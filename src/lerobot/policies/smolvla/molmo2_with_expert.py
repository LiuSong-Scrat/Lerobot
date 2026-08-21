# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-only Molmo2-ER backbone paired with the SmolVLA Action Expert.

This module deliberately does not instantiate ``Molmo2Model``.  That class
always constructs the SigLIP vision backbone and all 36 language blocks, while
SmolVLA only needs the first half of the language stack.  The implementation
below owns exactly these frozen Molmo parameters:

* ``model.transformer.wte``;
* ``model.transformer.blocks.0`` through ``blocks.17``;
* ``model.transformer.ln_f``.

There is no vision module, language-model head, or block 18--35 placeholder.
Weights are read selectively from sharded safetensors, so excluded tensors are
never materialized in host memory.  Molmo's fused QKV projection, Qwen3-style
per-head Q/K RMSNorm, grouped-query attention, RoPE, pre-norm residual layout,
and gated SwiGLU are retained.

The public ``forward`` contract mirrors ``SmolVLMWithExpertModel``.  In
particular, the frozen prefix is *not* evaluated under ``torch.no_grad``:
gradients from the Action Expert can still reach trainable point-cloud prefix
projections even though no Molmo parameter is trainable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AutoTokenizer

_CONFIG_NAME = "config.json"
_SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
_SAFETENSORS_SINGLE_NAME = "model.safetensors"


@dataclass(frozen=True)
class Molmo2TextSpec:
    """The text fields needed by this intentionally small Molmo runtime."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    additional_vocab_size: int
    hidden_act: str
    layer_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    qkv_bias: bool
    use_qk_norm: bool
    qk_norm_type: str
    embedding_dropout: float
    attention_dropout: float
    residual_dropout: float
    initializer_range: float

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size + self.additional_vocab_size

    @classmethod
    def from_model_directory(cls, model_directory: Path) -> Molmo2TextSpec:
        config_path = model_directory / _CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"Molmo2 config is missing: {config_path}")
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
        text = raw_config.get("text_config")
        if not isinstance(text, dict):
            raise ValueError(f"{config_path} does not contain a text_config object.")
        return cls(
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            vocab_size=int(text["vocab_size"]),
            additional_vocab_size=int(text.get("additional_vocab_size") or 0),
            hidden_act=str(text.get("hidden_act", "silu")),
            layer_norm_eps=float(text.get("layer_norm_eps", 1e-6)),
            rope_theta=float(text.get("rope_theta", 10_000.0)),
            max_position_embeddings=int(text.get("max_position_embeddings", 4096)),
            qkv_bias=bool(text.get("qkv_bias", False)),
            use_qk_norm=bool(text.get("use_qk_norm", False)),
            qk_norm_type=str(text.get("qk_norm_type", "olmo")),
            embedding_dropout=float(text.get("embedding_dropout", 0.0)),
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            residual_dropout=float(text.get("residual_dropout", 0.0)),
            initializer_range=float(text.get("initializer_range", 0.02)),
        )


_MOLMO2_ER_EXPECTED_TEXT_CONTRACT: dict[str, int | float | str | bool] = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "additional_vocab_size": 128,
    "hidden_act": "silu",
    "rope_theta": 5_000_000.0,
    "qkv_bias": False,
    "use_qk_norm": True,
    "qk_norm_type": "qwen3",
}


def validate_molmo2_er_text_contract(spec: Molmo2TextSpec) -> None:
    """Fail loudly instead of silently adapting an incompatible Molmo model."""

    actual = asdict(spec)
    mismatches = {
        key: (expected, actual.get(key))
        for key, expected in _MOLMO2_ER_EXPECTED_TEXT_CONTRACT.items()
        if actual.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected={expected!r}, actual={value!r}" for key, (expected, value) in mismatches.items()
        )
        raise ValueError(f"The supplied checkpoint is not the expected Molmo2-ER 4B text model ({details}).")


def get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: float = 4, multiple_of: int = 256) -> int:
    """Match the Action Expert width rule used by the original SmolVLA code."""

    intermediate = int(2 * hidden_dim / 3)
    intermediate = int(ffn_dim_multiplier * intermediate)
    return multiple_of * ((intermediate + multiple_of - 1) // multiple_of)


class Molmo2RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(dtype=original_dtype)


class Molmo2Embedding(nn.Module):
    """Molmo's split base/additional vocabulary, with checkpoint-compatible names."""

    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, features, device=device, dtype=dtype))
        self.new_embedding = nn.Parameter(
            torch.empty(num_new_embeddings, features, device=device, dtype=dtype)
        )

    @property
    def num_embeddings(self) -> int:
        return self.embedding.shape[0] + self.new_embedding.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.embedding.shape[1]

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= self.num_embeddings):
            raise IndexError(
                f"Molmo token id outside [0, {self.num_embeddings}): "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}."
            )
        return F.embedding(token_ids, torch.cat([self.embedding, self.new_embedding], dim=0))


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_molmo2_rope(x: Tensor, position_ids: Tensor, rope_theta: float) -> Tensor:
    """Apply Molmo/Qwen-style RoPE to ``x`` in ``(B,L,H,D)`` layout."""

    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")
    inv_freq = 1.0 / (
        float(rope_theta) ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    frequencies = position_ids.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat([frequencies, frequencies], dim=-1)
    cos = angles.cos().to(dtype=x.dtype).unsqueeze(2)
    sin = angles.sin().to(dtype=x.dtype).unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


class Molmo2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.ff_proj = nn.Linear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.ff_out = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.ff_proj(x).chunk(2, dim=-1)
        return self.ff_out(F.silu(gate) * value)


class Molmo2FusedAttention(nn.Module):
    """Native Molmo fused-QKV attention projections."""

    def __init__(self, spec: Molmo2TextSpec, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.num_heads = spec.num_attention_heads
        self.num_key_value_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.rope_theta = spec.rope_theta
        self.fused_dims = (
            self.num_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
        )
        self.att_proj = nn.Linear(
            spec.hidden_size,
            sum(self.fused_dims),
            bias=spec.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.q_norm = Molmo2RMSNorm(spec.head_dim, spec.layer_norm_eps, device=device, dtype=dtype)
        self.k_norm = Molmo2RMSNorm(spec.head_dim, spec.layer_norm_eps, device=device, dtype=dtype)
        self.attn_out = nn.Linear(
            self.num_heads * self.head_dim,
            spec.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def project(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self.att_proj(hidden_states).split(self.fused_dims, dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = apply_molmo2_rope(query, position_ids, self.rope_theta)
        key = apply_molmo2_rope(key, position_ids, self.rope_theta)
        return query, key, value


class Molmo2CrossAttention(nn.Module):
    """Action-query projection and learned remapping of frozen Molmo KV."""

    def __init__(
        self,
        expert_spec: Molmo2TextSpec,
        prefix_spec: Molmo2TextSpec,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.num_heads = expert_spec.num_attention_heads
        self.num_key_value_heads = expert_spec.num_key_value_heads
        self.head_dim = expert_spec.head_dim
        self.rope_theta = expert_spec.rope_theta
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        prefix_kv_dim = prefix_spec.num_key_value_heads * prefix_spec.head_dim
        self.q_proj = nn.Linear(
            expert_spec.hidden_size,
            query_dim,
            bias=expert_spec.qkv_bias,
            device=device,
            dtype=dtype,
        )
        # These are intentionally prefix-KV-width -> expert-KV-width.  Keeping
        # expert-hidden-width inputs here would add 16.5M unused parameters and
        # would not match SmolVLA's alternating cross-attention construction.
        self.k_proj = nn.Linear(prefix_kv_dim, kv_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(prefix_kv_dim, kv_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = Molmo2RMSNorm(
            expert_spec.head_dim, expert_spec.layer_norm_eps, device=device, dtype=dtype
        )
        self.k_norm = Molmo2RMSNorm(
            expert_spec.head_dim, expert_spec.layer_norm_eps, device=device, dtype=dtype
        )
        self.attn_out = nn.Linear(
            query_dim,
            expert_spec.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def project_query(self, hidden_states: Tensor, position_ids: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        query = self.q_norm(query)
        return apply_molmo2_rope(query, position_ids, self.rope_theta)

    def project_prefix_kv(self, prefix_key: Tensor, prefix_value: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, seq_len = prefix_key.shape[:2]
        key = self.k_proj(prefix_key.flatten(2)).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(prefix_value.flatten(2)).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        return self.k_norm(key), value


class Molmo2DecoderLayer(nn.Module):
    def __init__(self, spec: Molmo2TextSpec, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.self_attn = Molmo2FusedAttention(spec, device=device, dtype=dtype)
        self.attn_norm = Molmo2RMSNorm(spec.hidden_size, spec.layer_norm_eps, device=device, dtype=dtype)
        self.mlp = Molmo2MLP(spec.hidden_size, spec.intermediate_size, device=device, dtype=dtype)
        self.ff_norm = Molmo2RMSNorm(spec.hidden_size, spec.layer_norm_eps, device=device, dtype=dtype)
        self.residual_dropout = float(spec.residual_dropout)

    def finish_attention(self, hidden_states: Tensor, attention_output: Tensor) -> Tensor:
        projected = self.self_attn.attn_out(attention_output)
        hidden_states = hidden_states + F.dropout(projected, p=self.residual_dropout, training=self.training)
        mlp_output = self.mlp(self.ff_norm(hidden_states))
        return hidden_states + F.dropout(mlp_output, p=self.residual_dropout, training=self.training)


class Molmo2ExpertLayer(nn.Module):
    def __init__(
        self,
        expert_spec: Molmo2TextSpec,
        prefix_spec: Molmo2TextSpec,
        *,
        is_cross_attention: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.self_attn = Molmo2CrossAttention(
                expert_spec,
                prefix_spec,
                device=device,
                dtype=dtype,
            )
        else:
            self.self_attn = Molmo2FusedAttention(expert_spec, device=device, dtype=dtype)
        self.attn_norm = Molmo2RMSNorm(
            expert_spec.hidden_size,
            expert_spec.layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.mlp = Molmo2MLP(
            expert_spec.hidden_size,
            expert_spec.intermediate_size,
            device=device,
            dtype=dtype,
        )
        self.ff_norm = Molmo2RMSNorm(
            expert_spec.hidden_size,
            expert_spec.layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.residual_dropout = float(expert_spec.residual_dropout)

    def finish_attention(self, hidden_states: Tensor, attention_output: Tensor) -> Tensor:
        projected = self.self_attn.attn_out(attention_output)
        hidden_states = hidden_states + F.dropout(projected, p=self.residual_dropout, training=self.training)
        mlp_output = self.mlp(self.ff_norm(hidden_states))
        return hidden_states + F.dropout(mlp_output, p=self.residual_dropout, training=self.training)


class Molmo2TextBackbone(nn.Module):
    def __init__(
        self,
        spec: Molmo2TextSpec,
        num_layers: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.spec = spec
        self.wte = Molmo2Embedding(
            spec.vocab_size,
            spec.additional_vocab_size,
            spec.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.emb_drop = nn.Dropout(spec.embedding_dropout)
        self.blocks = nn.ModuleList(
            [Molmo2DecoderLayer(spec, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.ln_f = Molmo2RMSNorm(spec.hidden_size, spec.layer_norm_eps, device=device, dtype=dtype)

    @property
    def layers(self) -> nn.ModuleList:
        return self.blocks

    @property
    def norm(self) -> Molmo2RMSNorm:
        return self.ln_f

    def get_input_embeddings(self) -> Molmo2Embedding:
        return self.wte


class Molmo2ExpertBackbone(nn.Module):
    def __init__(
        self,
        expert_spec: Molmo2TextSpec,
        prefix_spec: Molmo2TextSpec,
        num_layers: int,
        *,
        self_attn_every_n_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.spec = expert_spec
        self.layers = nn.ModuleList(
            [
                Molmo2ExpertLayer(
                    expert_spec,
                    prefix_spec,
                    is_cross_attention=(layer_idx % self_attn_every_n_layers != 0),
                    device=device,
                    dtype=dtype,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.norm = Molmo2RMSNorm(
            expert_spec.hidden_size,
            expert_spec.layer_norm_eps,
            device=device,
            dtype=dtype,
        )


def _initialize_module(module: nn.Module, std: float) -> None:
    """Match Molmo/transformers initialization for freshly trained tensors."""

    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear) and not child.weight.is_meta:
                child.weight.normal_(mean=0.0, std=std)
                if child.bias is not None:
                    child.bias.zero_()
            elif isinstance(child, Molmo2Embedding):
                if not child.embedding.is_meta:
                    child.embedding.normal_(mean=0.0, std=std)
                    child.new_embedding.normal_(mean=0.0, std=std)
            elif isinstance(child, Molmo2RMSNorm) and not child.weight.is_meta:
                child.weight.fill_(1.0)


def _repeat_kv(states: Tensor, repeats: int) -> Tensor:
    # (B,L,KVH,D) -> (B,H,L,D), without allocating when repeats == 1.
    states = states.transpose(1, 2)
    if repeats == 1:
        return states
    batch_size, kv_heads, seq_len, head_dim = states.shape
    states = states[:, :, None, :, :].expand(batch_size, kv_heads, repeats, seq_len, head_dim)
    return states.reshape(batch_size, kv_heads * repeats, seq_len, head_dim)


def _scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
    """Grouped-query SDPA returning ``(B,L,H*D)`` like Molmo ``attn_out`` expects."""

    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    if mask.shape != (query.shape[0], query.shape[1], key.shape[1]):
        raise ValueError(
            "Attention mask has the wrong shape: "
            f"expected {(query.shape[0], query.shape[1], key.shape[1])}, got {tuple(mask.shape)}."
        )
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError(f"Query heads {query.shape[2]} are not divisible by KV heads {key.shape[2]}.")
    repeats = query.shape[2] // key.shape[2]
    query_heads = query.transpose(1, 2)
    key_heads = _repeat_kv(key, repeats)
    value_heads = _repeat_kv(value, repeats)
    output = F.scaled_dot_product_attention(
        query_heads,
        key_heads,
        value_heads,
        attn_mask=mask[:, None, :, :],
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous().flatten(2)


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Molmo dtype: {dtype!r}.") from exc


def _checkpoint_weight_map(model_directory: Path) -> dict[str, Path]:
    index_path = model_directory / _SAFETENSORS_INDEX_NAME
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            index = json.load(index_file)
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict):
            raise ValueError(f"Malformed safetensors index: {index_path}")
        return {str(key): model_directory / str(filename) for key, filename in raw_map.items()}

    single_path = model_directory / _SAFETENSORS_SINGLE_NAME
    if not single_path.is_file():
        raise FileNotFoundError(
            f"Expected {_SAFETENSORS_INDEX_NAME} or {_SAFETENSORS_SINGLE_NAME} in {model_directory}."
        )
    with safe_open(single_path, framework="pt", device="cpu") as checkpoint:
        return dict.fromkeys(checkpoint, single_path)


def load_selective_molmo2_text_weights(
    backbone: Molmo2TextBackbone,
    model_directory: str | Path,
) -> dict[str, Any]:
    """Load only retained text tensors, one mmap-backed shard at a time."""

    model_directory = Path(model_directory).expanduser().resolve()
    weight_map = _checkpoint_weight_map(model_directory)
    target_state = backbone.state_dict()
    source_for_target: dict[str, str] = {}
    for target_key in target_state:
        source_key = f"model.transformer.{target_key}"
        if source_key not in weight_map:
            raise KeyError(f"Required retained Molmo tensor is missing: {source_key}")
        source_for_target[target_key] = source_key

    assignments_by_file: dict[Path, list[tuple[str, str]]] = {}
    for target_key, source_key in source_for_target.items():
        assignments_by_file.setdefault(weight_map[source_key], []).append((target_key, source_key))

    selected_parameters = 0
    with torch.no_grad():
        for shard_path, assignments in assignments_by_file.items():
            if not shard_path.is_file():
                raise FileNotFoundError(f"Molmo safetensors shard is missing: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
                for target_key, source_key in assignments:
                    source = checkpoint.get_tensor(source_key)
                    target = target_state[target_key]
                    if source.shape != target.shape:
                        raise ValueError(
                            f"Shape mismatch for {source_key}: checkpoint={tuple(source.shape)}, "
                            f"model={tuple(target.shape)}."
                        )
                    if target.is_meta:
                        raise RuntimeError("Cannot load Molmo weights into a meta-device model.")
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                    selected_parameters += target.numel()

    ignored_keys = set(weight_map).difference(source_for_target.values())
    retained_block_prefix = "model.transformer.blocks."
    dropped_block_keys = 0
    vision_keys = 0
    lm_head_keys = 0
    other_ignored_keys = 0
    for key in ignored_keys:
        if key.startswith(retained_block_prefix):
            suffix = key[len(retained_block_prefix) :]
            block_index = suffix.split(".", maxsplit=1)[0]
            if block_index.isdigit() and int(block_index) >= len(backbone.blocks):
                dropped_block_keys += 1
            else:
                other_ignored_keys += 1
        elif key.startswith("model.vision_backbone."):
            vision_keys += 1
        elif key.startswith("lm_head."):
            lm_head_keys += 1
        else:
            other_ignored_keys += 1

    if other_ignored_keys:
        unexpected = sorted(ignored_keys)[:8]
        raise RuntimeError(
            "Selective Molmo loading found ignored tensors outside the explicitly excluded "
            f"vision/dropped-block/lm-head groups (count={other_ignored_keys}, preview={unexpected})."
        )

    return {
        "model_directory": str(model_directory),
        "selected_tensors": len(source_for_target),
        "selected_parameters": selected_parameters,
        "retained_blocks": len(backbone.blocks),
        "ignored_vision_tensors": vision_keys,
        "ignored_dropped_block_tensors": dropped_block_keys,
        "ignored_lm_head_tensors": lm_head_keys,
    }


class Molmo2WithExpertModel(nn.Module):
    """First-half Molmo2-ER text stack plus an 18-layer 0.75x Action Expert."""

    # SmolVLM embeddings were multiplied by sqrt(H) in the original policy.
    # Native Molmo embeddings are not, so integration code must leave both the
    # language and projected point tokens at their native scale.
    scale_input_embeddings = False

    def __init__(
        self,
        model_id: str,
        vlm_weights_path: str | None = None,
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = True,
        attention_mode: str = "cross_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = 18,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
        device: str | torch.device = "auto",
        torch_dtype: str | torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        del freeze_vision_encoder  # There is no vision module to freeze.

        architecture_directory = Path(model_id).expanduser().resolve()
        weight_directory = (
            Path(vlm_weights_path).expanduser().resolve()
            if vlm_weights_path and str(vlm_weights_path).lower() not in {"none", "false", "off", "0"}
            else architecture_directory
        )
        if not architecture_directory.is_dir():
            raise ValueError(
                "Molmo2WithExpertModel requires a local Molmo2-ER directory so config, tokenizer, "
                f"and sharded weights remain reproducible; got {model_id!r}."
            )

        text_spec = Molmo2TextSpec.from_model_directory(architecture_directory)
        validate_molmo2_er_text_contract(text_spec)
        if num_vlm_layers != 18:
            raise ValueError(f"Molmo2-ER 3B keeps exactly the first 18/36 VLM blocks, got {num_vlm_layers}.")
        if num_expert_layers <= 0:
            num_expert_layers = num_vlm_layers
        if num_expert_layers != 18:
            raise ValueError(
                f"The controlled 3B Action Expert has exactly 18 blocks, got {num_expert_layers}."
            )
        if not math.isclose(expert_width_multiplier, 0.75, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "The controlled Molmo2-ER Action Expert width multiplier is 0.75 "
                f"(H=1920), got {expert_width_multiplier}."
            )
        if "cross" not in attention_mode:
            raise ValueError(f"Molmo2-ER 3B requires alternating cross attention, got {attention_mode!r}.")
        if self_attn_every_n_layers != 2:
            raise ValueError(
                "Molmo2-ER 3B requires even Action Expert blocks to use self-attention and odd "
                f"blocks to use cross-attention; got interval {self_attn_every_n_layers}."
            )

        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(torch_dtype)
        expert_hidden_size = int(text_spec.hidden_size * expert_width_multiplier)
        expert_spec = Molmo2TextSpec(
            hidden_size=expert_hidden_size,
            intermediate_size=get_intermediate_size(expert_hidden_size),
            num_hidden_layers=num_expert_layers,
            num_attention_heads=text_spec.num_attention_heads,
            num_key_value_heads=text_spec.num_key_value_heads,
            head_dim=text_spec.head_dim,
            vocab_size=text_spec.vocab_size,
            additional_vocab_size=text_spec.additional_vocab_size,
            hidden_act=text_spec.hidden_act,
            layer_norm_eps=text_spec.layer_norm_eps,
            rope_theta=text_spec.rope_theta,
            max_position_embeddings=text_spec.max_position_embeddings,
            qkv_bias=text_spec.qkv_bias,
            use_qk_norm=text_spec.use_qk_norm,
            qk_norm_type=text_spec.qk_norm_type,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
            initializer_range=text_spec.initializer_range,
        )

        self.vlm = Molmo2TextBackbone(
            text_spec,
            num_vlm_layers,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        self.lm_expert = Molmo2ExpertBackbone(
            expert_spec,
            text_spec,
            num_expert_layers,
            self_attn_every_n_layers=self_attn_every_n_layers,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        # Every retained VLM tensor is loaded below when weights are enabled.
        # Initializing 2.2B frozen values first would only consume RNG and make
        # the trainable Expert initialization depend on an unused random draw.
        if not load_vlm_weights:
            _initialize_module(self.vlm, text_spec.initializer_range)
        _initialize_module(self.lm_expert, expert_spec.initializer_range)

        self.load_report: dict[str, Any] | None = None
        if load_vlm_weights:
            self.load_report = load_selective_molmo2_text_weights(self.vlm, weight_directory)
        elif vlm_weights_path:
            raise ValueError("vlm_weights_path was provided but load_vlm_weights=False.")

        tokenizer = AutoTokenizer.from_pretrained(architecture_directory, trust_remote_code=False)
        # Point-only mode never consumes these SmolVLM-only values, but retaining
        # the attributes lets old initialization code remain harmless until its
        # image-only paths are removed.
        if getattr(tokenizer, "fake_image_token_id", None) is None:
            tokenizer.fake_image_token_id = -1
        if getattr(tokenizer, "global_image_token_id", None) is None:
            tokenizer.global_image_token_id = -1
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.eos_token_id
        if bos_token_id is None:
            raise ValueError("Molmo2-ER tokenizer must define a BOS or EOS token ID.")
        self.bos_token_id = int(bos_token_id)
        self.processor = SimpleNamespace(tokenizer=tokenizer)

        self.num_vlm_layers = num_vlm_layers
        self.num_expert_layers = num_expert_layers
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.attention_mode = attention_mode
        self.train_expert_only = train_expert_only
        self.expert_hidden_size = expert_hidden_size
        self.num_attention_heads = text_spec.num_attention_heads
        self.num_key_value_heads = text_spec.num_key_value_heads
        self.head_dim = text_spec.head_dim
        self.text_spec = text_spec
        self.expert_spec = expert_spec
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(**asdict(text_spec), total_vocab_size=text_spec.total_vocab_size),
            model_type="molmo2_text_only_smolvla",
        )
        self.set_requires_grad()

    @property
    def architecture_contract(self) -> dict[str, Any]:
        return {
            "source_vlm_layers": self.text_spec.num_hidden_layers,
            "retained_vlm_layers": self.num_vlm_layers,
            "retained_vlm_layer_indices": list(range(self.num_vlm_layers)),
            "vlm_hidden_size": self.text_spec.hidden_size,
            "vlm_intermediate_size": self.text_spec.intermediate_size,
            "expert_layers": self.num_expert_layers,
            "expert_hidden_size": self.expert_spec.hidden_size,
            "expert_intermediate_size": self.expert_spec.intermediate_size,
            "self_attention_layers": list(range(0, self.num_expert_layers, 2)),
            "cross_attention_layers": list(range(1, self.num_expert_layers, 2)),
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.text_spec.rope_theta,
            "vision_module_present": False,
            "lm_head_present": False,
        }

    def get_vlm_model(self) -> Molmo2TextBackbone:
        return self.vlm

    def set_requires_grad(self) -> None:
        if not self.train_expert_only:
            raise ValueError(
                "The controlled Molmo2-ER 3B experiment freezes the retained VLM; "
                "train_expert_only must remain True."
            )
        self.vlm.requires_grad_(False)
        self.lm_expert.requires_grad_(True)
        self.vlm.eval()

    def train(self, mode: bool = True) -> Molmo2WithExpertModel:
        super().train(mode)
        # Frozen VLM dropout must stay disabled, while autograd through its
        # operations remains enabled for trainable point-prefix projections.
        self.vlm.eval()
        return self

    def embed_image(self, image: Tensor) -> Tensor:
        del image
        raise RuntimeError("Molmo2-ER 3B is point-cloud-only and has no image/vision module.")

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        # Native Molmo applies no sqrt(hidden_size) rescaling.
        return self.vlm.emb_drop(self.vlm.wte(tokens))

    @staticmethod
    def _mask_slice(
        attention_mask: Tensor | None,
        row_slice: slice,
        col_slice: slice,
        *,
        batch_size: int,
        rows: int,
        cols: int,
        device: torch.device,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(batch_size, rows, cols, dtype=torch.bool, device=device)
        return attention_mask[:, row_slice, col_slice].to(device=device, dtype=torch.bool)

    def _forward_vlm_layer(
        self,
        layer: Molmo2DecoderLayer,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = layer.attn_norm(hidden_states)
        query, key, value = layer.self_attn.project(normalized, position_ids)
        attention_output = _scaled_dot_product_attention(query, key, value, attention_mask)
        return layer.finish_attention(hidden_states, attention_output), key, value

    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        if inputs_embeds is None or len(inputs_embeds) != 2:
            raise ValueError("inputs_embeds must be [prefix_embeddings, expert_embeddings].")
        prefix, expert = inputs_embeds
        if prefix is None and expert is None:
            raise ValueError("At least one embedding stream must be present.")

        # PointSeg/action projections intentionally stay in FP32, while the
        # retained Molmo and Action Expert weights are BF16.  Concatenating an
        # FP32 point token with BF16 language embeddings promotes the complete
        # prefix to FP32; without this explicit boundary cast the first BF16
        # Linear fails when AMP is disabled (the controlled v0.4.3 recipe).
        # ``Tensor.to`` remains differentiable, so gradients still reach every
        # trainable point/action projection across this cast.
        if prefix is not None:
            prefix = prefix.to(dtype=self.vlm.wte.embedding.dtype)
        if expert is not None:
            expert = expert.to(dtype=self.lm_expert.norm.weight.dtype)

        reference = prefix if prefix is not None else expert
        assert reference is not None
        batch_size = reference.shape[0]
        if position_ids is None:
            total_length = (prefix.shape[1] if prefix is not None else 0) + (
                expert.shape[1] if expert is not None else 0
            )
            position_ids = (
                torch.arange(total_length, device=reference.device).unsqueeze(0).expand(batch_size, -1)
            )
        use_cache = bool(use_cache)
        fill_kv_cache = bool(fill_kv_cache)
        if fill_kv_cache and prefix is None:
            raise ValueError("fill_kv_cache=True requires prefix embeddings.")
        if use_cache and past_key_values is None:
            past_key_values = {}

        for layer_idx, (vlm_layer, expert_layer) in enumerate(
            zip(self.vlm.blocks, self.lm_expert.layers, strict=True)
        ):
            prefix_len = prefix.shape[1] if prefix is not None else 0
            expert_len = expert.shape[1] if expert is not None else 0
            is_self_attention = layer_idx % self.self_attn_every_n_layers == 0

            if fill_kv_cache or (prefix is not None and expert is None):
                assert prefix is not None
                prefix_positions = position_ids[:, :prefix_len]
                prefix_mask = self._mask_slice(
                    attention_mask,
                    slice(0, prefix_len),
                    slice(0, prefix_len),
                    batch_size=batch_size,
                    rows=prefix_len,
                    cols=prefix_len,
                    device=prefix.device,
                )
                prefix, prefix_key, prefix_value = self._forward_vlm_layer(
                    vlm_layer, prefix, prefix_positions, prefix_mask
                )
                if use_cache:
                    assert past_key_values is not None
                    past_key_values[layer_idx] = {
                        "key_states": prefix_key,
                        "value_states": prefix_value,
                    }
                continue

            if prefix is not None and expert is not None:
                prefix_positions = position_ids[:, :prefix_len]
                expert_positions = position_ids[:, prefix_len : prefix_len + expert_len]
                prefix_normalized = vlm_layer.attn_norm(prefix)
                prefix_query, prefix_key, prefix_value = vlm_layer.self_attn.project(
                    prefix_normalized, prefix_positions
                )
                expert_normalized = expert_layer.attn_norm(expert)

                if is_self_attention:
                    assert isinstance(expert_layer.self_attn, Molmo2FusedAttention)
                    expert_query, expert_key, expert_value = expert_layer.self_attn.project(
                        expert_normalized, expert_positions
                    )
                    query = torch.cat([prefix_query, expert_query], dim=1)
                    key = torch.cat([prefix_key, expert_key], dim=1)
                    value = torch.cat([prefix_value, expert_value], dim=1)
                    joint_len = prefix_len + expert_len
                    joint_mask = self._mask_slice(
                        attention_mask,
                        slice(0, joint_len),
                        slice(0, joint_len),
                        batch_size=batch_size,
                        rows=joint_len,
                        cols=joint_len,
                        device=prefix.device,
                    )
                    joint_output = _scaled_dot_product_attention(query, key, value, joint_mask)
                    prefix_output = joint_output[:, :prefix_len]
                    expert_output = joint_output[:, prefix_len:]
                else:
                    assert isinstance(expert_layer.self_attn, Molmo2CrossAttention)
                    prefix_mask = self._mask_slice(
                        attention_mask,
                        slice(0, prefix_len),
                        slice(0, prefix_len),
                        batch_size=batch_size,
                        rows=prefix_len,
                        cols=prefix_len,
                        device=prefix.device,
                    )
                    prefix_output = _scaled_dot_product_attention(
                        prefix_query, prefix_key, prefix_value, prefix_mask
                    )
                    relative_expert_positions = (
                        expert_positions - expert_positions.min(dim=1, keepdim=True).values
                    )
                    expert_query = expert_layer.self_attn.project_query(
                        expert_normalized, relative_expert_positions
                    )
                    expert_key, expert_value = expert_layer.self_attn.project_prefix_kv(
                        prefix_key, prefix_value
                    )
                    cross_mask = self._mask_slice(
                        attention_mask,
                        slice(prefix_len, prefix_len + expert_len),
                        slice(0, prefix_len),
                        batch_size=batch_size,
                        rows=expert_len,
                        cols=prefix_len,
                        device=expert.device,
                    )
                    expert_output = _scaled_dot_product_attention(
                        expert_query, expert_key, expert_value, cross_mask
                    )

                prefix = vlm_layer.finish_attention(prefix, prefix_output)
                expert = expert_layer.finish_attention(expert, expert_output)
                if use_cache:
                    assert past_key_values is not None
                    past_key_values[layer_idx] = {
                        "key_states": prefix_key,
                        "value_states": prefix_value,
                    }
                continue

            if prefix is None and expert is not None:
                if past_key_values is None or layer_idx not in past_key_values:
                    raise ValueError(
                        f"Suffix-only inference requires a populated prefix KV cache at layer {layer_idx}."
                    )
                cached_key = past_key_values[layer_idx]["key_states"]
                cached_value = past_key_values[layer_idx]["value_states"]
                prefix_len = cached_key.shape[1]
                expert_positions = position_ids[:, -expert_len:]
                expert_normalized = expert_layer.attn_norm(expert)
                if is_self_attention:
                    assert isinstance(expert_layer.self_attn, Molmo2FusedAttention)
                    expert_query, expert_key, expert_value = expert_layer.self_attn.project(
                        expert_normalized, expert_positions
                    )
                    key = torch.cat([cached_key, expert_key], dim=1)
                    value = torch.cat([cached_value, expert_value], dim=1)
                    suffix_mask = self._mask_slice(
                        attention_mask,
                        slice(0, expert_len),
                        slice(0, prefix_len + expert_len),
                        batch_size=batch_size,
                        rows=expert_len,
                        cols=prefix_len + expert_len,
                        device=expert.device,
                    )
                    expert_output = _scaled_dot_product_attention(expert_query, key, value, suffix_mask)
                else:
                    assert isinstance(expert_layer.self_attn, Molmo2CrossAttention)
                    relative_positions = expert_positions - expert_positions.min(dim=1, keepdim=True).values
                    expert_query = expert_layer.self_attn.project_query(expert_normalized, relative_positions)
                    expert_key, expert_value = expert_layer.self_attn.project_prefix_kv(
                        cached_key, cached_value
                    )
                    cross_mask = self._mask_slice(
                        attention_mask,
                        slice(0, expert_len),
                        slice(0, prefix_len),
                        batch_size=batch_size,
                        rows=expert_len,
                        cols=prefix_len,
                        device=expert.device,
                    )
                    expert_output = _scaled_dot_product_attention(
                        expert_query, expert_key, expert_value, cross_mask
                    )
                expert = expert_layer.finish_attention(expert, expert_output)

        outputs: list[Tensor | None] = [
            self.vlm.ln_f(prefix) if prefix is not None else None,
            self.lm_expert.norm(expert) if expert is not None else None,
        ]
        return outputs, past_key_values


# A descriptive alias used by architecture checks and downstream experiments.
Molmo2TextWithExpertModel = Molmo2WithExpertModel
Molmo2TextHalfWithExpertModel = Molmo2WithExpertModel


__all__ = [
    "Molmo2TextSpec",
    "Molmo2WithExpertModel",
    "Molmo2TextWithExpertModel",
    "Molmo2TextHalfWithExpertModel",
    "apply_molmo2_rope",
    "get_intermediate_size",
    "load_selective_molmo2_text_weights",
    "validate_molmo2_er_text_contract",
]
