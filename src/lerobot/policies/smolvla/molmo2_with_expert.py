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

This module deliberately does not instantiate ``Molmo2Model``.  The compact
runtime can retain either the 18-layer point-only control or all 36 text
layers used by the full Molmo2-ER backend without constructing a second vision
backbone.  It owns exactly the selected frozen Molmo text parameters:

* ``model.transformer.wte``;
* ``model.transformer.blocks.0`` through the configured retained layer;
* ``model.transformer.ln_f``.

There is no language-model head.  The point-only control has no vision module;
the full backend owns the single native vision module in its wrapper.  Weights
are read selectively from sharded safetensors, so excluded tensors are never
materialized in host memory.  Molmo's fused QKV projection, Qwen3-style
per-head Q/K RMSNorm, grouped-query attention, RoPE, pre-norm residual layout,
and gated SwiGLU are retained.

The public ``forward`` contract mirrors ``SmolVLMWithExpertModel``.  The Full
backend keeps native Molmo image/language tokens in an independent read-only
stream.  Trainable FG/BG tokens form a small scene-shadow stream which reads
native per-layer K/V without ever changing native hidden states.  This is the
largest WEPVLA-compatible directed graph that also preserves Molmo's native
path: scene reads native, actions read native+scene, and native never reads
scene/action.  The historical ``inference_only_vlm`` path remains for old
point-memory experiments.
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
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from lerobot.policies.smolvla.constants import FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY

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
    norm_after: bool = False
    rope_scaling: dict[str, Any] | None = None
    rope_scaling_layers: tuple[int, ...] | None = None
    attn_implementation: str = "sdpa"

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
            norm_after=bool(text.get("norm_after", False)),
            rope_scaling=text.get("rope_scaling"),
            rope_scaling_layers=(
                tuple(int(index) for index in text["rope_scaling_layers"])
                if text.get("rope_scaling_layers") is not None
                else None
            ),
            attn_implementation=str(text.get("attn_implementation", "sdpa")),
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
    "layer_norm_eps": 1e-6,
    "max_position_embeddings": 16_384,
    "embedding_dropout": 0.0,
    "attention_dropout": 0.0,
    "residual_dropout": 0.0,
    "initializer_range": 0.02,
    "norm_after": False,
    "rope_scaling": None,
    "rope_scaling_layers": None,
    "attn_implementation": "sdpa",
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
        # Molmo stores the base and additional vocabularies in two checkpoint
        # tensors.  Concatenating them here copied the complete ~152k x 2560
        # table on every batch (roughly 0.78 GiB in bf16), even though the
        # table is frozen.  Look up both small token batches independently and
        # select the correct result instead.  The deliberately *unclamped*
        # selected ids retain F.embedding's native out-of-range error for both
        # negative and over-large token ids without a separate CUDA reduction.
        base_vocab_size = self.embedding.shape[0]
        if self.new_embedding.shape[0] == 0:
            return F.embedding(token_ids, self.embedding)

        uses_new_vocabulary = token_ids >= base_vocab_size
        zero = torch.zeros((), dtype=token_ids.dtype, device=token_ids.device)
        base_token_ids = torch.where(uses_new_vocabulary, zero, token_ids)
        new_token_ids = torch.where(uses_new_vocabulary, token_ids - base_vocab_size, zero)
        try:
            base_embeddings = F.embedding(base_token_ids, self.embedding)
            new_embeddings = F.embedding(new_token_ids, self.new_embedding)
        except IndexError as error:
            raise IndexError(
                f"Molmo token id is outside the complete split vocabulary [0, {self.num_embeddings})."
            ) from error
        return torch.where(uses_new_vocabulary.unsqueeze(-1), new_embeddings, base_embeddings)


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _molmo2_rope_factors(
    position_ids: Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build the shared Molmo/Qwen RoPE factors for one Q/K projection."""

    if head_dim % 2:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")
    inv_freq = 1.0 / (
        float(rope_theta) ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    frequencies = position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat([frequencies, frequencies], dim=-1)
    return angles.cos().to(dtype=dtype).unsqueeze(2), angles.sin().to(dtype=dtype).unsqueeze(2)


def _apply_molmo2_rope_factors(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + _rotate_half(x) * sin


def apply_molmo2_rope(x: Tensor, position_ids: Tensor, rope_theta: float) -> Tensor:
    """Apply Molmo/Qwen-style RoPE to ``x`` in ``(B,L,H,D)`` layout."""

    cos, sin = _molmo2_rope_factors(
        position_ids,
        head_dim=x.shape[-1],
        rope_theta=rope_theta,
        device=x.device,
        dtype=x.dtype,
    )
    return _apply_molmo2_rope_factors(x, cos, sin)


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
        # Q and K have identical positions/head width/dtype. Build their RoPE
        # factors once per projection instead of repeating pow/cos/sin twice;
        # this also avoids the duplicate work during checkpoint recomputation.
        cos, sin = _molmo2_rope_factors(
            position_ids,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            device=query.device,
            dtype=query.dtype,
        )
        query = _apply_molmo2_rope_factors(query, cos, sin)
        key = _apply_molmo2_rope_factors(key, cos, sin)
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
    # PyTorch 2.5's native GQA path falls back to the math kernel when this
    # model's arbitrary boolean attention mask is present.  Explicitly map
    # Molmo's 8 KV heads onto its 32 query heads so CUDA can retain the
    # efficient equal-head SDPA kernel used by the reference implementation.
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

    def _forward_coupled_prefix_expert_layer(
        self,
        layer_idx: int,
        prefix: Tensor,
        expert: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run one exact WEP prefix/Action-Expert layer without cache side effects.

        This pure tensor boundary is used by non-reentrant activation
        checkpointing.  Even layers perform masked joint MHA; odd layers update
        the prefix with VLM self-attention while actions cross-attend the whole
        evolving prefix.  Molmo parameters may be frozen, but this function is
        intentionally differentiable with respect to prefix inputs.
        """

        vlm_layer = self.vlm.blocks[layer_idx]
        expert_layer = self.lm_expert.layers[layer_idx]
        batch_size = prefix.shape[0]
        prefix_len = prefix.shape[1]
        expert_len = expert.shape[1]
        prefix_positions = position_ids[:, :prefix_len]
        expert_positions = position_ids[:, prefix_len : prefix_len + expert_len]

        prefix_normalized = vlm_layer.attn_norm(prefix)
        prefix_query, prefix_key, prefix_value = vlm_layer.self_attn.project(
            prefix_normalized, prefix_positions
        )
        expert_normalized = expert_layer.attn_norm(expert)
        is_self_attention = layer_idx % self.self_attn_every_n_layers == 0

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

        return (
            vlm_layer.finish_attention(prefix, prefix_output),
            expert_layer.finish_attention(expert, expert_output),
        )

    def _prefill_inference_only_vlm(
        self,
        prefix: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, dict[int, dict[str, Tensor]]]:
        """Run the frozen VLM as a read-only per-layer memory producer.

        This is the hard Scheme-B boundary: no trainable token is accepted by
        this method and every Molmo decoder operation executes under
        ``torch.no_grad``.  The returned K/V tensors are ordinary detached
        tensors, so Action Expert backward cannot enter the VLM graph.
        """

        prefix = prefix.to(dtype=self.vlm.wte.embedding.dtype)
        batch_size, prefix_len = prefix.shape[:2]
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
        cache: dict[int, dict[str, Tensor]] = {}
        with torch.no_grad():
            hidden = prefix
            for layer_idx, vlm_layer in enumerate(self.vlm.blocks):
                hidden, key, value = self._forward_vlm_layer(
                    vlm_layer,
                    hidden,
                    prefix_positions,
                    prefix_mask,
                )
                cache[layer_idx] = {
                    "key_states": key.detach(),
                    "value_states": value.detach(),
                }
            hidden = self.vlm.ln_f(hidden).detach()
        return hidden, cache

    def _prefill_native_vlm(
        self,
        native: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, dict[int, dict[str, Tensor]]]:
        """Produce the immutable native Molmo memory used by the v5 graph.

        Only official image/language embeddings are accepted here.  In
        particular, FG/BG and action embeddings must have been split off by
        :meth:`forward` before this boundary.  Running the native stream as a
        separate attention problem (rather than merely masking extra columns
        in a larger SDPA call) also keeps its kernel shape independent of the
        trainable streams.
        """

        return self._prefill_inference_only_vlm(native, position_ids, attention_mask)

    def _prefill_scene_from_native_memory(
        self,
        scene: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        native_cache: dict[int, dict[str, Tensor]],
    ) -> tuple[Tensor, dict[int, dict[str, Tensor]]]:
        """Evolve FG/BG through frozen Molmo blocks without modifying native.

        Scene queries read the detached native K/V and every scene token.
        Native queries are absent from this computation, so active point-cloud
        inputs cannot condition Molmo's pretrained image/language hidden
        states.  The scene K/V remain differentiable and are consumed by every
        Action Expert layer.
        """

        scene = scene.to(dtype=self.vlm.wte.embedding.dtype)
        batch_size, scene_len = scene.shape[:2]
        first_native_key = native_cache.get(0, {}).get("key_states")
        if not torch.is_tensor(first_native_key):
            raise ValueError("v5 scene prefill requires a complete native Molmo cache.")
        native_len = first_native_key.shape[1]
        prefix_len = native_len + scene_len
        if position_ids.shape[1] < prefix_len:
            raise ValueError(
                "v5 position_ids must cover native+scene prefix; "
                f"got {tuple(position_ids.shape)} for {native_len}+{scene_len} tokens."
            )
        scene_positions = position_ids[:, native_len:prefix_len]
        scene_mask = self._mask_slice(
            attention_mask,
            slice(native_len, prefix_len),
            slice(0, prefix_len),
            batch_size=batch_size,
            rows=scene_len,
            cols=prefix_len,
            device=scene.device,
        )

        scene_cache: dict[int, dict[str, Tensor]] = {}
        for layer_idx, vlm_layer in enumerate(self.vlm.blocks):
            native_entry = native_cache.get(layer_idx)
            if native_entry is None:
                raise ValueError(f"Native Molmo cache is missing layer {layer_idx}.")
            native_key = native_entry["key_states"]
            native_value = native_entry["value_states"]
            if native_key.requires_grad or native_value.requires_grad:
                raise RuntimeError("Native Molmo K/V must remain detached in the v5 graph.")

            scene_normalized = vlm_layer.attn_norm(scene)
            scene_query, scene_key, scene_value = vlm_layer.self_attn.project(
                scene_normalized,
                scene_positions,
            )
            scene_output = _scaled_dot_product_attention(
                scene_query,
                torch.cat([native_key, scene_key], dim=1),
                torch.cat([native_value, scene_value], dim=1),
                scene_mask,
            )
            # Cache the pre-update K/V for the same layer, matching standard
            # transformer cache semantics.  Do not detach these tensors:
            # Action loss must reach the FG/BG projections through them.
            scene_cache[layer_idx] = {
                "key_states": scene_key,
                "value_states": scene_value,
            }
            scene = vlm_layer.finish_attention(scene, scene_output)

        return self.vlm.ln_f(scene), scene_cache

    def _forward_action_layer_from_native_scene_memory(
        self,
        layer_idx: int,
        expert: Tensor,
        expert_positions: Tensor,
        suffix_attention_mask: Tensor,
        native_entry: dict[str, Tensor],
        scene_key: Tensor,
        scene_value: Tensor,
    ) -> Tensor:
        """Run one WEP Action Expert layer against native+scene memory."""

        expert_layer = self.lm_expert.layers[layer_idx]
        native_key = native_entry["key_states"]
        native_value = native_entry["value_states"]
        if native_key.requires_grad or native_value.requires_grad:
            raise RuntimeError("Native Molmo memory must never require gradients.")
        prefix_key = torch.cat([native_key, scene_key], dim=1)
        prefix_value = torch.cat([native_value, scene_value], dim=1)
        prefix_len = prefix_key.shape[1]
        expert_len = expert.shape[1]
        expert_normalized = expert_layer.attn_norm(expert)
        is_self_attention = layer_idx % self.self_attn_every_n_layers == 0

        if is_self_attention:
            assert isinstance(expert_layer.self_attn, Molmo2FusedAttention)
            expert_query, expert_key, expert_value = expert_layer.self_attn.project(
                expert_normalized,
                expert_positions,
            )
            expert_output = _scaled_dot_product_attention(
                expert_query,
                torch.cat([prefix_key, expert_key], dim=1),
                torch.cat([prefix_value, expert_value], dim=1),
                suffix_attention_mask[:, :, : prefix_len + expert_len],
            )
        else:
            assert isinstance(expert_layer.self_attn, Molmo2CrossAttention)
            relative_positions = expert_positions - expert_positions.min(
                dim=1, keepdim=True
            ).values
            expert_query = expert_layer.self_attn.project_query(
                expert_normalized,
                relative_positions,
            )
            expert_key, expert_value = expert_layer.self_attn.project_prefix_kv(
                prefix_key,
                prefix_value,
            )
            expert_output = _scaled_dot_product_attention(
                expert_query,
                expert_key,
                expert_value,
                suffix_attention_mask[:, :, :prefix_len],
            )
        return expert_layer.finish_attention(expert, expert_output)

    def _forward_action_from_native_scene_memory(
        self,
        expert: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        native_cache: dict[int, dict[str, Tensor]],
        scene_cache: dict[int, dict[str, Tensor]],
    ) -> Tensor:
        """Run Action Expert while checkpointing only the trainable A stream."""

        expert = expert.to(dtype=self.lm_expert.norm.weight.dtype)
        batch_size, expert_len = expert.shape[:2]
        native_key = native_cache.get(0, {}).get("key_states")
        scene_key = scene_cache.get(0, {}).get("key_states")
        if not torch.is_tensor(native_key) or not torch.is_tensor(scene_key):
            raise ValueError("v5 Action Expert requires complete native and scene caches.")
        prefix_len = native_key.shape[1] + scene_key.shape[1]
        if position_ids.shape[1] == expert_len:
            expert_positions = position_ids
        elif position_ids.shape[1] == prefix_len + expert_len:
            expert_positions = position_ids[:, prefix_len:]
        else:
            raise ValueError(
                "v5 position_ids must cover either action-only or native+scene+action; "
                f"got {tuple(position_ids.shape)}, prefix={prefix_len}, action={expert_len}."
            )

        if attention_mask is None:
            suffix_mask = torch.ones(
                batch_size,
                expert_len,
                prefix_len + expert_len,
                dtype=torch.bool,
                device=expert.device,
            )
        elif attention_mask.shape[1] == expert_len:
            suffix_mask = attention_mask.to(device=expert.device, dtype=torch.bool)
        elif attention_mask.shape[1] == prefix_len + expert_len:
            suffix_mask = attention_mask[:, prefix_len:, :].to(
                device=expert.device, dtype=torch.bool
            )
        else:
            raise ValueError(
                "v5 attention rows must cover either action-only or full layout; "
                f"got {tuple(attention_mask.shape)}, prefix={prefix_len}, action={expert_len}."
            )
        if suffix_mask.shape[2] != prefix_len + expert_len:
            raise ValueError(
                "v5 action attention columns must cover native+scene+action; "
                f"got {suffix_mask.shape[2]} instead of {prefix_len + expert_len}."
            )

        use_gradient_checkpointing = bool(
            self.training
            and getattr(self, "gradient_checkpointing", False)
            and torch.is_grad_enabled()
        )
        if use_gradient_checkpointing:
            segment_size = int(
                getattr(self, "gradient_checkpointing_layers_per_segment", 1)
            )
            if segment_size < 1:
                raise ValueError("Gradient-checkpoint segment size must be positive.")
            layer_count = len(self.lm_expert.layers)
            for segment_start in range(0, layer_count, segment_size):
                segment_end = min(segment_start + segment_size, layer_count)
                scene_inputs: list[Tensor] = []
                for current_layer_idx in range(segment_start, segment_end):
                    scene_entry = scene_cache.get(current_layer_idx)
                    if scene_entry is None:
                        raise ValueError(
                            f"Scene cache is missing layer {current_layer_idx}."
                        )
                    scene_inputs.extend(
                        [scene_entry["key_states"], scene_entry["value_states"]]
                    )

                def checkpointed_action_segment(
                    expert_hidden: Tensor,
                    *segment_scene_tensors: Tensor,
                    start: int = segment_start,
                    end: int = segment_end,
                ) -> Tensor:
                    scene_tensor_index = 0
                    for current_layer_idx in range(start, end):
                        current_scene_key = segment_scene_tensors[scene_tensor_index]
                        current_scene_value = segment_scene_tensors[scene_tensor_index + 1]
                        scene_tensor_index += 2
                        expert_hidden = self._forward_action_layer_from_native_scene_memory(
                            current_layer_idx,
                            expert_hidden,
                            expert_positions,
                            suffix_mask,
                            native_cache[current_layer_idx],
                            current_scene_key,
                            current_scene_value,
                        )
                    return expert_hidden

                expert = checkpoint(
                    checkpointed_action_segment,
                    expert,
                    *scene_inputs,
                    use_reentrant=False,
                )
        else:
            for layer_idx in range(len(self.lm_expert.layers)):
                scene_entry = scene_cache.get(layer_idx)
                native_entry = native_cache.get(layer_idx)
                if scene_entry is None or native_entry is None:
                    raise ValueError(f"v5 prefix memory is missing layer {layer_idx}.")
                expert = self._forward_action_layer_from_native_scene_memory(
                    layer_idx,
                    expert,
                    expert_positions,
                    suffix_mask,
                    native_entry,
                    scene_entry["key_states"],
                    scene_entry["value_states"],
                )
        return self.lm_expert.norm(expert)

    def _build_v5_inference_cache(
        self,
        native_cache: dict[int, dict[str, Tensor]],
        scene_cache: dict[int, dict[str, Tensor]],
    ) -> dict[int, dict[str, Any]]:
        """Build one denoising cache, pre-projecting odd-layer cross K/V."""

        cache: dict[int, dict[str, Any]] = {}
        with torch.no_grad():
            for layer_idx, expert_layer in enumerate(self.lm_expert.layers):
                native_entry = native_cache[layer_idx]
                scene_entry = scene_cache[layer_idx]
                native_length = int(native_entry["key_states"].shape[1])
                scene_length = int(scene_entry["key_states"].shape[1])
                key = torch.cat(
                    [native_entry["key_states"], scene_entry["key_states"]], dim=1
                )
                value = torch.cat(
                    [native_entry["value_states"], scene_entry["value_states"]], dim=1
                )
                if layer_idx % self.self_attn_every_n_layers != 0:
                    assert isinstance(expert_layer.self_attn, Molmo2CrossAttention)
                    key, value = expert_layer.self_attn.project_prefix_kv(key, value)
                    cache[layer_idx] = {
                        "key_states": key.detach(),
                        "value_states": value.detach(),
                        "cross_attention_projected": True,
                        "cache_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
                        "native_length": native_length,
                        "scene_length": scene_length,
                    }
                else:
                    cache[layer_idx] = {
                        "key_states": key.detach(),
                        "value_states": value.detach(),
                        "cross_attention_projected": False,
                        "cache_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
                        "native_length": native_length,
                        "scene_length": scene_length,
                    }
        return cache

    def _validate_v5_inference_cache(
        self,
        cache: dict[int, dict[str, Any]],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> int:
        """Fail closed on stale/raw caches before a Full-Molmo denoising step."""

        expected_layers = len(self.lm_expert.layers)
        if set(cache) != set(range(expected_layers)):
            raise ValueError(
                "Full-Molmo2-ER v5 cache must contain every Expert layer exactly once; "
                f"expected={expected_layers}, keys={sorted(cache)}."
            )
        reference_shape: tuple[int, ...] | None = None
        reference_device: torch.device | None = None
        reference_dtype: torch.dtype | None = None
        native_length: int | None = None
        scene_length: int | None = None
        first_expert_attention = self.lm_expert.layers[0].self_attn
        expected_head_shape = (
            int(first_expert_attention.num_key_value_heads),
            int(first_expert_attention.head_dim),
        )
        for layer_idx in range(expected_layers):
            entry = cache[layer_idx]
            if entry.get("cache_topology") != FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY:
                raise ValueError(
                    f"Layer {layer_idx} does not contain a v5 Full-Molmo cache topology marker."
                )
            expected_projected = layer_idx % self.self_attn_every_n_layers != 0
            if entry.get("cross_attention_projected") is not expected_projected:
                raise ValueError(
                    "Full-Molmo2-ER cache projection marker disagrees with the Expert schedule "
                    f"at layer {layer_idx}."
                )
            key = entry.get("key_states")
            value = entry.get("value_states")
            if (
                not torch.is_tensor(key)
                or not torch.is_tensor(value)
                or key.shape != value.shape
                or key.ndim != 4
            ):
                raise ValueError(f"Layer {layer_idx} has malformed Full-Molmo K/V tensors.")
            if key.requires_grad or value.requires_grad:
                raise RuntimeError("Full-Molmo inference cache tensors must be detached.")
            if key.device != device or value.device != device:
                raise ValueError(
                    f"Layer {layer_idx} cache is on {key.device}/{value.device}, "
                    f"but the Action Expert is on {device}."
                )
            if key.dtype != dtype or value.dtype != dtype:
                raise ValueError(
                    f"Layer {layer_idx} cache uses {key.dtype}/{value.dtype}, "
                    f"but the Action Expert uses {dtype}."
                )
            if key.shape[2:] != expected_head_shape:
                raise ValueError(
                    f"Layer {layer_idx} cache head shape {tuple(key.shape[2:])} disagrees with "
                    f"the Expert contract {expected_head_shape}."
                )
            current_native_length = entry.get("native_length")
            current_scene_length = entry.get("scene_length")
            if type(current_native_length) is not int or type(current_scene_length) is not int:
                raise ValueError(f"Layer {layer_idx} is missing integer native/scene cache lengths.")
            if current_native_length < 1 or current_scene_length not in {2, 4}:
                raise ValueError(
                    f"Layer {layer_idx} has invalid native/scene lengths "
                    f"{current_native_length}/{current_scene_length}."
                )
            if key.shape[0] != batch_size or key.shape[1] != current_native_length + current_scene_length:
                raise ValueError(
                    f"Layer {layer_idx} cache shape {tuple(key.shape)} disagrees with "
                    f"batch/native/scene={batch_size}/{current_native_length}/{current_scene_length}."
                )
            if reference_shape is None:
                reference_shape = tuple(key.shape)
                reference_device = key.device
                reference_dtype = key.dtype
                native_length = current_native_length
                scene_length = current_scene_length
            elif (
                tuple(key.shape) != reference_shape
                or key.device != reference_device
                or key.dtype != reference_dtype
                or current_native_length != native_length
                or current_scene_length != scene_length
            ):
                raise ValueError(f"Layer {layer_idx} Full-Molmo cache metadata drifted across layers.")
        assert native_length is not None and scene_length is not None
        return native_length + scene_length

    def _forward_expert_from_frozen_memory(
        self,
        expert: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        cache: dict[int, dict[str, Any]],
    ) -> Tensor:
        """Run only trainable Expert layers against detached Molmo memory."""

        expert = expert.to(dtype=self.lm_expert.norm.weight.dtype)
        batch_size, expert_len = expert.shape[:2]
        first_key = cache.get(0, {}).get("key_states")
        if not torch.is_tensor(first_key):
            raise ValueError("Scheme-B Expert forward requires a complete frozen Molmo layer cache.")
        prefix_len = first_key.shape[1]
        if position_ids.shape[1] == expert_len:
            expert_positions = position_ids
        elif position_ids.shape[1] == prefix_len + expert_len:
            expert_positions = position_ids[:, prefix_len:]
        else:
            raise ValueError(
                "Scheme-B position_ids must cover either suffix-only or prefix+suffix layout; "
                f"got {tuple(position_ids.shape)}, prefix={prefix_len}, suffix={expert_len}."
            )

        if attention_mask is None:
            suffix_mask = torch.ones(
                batch_size,
                expert_len,
                prefix_len + expert_len,
                dtype=torch.bool,
                device=expert.device,
            )
        elif attention_mask.shape[1] == expert_len:
            suffix_mask = attention_mask.to(device=expert.device, dtype=torch.bool)
        elif attention_mask.shape[1] == prefix_len + expert_len:
            suffix_mask = attention_mask[:, prefix_len:, :].to(device=expert.device, dtype=torch.bool)
        else:
            raise ValueError(
                "Scheme-B attention rows must cover either suffix-only or prefix+suffix layout; "
                f"got {tuple(attention_mask.shape)}, prefix={prefix_len}, suffix={expert_len}."
            )

        for layer_idx, expert_layer in enumerate(self.lm_expert.layers):
            cached = cache.get(layer_idx)
            if cached is None:
                raise ValueError(f"Frozen Molmo cache is missing layer {layer_idx}.")
            cached_key = cached["key_states"]
            cached_value = cached["value_states"]
            if cached_key.requires_grad or cached_value.requires_grad:
                raise RuntimeError("Frozen Molmo memory must never require gradients.")

            expert_normalized = expert_layer.attn_norm(expert)
            is_self_attention = layer_idx % self.self_attn_every_n_layers == 0
            if is_self_attention:
                assert isinstance(expert_layer.self_attn, Molmo2FusedAttention)
                expert_query, expert_key, expert_value = expert_layer.self_attn.project(
                    expert_normalized,
                    expert_positions,
                )
                key = torch.cat([cached_key, expert_key], dim=1)
                value = torch.cat([cached_value, expert_value], dim=1)
                expert_output = _scaled_dot_product_attention(
                    expert_query,
                    key,
                    value,
                    suffix_mask,
                )
            else:
                assert isinstance(expert_layer.self_attn, Molmo2CrossAttention)
                relative_positions = expert_positions - expert_positions.min(dim=1, keepdim=True).values
                expert_query = expert_layer.self_attn.project_query(
                    expert_normalized,
                    relative_positions,
                )
                expert_key, expert_value = expert_layer.self_attn.project_prefix_kv(
                    cached_key,
                    cached_value,
                )
                expert_output = _scaled_dot_product_attention(
                    expert_query,
                    expert_key,
                    expert_value,
                    suffix_mask[:, :, :prefix_len],
                )
            expert = expert_layer.finish_attention(expert, expert_output)
        return self.lm_expert.norm(expert)

    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: dict[int, dict[str, Any]] | None = None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        prefix_scene_length: int = 0,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Any]] | None]:
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

        # Full-Molmo v5 keeps three logically distinct streams in the public
        # two-stream API: prefix=[native N, scene S], expert=actions A.  N is
        # evaluated once under no_grad, S is evaluated once with its tiny
        # input-autograd graph, and activation checkpointing recomputes A only.
        # This both protects the native Molmo representation and removes the
        # dominant frozen-prefix backward recomputation.
        prefix_scene_length = int(prefix_scene_length)
        if prefix_scene_length < 0:
            raise ValueError("prefix_scene_length must be non-negative.")
        if (
            prefix is not None
            and bool(getattr(self, "requires_native_scene_split", False))
            and prefix_scene_length == 0
        ):
            raise ValueError(
                "Full-Molmo2-ER v5 requires an explicit non-zero prefix_scene_length; "
                "refusing to fall back to the native-conditioned coupled-prefix graph."
            )
        if prefix_scene_length:
            if prefix is None:
                raise ValueError(
                    "prefix_scene_length is only valid while native+scene prefix embeddings are provided."
                )
            if prefix_scene_length >= prefix.shape[1]:
                raise ValueError(
                    "v5 prefix must contain at least one native token before scene tokens; "
                    f"got prefix={prefix.shape[1]}, scene={prefix_scene_length}."
                )
            if past_key_values and not fill_kv_cache:
                raise ValueError("v5 joint forward does not accept an existing prefix cache.")

            native_len = prefix.shape[1] - prefix_scene_length
            native = prefix[:, :native_len]
            scene = prefix[:, native_len:]
            if expert is not None and (use_cache or fill_kv_cache):
                raise ValueError(
                    "Full-Molmo2-ER v5 joint training forward does not produce a reusable cache; "
                    "set use_cache=False and fill_kv_cache=False."
                )
            native_positions = position_ids[:, :native_len]
            native_attention = self._mask_slice(
                attention_mask,
                slice(0, native_len),
                slice(0, native_len),
                batch_size=batch_size,
                rows=native_len,
                cols=native_len,
                device=native.device,
            )
            native_output, native_cache = self._prefill_native_vlm(
                native,
                native_positions,
                native_attention,
            )

            if expert is None:
                if use_cache != fill_kv_cache:
                    raise ValueError(
                        "Full-Molmo2-ER v5 prefix-only forward requires use_cache and "
                        "fill_kv_cache to be either both enabled or both disabled."
                    )
                # KV prefill is an inference boundary; avoid retaining the
                # trainable scene-projection graph across all denoising steps.
                with torch.no_grad():
                    scene_output, scene_cache = self._prefill_scene_from_native_memory(
                        scene,
                        position_ids,
                        attention_mask,
                        native_cache,
                    )
                    v5_cache = (
                        self._build_v5_inference_cache(native_cache, scene_cache)
                        if use_cache
                        else None
                    )
                return [torch.cat([native_output, scene_output], dim=1), None], v5_cache

            scene_output, scene_cache = self._prefill_scene_from_native_memory(
                scene,
                position_ids,
                attention_mask,
                native_cache,
            )
            expert_output = self._forward_action_from_native_scene_memory(
                expert,
                position_ids,
                attention_mask,
                native_cache,
                scene_cache,
            )
            return [torch.cat([native_output, scene_output], dim=1), expert_output], (
                past_key_values if use_cache else None
            )

        if bool(getattr(self, "inference_only_vlm", False)):
            prefix_output = None
            frozen_cache = past_key_values
            if prefix is not None:
                prefix_output, frozen_cache = self._prefill_inference_only_vlm(
                    prefix,
                    position_ids,
                    attention_mask,
                )
            if expert is None:
                if frozen_cache is None:
                    raise RuntimeError("Scheme-B VLM prefill failed to produce frozen memory.")
                return [prefix_output, None], frozen_cache
            if frozen_cache is None:
                raise ValueError("Scheme-B suffix forward requires frozen per-layer Molmo memory.")
            expert_output = self._forward_expert_from_frozen_memory(
                expert,
                position_ids,
                attention_mask,
                frozen_cache,
            )
            return [prefix_output, expert_output], frozen_cache if use_cache else past_key_values

        use_gradient_checkpointing = bool(
            self.training
            and getattr(self, "gradient_checkpointing", False)
            and torch.is_grad_enabled()
            and not use_cache
            and not fill_kv_cache
            and prefix is not None
            and expert is not None
        )
        if use_gradient_checkpointing:
            segment_size = int(
                getattr(self, "gradient_checkpointing_layers_per_segment", 1)
            )
            if segment_size < 1:
                raise ValueError("Gradient-checkpoint segment size must be positive.")
            layer_count = len(self.vlm.blocks)
            for segment_start in range(0, layer_count, segment_size):
                segment_end = min(segment_start + segment_size, layer_count)

                def checkpointed_segment(
                    prefix_hidden: Tensor,
                    expert_hidden: Tensor,
                    *,
                    start: int = segment_start,
                    end: int = segment_end,
                ) -> tuple[Tensor, Tensor]:
                    for current_layer_idx in range(start, end):
                        prefix_hidden, expert_hidden = (
                            self._forward_coupled_prefix_expert_layer(
                                current_layer_idx,
                                prefix_hidden,
                                expert_hidden,
                                position_ids,
                                attention_mask,
                            )
                        )
                    return prefix_hidden, expert_hidden

                prefix, expert = checkpoint(
                    checkpointed_segment,
                    prefix,
                    expert,
                    use_reentrant=False,
                )
            return [self.vlm.ln_f(prefix), self.lm_expert.norm(expert)], past_key_values

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
                if layer_idx == 0 and bool(getattr(self, "requires_native_scene_split", False)):
                    self._validate_v5_inference_cache(
                        past_key_values,
                        batch_size=batch_size,
                        device=expert.device,
                        dtype=expert.dtype,
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
                    if bool(
                        past_key_values[layer_idx].get("cross_attention_projected", False)
                    ):
                        expert_key, expert_value = cached_key, cached_value
                    else:
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
