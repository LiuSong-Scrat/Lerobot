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

"""Frozen full Molmo2-ER paired with a 36-layer SmolVLA Action Expert.

This backend is the executable counterpart of
``docs/SCHEME_B_FROZEN_FULL_MOLMO2ER_WEP_VLA_DESIGN.md``.  It intentionally reuses the
checkpoint-compatible text and Action Expert implementation from
``molmo2_with_expert`` while adding the *native* Molmo2 vision backbone and
connector.  The active source tensors are exactly:

* ``model.vision_backbone`` (ViT, learned 2-D pooling and projector);
* ``model.transformer.wte``;
* all ``model.transformer.blocks.0`` through ``blocks.35``;
* ``model.transformer.ln_f``.

``lm_head`` is never instantiated. Every base Molmo parameter is frozen and
held in evaluation mode; the optional V3-LoRA A/B tensors are the only
trainable parameters inside the Molmo decoder. Native vision/WTE embeddings are produced under
``torch.no_grad`` and detached, then trainable FG/BG tokens are inserted into
the prefix.  The 36 decoder blocks therefore remain parameter-frozen but are
part of the input-autograd graph, exactly like frozen WEPVLA: IMAGE/LANGUAGE
and FG/BG condition each other at every layer before Action Expert queries use
that evolving prefix.  Two-layer non-reentrant activation-checkpoint segments
keep this topology without retaining all decoder activations at once.

For a 256 x 256 square input, Molmo's processor emits two pixel-identical
378-square crops.  The backend asserts that equality at runtime and, when it
holds, encodes the ViT only once before reusing the selected -3/-9 features for
both crops.  Any failed assertion automatically falls back to the official
multi-crop path without changing the token template or connector computation.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from lerobot.policies.smolvla.configuration_smolvla import (
    FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
)
from lerobot.policies.smolvla.molmo2_with_expert import (
    MOLMO_V3_LORA_TARGET_MODULES,
    Molmo2ExpertBackbone,
    Molmo2TextBackbone,
    Molmo2TextSpec,
    Molmo2WithExpertModel,
    MolmoV3LoRALinear,
    _checkpoint_weight_map,
    _initialize_module,
    _resolve_device,
    _resolve_dtype,
    expected_molmo_v3_lora_parameters,
    get_intermediate_size,
    inject_molmo_v3_attention_lora,
    is_molmo_v3_lora_parameter_name,
    load_selective_molmo2_text_weights,
    validate_molmo2_er_text_contract,
)

_VISION_SOURCE_PREFIX = "model.vision_backbone."
_TEXT_SOURCE_PREFIX = "model.transformer."
_LM_HEAD_SOURCE_PREFIX = "lm_head."

# Exact active-backend parameter budget.  PointSeg, scene projections,
# PointActionFusion and action/time projections live in VLAFlowMatching and are
# therefore intentionally not included here.
_EXPECTED_VISION_PARAMETERS = 439_117_264
_EXPECTED_TEXT_PARAMETERS = 4_022_795_776
_EXPECTED_EXPERT_PARAMETERS = 1_736_591_232
_EXPECTED_FROZEN_PARAMETERS = _EXPECTED_VISION_PARAMETERS + _EXPECTED_TEXT_PARAMETERS
_EXPECTED_BACKEND_PARAMETERS = _EXPECTED_FROZEN_PARAMETERS + _EXPECTED_EXPERT_PARAMETERS

_EXPECTED_VISION_CONTRACT: dict[str, dict[str, Any]] = {
    "vit_config": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 72,
        "hidden_act": "gelu_pytorch_tanh",
        "image_default_input_size": (378, 378),
        "image_num_pos": 729,
        "image_patch_size": 14,
        "float32_attention": True,
    },
    "adapter_config": {
        "hidden_size": 1152,
        "intermediate_size": 9728,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 72,
        "hidden_act": "silu",
        "text_hidden_size": 2560,
        "vit_layers": (-3, -9),
        "pooling_attention_mask": True,
        "float32_attention": True,
    },
}

_EXPECTED_IMAGE_TOKEN_IDS = {
    "image_start_token_id": 151936,
    "image_end_token_id": 151937,
    "image_patch_id": 151938,
    "image_col_id": 151939,
    "low_res_image_start_token_id": 151940,
    "image_low_res_id": 151942,
}


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _normalise_contract_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def validate_molmo2_er_vision_contract(config: Any) -> None:
    """Reject a local model whose native vision/token contract has drifted."""

    mismatches: list[str] = []
    for section_name, expected_fields in _EXPECTED_VISION_CONTRACT.items():
        section = getattr(config, section_name, None)
        if section is None:
            mismatches.append(f"{section_name}: missing")
            continue
        for field_name, expected in expected_fields.items():
            actual = _normalise_contract_value(getattr(section, field_name, None))
            if actual != expected:
                mismatches.append(f"{section_name}.{field_name}: expected={expected!r}, actual={actual!r}")

    for field_name, expected in _EXPECTED_IMAGE_TOKEN_IDS.items():
        actual = getattr(config, field_name, None)
        if actual != expected:
            mismatches.append(f"{field_name}: expected={expected!r}, actual={actual!r}")

    if mismatches:
        raise ValueError(
            "The supplied checkpoint is not the frozen Molmo2-ER native-vision contract ("
            + ", ".join(mismatches)
            + ")."
        )


@contextmanager
def _parameter_creation_context(device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    """Create the remote-code vision module directly in its runtime dtype.

    Molmo's remote vision constructors do not expose a dtype argument.  A
    tightly scoped default-dtype context avoids first allocating 439M FP32
    parameters and then allocating a second BF16 copy.  The previous global
    default is restored even if construction fails.
    """

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous_dtype)


def _local_vision_backbone_class(model_directory: Path, config: Any) -> type[nn.Module]:
    auto_map = getattr(config, "auto_map", None)
    if not isinstance(auto_map, dict):
        raise ValueError("Molmo2-ER config must contain an auto_map dictionary for local modeling code.")
    class_reference = auto_map.get("AutoModelForImageTextToText")
    if not isinstance(class_reference, str):
        raise ValueError("Molmo2-ER config is missing AutoModelForImageTextToText remote-code mapping.")

    generation_class = get_class_from_dynamic_module(
        class_reference,
        str(model_directory),
        local_files_only=True,
    )
    modeling_module = sys.modules.get(generation_class.__module__)
    vision_class = getattr(modeling_module, "Molmo2VisionBackbone", None)
    if not isinstance(vision_class, type) or not issubclass(vision_class, nn.Module):
        raise TypeError(
            "Local Molmo2-ER modeling code does not expose a torch.nn.Module named Molmo2VisionBackbone."
        )
    return vision_class


def _instantiate_local_vision_backbone(
    model_directory: Path,
    config: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    vision_class = _local_vision_backbone_class(model_directory, config)
    with _parameter_creation_context(device, dtype):
        vision_backbone = vision_class(config.vit_config, config.adapter_config)
    return vision_backbone


def load_selective_molmo2_vision_weights(
    vision_backbone: nn.Module,
    model_directory: str | Path,
) -> dict[str, Any]:
    """Strictly load every native vision/connector tensor and nothing else."""

    model_directory = Path(model_directory).expanduser().resolve()
    weight_map = _checkpoint_weight_map(model_directory)
    target_state = vision_backbone.state_dict()
    source_for_target = {target_key: f"{_VISION_SOURCE_PREFIX}{target_key}" for target_key in target_state}

    missing_source = sorted(set(source_for_target.values()).difference(weight_map))
    source_vision_keys = {key for key in weight_map if key.startswith(_VISION_SOURCE_PREFIX)}
    unexpected_source = sorted(source_vision_keys.difference(source_for_target.values()))
    if missing_source or unexpected_source:
        raise RuntimeError(
            "Native Molmo vision state does not match the local checkpoint exactly: "
            f"missing={missing_source[:8]}, unexpected={unexpected_source[:8]}."
        )

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
                        raise RuntimeError("Cannot load Molmo vision weights into a meta-device model.")
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                    selected_parameters += target.numel()

    if selected_parameters != _EXPECTED_VISION_PARAMETERS:
        raise RuntimeError(
            "Loaded native Molmo vision parameter count drifted: "
            f"expected={_EXPECTED_VISION_PARAMETERS:,}, actual={selected_parameters:,}."
        )
    return {
        "model_directory": str(model_directory),
        "selected_tensors": len(source_for_target),
        "selected_parameters": selected_parameters,
        "strict_state_match": True,
        "includes_vit": True,
        "includes_connector": True,
    }


def _audit_source_checkpoint(model_directory: Path) -> dict[str, Any]:
    """Prove that the only source tensor omitted by the active model is lm_head."""

    weight_map = _checkpoint_weight_map(model_directory)
    grouped = {
        "vision_tensors": sum(key.startswith(_VISION_SOURCE_PREFIX) for key in weight_map),
        "text_tensors": sum(key.startswith(_TEXT_SOURCE_PREFIX) for key in weight_map),
        "lm_head_tensors": sum(key.startswith(_LM_HEAD_SOURCE_PREFIX) for key in weight_map),
    }
    grouped_total = sum(grouped.values())
    if grouped_total != len(weight_map) or grouped["lm_head_tensors"] != 1:
        unknown = sorted(
            key
            for key in weight_map
            if not key.startswith((_VISION_SOURCE_PREFIX, _TEXT_SOURCE_PREFIX, _LM_HEAD_SOURCE_PREFIX))
        )
        raise RuntimeError(
            "Molmo source checkpoint contains an unexpected top-level state layout: "
            f"total={len(weight_map)}, grouped={grouped_total}, unknown={unknown[:8]}."
        )
    return {
        "total_source_tensors": len(weight_map),
        **grouped,
        "physically_omitted_groups": ["lm_head"],
        "lm_head_instantiated": False,
    }


class Molmo2FullWithExpertModel(Molmo2WithExpertModel):
    """Full 36/36 base-frozen Molmo2-ER plus a 36-layer 0.75x Action Expert."""

    scale_input_embeddings = False
    inference_only_vlm = False

    def __init__(
        self,
        model_id: str,
        vlm_weights_path: str | None = None,
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = True,
        attention_mode: str = "cross_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = 36,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
        device: str | torch.device = "auto",
        torch_dtype: str | torch.dtype = torch.bfloat16,
        exact_vision_reuse: bool = True,
        inference_only_vlm: bool = False,
        gradient_checkpointing: bool = True,
        gradient_checkpointing_layers_per_segment: int = 2,
        molmo_lora_enable: bool = False,
        molmo_lora_rank: int = 8,
        molmo_lora_alpha: float = 8.0,
        molmo_lora_dropout: float = 0.0,
    ):
        # Do not call the half-backend initializer: its 18-layer checks are an
        # intentional contract for the historical point-only control.
        nn.Module.__init__(self)

        architecture_directory = Path(model_id).expanduser().resolve()
        weight_directory = (
            Path(vlm_weights_path).expanduser().resolve()
            if vlm_weights_path and str(vlm_weights_path).lower() not in {"none", "false", "off", "0"}
            else architecture_directory
        )
        if not architecture_directory.is_dir():
            raise ValueError(
                "Molmo2FullWithExpertModel requires a local Molmo2-ER directory containing "
                f"config, processor, modeling code and tokenizer; got {model_id!r}."
            )
        if not weight_directory.is_dir():
            raise ValueError(f"Local Molmo2-ER weight directory does not exist: {weight_directory}")
        if not freeze_vision_encoder:
            raise ValueError("Full-Molmo2-ER requires the native ViT and connector to remain frozen.")
        if not train_expert_only:
            raise ValueError("Full-Molmo2-ER freezes every base Molmo tensor; train_expert_only must be True.")
        if inference_only_vlm:
            raise ValueError(
                "WEP-compatible Full-Molmo2-ER requires molmo_inference_only=False: "
                "trainable FG/BG must remain in the VLM prefix and receive input gradients."
            )
        if num_vlm_layers != 36:
            raise ValueError(f"Full-Molmo2-ER retains exactly all 36/36 text blocks, got {num_vlm_layers}.")
        if num_expert_layers <= 0:
            num_expert_layers = num_vlm_layers
        if num_expert_layers != 36:
            raise ValueError(f"Full-Molmo2-ER requires a 36-layer Action Expert, got {num_expert_layers}.")
        if not math.isclose(expert_width_multiplier, 0.75, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "Full-Molmo2-ER requires expert_width_multiplier=0.75 (H=1920), "
                f"got {expert_width_multiplier}."
            )
        if "cross" not in attention_mode:
            raise ValueError(f"Full-Molmo2-ER requires alternating cross attention, got {attention_mode!r}.")
        if self_attn_every_n_layers != 2:
            raise ValueError(
                "Full-Molmo2-ER requires even Expert layers to use joint self-attention and odd "
                f"layers to use pure cross-attention; got interval {self_attn_every_n_layers}."
            )

        text_spec = Molmo2TextSpec.from_model_directory(architecture_directory)
        validate_molmo2_er_text_contract(text_spec)
        native_config = AutoConfig.from_pretrained(
            architecture_directory,
            trust_remote_code=True,
            local_files_only=True,
        )
        validate_molmo2_er_vision_contract(native_config)

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
        self.vision_backbone = _instantiate_local_vision_backbone(
            architecture_directory,
            native_config,
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

        # Frozen tensors are either fully overwritten below or initialized only
        # for explicit meta/CPU structure tests.  The trainable Expert is always
        # a fresh random initialization.
        if not load_vlm_weights:
            _initialize_module(self.vlm, text_spec.initializer_range)
        _initialize_module(self.lm_expert, expert_spec.initializer_range)

        source_report = _audit_source_checkpoint(weight_directory) if load_vlm_weights else None
        text_load_report = None
        vision_load_report = None
        if load_vlm_weights:
            text_load_report = load_selective_molmo2_text_weights(self.vlm, weight_directory)
            vision_load_report = load_selective_molmo2_vision_weights(self.vision_backbone, weight_directory)
        elif vlm_weights_path:
            raise ValueError("vlm_weights_path was provided but load_vlm_weights=False.")

        # Inject only after the official Molmo tensors have been loaded.  The
        # source checkpoint intentionally has no LoRA keys, while policy
        # checkpoints produced by this variant persist the adapters normally.
        self.molmo_lora_enable = bool(molmo_lora_enable)
        self.molmo_lora_rank = int(molmo_lora_rank)
        self.molmo_lora_alpha = float(molmo_lora_alpha)
        self.molmo_lora_dropout = float(molmo_lora_dropout)
        self.molmo_lora_target_modules = MOLMO_V3_LORA_TARGET_MODULES
        self.molmo_lora_module_names: tuple[str, ...] = ()
        if self.molmo_lora_enable:
            self.molmo_lora_module_names = inject_molmo_v3_attention_lora(
                self.vlm,
                rank=self.molmo_lora_rank,
                alpha=self.molmo_lora_alpha,
                dropout=self.molmo_lora_dropout,
            )
        self.molmo_lora_parameter_count = (
            expected_molmo_v3_lora_parameters(
                rank=self.molmo_lora_rank,
                num_layers=num_vlm_layers,
            )
            if self.molmo_lora_enable
            else 0
        )

        self.processor = AutoProcessor.from_pretrained(
            architecture_directory,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        tokenizer = self.processor.tokenizer
        # Legacy initialization reads these attributes even though the full
        # native path never uses SmolVLM's synthetic image tokens.
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
        self.image_patch_id = int(native_config.image_patch_id)
        self.image_start_token_id = int(native_config.image_start_token_id)
        self.image_end_token_id = int(native_config.image_end_token_id)
        self.image_col_id = int(native_config.image_col_id)
        self.num_vlm_layers = num_vlm_layers
        self.num_expert_layers = num_expert_layers
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.attention_mode = attention_mode
        self.train_expert_only = train_expert_only
        self.freeze_vision_encoder = freeze_vision_encoder
        self.expert_hidden_size = expert_hidden_size
        self.num_attention_heads = text_spec.num_attention_heads
        self.num_key_value_heads = text_spec.num_key_value_heads
        self.head_dim = text_spec.head_dim
        self.text_spec = text_spec
        self.expert_spec = expert_spec
        self.native_config = native_config
        self.exact_vision_reuse = bool(exact_vision_reuse)
        self.inference_only_vlm = False
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_checkpointing_layers_per_segment = int(
            gradient_checkpointing_layers_per_segment
        )
        if not 1 <= self.gradient_checkpointing_layers_per_segment <= self.num_vlm_layers:
            raise ValueError(
                "gradient_checkpointing_layers_per_segment must be between one and "
                f"{self.num_vlm_layers}."
            )
        self.vision_reuse_calls = 0
        self.vision_fallback_calls = 0
        self.last_vision_encode_report: dict[str, Any] | None = None
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(**asdict(text_spec)),
            vision_config=native_config.vit_config,
            adapter_config=native_config.adapter_config,
            model_type="molmo2_full_smolvla",
        )
        # Dataclass fields plus a derived value expected by existing SmolVLA
        # initialization code.
        self.config.text_config.total_vocab_size = text_spec.total_vocab_size

        actual_counts = {
            "vision": _parameter_count(self.vision_backbone),
            "text": _parameter_count(self.vlm),
            "expert": _parameter_count(self.lm_expert),
        }
        expected_counts = {
            "vision": _EXPECTED_VISION_PARAMETERS,
            "text": _EXPECTED_TEXT_PARAMETERS + self.molmo_lora_parameter_count,
            "expert": _EXPECTED_EXPERT_PARAMETERS,
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                "Full-Molmo2-ER active parameter contract drifted: "
                f"expected={expected_counts}, actual={actual_counts}."
            )

        self.load_report: dict[str, Any] = {
            "loaded": bool(load_vlm_weights),
            "architecture_directory": str(architecture_directory),
            "weight_directory": str(weight_directory),
            "strict_local_only": True,
            "text": text_load_report,
            "vision_and_connector": vision_load_report,
            "source_checkpoint": source_report,
            "active_parameter_counts": {
                **actual_counts,
                "frozen_molmo_base": _EXPECTED_FROZEN_PARAMETERS,
                "trainable_molmo_lora": self.molmo_lora_parameter_count,
                "trainable_expert": actual_counts["expert"],
                "backend_total": sum(actual_counts.values()),
            },
            "lm_head_instantiated": False,
        }
        self.set_requires_grad()

    @property
    def architecture_contract(self) -> dict[str, Any]:
        return {
            "full_molmo_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
            "source_model": "Molmo2-ER",
            "source_vlm_layers": self.text_spec.num_hidden_layers,
            "retained_vlm_layers": self.num_vlm_layers,
            "retained_vlm_layer_indices": list(range(36)),
            "vlm_hidden_size": self.text_spec.hidden_size,
            "vlm_intermediate_size": self.text_spec.intermediate_size,
            "expert_layers": self.num_expert_layers,
            "expert_hidden_size": self.expert_spec.hidden_size,
            "expert_intermediate_size": self.expert_spec.intermediate_size,
            "self_attention_layers": list(range(0, 36, 2)),
            "cross_attention_layers": list(range(1, 36, 2)),
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.text_spec.rope_theta,
            "vision_module_present": True,
            "vision_native_processor": True,
            "vision_native_crop_size": (378, 378),
            "vision_selected_layers": (-3, -9),
            "vision_features_for_256_square_rgb": 392,
            "native_image_positions_for_256_square_rgb": 410,
            "exact_identical_two_crop_reuse": self.exact_vision_reuse,
            "total_vocab_size": self.text_spec.total_vocab_size,
            "wte_applied_to_all_native_tokens": True,
            "vision_feature_fusion": "add_only_at_image_patch_id",
            "molmo_frozen_eval": True,
            "vlm_execution": (
                "frozen_base_with_lora_and_prefix_input_autograd"
                if self.molmo_lora_enable
                else "frozen_parameters_with_prefix_input_autograd"
            ),
            "per_layer_memory": "evolving_image_text_fg_bg_prefix",
            "fg_bg_location": "trainable_vlm_prefix",
            "action_location": "expert_suffix_only",
            "text_prefix_autograd_preserved": True,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_layers_per_segment": (
                self.gradient_checkpointing_layers_per_segment
            ),
            "molmo_lora_enable": self.molmo_lora_enable,
            "molmo_lora_rank": self.molmo_lora_rank,
            "molmo_lora_alpha": self.molmo_lora_alpha,
            "molmo_lora_dropout": self.molmo_lora_dropout,
            "molmo_lora_target_modules": self.molmo_lora_target_modules,
            "molmo_lora_module_count": len(self.molmo_lora_module_names),
            "molmo_lora_parameter_count": self.molmo_lora_parameter_count,
            "base_molmo_parameters_frozen": True,
            "lm_head_present": False,
            "backend_parameters": _EXPECTED_BACKEND_PARAMETERS + self.molmo_lora_parameter_count,
            "frozen_molmo_parameters": _EXPECTED_FROZEN_PARAMETERS,
            "trainable_expert_parameters": _EXPECTED_EXPERT_PARAMETERS,
            "trainable_molmo_lora_parameters": self.molmo_lora_parameter_count,
        }

    def named_molmo_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.vlm.named_parameters():
            if is_molmo_v3_lora_parameter_name(name):
                yield f"vlm.{name}", parameter

    def named_frozen_molmo_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.vlm.named_parameters():
            if not is_molmo_v3_lora_parameter_name(name):
                yield f"vlm.{name}", parameter
        for name, parameter in self.vision_backbone.named_parameters():
            yield f"vision_backbone.{name}", parameter

    def set_requires_grad(self) -> None:
        if not self.train_expert_only or not self.freeze_vision_encoder:
            raise ValueError("Full-Molmo2-ER requires a frozen vision/text backbone and trainable Expert.")
        self.vlm.requires_grad_(False)
        self.vision_backbone.requires_grad_(False)
        self.lm_expert.requires_grad_(True)
        for _, parameter in self.named_molmo_lora_parameters():
            parameter.requires_grad_(True)
        self.vlm.eval()
        self.vision_backbone.eval()

    def train(self, mode: bool = True) -> Molmo2FullWithExpertModel:
        # Molmo remains deterministic and parameter-frozen in every train mode.
        # Decoder input autograd is intentionally preserved for prefix FG/BG.
        super().train(mode)
        self.vlm.eval()
        self.vision_backbone.eval()
        # Base Molmo stays in eval mode.  Only adapter dropout follows the
        # policy mode; the registered default is zero for deterministic V3.
        for module in self.vlm.modules():
            if isinstance(module, MolmoV3LoRALinear):
                module.lora_dropout.train(mode)
        return self

    @staticmethod
    def _visual_input(visual_inputs: Mapping[str, Tensor], field_name: str) -> Tensor:
        """Resolve both raw field names and ``observation.molmo.*`` constants."""

        direct = visual_inputs.get(field_name)
        if torch.is_tensor(direct):
            return direct
        suffix = f".{field_name}"
        matches = [value for key, value in visual_inputs.items() if str(key).endswith(suffix)]
        if len(matches) != 1 or not torch.is_tensor(matches[0]):
            available = sorted(str(key) for key in visual_inputs)
            raise KeyError(
                f"Expected one native Molmo visual tensor ending in {suffix!r}; available={available}."
            )
        return matches[0]

    def _build_batched_images(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        image_token_pooling: Tensor,
        image_grids: Tensor,
        image_num_crops: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reproduce Molmo2Model.build_batched_images without a full LM model."""

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,L], got {tuple(input_ids.shape)}.")
        if pixel_values.ndim != 3:
            raise ValueError(
                "Molmo pixel_values must be flattened native crops [total_crops,patches,pixels], "
                f"got {tuple(pixel_values.shape)}."
            )
        if image_token_pooling.ndim != 2:
            raise ValueError(
                f"image_token_pooling must be [total_features,pool_width], got {image_token_pooling.shape}."
            )
        if image_grids.ndim != 2 or image_grids.shape[1] != 4:
            raise ValueError(f"image_grids must be [num_images,4], got {tuple(image_grids.shape)}.")
        if image_num_crops.ndim != 1:
            raise ValueError(f"image_num_crops must be [num_images], got {tuple(image_num_crops.shape)}.")

        device = self.vision_backbone.device
        input_ids = input_ids.to(device=device)
        pixel_values = pixel_values.to(device=device)
        image_token_pooling = image_token_pooling.to(device=device, dtype=torch.long)
        image_grids = image_grids.to(device=device, dtype=torch.long)
        image_num_crops = image_num_crops.to(device=device, dtype=torch.long)

        raw_counts = (input_ids == self.image_end_token_id).sum(dim=1)
        if bool((raw_counts % 2 != 0).any()):
            raise ValueError(
                "Each native Molmo image must have global/high-resolution image-end markers; "
                f"got per-sample counts {raw_counts.tolist()}."
            )
        counts = raw_counts // 2
        batch_size = input_ids.shape[0]
        num_images = int(counts.sum().item())
        if image_grids.shape[0] != num_images or image_num_crops.shape[0] != num_images:
            raise ValueError(
                "Native Molmo image metadata count mismatch: "
                f"input_ids imply {num_images}, grids={image_grids.shape[0]}, "
                f"num_crops={image_num_crops.shape[0]}."
            )

        pooled_per_image = image_grids[:, :2].prod(dim=1) + image_grids[:, 2:].prod(dim=1)
        example_for_image = torch.arange(batch_size, device=device).repeat_interleave(counts)
        crops_per_example = torch.zeros(batch_size, dtype=torch.long, device=device)
        crops_per_example.index_add_(0, example_for_image, image_num_crops)
        patches_per_image = image_num_crops * pixel_values.shape[1]

        counts_list = counts.tolist()
        offsets_per_example: list[list[int]] = []
        image_offset = 0
        for count in counts_list:
            per_image = patches_per_image[image_offset : image_offset + count]
            offsets_per_example.append([0] + per_image.cumsum(0).tolist()[:-1])
            image_offset += count

        pooled_per_example = torch.zeros(batch_size, dtype=torch.long, device=device)
        pooled_per_example.index_add_(0, example_for_image, pooled_per_image)
        total_crops = int(crops_per_example.sum().item())
        total_pooled = int(pooled_per_example.sum().item())
        if total_crops != pixel_values.shape[0] or total_pooled != image_token_pooling.shape[0]:
            raise ValueError(
                "Native Molmo flattened visual tensors are inconsistent: "
                f"expected crops/features={total_crops}/{total_pooled}, "
                f"actual={pixel_values.shape[0]}/{image_token_pooling.shape[0]}."
            )

        # Registered training/evaluation uses exactly one image and two crops
        # per sample.  In that contract the flattened processor tensors are
        # already batch-major, so reshape them as views instead of allocating
        # and copying a second full pixel/pooling buffer on every rank.
        fixed_single_image_two_crop = bool(
            (counts == 1).all()
            and (image_num_crops == 2).all()
            and (crops_per_example == 2).all()
            and (pooled_per_example == pooled_per_example[0]).all()
            and pixel_values.is_contiguous()
            and image_token_pooling.is_contiguous()
        )
        if fixed_single_image_two_crop:
            images = pixel_values.view(
                batch_size,
                2,
                pixel_values.shape[1],
                pixel_values.shape[2],
            )
            pooling = image_token_pooling.view(
                batch_size,
                int(pooled_per_example[0].item()),
                image_token_pooling.shape[1],
            )
            return images, pooling

        max_crops = int(crops_per_example.max().item())
        images = torch.full(
            (batch_size, max_crops, pixel_values.shape[1], pixel_values.shape[2]),
            -1,
            dtype=pixel_values.dtype,
            device=device,
        )
        crop_offset = 0
        for batch_index in range(batch_size):
            count = int(crops_per_example[batch_index].item())
            images[batch_index, :count] = pixel_values[crop_offset : crop_offset + count]
            crop_offset += count

        max_pooled = int(pooled_per_example.max().item())
        pooling = torch.full(
            (batch_size, max_pooled, image_token_pooling.shape[1]),
            -1,
            dtype=torch.long,
            device=device,
        )
        pool_offset = 0
        image_offset = 0
        for batch_index, image_count in enumerate(counts_list):
            count = int(pooled_per_example[batch_index].item())
            current = image_token_pooling[pool_offset : pool_offset + count].clone()
            local_offset = 0
            per_image_pooled = pooled_per_image[image_offset : image_offset + image_count]
            for index_offset, num_pooled in zip(
                offsets_per_example[batch_index], per_image_pooled.tolist(), strict=True
            ):
                current_slice = current[local_offset : local_offset + num_pooled]
                current[local_offset : local_offset + num_pooled] = torch.where(
                    current_slice >= 0,
                    current_slice + int(index_offset),
                    current_slice,
                )
                local_offset += num_pooled
            pooling[batch_index, :count] = current
            pool_offset += count
            image_offset += image_count
        return images, pooling

    def _pool_and_project_preencoded(self, image_features: Tensor, pooled_patches_idx: Tensor) -> Tensor:
        """Run the untouched native connector on already reused ViT features."""

        vision = self.vision_backbone
        image_features = vision.image_feature_dropout(image_features)
        batch_size = image_features.shape[0]
        feature_dim = image_features.shape[-1]
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)
        batch_idx = torch.arange(
            pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device
        )
        batch_idx = torch.tile(
            batch_idx.view(batch_size, 1, 1),
            [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]],
        )
        to_pool = image_features.reshape(batch_size, -1, feature_dim)[
            batch_idx, torch.clip(pooled_patches_idx, 0)
        ]
        to_pool = to_pool * valid.to(dtype=vision.dtype)[:, :, :, None]
        to_pool = to_pool.reshape(-1, pooled_patches_idx.shape[-1], feature_dim)
        if vision.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape(-1, 1, 1, valid.shape[-1])
            denominator = valid.view(-1, to_pool.shape[-2]).float().sum(dim=-1)
            denominator = torch.where(denominator == 0, 1, denominator)
            query = to_pool.sum(dim=-2, keepdim=True) / denominator[:, None, None].to(to_pool.dtype)
        else:
            attn_mask = None
            query = to_pool.mean(dim=-2, keepdim=True)
        pooled = vision.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled = pooled.reshape(batch_size, -1, pooled.shape[-1])
        pooled = vision.image_projector(pooled)
        return pooled.reshape(-1, pooled.shape[-1])[valid_token.flatten()]

    def _encode_native_vision(self, images: Tensor, pooled_patches_idx: Tensor) -> Tensor:
        images = images.to(device=self.vision_backbone.device, dtype=self.vision_backbone.dtype)
        pooled_patches_idx = pooled_patches_idx.to(device=self.vision_backbone.device, dtype=torch.long)
        fallback_reason: str | None = None
        if self.exact_vision_reuse:
            try:
                assert images.ndim == 4, f"expected [B,crops,patches,pixels], got {images.shape}"
                assert images.shape[1] == 2, f"expected exactly two crops, got {images.shape[1]}"
                assert torch.equal(images[:, 0], images[:, 1]), (
                    "global and high-resolution crops are not pixel-identical"
                )
                encoded_once = self.vision_backbone.encode_image(images[:, :1])
                reused_features = encoded_once.expand(-1, 2, -1, -1)
                output = self._pool_and_project_preencoded(reused_features, pooled_patches_idx)
                self.vision_reuse_calls += 1
                self.last_vision_encode_report = {
                    "path": "exact_single_encode_reuse",
                    "input_crops_per_sample": 2,
                    "vit_encoded_crops_per_sample": 1,
                    "pixel_identity_asserted": True,
                    "output_features": int(output.shape[0]),
                }
                return output
            except AssertionError as error:
                # Shape/equality assertions define optimization eligibility,
                # not input validity.  Native Molmo remains the exact fallback.
                fallback_reason = str(error)

        output = self.vision_backbone(images, pooled_patches_idx)
        self.vision_fallback_calls += 1
        self.last_vision_encode_report = {
            "path": "native_multi_crop_fallback",
            "input_crops_per_sample": int(images.shape[1]),
            "vit_encoded_crops_per_sample": int(images.shape[1]),
            "pixel_identity_asserted": False,
            "fallback_reason": fallback_reason or "exact reuse disabled",
            "output_features": int(output.shape[0]),
        }
        return output

    def embed_multimodal_tokens(
        self,
        input_ids: Tensor,
        visual_inputs: Mapping[str, Tensor],
        *,
        disable_vision: bool = False,
    ) -> Tensor:
        """Return native ``WTE(all tokens) + vision(<im_patch> only)`` embeddings.

        ``visual_inputs`` accepts the registered ``observation.molmo.*`` keys
        as well as their unprefixed processor field names.  Structural image
        start/end/column tokens retain their WTE embeddings; connector features
        are added only where ``input_ids == image_patch_id``.
        """

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,L], got {tuple(input_ids.shape)}.")
        embedding_device = self.vlm.wte.embedding.device
        native_input_ids = input_ids.to(device=embedding_device, dtype=torch.long)
        if bool((native_input_ids < -1).any()):
            raise ValueError("Molmo input_ids may use -1 only as a padding sentinel.")
        if bool((native_input_ids >= self.text_spec.total_vocab_size).any()):
            raise IndexError(
                "Molmo input token is outside the complete base+additional vocabulary "
                f"[0, {self.text_spec.total_vocab_size})."
            )
        padding = native_input_ids == -1
        safe_input_ids = native_input_ids.masked_fill(padding, 0)

        # Both WTE and vision are frozen and have no differentiable inputs.
        # Autograd resumes after trainable scene tokens are inserted upstream.
        with torch.no_grad():
            embeddings = self.vlm.wte(safe_input_ids)
            embeddings = embeddings.masked_fill(padding.unsqueeze(-1), 0)
            if disable_vision:
                self.last_vision_encode_report = {
                    "path": "disabled_ablation",
                    "output_features": 0,
                    "wte_image_template_preserved": True,
                }
                return self.vlm.emb_drop(embeddings)

            pixel_values = self._visual_input(visual_inputs, "pixel_values")
            image_token_pooling = self._visual_input(visual_inputs, "image_token_pooling")
            image_grids = self._visual_input(visual_inputs, "image_grids")
            image_num_crops = self._visual_input(visual_inputs, "image_num_crops")
            images, pooled_patches_idx = self._build_batched_images(
                safe_input_ids,
                pixel_values,
                image_token_pooling,
                image_grids,
                image_num_crops,
            )
            image_features = self._encode_native_vision(images, pooled_patches_idx).to(
                device=embedding_device,
                dtype=embeddings.dtype,
            )
            is_image_patch = safe_input_ids == self.image_patch_id
            patch_count = int(is_image_patch.sum().item())
            if patch_count != image_features.shape[0]:
                raise RuntimeError(
                    "Molmo native image feature/template mismatch: "
                    f"patch_positions={patch_count}, connector_features={image_features.shape[0]}."
                )
            embeddings = embeddings.clone()
            embeddings.view(-1, embeddings.shape[-1])[is_image_patch.flatten()] += image_features
            return self.vlm.emb_drop(embeddings)

    def embed_image(self, image: Tensor) -> Tensor:
        del image
        raise RuntimeError(
            "Full-Molmo2-ER requires native processor token/crop/pooling metadata; "
            "call embed_multimodal_tokens instead of the legacy embed_image path."
        )


# Descriptive aliases for architecture audits and experiment manifests.
Molmo2FullVisionWithExpertModel = Molmo2FullWithExpertModel
Molmo2ERFullWithExpertModel = Molmo2FullWithExpertModel


__all__ = [
    "Molmo2ERFullWithExpertModel",
    "Molmo2FullVisionWithExpertModel",
    "Molmo2FullWithExpertModel",
    "load_selective_molmo2_vision_weights",
    "validate_molmo2_er_vision_contract",
]
