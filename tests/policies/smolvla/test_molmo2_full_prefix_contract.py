#!/usr/bin/env python

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import _validate_full_molmo_checkpoint_topology
from lerobot.policies.smolvla import molmo2_with_expert as molmo_core
from lerobot.policies.smolvla.configuration_smolvla import (
    FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
    SmolVLAConfig,
)
from lerobot.policies.smolvla.constants import (
    FULL_MOLMO2ER_EXPERT_HIDDEN_SIZE,
    FULL_MOLMO2ER_EXPERT_INTERMEDIATE_SIZE,
)
from lerobot.policies.smolvla.modeling_smolvla import (
    FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET,
    FULL_MOLMO2ER_WORLDFLOW_OFF_PARAMETER_BUDGET,
    FULL_MOLMO2ER_WORLDFLOW_ON_PARAMETER_BUDGET,
    VLAFlowMatching,
    WorldFlowActionBranch,
    expected_full_molmo2er_parameter_budget,
    full_molmo2er_trainable_parameter_prefixes,
    make_att_2d_masks,
)
from lerobot.policies.smolvla.molmo2_full_with_expert import Molmo2FullWithExpertModel

TEXT_ROLE = 0
IMAGE_ROLE = 1
SCENE_ROLE = 2
PAD_ROLE = 3
NATIVE_IMAGE_POSITIONS = 410


def _full_worldflow_config(**overrides) -> SmolVLAConfig:
    values = {
        "vlm_backend": "molmo2_full",
        "vlm_model_name": "/does/not/load/during/config-test",
        "vlm_weights_path": "/does/not/load/during/config-test",
        "load_vlm_weights": True,
        "train_expert_only": True,
        "freeze_vision_encoder": True,
        "camera_views": "agentview",
        "rgb_camera_views": "agentview",
        "num_vlm_layers": 36,
        "num_expert_layers": 36,
        "expert_width_multiplier": 0.28125,
        "pointseg_enable": True,
        "pointseg_grid_size": 0.01,
        "pointseg_feature_dim": 64,
        "pointseg_foreground_ratio": 0.025,
        "pointseg_background_ratio": 0.025,
        "pointseg_min_foreground_points": 2500,
        "pointseg_min_background_points": 0,
        "pointseg_aux_loss_weight": 0.0005,
        "pointseg_use_temporal_priors_as_input": False,
        "pointseg_use_pseudo_selection": False,
        "worldflow_enable": True,
        "worldflow_target_type": "world_eef_trajectory",
        "worldflow_world_eef_velocity_mode": "base_pose9_euclidean",
        "worldflow_reference_frame": "robot_base",
        "worldflow_frame_origin": "global",
        "worldflow_scene_frame_origin": "global",
        "worldflow_noise_coupling": "left_compose_ego",
        "worldflow_action_fusion": "point_action_expert_conjugate_bridge",
        "worldflow_action_expert_mode": "shared",
        "worldflow_current_ee_pose_token": False,
        "worldflow_bootstrap_from_ego": False,
        "worldflow_freeze_pretrained_ego": False,
        "worldflow_training_coordinate_frame_augmentation": False,
        "worldflow_feature_dim": 64,
        "worldflow_grid_size": 0.01,
        "worldflow_loss_weight": 1.0,
        "worldflow_geo_loss_weight": 0.0,
        "worldflow_bridge_loss_weight": 0.0,
        "worldflow_equiv_loss_weight": 0.0,
        "worldflow_pretrained_lr_multiplier": 1.0,
        "worldflow_new_lr_multiplier": 1.0,
        "worldflow_max_points": 2048,
        "worldflow_require_action_target_sidecar": True,
        "worldflow_ego_residual_gate_init": None,
        "scheduler_decay_steps": 30_000,
        "scheduler_decay_lr": 3e-5,
        "device": "cpu",
    }
    values.update(overrides)
    return SmolVLAConfig(**values)


def test_full_config_locks_v043_multiview_doubleflow_contract() -> None:
    config = _full_worldflow_config()
    assert config.selected_camera_views == ("agentview",)
    assert config.selected_rgb_camera_views == ("agentview",)
    assert not config.molmo_inference_only and config.molmo_image_fast_path
    assert config.molmo_gradient_checkpointing
    assert config.molmo_gradient_checkpointing_layers_per_segment == 2
    assert config.full_molmo_topology == FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY
    assert config.full_molmo_topology == "v3_feature_align_language_casual"
    assert config.num_vlm_layers == config.num_expert_layers == 36
    assert config.worldflow_action_fusion == "point_action_expert_conjugate_bridge"
    assert config.worldflow_action_expert_mode == "shared"
    assert config.worldflow_noise_coupling == "left_compose_ego"
    assert config.worldflow_target_type == "world_eef_trajectory"
    assert config.worldflow_world_eef_velocity_mode == "base_pose9_euclidean"
    assert config.pose9_action_noise_enable is False


def test_full_config_allows_pose9_action_noise_opt_in() -> None:
    config = _full_worldflow_config(pose9_action_noise_enable=True)
    assert config.pose9_action_noise_enable is True


def test_full_config_allows_native_v3_lora_without_generic_peft() -> None:
    config = _full_worldflow_config(
        molmo_lora_enable=True,
        molmo_lora_rank=8,
        molmo_lora_alpha=8.0,
        molmo_lora_dropout=0.0,
        molmo_lora_lr_multiplier=0.1,
    )
    assert config.molmo_lora_enable and not config.use_peft
    assert config.molmo_lora_rank == 8
    assert config.molmo_lora_lr_multiplier == 0.1


@pytest.mark.parametrize("invalid_rank", [True, 0, -1, 1.5])
def test_full_config_rejects_invalid_v3_lora_rank(invalid_rank) -> None:
    with pytest.raises(ValueError, match="molmo_lora_rank"):
        _full_worldflow_config(molmo_lora_enable=True, molmo_lora_rank=invalid_rank)


def test_full_config_allows_cli_scheduler_override() -> None:
    config = _full_worldflow_config(
        scheduler_warmup_steps=50,
        scheduler_decay_steps=1_500,
        scheduler_decay_lr=1e-5,
    )
    assert config.scheduler_warmup_steps == 50
    assert config.scheduler_decay_steps == 1_500
    assert config.scheduler_decay_lr == 1e-5


def test_full_checkpoint_topology_gate_rejects_old_and_accepts_native_readout(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"

    config_path.write_text(
        json.dumps({"vlm_backend": "molmo2_full", "molmo_inference_only": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="topology"):
        _validate_full_molmo_checkpoint_topology(checkpoint)

    for invalid_topology in (
        None,
        "detached_scene_suffix_v2",
        "wepvla_scene_in_vlm_prefix_v3",
        "wepvla_scene_in_vlm_prefix_v3_feature_align",
        "unknown",
    ):
        legacy_config = {
            "vlm_backend": "molmo2_full",
            "molmo_inference_only": False,
        }
        if invalid_topology is not None:
            legacy_config["full_molmo_topology"] = invalid_topology
        config_path.write_text(json.dumps(legacy_config), encoding="utf-8")
        with pytest.raises(ValueError, match="topology"):
            _validate_full_molmo_checkpoint_topology(checkpoint)

    config_path.write_text(
        json.dumps(
            {
                "vlm_backend": "molmo2_full",
                "molmo_inference_only": False,
                "full_molmo_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
            }
        ),
        encoding="utf-8",
    )
    _validate_full_molmo_checkpoint_topology(checkpoint)


def test_public_config_loader_rejects_missing_raw_topology_before_defaults(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config = _full_worldflow_config()
    config._save_pretrained(checkpoint)
    config_path = checkpoint / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config.pop("full_molmo_topology")
    raw_config["molmo_inference_only"] = False
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")

    with pytest.raises(ValueError, match="topology"):
        PreTrainedConfig.from_pretrained(checkpoint)


def test_public_config_loader_accepts_registered_native_readout_topology(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config = _full_worldflow_config()
    config._save_pretrained(checkpoint)

    loaded = PreTrainedConfig.from_pretrained(checkpoint)
    assert isinstance(loaded, SmolVLAConfig)
    assert loaded.full_molmo_topology == FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("molmo_inference_only", True),
        ("molmo_image_fast_path", False),
        ("worldflow_enable", False),
        ("worldflow_noise_coupling", "conjugate_ego"),
        ("worldflow_action_fusion", "cross_attention"),
        ("worldflow_action_expert_mode", "independent"),
        ("worldflow_require_action_target_sidecar", False),
        ("worldflow_max_points", 0),
        ("molmo_gradient_checkpointing_layers_per_segment", 0),
        ("expert_width_multiplier", 0.75),
    ),
)
def test_full_config_rejects_registered_contract_drift(field_name: str, bad_value) -> None:
    with pytest.raises(ValueError):
        _full_worldflow_config(**{field_name: bad_value})


def test_parameter_budget_matches_v043_shared_doubleflow() -> None:
    off = expected_full_molmo2er_parameter_budget(worldflow_enable=False)
    on = expected_full_molmo2er_parameter_budget(worldflow_enable=True)
    on_lora = expected_full_molmo2er_parameter_budget(
        worldflow_enable=True,
        molmo_lora_enable=True,
        molmo_lora_rank=8,
    )
    assert off == FULL_MOLMO2ER_WORLDFLOW_OFF_PARAMETER_BUDGET == {
        "total": 4_915_225_982,
        "trainable": 440_554_697,
        "frozen": 4_474_671_285,
    }
    assert on == FULL_MOLMO2ER_WORLDFLOW_ON_PARAMETER_BUDGET == {
        "total": 4_955_217_630,
        "trainable": 467_816_260,
        "frozen": 4_487_401_370,
    }
    assert on_lora == {
        "total": 4_959_588_062,
        "trainable": 472_186_692,
        "frozen": 4_487_401_370,
        "molmo_lora": 4_370_432,
    }
    assert {name: on[name] - off[name] for name in on} == (
        FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET
    ) == {
        "total": 39_991_648,
        "trainable": 27_261_563,
        "frozen": 12_730_085,
    }
    off_prefixes = full_molmo2er_trainable_parameter_prefixes(worldflow_enable=False)
    on_prefixes = full_molmo2er_trainable_parameter_prefixes(worldflow_enable=True)
    assert not any("worldflow" in prefix for prefix in off_prefixes)
    assert set(on_prefixes) - set(off_prefixes) == {
        "model.worldflow_branch.",
        "model.world_pointseg_object_proj.",
        "model.world_pointseg_background_proj.",
        "model.world_point_prefix_type_embedding",
        "model.world_action_out_proj.",
        "model.world_ego_action_type_embedding",
        "model.conjugate_motion_in_proj.",
    }


def test_feature_aligned_expert_is_wep_width_with_one_shared_context_adapter() -> None:
    prefix_spec = molmo_core.Molmo2TextSpec(
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        additional_vocab_size=128,
        hidden_act="silu",
        layer_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        max_position_embeddings=4096,
        qkv_bias=False,
        use_qk_norm=True,
        qk_norm_type="qwen3",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=0.02,
    )
    expert_spec = replace(
        prefix_spec,
        hidden_size=FULL_MOLMO2ER_EXPERT_HIDDEN_SIZE,
        intermediate_size=FULL_MOLMO2ER_EXPERT_INTERMEDIATE_SIZE,
    )
    expert = molmo_core.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        36,
        self_attn_every_n_layers=2,
        share_cross_attention_kv_projection=True,
        device=torch.device("meta"),
        dtype=torch.bfloat16,
    )

    assert expert.spec.hidden_size == 720
    assert expert.spec.intermediate_size == 2048
    assert expert.context_k_proj is not None and expert.context_v_proj is not None
    assert expert.context_k_proj.in_features == 1024
    assert expert.context_k_proj.out_features == 1024
    cross_layers = [layer for layer in expert.layers if layer.is_cross_attention]
    assert len(cross_layers) == 18
    assert all(layer.self_attn.k_proj is None for layer in cross_layers)
    assert all(layer.self_attn.v_proj is None for layer in cross_layers)
    assert sum(parameter.numel() for parameter in expert.parameters()) == 400_290_128


def test_worldflow_added_budget_is_the_v043_shared_branch_not_private_language() -> None:
    config = SmolVLAConfig()
    config.worldflow_target_type = "world_eef_trajectory"
    config.worldflow_world_eef_velocity_mode = "base_pose9_euclidean"
    config.worldflow_action_fusion = "point_action_expert_conjugate_bridge"
    config.worldflow_action_expert_mode = "shared"
    config.worldflow_feature_dim = 64
    config.worldflow_grid_size = 0.01
    branch = WorldFlowActionBranch(config, action_hidden_dim=720, language_vocab_size=152_064)
    assert branch.language_embedding is None
    branch_total = sum(parameter.numel() for parameter in branch.parameters())
    branch_trainable = sum(
        parameter.numel() for parameter in branch.parameters() if parameter.requires_grad
    )
    assert (branch_total, branch_trainable) == (39_640_438, 26_910_353)
    outer = (
        2 * (64 * 2560 + 2560)
        + (720 * 10 + 10)
        + (9 * 720 + 720)
        + 2560
        + 2 * 720
    )
    assert outer == 351_210
    assert branch_total + outer == FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET["total"]
    assert branch_trainable + outer == FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET["trainable"]


def test_fixed_native_visual_batch_reuses_flattened_storage() -> None:
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    backend.vision_backbone = SimpleNamespace(device=torch.device("cpu"))
    backend.image_end_token_id = 17
    input_ids = torch.tensor([[1, 17, 17, 2], [1, 17, 17, 3]])
    pixels = torch.randn(4, 3, 5)
    pooling = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [0, 1],
            [2, 3],
            [4, 5],
        ],
        dtype=torch.long,
    )
    grids = torch.tensor([[1, 1, 1, 2], [1, 1, 1, 2]], dtype=torch.long)
    crops = torch.tensor([2, 2], dtype=torch.long)

    images, batched_pooling = backend._build_batched_images(
        input_ids,
        pixels,
        pooling,
        grids,
        crops,
    )
    assert images.shape == (2, 2, 3, 5)
    assert batched_pooling.shape == (2, 3, 2)
    assert images.untyped_storage().data_ptr() == pixels.untyped_storage().data_ptr()
    assert batched_pooling.untyped_storage().data_ptr() == pooling.untyped_storage().data_ptr()


def _right_padded_native_batch(
    text_lengths: tuple[int, ...], *, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(text_lengths)
    length = 1 + NATIVE_IMAGE_POSITIONS + max(text_lengths)
    embeddings = torch.randn(batch_size, length, hidden_size)
    valid = torch.zeros(batch_size, length, dtype=torch.bool)
    image = torch.zeros_like(valid)
    for index, text_length in enumerate(text_lengths):
        valid[index, : 1 + NATIVE_IMAGE_POSITIONS + text_length] = True
        image[index, 1 : 1 + NATIVE_IMAGE_POSITIONS] = True
    return embeddings, valid, image


def _expected_native_scene_attention(
    valid: torch.Tensor,
    roles: torch.Tensor,
) -> torch.Tensor:
    """Reference visibility for native Molmo followed by one scene block."""

    if valid.shape != roles.shape:
        raise ValueError(f"Validity/role mismatch: {valid.shape} != {roles.shape}.")
    sequence_length = valid.shape[1]
    indices = torch.arange(sequence_length, device=valid.device)
    causal = indices[None, :] <= indices[:, None]
    native = (roles == TEXT_ROLE) | (roles == IMAGE_ROLE)
    image = roles == IMAGE_ROLE
    scene = roles == SCENE_ROLE
    native_to_native = (
        native[:, :, None] & native[:, None, :] & (causal[None] | (image[:, :, None] & image[:, None, :]))
    )
    scene_to_prefix = scene[:, :, None] & (native | scene)[:, None, :]
    return (native_to_native | scene_to_prefix) & valid[:, :, None] & valid[:, None, :]


def _make_prefix_shell(hidden_size: int = 8) -> VLAFlowMatching:
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.last_prefix_metrics = {}
    model.last_molmo_scene_insert_positions = None
    model.last_molmo_token_roles = None
    model.last_prefix_token_layout = ()
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=hidden_size)
    return model


def test_full_prefix_appends_scene_after_native_and_preserves_official_native_mask() -> None:
    model = _make_prefix_shell()
    native, valid, image = _right_padded_native_batch((2, 5), hidden_size=8)
    native.requires_grad_()
    scene = torch.randn(2, 2, 8, requires_grad=True)
    prefix, prefix_valid, boundaries = model._build_full_molmo_native_prefix(
        native,
        valid,
        image,
        scene,
        torch.ones(2, 2, dtype=torch.bool),
        ablate_language=False,
    )

    assert prefix.shape == (2, 418, 8)
    assert prefix.requires_grad
    assert torch.equal(model.last_molmo_scene_insert_positions, torch.tensor([413, 416]))
    assert torch.equal(prefix_valid.sum(dim=1), torch.tensor([415, 418]))
    assert prefix_valid[0, :415].all() and not prefix_valid[0, 415:].any()
    assert prefix_valid[1].all()
    assert torch.equal(prefix[0, :413], native[0, :413])
    assert torch.equal(prefix[1, :416], native[1, :416])
    assert torch.equal(prefix[0, 413:415], scene[0])
    assert torch.equal(prefix[1, 416:418], scene[1])
    assert not prefix[0, 415:].any()
    assert model.last_prefix_token_layout == (
        "native_bos_image_language",
        "ego_foreground",
        "ego_background",
    )
    roles = model.last_molmo_token_roles
    assert roles is not None
    assert set(roles.unique().tolist()) <= {TEXT_ROLE, IMAGE_ROLE, SCENE_ROLE, PAD_ROLE}
    assert (roles[0, 413:415] == SCENE_ROLE).all()
    assert (roles[1, 416:418] == SCENE_ROLE).all()

    # Native keeps Molmo's official causal-text + bidirectional-image rule.
    # The later scene block can read all valid Native/Scene keys, while no
    # Native query can read either scene token.
    allowed = make_att_2d_masks(prefix_valid, boundaries)
    expected = _expected_native_scene_attention(prefix_valid, roles)
    assert torch.equal(allowed, expected)
    for batch_index, native_length in enumerate((413, 416)):
        native_indices = torch.arange(native_length)
        scene_indices = torch.arange(native_length, native_length + 2)
        assert not allowed[batch_index, native_indices[:, None], scene_indices].any()
        assert allowed[batch_index, scene_indices[:, None], : native_length + 2].all()
        # Appending scene leaves every Native token at its original RoPE id.
        positions = torch.cumsum(prefix_valid[batch_index], dim=0, dtype=torch.long) - 1
        assert torch.equal(positions[:native_length], torch.arange(native_length))

    prefix.sum().backward()
    assert native.grad is None
    assert scene.grad is not None and scene.grad.abs().sum() > 0


def test_full_prefix_preserves_each_samples_native_image_block_boundary() -> None:
    model = _make_prefix_shell(hidden_size=4)
    native = torch.randn(2, 414, 4)
    native_valid = torch.ones(2, 414, dtype=torch.bool)
    native_image = torch.zeros_like(native_valid)
    native_image[0, 1:411] = True
    native_image[1, 2:412] = True
    scene = torch.randn(2, 2, 4)
    prefix, prefix_valid, boundaries = model._build_full_molmo_native_prefix(
        native,
        native_valid,
        native_image,
        scene,
        torch.ones(2, 2, dtype=torch.bool),
        ablate_language=False,
    )
    assert torch.equal(model.last_molmo_scene_insert_positions, torch.tensor([414, 414]))
    assert torch.equal(prefix[:, :414], native)
    assert torch.equal(prefix[:, 414:416], scene)
    assert prefix_valid.all()
    roles = model.last_molmo_token_roles
    assert roles is not None
    allowed = make_att_2d_masks(prefix_valid, boundaries)
    assert torch.equal(allowed, _expected_native_scene_attention(prefix_valid, roles))
    # Both image blocks remain bidirectional even though their physical start
    # differs; no image query can read later language or scene positions.
    for batch_index, image_start in enumerate((1, 2)):
        image_indices = torch.arange(image_start, image_start + NATIVE_IMAGE_POSITIONS)
        assert allowed[batch_index, image_indices[:, None], image_indices].all()
        assert not allowed[batch_index, image_indices, 413].any()
        assert not allowed[batch_index, image_indices, 414:].any()


def test_world_fg_bg_continue_scene_block_with_consecutive_positions() -> None:
    torch.manual_seed(5)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vlm_backend="molmo2_full")
    model.vlm_with_expert = SimpleNamespace(scale_input_embeddings=False)
    model.world_pointseg_object_proj = nn.Linear(4, 8)
    model.world_pointseg_background_proj = nn.Linear(4, 8)
    model.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(8))
    model.inference_ablation_modalities = frozenset()
    model.last_molmo_scene_insert_positions = torch.tensor([4, 6])
    prefix = torch.randn(2, 8, 8)
    prefix_valid = torch.tensor([[True, True, True, True, True, True, False, False], [True] * 8])
    prefix_boundaries = torch.tensor(
        [
            [True, True, False, True, True, False, False, False],
            [True, True, False, False, True, True, True, False],
        ]
    )
    scene_features = torch.randn(2, 2, 4)
    world_tokens = {
        "scene_global_tokens": scene_features,
        "scene_global_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    augmented, augmented_valid, boundaries, positions = model._append_shared_expert_world_scene_prefix(
        prefix, prefix_valid, prefix_boundaries, world_tokens
    )
    expected_world = torch.stack(
        [
            model.world_pointseg_object_proj(scene_features[:, 0]),
            model.world_pointseg_background_proj(scene_features[:, 1]),
        ],
        dim=1,
    )
    assert torch.allclose(augmented[:, -2:], expected_world)
    assert augmented_valid[:, -2:].all()
    assert torch.equal(boundaries[:, :-2], prefix_boundaries)
    assert not boundaries[:, -2:].any()
    base_positions = torch.cumsum(prefix_valid, dim=1, dtype=torch.long) - 1
    assert torch.equal(positions[:, :8], base_positions)
    assert torch.equal(positions[:, -2:], torch.tensor([[6, 7], [8, 9]]))

    augmented_roles = torch.tensor(
        [
            [
                TEXT_ROLE,
                IMAGE_ROLE,
                IMAGE_ROLE,
                TEXT_ROLE,
                SCENE_ROLE,
                SCENE_ROLE,
                PAD_ROLE,
                PAD_ROLE,
                SCENE_ROLE,
                SCENE_ROLE,
            ],
            [
                TEXT_ROLE,
                IMAGE_ROLE,
                IMAGE_ROLE,
                IMAGE_ROLE,
                TEXT_ROLE,
                TEXT_ROLE,
                SCENE_ROLE,
                SCENE_ROLE,
                SCENE_ROLE,
                SCENE_ROLE,
            ],
        ],
        dtype=torch.int8,
    )
    allowed = make_att_2d_masks(augmented_valid, boundaries)
    assert torch.equal(
        allowed,
        _expected_native_scene_attention(augmented_valid, augmented_roles),
    )
    for batch_index, native_length in enumerate((4, 6)):
        native_indices = torch.arange(native_length)
        scene_indices = torch.tensor([native_length, native_length + 1, 8, 9])
        assert not allowed[batch_index, native_indices[:, None], scene_indices].any()
        valid_prefix_indices = torch.nonzero(augmented_valid[batch_index], as_tuple=False).flatten()
        assert allowed[batch_index, scene_indices[:, None], valid_prefix_indices].all()


def test_valid_world_scene_cannot_leak_into_native_when_ego_scene_is_invalid() -> None:
    model = _make_prefix_shell(hidden_size=8)
    model.config = SimpleNamespace(vlm_backend="molmo2_full")
    model.vlm_with_expert.scale_input_embeddings = False
    model.world_pointseg_object_proj = nn.Linear(4, 8)
    model.world_pointseg_background_proj = nn.Linear(4, 8)
    model.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(8))
    model.inference_ablation_modalities = frozenset()
    native, native_valid, native_image = _right_padded_native_batch((2,), hidden_size=8)
    prefix, prefix_valid, prefix_boundaries = model._build_full_molmo_native_prefix(
        native,
        native_valid,
        native_image,
        torch.randn(1, 2, 8),
        torch.zeros(1, 2, dtype=torch.bool),
        ablate_language=False,
    )
    world_tokens = {
        "scene_global_tokens": torch.randn(1, 2, 4),
        "scene_global_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    _, augmented_valid, augmented_boundaries, positions = (
        model._append_shared_expert_world_scene_prefix(
            prefix,
            prefix_valid,
            prefix_boundaries,
            world_tokens,
        )
    )
    native_length = int(native_valid.sum())
    world_indices = torch.tensor([prefix.shape[1], prefix.shape[1] + 1])
    allowed = make_att_2d_masks(augmented_valid, augmented_boundaries)
    assert not allowed[0, :native_length, world_indices].any()
    assert allowed[0, world_indices[:, None], :native_length].all()
    assert allowed[0, world_indices[:, None], world_indices].all()
    assert torch.equal(positions[0, -2:], torch.tensor([native_length, native_length + 1]))


def test_shared_suffix_contains_only_paired_actions() -> None:
    torch.manual_seed(7)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    hidden = 8
    model.config = SimpleNamespace(vlm_backend="molmo2_full", chunk_size=2)
    model.conjugate_motion_in_proj = nn.Linear(9, hidden)
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, hidden))
    model.inference_ablation_modalities = frozenset()
    ego_actions = torch.randn(1, 2, hidden)
    world_tokens = {
        "scene_global_tokens": torch.randn(1, 2, 4),
        "scene_global_mask": torch.ones(1, 2, dtype=torch.bool),
        "action_tokens": torch.randn(1, 2, hidden),
        "action_mask": torch.ones(1, 2, dtype=torch.bool),
        "ego_conjugate_motion_pose9": torch.randn(1, 2, 9),
        "world_conjugate_motion_pose9": torch.randn(1, 2, 9),
        "conjugate_motion_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    suffix, mask, positions, layout = model._build_shared_point_action_expert_joint_suffix(
        ego_actions, torch.ones(1, 2, dtype=torch.bool), world_tokens
    )
    assert suffix.shape == (1, 4, hidden)
    assert mask.all()
    assert layout == {
        "ego_action": slice(0, 2),
        "world_action": slice(2, 4),
    }
    assert torch.equal(positions, torch.tensor([[0, 1, 0, 1]]))


def _tiny_spec(hidden_size: int, intermediate_size: int) -> molmo_core.Molmo2TextSpec:
    return molmo_core.Molmo2TextSpec(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=36,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=31,
        additional_vocab_size=1,
        hidden_act="silu",
        layer_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        max_position_embeddings=1024,
        qkv_bias=False,
        use_qk_norm=True,
        qk_norm_type="qwen3",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=0.02,
    )


def _make_tiny_backend(
    *,
    num_layers: int = 36,
    gradient_checkpointing: bool = False,
    checkpoint_segment_size: int = 2,
) -> Molmo2FullWithExpertModel:
    prefix_spec = _tiny_spec(8, 16)
    expert_spec = _tiny_spec(6, 12)
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    nn.Module.__init__(backend)
    backend.vlm = molmo_core.Molmo2TextBackbone(
        prefix_spec, num_layers, device=torch.device("cpu"), dtype=torch.float32
    )
    backend.lm_expert = molmo_core.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        num_layers,
        self_attn_every_n_layers=2,
        share_cross_attention_kv_projection=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.vision_backbone = nn.Linear(1, 1, bias=False)
    molmo_core._initialize_module(backend, 0.02)
    backend.num_vlm_layers = num_layers
    backend.num_expert_layers = num_layers
    backend.self_attn_every_n_layers = 2
    backend.attention_mode = "cross_attn"
    backend.train_expert_only = True
    backend.freeze_vision_encoder = True
    backend.text_spec = prefix_spec
    backend.expert_spec = expert_spec
    backend.inference_only_vlm = False
    backend.gradient_checkpointing = gradient_checkpointing
    backend.gradient_checkpointing_layers_per_segment = checkpoint_segment_size
    backend.set_requires_grad()
    backend.train()
    return backend


def test_scene_and_later_language_cannot_change_native_image_hidden() -> None:
    torch.manual_seed(19)
    backend = _make_tiny_backend(num_layers=2)
    backend.eval()
    # [BOS/TEXT, IMAGE, IMAGE, IMAGE, LANGUAGE, EGO_FG, EGO_BG]
    valid = torch.ones(1, 7, dtype=torch.bool)
    boundaries = torch.tensor([[True, True, False, False, True, True, False]])
    attention = make_att_2d_masks(valid, boundaries)
    positions = torch.arange(7, dtype=torch.long)[None]
    baseline = torch.randn(1, 7, 8)
    changed_scene = baseline.clone()
    changed_scene[:, 5:] += 100.0
    changed_language = baseline.clone()
    changed_language[:, 4] -= 100.0

    with torch.no_grad():
        (baseline_output, _), _ = backend(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[baseline, None],
            use_cache=False,
            fill_kv_cache=False,
        )
        (scene_output, _), _ = backend(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[changed_scene, None],
            use_cache=False,
            fill_kv_cache=False,
        )
        (language_output, _), _ = backend(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[changed_language, None],
            use_cache=False,
            fill_kv_cache=False,
        )

    # Scene is a pure readout: changing it cannot condition any Native row.
    assert torch.equal(baseline_output[:, :5], scene_output[:, :5])
    assert not torch.equal(baseline_output[:, 5:], scene_output[:, 5:])
    # Official causal text semantics prevent later language from changing BOS
    # or image hidden, while the following scene readouts can consume it.
    assert torch.equal(baseline_output[:, :4], language_output[:, :4])
    assert not torch.equal(baseline_output[:, 5:], language_output[:, 5:])


def _joint_attention_and_positions(
    prefix_valid: torch.Tensor,
    action_valid: torch.Tensor,
    prefix_boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_attention = make_att_2d_masks(prefix_valid, prefix_boundaries)
    prefix_to_action = torch.zeros(
        prefix_valid.shape[0],
        prefix_valid.shape[1],
        action_valid.shape[1],
        dtype=torch.bool,
    )
    action_to_prefix = action_valid[:, :, None] & prefix_valid[:, None, :]
    action_attention = action_valid[:, :, None] & action_valid[:, None, :]
    attention = torch.cat(
        [
            torch.cat([prefix_attention, prefix_to_action], dim=2),
            torch.cat([action_to_prefix, action_attention], dim=2),
        ],
        dim=1,
    )
    prefix_positions = torch.cumsum(prefix_valid, dim=1, dtype=torch.long) - 1
    offset = prefix_positions.masked_fill(~prefix_valid, -1).amax(dim=1, keepdim=True) + 1
    action_steps = action_valid.shape[1] // 2
    relative = torch.arange(action_steps)[None].repeat(prefix_valid.shape[0], 2)
    positions = torch.cat([prefix_positions, offset + relative], dim=1)
    return attention, positions


def test_action_loss_crosses_all_36_frozen_vlm_layers_into_all_scene_prefixes() -> None:
    torch.manual_seed(23)
    backend = _make_tiny_backend()
    builder = _make_prefix_shell()
    native, valid, image = _right_padded_native_batch((2,), hidden_size=8)
    native.requires_grad_()
    ego_scene_projection = nn.Linear(4, 8)
    world_scene_projection = nn.Linear(4, 8)
    ego_scene = ego_scene_projection(torch.randn(1, 2, 4))
    world_scene = world_scene_projection(torch.randn(1, 2, 4))
    ego_scene.retain_grad()
    world_scene.retain_grad()
    prefix, prefix_valid, prefix_boundaries = builder._build_full_molmo_native_prefix(
        native,
        valid,
        image,
        ego_scene,
        torch.ones(1, 2, dtype=torch.bool),
        ablate_language=False,
    )
    prefix = torch.cat([prefix, world_scene], dim=1)
    prefix_valid = torch.cat([prefix_valid, torch.ones(1, 2, dtype=torch.bool)], dim=1)
    prefix_boundaries = torch.cat([prefix_boundaries, torch.zeros(1, 2, dtype=torch.bool)], dim=1)
    assert prefix.requires_grad
    actions = torch.randn(1, 2, 6, requires_grad=True)
    action_valid = torch.ones(1, 2, dtype=torch.bool)
    attention, positions = _joint_attention_and_positions(prefix_valid, action_valid, prefix_boundaries)
    prefix_attention = attention[:, : prefix.shape[1], : prefix.shape[1]]
    roles = torch.cat(
        [
            builder.last_molmo_token_roles,
            torch.full((1, 2), SCENE_ROLE, dtype=torch.int8),
        ],
        dim=1,
    )
    assert torch.equal(
        prefix_attention,
        _expected_native_scene_attention(prefix_valid, roles),
    )
    assert not attention[:, : prefix.shape[1], prefix.shape[1] :].any()
    assert attention[:, prefix.shape[1] :, :].all()
    (_, output), cache = backend(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix, actions],
        use_cache=False,
        fill_kv_cache=False,
    )
    loss = output.square().mean()
    loss.backward()
    assert native.grad is None
    assert all(parameter.grad is None for parameter in backend.vlm.parameters())
    assert cache is None
    assert ego_scene.grad is not None and ego_scene.grad.abs().sum() > 0
    assert world_scene.grad is not None and world_scene.grad.abs().sum() > 0
    assert ego_scene_projection.weight.grad is not None
    assert world_scene_projection.weight.grad is not None
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in backend.lm_expert.parameters())
    assert backend.lm_expert.context_k_proj is not None
    assert backend.lm_expert.context_v_proj is not None
    assert backend.lm_expert.context_k_proj.weight.grad is not None
    assert backend.lm_expert.context_v_proj.weight.grad is not None


def test_layer_checkpointing_preserves_forward_and_input_expert_gradients() -> None:
    torch.manual_seed(41)
    plain = _make_tiny_backend(num_layers=3, gradient_checkpointing=False)
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing = True
    prefix_valid = torch.ones(1, 5, dtype=torch.bool)
    # [TEXT, IMAGE, IMAGE, TEXT, SCENE]
    prefix_boundaries = torch.tensor([[True, True, False, True, True]])
    action_valid = torch.ones(1, 4, dtype=torch.bool)
    attention, positions = _joint_attention_and_positions(prefix_valid, action_valid, prefix_boundaries)
    prefix_plain = torch.randn(1, 5, 8, requires_grad=True)
    action_plain = torch.randn(1, 4, 6, requires_grad=True)
    prefix_checkpointed = prefix_plain.detach().clone().requires_grad_()
    action_checkpointed = action_plain.detach().clone().requires_grad_()

    (_, plain_output), _ = plain(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix_plain, action_plain],
        use_cache=False,
        fill_kv_cache=False,
    )
    (_, checkpointed_output), _ = checkpointed(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix_checkpointed, action_checkpointed],
        use_cache=False,
        fill_kv_cache=False,
    )
    assert torch.allclose(plain_output, checkpointed_output, atol=1e-6, rtol=1e-6)
    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    assert torch.allclose(prefix_plain.grad, prefix_checkpointed.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(action_plain.grad, action_checkpointed.grad, atol=1e-6, rtol=1e-5)
    for (_, plain_parameter), (_, checkpointed_parameter) in zip(
        plain.lm_expert.named_parameters(),
        checkpointed.lm_expert.named_parameters(),
        strict=True,
    ):
        assert torch.allclose(
            plain_parameter.grad,
            checkpointed_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        )
    assert all(parameter.grad is None for parameter in plain.vlm.parameters())
    assert all(parameter.grad is None for parameter in checkpointed.vlm.parameters())


def test_full_prefix_cache_matches_joint_forward_with_action_only_suffix() -> None:
    torch.manual_seed(53)
    backend = _make_tiny_backend(num_layers=4)
    backend.eval()
    prefix = torch.randn(1, 7, 8)
    actions = torch.randn(1, 4, 6)
    prefix_valid = torch.ones(1, 7, dtype=torch.bool)
    # [TEXT, IMAGE, IMAGE, IMAGE, TEXT, SCENE, SCENE]
    prefix_boundaries = torch.tensor([[True, True, False, False, True, True, False]])
    action_valid = torch.ones(1, 4, dtype=torch.bool)
    joint_attention, joint_positions = _joint_attention_and_positions(
        prefix_valid, action_valid, prefix_boundaries
    )
    with torch.no_grad():
        (_, joint_output), _ = backend(
            attention_mask=joint_attention,
            position_ids=joint_positions,
            inputs_embeds=[prefix, actions],
            use_cache=False,
            fill_kv_cache=False,
        )
        (_, _), cache = backend(
            attention_mask=make_att_2d_masks(prefix_valid, prefix_boundaries),
            position_ids=joint_positions[:, : prefix.shape[1]],
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        assert cache is not None and len(cache) == 4
        suffix_attention = joint_attention[:, prefix.shape[1] :, :]
        (_, cached_output), _ = backend(
            attention_mask=suffix_attention,
            position_ids=joint_positions[:, prefix.shape[1] :],
            past_key_values=cache,
            inputs_embeds=[None, actions],
            use_cache=True,
            fill_kv_cache=False,
        )
    assert torch.allclose(joint_output, cached_output, atol=1e-5, rtol=1e-5)
    assert all(layer["key_states"].shape[1] == prefix.shape[1] for layer in cache.values())


def test_native_readout_backend_state_dict_strict_reload_is_exact() -> None:
    torch.manual_seed(61)
    source = _make_tiny_backend(num_layers=2)
    reloaded = _make_tiny_backend(num_layers=2)
    reloaded.load_state_dict(source.state_dict(), strict=True)
    source.eval()
    reloaded.eval()

    prefix = torch.randn(1, 5, 8)
    actions = torch.randn(1, 4, 6)
    prefix_boundaries = torch.tensor([[True, True, False, True, True]])
    attention, positions = _joint_attention_and_positions(
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
        prefix_boundaries,
    )
    with torch.no_grad():
        (_, source_output), _ = source(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[prefix, actions],
            use_cache=False,
            fill_kv_cache=False,
        )
        (_, reloaded_output), _ = reloaded(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[prefix, actions],
            use_cache=False,
            fill_kv_cache=False,
        )
    assert torch.equal(source_output, reloaded_output)
