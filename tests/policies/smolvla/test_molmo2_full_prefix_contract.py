#!/usr/bin/env python

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
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
from lerobot.policies.smolvla import molmo2_with_expert as molmo_core
from lerobot.policies.smolvla.molmo2_full_with_expert import Molmo2FullWithExpertModel

TEXT_ROLE = 0
IMAGE_ROLE = 1
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
        "expert_width_multiplier": 0.75,
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
    assert config.molmo_inference_only and config.molmo_image_fast_path
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


def test_full_config_allows_cli_scheduler_override() -> None:
    config = _full_worldflow_config(
        scheduler_warmup_steps=50,
        scheduler_decay_steps=1_500,
        scheduler_decay_lr=1e-5,
    )
    assert config.scheduler_warmup_steps == 50
    assert config.scheduler_decay_steps == 1_500
    assert config.scheduler_decay_lr == 1e-5


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("molmo_inference_only", False),
        ("molmo_image_fast_path", False),
        ("worldflow_enable", False),
        ("worldflow_noise_coupling", "conjugate_ego"),
        ("worldflow_action_fusion", "cross_attention"),
        ("worldflow_action_expert_mode", "independent"),
        ("worldflow_require_action_target_sidecar", False),
        ("worldflow_max_points", 0),
    ),
)
def test_full_config_rejects_registered_contract_drift(field_name: str, bad_value) -> None:
    with pytest.raises(ValueError):
        _full_worldflow_config(**{field_name: bad_value})


def test_parameter_budget_matches_v043_shared_doubleflow() -> None:
    off = expected_full_molmo2er_parameter_budget(worldflow_enable=False)
    on = expected_full_molmo2er_parameter_budget(worldflow_enable=True)
    assert off == FULL_MOLMO2ER_WORLDFLOW_OFF_PARAMETER_BUDGET == {
        "total": 6_261_132_686,
        "trainable": 1_786_461_401,
        "frozen": 4_474_671_285,
    }
    assert on == FULL_MOLMO2ER_WORLDFLOW_ON_PARAMETER_BUDGET == {
        "total": 6_310_743_694,
        "trainable": 1_823_342_324,
        "frozen": 4_487_401_370,
    }
    assert {name: on[name] - off[name] for name in on} == (
        FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET
    ) == {
        "total": 49_611_008,
        "trainable": 36_880_923,
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


def test_worldflow_added_budget_is_the_v043_shared_branch_not_private_language() -> None:
    config = SmolVLAConfig()
    config.worldflow_target_type = "world_eef_trajectory"
    config.worldflow_world_eef_velocity_mode = "base_pose9_euclidean"
    config.worldflow_action_fusion = "point_action_expert_conjugate_bridge"
    config.worldflow_action_expert_mode = "shared"
    config.worldflow_feature_dim = 64
    config.worldflow_grid_size = 0.01
    branch = WorldFlowActionBranch(config, action_hidden_dim=1920, language_vocab_size=152_064)
    assert branch.language_embedding is None
    branch_total = sum(parameter.numel() for parameter in branch.parameters())
    branch_trainable = sum(
        parameter.numel() for parameter in branch.parameters() if parameter.requires_grad
    )
    assert (branch_total, branch_trainable) == (49_317_238, 36_587_153)
    outer = (
        2 * (64 * 1920 + 1920)
        + (1920 * 10 + 10)
        + (9 * 1920 + 1920)
        + 1920
        + 2 * 1920
    )
    assert outer == 293_770
    assert branch_total + outer == FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET["total"]
    assert branch_trainable + outer == FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET["trainable"]


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


def _make_prefix_shell(hidden_size: int = 8) -> VLAFlowMatching:
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.last_prefix_metrics = {}
    model.last_molmo_scene_insert_positions = None
    model.last_molmo_token_roles = None
    model.last_prefix_token_layout = ()
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=hidden_size)
    return model


def test_native_prefix_contains_only_frozen_image_text() -> None:
    model = _make_prefix_shell()
    native, valid, image = _right_padded_native_batch((3,), hidden_size=8)
    native.requires_grad_()
    prefix, prefix_valid, boundaries = model._build_full_molmo_native_prefix(
        native, valid, image, ablate_language=False
    )
    assert torch.equal(prefix, native)
    assert not prefix.requires_grad
    assert model.last_prefix_token_layout == ("bos", "native_image", "causal_text")
    roles = model.last_molmo_token_roles
    assert roles is not None
    assert set(roles.unique().tolist()) <= {TEXT_ROLE, IMAGE_ROLE, PAD_ROLE}
    assert boundaries[:, 0].all() and boundaries[:, 1].all()
    assert not boundaries[:, 2:411].any()
    allowed = make_att_2d_masks(prefix_valid, boundaries)
    assert allowed[:, 1:411, 1:411].all()
    assert not allowed[:, 1:411, 411:].any()
    assert allowed[:, 411:, 1:411].all()


def test_shared_suffix_is_v043_fg_bg_pairs_plus_paired_actions() -> None:
    torch.manual_seed(7)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    hidden = 8
    model.config = SimpleNamespace(vlm_backend="molmo2_full", chunk_size=2)
    model.conjugate_motion_in_proj = nn.Linear(9, hidden)
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, hidden))
    model.world_pointseg_object_proj = nn.Linear(4, hidden)
    model.world_pointseg_background_proj = nn.Linear(4, hidden)
    model.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(hidden))
    model.last_expert_scene_tokens = torch.randn(1, 2, hidden)
    model.last_expert_scene_mask = torch.ones(1, 2, dtype=torch.bool)
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
    assert suffix.shape == (1, 8, hidden)
    assert mask.all()
    assert layout == {
        "ego_foreground": slice(0, 1),
        "ego_background": slice(1, 2),
        "world_foreground": slice(2, 3),
        "world_background": slice(3, 4),
        "ego_action": slice(4, 6),
        "world_action": slice(6, 8),
    }
    assert torch.equal(positions, torch.tensor([[0, 1, 0, 1, 2, 3, 2, 3]]))


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


def _make_tiny_backend() -> Molmo2FullWithExpertModel:
    prefix_spec = _tiny_spec(8, 16)
    expert_spec = _tiny_spec(6, 12)
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    nn.Module.__init__(backend)
    backend.vlm = molmo_core.Molmo2TextBackbone(
        prefix_spec, 36, device=torch.device("cpu"), dtype=torch.float32
    )
    backend.lm_expert = molmo_core.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        36,
        self_attn_every_n_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.vision_backbone = nn.Linear(1, 1, bias=False)
    molmo_core._initialize_module(backend, 0.02)
    backend.num_vlm_layers = 36
    backend.num_expert_layers = 36
    backend.self_attn_every_n_layers = 2
    backend.attention_mode = "cross_attn"
    backend.train_expert_only = True
    backend.freeze_vision_encoder = True
    backend.text_spec = prefix_spec
    backend.expert_spec = expert_spec
    backend.set_requires_grad()
    backend.train()
    return backend


def test_all_36_vlm_layers_are_detached_while_four_scene_tokens_train() -> None:
    torch.manual_seed(23)
    backend = _make_tiny_backend()
    builder = _make_prefix_shell()
    native, valid, image = _right_padded_native_batch((2,), hidden_size=8)
    native.requires_grad_()
    prefix, prefix_valid, prefix_boundaries = builder._build_full_molmo_native_prefix(
        native, valid, image, ablate_language=False
    )
    scene_projection = nn.Linear(4, 6)
    scene = scene_projection(torch.randn(1, 4, 4))
    scene.retain_grad()
    actions = torch.randn(1, 2, 6, requires_grad=True)
    expert = torch.cat([scene, actions], dim=1)
    expert_valid = torch.ones(1, 6, dtype=torch.bool)
    prefix_attention = make_att_2d_masks(prefix_valid, prefix_boundaries)
    suffix_is_action = torch.tensor([False, False, False, False, True, True])
    suffix_attention = (
        (suffix_is_action[:, None] | ~suffix_is_action[None, :])[None]
        & expert_valid[:, :, None]
        & expert_valid[:, None, :]
    )
    prefix_to_suffix = (
        prefix_valid[:, :, None]
        & expert_valid[:, None, :]
        & (~suffix_is_action)[None, None, :]
    )
    suffix_to_prefix = expert_valid[:, :, None] & prefix_valid[:, None, :]
    attention = torch.cat(
        [
            torch.cat([prefix_attention, prefix_to_suffix], dim=2),
            torch.cat([suffix_to_prefix, suffix_attention], dim=2),
        ],
        dim=1,
    )
    prefix_positions = torch.cumsum(prefix_valid, dim=1) - 1
    offset = prefix_positions.amax(dim=1, keepdim=True) + 1
    expert_positions = offset + torch.tensor([[0, 1, 0, 1, 2, 3]])
    positions = torch.cat([prefix_positions, expert_positions], dim=1)
    (_, output), cache = backend(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix, expert],
        use_cache=True,
        fill_kv_cache=False,
    )
    loss = output[:, -2:].square().mean()
    loss.backward()
    assert native.grad is None
    assert all(parameter.grad is None for parameter in backend.vlm.parameters())
    assert cache is not None and len(cache) == 36
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for layer in cache.values()
        for tensor in layer.values()
    )
    assert scene.grad is not None and scene.grad.abs().sum() > 0
    assert scene_projection.weight.grad is not None
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in backend.lm_expert.parameters())
