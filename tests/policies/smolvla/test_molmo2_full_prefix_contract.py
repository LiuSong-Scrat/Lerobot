#!/usr/bin/env python

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import _validate_full_molmo_checkpoint_topology
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import (
    FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
    SmolVLAConfig,
)
from lerobot.policies.smolvla.modeling_smolvla import (
    FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET,
    FULL_MOLMO2ER_WORLDFLOW_OFF_PARAMETER_BUDGET,
    FULL_MOLMO2ER_WORLDFLOW_ON_PARAMETER_BUDGET,
    VLAFlowMatching,
    WorldFlowActionBranch,
    _litept_valid_sample_mask,
    _pack_litept_tokens_by_batch,
    expected_full_molmo2er_parameter_budget,
    full_molmo2er_trainable_parameter_prefixes,
    make_att_2d_masks,
)
from lerobot.policies.smolvla import molmo2_with_expert as molmo_core
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
    assert not config.molmo_inference_only and config.molmo_image_fast_path
    assert config.molmo_gradient_checkpointing
    assert config.molmo_gradient_checkpointing_layers_per_segment == 2
    assert config.full_molmo_topology == FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY
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


def test_full_checkpoint_topology_gate_rejects_old_and_accepts_wep_prefix(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"

    config_path.write_text(
        json.dumps({"vlm_backend": "molmo2_full", "molmo_inference_only": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible Full-Molmo topology"):
        _validate_full_molmo_checkpoint_topology(checkpoint)

    for invalid_topology in (
        None,
        "detached_scene_suffix_v2",
        "wepvla_scene_in_vlm_prefix_v3",
        "molmo_native_hybrid_wepvla_expert_v4",
        "unknown",
    ):
        legacy_config = {
            "vlm_backend": "molmo2_full",
            "molmo_inference_only": False,
        }
        if invalid_topology is not None:
            legacy_config["full_molmo_topology"] = invalid_topology
        config_path.write_text(json.dumps(legacy_config), encoding="utf-8")
        with pytest.raises(ValueError, match="incompatible Full-Molmo topology"):
            _validate_full_molmo_checkpoint_topology(checkpoint)

    config_path.write_text(
        json.dumps(
            {
                "vlm_backend": "molmo2_full",
                "full_molmo_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible Full-Molmo topology"):
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


def test_full_checkpoint_topology_gate_rejects_missing_config(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="topology cannot be verified"):
        _validate_full_molmo_checkpoint_topology(checkpoint)


def test_explicit_v5_config_cannot_bypass_v3_raw_checkpoint_gate(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla",
                "vlm_backend": "molmo2_full",
                "molmo_inference_only": False,
                "full_molmo_topology": "wepvla_scene_in_vlm_prefix_v3",
            }
        ),
        encoding="utf-8",
    )

    class NeverInstantiatedPolicy(PreTrainedPolicy):
        config_class = SmolVLAConfig
        name = "never_instantiated_full_molmo_test"

        def __init__(self, config, **kwargs):
            raise AssertionError("raw topology validation must run before model construction")

    with pytest.raises(ValueError, match="incompatible Full-Molmo topology"):
        NeverInstantiatedPolicy.from_pretrained(
            checkpoint,
            config=_full_worldflow_config(),
        )


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

    with pytest.raises(ValueError, match="incompatible Full-Molmo topology"):
        PreTrainedConfig.from_pretrained(checkpoint)


def test_public_config_loader_rejects_missing_inference_mode_marker(tmp_path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config = _full_worldflow_config()
    config._save_pretrained(checkpoint)
    config_path = checkpoint / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config.pop("molmo_inference_only")
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible Full-Molmo topology"):
        PreTrainedConfig.from_pretrained(checkpoint)


def test_public_config_loader_accepts_registered_wep_prefix_topology(tmp_path) -> None:
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
    ),
)
def test_full_config_rejects_registered_contract_drift(field_name: str, bad_value) -> None:
    with pytest.raises(ValueError):
        _full_worldflow_config(**{field_name: bad_value})


def test_parameter_budget_matches_v043_shared_doubleflow() -> None:
    off = expected_full_molmo2er_parameter_budget(worldflow_enable=False)
    on = expected_full_molmo2er_parameter_budget(worldflow_enable=True)
    assert off == FULL_MOLMO2ER_WORLDFLOW_OFF_PARAMETER_BUDGET == {
        "total": 6_261_215_886,
        "trainable": 1_786_544_601,
        "frozen": 4_474_671_285,
    }
    assert on == FULL_MOLMO2ER_WORLDFLOW_ON_PARAMETER_BUDGET == {
        "total": 6_310_910_734,
        "trainable": 1_823_509_364,
        "frozen": 4_487_401_370,
    }
    assert {name: on[name] - off[name] for name in on} == (
        FULL_MOLMO2ER_WORLDFLOW_ADDED_PARAMETER_BUDGET
    ) == {
        "total": 49_694_848,
        "trainable": 36_964_763,
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
        2 * (64 * 2560 + 2560)
        + (1920 * 10 + 10)
        + (9 * 1920 + 1920)
        + 2560
        + 2 * 1920
    )
    assert outer == 377_610
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


def test_registered_392_feature_visual_fast_path_validates_once_and_returns_views() -> None:
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    backend.vision_backbone = SimpleNamespace(device=torch.device("cpu"))
    backend.image_end_token_id = 17
    input_ids = torch.tensor([[1, 17, 17, 2], [1, 17, 17, 3]])
    pixels = torch.randn(4, 3, 5)
    pooling = torch.arange(2 * 392 * 2, dtype=torch.long).view(2 * 392, 2)
    grids = torch.tensor([[14, 14, 14, 14], [14, 14, 14, 14]], dtype=torch.long)
    crops = torch.tensor([2, 2], dtype=torch.long)

    first_images, first_pooling = backend._build_batched_images(
        input_ids, pixels, pooling, grids, crops
    )
    assert backend._fixed_visual_batch_contract_validated is True
    assert first_images.shape == (2, 2, 3, 5)
    assert first_pooling.shape == (2, 392, 2)
    assert first_images.untyped_storage().data_ptr() == pixels.untyped_storage().data_ptr()
    assert first_pooling.untyped_storage().data_ptr() == pooling.untyped_storage().data_ptr()

    second_images, second_pooling = backend._build_batched_images(
        input_ids, pixels, pooling, grids, crops
    )
    assert torch.equal(first_images, second_images)
    assert torch.equal(first_pooling, second_pooling)


def test_vectorized_litept_validity_matches_per_sample_reference() -> None:
    pc = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 9.0, 9.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [8.0, 8.0, 8.0]],
            [[0.0, 0.0, 0.0], [float("nan"), 1.0, 1.0], [7.0, 7.0, 7.0]],
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
        ]
    )
    point_is_pad = torch.tensor(
        [
            [False, False, True],
            [False, False, True],
            [False, False, True],
            [True, True, True],
        ]
    )
    expected = []
    for batch_index in range(pc.shape[0]):
        valid = ~point_is_pad[batch_index]
        xyz = pc[batch_index, valid]
        if not bool(valid.any()):
            expected.append(False)
            continue
        extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).abs().sum()
        expected.append(bool(torch.isfinite(xyz).all()) and bool(extent >= 1e-6))
    assert torch.equal(
        _litept_valid_sample_mask(pc, point_is_pad),
        torch.tensor(expected, dtype=torch.bool),
    )


def test_vectorized_litept_token_packing_matches_reference_order_values_and_gradients() -> None:
    torch.manual_seed(19)
    point_batch = torch.tensor([1, 0, 2, 1, 0, 1], dtype=torch.long)
    valid_indices = torch.tensor([0, 2, 3], dtype=torch.long)
    xyz = torch.randn(6, 3, requires_grad=True)
    features = torch.randn(6, 5, requires_grad=True)

    actual = _pack_litept_tokens_by_batch(
        xyz,
        features,
        point_batch,
        valid_indices,
        total_batch_size=5,
        output_dtype=torch.float32,
    )

    counts = torch.bincount(point_batch, minlength=3)
    max_tokens = int(counts.max())
    expected_xyz = torch.zeros(5, max_tokens, 3)
    expected_features = torch.zeros(5, max_tokens, 5)
    expected_global = torch.zeros(5, 5)
    expected_mask = torch.zeros(5, max_tokens, dtype=torch.bool)
    for local_batch in range(3):
        indices = torch.nonzero(point_batch == local_batch, as_tuple=False).squeeze(1)
        count = indices.shape[0]
        global_batch = valid_indices[local_batch]
        expected_xyz[global_batch, :count] = xyz[indices]
        expected_features[global_batch, :count] = features[indices]
        expected_global[global_batch] = features[indices].max(dim=0).values
        expected_mask[global_batch, :count] = True

    for actual_tensor, expected_tensor in zip(actual, (expected_xyz, expected_features, expected_global, expected_mask), strict=True):
        assert torch.equal(actual_tensor, expected_tensor)

    actual_loss = actual[0].square().sum() + actual[1].square().sum() + actual[2].square().sum()
    expected_loss = (
        expected_xyz.square().sum()
        + expected_features.square().sum()
        + expected_global.square().sum()
    )
    actual_grads = torch.autograd.grad(actual_loss, (xyz, features), retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, (xyz, features))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.equal(actual_grad, expected_grad)


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
    model.last_molmo_world_scene_insert_positions = None
    model.last_molmo_token_roles = None
    model.last_molmo_prefix_position_ids = None
    model.last_molmo_native_length = 0
    model.last_molmo_scene_token_count = 0
    model.last_prefix_token_layout = ()
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=hidden_size)
    return model


def test_v5_prefix_layout_preserves_registered_positions_and_scene_split() -> None:
    model = _make_prefix_shell()
    model.config = SimpleNamespace(vlm_backend="molmo2_full")
    model.last_molmo_native_length = 5
    model.last_molmo_scene_token_count = 4
    model.last_molmo_prefix_position_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 2, 3, 2, 3]], dtype=torch.long
    )
    prefix = torch.randn(1, 9, 8)
    prefix_valid = torch.ones(1, 9, dtype=torch.bool)

    positions, scene_length = model._full_molmo_prefix_layout(prefix, prefix_valid)

    assert scene_length == 4
    assert torch.equal(positions, model.last_molmo_prefix_position_ids)
    suffix_valid = torch.tensor([[True, True, False]])
    joint_positions = model._compose_prefix_suffix_position_ids(
        prefix_valid,
        suffix_valid,
        positions,
    )
    assert torch.equal(joint_positions[:, :9], positions)
    assert torch.equal(joint_positions[:, 9:], torch.tensor([[5, 6, 6]]))
    assert torch.equal(
        model._cached_prefix_position_offsets(prefix_valid),
        torch.tensor([[5]]),
    )


def test_v5_prefix_layout_fails_closed_on_stale_scene_metadata() -> None:
    model = _make_prefix_shell()
    model.config = SimpleNamespace(vlm_backend="molmo2_full")
    model.last_molmo_native_length = 5
    model.last_molmo_scene_token_count = 3
    model.last_molmo_prefix_position_ids = torch.arange(8)[None]

    with pytest.raises(ValueError, match="prefix metadata drifted"):
        model._full_molmo_prefix_layout(
            torch.randn(1, 8, 8),
            torch.ones(1, 8, dtype=torch.bool),
        )


def test_full_prefix_appends_trainable_scene_without_changing_native_layout() -> None:
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
    assert torch.equal(model.last_molmo_scene_insert_positions, torch.tensor([416, 416]))
    assert torch.equal(prefix_valid.sum(dim=1), torch.tensor([415, 418]))
    assert prefix_valid[0, :413].all()
    assert not prefix_valid[0, 413:416].any()
    assert prefix_valid[0, 416:].all()
    assert prefix_valid[1].all()
    assert torch.equal(prefix[0, :413], native[0, :413])
    assert not prefix[0, 413:416].any()
    assert torch.equal(prefix[1, :416], native[1])
    assert torch.equal(prefix[:, 416:418], scene)
    assert model.last_prefix_token_layout == (
        "native_molmo_image_language",
        "ego_foreground_shadow",
        "ego_background_shadow",
    )
    roles = model.last_molmo_token_roles
    assert roles is not None
    assert set(roles.unique().tolist()) <= {TEXT_ROLE, IMAGE_ROLE, SCENE_ROLE, PAD_ROLE}
    assert (roles[:, 416:418] == SCENE_ROLE).all()

    # Native retains its hybrid mask. Scene is a later bidirectional block:
    # native cannot read scene, while scene reads every valid native/scene key.
    assert torch.equal(
        torch.nonzero(boundaries[0], as_tuple=False).flatten(),
        torch.tensor([0, 1, 411, 412, 416]),
    )
    assert torch.equal(
        torch.nonzero(boundaries[1], as_tuple=False).flatten(),
        torch.tensor([0, 1, 411, 412, 413, 414, 415, 416]),
    )
    allowed = make_att_2d_masks(prefix_valid, boundaries)
    expected = torch.zeros_like(allowed)
    native_columns = torch.arange(native.shape[1])
    native_causal = native_columns[None, :] <= native_columns[:, None]
    native_perception = image[:, :, None] & image[:, None, :]
    expected[:, :416, :416] = (
        prefix_valid[:, :416, None]
        & prefix_valid[:, None, :416]
        & (native_causal[None] | native_perception)
    )
    expected[:, 416:, :] = prefix_valid[:, 416:, None] & prefix_valid[:, None, :]
    assert torch.equal(allowed, expected)

    prefix.sum().backward()
    assert native.grad is None
    assert scene.grad is not None and scene.grad.abs().sum() > 0


def test_full_prefix_uses_each_samples_actual_image_boundary() -> None:
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
    assert torch.equal(
        torch.nonzero(boundaries[0], as_tuple=False).flatten(),
        torch.tensor([0, 1, 411, 412, 413, 414]),
    )
    assert torch.equal(
        torch.nonzero(boundaries[1], as_tuple=False).flatten(),
        torch.tensor([0, 1, 2, 412, 413, 414]),
    )
    assert torch.equal(
        model.last_molmo_prefix_position_ids[:, -2:],
        torch.tensor([[411, 412], [412, 413]]),
    )


def test_world_fg_bg_append_to_scene_shadow_without_moving_native() -> None:
    torch.manual_seed(5)
    model = _make_prefix_shell(hidden_size=8)
    model.config = SimpleNamespace(vlm_backend="molmo2_full")
    model.vlm_with_expert = SimpleNamespace(scale_input_embeddings=False)
    model.world_pointseg_object_proj = nn.Linear(4, 8)
    model.world_pointseg_background_proj = nn.Linear(4, 8)
    model.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(8))
    model.inference_ablation_modalities = frozenset()
    native, native_valid, native_image = _right_padded_native_batch(
        (2, 5), hidden_size=8
    )
    ego_scene = torch.randn(2, 2, 8)
    prefix, prefix_valid, prefix_boundaries = model._build_full_molmo_native_prefix(
        native,
        native_valid,
        native_image,
        ego_scene,
        torch.ones(2, 2, dtype=torch.bool),
        ablate_language=False,
    )
    scene_features = torch.randn(2, 2, 4)
    world_tokens = {
        "scene_global_tokens": scene_features,
        "scene_global_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    augmented, augmented_valid, boundaries, positions = (
        model._append_shared_expert_world_scene_prefix(
            prefix, prefix_valid, prefix_boundaries, world_tokens
        )
    )
    expected_world = torch.stack(
        [
            model.world_pointseg_object_proj(scene_features[:, 0]),
            model.world_pointseg_background_proj(scene_features[:, 1]),
        ],
        dim=1,
    )
    native_length = native.shape[1]
    assert torch.equal(augmented[:, :native_length], prefix[:, :native_length])
    assert torch.equal(augmented[:, native_length : native_length + 2], ego_scene)
    assert torch.allclose(augmented[:, -2:], expected_world)
    assert augmented_valid[:, -4:].all()
    assert torch.equal(boundaries[:, :-2], prefix_boundaries)
    assert not boundaries[:, -2:].any()
    assert torch.equal(
        positions[:, :native_length], model.last_molmo_prefix_position_ids[:, :native_length]
    )
    assert torch.equal(positions[:, -2:], positions[:, -4:-2])
    assert torch.equal(
        model.last_molmo_world_scene_insert_positions,
        torch.tensor([native_length + 2, native_length + 2]),
    )
    assert model.last_molmo_token_roles is not None
    assert (model.last_molmo_token_roles[:, -4:] == SCENE_ROLE).all()
    assert model.last_molmo_scene_token_count == 4
    assert model.last_prefix_token_layout[-2:] == (
        "world_foreground_shadow",
        "world_background_shadow",
    )


def test_disabled_scene_slots_reduce_exactly_to_native_molmo_attention_and_positions() -> None:
    torch.manual_seed(6)
    model = _make_prefix_shell(hidden_size=4)
    model.config = SimpleNamespace(vlm_backend="molmo2_full")
    model.vlm_with_expert = SimpleNamespace(
        expert_hidden_size=4,
        scale_input_embeddings=False,
    )
    model.world_pointseg_object_proj = nn.Linear(3, 4)
    model.world_pointseg_background_proj = nn.Linear(3, 4)
    model.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(4))
    model.inference_ablation_modalities = frozenset()

    native, native_valid, native_image = _right_padded_native_batch((3,), hidden_size=4)
    prefix, prefix_valid, prefix_boundaries = model._build_full_molmo_native_prefix(
        native,
        native_valid,
        native_image,
        torch.randn(1, 2, 4),
        torch.zeros(1, 2, dtype=torch.bool),
        ablate_language=False,
    )
    combined, combined_valid, combined_boundaries, positions = (
        model._append_shared_expert_world_scene_prefix(
            prefix,
            prefix_valid,
            prefix_boundaries,
            {
                "scene_global_tokens": torch.randn(1, 2, 3),
                "scene_global_mask": torch.zeros(1, 2, dtype=torch.bool),
            },
        )
    )

    assert combined.shape[1] == native.shape[1] + 4
    native_length = native.shape[1]
    assert torch.equal(positions[0, :native_length], torch.arange(native_length))
    combined_attention = make_att_2d_masks(combined_valid, combined_boundaries)
    native_attention = combined_attention[0, :native_length, :native_length]
    token_indices = torch.arange(native.shape[1])
    official_native_attention = (
        token_indices[None, :] <= token_indices[:, None]
    ) | (native_image[0, :, None] & native_image[0, None, :])
    assert torch.equal(native_attention, official_native_attention)
    assert not combined_attention[0, :native_length, native_length:].any()
    assert not combined_valid[0, native_length:].any()
    disabled_scene_indices = torch.arange(native_length, native_length + 4)
    assert not combined_valid[0, disabled_scene_indices].any()
    assert not combined_attention[0, disabled_scene_indices].any()
    assert not combined_attention[0, :, disabled_scene_indices].any()


def test_active_scene_shadow_preserves_native_molmo_hidden_states() -> None:
    """Native output must be invariant even when all four scene tokens are active."""

    torch.manual_seed(8)
    backend = _make_tiny_backend(num_layers=2)
    backend.eval()
    builder = _make_prefix_shell(hidden_size=8)
    builder.config = SimpleNamespace(vlm_backend="molmo2_full")
    builder.vlm_with_expert = SimpleNamespace(
        expert_hidden_size=8,
        scale_input_embeddings=False,
    )
    builder.world_pointseg_object_proj = nn.Linear(3, 8)
    builder.world_pointseg_background_proj = nn.Linear(3, 8)
    builder.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(8))
    builder.inference_ablation_modalities = frozenset()

    native, native_valid, native_image = _right_padded_native_batch((3,), hidden_size=8)
    native_indices_1d = torch.arange(native.shape[1])
    native_attention = (
        native_indices_1d[None, :] <= native_indices_1d[:, None]
    ) | (native_image[0, :, None] & native_image[0, None, :])
    native_attention = native_attention[None]
    native_positions = native_indices_1d[None]
    (native_hidden, _), _ = backend(
        attention_mask=native_attention,
        position_ids=native_positions,
        inputs_embeds=[native, None],
        use_cache=False,
        fill_kv_cache=False,
    )

    prefix, prefix_valid, prefix_boundaries = builder._build_full_molmo_native_prefix(
        native,
        native_valid,
        native_image,
        torch.randn(1, 2, 8),
        torch.ones(1, 2, dtype=torch.bool),
        ablate_language=False,
    )
    combined, combined_valid, combined_boundaries, combined_positions = (
        builder._append_shared_expert_world_scene_prefix(
            prefix,
            prefix_valid,
            prefix_boundaries,
            {
                "scene_global_tokens": torch.randn(1, 2, 3),
                "scene_global_mask": torch.ones(1, 2, dtype=torch.bool),
            },
        )
    )
    (combined_hidden, _), _ = backend(
        attention_mask=make_att_2d_masks(combined_valid, combined_boundaries),
        position_ids=combined_positions,
        inputs_embeds=[combined, None],
        use_cache=False,
        fill_kv_cache=False,
        prefix_scene_length=4,
    )
    assert torch.equal(combined_hidden[:, : native.shape[1]], native_hidden)


def test_grouped_query_head_mapping_matches_explicit_kv_repetition_forward_and_backward() -> None:
    torch.manual_seed(11)
    query = torch.randn(2, 7, 8, 16, requires_grad=True)
    key = torch.randn(2, 9, 2, 16, requires_grad=True)
    value = torch.randn(2, 9, 2, 16, requires_grad=True)
    mask = torch.rand(2, 7, 9) > 0.2
    mask[:, :, 0] = True

    actual = molmo_core._scaled_dot_product_attention(query, key, value, mask)
    repeated_key = key.transpose(1, 2)[:, :, None].expand(-1, -1, 4, -1, -1)
    repeated_value = value.transpose(1, 2)[:, :, None].expand(-1, -1, 4, -1, -1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        repeated_key.reshape(2, 8, 9, 16),
        repeated_value.reshape(2, 8, 9, 16),
        attn_mask=mask[:, None],
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2).contiguous().flatten(2)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    actual_grads = torch.autograd.grad(
        actual.square().sum(), (query, key, value), retain_graph=True
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (query, key, value)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual_grad, expected_grad, atol=2e-6, rtol=1e-5)


def test_split_molmo_embedding_matches_concatenated_table_forward_and_backward() -> None:
    torch.manual_seed(13)
    embedding = molmo_core.Molmo2Embedding(
        7,
        3,
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        embedding.embedding.normal_()
        embedding.new_embedding.normal_()
    token_ids = torch.tensor([[0, 6, 7, 9], [8, 2, 7, 1]])

    actual = embedding(token_ids)
    expected = torch.nn.functional.embedding(
        token_ids,
        torch.cat([embedding.embedding, embedding.new_embedding], dim=0),
    )
    assert torch.equal(actual, expected)

    actual_grads = torch.autograd.grad(
        actual.square().sum(),
        (embedding.embedding, embedding.new_embedding),
        retain_graph=True,
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(),
        (embedding.embedding, embedding.new_embedding),
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.equal(actual_grad, expected_grad)

    with pytest.raises(IndexError):
        embedding(torch.tensor([[10]]))
    with pytest.raises(IndexError):
        embedding(torch.tensor([[-1]]))


def test_shared_qk_rope_factors_match_two_independent_reference_calls() -> None:
    torch.manual_seed(17)
    query = torch.randn(2, 7, 8, 16, requires_grad=True)
    key = torch.randn(2, 7, 2, 16, requires_grad=True)
    positions = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 3, 3, 3]], dtype=torch.long
    )
    theta = 5_000_000.0

    cos, sin = molmo_core._molmo2_rope_factors(
        positions,
        head_dim=16,
        rope_theta=theta,
        device=query.device,
        dtype=query.dtype,
    )
    actual_query = molmo_core._apply_molmo2_rope_factors(query, cos, sin)
    actual_key = molmo_core._apply_molmo2_rope_factors(key, cos, sin)
    expected_query = molmo_core.apply_molmo2_rope(query, positions, theta)
    expected_key = molmo_core.apply_molmo2_rope(key, positions, theta)
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)

    actual_grads = torch.autograd.grad(
        actual_query.square().sum() + actual_key.square().sum(),
        (query, key),
        retain_graph=True,
    )
    expected_grads = torch.autograd.grad(
        expected_query.square().sum() + expected_key.square().sum(),
        (query, key),
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.equal(actual_grad, expected_grad)


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


def _joint_attention_and_positions(
    prefix_valid: torch.Tensor,
    action_valid: torch.Tensor,
    prefix_boundaries: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_attention = (
        prefix_valid[:, :, None] & prefix_valid[:, None, :]
        if prefix_boundaries is None
        else make_att_2d_masks(prefix_valid, prefix_boundaries)
    )
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


def _tiny_v5_attention_and_positions() -> tuple[torch.Tensor, torch.Tensor]:
    """One N=5, S=4, A=4 layout with WEP scene/action position reuse."""

    prefix_valid = torch.ones(1, 9, dtype=torch.bool)
    action_valid = torch.ones(1, 4, dtype=torch.bool)
    # N: causal token, two-token perception block, two causal language tokens.
    # S: one four-token bidirectional scene block after native.
    boundaries = torch.tensor(
        [[True, True, False, True, True, True, False, False, False]]
    )
    attention, _ = _joint_attention_and_positions(
        prefix_valid, action_valid, boundaries
    )
    positions = torch.tensor([[0, 1, 2, 3, 4, 2, 3, 2, 3, 5, 6, 5, 6]])
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
    prefix_valid = torch.cat(
        [prefix_valid, torch.ones(1, 2, dtype=torch.bool)], dim=1
    )
    prefix_boundaries = torch.cat(
        [prefix_boundaries, torch.zeros(1, 2, dtype=torch.bool)], dim=1
    )
    prefix_positions = torch.cat(
        [
            builder.last_molmo_prefix_position_ids,
            builder.last_molmo_prefix_position_ids[:, -2:],
        ],
        dim=1,
    )
    assert prefix.requires_grad
    assert torch.equal(
        torch.nonzero(prefix_boundaries[0], as_tuple=False).flatten(),
        torch.tensor([0, 1, 411, 412, 413]),
    )
    actions = torch.randn(1, 2, 6, requires_grad=True)
    action_valid = torch.ones(1, 2, dtype=torch.bool)
    attention, _ = _joint_attention_and_positions(
        prefix_valid, action_valid, prefix_boundaries
    )
    action_offset = (
        prefix_positions.masked_fill(~prefix_valid, -1).amax(dim=1, keepdim=True) + 1
    )
    positions = torch.cat(
        [prefix_positions, action_offset.expand(-1, actions.shape[1])], dim=1
    )
    (_, output), cache = backend(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix, actions],
        use_cache=False,
        fill_kv_cache=False,
        prefix_scene_length=4,
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


def test_layer_checkpointing_preserves_forward_and_input_expert_gradients() -> None:
    torch.manual_seed(41)
    plain = _make_tiny_backend(num_layers=3, gradient_checkpointing=False)
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing = True
    prefix_valid = torch.ones(1, 5, dtype=torch.bool)
    action_valid = torch.ones(1, 4, dtype=torch.bool)
    attention, positions = _joint_attention_and_positions(prefix_valid, action_valid)
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


def test_v5_action_only_checkpoint_preserves_scene_and_expert_gradients() -> None:
    torch.manual_seed(47)
    plain = _make_tiny_backend(
        num_layers=4, gradient_checkpointing=False
    )
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing = True
    checkpointed.gradient_checkpointing_layers_per_segment = 2
    attention, positions = _tiny_v5_attention_and_positions()
    prefix_plain = torch.randn(1, 9, 8, requires_grad=True)
    action_plain = torch.randn(1, 4, 6, requires_grad=True)
    prefix_checkpointed = prefix_plain.detach().clone().requires_grad_()
    action_checkpointed = action_plain.detach().clone().requires_grad_()

    (_, plain_output), _ = plain(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix_plain, action_plain],
        use_cache=False,
        fill_kv_cache=False,
        prefix_scene_length=4,
    )
    (_, checkpointed_output), _ = checkpointed(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix_checkpointed, action_checkpointed],
        use_cache=False,
        fill_kv_cache=False,
        prefix_scene_length=4,
    )
    assert torch.allclose(plain_output, checkpointed_output, atol=1e-6, rtol=1e-6)
    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()

    assert prefix_plain.grad is not None and prefix_checkpointed.grad is not None
    assert not prefix_plain.grad[:, :5].any()
    assert not prefix_checkpointed.grad[:, :5].any()
    assert prefix_plain.grad[:, 5:].abs().sum() > 0
    assert prefix_checkpointed.grad[:, 5:].abs().sum() > 0
    assert torch.allclose(
        prefix_plain.grad[:, 5:],
        prefix_checkpointed.grad[:, 5:],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        action_plain.grad, action_checkpointed.grad, atol=1e-6, rtol=1e-5
    )
    for (_, plain_parameter), (_, checkpointed_parameter) in zip(
        plain.lm_expert.named_parameters(),
        checkpointed.lm_expert.named_parameters(),
        strict=True,
    ):
        assert plain_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        assert torch.allclose(
            plain_parameter.grad,
            checkpointed_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        )
    assert all(parameter.grad is None for parameter in plain.vlm.parameters())
    assert all(parameter.grad is None for parameter in checkpointed.vlm.parameters())


def test_production_full_backend_refuses_missing_native_scene_split() -> None:
    backend = _make_tiny_backend(num_layers=2)
    backend.requires_native_scene_split = True
    attention, positions = _tiny_v5_attention_and_positions()

    with pytest.raises(ValueError, match="explicit non-zero prefix_scene_length"):
        backend(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[torch.randn(1, 9, 8), torch.randn(1, 4, 6)],
            use_cache=False,
            fill_kv_cache=False,
        )


def test_full_prefix_cache_matches_joint_forward_with_action_only_suffix() -> None:
    torch.manual_seed(53)
    backend = _make_tiny_backend(num_layers=4)
    backend.eval()
    prefix = torch.randn(1, 7, 8)
    actions = torch.randn(1, 4, 6)
    prefix_valid = torch.ones(1, 7, dtype=torch.bool)
    prefix_boundaries = torch.tensor([[True, True, False, False, True, True, True]])
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


def test_v5_native_scene_prefill_cache_matches_joint_action_forward() -> None:
    torch.manual_seed(59)
    backend = _make_tiny_backend(num_layers=4)
    backend.eval()
    prefix = torch.randn(1, 9, 8)
    actions = torch.randn(1, 4, 6)
    attention, positions = _tiny_v5_attention_and_positions()
    prefix_attention = attention[:, :9, :9]
    suffix_attention = attention[:, 9:, :]

    with torch.no_grad():
        (joint_prefix, joint_output), _ = backend(
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[prefix, actions],
            use_cache=False,
            fill_kv_cache=False,
            prefix_scene_length=4,
        )
        (cached_prefix, _), cache = backend(
            attention_mask=prefix_attention,
            position_ids=positions[:, :9],
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
            prefix_scene_length=4,
        )
        assert cache is not None and len(cache) == 4
        (_, cached_output), _ = backend(
            attention_mask=suffix_attention,
            position_ids=positions[:, 9:],
            past_key_values=cache,
            inputs_embeds=[None, actions],
            use_cache=True,
            fill_kv_cache=False,
        )

    assert torch.equal(joint_prefix, cached_prefix)
    assert torch.allclose(joint_output, cached_output, atol=1e-6, rtol=1e-6)
    assert not cache[0]["cross_attention_projected"]
    assert cache[1]["cross_attention_projected"]
    assert all(layer["key_states"].shape[1] == 9 for layer in cache.values())
    assert all(
        layer["cache_topology"] == FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY
        and layer["native_length"] == 5
        and layer["scene_length"] == 4
        for layer in cache.values()
    )


def test_v5_cache_rejects_missing_or_wrong_projection_marker() -> None:
    backend = _make_tiny_backend(num_layers=2)
    backend.requires_native_scene_split = True
    attention, positions = _tiny_v5_attention_and_positions()
    prefix_attention = attention[:, :9, :9]
    suffix_attention = attention[:, 9:, :]
    prefix = torch.randn(1, 9, 8)
    actions = torch.randn(1, 4, 6)

    (_, _), cache = backend(
        attention_mask=prefix_attention,
        position_ids=positions[:, :9],
        inputs_embeds=[prefix, None],
        use_cache=True,
        fill_kv_cache=True,
        prefix_scene_length=4,
    )
    assert cache is not None
    cache[1].pop("cache_topology")
    with pytest.raises(ValueError, match="topology marker"):
        backend(
            attention_mask=suffix_attention,
            position_ids=positions[:, 9:],
            past_key_values=cache,
            inputs_embeds=[None, actions],
            use_cache=True,
            fill_kv_cache=False,
        )


def test_wep_prefix_backend_state_dict_strict_reload_is_exact() -> None:
    torch.manual_seed(61)
    source = _make_tiny_backend(num_layers=2)
    reloaded = _make_tiny_backend(num_layers=2)
    reloaded.load_state_dict(source.state_dict(), strict=True)
    source.eval()
    reloaded.eval()

    prefix = torch.randn(1, 5, 8)
    actions = torch.randn(1, 4, 6)
    attention, positions = _joint_attention_and_positions(
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
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
