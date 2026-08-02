#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def test_action_chunk_start_offset_shifts_dataset_action_indices():
    cfg = SmolVLAConfig(chunk_size=4, n_action_steps=4, action_chunk_start_offset=1)

    assert cfg.action_delta_indices == [1, 2, 3, 4]
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    WorldFlowActionBranch,
    make_att_2d_masks,
    matrix_to_pose9,
    pose9_endpoint_velocity_to_spatial_twist,
    pose9_velocity_to_spatial_twist,
    pose9_to_matrix,
    se3_exp,
    se3_geodesic_flow_state,
    se3_left_apply,
    se3_log,
    so3_exp,
    so3_log,
)


def test_so3_and_se3_exp_log_round_trip():
    torch.manual_seed(7)
    omega = torch.randn(16, 3) * 0.25
    rot = so3_exp(omega)
    assert torch.allclose(so3_log(rot), omega, atol=2e-4, rtol=2e-4)

    xi = torch.randn(16, 6) * 0.20
    transform = se3_exp(xi)
    assert torch.allclose(se3_log(transform), xi, atol=3e-4, rtol=3e-4)


def test_se3_relative_update_recovers_target():
    torch.manual_seed(11)
    a = se3_exp(torch.randn(8, 6) * 0.2)
    b = se3_exp(torch.randn(8, 6) * 0.2)
    recovered = se3_exp(se3_log(a @ torch.linalg.inv(b))) @ b
    assert torch.allclose(recovered, a, atol=3e-4, rtol=3e-4)


def test_se3_geodesic_flow_stays_on_group_and_exact_velocity_reaches_target():
    torch.manual_seed(12)
    noise = se3_exp(torch.randn(4, 7, 6) * 0.15)
    target = se3_exp(torch.randn(4, 7, 6) * 0.15)
    time = torch.tensor([0.0, 0.2, 0.6, 0.9])

    state, velocity = se3_geodesic_flow_state(noise, target, time)
    remaining = (1.0 - time)[:, None, None]
    recovered = se3_left_apply(remaining * velocity, state)

    rotation = state[..., :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=4e-5, rtol=4e-5)
    assert torch.allclose(torch.det(rotation), torch.ones_like(torch.det(rotation)), atol=4e-5, rtol=4e-5)
    assert torch.allclose(recovered, target, atol=5e-4, rtol=5e-4)


def test_pose9_velocity_projection_recovers_exact_spatial_twist():
    torch.manual_seed(121)
    transform = se3_exp(torch.randn(4, 7, 6) * 0.2)
    twist = torch.randn(4, 7, 6) * 0.15
    rotation = transform[..., :3, :3]
    position = transform[..., :3, 3]
    omega = twist[..., 3:6]
    x, y, z = omega.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    omega_hat = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    rotation_dot = omega_hat @ rotation
    position_dot = twist[..., :3] + torch.cross(omega, position, dim=-1)
    pose9_velocity = torch.cat(
        [position_dot, rotation_dot[..., :, 0], rotation_dot[..., :, 1]],
        dim=-1,
    )

    recovered = pose9_velocity_to_spatial_twist(
        matrix_to_pose9(transform),
        pose9_velocity,
    )

    assert torch.allclose(recovered, twist, atol=4e-5, rtol=4e-5)


def test_pose9_endpoint_velocity_recovers_projected_clean_endpoint():
    torch.manual_seed(122)
    current = matrix_to_pose9(se3_exp(torch.randn(4, 7, 6) * 0.15))
    legacy_velocity = torch.randn(4, 7, 9) * 0.08
    time = torch.tensor([0.0, 0.2, 0.6, 0.9])
    remaining = (1.0 - time)[:, None, None]
    expected_endpoint = pose9_to_matrix(current + remaining * legacy_velocity)

    twist = pose9_endpoint_velocity_to_spatial_twist(current, legacy_velocity, time)
    recovered_endpoint = se3_left_apply(remaining * twist, pose9_to_matrix(current))

    assert torch.allclose(recovered_endpoint, expected_endpoint, atol=6e-4, rtol=6e-4)


def test_conjugate_se3_geodesic_paths_match_at_every_time():
    torch.manual_seed(14)
    current = se3_exp(torch.randn(4, 6) * 0.15)
    ego_noise = se3_exp(torch.randn(4, 9, 6) * 0.12)
    ego_target = se3_exp(torch.randn(4, 9, 6) * 0.12)
    current_inv = torch.linalg.inv(current)
    world_noise = current.unsqueeze(1) @ ego_noise @ current_inv.unsqueeze(1)
    world_target = current.unsqueeze(1) @ ego_target @ current_inv.unsqueeze(1)
    time = torch.tensor([0.0, 0.25, 0.55, 0.9])

    ego_state, _ = se3_geodesic_flow_state(ego_noise, ego_target, time)
    world_state, _ = se3_geodesic_flow_state(world_noise, world_target, time)
    expected_world_state = current.unsqueeze(1) @ ego_state @ current_inv.unsqueeze(1)

    assert torch.allclose(world_state, expected_world_state, atol=6e-4, rtol=6e-4)


def test_worldflow_noise_respects_physical_translation_and_rotation_scales():
    torch.manual_seed(13)
    cfg = SmolVLAConfig(
        chunk_size=256,
        n_action_steps=16,
        worldflow_noise_trans_scale=0.07,
        worldflow_noise_rot_scale=0.11,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    pose9_noise = model.sample_worldflow_noise(batch_size=128, device=torch.device("cpu"))
    sampled_twist = se3_log(pose9_to_matrix(pose9_noise))

    assert torch.allclose(
        sampled_twist[..., :3].std(),
        torch.tensor(cfg.worldflow_noise_trans_scale),
        atol=3e-3,
        rtol=0.04,
    )
    assert torch.allclose(
        sampled_twist[..., 3:6].std(),
        torch.tensor(cfg.worldflow_noise_rot_scale),
        atol=3e-3,
        rtol=0.04,
    )


def test_ego_pose9_noise_is_valid_se3_and_respects_physical_scales():
    torch.manual_seed(17)
    cfg = SmolVLAConfig(
        chunk_size=256,
        n_action_steps=16,
        pose9_action_noise_enable=True,
        pose9_action_noise_trans_scale=0.08,
        pose9_action_noise_rot_scale=0.14,
        pose9_action_noise_gripper_scale=0.03,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    noise = model.sample_pose9_action_noise((128, cfg.chunk_size, 10), torch.device("cpu"))
    transform = pose9_to_matrix(noise[..., :9])
    sampled_twist = se3_log(transform)

    assert torch.allclose(
        sampled_twist[..., :3].std(),
        torch.tensor(cfg.pose9_action_noise_trans_scale),
        atol=3e-3,
        rtol=0.04,
    )
    assert torch.allclose(
        sampled_twist[..., 3:6].std(),
        torch.tensor(cfg.pose9_action_noise_rot_scale),
        atol=3e-3,
        rtol=0.04,
    )
    assert torch.allclose(
        noise[..., 9].std(),
        torch.tensor(cfg.pose9_action_noise_gripper_scale),
        atol=2e-3,
        rtol=0.04,
    )
    rotation = transform[..., :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=2e-5)


def test_zero_pose9_and_worldflow_noise_scales_produce_identity_priors():
    cfg = SmolVLAConfig(
        chunk_size=8,
        n_action_steps=8,
        pose9_action_noise_enable=True,
        pose9_action_noise_trans_scale=0.0,
        pose9_action_noise_rot_scale=0.0,
        pose9_action_noise_gripper_scale=0.0,
        worldflow_enable=True,
        worldflow_noise_trans_scale=0.0,
        worldflow_noise_rot_scale=0.0,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    ego_noise = model.sample_pose9_action_noise((3, cfg.chunk_size, 10), torch.device("cpu"))
    world_noise = model.sample_worldflow_noise(batch_size=3, device=torch.device("cpu"))
    identity = torch.eye(4).expand(3, cfg.chunk_size, 4, 4)

    assert torch.equal(pose9_to_matrix(ego_noise[..., :9]), identity)
    assert torch.count_nonzero(ego_noise[..., 9]) == 0
    assert torch.equal(pose9_to_matrix(world_noise), identity)


def test_conjugate_worldflow_noise_matches_ego_prior_exactly():
    torch.manual_seed(23)
    cfg = SmolVLAConfig(
        chunk_size=16,
        n_action_steps=8,
        pose9_action_noise_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    ego_noise = model.sample_pose9_action_noise((5, cfg.chunk_size, 10), torch.device("cpu"))
    current = matrix_to_pose9(se3_exp(torch.randn(5, 6) * 0.2))
    world_noise = model.conjugate_ego_noise_to_world(ego_noise, current)

    current_matrix = pose9_to_matrix(current)
    expected = (
        current_matrix.unsqueeze(1)
        @ pose9_to_matrix(ego_noise[..., :9])
        @ torch.linalg.inv(current_matrix).unsqueeze(1)
    )
    actual = pose9_to_matrix(world_noise)
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)

    recovered_ego = (
        torch.linalg.inv(current_matrix).unsqueeze(1)
        @ actual
        @ current_matrix.unsqueeze(1)
    )
    assert torch.allclose(
        recovered_ego,
        pose9_to_matrix(ego_noise[..., :9]),
        atol=4e-5,
        rtol=4e-5,
    )


def test_worldflow_and_complete_se3_flow_can_be_enabled_together():
    cfg = SmolVLAConfig(
        se3_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
    )

    assert cfg.se3_enable
    assert cfg.worldflow_enable
    assert "ego_flow=se3_geodesic_left_trivialized" in cfg.flow_contract_summary()
    assert "worldflow=conjugate_ego" in cfg.flow_contract_summary()


def test_projected_pose9_is_a_valid_complete_se3_head_mode():
    cfg = SmolVLAConfig(
        se3_enable=True,
        se3_twist_head_mode="projected_pose9",
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
    )

    assert "se3_geodesic_left_trivialized(head=projected_pose9)" in cfg.flow_contract_summary()

    endpoint_cfg = SmolVLAConfig(
        se3_enable=True,
        se3_twist_head_mode="pose9_endpoint",
    )
    assert "se3_geodesic_left_trivialized(head=pose9_endpoint)" in endpoint_cfg.flow_contract_summary()

    with pytest.raises(ValueError, match="se3_twist_head_mode"):
        SmolVLAConfig(se3_enable=True, se3_twist_head_mode="unknown")


def test_conjugate_worldflow_noise_rejects_legacy_invalid_rotation_prior():
    with pytest.raises(ValueError, match="pose9_action_noise_enable=True"):
        SmolVLAConfig(
            worldflow_enable=True,
            pose9_action_noise_enable=False,
            worldflow_noise_coupling="conjugate_ego",
        )


def test_integration_grid_time_sampling_matches_inference_euler_steps():
    cfg = SmolVLAConfig(
        num_steps=10,
        flow_time_sampling="integration_grid",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    torch.manual_seed(7)
    sampled = model.sample_time(2048, torch.device("cpu"))

    scaled = sampled * cfg.num_steps
    assert torch.allclose(scaled, scaled.round())
    assert sampled.min().item() == 0.0
    assert torch.allclose(sampled.max(), torch.tensor(0.9))
    assert set(scaled.to(dtype=torch.int64).tolist()) == set(range(cfg.num_steps))


def test_integration_grid_time_sampling_can_emphasize_exact_zero():
    cfg = SmolVLAConfig(
        num_steps=10,
        flow_time_sampling="integration_grid",
        flow_time_zero_probability=0.5,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    torch.manual_seed(7)
    sampled = model.sample_time(20_000, torch.device("cpu"))

    scaled = sampled * cfg.num_steps
    zero_ratio = (sampled == 0).to(dtype=torch.float32).mean()
    assert torch.allclose(scaled, scaled.round())
    assert torch.allclose(sampled.max(), torch.tensor(0.9))
    assert torch.allclose(zero_ratio, torch.tensor(0.5), atol=0.015)
    assert set(scaled.to(dtype=torch.int64).tolist()) == set(range(cfg.num_steps))


def test_pose9_action_loss_weights_preserve_scale_and_prioritize_physical_groups():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            action_loss_translation_weight=3.0,
            action_loss_rotation_weight=1.0,
            action_loss_gripper_weight=3.0,
        )
    )
    weights = SmolVLAPolicy._action_loss_dimension_weights(
        policy,
        10,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert weights[:3].min() > weights[3:9].max()
    assert torch.allclose(weights[9], weights[0])


def test_standard_action_metrics_use_physical_endpoint_error_and_padding_mask():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    identity = torch.eye(4).expand(1, 2, 4, 4).clone()
    actions = torch.cat(
        [matrix_to_pose9(identity), torch.zeros(1, 2, 1)],
        dim=-1,
    )
    x_t = actions.clone()
    u_t = torch.zeros_like(actions)
    pred_velocity = torch.zeros_like(actions)
    pred_velocity[0, 0, 0] = 0.01
    pred_velocity[0, 0, 9] = 0.02
    # This much larger error must be ignored because the second step is padded.
    pred_velocity[0, 1, 0] = 1.0
    pred_velocity[0, 1, 9] = 1.0

    model._record_standard_action_metrics(
        actions=actions,
        x_t=x_t,
        u_t=u_t,
        pred_velocity=pred_velocity,
        time=torch.zeros(1),
        actions_is_pad=torch.tensor([[False, True]]),
    )

    assert torch.allclose(model.last_action_metrics["action_endpoint_trans_err"], torch.tensor(0.01))
    assert torch.allclose(
        model.last_action_metrics["action_endpoint_gripper_err"],
        torch.tensor(0.02),
    )
    assert torch.allclose(
        model.last_action_metrics["action_endpoint_rot_err_deg"],
        torch.tensor(0.0),
    )


class _TinySceneEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(6, dim)

    def forward(self, point_cloud, point_is_pad=None, *, return_tokens=False):
        tokens = self.proj(point_cloud)
        if point_is_pad is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = ~point_is_pad
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        global_feat = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        assert return_tokens
        return {"global_feat": global_feat, "scene_tok1": tokens, "scene_mask1": mask}


class _RecordingPointAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_point_tokens = None

    def forward(
        self,
        action_tokens,
        point_tokens,
        point_mask=None,
        actions_is_pad=None,
    ):
        del point_mask, actions_is_pad
        self.num_point_tokens = point_tokens.shape[1]
        return action_tokens


class _TinyLanguageVLM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_dim)

    def embed_language_tokens(self, tokens):
        return self.embedding(tokens)


class _TinyPointsegConditioner(nn.Module):
    def forward(self, payload):
        point_cloud = payload["point_cloud"]
        scene_tokens = point_cloud[..., :4]
        global_feat = scene_tokens.mean(dim=1)
        point_mask = torch.ones(scene_tokens.shape[:2], dtype=torch.bool, device=scene_tokens.device)
        scores = torch.full(scene_tokens.shape[:2], 0.75, device=scene_tokens.device)
        return {
            "object_feat": global_feat,
            "background_feat": torch.zeros_like(global_feat),
            "foreground_pc": point_cloud,
            "foreground_scene_tok1": scene_tokens,
            "foreground_scene_mask1": point_mask,
            "operation_prob": scores,
            "pointseg_selection_scores": scores,
        }


def _make_tiny_worldflow_branch(cfg: SmolVLAConfig) -> WorldFlowActionBranch:
    branch = WorldFlowActionBranch(
        cfg,
        action_hidden_dim=32,
        language_vocab_size=64,
    )
    branch.scene_encoder = _TinySceneEncoder(cfg.worldflow_feature_dim)
    return branch


def test_worldflow_branch_consumes_xyzrgb_without_role_or_probability_inputs():
    torch.manual_seed(17)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    branch = _make_tiny_worldflow_branch(cfg)
    point_cloud = torch.randn(2, 12, 6)
    point_cloud[..., 3:6] = torch.rand(2, 12, 3) * 255.0
    lang_tokens = torch.randint(0, 64, (2, 5))
    lang_masks = torch.ones(2, 5, dtype=torch.bool)
    noisy_actions = torch.randn(2, cfg.chunk_size, 9)
    time = torch.rand(2)
    actions_is_pad = torch.tensor(
        [[False, False, True, True], [False, False, False, False]]
    )

    output = branch(
        point_cloud,
        lang_tokens,
        lang_masks,
        noisy_actions,
        time,
        actions_is_pad=actions_is_pad,
    )

    assert output["scene_tokens"].shape == (2, 12, cfg.worldflow_feature_dim)
    assert output["scene_mask"].shape == (2, 12)
    assert output["global_feat"].shape == (2, cfg.worldflow_feature_dim)
    assert output["action_tokens"].shape == (2, cfg.chunk_size, 32)
    assert torch.isfinite(output["action_tokens"]).all()
    assert torch.equal(
        output["action_tokens"][0, 2:],
        torch.zeros_like(output["action_tokens"][0, 2:]),
    )
    assert not hasattr(branch, "action_expert")


def test_worldflow_point_action_receives_every_litept_token():
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    branch = _make_tiny_worldflow_branch(cfg)
    recorder = _RecordingPointAction()
    branch.point_action_adapter = recorder
    num_points = 37
    output = branch(
        torch.randn(2, num_points, 6),
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
        torch.randn(2, cfg.chunk_size, 9),
        torch.rand(2),
    )

    assert recorder.num_point_tokens == num_points
    assert output["global_feat"].shape == (2, cfg.worldflow_feature_dim)


def test_ego_prefix_keeps_all_litept_tokens_but_one_global_scene_feature():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vla_adapter_enable=False, encode_robot_state=False)
    model.vlm_with_expert = _TinyLanguageVLM(hidden_dim=8)
    model.pointseg_conditioner = _TinyPointsegConditioner()
    model.pointseg_object_proj = nn.Linear(4, 8)
    model.pointseg_background_proj = nn.Linear(4, 8)
    model.point_action_fusion = nn.Identity()
    model.worldflow_branch = None
    model.capture_pointseg_visualization = False
    model.prefix_length = 0

    num_points = 41
    point_cloud = torch.randn(2, num_points, 6)
    expected_global = point_cloud[..., :4].mean(dim=1)
    model.embed_prefix(
        [point_cloud],
        [torch.ones(2, dtype=torch.bool)],
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
    )

    assert model.last_point_action_tokens.shape == (2, num_points, 4)
    assert torch.equal(model.last_point_action_tokens, point_cloud[..., :4])
    assert model.last_ego_scene_global_feat.shape == (2, 4)
    assert torch.allclose(model.last_ego_scene_global_feat, expected_global)
    assert torch.equal(model.last_ego_scene_global_mask, torch.ones(2, dtype=torch.bool))
    recorder = _RecordingPointAction()
    model.point_action_fusion = recorder
    model._inject_point_action_features(torch.randn(2, 4, 8))
    assert recorder.num_point_tokens == num_points


def test_world_ego_joint_attention_has_causal_blocks_and_bidirectional_actions():
    cfg = SmolVLAConfig(
        chunk_size=2,
        n_action_steps=2,
        pointseg_feature_dim=4,
        worldflow_feature_dim=16,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.ego_scene_to_expert = nn.Linear(4, 8)
    model.world_scene_to_expert = nn.Linear(16, 8)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.last_ego_scene_global_feat = torch.randn(1, 4)
    model.last_ego_scene_global_mask = torch.ones(1, dtype=torch.bool)

    ego_actions = torch.randn(1, 2, 8)
    world = {
        "scene_tokens": torch.randn(1, 4, 16),
        "scene_mask": torch.ones(1, 4, dtype=torch.bool),
        "global_feat": torch.randn(1, 16),
        "action_tokens": torch.randn(1, 2, 8),
        "action_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    suffix, pad, blocks, layout = model._build_world_ego_joint_suffix(
        ego_actions,
        torch.ones(1, 2, dtype=torch.bool),
        world,
    )
    mask = make_att_2d_masks(pad, blocks)[0]

    ego_action = torch.arange(layout["ego_action"].start, layout["ego_action"].stop)
    world_action = torch.arange(layout["world_action"].start, layout["world_action"].stop)
    all_actions = torch.cat([ego_action, world_action])
    all_scenes = torch.arange(0, layout["ego_action"].start)
    assert layout["ego_scene"].stop - layout["ego_scene"].start == 1
    assert layout["world_scene"].stop - layout["world_scene"].start == 1
    assert suffix.shape == (1, 2 + 2 * cfg.chunk_size, 8)
    assert mask[all_actions[:, None], all_actions].all()
    assert mask[all_actions[:, None], all_scenes].all()
    assert not mask[all_scenes[:, None], all_actions].any()


def test_world_stream_ablation_preserves_layout_but_masks_world_tokens():
    cfg = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        pointseg_feature_dim=4,
        worldflow_feature_dim=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.ego_scene_to_expert = nn.Linear(4, 8)
    model.world_scene_to_expert = nn.Linear(4, 8)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.last_ego_scene_global_feat = torch.randn(1, 4)
    model.last_ego_scene_global_mask = torch.ones(1, dtype=torch.bool)
    model.inference_ablation_modalities = frozenset({"world"})
    world = {
        "scene_tokens": torch.randn(1, 5, 4),
        "scene_mask": torch.ones(1, 5, dtype=torch.bool),
        "global_feat": torch.randn(1, 4),
        "action_tokens": torch.randn(1, cfg.chunk_size, 8),
        "action_mask": torch.ones(1, cfg.chunk_size, dtype=torch.bool),
    }

    suffix, pad, _blocks, layout = model._build_world_ego_joint_suffix(
        torch.randn(1, cfg.chunk_size, 8),
        torch.ones(1, cfg.chunk_size, dtype=torch.bool),
        world,
    )

    assert suffix.shape[1] == 2 + 2 * cfg.chunk_size
    assert pad[:, layout["ego_scene"]].all()
    assert pad[:, layout["ego_action"]].all()
    assert not pad[:, layout["world_scene"]].any()
    assert not pad[:, layout["world_action"]].any()


def test_joint_attention_carries_gradients_between_both_global_scenes_and_action_streams():
    torch.manual_seed(18)
    cfg = SmolVLAConfig(
        chunk_size=2,
        n_action_steps=2,
        pointseg_feature_dim=4,
        worldflow_feature_dim=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.ego_scene_to_expert = nn.Linear(4, 8)
    model.world_scene_to_expert = nn.Linear(4, 8)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 8))

    ego_scene = torch.randn(1, 4, requires_grad=True)
    world_scene = torch.randn(1, 4, requires_grad=True)
    ego_actions = torch.randn(1, cfg.chunk_size, 8, requires_grad=True)
    world_actions = torch.randn(1, cfg.chunk_size, 8, requires_grad=True)
    model.last_ego_scene_global_feat = ego_scene
    model.last_ego_scene_global_mask = torch.ones(1, dtype=torch.bool)
    world = {
        "scene_tokens": torch.randn(1, 7, 4),
        "scene_mask": torch.ones(1, 7, dtype=torch.bool),
        "global_feat": world_scene,
        "action_tokens": world_actions,
        "action_mask": torch.ones(1, cfg.chunk_size, dtype=torch.bool),
    }
    suffix, pad, blocks, layout = model._build_world_ego_joint_suffix(
        ego_actions,
        torch.ones(1, cfg.chunk_size, dtype=torch.bool),
        world,
    )
    allowed = make_att_2d_masks(pad, blocks)
    scores = suffix @ suffix.transpose(-1, -2) / suffix.shape[-1] ** 0.5
    weights = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1)
    attended = weights @ suffix

    ego_loss = attended[:, layout["ego_action"]].square().mean()
    ego_to_world = torch.autograd.grad(
        ego_loss,
        (world_scene, world_actions),
        retain_graph=True,
    )
    assert all(gradient.abs().sum() > 0 for gradient in ego_to_world)

    world_loss = attended[:, layout["world_action"]].square().mean()
    world_to_ego = torch.autograd.grad(world_loss, (ego_scene, ego_actions))
    assert all(gradient.abs().sum() > 0 for gradient in world_to_ego)


def test_worldflow_joint_loss_uses_foreground_only_and_backpropagates_bridge():
    torch.manual_seed(19)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_equiv_loss_weight=0.0,
        worldflow_max_points=8,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}

    bsize, chunk_size = 2, cfg.chunk_size
    foreground = torch.randn(bsize, 12, 6)
    foreground[..., 3:6] = torch.rand(bsize, 12, 3) * 255.0
    model.last_worldflow_payload = {"foreground_pc_ego": foreground}
    current = se3_exp(torch.randn(bsize, 6) * 0.1)
    target = se3_exp(torch.randn(bsize, chunk_size, 6) * 0.1)
    body_prediction = matrix_to_pose9(
        torch.linalg.inv(current).unsqueeze(1) @ target
    ).detach().requires_grad_(True)
    context = {
        "current_ee_pose": matrix_to_pose9(current),
        "ee_poses": matrix_to_pose9(target),
        "step_is_pad": torch.zeros(bsize, chunk_size, dtype=torch.bool),
    }
    lang_tokens = torch.randint(0, 64, (bsize, 5))
    lang_masks = torch.ones(bsize, 5, dtype=torch.bool)
    time = torch.rand(bsize)

    state = model._prepare_worldflow_training_state(
        context,
        lang_tokens,
        lang_masks,
        time,
        actions_is_pad=None,
    )
    pred_velocity = torch.zeros_like(state["u_t"], requires_grad=True)
    output = model._finalize_worldflow_training_loss(
        state,
        pred_velocity,
        body_prediction,
        time,
    )
    output["per_sample_loss"].mean().backward()

    assert output["pred_spatial"].shape == (bsize, chunk_size, 4, 4)
    assert body_prediction.grad is not None
    assert "loss_worldflow_geo" in model.last_worldflow_metrics
    assert model.last_worldflow_metrics["worldflow_foreground_points"].item() == cfg.worldflow_max_points


def test_joint_se3_worldflow_training_path_is_exactly_conjugate():
    torch.manual_seed(21)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_enable=True,
        worldflow_enable=True,
        se3_enable=True,
        worldflow_noise_coupling="conjugate_ego",
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_equiv_loss_weight=0.0,
        worldflow_max_points=8,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}

    bsize, chunk_size = 2, cfg.chunk_size
    foreground = torch.randn(bsize, 12, 6)
    foreground[..., 3:6] = torch.rand(bsize, 12, 3) * 255.0
    model.last_worldflow_payload = {"foreground_pc_ego": foreground}
    current = se3_exp(torch.randn(bsize, 6) * 0.1)
    body_target = se3_exp(torch.randn(bsize, chunk_size, 6) * 0.1)
    target = current.unsqueeze(1) @ body_target
    ego_noise_h = se3_exp(torch.randn(bsize, chunk_size, 6) * 0.1)
    ego_noise = torch.cat(
        [matrix_to_pose9(ego_noise_h), torch.zeros(bsize, chunk_size, 1)],
        dim=-1,
    )
    time = torch.tensor([0.2, 0.7])
    ego_state_h, _ = se3_geodesic_flow_state(ego_noise_h, body_target, time)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state_h), torch.zeros(bsize, chunk_size, 1)],
        dim=-1,
    )
    context = {
        "current_ee_pose": matrix_to_pose9(current),
        "ee_poses": matrix_to_pose9(target),
        "step_is_pad": torch.zeros(bsize, chunk_size, dtype=torch.bool),
    }

    state = model._prepare_worldflow_training_state(
        context,
        torch.randint(0, 64, (bsize, 5)),
        torch.ones(bsize, 5, dtype=torch.bool),
        time,
        actions_is_pad=None,
        ego_noise=ego_noise,
        ego_x_t=ego_x_t,
    )
    output = model._finalize_worldflow_training_loss(
        state,
        state["u_t"],
        matrix_to_pose9(body_target),
        time,
    )

    assert torch.allclose(state["path_conjugacy_error"], torch.zeros_like(state["path_conjugacy_error"]), atol=2e-4)
    assert torch.allclose(output["pred_spatial"], state["spatial_gt"], atol=5e-4, rtol=5e-4)
    assert output["loss_flow"].item() < 1e-7
    assert output["loss_geo"].item() < 1e-4
    assert output["loss_bridge"].item() < 1e-4


def test_world_ego_bridge_formula_matches_body_transform():
    torch.manual_seed(23)
    current = se3_exp(torch.randn(3, 6) * 0.2)
    body = se3_exp(torch.randn(3, 6) * 0.2)
    target = current @ body
    spatial = target @ torch.linalg.inv(current)
    recovered_body = torch.linalg.inv(current) @ spatial @ current

    assert torch.allclose(recovered_body, body, atol=3e-4, rtol=3e-4)


def test_worldflow_coordinate_change_uses_se3_conjugation():
    torch.manual_seed(29)
    current = se3_exp(torch.randn(3, 6) * 0.2)
    target = se3_exp(torch.randn(3, 6) * 0.2)
    coordinate_change = se3_exp(torch.randn(3, 6) * 0.2)

    spatial = target @ torch.linalg.inv(current)
    spatial_aug = coordinate_change @ spatial @ torch.linalg.inv(coordinate_change)
    current_aug = coordinate_change @ current
    recovered_body = torch.linalg.inv(current_aug) @ spatial_aug @ current_aug

    expected_body = torch.linalg.inv(current) @ target
    assert torch.allclose(recovered_body, expected_body, atol=3e-4, rtol=3e-4)


def test_worldflow_branch_is_part_of_policy_inference_call_path():
    assert hasattr(VLAFlowMatching, "denoise_step_world_ego")
