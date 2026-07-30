#!/usr/bin/env python

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    WorldFlowActionBranch,
    make_att_2d_masks,
    matrix_to_pose9,
    se3_exp,
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
