#!/usr/bin/env python

import inspect

import torch
from torch import nn
from transformers import Gemma2Config

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    WorldFlowActionBranch,
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


class _IdentityPointAction(nn.Module):
    def forward(
        self,
        action_tokens,
        point_tokens,
        point_mask=None,
        actions_is_pad=None,
    ):
        del point_tokens, point_mask, actions_is_pad
        return action_tokens


def _make_tiny_worldflow_branch(cfg: SmolVLAConfig) -> WorldFlowActionBranch:
    expert_config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    branch = WorldFlowActionBranch(
        cfg,
        action_hidden_dim=32,
        language_vocab_size=64,
        action_expert_config=expert_config,
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
        worldflow_action_expert_layers=1,
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

    velocity = branch(
        point_cloud,
        lang_tokens,
        lang_masks,
        noisy_actions,
        time,
        actions_is_pad=actions_is_pad,
    )

    assert velocity.shape == noisy_actions.shape
    assert torch.isfinite(velocity).all()
    assert torch.equal(velocity[0, 2:], torch.zeros_like(velocity[0, 2:]))


def test_worldflow_action_expert_attention_is_bidirectional():
    torch.manual_seed(18)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_action_expert_layers=1,
    )
    branch = _make_tiny_worldflow_branch(cfg).eval()
    branch.point_action_adapter = _IdentityPointAction()
    point_cloud = torch.randn(1, 12, 6)
    point_cloud[..., 3:6] = torch.rand(1, 12, 3) * 255.0
    lang_tokens = torch.randint(0, 64, (1, 5))
    lang_masks = torch.ones(1, 5, dtype=torch.bool)
    noisy_actions = torch.randn(1, cfg.chunk_size, 9)
    changed_actions = noisy_actions.clone()
    changed_actions[:, -1] += 1.0
    time = torch.rand(1)

    first = branch(point_cloud, lang_tokens, lang_masks, noisy_actions, time)
    changed = branch(point_cloud, lang_tokens, lang_masks, changed_actions, time)

    assert not torch.allclose(first[:, 0], changed[:, 0])


def test_worldflow_auxiliary_uses_foreground_only_and_backpropagates_bridge():
    torch.manual_seed(19)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_action_expert_layers=1,
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
    model.last_body_pose9_prediction = matrix_to_pose9(
        torch.linalg.inv(current).unsqueeze(1) @ target
    ).detach().requires_grad_(True)
    batch = {
        "worldflow.current_ee_pose": matrix_to_pose9(current),
        "worldflow.ee_poses": matrix_to_pose9(target),
        "worldflow.step_is_pad": torch.zeros(bsize, chunk_size, dtype=torch.bool),
    }
    lang_tokens = torch.randint(0, 64, (bsize, 5))
    lang_masks = torch.ones(bsize, 5, dtype=torch.bool)

    output = model.compute_worldflow_aux_loss(batch, lang_tokens, lang_masks)
    output["per_sample_loss"].mean().backward()

    assert output["pred_spatial"].shape == (bsize, chunk_size, 4, 4)
    assert model.last_body_pose9_prediction.grad is not None
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


def test_worldflow_branch_is_absent_from_policy_inference_call_path():
    assert "worldflow_branch(" in inspect.getsource(VLAFlowMatching.compute_worldflow_aux_loss)
    assert "worldflow_branch(" not in inspect.getsource(VLAFlowMatching.sample_actions)
    assert "worldflow_branch(" not in inspect.getsource(VLAFlowMatching.denoise_step)
