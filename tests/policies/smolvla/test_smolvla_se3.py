#!/usr/bin/env python

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    DenseRigidObjectFlowHead,
    select_objectflow_points,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
    weighted_kabsch_transform,
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


def test_weighted_kabsch_recovers_spatial_transform():
    torch.manual_seed(13)
    source = torch.randn(2, 4, 12, 3)
    transform = se3_exp(torch.randn(2, 4, 6) * 0.15)
    target = (
        source @ transform[..., :3, :3].transpose(-1, -2)
        + transform[..., :3, 3].unsqueeze(-2)
    )
    weights = torch.ones(2, 4, 12)

    recovered = weighted_kabsch_transform(source, target, weights)

    assert torch.allclose(recovered[..., :3, 3], transform[..., :3, 3], atol=2e-4, rtol=2e-4)
    assert torch.allclose(recovered[..., :3, :3], transform[..., :3, :3], atol=2e-4, rtol=2e-4)


def test_dense_objectflow_head_uses_automatic_roles_and_starts_from_zero_flow():
    torch.manual_seed(17)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_backbone_type="mlp",
        worldflow_feature_dim=24,
        worldflow_grid_size=0.01,
    )
    head = DenseRigidObjectFlowHead(cfg, language_dim=12)
    head.eval()

    point_cloud = torch.randn(2, 8, 6)
    point_cloud[..., 3:6] = torch.rand(2, 8, 3) * 255.0
    role_scores = torch.zeros(2, 8, 3)
    role_scores[:, :3, 1] = 1.0
    role_scores[:, 3:6, 2] = 1.0
    lang_emb = torch.randn(2, 5, 12)
    lang_mask = torch.ones(2, 5, dtype=torch.bool)

    flow = head(point_cloud, role_scores, lang_emb, lang_mask)

    assert flow.shape == (2, cfg.chunk_size, 8, 3)
    assert torch.allclose(flow, torch.zeros_like(flow))


def test_objectflow_point_cap_balances_condition_and_target_scores():
    torch.manual_seed(19)
    point_cloud = torch.randn(1, 10, 6)
    role_scores = torch.zeros(1, 10, 3)
    role_scores[0, :5, 1] = torch.linspace(1.0, 0.6, 5)
    role_scores[0, 5:, 2] = torch.linspace(1.0, 0.6, 5)

    selected_pc, selected_roles, selected_is_pad = select_objectflow_points(
        point_cloud,
        role_scores,
        max_points=6,
    )

    assert selected_pc.shape == (1, 6, 6)
    assert selected_roles.shape == (1, 6, 3)
    assert not bool(selected_is_pad.any())
    assert int((selected_roles[..., 1] > 0).sum().item()) >= 2
    assert int((selected_roles[..., 2] > 0).sum().item()) >= 2


def test_world_ego_bridge_formula_matches_body_transform():
    torch.manual_seed(23)
    current = se3_exp(torch.randn(3, 6) * 0.2)
    body = se3_exp(torch.randn(3, 6) * 0.2)
    target = current @ body
    spatial = target @ torch.linalg.inv(current)
    recovered_body = torch.linalg.inv(current) @ spatial @ current

    assert torch.allclose(recovered_body, body, atol=3e-4, rtol=3e-4)
