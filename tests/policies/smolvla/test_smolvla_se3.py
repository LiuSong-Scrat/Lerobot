#!/usr/bin/env python

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    DenseRigidObjectFlowHead,
    se3_exp,
    se3_log,
    select_objectflow_points,
    so3_exp,
    so3_log,
    weighted_kabsch_transform,
)
from lerobot.policies.smolvla.song_pointseg import invert_transform, matrix_to_pose9, pose9_to_matrix


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
    source = torch.randn(2, 32, 3)
    transform = se3_exp(torch.randn(2, 4, 6) * 0.15)
    target = torch.einsum("btij,bnj->btni", transform[..., :3, :3], source)
    target = target + transform[..., :3, 3].unsqueeze(2)
    weights = torch.rand(2, 32)
    weights[:, :8] = 0.0

    fitted, valid = weighted_kabsch_transform(source, target, weights)

    assert valid.all()
    assert torch.allclose(fitted[..., :3, 3], transform[..., :3, 3], atol=2e-5, rtol=2e-5)
    assert torch.allclose(fitted[..., :3, :3], transform[..., :3, :3], atol=2e-5, rtol=2e-5)


def test_dense_objectflow_head_uses_automatic_roles_and_starts_from_zero_flow():
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=24,
    )
    head = DenseRigidObjectFlowHead(cfg, language_dim=12)
    point_cloud = torch.randn(2, 20, 6)
    point_cloud[..., 3:] = torch.rand(2, 20, 3) * 255
    role_scores = torch.softmax(torch.randn(2, 20, 3), dim=-1)
    lang_emb = torch.randn(2, 5, 12)
    lang_mask = torch.ones(2, 5, dtype=torch.bool)
    point_is_pad = torch.zeros(2, 20, dtype=torch.bool)

    flow = head(point_cloud, role_scores, lang_emb, lang_mask, point_is_pad)

    assert flow.shape == (2, 4, 20, 3)
    assert torch.count_nonzero(flow) == 0


def test_objectflow_point_cap_balances_condition_and_target_scores():
    point_cloud = torch.arange(1 * 12 * 6, dtype=torch.float32).reshape(1, 12, 6)
    roles = torch.zeros(1, 12, 3)
    roles[0, :6, 1] = torch.linspace(1.0, 0.5, 6)
    roles[0, 6:, 2] = torch.linspace(1.0, 0.5, 6)
    point_is_pad = torch.zeros(1, 12, dtype=torch.bool)

    selected_pc, selected_roles, selected_pad = select_objectflow_points(
        point_cloud, roles, point_is_pad, max_points=6
    )

    assert selected_pc.shape == (1, 6, 6)
    assert selected_roles[0, :3, 1].gt(0).all()
    assert selected_roles[0, 3:, 2].gt(0).all()
    assert not selected_pad.any()


def test_world_ego_bridge_has_expected_invariance_and_equivariance():
    torch.manual_seed(17)
    world_from_ego = se3_exp(torch.randn(3, 6) * 0.2)
    body = se3_exp(torch.randn(3, 5, 6) * 0.15)
    spatial = world_from_ego[:, None] @ body @ invert_transform(world_from_ego)[:, None]
    bridged = invert_transform(world_from_ego)[:, None] @ spatial @ world_from_ego[:, None]
    assert torch.allclose(bridged, body, atol=3e-5, rtol=3e-5)

    world_change = se3_exp(torch.randn(3, 6) * 0.2)
    changed_world_from_ego = world_change @ world_from_ego
    changed_spatial = world_change[:, None] @ spatial @ invert_transform(world_change)[:, None]
    changed_bridge = (
        invert_transform(changed_world_from_ego)[:, None]
        @ changed_spatial
        @ changed_world_from_ego[:, None]
    )
    assert torch.allclose(changed_bridge, body, atol=3e-5, rtol=3e-5)

    # A constant grasp-frame offset conjugates Body flow but cancels from Spatial flow.
    grasp_offset = se3_exp(torch.randn(3, 6) * 0.1)
    changed_body = invert_transform(grasp_offset)[:, None] @ body @ grasp_offset[:, None]
    grasp_world_from_ego = world_from_ego @ grasp_offset
    grasp_spatial = grasp_world_from_ego[:, None] @ changed_body @ invert_transform(
        grasp_world_from_ego
    )[:, None]
    assert torch.allclose(grasp_spatial, spatial, atol=3e-5, rtol=3e-5)

    # Keep pose9 conversion covered for the bridge output used by the model.
    assert torch.allclose(pose9_to_matrix(matrix_to_pose9(changed_bridge)), changed_bridge, atol=3e-5)
