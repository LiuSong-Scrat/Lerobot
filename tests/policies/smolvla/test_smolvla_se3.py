#!/usr/bin/env python

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    SE3CanonicalTrajectoryHead,
    _transform_point_cloud_xyzrgb,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)
from lerobot.policies.smolvla.song_pointseg import pose9_to_matrix


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


def test_se3_canonical_trajectory_head_is_equivariant_for_nondegenerate_cloud():
    torch.manual_seed(13)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_backbone_type="mlp",
        worldflow_feature_dim=24,
        worldflow_grid_size=0.01,
    )
    head = SE3CanonicalTrajectoryHead(cfg, language_dim=12, feature_dim=24)
    head.eval()

    xyz = torch.tensor(
        [
            [-0.3, -0.1, 0.0],
            [0.4, -0.2, 0.1],
            [0.2, 0.5, -0.2],
            [-0.1, 0.3, 0.4],
            [0.6, 0.2, 0.3],
            [-0.5, 0.4, -0.3],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    rgb = torch.linspace(0, 255, xyz.shape[1] * 3, dtype=torch.float32).reshape(1, xyz.shape[1], 3)
    point_cloud = torch.cat([xyz, rgb], dim=-1)
    lang_emb = torch.randn(1, 5, 12)
    lang_mask = torch.ones(1, 5, dtype=torch.bool)
    transform = se3_exp(torch.tensor([[0.2, -0.1, 0.3, 0.15, -0.25, 0.1]], dtype=torch.float32))
    point_cloud_aug = _transform_point_cloud_xyzrgb(point_cloud, transform)

    pred = pose9_to_matrix(head(point_cloud, lang_emb, lang_mask))
    pred_aug = pose9_to_matrix(head(point_cloud_aug, lang_emb, lang_mask))
    expected = transform.unsqueeze(1) @ pred

    assert torch.allclose(pred_aug[..., :3, 3], expected[..., :3, 3], atol=1e-4, rtol=1e-4)
    assert torch.allclose(pred_aug[..., :3, :3], expected[..., :3, :3], atol=1e-4, rtol=1e-4)
