#!/usr/bin/env python

import pytest
import torch
from torch import nn

from lerobot.configs.types import NormalizationMode
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    WorldSE3TrajectoryHead,
    _transform_point_cloud_xyzrgb,
    se3_exp,
    se3_log,
    select_worldflow_evidence_points,
    so3_exp,
    so3_log,
    worldflow_evidence_weights,
)
from lerobot.policies.smolvla.song_pointseg import matrix_to_pose9


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
    target = se3_exp(torch.randn(8, 6) * 0.2)
    current = se3_exp(torch.randn(8, 6) * 0.2)
    body = torch.linalg.inv(current) @ target
    assert torch.allclose(current @ body, target, atol=3e-4, rtol=3e-4)


def test_cache_v7_evidence_maps_to_transport_and_interaction():
    evidence = torch.tensor(
        [
            [
                [0.8, 0.0, 0.0],
                [0.0, 0.3, 0.5],
                [0.4, 0.2, 0.1],
            ]
        ]
    )
    point_is_pad = torch.tensor([[False, False, True]])

    transport, interaction = worldflow_evidence_weights(evidence, point_is_pad)

    assert torch.allclose(transport, torch.tensor([[0.8, 0.0, 0.0]]))
    # Probabilistic OR: 1 - (1 - approach) * (1 - near_contact).
    assert torch.allclose(interaction, torch.tensor([[0.0, 0.65, 0.0]]))


def test_worldflow_point_cap_balances_transport_and_interaction_evidence():
    point_cloud = torch.zeros(1, 10, 6)
    point_cloud[0, :, 0] = torch.arange(10)
    point_cloud[..., 3:6] = 127.0
    evidence = torch.zeros(1, 10, 3)
    evidence[0, :5, 0] = torch.linspace(1.0, 0.6, 5)
    evidence[0, 5:, 1] = torch.linspace(1.0, 0.6, 5)

    selected = select_worldflow_evidence_points(point_cloud, evidence, max_points=6)

    assert selected["transport_points"].shape == (1, 3, 7)
    assert selected["interaction_points"].shape == (1, 3, 7)
    assert not bool(selected["transport_is_pad"].any())
    assert not bool(selected["interaction_is_pad"].any())
    assert bool((selected["transport_points"][0, :, 0] < 5).all())
    assert bool((selected["interaction_points"][0, :, 0] >= 5).all())
    assert bool((selected["transport_points"][..., 6] > 0).all())
    assert bool((selected["interaction_points"][..., 6] > 0).all())


def test_worldflow_selection_masks_zero_evidence_instead_of_using_arbitrary_geometry():
    point_cloud = torch.randn(2, 12, 6)
    evidence = torch.zeros(2, 12, 3)
    evidence[1, :4, 0] = 0.8

    selected = select_worldflow_evidence_points(point_cloud, evidence, max_points=8)

    assert bool(selected["transport_is_pad"][0].all())
    assert bool(selected["interaction_is_pad"].all())
    assert int((~selected["transport_is_pad"][1]).sum()) == 4


def test_se3_augmentation_preserves_rgb_and_soft_evidence_channel():
    point_cloud = torch.randn(2, 7, 7)
    point_cloud[..., 3:6] = torch.rand(2, 7, 3) * 255.0
    point_cloud[..., 6] = torch.rand(2, 7) * 255.0
    transform = se3_exp(torch.randn(2, 6) * 0.1)

    transformed = _transform_point_cloud_xyzrgb(point_cloud, transform)

    assert transformed.shape == point_cloud.shape
    assert torch.equal(transformed[..., 3:], point_cloud[..., 3:])


class _MeanEvidenceEncoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, points: torch.Tensor, point_is_pad: torch.Tensor) -> torch.Tensor:
        valid = (~point_is_pad).unsqueeze(-1).to(dtype=points.dtype)
        mean = (points[..., :1] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return mean.expand(-1, self.feature_dim)


def _test_head() -> tuple[WorldSE3TrajectoryHead, SmolVLAConfig]:
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=24,
        worldflow_grid_size=0.01,
    )
    head = WorldSE3TrajectoryHead(cfg, language_dim=12)
    # Keep unit tests independent of optional pointops kernels.
    head.transport_encoder = _MeanEvidenceEncoder(cfg.worldflow_feature_dim)
    head.interaction_encoder = _MeanEvidenceEncoder(cfg.worldflow_feature_dim)
    return head, cfg


def test_direct_world_se3_head_starts_from_identity_trajectory():
    head, cfg = _test_head()
    head.eval()
    selected = {
        "transport_points": torch.randn(2, 8, 7),
        "transport_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        "interaction_points": torch.randn(2, 8, 7),
        "interaction_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }
    lang_emb = torch.randn(2, 5, 12)
    lang_mask = torch.ones(2, 5, dtype=torch.bool)

    body = head(selected, lang_emb, lang_mask)
    identity = torch.eye(4).expand_as(body)

    assert body.shape == (2, cfg.chunk_size, 4, 4)
    assert torch.allclose(body, identity)


def test_direct_world_se3_head_handles_missing_interaction_evidence():
    head, cfg = _test_head()
    selected = {
        "transport_points": torch.randn(2, 8, 7),
        "transport_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        "interaction_points": torch.randn(2, 8, 7),
        "interaction_is_pad": torch.ones(2, 8, dtype=torch.bool),
    }
    body = head(
        selected,
        torch.randn(2, 5, 12),
        torch.ones(2, 5, dtype=torch.bool),
    )

    assert body.shape == (2, cfg.chunk_size, 4, 4)
    assert torch.isfinite(body).all()


def test_world_ego_conjugacy_recovers_body_transform():
    torch.manual_seed(23)
    current = se3_exp(torch.randn(3, 6) * 0.2)
    body = se3_exp(torch.randn(3, 6) * 0.2)
    target = current @ body
    spatial = target @ torch.linalg.inv(current)
    recovered_body = torch.linalg.inv(current) @ spatial @ current

    assert torch.allclose(recovered_body, body, atol=3e-4, rtol=3e-4)


def test_worldflow_auxiliary_loss_updates_head_and_confidence_gated_action_bridge():
    head, cfg = _test_head()
    cfg.worldflow_enable = True
    cfg.worldflow_max_points = 16
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_head = head
    model.last_worldflow_metrics = {}

    point_cloud = torch.randn(2, 16, 6)
    point_cloud[..., 3:6] = torch.rand(2, 16, 3) * 255.0
    evidence = torch.zeros(2, 16, 3)
    evidence[:, :8, 0] = 0.9
    evidence[:, 8:, 1] = 0.8
    model.last_worldflow_payload = {
        "point_cloud_ego": point_cloud,
        "role_scores": evidence,
        "point_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }

    current = se3_exp(torch.randn(2, 6) * 0.1)
    body_gt = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.05)
    target = current.unsqueeze(1) @ body_gt
    action_body_pose9 = matrix_to_pose9(body_gt).detach().clone().requires_grad_(True)
    model.last_body_pose9_prediction = action_body_pose9
    cached_lang = torch.randn(2, 5, 12, requires_grad=True)
    batch = {
        "worldflow.current_ee_pose": matrix_to_pose9(current),
        "worldflow.ee_poses": matrix_to_pose9(target),
        "worldflow.step_is_pad": torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
    }

    result = model.compute_worldflow_aux_loss(
        batch,
        lang_tokens=torch.zeros(2, 5, dtype=torch.long),
        lang_masks=torch.ones(2, 5, dtype=torch.bool),
        actions_is_pad=torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
        cached_lang_emb=cached_lang,
    )
    assert result is not None
    assert result["pred_body"].shape == (2, cfg.chunk_size, 4, 4)
    assert result["per_sample_loss"].shape == (2,)
    assert torch.isfinite(result["per_sample_loss"]).all()

    result["per_sample_loss"].mean().backward()
    final = head.trajectory_decoder[-1]
    assert isinstance(final, nn.Linear)
    assert final.weight.grad is not None
    assert bool((final.weight.grad.abs() > 0).any())
    assert action_body_pose9.grad is not None
    assert cached_lang.grad is None
    assert "worldflow_bridge_confidence" in model.last_worldflow_metrics


def test_worldflow_is_disabled_by_default_and_requires_metric_actions():
    assert SmolVLAConfig().worldflow_enable is False
    with pytest.raises(ValueError, match="ACTION normalization"):
        SmolVLAConfig(
            worldflow_enable=True,
            normalization_mapping={"ACTION": NormalizationMode.MEAN_STD},
        )
    with pytest.raises(ValueError, match="at least 3"):
        SmolVLAConfig(worldflow_enable=True, worldflow_min_transport_points=2)
