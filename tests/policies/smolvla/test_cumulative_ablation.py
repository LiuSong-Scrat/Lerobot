from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    PointActionSelfAttention,
    SmolVLAPolicy,
    VLAFlowMatching,
)


@pytest.mark.parametrize(
    ("name", "gates"),
    [
        ("smolvla_src", (False, False, False)),
        ("smolvla_pointcloud", (True, False, False)),
        ("smolvla_pointcloud_effseg", (True, True, False)),
        ("smolvla_pointcloud_effseg_pointaction", (True, True, True)),
    ],
)
def test_cumulative_ablation_presets(name, gates):
    config = SmolVLAConfig(ablation_variant=name)

    assert (config.pointcloud_enable, config.pointseg_enable, config.point_action_fusion_enable) == gates
    assert config.pointcloud_input_points == 10_000
    assert config.vla_adapter_enable
    assert config.vla_adapter_freeze_vlm
    assert config.encode_robot_state
    assert config.train_state_proj
    assert not config.worldflow_enable


@pytest.mark.parametrize("name", ["smolvla_pointcloud", "smolvla_pointcloud_effseg"])
def test_pre_pointaction_ablations_do_not_enable_point_action_adapter(name):
    config = SmolVLAConfig(ablation_variant=name)

    assert not config.point_action_fusion_enable


class _StubSmolVLMWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8, vocab_size=32))
        self.expert_hidden_size = 8
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(fake_image_token_id=1, global_image_token_id=2)
        )
        self.lm_expert = nn.Linear(1, 1)


@pytest.mark.parametrize(
    ("name", "has_point_action"),
    [
        ("smolvla_src", False),
        ("smolvla_pointcloud", False),
        ("smolvla_pointcloud_effseg", False),
        ("smolvla_pointcloud_effseg_pointaction", True),
    ],
)
def test_cumulative_ablation_constructs_exact_state_and_point_action_modules(name, has_point_action):
    config = SmolVLAConfig(ablation_variant=name, device="cpu")
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel",
        return_value=_StubSmolVLMWithExpert(),
    ):
        model = VLAFlowMatching(config)

    assert model.state_proj.weight.requires_grad
    assert isinstance(model.point_action_fusion, PointActionSelfAttention) is has_point_action


def _minimal_policy(*, pointcloud_enable: bool, pointcloud_input_points: int) -> SmolVLAPolicy:
    policy = object.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        pointcloud_enable=pointcloud_enable,
        pointcloud_input_points=pointcloud_input_points,
    )
    policy.model = SimpleNamespace(pointseg_conditioner=object() if pointcloud_enable else None)
    return policy


def test_2d_ablation_does_not_require_point_cloud_batch_key():
    policy = _minimal_policy(pointcloud_enable=False, pointcloud_input_points=10_000)
    assert policy.prepare_point_clouds({}) == ([], [])


def test_point_ablation_normalizes_cloud_and_effseg_targets_to_exact_count():
    policy = _minimal_policy(pointcloud_enable=True, pointcloud_input_points=10)
    point_cloud = torch.arange(2 * 7 * 6, dtype=torch.float32).reshape(2, 7, 6)
    point_is_pad = torch.tensor(
        [[False, False, False, False, False, True, True], [False] * 7]
    )
    labels = torch.arange(14).reshape(2, 1, 7)

    point_clouds, masks = policy.prepare_point_clouds(
        {
            "observation.point_cloud": point_cloud,
            "observation.point_cloud_is_pad": point_is_pad,
            "pointseg.labels": labels,
        }
    )

    payload = point_clouds[0]
    assert payload["point_cloud"].shape == (2, 10, 6)
    assert payload["point_is_pad"].shape == (2, 10)
    assert not payload["point_is_pad"].any()
    assert payload["pointseg.labels"].shape == (2, 10)
    assert payload["pointseg.labels"][0].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert masks[0].tolist() == [True, True]
