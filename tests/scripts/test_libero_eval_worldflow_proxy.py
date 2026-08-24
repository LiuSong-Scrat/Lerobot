from __future__ import annotations

import os
import queue
from types import SimpleNamespace

import pytest

os.environ.setdefault("SONG_LIBERO_ENV_WORKER", "1")

from benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_eval import (  # noqa: E402
    ProcessInferenceProxy,
    evaluation_protocol_for_config,
    inspect_policy_camera_alignment,
    policy_requires_rgb,
    reconcile_eval_camera_views_with_loaded_policy,
    validate_control_frequency,
)


def test_process_proxy_preserves_molmo_rgb_and_worldflow_contracts():
    proxy = ProcessInferenceProxy(
        worker_id=0,
        request_queue=queue.Queue(),
        response_queue=queue.Queue(),
        vla_adapter_enable=False,
        image_feature_keys=["observation.images.agentview"],
        requires_rgb=True,
        worldflow_enable=True,
        worldflow_reference_frame="robot_base",
    )

    assert proxy.policy.config.vla_adapter_enable is False
    assert proxy.policy.config.requires_rgb is True
    assert list(proxy.policy.config.image_features) == ["observation.images.agentview"]
    assert proxy.policy.config.worldflow_enable is True
    assert proxy.policy.config.worldflow_reference_frame == "robot_base"


def test_molmo_rgb_requirement_is_independent_of_vla_adapter_flag():
    policy_config = SimpleNamespace(
        vla_adapter_enable=False,
        requires_rgb=True,
        rgb_camera_views="agentview",
        image_features={"observation.images.agentview": None},
    )
    infer = SimpleNamespace(
        camera_views=("agentview",),
        policy=SimpleNamespace(config=policy_config),
    )
    cfg = {
        "pointcloud_camera_names": ["agentview"],
        "image_cameras": ["agentview"],
    }

    alignment = inspect_policy_camera_alignment(infer, cfg)

    assert policy_requires_rgb(policy_config) is True
    assert alignment["checkpoint_rgb"] == ("agentview",)
    assert alignment["rgb_matches_checkpoint"] is True

    reconciled = reconcile_eval_camera_views_with_loaded_policy(infer, cfg)
    assert cfg["image_cameras"] == ["agentview"]
    assert cfg["camera_names"] == ["agentview"]
    assert reconciled["rgb_matches_checkpoint"] is True


def test_nonstandard_positive_control_frequency_is_allowed_but_not_comparable():
    assert validate_control_frequency(5) == 5.0

    protocol = evaluation_protocol_for_config({"control": {"control_freq": 5}})

    assert protocol["name"] == "nonstandard_control_frequency_rollout"
    assert protocol["control_freq"] == 5.0
    assert protocol["official_control_freq"] == 20.0
    assert protocol["benchmark_comparable"] is False


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_control_frequency_must_be_finite_and_positive(invalid):
    with pytest.raises(ValueError, match="finite positive frequency"):
        validate_control_frequency(invalid)
