import os
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.song_real_libero.scripts.libero_setting import libero_pointcloud_fusion as light
from lerobot.policies.smolvla import song_pointseg as policy


@pytest.mark.parametrize(
    "value",
    [None, "agentview", "agentview,robot0_eye_in_hand", ["robot0_eye_in_hand"]],
)
def test_lightweight_camera_view_parser_matches_policy(value) -> None:
    assert light.parse_camera_views(value) == policy.parse_camera_views(value)


@pytest.mark.parametrize(
    "fusion",
    [
        "legacy_budget",
        "uniform_union",
        "fps",
        "voxel_fps",
        "voxel_cover_fps",
        "novelty_union",
        "multiscale_novelty_union",
        "consensus_multiscale_novelty_union",
        "transport_novelty_union",
        "full_union",
        "primary_residual",
    ],
)
def test_lightweight_composition_matches_policy(fusion: str) -> None:
    first = np.arange(72, dtype=np.float32).reshape(12, 6)
    second = first + 1_000
    kwargs = {"gripper_points": 2, "seed": 17, "fusion": fusion}
    expected = policy.compose_point_cloud_views([first, second], **kwargs)
    actual = light.compose_point_cloud_views([first, second], **kwargs)
    np.testing.assert_array_equal(actual, expected)


def test_lightweight_weighted_composition_matches_policy() -> None:
    first = np.arange(72, dtype=np.float32).reshape(12, 6)
    second = first + 1_000
    kwargs = {
        "gripper_points": 2,
        "seed": 23,
        "view_weights": (3.0, 1.0),
        "fusion": "legacy_budget",
    }
    expected = policy.compose_point_cloud_views([first, second], **kwargs)
    actual = light.compose_point_cloud_views([first, second], **kwargs)
    np.testing.assert_array_equal(actual, expected)


def test_libero_pointcloud_utils_does_not_import_torch() -> None:
    command = (
        "from benchmarks.song_real_libero.scripts.libero_setting "
        "import libero_pointcloud_utils; "
        "import sys; "
        "assert 'torch' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(os.getcwd())
    subprocess.run([sys.executable, "-c", command], env=environment, check=True)
