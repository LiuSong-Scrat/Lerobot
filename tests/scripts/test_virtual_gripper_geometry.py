from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/song_real_libero/scripts"
)
sys.path.insert(0, str(SCRIPTS))

from libero_setting.libero_pointcloud_utils import (  # noqa: E402
    action_pose9_to_libero,
    add_local_gripper_cloud_to_point_cloud,
    normalize_gripper_widths,
)


def test_metre_widths_are_not_normalized() -> None:
    widths = normalize_gripper_widths(
        np.array([0.01, 0.03, 0.06], dtype=np.float32),
        already_normalized=False,
        max_physical_width=0.08,
    )
    np.testing.assert_array_equal(widths, np.array([0.01, 0.03, 0.06], dtype=np.float32))


def test_libero_tail_is_canonical_and_pure_red() -> None:
    scene = np.zeros((10_000, 6), dtype=np.float32)
    cloud = add_local_gripper_cloud_to_point_cloud(
        scene,
        0.04,
        total_points=10_000,
        gripper_points=500,
        gripper_template="rh20t_v3",
        gripper_max_width=0.08,
        seed=23,
        drop_strategy="tail",
    )
    tail = cloud[-500:]
    np.testing.assert_array_equal(
        tail[:, 3:], np.broadcast_to(np.array([255.0, 0.0, 0.0]), (500, 3))
    )
    # RH20T v3 surface counts for 500 points are fingers 129/129,
    # 100 mm palm 159, and handle 83.
    left, right, palm = tail[:129, :3], tail[129:258, :3], tail[258:417, :3]
    np.testing.assert_allclose(right[:, 1].min() - left[:, 1].max(), 0.04, atol=1e-7)
    np.testing.assert_allclose(np.ptp(palm[:, 1]), 0.10, atol=1e-7)


def test_eval_threshold_uses_physical_metres() -> None:
    identity_pose9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    open_action = np.r_[identity_pose9, np.float32(0.05)]
    close_action = np.r_[identity_pose9, np.float32(0.03)]
    assert action_pose9_to_libero(open_action, 0.05, 0.5, 0.04)[-1] == -1.0
    assert action_pose9_to_libero(close_action, 0.05, 0.5, 0.04)[-1] == 1.0
