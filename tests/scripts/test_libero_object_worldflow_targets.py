from __future__ import annotations

import numpy as np

from benchmarks.song_real_libero.scripts.libero_setting.libero_hdf5_to_dataset import (
    build_centered_object_motion_targets,
    homo_to_pose9,
    pose9_to_homo_np,
)


def _pose(translation, rotation=None):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    return transform


def test_centered_object_motion_selects_nonrobot_body_and_has_no_origin_lever_arm():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, None], 3, axis=0)
    poses = np.repeat(poses, 3, axis=1)
    # Robot body moves much more but must be excluded.
    poses[:, 1, :3, 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    # Object rotates 90 degrees around its own center far from the base and
    # translates by only 1 cm. A global-origin spatial transform would contain
    # a large artificial translation; the centered descriptor must not.
    poses[:, 2, :3, 3] = [[2.0, 3.0, 0.0], [2.01, 3.0, 0.0], [2.02, 3.0, 0.0]]
    poses[1:, 2, :3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    targets, body_name, score = build_centered_object_motion_targets(
        poses,
        ["world", "robot0_link", "target_object"],
        np.asarray([True, True, False]),
        horizon=2,
    )

    assert body_name == "target_object"
    assert score > 0
    first_step = pose9_to_homo_np(targets[0, 0])
    np.testing.assert_allclose(first_step[:3, 3], [0.01, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(first_step[:3, :3], poses[1, 2, :3, :3], atol=1e-6)


def test_centered_object_motion_horizon_is_next_state_cumulative_motion():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, None], 4, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    poses[:, 1, 0, 3] = [0.0, 0.1, 0.3, 0.6]

    targets, _, _ = build_centered_object_motion_targets(
        poses,
        ["world", "object"],
        np.asarray([True, False]),
        horizon=3,
    )

    np.testing.assert_allclose(targets[0, :, 0], [0.1, 0.3, 0.6], atol=1e-6)
    identity_pose9 = homo_to_pose9(np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(targets[-1], np.repeat(identity_pose9[None], 3, axis=0))
