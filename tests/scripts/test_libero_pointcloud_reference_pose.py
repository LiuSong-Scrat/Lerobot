from __future__ import annotations

import numpy as np

from benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_utils import (
    eef_pose9_world_to_reference_np,
    matrix_to_pose9_np,
    pose9_to_homo_np,
)


def _transform(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def test_eef_world_pose_is_expressed_in_fixed_reference_and_preserves_gripper():
    reference_to_world = _transform(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 2.0, 3.0],
    )
    eef_to_world = _transform(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [4.0, 5.0, 6.0],
    )
    gripper_width = np.asarray([0.037], dtype=np.float32)
    eef_pose_world = np.concatenate([matrix_to_pose9_np(eef_to_world), gripper_width])

    converted = eef_pose9_world_to_reference_np(eef_pose_world, reference_to_world)

    expected = np.linalg.inv(reference_to_world) @ eef_to_world
    np.testing.assert_allclose(pose9_to_homo_np(converted[:9]), expected, atol=1e-6)
    np.testing.assert_array_equal(converted[9:], gripper_width)


def test_eef_world_pose_conversion_supports_batches():
    reference_to_world = np.eye(4, dtype=np.float32)
    first = np.concatenate([matrix_to_pose9_np(np.eye(4, dtype=np.float32)), [0.01]])
    translated = np.eye(4, dtype=np.float32)
    translated[:3, 3] = [0.2, -0.3, 0.4]
    second = np.concatenate([matrix_to_pose9_np(translated), [0.02]])

    converted = eef_pose9_world_to_reference_np(
        np.stack([first, second]).astype(np.float32),
        reference_to_world,
    )

    assert converted.shape == (2, 10)
    np.testing.assert_allclose(converted, np.stack([first, second]), atol=1e-6)
