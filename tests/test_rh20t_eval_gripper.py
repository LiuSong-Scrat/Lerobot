import numpy as np
from scipy.spatial.transform import Rotation as R

from benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_utils import (
    add_world_gripper_cloud_to_point_cloud,
    create_gripper_cloud_rgb,
    gripper_width_m_from_scalar,
    normalize_gripper_widths,
    previous_eef_pose9_to_virtual_eef_pose9,
    pose9_to_homo_np,
    reference_point_cloud_to_current_eff,
)
from benchmarks.song_real_libero.scripts.virtual_gripper_geometry import (
    GRIPPER_COLOR_RGB,
    T_PREVIOUS_EEF_VIRTUAL_EEF,
    sample_gripper_width,
)


def test_eval_gripper_clamps_finite_joint_limit_noise() -> None:
    assert gripper_width_m_from_scalar(-0.0005021113902330399, 0.08) == 0.0
    assert gripper_width_m_from_scalar(0.09, 0.08) == 0.08


def test_eval_gripper_uses_physical_width_and_canonical_geometry() -> None:
    widths = normalize_gripper_widths(
        np.array([0.0, 0.04, 0.10], dtype=np.float32),
        max_physical_width=0.08,
    )
    np.testing.assert_allclose(widths, [0.0, 0.04, 0.08])

    seed = 23
    expected_xyz = sample_gripper_width(0.04, 500, np.random.default_rng(seed))
    actual = create_gripper_cloud_rgb(
        0.04,
        np.zeros(6, dtype=np.float32),
        500,
        np.random.default_rng(seed),
        gripper_len=0.06,
        gripper_template="rh20t_v3",
        gripper_max_width=0.08,
    )
    np.testing.assert_array_equal(actual[:, :3], expected_xyz.astype(np.float32))
    np.testing.assert_array_equal(actual[:, 3:], np.broadcast_to(GRIPPER_COLOR_RGB, (500, 3)))


def test_world_pose_round_trip_preserves_canonical_eef_tail_and_legacy_alias() -> None:
    seed = 47
    width_m = 0.03
    rotation = R.from_euler("zyx", [0.3, -0.2, 0.4]).as_matrix().astype(np.float32)
    pose9_gripper = np.concatenate(
        (
            np.array([0.31, -0.17, 0.42], dtype=np.float32),
            rotation[:, 0],
            rotation[:, 1],
            np.array([width_m], dtype=np.float32),
        )
    )
    merged_eff = add_world_gripper_cloud_to_point_cloud(
        np.zeros((50_000, 6), dtype=np.float32),
        pose9_gripper,
        width_m,
        total_points=50_000,
        gripper_points=500,
        gripper_template="reap",
        gripper_max_width=0.08,
        seed=seed,
        drop_strategy="tail",
    )
    expected_local = sample_gripper_width(width_m, 500, np.random.default_rng(seed))
    np.testing.assert_allclose(merged_eff[-500:, :3], expected_local, atol=1e-6, rtol=0.0)
    np.testing.assert_array_equal(
        merged_eff[-500:, 3:],
        np.broadcast_to(GRIPPER_COLOR_RGB, (500, 3)),
    )


def test_previous_controller_eef_is_shifted_to_dataset_virtual_eef() -> None:
    rotation = R.from_euler("zyx", [0.35, -0.27, 0.18]).as_matrix().astype(np.float32)
    previous = np.concatenate(
        (
            np.array([0.42, -0.11, 0.73], dtype=np.float32),
            rotation[:, 0],
            rotation[:, 1],
            np.array([0.047], dtype=np.float32),
        )
    )

    virtual = previous_eef_pose9_to_virtual_eef_pose9(previous)
    expected = pose9_to_homo_np(previous[:9]) @ T_PREVIOUS_EEF_VIRTUAL_EEF

    np.testing.assert_allclose(pose9_to_homo_np(virtual[:9]), expected, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(virtual[-1], previous[-1], atol=0.0, rtol=0.0)


def test_virtual_eef_controller_mapping_round_trip() -> None:
    """The evaluator must execute virtual targets through the physical EEF."""

    previous_world = np.eye(4, dtype=np.float32)
    previous_world[:3, :3] = R.from_euler("zyx", [-0.2, 0.3, 0.1]).as_matrix()
    previous_world[:3, 3] = [0.31, -0.24, 0.81]
    virtual_world = previous_world @ T_PREVIOUS_EEF_VIRTUAL_EEF
    virtual_to_previous = np.linalg.inv(virtual_world) @ previous_world

    relative_target = np.eye(4, dtype=np.float32)
    relative_target[:3, :3] = R.from_euler("zyx", [0.05, -0.04, 0.03]).as_matrix()
    relative_target[:3, 3] = [0.02, -0.01, 0.015]
    target_virtual_world = virtual_world @ relative_target
    target_previous_world = target_virtual_world @ virtual_to_previous

    np.testing.assert_allclose(
        target_previous_world @ T_PREVIOUS_EEF_VIRTUAL_EEF,
        target_virtual_world,
        atol=1e-6,
        rtol=0.0,
    )


def test_scene_reexpression_matches_libero_dataset_migration_formula() -> None:
    rotation = R.from_euler("zyx", [0.22, 0.13, -0.31]).as_matrix().astype(np.float32)
    previous = np.concatenate(
        (
            np.array([0.37, 0.08, 0.69], dtype=np.float32),
            rotation[:, 0],
            rotation[:, 1],
            np.array([0.052], dtype=np.float32),
        )
    )
    virtual = previous_eef_pose9_to_virtual_eef_pose9(previous)

    scene_previous = np.array(
        [
            [0.12, -0.04, 0.23, 10.0, 20.0, 30.0],
            [-0.17, 0.09, 0.31, 40.0, 50.0, 60.0],
        ],
        dtype=np.float32,
    )
    previous_world = pose9_to_homo_np(previous[:9])
    scene_world = scene_previous.copy()
    scene_world[:, :3] = (
        scene_previous[:, :3] @ previous_world[:3, :3].T
        + previous_world[:3, 3]
    )

    actual_virtual = reference_point_cloud_to_current_eff(scene_world, virtual)
    expected_virtual = scene_previous.copy()
    expected_virtual[:, :3] = (
        scene_previous[:, :3] - T_PREVIOUS_EEF_VIRTUAL_EEF[:3, 3]
    )

    np.testing.assert_allclose(actual_virtual, expected_virtual, atol=1e-6, rtol=0.0)
