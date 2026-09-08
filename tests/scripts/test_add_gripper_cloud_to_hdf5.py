from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/song_real_libero/scripts/add_gripper_cloud_to_hdf5.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("add_gripper_cloud_to_hdf5", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        overwrite=False,
        in_place=False,
        pose_path="observations/pose_eular",
        eff_angular_path="observations/eff_angular",
        cloud_group_path="observations/cloud_rgb",
        camera="all",
        output_compression="none",
        gzip_level=4,
        eff_angular_is_normalized=False,
        gripper_max_width_m=0.085,
        gripper_points=500,
        gripper_len=0.06,
        batch_frames=1,
        drop_strategy="tail",
        shuffle=False,
        no_shuffle=False,
    )


def test_achieved_width_uses_fixed_physical_scale_not_episode_minmax() -> None:
    widths_m = MODULE.achieved_gripper_widths(
        np.array([0.0, 0.0425, 0.085, 0.12], dtype=np.float32),
        max_width_m=0.085,
    )

    np.testing.assert_allclose(widths_m, [0.0, 0.0425, 0.085, 0.085])


def test_template_is_transformed_by_achieved_eef_pose() -> None:
    pose = np.array([0.31, -0.22, 0.73, 0.4, -0.2, 0.1], dtype=np.float64)
    expected_local = MODULE.reap_gripper_template(
        0.0425,
        500,
        np.random.default_rng(13),
        gripper_len=0.06,
    )
    expected = MODULE.transform_eef_template_to_reference(expected_local, pose)
    actual = MODULE.create_gripper_points(
        0.0425,
        pose,
        500,
        np.random.default_rng(13),
        gripper_len=0.06,
    )

    np.testing.assert_allclose(actual, expected)


def test_template_uses_physical_gap_and_fixed_100mm_palm() -> None:
    width_m = 0.0425
    points = MODULE.reap_gripper_template(
        width_m,
        500,
        np.random.default_rng(19),
        gripper_len=0.06,
    )
    params = MODULE.CanonicalGripperParameters(max_width_m=0.085)
    sizes = [
        np.array([params.finger_thickness_m, params.finger_thickness_m, params.finger_length_m]),
        np.array([params.finger_thickness_m, params.finger_thickness_m, params.finger_length_m]),
        np.array([params.finger_thickness_m, params.palm_width_m, params.palm_depth_m]),
        np.array([params.palm_depth_m, params.palm_depth_m, params.handle_length_m]),
    ]
    counts = MODULE.allocate_surface_counts(500, sizes)
    cuts = np.concatenate(([0], np.cumsum(counts)))
    left = points[cuts[0] : cuts[1]]
    right = points[cuts[1] : cuts[2]]
    palm = points[cuts[2] : cuts[3]]

    # The inner faces sit at -width/2 and +width/2: each finger receives half
    # of the real opening, with no normalized-fraction geometry path.
    np.testing.assert_allclose(right[:, 1].min() - left[:, 1].max(), width_m, atol=1e-12)
    np.testing.assert_allclose(np.ptp(palm[:, 1]), 0.10, atol=1e-12)
    np.testing.assert_allclose(points[:, 2].min(), -0.11, atol=1e-12)
    np.testing.assert_allclose(points[:, 2].max(), 0.03, atol=1e-12)


def test_hdf5_output_has_rh20t_tail_and_preserves_task_and_alias(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    task = "Fold the towel from the bottom corners."
    poses = np.array(
        [
            [0.1, -0.2, 0.7, 0.0, 0.0, 0.0],
            [-0.1, 0.3, 0.8, 0.2, -0.1, 0.3],
        ],
        dtype=np.float32,
    )
    widths = np.array([[0.0425], [0.12]], dtype=np.float64)
    scene = np.zeros((2, 50_000, 6), dtype=np.float32)
    scene[..., :3] = np.arange(50_000, dtype=np.float32)[None, :, None]
    scene[..., 3:] = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    with h5py.File(source, "w") as h5_file:
        h5_file.attrs["task"] = task
        h5_file.attrs["pose_frame"] = "world"
        observations = h5_file.create_group("observations")
        observations.create_dataset("pose_eular", data=poses)
        observations.create_dataset("eff_angular", data=widths)
        clouds = observations.create_group("cloud_rgb")
        overhead = clouds.create_dataset("overhead", data=scene)
        clouds["hand"] = overhead

    args = make_args()
    assert MODULE.add_gripper_to_new_file(source, output, args, np.random.default_rng(17))

    with h5py.File(output, "r") as h5_file:
        assert h5_file.attrs["task"] == task
        assert h5_file.attrs["virtual_gripper_contract"] == MODULE.RH20T_GRIPPER_CONTRACT
        assert h5_file.attrs["virtual_gripper_pose_semantics"] == "achieved_eef_pose"
        assert h5_file.attrs["point_cloud_layout"] == "scene=49500, virtual_gripper_tail=500"
        overhead = h5_file["observations/cloud_rgb/overhead"]
        hand = h5_file["observations/cloud_rgb/hand"]
        assert MODULE.dataset_addr(overhead) == MODULE.dataset_addr(hand)
        assert overhead.shape == (2, 50_000, 6)
        np.testing.assert_array_equal(overhead[:, :49_500], scene[:, :49_500])
        np.testing.assert_array_equal(
            overhead[:, -500:, 3:6],
            np.broadcast_to(np.array([255.0, 0.0, 0.0]), (2, 500, 3)),
        )

        expected_first = MODULE.create_gripper_cloud_rgb(
            0.0425,
            poses[0],
            500,
            np.random.default_rng(17),
            gripper_len=0.06,
        )
        np.testing.assert_allclose(overhead[0, -500:], expected_first, rtol=1e-6, atol=1e-6)
        assert overhead.attrs["virtual_gripper_width_max_m"] == 0.085
        assert overhead.attrs["virtual_gripper_points"] == 500
        assert overhead.attrs["virtual_gripper_palm_width_m"] == 0.10
        assert not bool(overhead.attrs["virtual_gripper_width_normalized_for_geometry"])
