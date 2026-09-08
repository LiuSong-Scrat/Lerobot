#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect RLBench expert demonstrations into the Song LeRobot format.

This file is deliberately independent from RLBench's original
``dataset_generator.py``.  RLBench's generator removes point clouds before
writing a demo, while this collector reads the live observations first and
stores the clouds in LeRobot-compatible episode sidecars.

The main table has the same 10-dimensional convention used by the existing
CALVIN/Song data:

    [x, y, z, first_rotation_column(3), second_rotation_column(3), width]

For ``observation.state``, the first nine values are the achieved EEF pose
expressed relative to the EEF pose in frame zero. By default, ``action`` is the
RLBench expert's commanded Panda joint target converted by FK and expressed
relative to the same EEF0. ``--action-label-mode executed`` instead uses the
achieved next EEF state as the action label. The point cloud sidecar contains
the finite front-camera world cloud transformed to the current EEF frame,
followed by RGB in [0, 255].

RLBench's expert itself uses JointVelocity + Discrete gripper internally.
Those original expert commands are saved separately as
``raw_expert_actions/episode_XXXXXX.npy``.  They are not silently substituted
for the EEF labels in the main LeRobot action column.
"""

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import zarr


# Make the repository's LeRobot package and the existing point-cloud helpers
# importable when this file is run directly by an absolute path.
REPO_ROOT = Path(__file__).resolve().parents[3]
SONG_SCRIPT_ROOT = REPO_ROOT / "benchmarks" / "song_real_libero" / "scripts"
COPPELIASIM_ROOT = REPO_ROOT / "benchmarks" / "CoppeliaSim"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "benchmarks"
    / "RLBench"
    / "datasets"
    / ("rlbench_lerobot_" + time.strftime("%Y%m%d_%H%M%S"))
)
os.environ.setdefault("DISPLAY", ":99")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("COPPELIASIM_ROOT", str(COPPELIASIM_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(COPPELIASIM_ROOT))
library_path = os.environ.get("LD_LIBRARY_PATH", "")
if str(COPPELIASIM_ROOT) not in library_path.split(":"):
    os.environ["LD_LIBRARY_PATH"] = str(COPPELIASIM_ROOT) + (":" + library_path if library_path else "")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SONG_SCRIPT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero_setting.libero_pointcloud_utils import (
    RLBENCH_PANDA_GRIPPER_TEMPLATE,
    RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION,
    RLBENCH_PANDA_MAX_WIDTH,
    add_world_gripper_clouds_to_episode,
    fast_inverse_homogeneous,
    pose9_to_homo_np,
    sample_or_repeat_points,
)
from rlbench_reap_gripper import (
    LIBERO_GRIPPER_TEMPLATE,
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    LIBERO_REAP_TEMPLATE_MAX_WIDTH,
    canonical_reap_metadata,
    libero_reap_width_percent_from_physical,
)
from rlbench_worldflow_sidecars import (
    RLBENCH_PANDA_LINK0_FRAME_VERSION,
    RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE,
    WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
    WORLD_BASE_EE_POSE_DIR,
    build_robot_base_episode_sidecars,
    rlbench_panda_link0_to_world_matrix,
    validate_rigid_transform,
    write_sidecar_metadata as write_robot_base_sidecar_metadata,
)
from lerobot.policies.smolvla.song_pointseg import save_point_clouds_zarr


DEFAULT_TASKS = [
    # "close_box",
    # "close_laptop_lid",
    "toilet_seat_down",
    "sweep_to_dustpan",
    "close_fridge",
    # "phone_on_base",
    # "take_umbrella_out_of_umbrella_stand",
    # "take_frame_off_hanger",
    # "stack_wine",
    "water_plants",
]

FEATURE_NAMES = ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]
ACTION_LABEL_MODES = ("expert_target", "executed")
POINT_DIR = "point_clouds"
POSE_DIR = "world_ee_poses"
RAW_ACTION_DIR = "raw_expert_actions"
RAW_ACTION_FULL_DIR = "raw_expert_actions_full"
TASK_STATE_DIR = "initial_task_states"
OBJECT_STATE_DIR = "initial_object_states"
# Historical Song/RLBench point-cloud crop kept for dataset/evaluation compatibility.
RLBENCH_SCENE_BOUNDS = np.asarray(
    [-0.5, -1, 0.7505, 1.5, 1, 2.0], dtype=np.float32
)
OBJECT_STATE_KEYS = [
    "initial_object_names",
    "initial_object_handles",
    "initial_object_types",
    "initial_object_parent_handles",
    "initial_object_parent_names",
    "initial_object_poses",
    "initial_object_linear_velocities",
    "initial_object_angular_velocities",
    "initial_object_joint_positions",
    "initial_object_joint_velocities",
    "initial_object_joint_target_positions",
    "initial_object_joint_target_velocities",
]


def action_semantics(action_label_mode):
    if action_label_mode == "expert_target":
        return "expert Panda joint target converted by FK relative to episode EEF0"
    if action_label_mode == "executed":
        return "achieved next-frame EEF pose relative to episode EEF0"
    raise ValueError("Unknown action label mode: " + str(action_label_mode))


def action_semantics_version(action_label_mode):
    if action_label_mode == "expert_target":
        return "rlbench_expert_joint_target_fk_eef0_object_state_v3"
    if action_label_mode == "executed":
        return "rlbench_achieved_next_eef_state_eef0_v1"
    raise ValueError("Unknown action label mode: " + str(action_label_mode))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect RLBench expert data as Song LeRobot point clouds.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset output directory. Defaults to a timestamped directory under benchmarks/RLBench/datasets.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit artifact directory. Normally this is derived from "
            "--output-root; it is required when --pack-only rebuilds a dataset at "
            "a different temporary output path."
        ),
    )
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help=(
            "Do not launch RLBench or collect demos. Read the completed records and "
            "collection settings from --artifact-root/manifest.json and rebuild only "
            "the LeRobot dataset."
        ),
    )
    parser.add_argument("--repo-id", default="rlbench_song_pointcloud")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument(
        "--collection-seed",
        type=int,
        default=None,
        help="Optional NumPy seed for reproducible live-demo initialization.",
    )
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument(
        "--cache-current-points",
        type=int,
        default=None,
        help="Number of points retained in the current-frame PointSeg cache. Defaults to --num-points.",
    )
    parser.add_argument(
        "--cache-future-points",
        type=int,
        default=None,
        help="Number of points retained in the future-frame PointSeg cache. Defaults to --cache-current-points.",
    )
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument(
        "--gripper-template",
        choices=[RLBENCH_PANDA_GRIPPER_TEMPLATE, LIBERO_GRIPPER_TEMPLATE],
        default=LIBERO_GRIPPER_TEMPLATE,
        help=(
            "Virtual gripper merged into point clouds before PointSeg caching. "
            "reap is the canonical RLBench-aligned REAP v4 gripper."
        ),
    )
    parser.add_argument("--gripper-max-width", type=float, default=RLBENCH_PANDA_MAX_WIDTH)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-demo-attempts", type=int, default=10)
    parser.add_argument(
        "--abort-on-episode-failure",
        action="store_true",
        help=(
            "Abort collection after the configured attempts for one episode "
            "instead of retrying that episode indefinitely. This is intended "
            "for isolated exact-scene planner diagnostics."
        ),
    )
    parser.add_argument(
        "--expert-path-mode",
        choices=[
            "linear_then_rrt",
            "linear_only",
            "segmented_linear",
            "segmented_linear_then_rrt",
            "rrt_only",
        ],
        default="linear_then_rrt",
        help=(
            "Expert waypoint planner. linear_then_rrt preserves RLBench's "
            "linear-first/RRTConnect-fallback behavior; linear_only forces "
            "ordinary Point waypoints through Cartesian linear IK and never "
            "calls the nonlinear planner; segmented_linear divides the same "
            "Cartesian line and quaternion SLERP into short IK segments; "
            "segmented_linear_then_rrt uses the same short targets but permits "
            "RRTConnect only for an individually infeasible segment; "
            "rrt_only forces ordinary Point "
            "waypoints directly through nonlinear RRTConnect. Predefined "
            "Cartesian paths are preserved."
        ),
    )
    parser.add_argument(
        "--segmented-linear-segments",
        type=int,
        default=10,
        help=(
            "Number of short Cartesian IK segments per ordinary waypoint when "
            "--expert-path-mode=segmented_linear (default: 10)."
        ),
    )
    parser.add_argument(
        "--replay-random-seeds-from-artifacts",
        type=Path,
        default=None,
        help=(
            "Optional artifact root from an earlier collection. Before each live "
            "demo reset, restore that episode's complete saved NumPy MT19937 state. "
            "This provides matched initial scenes for planner A/B comparisons."
        ),
    )
    parser.add_argument(
        "--phone-base-max-initial-distance-m",
        type=float,
        default=None,
        help=(
            "For phone_on_base only, retain demos whose initial phone-center to "
            "success-sensor-center distance is at most this many meters."
        ),
    )
    parser.add_argument(
        "--phone-eef-max-initial-distance-m",
        type=float,
        default=None,
        help=(
            "For phone_on_base only, sample the scene before expert planning and "
            "retain it only when the initial phone-center to physical EEF-tip "
            "distance is at most this many meters."
        ),
    )
    parser.add_argument(
        "--replay-scenes-from-artifacts",
        type=Path,
        default=None,
        help=(
            "Artifact root from a matched collection. Restore each episode's "
            "configuration tree and readable per-object snapshot before expert "
            "planning, enabling strict planner A/B collection on the same scenes."
        ),
    )
    parser.add_argument(
        "--collection-workers",
        type=int,
        default=1,
        help=(
            "Number of independent RLBench environments used in parallel. "
            "Each worker needs its own X display; default is 1."
        ),
    )
    parser.add_argument(
        "--collection-display-base",
        type=int,
        default=None,
        help=(
            "Base X display number for parallel collection. Worker i uses "
            ":(base+i). If omitted, the current DISPLAY number is used."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--delete-artifacts-after-pack",
        action="store_true",
        help=(
            "After each episode is successfully saved and its sidecars are verified, "
            "delete that episode's raw artifact directory to reduce peak disk use."
        ),
    )
    parser.add_argument("--skip-pointseg-cache", action="store_true")
    parser.add_argument("--cache-output-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-python",
        type=Path,
        default=Path("/home/liusong/miniconda3/envs/reap_metaworld/bin/python"),
    )
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--cache-num-workers", type=int, default=8)
    parser.add_argument("--cache-device", default="cuda")
    parser.add_argument("--cache-vis-count", type=int, default=8)
    parser.add_argument("--cache-vis-one-episode-per-task", action="store_true")
    parser.add_argument("--motion-rotation-radius", type=float, default=0.18)
    parser.add_argument("--motion-baseline-threshold", type=float, default=0.010)
    parser.add_argument("--motion-baseline-temperature", type=float, default=0.006)
    parser.add_argument("--motion-relative-margin", type=float, default=0.05)
    parser.add_argument("--motion-relative-tau", type=float, default=0.08)
    parser.add_argument("--trajectory-sigma", type=float, default=0.22)
    parser.add_argument("--contact-radius", type=float, default=0.22)
    parser.add_argument("--contact-temperature", type=float, default=0.055)
    parser.add_argument("--approach-margin", type=float, default=0.0)
    parser.add_argument("--approach-tau", type=float, default=0.04)
    parser.add_argument("--background-trajectory-sigma", type=float, default=0.32)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--action-alignment",
        choices=["transition", "observation"],
        default="transition",
        help=(
            "transition stores observation[t] with the expert command that moves "
            "to the next frame; observation preserves RLBench's post-step misc action."
        ),
    )
    parser.add_argument(
        "--action-label-mode",
        choices=ACTION_LABEL_MODES,
        default="expert_target",
        help=(
            "expert_target keeps the nominal expert joint-target FK label; "
            "executed uses the achieved EEF state at the aligned next frame."
        ),
    )
    parser.add_argument(
        "--generate-world-base-worldflow-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate achieved and commanded EEF pose sidecars in the complete "
            "Panda robot-base frame (default: enabled)."
        ),
    )
    return parser.parse_args()


def pose7_to_pose9(pose7):
    """Convert RLBench [xyz, qx, qy, qz, qw] to the Song 9D pose."""
    from scipy.spatial.transform import Rotation

    pose7 = np.asarray(pose7, dtype=np.float32).reshape(-1)
    rotation = Rotation.from_quat(pose7[3:7]).as_matrix().astype(np.float32)
    return np.concatenate((pose7[:3], rotation[:, 0], rotation[:, 1])).astype(np.float32)


def configuration_tree_to_bytes(configuration_tree):
    """Copy a CoppeliaSim configuration tree into ordinary Python bytes.

    PyRep 4.1 currently returns CoppeliaSim's native ``char *`` pointer even
    though ``get_configuration_tree()`` is annotated as returning ``bytes``.
    The first four bytes of this binary buffer store its complete byte length.
    It must not be copied with ``ffi.string()`` because configuration trees
    contain zero bytes and would be truncated at the first zero.
    """
    if isinstance(configuration_tree, bytes):
        return configuration_tree
    if isinstance(configuration_tree, (bytearray, memoryview)):
        return bytes(configuration_tree)

    from pyrep.backend import sim

    try:
        header = bytes(sim.ffi.buffer(configuration_tree, 4))
        byte_count = int.from_bytes(header, byteorder="little", signed=False)
        if byte_count < 4 or byte_count > 256 * 1024 * 1024:
            raise RuntimeError(
                "Invalid CoppeliaSim configuration-tree size: " + str(byte_count)
            )
        return bytes(sim.ffi.buffer(configuration_tree, byte_count))
    finally:
        # simGetConfigurationTree allocates this native buffer. After making
        # the Python copy, release it exactly once to avoid one leak per demo.
        sim.simReleaseBuffer(configuration_tree)


def capture_initial_object_states(task):
    """Capture readable per-object state at the recorded first frame."""
    objects = task.get_base().get_objects_in_tree(
        exclude_base=False, first_generation_only=False
    )
    count = len(objects)
    names = []
    handles = np.empty(count, dtype=np.int64)
    types = []
    parent_handles = np.full(count, -1, dtype=np.int64)
    parent_names = [""] * count
    poses = np.full((count, 7), np.nan, dtype=np.float32)
    linear_velocities = np.full((count, 3), np.nan, dtype=np.float32)
    angular_velocities = np.full((count, 3), np.nan, dtype=np.float32)
    joint_positions = np.full(count, np.nan, dtype=np.float32)
    joint_velocities = np.full(count, np.nan, dtype=np.float32)
    joint_target_positions = np.full(count, np.nan, dtype=np.float32)
    joint_target_velocities = np.full(count, np.nan, dtype=np.float32)

    for index, obj in enumerate(objects):
        names.append(obj.get_name())
        handles[index] = obj.get_handle()
        object_type = obj.get_type()
        types.append(getattr(object_type, "name", str(object_type)))
        try:
            parent = obj.get_parent()
            if parent is not None:
                parent_handles[index] = parent.get_handle()
                parent_names[index] = parent.get_name()
        except Exception:
            pass
        try:
            poses[index] = np.asarray(obj.get_pose(), dtype=np.float32)
        except Exception:
            pass
        try:
            linear, angular = obj.get_velocity()
            linear_velocities[index] = np.asarray(linear, dtype=np.float32)
            angular_velocities[index] = np.asarray(angular, dtype=np.float32)
        except Exception:
            pass
        if hasattr(obj, "get_joint_position"):
            try:
                joint_positions[index] = float(obj.get_joint_position())
                joint_velocities[index] = float(obj.get_joint_velocity())
                joint_target_positions[index] = float(obj.get_joint_target_position())
                joint_target_velocities[index] = float(obj.get_joint_target_velocity())
            except Exception:
                pass

    return {
        "initial_object_names": np.asarray(names, dtype="U256"),
        "initial_object_handles": handles,
        "initial_object_types": np.asarray(types, dtype="U64"),
        "initial_object_parent_handles": parent_handles,
        "initial_object_parent_names": np.asarray(parent_names, dtype="U256"),
        "initial_object_poses": poses,
        "initial_object_linear_velocities": linear_velocities,
        "initial_object_angular_velocities": angular_velocities,
        "initial_object_joint_positions": joint_positions,
        "initial_object_joint_velocities": joint_velocities,
        "initial_object_joint_target_positions": joint_target_positions,
        "initial_object_joint_target_velocities": joint_target_velocities,
    }


def demo_random_state_from_arrays(arrays):
    """Rebuild the complete NumPy RNG state saved with a collected demo."""
    return (
        "MT19937",
        np.asarray(arrays["demo_random_seed_state"], dtype=np.uint32).copy(),
        int(arrays["demo_random_seed_position"]),
        int(arrays["demo_random_seed_has_gauss"]),
        float(arrays["demo_random_seed_cached_gaussian"]),
    )


def restore_initial_object_states_from_arrays(task, arrays):
    """Restore the readable per-object snapshot stored in one collection artifact."""
    objects = task.get_base().get_objects_in_tree(
        exclude_base=False, first_generation_only=False
    )
    current_by_name = {obj.get_name(): obj for obj in objects}
    recorded_names = [str(name) for name in arrays["initial_object_names"]]
    missing = [name for name in recorded_names if name not in current_by_name]
    if missing:
        raise RuntimeError(
            "Cannot restore matched scene; task objects are missing: "
            + ", ".join(missing)
        )

    for index, name in enumerate(recorded_names):
        obj = current_by_name[name]
        pose = np.asarray(arrays["initial_object_poses"][index], dtype=np.float64)
        if np.isfinite(pose).all():
            obj.set_pose(pose.tolist(), reset_dynamics=True)
        joint_position = float(arrays["initial_object_joint_positions"][index])
        if np.isfinite(joint_position) and hasattr(obj, "set_joint_position"):
            obj.set_joint_position(joint_position, disable_dynamics=True)
        joint_target = float(
            arrays["initial_object_joint_target_positions"][index]
        )
        if np.isfinite(joint_target) and hasattr(obj, "set_joint_target_position"):
            obj.set_joint_target_position(joint_target)
        joint_target_velocity = float(
            arrays["initial_object_joint_target_velocities"][index]
        )
        if np.isfinite(joint_target_velocity) and hasattr(
            obj, "set_joint_target_velocity"
        ):
            obj.set_joint_target_velocity(joint_target_velocity)

    # Joint setters can step or move descendants. Reapply absolute poses last.
    for index, name in enumerate(recorded_names):
        pose = np.asarray(arrays["initial_object_poses"][index], dtype=np.float64)
        if np.isfinite(pose).all():
            current_by_name[name].set_pose(pose.tolist(), reset_dynamics=True)


def restore_task_environment_from_artifact_arrays(task_env, arrays):
    """Reset robot/task bookkeeping, then restore one exact recorded task scene."""
    random_state = demo_random_state_from_arrays(arrays)
    reset_attempts = int(arrays["demo_num_reset_attempts"])

    class RecordedDemoPlacement:
        num_reset_attempts = reset_attempts

    np.random.set_state(random_state)
    descriptions, _ = task_env.reset(RecordedDemoPlacement())
    configuration_bytes = np.asarray(
        arrays["initial_task_state_bytes"], dtype=np.uint8
    ).tobytes()
    object_count = int(arrays["initial_task_state_object_count"])
    task_env._task.restore_state((configuration_bytes, object_count))
    restore_initial_object_states_from_arrays(task_env._task, arrays)
    return descriptions, task_env.get_observation(), random_state, reset_attempts


def collect_live_demo_from_current_scene(task_env):
    """Run the RLBench expert without allowing TaskEnvironment to reset again."""
    control_loop_enabled = task_env._robot.arm.joints[0].is_control_loop_enabled()
    task_env._robot.arm.set_control_loop_enabled(True)
    try:
        return task_env._scene.get_demo()
    finally:
        task_env._robot.arm.set_control_loop_enabled(control_loop_enabled)


def pose9_to_umi(pose_sequence):
    """Express every world pose in the first EEF frame of this episode."""
    poses = np.asarray(pose_sequence, dtype=np.float32)
    transforms = pose9_to_homo_np(poses)
    first_inverse = fast_inverse_homogeneous(transforms[0])
    relative = first_inverse[None] @ transforms
    return np.concatenate((relative[:, :3, 3], relative[:, :3, 0], relative[:, :3, 1]), axis=1)


def matrix_to_pose9(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[:3, 3], matrix[:3, 0], matrix[:3, 1])).astype(np.float32)


def rotation_about_axis(axis, point, angle):
    axis = np.asarray(axis, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [cosine + x * x * one_minus_cosine,
             x * y * one_minus_cosine - z * sine,
             x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine,
             cosine + y * y * one_minus_cosine,
             y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine,
             z * y * one_minus_cosine + x * sine,
             cosine + z * z * one_minus_cosine],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return transform


def read_panda_fk_model(environment):
    """Read the fixed Panda kinematic model once before collecting tasks."""
    arm = environment._robot.arm
    original_positions = arm.get_joint_positions()
    arm.set_joint_positions([0.0] * 7, disable_dynamics=True)
    home_joint_positions = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    axes = []
    points = []
    for joint in arm.joints:
        matrix = np.asarray(joint.get_matrix(), dtype=np.float64)
        points.append(matrix[:3, 3].copy())
        axes.append(matrix[:3, 2].copy())
    home_tip = np.asarray(arm.get_tip().get_matrix(), dtype=np.float64)
    arm.set_joint_positions(original_positions, disable_dynamics=True)
    return axes, points, home_tip, home_joint_positions


def fk_pose(model, joint_positions):
    axes, points, home_tip, home_joint_positions = model
    transform = np.eye(4, dtype=np.float64)
    for axis, point, angle, home_angle in zip(
        axes, points, joint_positions, home_joint_positions
    ):
        transform = transform @ rotation_about_axis(axis, point, angle - home_angle)
    return transform @ home_tip


def expert_actions_to_eef0(raw_actions, world_poses, first_width, gripper_max_width, fk_model):
    """Convert recorded Panda joint targets directly into EEF0 pose targets."""
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    world_poses = np.asarray(world_poses, dtype=np.float32)
    world_to_eef0 = fast_inverse_homogeneous(pose9_to_homo_np(world_poses[0]))
    actions = np.empty((len(raw_actions), 10), dtype=np.float32)
    actions[0] = 0.0
    actions[0, 3] = 1.0
    actions[0, 7] = 1.0
    actions[0, 9] = float(first_width)
    position_errors = []
    for frame_index in range(1, len(raw_actions)):
        raw_action = raw_actions[frame_index]
        if raw_action.shape != (8,) or not np.isfinite(raw_action).all():
            raise RuntimeError("Missing RLBench expert joint target at frame " + str(frame_index))
        target_world = fk_pose(fk_model, raw_action[:7])
        target_eef0 = world_to_eef0 @ target_world
        actions[frame_index, :9] = matrix_to_pose9(target_eef0)
        actions[frame_index, 9] = float(np.clip(raw_action[7], 0.0, 1.0)) * gripper_max_width
        position_errors.append(float(np.linalg.norm(target_world[:3, 3] - world_poses[frame_index, :3])))
    return actions, np.asarray(position_errors, dtype=np.float32)


def task_class_from_name(task_name):
    class_name = "".join(part[:1].upper() + part[1:] for part in task_name.split("_"))
    module = importlib.import_module("rlbench.tasks." + task_name)
    return getattr(module, class_name)


def resolve_tasks(args):
    if args.all_tasks:
        import rlbench

        names = []
        for name in rlbench.TASKS:
            name = name.replace(".py", "")
            if not name.startswith("place_holder"):
                names.append(name)
        return sorted(names)
    if args.tasks:
        return list(args.tasks)
    return list(DEFAULT_TASKS)


def make_observation_config(image_size):
    from rlbench import CameraConfig, ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.front_camera = CameraConfig(
        rgb=True, depth=False, point_cloud=True, mask=False, image_size=(image_size, image_size)
    )
    config.wrist_camera = CameraConfig(
        rgb=False, depth=False, point_cloud=False, mask=False, image_size=(image_size, image_size)
    )
    config.gripper_open = True
    config.gripper_pose = True
    config.joint_positions = True
    config.joint_velocities = True
    config.record_gripper_closing = True
    return config


def color_to_uint8(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if rgb.size and float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def observation_cloud(observation, fallback_to_all_finite=False):
    """Return front-camera points inside the historical world-space crop."""
    if observation.front_point_cloud is None:
        raise RuntimeError("RLBench did not return a front-camera point cloud.")
    xyz = np.asarray(observation.front_point_cloud, dtype=np.float32).reshape(-1, 3)
    if observation.front_rgb is None:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    else:
        rgb = color_to_uint8(observation.front_rgb).reshape(-1, 3)
    count = min(len(xyz), len(rgb))
    xyz = xyz[:count]
    rgb = rgb[:count]
    lower = RLBENCH_SCENE_BOUNDS[:3]
    upper = RLBENCH_SCENE_BOUNDS[3:]
    valid = np.isfinite(xyz).all(axis=1)
    valid &= (xyz >= lower).all(axis=1)
    valid &= (xyz <= upper).all(axis=1)
    if not valid.any():
        finite = np.isfinite(xyz).all(axis=1)
        if fallback_to_all_finite and finite.any():
            # A live evaluation camera can see only floor/background points
            # outside the training crop after task randomization. Preserve
            # those finite observations instead of aborting before inference.
            valid = finite
        else:
            raise RuntimeError(
                "RLBench returned no usable front-camera point-cloud points "
                "(finite=%d, in_bounds=%d)." % (int(finite.sum()), int(valid.sum()))
            )
    return np.concatenate((xyz[valid], rgb[valid].astype(np.float32)), axis=1).astype(np.float32)


def make_episode_arrays(
    demo,
    num_points,
    gripper_points,
    gripper_template,
    gripper_max_width,
    fps,
    seed,
    fk_model,
    t_world_base,
    generate_world_base_worldflow_sidecars,
    action_alignment,
    action_label_mode,
):
    """Convert one successful RLBench Demo object into numpy arrays."""
    observations = [demo[index] for index in range(len(demo))]
    if len(observations) < 2:
        raise RuntimeError("The expert demo contains fewer than two observations.")

    world_poses = []
    grippers = []
    world_clouds = []
    raw_actions = []
    images = []

    for frame_index, observation in enumerate(observations):
        world_poses.append(pose7_to_pose9(observation.gripper_pose))
        # Keep the continuous simulator opening amount. RLBench's Panda
        # physical width is represented as [0, gripper_max_width] meters.
        open_value = float(np.clip(observation.gripper_open, 0.0, 1.0))
        grippers.append(open_value * float(gripper_max_width))
        world_clouds.append(
            sample_or_repeat_points(
                # Some task cameras are entirely outside the historical crop;
                # retain their finite camera points instead of rejecting the demo.
                observation_cloud(observation, fallback_to_all_finite=True),
                num_points,
                seed + frame_index,
            )
        )
        images.append(np.asarray(observation.front_rgb, dtype=np.uint8))

        command = observation.misc.get("joint_position_action")
        if command is None:
            raw_actions.append(np.full((8,), np.nan, dtype=np.float32))
        else:
            command = np.asarray(command, dtype=np.float32).reshape(-1)
            if command.size != 8:
                raise RuntimeError("RLBench expert joint_position_action is not 8-dimensional.")
            raw_actions.append(command)

    world_poses = np.asarray(world_poses, dtype=np.float32)
    grippers = np.asarray(grippers, dtype=np.float32)
    world_clouds = np.asarray(world_clouds, dtype=np.float32)

    if gripper_template == LIBERO_GRIPPER_TEMPLATE:
        # Preserve LIBERO's physical width / 0.1 normalization and fixed
        # four-box body dimensions, but recover the two-finger aperture with
        # the same 0.1 m scale so the virtual gap equals RLBench's physical gap.
        # The shape is translated to local Z=-0.09 m so its forward tip aligns
        # with the RLBench Panda tip. RLBench state/action widths remain in
        # their native physical [0, 0.08] m range.
        cloud_gripper_widths = libero_reap_width_percent_from_physical(grippers)
        cloud_widths_are_normalized = True
        cloud_gripper_max_width = None
        cloud_gripper_opening_max_width = LIBERO_REAP_OPENING_MAX_WIDTH
        cloud_gripper_len = LIBERO_REAP_GRIPPER_LEN
    else:
        cloud_gripper_widths = grippers
        cloud_widths_are_normalized = False
        cloud_gripper_max_width = gripper_max_width
        cloud_gripper_opening_max_width = None
        cloud_gripper_len = 0.0

    # Merge the selected template before PointSeg sees the stored point cloud.
    world_clouds = add_world_gripper_clouds_to_episode(
        world_clouds,
        world_poses,
        cloud_gripper_widths,
        total_points=num_points,
        gripper_points=gripper_points,
        gripper_template=gripper_template,
        gripper_len=cloud_gripper_len,
        seed=seed,
        drop_strategy="tail",
        shuffle_points=False,
        widths_are_normalized=cloud_widths_are_normalized,
        gripper_max_width=cloud_gripper_max_width,
        gripper_opening_max_width=cloud_gripper_opening_max_width,
    )
    # add_world_gripper_clouds_to_episode already returns current-EFF clouds.
    point_clouds = world_clouds
    relative_poses = pose9_to_umi(world_poses)
    states = np.concatenate((relative_poses, grippers[:, None]), axis=1).astype(np.float32)
    expert_target_actions, fk_position_errors = expert_actions_to_eef0(
        raw_actions,
        world_poses,
        grippers[0],
        gripper_max_width,
        fk_model,
    )
    actions = expert_target_actions

    if action_label_mode == "executed":
        # The alignment slices below turn this into state[t+1] for transition
        # mode and state[t] for observation mode.
        actions = states.copy()
    elif action_label_mode != "expert_target":
        raise ValueError("Unknown action label mode: " + str(action_label_mode))

    # RLBench attaches the command to the observation recorded after that
    # command was executed. For a LeRobot transition, pair command i+1 with
    # observation i and drop the terminal observation with no next command.
    raw_actions_array = np.asarray(raw_actions, dtype=np.float32)
    if action_alignment == "transition":
        if len(actions) < 2:
            raise RuntimeError("A transition-aligned episode needs at least two observations.")
        output = {
            "actions": actions[1:],
            "states": states[:-1],
            "point_clouds": point_clouds[:-1],
            "world_ee_poses": world_poses[:-1],
            "raw_expert_actions": raw_actions_array[1:],
            "raw_expert_actions_full": raw_actions_array,
            "images": np.asarray(images[:-1], dtype=np.uint8),
            "timestamps": np.arange(len(actions) - 1, dtype=np.float32) / float(fps),
            "fk_position_errors": fk_position_errors,
            "action_alignment": action_alignment,
            "action_label_mode": action_label_mode,
        }
        commanded_targets_eef0 = expert_target_actions[1:, :9]
    elif action_alignment == "observation":
        output = {
            "actions": actions,
            "states": states,
            "point_clouds": point_clouds,
            "world_ee_poses": world_poses,
            "raw_expert_actions": raw_actions_array,
            "raw_expert_actions_full": raw_actions_array,
            "images": np.asarray(images, dtype=np.uint8),
            "timestamps": np.arange(len(actions), dtype=np.float32) / float(fps),
            "fk_position_errors": fk_position_errors,
            "action_alignment": action_alignment,
            "action_label_mode": action_label_mode,
        }
        commanded_targets_eef0 = expert_target_actions[:, :9]
    else:
        raise ValueError("Unknown action alignment: " + str(action_alignment))

    if generate_world_base_worldflow_sidecars:
        sidecar_actions = np.zeros((len(commanded_targets_eef0), 10), dtype=np.float32)
        sidecar_actions[:, :9] = commanded_targets_eef0
        base_ee, base_target, sidecar_validation = build_robot_base_episode_sidecars(
            output["world_ee_poses"],
            sidecar_actions,
            t_world_base,
        )
        output["world_base_ee_poses"] = base_ee
        output["world_base_action_target_ee_poses"] = base_target
        output["worldflow_base_sidecar_validation"] = sidecar_validation
        output["T_world_base"] = np.asarray(t_world_base, dtype=np.float64)
        output["T_base_world"] = np.linalg.inv(np.asarray(t_world_base, dtype=np.float64))
    return output


def validate_captured_episode(arrays, task_name, episode_index, gripper_points):
    """Reject camera/point-cloud failures before an episode reaches the dataset."""
    images = np.asarray(arrays["images"])
    if images.size == 0 or int(images.max()) == 0:
        raise RuntimeError(
            "front RGB is completely black for task="
            + str(task_name)
            + " episode="
            + str(episode_index)
            + "; the worker display/camera did not render"
        )

    point_clouds = np.asarray(arrays["point_clouds"], dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 6:
        raise RuntimeError("captured point-cloud array has an invalid shape: " + str(point_clouds.shape))
    scene_points = point_clouds[:, :-int(min(gripper_points, point_clouds.shape[1])), :]
    scene_rgb = scene_points[..., 3:6]
    if scene_rgb.size == 0 or float(np.max(scene_rgb)) <= 0.0:
        raise RuntimeError(
            "front point-cloud RGB is empty for task="
            + str(task_name)
            + " episode="
            + str(episode_index)
        )

    # The stored cloud is in the current EEF frame. Reconstruct frame-0 world
    # coordinates and verify that the camera cloud still lies in the RLBench crop.
    world_pose0 = pose9_to_homo_np(np.asarray(arrays["world_ee_poses"])[0, :9])
    world_xyz = (
        scene_points[0, ..., :3] @ world_pose0[:3, :3].T
        + world_pose0[:3, 3]
    )
    lower = RLBENCH_SCENE_BOUNDS[:3] - 0.05
    upper = RLBENCH_SCENE_BOUNDS[3:] + 0.05
    in_bounds = np.isfinite(world_xyz).all(axis=1) & (world_xyz >= lower).all(axis=1) & (world_xyz <= upper).all(axis=1)
    if float(in_bounds.mean()) < 0.98:
        raise RuntimeError(
            "frame-0 point cloud is inconsistent with the RLBench world crop for task="
            + str(task_name)
            + " episode="
            + str(episode_index)
            + f" (in_bounds={float(in_bounds.mean()):.3f})"
        )


def episode_name(task_name, local_index):
    return task_name + "__episode_" + str(local_index).zfill(5)


def artifact_is_complete(path, require_world_base_worldflow_sidecars=True):
    arrays_path = path / "arrays.npz"
    if not (
        path.is_dir()
        and arrays_path.is_file()
        and (path / "point_clouds.zarr").is_dir()
        and (path / "record.json").is_file()
    ):
        return False

    # v1 artifacts did not save the task configuration tree. They cannot
    # restore the first-frame object layout and must not be reused silently.
    try:
        with np.load(arrays_path) as arrays:
            complete = (
                "initial_task_state_bytes" in arrays.files
                and "initial_task_state_object_count" in arrays.files
                and "raw_expert_actions_full" in arrays.files
                and all(key in arrays.files for key in OBJECT_STATE_KEYS)
            )
            if require_world_base_worldflow_sidecars:
                complete = complete and all(
                    key in arrays.files
                    for key in (
                        "world_base_ee_poses",
                        "world_base_action_target_ee_poses",
                        "T_world_base",
                        "T_base_world",
                    )
                )
            return complete
    except Exception:
        return False


def save_artifact(path, task_name, local_index, arrays, description, variation):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    artifact_arrays = {
        "actions": arrays["actions"],
        "states": arrays["states"],
        "world_ee_poses": arrays["world_ee_poses"],
        "raw_expert_actions": arrays["raw_expert_actions"],
        "raw_expert_actions_full": arrays["raw_expert_actions_full"],
        "images": arrays["images"],
        "timestamps": arrays["timestamps"],
        "fk_position_errors": arrays["fk_position_errors"],
        "initial_task_state_bytes": arrays["initial_task_state_bytes"],
        "initial_task_state_object_count": arrays["initial_task_state_object_count"],
        "demo_random_seed_state": arrays["demo_random_seed_state"],
        "demo_random_seed_position": arrays["demo_random_seed_position"],
        "demo_random_seed_has_gauss": arrays["demo_random_seed_has_gauss"],
        "demo_random_seed_cached_gaussian": arrays["demo_random_seed_cached_gaussian"],
        "demo_num_reset_attempts": arrays["demo_num_reset_attempts"],
    }
    artifact_arrays.update(
        {key: arrays[key] for key in OBJECT_STATE_KEYS if key in arrays}
    )
    artifact_arrays.update(
        {
            key: arrays[key]
            for key in (
                "world_base_ee_poses",
                "world_base_action_target_ee_poses",
                "T_world_base",
                "T_base_world",
                "expert_path_mode",
                "expert_linear_path_calls",
                "expert_rrt_path_calls",
                "initial_phone_to_base_sensor_distance_m",
                "initial_phone_to_eef_distance_m",
                "matched_scene_source",
            )
            if key in arrays
        }
    )
    np.savez(
        temp / "arrays.npz",
        **artifact_arrays,
    )
    save_point_clouds_zarr(temp / "point_clouds.zarr", arrays["point_clouds"], compression_level=3)
    record = {
        "task": task_name,
        "local_episode_index": int(local_index),
        "description": str(description),
        "variation": int(variation),
        "frames": int(len(arrays["actions"])),
        "reset_first_rgb_mae": float(arrays["reset_first_rgb_mae"]),
        "initial_task_state_object_count": int(arrays["initial_task_state_object_count"]),
        "initial_object_state_count": int(len(arrays.get("initial_object_names", []))),
        "fk_target_vs_achieved_position_error_median_m": (
            float(np.median(arrays["fk_position_errors"]))
            if len(arrays["fk_position_errors"])
            else 0.0
        ),
        "fk_target_vs_achieved_position_error_max_m": (
            float(np.max(arrays["fk_position_errors"]))
            if len(arrays["fk_position_errors"])
            else 0.0
        ),
        "action_alignment": str(arrays["action_alignment"]),
        "action_label_mode": str(arrays["action_label_mode"]),
        "world_base_worldflow_sidecars": bool("world_base_ee_poses" in arrays),
        "expert_path_mode": str(arrays.get("expert_path_mode", "linear_then_rrt")),
        "expert_linear_path_calls": int(arrays.get("expert_linear_path_calls", 0)),
        "expert_rrt_path_calls": int(arrays.get("expert_rrt_path_calls", 0)),
        "worldflow_base_sidecar_validation": arrays.get(
            "worldflow_base_sidecar_validation"
        ),
        "created_unix_s": time.time(),
    }
    if "initial_phone_to_base_sensor_distance_m" in arrays:
        record["initial_phone_to_base_sensor_distance_m"] = float(
            arrays["initial_phone_to_base_sensor_distance_m"]
        )
    if "initial_phone_to_eef_distance_m" in arrays:
        record["initial_phone_to_eef_distance_m"] = float(
            arrays["initial_phone_to_eef_distance_m"]
        )
    if "matched_scene_source" in arrays:
        record["matched_scene_source"] = str(arrays["matched_scene_source"])
    with open(temp / "record.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    if path.exists():
        shutil.rmtree(path)
    temp.rename(path)


def copy_tree_with_hardlinks(source, destination):
    """Copy a Zarr directory without duplicating data blocks on this disk."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir()
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copy2(source_path, destination_path)


def write_sidecar_meta(root, t_world_base=None, action_alignment="transition"):
    point_dir = root / POINT_DIR
    pose_dir = root / POSE_DIR
    raw_dir = root / RAW_ACTION_DIR
    raw_full_dir = root / RAW_ACTION_FULL_DIR
    task_state_dir = root / TASK_STATE_DIR
    object_state_dir = root / OBJECT_STATE_DIR
    point_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_full_dir.mkdir(parents=True, exist_ok=True)
    task_state_dir.mkdir(parents=True, exist_ok=True)
    object_state_dir.mkdir(parents=True, exist_ok=True)
    if t_world_base is not None:
        write_robot_base_sidecar_metadata(
            root,
            t_world_base,
            RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE,
            action_alignment=action_alignment,
            base_frame_definition=RLBENCH_PANDA_LINK0_FRAME_VERSION,
        )
    with open(point_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "observation.point_cloud",
                "dtype": "float32",
                "shape": [None, 6],
                "layout": "episode_array",
                "storage_format": "zarr",
                "zarr_encoding": "packed_xyz_float16_rgb_uint8",
                "path_format": "point_clouds/episode_{episode_index:06d}.zarr",
                "coordinate_frame": "current_eff",
                "source_frame": "RLBench_world",
            },
            file,
            indent=2,
        )
    with open(pose_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "worldflow.ee_poses",
                "shape": [9],
                "dtype": "float32",
                "coordinate_frame": "RLBench_world",
                "path_format": "world_ee_poses/episode_{episode_index:06d}.npy",
            },
            file,
            indent=2,
        )
    with open(raw_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.raw_expert_action",
                "shape": [8],
                "dtype": "float32",
                "layout": "episode_npy",
                "values": "7 joint positions followed by 0=closed/1=open",
                "alignment": "same transition index as action; first command moves observation[t] to observation[t+1]",
                "path_format": "raw_expert_actions/episode_{episode_index:06d}.npy",
            },
            file,
            indent=2,
        )
    with open(raw_full_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.raw_expert_action_full_demo",
                "shape": [None, 8],
                "dtype": "float32",
                "values": "RLBench demo observations including row 0 NaN and post-step commands",
                "path_format": "raw_expert_actions_full/episode_{episode_index:06d}.npy",
            },
            file,
            indent=2,
        )
    with open(task_state_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.initial_task_state",
                "layout": "episode_npz",
                "path_format": "initial_task_states/episode_{episode_index:06d}.npz",
                "configuration_bytes": "RLBench Task.get_state() configuration tree as uint8",
                "object_count": "Object count checked by RLBench Task.restore_state()",
                "purpose": "Restore the recorded first-frame task-object layout before action replay",
            },
            file,
            indent=2,
        )
    with open(object_state_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.initial_object_states",
                "layout": "episode_npz",
                "path_format": "initial_object_states/episode_{episode_index:06d}.npz",
                "coordinate_frame": "RLBench_world",
                "fields": {
                    "initial_object_names": "object names",
                    "initial_object_handles": "CoppeliaSim handles",
                    "initial_object_types": "PyRep ObjectType names",
                    "initial_object_parent_handles": "parent handles, -1 means none",
                    "initial_object_parent_names": "parent names, empty means none",
                    "initial_object_poses": "[x,y,z,qx,qy,qz,qw]",
                    "initial_object_linear_velocities": "world linear velocity m/s",
                    "initial_object_angular_velocities": "world angular velocity rad/s",
                    "initial_object_joint_positions": "NaN for non-joints",
                    "initial_object_joint_velocities": "NaN for non-joints",
                    "initial_object_joint_target_positions": "NaN for non-joints",
                    "initial_object_joint_target_velocities": "NaN for non-joints",
                },
                "purpose": "Readable per-object reset snapshot; exact replay uses initial_task_states.",
            },
            file,
            indent=2,
        )


def write_frame_to_dataset(dataset, task, arrays, frame_index):
    image_key = "observation.images.front"
    dataset.add_frame(
        {
            "task": task,
            "action": arrays["actions"][frame_index],
            "observation.state": arrays["states"][frame_index],
            image_key: arrays["images"][frame_index],
        }
    )


def create_dataset(
    output_root,
    repo_id,
    fps,
    image_size,
    t_world_base=None,
    action_alignment="transition",
):
    features = {
        "action": {"dtype": "float32", "shape": (10,), "names": FEATURE_NAMES},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": FEATURE_NAMES},
        "observation.images.front": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        features=features,
        robot_type="rlbench_panda",
        use_videos=False,
    )
    write_sidecar_meta(
        dataset.root,
        t_world_base=t_world_base,
        action_alignment=action_alignment,
    )
    return dataset


def verify_packed_episode(
    output_root,
    episode_index,
    expected_frames,
    expected_points,
    require_world_base_worldflow_sidecars=True,
):
    """Verify durable episode outputs before a source artifact may be deleted."""

    stem = "episode_" + str(episode_index).zfill(6)
    point_path = output_root / POINT_DIR / (stem + ".zarr")
    point_group = zarr.open(str(point_path), mode="r")
    expected_shape = (int(expected_frames), int(expected_points), 3)
    xyz_shape = tuple(point_group["xyz"].shape)
    rgb_shape = tuple(point_group["rgb"].shape)
    if xyz_shape != expected_shape or rgb_shape != expected_shape:
        raise RuntimeError(
            "Packed point-cloud shape mismatch for "
            + stem
            + ": xyz="
            + str(xyz_shape)
            + " rgb="
            + str(rgb_shape)
            + " expected="
            + str(expected_shape)
        )
    required = [
        output_root / POSE_DIR / (stem + ".npy"),
        output_root / RAW_ACTION_DIR / (stem + ".npy"),
        output_root / RAW_ACTION_FULL_DIR / (stem + ".npy"),
        output_root / TASK_STATE_DIR / (stem + ".npz"),
    ]
    if require_world_base_worldflow_sidecars:
        required.extend(
            [
                output_root / WORLD_BASE_EE_POSE_DIR / (stem + ".npy"),
                output_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR / (stem + ".npy"),
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Packed episode sidecars are missing before artifact cleanup: "
            + ", ".join(missing)
        )
    if require_world_base_worldflow_sidecars:
        for directory_name in (
            WORLD_BASE_EE_POSE_DIR,
            WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
        ):
            pose_path = output_root / directory_name / (stem + ".npy")
            poses = np.load(pose_path, mmap_mode="r")
            if poses.shape != (int(expected_frames), 9) or poses.dtype != np.float32:
                raise RuntimeError(
                    "Packed WorldFlow sidecar shape/dtype mismatch: "
                    + str(pose_path)
                    + " shape="
                    + str(poses.shape)
                    + " dtype="
                    + str(poses.dtype)
                )
            if not np.isfinite(poses).all():
                raise RuntimeError("Packed WorldFlow sidecar contains NaN/Inf: " + str(pose_path))


def pack_artifacts(args, artifact_root, records, expected_episode_count):
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        complete_marker = output_root / "meta" / "rlbench_conversion_complete.json"
        if complete_marker.is_file() and not args.overwrite:
            try:
                with open(complete_marker, "r", encoding="utf-8") as file:
                    complete_meta = json.load(file)
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError(
                    "Could not read the existing RLBench conversion marker: "
                    + str(complete_marker)
                ) from error
            existing_mode = str(complete_meta.get("action_label_mode", "expert_target"))
            if existing_mode != str(args.action_label_mode):
                raise RuntimeError(
                    "Existing dataset uses action_label_mode="
                    + existing_mode
                    + ", requested "
                    + str(args.action_label_mode)
                    + ". Use a different --output-root or pass --overwrite."
                )
            if args.generate_world_base_worldflow_sidecars:
                missing_worldflow = [
                    str(output_root / directory_name)
                    for directory_name in (
                        WORLD_BASE_EE_POSE_DIR,
                        WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
                    )
                    if not (output_root / directory_name / "meta.json").is_file()
                ]
                if missing_worldflow:
                    raise RuntimeError(
                        "Existing complete dataset predates robot-base WorldFlow sidecars: "
                        + ", ".join(missing_worldflow)
                        + ". Run the standalone sidecar backfill tool; the collector will not "
                        "delete a complete dataset implicitly."
                    )
                existing_base_frame = complete_meta.get("world_base_frame_definition")
                if existing_base_frame != RLBENCH_PANDA_LINK0_FRAME_VERSION:
                    raise RuntimeError(
                        "Existing complete dataset uses an incompatible or unversioned "
                        "WorldFlow base frame: "
                        + repr(existing_base_frame)
                        + "; required "
                        + repr(RLBENCH_PANDA_LINK0_FRAME_VERSION)
                        + ". Run the standalone sidecar backfill tool with the Panda-link0 "
                        "transform; the collector will not silently reuse old-base sidecars."
                    )
            print("[skip pack] complete output already exists: " + str(output_root))
            return
        print("[pack] removing an incomplete output before rebuilding: " + str(output_root))
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    t_world_base = None
    if args.generate_world_base_worldflow_sidecars:
        if not records:
            raise RuntimeError("Cannot pack robot-base WorldFlow sidecars without episodes")
        first_record = records[0]
        first_artifact = artifact_root / episode_name(
            first_record["task"], first_record["local_episode_index"]
        )
        with np.load(first_artifact / "arrays.npz") as first_arrays:
            t_world_base = validate_rigid_transform(
                first_arrays["T_world_base"],
                "packed T_world_base",
            )
    dataset = create_dataset(
        output_root,
        args.repo_id,
        args.fps,
        args.image_size,
        t_world_base=t_world_base,
        action_alignment=args.action_alignment,
    )
    worldflow_validation = None
    if args.generate_world_base_worldflow_sidecars:
        worldflow_validation = {
            "episode_count": 0,
            "frame_count": 0,
            "achieved_roundtrip_max_abs": 0.0,
            "action_target_roundtrip_max_abs": 0.0,
            "achieved_rotation_orthogonality_max_abs": 0.0,
            "target_rotation_orthogonality_max_abs": 0.0,
            "achieved_rotation_determinant_min": float("inf"),
            "achieved_rotation_determinant_max": float("-inf"),
            "target_rotation_determinant_min": float("inf"),
            "target_rotation_determinant_max": float("-inf"),
        }
    try:
        for episode_index, record in enumerate(records):
            artifact = artifact_root / episode_name(record["task"], record["local_episode_index"])
            with np.load(artifact / "arrays.npz") as arrays_file:
                arrays = {
                    "actions": arrays_file["actions"],
                    "states": arrays_file["states"],
                    "world_ee_poses": arrays_file["world_ee_poses"],
                    "raw_expert_actions": arrays_file["raw_expert_actions"],
                    "raw_expert_actions_full": arrays_file["raw_expert_actions_full"],
                    "images": arrays_file["images"],
                    "timestamps": arrays_file["timestamps"],
                    "initial_task_state_bytes": arrays_file["initial_task_state_bytes"],
                    "initial_task_state_object_count": arrays_file["initial_task_state_object_count"],
                    "demo_random_seed_state": arrays_file["demo_random_seed_state"],
                    "demo_random_seed_position": arrays_file["demo_random_seed_position"],
                    "demo_random_seed_has_gauss": arrays_file["demo_random_seed_has_gauss"],
                    "demo_random_seed_cached_gaussian": arrays_file["demo_random_seed_cached_gaussian"],
                    "demo_num_reset_attempts": arrays_file["demo_num_reset_attempts"],
                }
                arrays.update(
                    {
                        key: arrays_file[key]
                        for key in OBJECT_STATE_KEYS
                        if key in arrays_file.files
                    }
                )
                arrays.update(
                    {
                        key: arrays_file[key]
                        for key in (
                            "world_base_ee_poses",
                            "world_base_action_target_ee_poses",
                            "T_world_base",
                            "T_base_world",
                        )
                        if key in arrays_file.files
                    }
                )
            if args.generate_world_base_worldflow_sidecars:
                episode_t_world_base = validate_rigid_transform(
                    arrays["T_world_base"],
                    "episode T_world_base",
                )
                if not np.allclose(
                    episode_t_world_base,
                    t_world_base,
                    atol=2e-6,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "RLBench Panda base transform changed across collection workers/episodes"
                    )
                with open(artifact / "record.json", "r", encoding="utf-8") as file:
                    artifact_record = json.load(file)
                metrics = artifact_record.get("worldflow_base_sidecar_validation")
                if not isinstance(metrics, dict):
                    raise RuntimeError(
                        "Artifact is missing WorldFlow sidecar validation: " + str(artifact)
                    )
                worldflow_validation["episode_count"] += 1
                worldflow_validation["frame_count"] += int(metrics["frames"])
                for key in (
                    "achieved_roundtrip_max_abs",
                    "action_target_roundtrip_max_abs",
                    "achieved_rotation_orthogonality_max_abs",
                    "target_rotation_orthogonality_max_abs",
                    "achieved_rotation_determinant_max",
                    "target_rotation_determinant_max",
                ):
                    worldflow_validation[key] = max(
                        float(worldflow_validation[key]), float(metrics[key])
                    )
                for key in (
                    "achieved_rotation_determinant_min",
                    "target_rotation_determinant_min",
                ):
                    worldflow_validation[key] = min(
                        float(worldflow_validation[key]), float(metrics[key])
                    )
            copy_tree_with_hardlinks(
                artifact / "point_clouds.zarr",
                output_root / POINT_DIR / ("episode_" + str(episode_index).zfill(6) + ".zarr"),
            )
            np.save(output_root / POSE_DIR / ("episode_" + str(episode_index).zfill(6) + ".npy"), arrays["world_ee_poses"])
            np.save(output_root / RAW_ACTION_DIR / ("episode_" + str(episode_index).zfill(6) + ".npy"), arrays["raw_expert_actions"])
            np.save(
                output_root / RAW_ACTION_FULL_DIR / ("episode_" + str(episode_index).zfill(6) + ".npy"),
                arrays["raw_expert_actions_full"],
            )
            if args.generate_world_base_worldflow_sidecars:
                stem = "episode_" + str(episode_index).zfill(6) + ".npy"
                np.save(
                    output_root / WORLD_BASE_EE_POSE_DIR / stem,
                    np.ascontiguousarray(arrays["world_base_ee_poses"], dtype=np.float32),
                )
                np.save(
                    output_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR / stem,
                    np.ascontiguousarray(
                        arrays["world_base_action_target_ee_poses"], dtype=np.float32
                    ),
                )
            np.savez(
                output_root / TASK_STATE_DIR / ("episode_" + str(episode_index).zfill(6) + ".npz"),
                configuration_bytes=arrays["initial_task_state_bytes"],
                object_count=arrays["initial_task_state_object_count"],
                demo_random_seed_state=arrays["demo_random_seed_state"],
                demo_random_seed_position=arrays["demo_random_seed_position"],
                demo_random_seed_has_gauss=arrays["demo_random_seed_has_gauss"],
                demo_random_seed_cached_gaussian=arrays["demo_random_seed_cached_gaussian"],
                demo_num_reset_attempts=arrays["demo_num_reset_attempts"],
            )
            object_state_arrays = {
                key: arrays[key] for key in OBJECT_STATE_KEYS if key in arrays
            }
            if object_state_arrays:
                np.savez_compressed(
                    output_root
                    / OBJECT_STATE_DIR
                    / ("episode_" + str(episode_index).zfill(6) + ".npz"),
                    **object_state_arrays,
                )
            for frame_index in range(len(arrays["actions"])):
                write_frame_to_dataset(dataset, record["description"], arrays, frame_index)
            dataset.save_episode()
            print("[pack] episode=" + str(episode_index) + " task=" + record["task"] + " frames=" + str(record["frames"]))
            if args.delete_artifacts_after_pack:
                verify_packed_episode(
                    output_root,
                    episode_index,
                    expected_frames=len(arrays["actions"]),
                    expected_points=args.num_points,
                    require_world_base_worldflow_sidecars=(
                        args.generate_world_base_worldflow_sidecars
                    ),
                )
                shutil.rmtree(artifact)
                print(
                    "[artifact-cleanup] verified packed episode="
                    + str(episode_index)
                    + "; deleted="
                    + str(artifact),
                    flush=True,
                )
    finally:
        dataset.finalize()
    if args.generate_world_base_worldflow_sidecars:
        expected_frames = sum(int(record["frames"]) for record in records)
        if int(worldflow_validation["episode_count"]) != len(records):
            raise RuntimeError("WorldFlow validation episode count does not match packed records")
        if int(worldflow_validation["frame_count"]) != expected_frames:
            raise RuntimeError("WorldFlow validation frame count does not match packed records")
        achieved_files = list((output_root / WORLD_BASE_EE_POSE_DIR).glob("episode_*.npy"))
        target_files = list(
            (output_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR).glob("episode_*.npy")
        )
        if len(achieved_files) != len(records) or len(target_files) != len(records):
            raise RuntimeError(
                "Packed WorldFlow sidecar file count does not match dataset episodes"
            )
        with open(
            output_root / "meta" / "rlbench_worldflow_robot_base_sidecars.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "complete": True,
                    "dataset_root": str(output_root),
                    "action_alignment": str(args.action_alignment),
                    "target_semantics": "commanded action target; never achieved next pose",
                    "transform_source": (
                        RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE
                    ),
                    "base_frame_definition": RLBENCH_PANDA_LINK0_FRAME_VERSION,
                    "T_world_base": np.asarray(t_world_base, dtype=np.float64).tolist(),
                    "T_base_world": np.linalg.inv(
                        np.asarray(t_world_base, dtype=np.float64)
                    ).tolist(),
                    "validation": worldflow_validation,
                },
                file,
                indent=2,
            )
    with open(output_root / "meta" / "rlbench_conversion.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "source": "RLBench live expert policy",
                "expert_action_mode": "MoveArmThenGripper(JointVelocity, Discrete)",
                "main_action_semantics": action_semantics(args.action_label_mode),
                "action_alignment": str(args.action_alignment),
                "action_label_mode": str(args.action_label_mode),
                "observation_state_semantics": "achieved EEF pose relative to episode EEF0",
                "world_base_worldflow_sidecars": bool(
                    args.generate_world_base_worldflow_sidecars
                ),
                "world_base_frame_definition": (
                    RLBENCH_PANDA_LINK0_FRAME_VERSION
                    if args.generate_world_base_worldflow_sidecars
                    else None
                ),
                "world_base_ee_pose_semantics": (
                    "T_base_ee = inverse(T_world_base) @ T_world_ee"
                    if args.generate_world_base_worldflow_sidecars
                    else None
                ),
                "world_base_action_target_semantics": (
                    "commanded expert target: inverse(T_world_base) @ "
                    "T_world_eef0 @ T_eef0_target"
                    if args.generate_world_base_worldflow_sidecars
                    else None
                ),
                "T_world_base": (
                    np.asarray(t_world_base, dtype=np.float64).tolist()
                    if t_world_base is not None
                    else None
                ),
                "T_base_world": (
                    np.linalg.inv(np.asarray(t_world_base, dtype=np.float64)).tolist()
                    if t_world_base is not None
                    else None
                ),
                "point_cloud_semantics": "finite front-camera world cloud plus selected virtual gripper template transformed to current EEF",
                "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
                "gripper_template": (
                    LIBERO_GRIPPER_TEMPLATE_VERSION
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
                ),
                "gripper_template_name": str(args.gripper_template),
                "gripper_template_version": (
                    LIBERO_GRIPPER_TEMPLATE_VERSION
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
                ),
                "virtual_gripper_width_normalization_max_m": (
                    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else float(args.gripper_max_width)
                ),
                "virtual_gripper_geometry_max_width_m": (
                    LIBERO_REAP_TEMPLATE_MAX_WIDTH
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else RLBENCH_PANDA_MAX_WIDTH
                ),
                "virtual_gripper_opening_max_width_m": (
                    LIBERO_REAP_OPENING_MAX_WIDTH
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else float(args.gripper_max_width)
                ),
                "virtual_gripper_local_offset_m": (
                    [0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN]
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else [0.0, 0.0, 0.0]
                ),
                "initial_task_state_semantics": "RLBench task configuration tree at reset_to_demo first frame",
                "initial_object_state_semantics": "Readable per-object world state at reset_to_demo first frame",
                "collection_workers": int(args.collection_workers),
                "expert_path_mode": str(args.expert_path_mode),
                "replay_random_seeds_from_artifacts": (
                    None
                    if args.replay_random_seeds_from_artifacts is None
                    else str(
                        Path(args.replay_random_seeds_from_artifacts)
                        .expanduser()
                        .resolve()
                    )
                ),
                "collection_seed": (
                    None if args.collection_seed is None else int(args.collection_seed)
                ),
                "phone_base_max_initial_distance_m": (
                    None
                    if args.phone_base_max_initial_distance_m is None
                    else float(args.phone_base_max_initial_distance_m)
                ),
                "phone_eef_max_initial_distance_m": (
                    None
                    if args.phone_eef_max_initial_distance_m is None
                    else float(args.phone_eef_max_initial_distance_m)
                ),
                "replay_scenes_from_artifacts": (
                    None
                    if args.replay_scenes_from_artifacts is None
                    else str(Path(args.replay_scenes_from_artifacts).expanduser().resolve())
                ),
                "source_artifacts_retained": not bool(args.delete_artifacts_after_pack),
                "artifact_cleanup_mode": (
                    "delete_each_episode_after_verified_pack"
                    if args.delete_artifacts_after_pack
                    else "retain"
                ),
                "episode_count": len(records),
                "tasks": sorted(set(item["task"] for item in records)),
            },
            file,
            indent=2,
        )
    if len(records) == expected_episode_count:
        with open(output_root / "meta" / "rlbench_conversion_complete.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "complete": True,
                    "episode_count": len(records),
                    "action_label_mode": str(args.action_label_mode),
                    "world_base_worldflow_sidecars": bool(
                        args.generate_world_base_worldflow_sidecars
                    ),
                    "world_base_frame_definition": (
                        RLBENCH_PANDA_LINK0_FRAME_VERSION
                        if args.generate_world_base_worldflow_sidecars
                        else None
                    ),
                },
                file,
                indent=2,
            )


def collection_config_signature(args, tasks):
    template_version = (
        LIBERO_GRIPPER_TEMPLATE_VERSION
        if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
        else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
    )
    return {
        "tasks": list(tasks),
        "episodes_per_task": int(args.episodes_per_task),
        "episode_start": int(getattr(args, "episode_start", 0)),
        "variation": int(args.variation),
        "collection_seed": (
            None if args.collection_seed is None else int(args.collection_seed)
        ),
        "num_points": int(args.num_points),
        "gripper_points": int(args.gripper_points),
        "gripper_max_width": float(args.gripper_max_width),
        "gripper_template": str(args.gripper_template),
        "gripper_template_version": template_version,
        "virtual_gripper_width_normalization_max_m": (
            LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else float(args.gripper_max_width)
        ),
        "virtual_gripper_geometry_max_width_m": (
            LIBERO_REAP_TEMPLATE_MAX_WIDTH
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else RLBENCH_PANDA_MAX_WIDTH
        ),
        "virtual_gripper_opening_max_width_m": (
            LIBERO_REAP_OPENING_MAX_WIDTH
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else float(args.gripper_max_width)
        ),
        "virtual_gripper_len_m": (
            LIBERO_REAP_GRIPPER_LEN
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else 0.0
        ),
        "image_size": int(args.image_size),
        "fps": int(args.fps),
        "collection_workers": int(args.collection_workers),
        "expert_path_mode": str(args.expert_path_mode),
        "segmented_linear_segments": int(args.segmented_linear_segments),
        "replay_random_seeds_from_artifacts": (
            None
            if args.replay_random_seeds_from_artifacts is None
            else str(Path(args.replay_random_seeds_from_artifacts).expanduser().resolve())
        ),
        "phone_base_max_initial_distance_m": (
            None
            if args.phone_base_max_initial_distance_m is None
            else float(args.phone_base_max_initial_distance_m)
        ),
        "phone_eef_max_initial_distance_m": (
            None
            if args.phone_eef_max_initial_distance_m is None
            else float(args.phone_eef_max_initial_distance_m)
        ),
        "replay_scenes_from_artifacts": (
            None
            if args.replay_scenes_from_artifacts is None
            else str(Path(args.replay_scenes_from_artifacts).expanduser().resolve())
        ),
        "action_semantics_version": action_semantics_version(args.action_label_mode),
        "action_label_mode": str(args.action_label_mode),
        "action_alignment": str(args.action_alignment),
        "generate_world_base_worldflow_sidecars": bool(
            args.generate_world_base_worldflow_sidecars
        ),
        "world_base_frame_definition": (
            RLBENCH_PANDA_LINK0_FRAME_VERSION
            if args.generate_world_base_worldflow_sidecars
            else None
        ),
        "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
    }


def collect(args, tasks, artifact_root):
    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.waypoints import Point
    from pyrep.const import ConfigurationPathAlgorithms as Algos
    from pyrep.errors import ConfigurationPathError

    if args.collection_seed is not None:
        effective_collection_seed = int(args.collection_seed) + int(
            getattr(args, "episode_start", 0)
        )
        np.random.seed(effective_collection_seed)
        print(
            "[collection-seed] base="
            + str(args.collection_seed)
            + " effective="
            + str(effective_collection_seed),
            flush=True,
        )

    original_point_get_path = Point.get_path
    linear_path_calls = 0
    rrt_path_calls = 0
    if args.expert_path_mode == "linear_only":
        def linear_only_point_get_path(point, ignore_collisions=False):
            nonlocal linear_path_calls
            linear_path_calls += 1
            arm = point._robot.arm
            return arm.get_linear_path(
                point._waypoint.get_position(),
                euler=point._waypoint.get_orientation(),
                ignore_collisions=(point._ignore_collisions or ignore_collisions),
            )

        Point.get_path = linear_only_point_get_path
    elif args.expert_path_mode in (
        "segmented_linear",
        "segmented_linear_then_rrt",
    ):
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )
        from scipy.spatial.transform import Rotation, Slerp

        def segmented_linear_point_get_path(point, ignore_collisions=False):
            nonlocal linear_path_calls, rrt_path_calls
            arm = point._robot.arm
            segment_count = int(args.segmented_linear_segments)
            if segment_count <= 0:
                raise ValueError("--segmented-linear-segments must be positive")
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_position = np.asarray(arm.get_tip().get_position(), dtype=np.float64)
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            rotations = Rotation.from_quat(
                np.stack((start_quaternion, target_quaternion), axis=0)
            )
            slerp = Slerp([0.0, 1.0], rotations)
            fractions = np.linspace(0.0, 1.0, segment_count + 1)[1:]
            segment_quaternions = slerp(fractions).as_quat()
            segment_steps = max(2, int(math.ceil(50.0 / segment_count)))
            combined = []
            try:
                for segment_index, (fraction, quaternion) in enumerate(
                    zip(fractions, segment_quaternions)
                ):
                    position = (
                        start_position
                        + float(fraction) * (target_position - start_position)
                    )
                    linear_path_calls += 1
                    try:
                        path = arm.get_linear_path(
                            position,
                            quaternion=quaternion,
                            steps=segment_steps,
                            ignore_collisions=(
                                point._ignore_collisions or ignore_collisions
                            ),
                        )
                    except ConfigurationPathError:
                        if args.expert_path_mode == "segmented_linear":
                            raise
                        rrt_path_calls += 1
                        path = arm.get_nonlinear_path(
                            position,
                            quaternion=quaternion,
                            ignore_collisions=(
                                point._ignore_collisions or ignore_collisions
                            ),
                            trials=100,
                            max_configs=10,
                            trials_per_goal=10,
                            algorithm=Algos.RRTConnect,
                        )
                    points = np.asarray(path._path_points, dtype=np.float64).reshape(
                        -1, int(arm.get_joint_count())
                    )
                    if segment_index > 0 and len(points):
                        points = points[1:]
                    combined.append(points)
                    path.set_to_end(disable_dynamics=True)
            finally:
                arm.set_joint_positions(start_joints.tolist(), disable_dynamics=True)
            if not combined or not any(len(points) for points in combined):
                raise RuntimeError("Segmented linear IK returned an empty path")
            return ArmConfigurationPath(
                arm, np.concatenate(combined, axis=0).reshape(-1)
            )

        Point.get_path = segmented_linear_point_get_path
    elif args.expert_path_mode == "rrt_only":
        def rrt_only_point_get_path(point, ignore_collisions=False):
            nonlocal rrt_path_calls
            rrt_path_calls += 1
            arm = point._robot.arm
            return arm.get_nonlinear_path(
                point._waypoint.get_position(),
                euler=point._waypoint.get_orientation(),
                ignore_collisions=(point._ignore_collisions or ignore_collisions),
                trials=100,
                max_configs=10,
                trials_per_goal=10,
                algorithm=Algos.RRTConnect,
            )

        Point.get_path = rrt_only_point_get_path
    print(
        "[expert-path-mode] mode="
        + str(args.expert_path_mode)
        + " ordinary_point_planner="
        + {
            "linear_only": "linear_ik_only",
            "segmented_linear": (
                "segmented_linear_ik_"
                + str(int(args.segmented_linear_segments))
                + "x"
            ),
            "segmented_linear_then_rrt": (
                "segmented_linear_ik_then_short_rrtconnect_"
                + str(int(args.segmented_linear_segments))
                + "x"
            ),
            "rrt_only": "forced_rrtconnect",
            "linear_then_rrt": "linear_ik_then_rrtconnect_fallback",
        }[str(args.expert_path_mode)]
        + " predefined_cartesian_paths=preserved",
        flush=True,
    )

    manifest_path = artifact_root / "manifest.json"
    artifact_root.mkdir(parents=True, exist_ok=True)
    config_signature = collection_config_signature(args, tasks)
    if args.no_resume and manifest_path.exists():
        manifest_path.unlink()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("config") != config_signature:
            raise RuntimeError(
                "Resume parameters differ from the artifact manifest. "
                "Use the original parameters or pass --no-resume to recollect."
            )
    else:
        manifest = {"config": config_signature, "records": []}
    records = manifest["records"]
    known = set(item["task"] + ":" + str(item["local_episode_index"]) for item in records)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=make_observation_config(args.image_size),
        headless=True,
        static_positions=False,
    )
    env.launch()
    try:
        fk_model = read_panda_fk_model(env)
        t_world_base = rlbench_panda_link0_to_world_matrix(env._robot.arm)
        print(
            "[worldflow-base] frame="
            + RLBENCH_PANDA_LINK0_FRAME_VERSION
            + " T_world_base="
            + json.dumps(t_world_base.tolist(), separators=(",", ":")),
            flush=True,
        )
        for task_name in tasks:
            task_class = task_class_from_name(task_name)
            task_env = env.get_task(task_class)
            task_env.set_variation(args.variation)
            completed_indices = {
                int(item["local_episode_index"])
                for item in records
                if item["task"] == task_name
                and artifact_is_complete(
                    artifact_root / episode_name(task_name, int(item["local_episode_index"])),
                    require_world_base_worldflow_sidecars=(
                        args.generate_world_base_worldflow_sidecars
                    ),
                )
            }
            successful_count = len(completed_indices)
            local_index = int(getattr(args, "episode_start", 0))
            while successful_count < args.episodes_per_task:
                key = task_name + ":" + str(local_index)
                artifact = artifact_root / episode_name(task_name, local_index)
                if key in known and artifact_is_complete(
                    artifact,
                    require_world_base_worldflow_sidecars=(
                        args.generate_world_base_worldflow_sidecars
                    ),
                ):
                    print("[skip] task=" + task_name + " episode=" + str(local_index))
                    local_index += 1
                    continue
                if key in known:
                    records[:] = [
                        item
                        for item in records
                        if item["task"] + ":" + str(item["local_episode_index"]) != key
                    ]
                    known.remove(key)
                success = False
                last_error = None
                for attempt in range(args.max_demo_attempts):
                    try:
                        linear_calls_before = int(linear_path_calls)
                        rrt_calls_before = int(rrt_path_calls)
                        replay_seed_source = None
                        replay_scene_source = None
                        replay_scene_arrays = None
                        if args.replay_scenes_from_artifacts is not None:
                            replay_scene_source = (
                                Path(args.replay_scenes_from_artifacts)
                                .expanduser()
                                .resolve()
                                / episode_name(task_name, local_index)
                                / "arrays.npz"
                            )
                            if not replay_scene_source.is_file():
                                raise FileNotFoundError(
                                    "Matched scene artifact is missing: "
                                    + str(replay_scene_source)
                                )
                            required_scene_keys = (
                                "initial_task_state_bytes",
                                "initial_task_state_object_count",
                                "initial_object_names",
                                "initial_object_poses",
                                "initial_object_joint_positions",
                                "initial_object_joint_target_positions",
                                "initial_object_joint_target_velocities",
                                "demo_random_seed_state",
                                "demo_random_seed_position",
                                "demo_random_seed_has_gauss",
                                "demo_random_seed_cached_gaussian",
                                "demo_num_reset_attempts",
                            )
                            with np.load(replay_scene_source, allow_pickle=False) as source:
                                missing_scene_keys = [
                                    key for key in required_scene_keys if key not in source.files
                                ]
                                if missing_scene_keys:
                                    raise RuntimeError(
                                        "Matched scene artifact lacks required fields: "
                                        + ", ".join(missing_scene_keys)
                                    )
                                replay_scene_arrays = {
                                    key: np.asarray(source[key]).copy()
                                    for key in required_scene_keys
                                }
                            (
                                descriptions,
                                reset_observation,
                                replay_random_state,
                                replay_reset_attempts,
                            ) = restore_task_environment_from_artifact_arrays(
                                task_env, replay_scene_arrays
                            )
                            demo = collect_live_demo_from_current_scene(task_env)
                            demo.random_seed = replay_random_state
                            demo.num_reset_attempts = replay_reset_attempts
                            # Return to the exact matched starting state for sidecars,
                            # reset-image validation, and initial-distance reporting.
                            descriptions, reset_observation, _, _ = (
                                restore_task_environment_from_artifact_arrays(
                                    task_env, replay_scene_arrays
                                )
                            )
                            print(
                                "[matched-scene-restored] task="
                                + task_name
                                + " episode="
                                + str(local_index)
                                + " source="
                                + str(replay_scene_source),
                                flush=True,
                            )
                        elif (
                            task_name == "phone_on_base"
                            and args.phone_eef_max_initial_distance_m is not None
                        ):
                            candidate_random_state = np.random.get_state()
                            descriptions, candidate_observation = task_env.reset()
                            candidate_phone_position = np.asarray(
                                task_env._task.phone.get_position(), dtype=np.float64
                            )
                            candidate_eef_position = np.asarray(
                                candidate_observation.gripper_pose[:3], dtype=np.float64
                            )
                            candidate_distance = float(
                                np.linalg.norm(
                                    candidate_phone_position - candidate_eef_position
                                )
                            )
                            if candidate_distance > float(
                                args.phone_eef_max_initial_distance_m
                            ):
                                print(
                                    "[scene-reject] task=phone_on_base episode="
                                    + str(local_index)
                                    + " attempt="
                                    + str(attempt + 1)
                                    + " phone_to_eef_m="
                                    + format(candidate_distance, ".6f")
                                    + " threshold_m="
                                    + format(
                                        float(args.phone_eef_max_initial_distance_m),
                                        ".6f",
                                    ),
                                    flush=True,
                                )
                                continue
                            demo = collect_live_demo_from_current_scene(task_env)
                            demo.random_seed = candidate_random_state
                            descriptions, reset_observation = task_env.reset_to_demo(demo)
                        else:
                            if args.replay_random_seeds_from_artifacts is not None:
                                replay_seed_source = (
                                    Path(args.replay_random_seeds_from_artifacts)
                                    .expanduser()
                                    .resolve()
                                    / episode_name(task_name, local_index)
                                    / "arrays.npz"
                                )
                                if not replay_seed_source.is_file():
                                    raise FileNotFoundError(
                                        "Matched-scene seed artifact is missing: "
                                        + str(replay_seed_source)
                                    )
                                with np.load(replay_seed_source, allow_pickle=False) as source:
                                    required_seed_keys = (
                                        "demo_random_seed_state",
                                        "demo_random_seed_position",
                                        "demo_random_seed_has_gauss",
                                        "demo_random_seed_cached_gaussian",
                                    )
                                    missing_seed_keys = [
                                        key for key in required_seed_keys if key not in source.files
                                    ]
                                    if missing_seed_keys:
                                        raise RuntimeError(
                                            "Matched-scene artifact lacks saved RNG fields: "
                                            + ", ".join(missing_seed_keys)
                                        )
                                    np.random.set_state(
                                        (
                                            "MT19937",
                                            np.asarray(
                                                source["demo_random_seed_state"],
                                                dtype=np.uint32,
                                            ).copy(),
                                            int(source["demo_random_seed_position"]),
                                            int(source["demo_random_seed_has_gauss"]),
                                            float(source["demo_random_seed_cached_gaussian"]),
                                        )
                                    )
                                print(
                                    "[matched-scene-seed] task="
                                    + task_name
                                    + " episode="
                                    + str(local_index)
                                    + " source="
                                    + str(replay_seed_source),
                                    flush=True,
                                )
                            # The surrounding loop owns the retry budget. Keeping RLBench's
                            # internal budget at one avoids multiplying retries by 10x.
                            demos = task_env.get_demos(1, live_demos=True, max_attempts=1)
                            demo = demos[0]
                            descriptions, reset_observation = task_env.reset_to_demo(demo)
                        initial_phone_to_base_sensor_distance_m = None
                        initial_phone_to_eef_distance_m = None
                        if task_name == "phone_on_base":
                            phone_position = np.asarray(
                                task_env._task.phone.get_position(), dtype=np.float64
                            )
                            base_sensor_position = np.asarray(
                                task_env._task.success_sensor.get_position(), dtype=np.float64
                            )
                            initial_phone_to_base_sensor_distance_m = float(
                                np.linalg.norm(phone_position - base_sensor_position)
                            )
                            initial_phone_to_eef_distance_m = float(
                                np.linalg.norm(
                                    phone_position
                                    - np.asarray(
                                        reset_observation.gripper_pose[:3],
                                        dtype=np.float64,
                                    )
                                )
                            )
                            if (
                                args.phone_base_max_initial_distance_m is not None
                                and initial_phone_to_base_sensor_distance_m
                                > float(args.phone_base_max_initial_distance_m)
                            ):
                                raise RuntimeError(
                                    "phone_on_base initial distance rejected: "
                                    + str(initial_phone_to_base_sensor_distance_m)
                                    + " > "
                                    + str(args.phone_base_max_initial_distance_m)
                                )
                            if (
                                args.phone_eef_max_initial_distance_m is not None
                                and initial_phone_to_eef_distance_m
                                > float(args.phone_eef_max_initial_distance_m) + 1e-5
                            ):
                                raise RuntimeError(
                                    "phone_on_base restored phone-to-EEF distance changed: "
                                    + str(initial_phone_to_eef_distance_m)
                                    + " > "
                                    + str(args.phone_eef_max_initial_distance_m)
                                )
                        configuration_tree, object_count = task_env._task.get_state()
                        configuration_bytes = configuration_tree_to_bytes(
                            configuration_tree
                        )
                        object_states = capture_initial_object_states(task_env._task)
                        if len(object_states["initial_object_names"]) != int(object_count):
                            raise RuntimeError(
                                "Task state and per-object state counts differ: "
                                + str(object_count)
                                + " versus "
                                + str(len(object_states["initial_object_names"]))
                            )
                        description = descriptions[0] if descriptions else task_name.replace("_", " ")
                        arrays = make_episode_arrays(
                            demo,
                            args.num_points,
                            args.gripper_points,
                            args.gripper_template,
                            args.gripper_max_width,
                            args.fps,
                            seed=local_index * 100000 + attempt * 1000,
                            fk_model=fk_model,
                            t_world_base=t_world_base,
                            generate_world_base_worldflow_sidecars=(
                                args.generate_world_base_worldflow_sidecars
                            ),
                            action_alignment=args.action_alignment,
                            action_label_mode=args.action_label_mode,
                        )
                        validate_captured_episode(
                            arrays, task_name, local_index, args.gripper_points
                        )
                        reset_image = np.asarray(reset_observation.front_rgb, dtype=np.uint8)
                        demo_first_image = np.asarray(demo[0].front_rgb, dtype=np.uint8)
                        if reset_image.shape != demo_first_image.shape:
                            raise RuntimeError(
                                "reset_to_demo image shape does not match demo frame 0: "
                                + str(reset_image.shape)
                                + " versus "
                                + str(demo_first_image.shape)
                            )
                        arrays["reset_first_rgb_mae"] = np.float32(
                            np.mean(
                                np.abs(
                                    reset_image.astype(np.float32)
                                    - demo_first_image.astype(np.float32)
                                )
                            )
                        )
                        arrays["initial_task_state_bytes"] = np.frombuffer(
                            configuration_bytes, dtype=np.uint8
                        ).copy()
                        arrays["initial_task_state_object_count"] = np.int64(object_count)
                        arrays.update(object_states)
                        random_seed = demo.random_seed
                        arrays["demo_random_seed_state"] = np.asarray(
                            random_seed[1], dtype=np.uint32
                        )
                        arrays["demo_random_seed_position"] = np.int64(random_seed[2])
                        arrays["demo_random_seed_has_gauss"] = np.int64(random_seed[3])
                        arrays["demo_random_seed_cached_gaussian"] = np.float64(random_seed[4])
                        arrays["demo_num_reset_attempts"] = np.int64(demo.num_reset_attempts)
                        arrays["expert_path_mode"] = np.asarray(str(args.expert_path_mode))
                        arrays["expert_linear_path_calls"] = np.int64(
                            int(linear_path_calls) - linear_calls_before
                        )
                        arrays["expert_rrt_path_calls"] = np.int64(
                            int(rrt_path_calls) - rrt_calls_before
                        )
                        if initial_phone_to_base_sensor_distance_m is not None:
                            arrays["initial_phone_to_base_sensor_distance_m"] = np.float64(
                                initial_phone_to_base_sensor_distance_m
                            )
                        if initial_phone_to_eef_distance_m is not None:
                            arrays["initial_phone_to_eef_distance_m"] = np.float64(
                                initial_phone_to_eef_distance_m
                            )
                        if replay_scene_source is not None:
                            arrays["matched_scene_source"] = np.asarray(
                                str(replay_scene_source)
                            )
                        save_artifact(artifact, task_name, local_index, arrays, description, args.variation)
                        record = {
                            "task": task_name,
                            "local_episode_index": local_index,
                            "description": str(description),
                            "variation": args.variation,
                            "frames": int(len(arrays["actions"])),
                            "fk_target_vs_achieved_position_error_median_m": (
                                float(np.median(arrays["fk_position_errors"]))
                                if len(arrays["fk_position_errors"])
                                else 0.0
                            ),
                            "fk_target_vs_achieved_position_error_max_m": (
                                float(np.max(arrays["fk_position_errors"]))
                                if len(arrays["fk_position_errors"])
                                else 0.0
                            ),
                            "action_label_mode": str(args.action_label_mode),
                            "expert_path_mode": str(args.expert_path_mode),
                            "matched_scene_seed_source": (
                                None if replay_seed_source is None else str(replay_seed_source)
                            ),
                            "matched_scene_source": (
                                None if replay_scene_source is None else str(replay_scene_source)
                            ),
                            "expert_linear_path_calls": int(
                                int(linear_path_calls) - linear_calls_before
                            ),
                            "expert_rrt_path_calls": int(
                                int(rrt_path_calls) - rrt_calls_before
                            ),
                        }
                        if initial_phone_to_base_sensor_distance_m is not None:
                            record["initial_phone_to_base_sensor_distance_m"] = float(
                                initial_phone_to_base_sensor_distance_m
                            )
                        if initial_phone_to_eef_distance_m is not None:
                            record["initial_phone_to_eef_distance_m"] = float(
                                initial_phone_to_eef_distance_m
                            )
                        records.append(record)
                        known.add(key)
                        with open(manifest_path, "w", encoding="utf-8") as file:
                            json.dump({"config": config_signature, "records": records}, file, indent=2)
                        print("[ok] task=" + task_name + " episode=" + str(local_index) + " frames=" + str(record["frames"]))
                        success = True
                        successful_count += 1
                        local_index += 1
                        break
                    except Exception as error:
                        last_error = error
                        print(
                            "[retry] task="
                            + task_name
                            + " episode="
                            + str(local_index)
                            + " error="
                            + repr(error)
                            + "\n"
                            + traceback.format_exc(),
                            flush=True,
                        )
                if not success:
                    print(
                        "[failed] task="
                        + task_name
                        + " episode="
                        + str(local_index)
                        + " collected="
                        + str(successful_count)
                        + "/"
                        + str(args.episodes_per_task)
                        + " retrying error="
                        + repr(last_error)
                    )
                    if args.abort_on_episode_failure:
                        raise RuntimeError(
                            "Episode failed after "
                            + str(args.max_demo_attempts)
                            + " attempt(s): task="
                            + task_name
                            + " episode="
                            + str(local_index)
                        ) from last_error
    finally:
        Point.get_path = original_point_get_path
        env.shutdown()
    records.sort(key=lambda item: (tasks.index(item["task"]), item["local_episode_index"]))
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump({"config": config_signature, "records": records}, file, indent=2)
    return records


def collection_worker_entry(worker_id, tasks, args, artifact_root, display):
    """Run one isolated RLBench process on a task shard."""
    os.environ["DISPLAY"] = str(display)
    worker_root = Path(artifact_root) / "_workers" / ("worker_" + str(worker_id).zfill(2))
    print(
        "[collection-worker] id="
        + str(worker_id)
        + " display="
        + str(display)
        + " tasks="
        + ",".join(tasks),
        flush=True,
    )
    records = collect(args, tasks, worker_root)
    return {
        "worker_id": int(worker_id),
        "artifact_root": str(worker_root),
        "records": records,
    }


def collection_display_list(args, worker_count):
    if worker_count <= 1:
        return [os.environ.get("DISPLAY", ":99")]
    if args.collection_display_base is not None:
        base = int(args.collection_display_base)
    else:
        current_display = os.environ.get("DISPLAY", "")
        if not current_display.startswith(":"):
            raise RuntimeError(
                "Parallel RLBench collection needs a local DISPLAY such as :99 "
                "or an explicit --collection-display-base."
            )
        base = int(current_display[1:].split(".", 1)[0])
    return [":" + str(base + worker_id) for worker_id in range(worker_count)]


def validate_collection_displays(displays):
    for display in displays:
        if not str(display).startswith(":"):
            continue
        display_number = str(display)[1:].split(".", 1)[0]
        x_socket = Path("/tmp/.X11-unix") / ("X" + display_number)
        if not x_socket.exists():
            raise RuntimeError(
                "RLBench collection worker needs a running X server for "
                + str(display)
                + "; missing "
                + str(x_socket)
                + ". Start one X server per worker before collection."
            )
        xdpyinfo = shutil.which("xdpyinfo")
        if xdpyinfo is not None:
            probe = subprocess.run(
                [xdpyinfo, "-display", str(display)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if probe.returncode != 0:
                error = probe.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    "RLBench collection worker cannot connect to "
                    + str(display)
                    + "; xdpyinfo failed: "
                    + error
                )


def collect_parallel(args, tasks, artifact_root):
    """Collect task shards in spawned processes and merge their artifacts."""
    single_task_episode_parallel = len(tasks) == 1 and int(args.collection_workers) > 1
    if single_task_episode_parallel:
        worker_count = min(int(args.collection_workers), int(args.episodes_per_task))
    else:
        worker_count = min(int(args.collection_workers), len(tasks))
    if worker_count <= 1:
        return collect(args, tasks, artifact_root)

    displays = collection_display_list(args, worker_count)
    validate_collection_displays(displays)
    if single_task_episode_parallel:
        task_shards = [[tasks[0]] for _ in range(worker_count)]
        base, remainder = divmod(int(args.episodes_per_task), worker_count)
        episode_ranges = []
        start = 0
        for worker_id in range(worker_count):
            count = base + (1 if worker_id < remainder else 0)
            episode_ranges.append((start, count))
            start += count
    else:
        task_shards = [[] for _ in range(worker_count)]
        for task_index, task_name in enumerate(tasks):
            task_shards[task_index % worker_count].append(task_name)
        task_shards = [shard for shard in task_shards if shard]
        episode_ranges = [(0, int(args.episodes_per_task)) for _ in task_shards]
    artifact_root.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for worker_id, shard in enumerate(task_shards):
        worker_namespace = argparse.Namespace(**vars(args))
        episode_start, episode_count = episode_ranges[worker_id]
        worker_namespace.episode_start = int(episode_start)
        worker_namespace.episodes_per_task = int(episode_count)
        worker_args.append(
            (worker_id, shard, worker_namespace, artifact_root, displays[worker_id])
        )
    context = mp.get_context("spawn")
    print(
        "[collection] starting "
        + str(len(worker_args))
        + " workers on displays "
        + ",".join(displays),
        flush=True,
    )
    with context.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(collection_worker_entry, worker_args)

    records = []
    for result in results:
        worker_root = Path(result["artifact_root"])
        for record in result["records"]:
            source = worker_root / episode_name(
                record["task"], int(record["local_episode_index"])
            )
            destination = artifact_root / source.name
            if not artifact_is_complete(
                source,
                require_world_base_worldflow_sidecars=(
                    args.generate_world_base_worldflow_sidecars
                ),
            ):
                raise RuntimeError("Worker artifact is incomplete: " + str(source))
            if destination.exists():
                shutil.rmtree(destination)
            copy_tree_with_hardlinks(source, destination)
            records.append(record)

    records.sort(key=lambda item: (tasks.index(item["task"]), item["local_episode_index"]))
    with open(artifact_root / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(
            {"config": collection_config_signature(args, tasks), "records": records},
            file,
            indent=2,
        )
    return records


def run_pointseg_cache(args):
    if args.skip_pointseg_cache:
        print("[cache] skipped by --skip-pointseg-cache")
        return None

    expected_template_version = (
        LIBERO_GRIPPER_TEMPLATE_VERSION
        if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
        else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
    )
    dataset_root = args.output_root.expanduser().resolve()
    cache_root = (
        args.cache_output_dir.expanduser().resolve()
        if args.cache_output_dir is not None
        else dataset_root.parent / (dataset_root.name + "_pointseg_cache")
    )
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file() and not args.overwrite_cache:
        pipeline_metadata_path = dataset_root / "meta" / "rlbench_collection_pipeline.json"
        with open(dataset_root / "meta" / "info.json", "r", encoding="utf-8") as file:
            dataset_info = json.load(file)
        with open(manifest_path, "r", encoding="utf-8") as file:
            cache_manifest = json.load(file)
        requested_pseudo_config = {
            "motion_rotation_radius": args.motion_rotation_radius,
            "motion_baseline_threshold": args.motion_baseline_threshold,
            "motion_baseline_temperature": args.motion_baseline_temperature,
            "motion_relative_margin": args.motion_relative_margin,
            "motion_relative_tau": args.motion_relative_tau,
            "trajectory_sigma": args.trajectory_sigma,
            "contact_radius": args.contact_radius,
            "contact_temperature": args.contact_temperature,
            "approach_margin": args.approach_margin,
            "approach_tau": args.approach_tau,
            "background_trajectory_sigma": args.background_trajectory_sigma,
        }
        cached_pseudo_config = cache_manifest.get("pseudo_label_config", {})
        pseudo_config_matches = all(
            key in cached_pseudo_config
            and np.isclose(float(cached_pseudo_config[key]), float(value), rtol=0.0, atol=1e-8)
            for key, value in requested_pseudo_config.items()
        )
        pipeline_metadata = {}
        if pipeline_metadata_path.is_file():
            with open(pipeline_metadata_path, "r", encoding="utf-8") as file:
                pipeline_metadata = json.load(file)
        frame_count_matches = int(cache_manifest.get("num_samples", -1)) == int(
            dataset_info["total_frames"]
        )
        template_matches = pipeline_metadata.get("gripper_template") == expected_template_version
        if frame_count_matches and template_matches and pseudo_config_matches:
            print("[cache] complete cache already exists: " + str(cache_root))
            return cache_root
        if not pseudo_config_matches:
            raise RuntimeError(
                "PointSeg cache pseudo-label parameters do not match the requested values. "
                "Pass --overwrite-cache to rebuild: " + str(cache_root)
            )
        raise RuntimeError(
            "PointSeg cache frame count or gripper-template version does not match this dataset. "
            "Pass --overwrite-cache to rebuild: " + str(cache_root)
        )

    cache_current_points = (
        args.num_points if args.cache_current_points is None else args.cache_current_points
    )
    cache_future_points = (
        cache_current_points if args.cache_future_points is None else args.cache_future_points
    )
    if cache_current_points <= 0 or cache_future_points <= 0:
        raise ValueError("--cache-current-points and --cache-future-points must be positive")

    cache_python = args.cache_python.expanduser().resolve()
    if not cache_python.is_file():
        raise FileNotFoundError("PointSeg cache Python does not exist: " + str(cache_python))
    # Keep the original RLBench cache entrypoint so preview PLY files retain
    # the original continuous heatmap colors.
    cache_script = Path(__file__).resolve().parent / "RE_rlbench_cache_pointseg_samples.py"
    command = [
        str(cache_python),
        str(cache_script),
        "--dataset.repo_id=" + str(dataset_root),
        "--point-cloud-dir=" + str(dataset_root / POINT_DIR),
        "--output-dir=" + str(cache_root),
        "--current-points=" + str(cache_current_points),
        "--future-points=" + str(cache_future_points),
        "--batch-size=" + str(args.cache_batch_size),
        "--num-workers=" + str(args.cache_num_workers),
        "--device=" + str(args.cache_device),
        "--vis-count=" + str(args.cache_vis_count),
        "--motion-rotation-radius=" + str(args.motion_rotation_radius),
        "--motion-baseline-threshold=" + str(args.motion_baseline_threshold),
        "--motion-baseline-temperature=" + str(args.motion_baseline_temperature),
        "--motion-relative-margin=" + str(args.motion_relative_margin),
        "--motion-relative-tau=" + str(args.motion_relative_tau),
        "--trajectory-sigma=" + str(args.trajectory_sigma),
        "--contact-radius=" + str(args.contact_radius),
        "--contact-temperature=" + str(args.contact_temperature),
        "--approach-margin=" + str(args.approach_margin),
        "--approach-tau=" + str(args.approach_tau),
        "--background-trajectory-sigma=" + str(args.background_trajectory_sigma),
    ]
    if args.cache_vis_one_episode_per_task:
        command.append("--vis-one-episode-per-task")
    if args.overwrite_cache:
        command.append("--overwrite")
    print("[cache] starting: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    with open(dataset_root / "meta" / "rlbench_collection_pipeline.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "complete": True,
                "dataset_root": str(dataset_root),
                "pointseg_cache": str(cache_root),
                "raw_point_count": int(args.num_points),
                "cache_current_points": int(cache_current_points),
                "cache_future_points": int(cache_future_points),
                "action_semantics": action_semantics(args.action_label_mode),
                "action_alignment": str(args.action_alignment),
                "action_label_mode": str(args.action_label_mode),
                "state_semantics": "achieved EEF pose in episode EEF0",
                "world_base_worldflow_sidecars": bool(
                    args.generate_world_base_worldflow_sidecars
                ),
                "rgb": "observation.images.front",
                "point_cloud": "finite front-camera cloud plus selected virtual gripper template in current EEF",
                "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
                "gripper_template": expected_template_version,
                "virtual_gripper": (
                    canonical_reap_metadata()
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else None
                ),
                "collection_workers": int(args.collection_workers),
            },
            file,
            indent=2,
        )
    return cache_root


def main():
    args = parse_args()
    print(
        "[collector-runtime] task_environment="
        + str(REPO_ROOT / "benchmarks" / "RLBench" / "rlbench" / "task_environment.py")
        + " water_plant_collision="
        + os.environ.get("RLBENCH_WATER_PLANT_COLLISION", "enabled")
        + " water_drop_collision="
        + os.environ.get("RLBENCH_WATER_DROP_COLLISION", "original"),
        flush=True,
    )
    artifact_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else args.output_root.expanduser().resolve().parent
        / (args.output_root.name + "_artifacts")
    )
    pack_only_manifest = None
    if args.pack_only:
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "--pack-only requires a completed artifact manifest: " + str(manifest_path)
            )
        with open(manifest_path, "r", encoding="utf-8") as file:
            pack_only_manifest = json.load(file)
        config = pack_only_manifest.get("config")
        records = pack_only_manifest.get("records")
        if not isinstance(config, dict) or not isinstance(records, list):
            raise RuntimeError("Artifact manifest must contain config and records: " + str(manifest_path))
        tasks = list(config.get("tasks") or [])
        if not tasks:
            raise RuntimeError("Artifact manifest has no task list: " + str(manifest_path))
        requested_tasks = resolve_tasks(args) if (args.tasks is not None or args.all_tasks) else tasks
        if requested_tasks != tasks:
            raise RuntimeError(
                "--pack-only task order must match the artifact manifest: requested="
                + repr(requested_tasks)
                + " manifest="
                + repr(tasks)
            )
        # Historical manifests predate this flag and their artifacts do not
        # contain T_world_base/world-base sidecars. Preserve that exact layout
        # when repacking instead of silently enabling a newer default.
        if "generate_world_base_worldflow_sidecars" not in config:
            args.generate_world_base_worldflow_sidecars = False
        for name in (
            "episodes_per_task",
            "variation",
            "num_points",
            "gripper_points",
            "gripper_max_width",
            "gripper_template",
            "image_size",
            "fps",
            "collection_workers",
            "expert_path_mode",
            "segmented_linear_segments",
            "replay_random_seeds_from_artifacts",
            "collection_seed",
            "phone_base_max_initial_distance_m",
            "phone_eef_max_initial_distance_m",
            "replay_scenes_from_artifacts",
            "action_label_mode",
            "action_alignment",
            "generate_world_base_worldflow_sidecars",
        ):
            if name in config:
                setattr(args, name, config[name])
    else:
        tasks = resolve_tasks(args)
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if args.collection_workers <= 0:
        raise ValueError("--collection-workers must be positive")
    if args.segmented_linear_segments <= 0:
        raise ValueError("--segmented-linear-segments must be positive")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if not np.isclose(args.gripper_max_width, RLBENCH_PANDA_MAX_WIDTH, atol=1e-8):
        raise ValueError(
            "RLBench Panda has a fixed 0.08 m total opening; "
            "--gripper-max-width must be " + str(RLBENCH_PANDA_MAX_WIDTH)
        )
    if args.cache_batch_size <= 0 or args.cache_num_workers < 0:
        raise ValueError("PointSeg cache batch size/workers are invalid")
    if not args.pack_only and not os.environ.get("DISPLAY"):
        print("[warning] DISPLAY is not set. RLBench camera rendering usually needs DISPLAY=:99.")
    print("[tasks] " + ", ".join(tasks))
    print("[artifacts] " + str(artifact_root))
    if args.pack_only:
        expected_episode_count = len(tasks) * int(args.episodes_per_task)
        if len(records) != expected_episode_count:
            raise RuntimeError(
                "Artifact manifest is incomplete: records="
                + str(len(records))
                + " expected="
                + str(expected_episode_count)
            )
        print("[pack-only] rebuilding from validated existing artifacts")
        pack_artifacts(args, artifact_root, records, expected_episode_count)
        cache_root = run_pointseg_cache(args)
        print("[done] output=" + str(args.output_root.expanduser().resolve()))
        if cache_root is not None:
            print("[done] pointseg_cache=" + str(cache_root))
        return
    if len(tasks) == 1:
        worker_count = min(args.collection_workers, args.episodes_per_task)
    else:
        worker_count = min(args.collection_workers, len(tasks))
    displays = collection_display_list(args, worker_count)
    validate_collection_displays(displays)
    print("[collection-workers] " + str(worker_count) + " displays=" + ",".join(displays))
    records = collect_parallel(args, tasks, artifact_root)
    if len(records) != len(tasks) * args.episodes_per_task:
        print("[warning] not all requested episodes succeeded; packing the successful episodes only")
    pack_artifacts(args, artifact_root, records, len(tasks) * args.episodes_per_task)
    cache_root = run_pointseg_cache(args)
    print("[done] output=" + str(args.output_root.expanduser().resolve()))
    if cache_root is not None:
        print("[done] pointseg_cache=" + str(cache_root))


if __name__ == "__main__":
    main()
