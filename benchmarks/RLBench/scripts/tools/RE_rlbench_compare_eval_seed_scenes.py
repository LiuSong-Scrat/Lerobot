#!/usr/bin/env python
"""Compare ordinary RLBench eval seeds with recorded dataset initial scenes.

For an online evaluation without ``--dataset-root``, the evaluator performs
``np.random.seed(base_seed + episode_index)`` immediately before
``task_env.reset()``.  This utility reproduces that initialization, stores the
complete task-object state tree, and compares seed ``i`` with both dataset
episode ``i`` and every other episode of the same task.

Run one task per process (normally under separate ``xvfb-run`` instances), then
use ``--aggregate-only`` to build the ten-task report.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as pyarrow_dataset


TASK_DESCRIPTIONS = {
    "close_box": "close box",
    "close_fridge": "close fridge",
    "close_laptop_lid": "close laptop lid",
    "phone_on_base": "put the phone on the base",
    "stack_wine": "stack wine bottle",
    "sweep_to_dustpan": "sweep dirt to dustpan",
    "take_frame_off_hanger": "take frame off hanger",
    "take_umbrella_out_of_umbrella_stand": (
        "take umbrella out of umbrella stand"
    ),
    "toilet_seat_down": "toilet seat down",
    "water_plants": "water plant",
}

TASKS = tuple(TASK_DESCRIPTIONS)
POSITION_EXACT_TOLERANCE_M = 1e-6
ROTATION_EXACT_TOLERANCE_RAD = 1e-6
JOINT_EXACT_TOLERANCE_RAD = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def task_class_from_name(task_name: str):
    class_name = "".join(part[:1].upper() + part[1:] for part in task_name.split("_"))
    module = importlib.import_module("rlbench.tasks." + task_name)
    return getattr(module, class_name)


def make_observation_config(image_size: int):
    # Keep the evaluator's front-camera observation contract.  Observation
    # capture does not alter task placement, but matching it avoids relying on
    # that implementation detail.
    from rlbench import CameraConfig, ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.front_camera = CameraConfig(
        rgb=True,
        depth=False,
        point_cloud=True,
        mask=False,
        image_size=(image_size, image_size),
    )
    config.gripper_open = True
    config.gripper_pose = True
    config.joint_positions = True
    config.joint_velocities = True
    config.record_gripper_closing = True
    return config


def capture_initial_object_states(task) -> dict[str, np.ndarray]:
    objects = task.get_base().get_objects_in_tree(
        exclude_base=False, first_generation_only=False
    )
    count = len(objects)
    names: list[str] = []
    types: list[str] = []
    poses = np.full((count, 7), np.nan, dtype=np.float64)
    joints = np.full(count, np.nan, dtype=np.float64)
    for index, obj in enumerate(objects):
        names.append(obj.get_name())
        object_type = obj.get_type()
        types.append(getattr(object_type, "name", str(object_type)))
        try:
            poses[index] = np.asarray(obj.get_pose(), dtype=np.float64)
        except Exception:
            pass
        if hasattr(obj, "get_joint_position"):
            try:
                joints[index] = float(obj.get_joint_position())
            except Exception:
                pass
    if len(set(names)) != len(names):
        raise RuntimeError("Task object names are not unique: " + repr(names))
    return {
        "initial_object_names": np.asarray(names, dtype="U256"),
        "initial_object_types": np.asarray(types, dtype="U64"),
        "initial_object_poses": poses,
        "initial_object_joint_positions": joints,
    }


def load_snapshot(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            "initial_object_names": np.asarray(source["initial_object_names"]),
            "initial_object_types": np.asarray(source["initial_object_types"]),
            "initial_object_poses": np.asarray(
                source["initial_object_poses"], dtype=np.float64
            ),
            "initial_object_joint_positions": np.asarray(
                source["initial_object_joint_positions"], dtype=np.float64
            ),
        }


def dataset_episodes_for_task(dataset_root: Path, task_name: str) -> list[int]:
    table = pyarrow_dataset.dataset(
        str(dataset_root / "meta" / "episodes"), format="parquet"
    ).to_table(columns=["episode_index", "tasks"])
    expected = TASK_DESCRIPTIONS[task_name].strip().lower().replace("_", " ")
    result = []
    for episode_index, descriptions in zip(
        table["episode_index"].to_pylist(), table["tasks"].to_pylist()
    ):
        description = descriptions[0] if descriptions else ""
        normalized = str(description).strip().lower().replace("_", " ")
        if normalized == expected:
            result.append(int(episode_index))
    return sorted(result)


def quaternion_angle_rad(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # RLBench poses use [qx, qy, qz, qw].  q and -q encode the same rotation.
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    dots = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def compare_snapshots(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    allowed_object_types: set[str] | None = None,
):
    left_names = left["initial_object_names"].tolist()
    right_names = right["initial_object_names"].tolist()
    left_index = {name: index for index, name in enumerate(left_names)}
    right_index = {name: index for index, name in enumerate(right_names)}
    left_types = {
        name: str(left["initial_object_types"][index])
        for name, index in left_index.items()
    }
    right_types = {
        name: str(right["initial_object_types"][index])
        for name, index in right_index.items()
    }
    if allowed_object_types is None:
        selected_left = set(left_index)
        selected_right = set(right_index)
    else:
        selected_left = {
            name for name, object_type in left_types.items()
            if object_type in allowed_object_types
        }
        selected_right = {
            name for name, object_type in right_types.items()
            if object_type in allowed_object_types
        }
    common = sorted(selected_left.intersection(selected_right))
    missing_left = sorted(selected_right.difference(selected_left))
    missing_right = sorted(selected_left.difference(selected_right))

    position_errors = []
    rotation_errors = []
    joint_errors = []
    per_object = []
    for name in common:
        li = left_index[name]
        ri = right_index[name]
        left_pose = left["initial_object_poses"][li]
        right_pose = right["initial_object_poses"][ri]
        position_error = math.nan
        rotation_error = math.nan
        if np.isfinite(left_pose).all() and np.isfinite(right_pose).all():
            position_error = float(np.linalg.norm(left_pose[:3] - right_pose[:3]))
            rotation_error = float(
                quaternion_angle_rad(left_pose[None, 3:7], right_pose[None, 3:7])[0]
            )
            position_errors.append(position_error)
            rotation_errors.append(rotation_error)
        left_joint = left["initial_object_joint_positions"][li]
        right_joint = right["initial_object_joint_positions"][ri]
        joint_error = math.nan
        if np.isfinite(left_joint) and np.isfinite(right_joint):
            joint_error = float(abs(left_joint - right_joint))
            joint_errors.append(joint_error)
        per_object.append((name, position_error, rotation_error, joint_error))

    pos = np.asarray(position_errors, dtype=np.float64)
    rot = np.asarray(rotation_errors, dtype=np.float64)
    joint = np.asarray(joint_errors, dtype=np.float64)
    max_position = float(np.max(pos)) if len(pos) else math.inf
    rms_position = float(np.sqrt(np.mean(pos * pos))) if len(pos) else math.inf
    max_rotation = float(np.max(rot)) if len(rot) else math.inf
    rms_rotation = float(np.sqrt(np.mean(rot * rot))) if len(rot) else math.inf
    max_joint = float(np.max(joint)) if len(joint) else 0.0
    exact = bool(
        not missing_left
        and not missing_right
        and max_position <= POSITION_EXACT_TOLERANCE_M
        and max_rotation <= ROTATION_EXACT_TOLERANCE_RAD
        and max_joint <= JOINT_EXACT_TOLERANCE_RAD
    )
    # Position dominates task placement. Rotation and articulated joint terms
    # break ties while keeping the score in approximately metre units.
    nearest_score = rms_position + 0.05 * rms_rotation + 0.05 * max_joint
    worst_object = ""
    if per_object:
        worst_object = max(
            per_object,
            key=lambda item: -math.inf if math.isnan(item[1]) else item[1],
        )[0]
    return {
        "comparable_objects": len(common),
        "missing_in_eval": missing_left,
        "missing_in_dataset": missing_right,
        "max_position_error_m": max_position,
        "rms_position_error_m": rms_position,
        "max_rotation_error_rad": max_rotation,
        "rms_rotation_error_rad": rms_rotation,
        "max_joint_error_rad": max_joint,
        "nearest_score": nearest_score,
        "worst_position_object": worst_object,
        "exact_match": exact,
    }


def task_output_dir(output_dir: Path, task_name: str) -> Path:
    return output_dir / "tasks" / task_name


def run_task(args: argparse.Namespace) -> None:
    if args.task is None:
        raise ValueError("--task is required unless --aggregate-only is used")
    if args.seed_start < 0 or args.seed_end < args.seed_start:
        raise ValueError("Invalid seed range")

    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        EndEffectorPoseViaPlanning,
        RelativeFrame,
    )
    from rlbench.action_modes.gripper_action_modes import Discrete

    dataset_root = args.dataset_root.expanduser().resolve()
    output = task_output_dir(args.output_dir.expanduser().resolve(), args.task)
    states_dir = output / "eval_initial_object_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    global_episodes = dataset_episodes_for_task(dataset_root, args.task)
    if len(global_episodes) <= args.seed_end:
        raise RuntimeError(
            f"{args.task}: dataset has {len(global_episodes)} episodes, "
            f"but comparison requests task-local episode {args.seed_end}"
        )
    dataset_snapshots = [
        load_snapshot(
            dataset_root
            / "initial_object_states"
            / f"episode_{global_episode:06d}.npz"
        )
        for global_episode in global_episodes
    ]

    env = Environment(
        MoveArmThenGripper(
            EndEffectorPoseViaPlanning(
                absolute_mode=True,
                frame=RelativeFrame.WORLD,
                collision_checking=False,
            ),
            Discrete(),
        ),
        obs_config=make_observation_config(args.image_size),
        headless=True,
        static_positions=False,
    )
    rows = []
    env.launch()
    try:
        task_env = env.get_task(task_class_from_name(args.task))
        task_env.set_variation(args.variation)
        for seed in range(args.seed_start, args.seed_end + 1):
            local_index = seed
            # Exact online-eval semantics for a base seed of zero: seed i is
            # applied immediately before reset for episode i.
            np.random.seed(seed)
            task_env.reset()
            eval_snapshot = capture_initial_object_states(task_env._task)
            state_path = states_dir / f"seed_{seed:03d}_episode_{local_index:03d}.npz"
            np.savez_compressed(state_path, **eval_snapshot)

            comparisons = [
                compare_snapshots(eval_snapshot, dataset_snapshot)
                for dataset_snapshot in dataset_snapshots
            ]
            same = comparisons[local_index]
            nearest_index = min(
                range(len(comparisons)),
                key=lambda index: comparisons[index]["nearest_score"],
            )
            nearest = comparisons[nearest_index]
            exact_indices = [
                index
                for index, comparison in enumerate(comparisons)
                if comparison["exact_match"]
            ]
            rows.append(
                {
                    "task": args.task,
                    "eval_seed": seed,
                    "eval_episode_index": local_index,
                    "dataset_local_episode_index": local_index,
                    "dataset_global_episode_index": global_episodes[local_index],
                    "object_count": len(eval_snapshot["initial_object_names"]),
                    "comparable_objects": same["comparable_objects"],
                    "same_index_max_position_error_mm": (
                        same["max_position_error_m"] * 1000.0
                    ),
                    "same_index_rms_position_error_mm": (
                        same["rms_position_error_m"] * 1000.0
                    ),
                    "same_index_max_rotation_error_deg": math.degrees(
                        same["max_rotation_error_rad"]
                    ),
                    "same_index_max_joint_error_deg": math.degrees(
                        same["max_joint_error_rad"]
                    ),
                    "same_index_worst_position_object": same[
                        "worst_position_object"
                    ],
                    "same_index_exact_match": same["exact_match"],
                    "nearest_dataset_local_episode_index": nearest_index,
                    "nearest_dataset_global_episode_index": global_episodes[
                        nearest_index
                    ],
                    "nearest_max_position_error_mm": (
                        nearest["max_position_error_m"] * 1000.0
                    ),
                    "nearest_rms_position_error_mm": (
                        nearest["rms_position_error_m"] * 1000.0
                    ),
                    "nearest_max_rotation_error_deg": math.degrees(
                        nearest["max_rotation_error_rad"]
                    ),
                    "same_index_is_nearest": nearest_index == local_index,
                    "exact_dataset_local_episode_indices": exact_indices,
                    "any_exact_task_episode_match": bool(exact_indices),
                    "eval_state_path": str(state_path),
                    "dataset_state_path": str(
                        dataset_root
                        / "initial_object_states"
                        / f"episode_{global_episodes[local_index]:06d}.npz"
                    ),
                }
            )
            print(
                f"[scene-compare] task={args.task} seed={seed} "
                f"same_max_mm={rows[-1]['same_index_max_position_error_mm']:.3f} "
                f"nearest_local={nearest_index} "
                f"nearest_max_mm={rows[-1]['nearest_max_position_error_mm']:.3f}",
                flush=True,
            )
    finally:
        env.shutdown()

    fieldnames = list(rows[0])
    with open(output / "episode_comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["exact_dataset_local_episode_indices"] = json.dumps(
                serialized["exact_dataset_local_episode_indices"], separators=(",", ":")
            )
            writer.writerow(serialized)
    with open(output / "episode_comparison.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    summary = summarize_task_rows(args.task, rows)
    with open(output / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("[task-summary] " + json.dumps(summary, separators=(",", ":")), flush=True)


def summarize_task_rows(task_name: str, rows: list[dict]) -> dict:
    same_max = np.asarray(
        [row["same_index_max_position_error_mm"] for row in rows], dtype=np.float64
    )
    nearest_max = np.asarray(
        [row["nearest_max_position_error_mm"] for row in rows], dtype=np.float64
    )
    summary = {
        "task": task_name,
        "episode_count": len(rows),
        "same_index_exact_matches": sum(row["same_index_exact_match"] for row in rows),
        "any_task_episode_exact_matches": sum(
            row["any_exact_task_episode_match"] for row in rows
        ),
        "same_index_is_nearest_count": sum(row["same_index_is_nearest"] for row in rows),
        "same_index_max_position_error_mm_mean": float(np.mean(same_max)),
        "same_index_max_position_error_mm_median": float(np.median(same_max)),
        "same_index_max_position_error_mm_min": float(np.min(same_max)),
        "same_index_max_position_error_mm_max": float(np.max(same_max)),
        "nearest_max_position_error_mm_mean": float(np.mean(nearest_max)),
        "nearest_max_position_error_mm_median": float(np.median(nearest_max)),
        "nearest_max_position_error_mm_min": float(np.min(nearest_max)),
        "nearest_max_position_error_mm_max": float(np.max(nearest_max)),
    }
    if "same_index_physical_max_position_error_mm" in rows[0]:
        physical_same = np.asarray(
            [row["same_index_physical_max_position_error_mm"] for row in rows],
            dtype=np.float64,
        )
        physical_nearest = np.asarray(
            [row["nearest_physical_max_position_error_mm"] for row in rows],
            dtype=np.float64,
        )
        summary.update(
            {
                "same_index_physical_exact_matches": sum(
                    row["same_index_physical_exact_match"] for row in rows
                ),
                "any_physical_exact_task_episode_matches": sum(
                    row["any_physical_exact_task_episode_match"] for row in rows
                ),
                "same_index_physical_is_nearest_count": sum(
                    row["same_index_physical_is_nearest"] for row in rows
                ),
                "same_index_physical_max_position_error_mm_mean": float(
                    np.mean(physical_same)
                ),
                "same_index_physical_max_position_error_mm_median": float(
                    np.median(physical_same)
                ),
                "same_index_physical_max_position_error_mm_min": float(
                    np.min(physical_same)
                ),
                "same_index_physical_max_position_error_mm_max": float(
                    np.max(physical_same)
                ),
                "nearest_physical_max_position_error_mm_mean": float(
                    np.mean(physical_nearest)
                ),
                "nearest_physical_max_position_error_mm_median": float(
                    np.median(physical_nearest)
                ),
                "nearest_physical_max_position_error_mm_min": float(
                    np.min(physical_nearest)
                ),
                "nearest_physical_max_position_error_mm_max": float(
                    np.max(physical_nearest)
                ),
            }
        )
    return summary


def aggregate(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    all_rows = []
    summaries = []
    dataset_root = args.dataset_root.expanduser().resolve()
    for task_name in TASKS:
        task_dir = task_output_dir(output, task_name)
        with open(task_dir / "episode_comparison.json", encoding="utf-8") as file:
            rows = json.load(file)
        global_episodes = dataset_episodes_for_task(dataset_root, task_name)
        dataset_snapshots = [
            load_snapshot(
                dataset_root
                / "initial_object_states"
                / f"episode_{global_episode:06d}.npz"
            )
            for global_episode in global_episodes
        ]
        all_max_position_matrix = np.empty((len(rows), len(dataset_snapshots)))
        all_rms_position_matrix = np.empty_like(all_max_position_matrix)
        physical_max_position_matrix = np.empty_like(all_max_position_matrix)
        physical_rms_position_matrix = np.empty_like(all_max_position_matrix)
        physical_max_rotation_matrix = np.empty_like(all_max_position_matrix)
        physical_exact_matrix = np.zeros_like(all_max_position_matrix, dtype=np.bool_)
        for row_index, row in enumerate(rows):
            eval_snapshot = load_snapshot(Path(row["eval_state_path"]))
            all_comparisons = [
                compare_snapshots(eval_snapshot, dataset_snapshot)
                for dataset_snapshot in dataset_snapshots
            ]
            physical_comparisons = [
                compare_snapshots(
                    eval_snapshot,
                    dataset_snapshot,
                    allowed_object_types={"SHAPE", "JOINT"},
                )
                for dataset_snapshot in dataset_snapshots
            ]
            all_max_position_matrix[row_index] = [
                comparison["max_position_error_m"] * 1000.0
                for comparison in all_comparisons
            ]
            all_rms_position_matrix[row_index] = [
                comparison["rms_position_error_m"] * 1000.0
                for comparison in all_comparisons
            ]
            physical_max_position_matrix[row_index] = [
                comparison["max_position_error_m"] * 1000.0
                for comparison in physical_comparisons
            ]
            physical_rms_position_matrix[row_index] = [
                comparison["rms_position_error_m"] * 1000.0
                for comparison in physical_comparisons
            ]
            physical_max_rotation_matrix[row_index] = [
                math.degrees(comparison["max_rotation_error_rad"])
                for comparison in physical_comparisons
            ]
            physical_exact_matrix[row_index] = [
                comparison["exact_match"] for comparison in physical_comparisons
            ]
            local_index = int(row["dataset_local_episode_index"])
            same = physical_comparisons[local_index]
            nearest_index = min(
                range(len(physical_comparisons)),
                key=lambda index: physical_comparisons[index]["nearest_score"],
            )
            nearest = physical_comparisons[nearest_index]
            exact_indices = [
                index
                for index, comparison in enumerate(physical_comparisons)
                if comparison["exact_match"]
            ]
            row.update(
                {
                    "physical_object_types": ["SHAPE", "JOINT"],
                    "physical_comparable_objects": same["comparable_objects"],
                    "same_index_physical_max_position_error_mm": (
                        same["max_position_error_m"] * 1000.0
                    ),
                    "same_index_physical_rms_position_error_mm": (
                        same["rms_position_error_m"] * 1000.0
                    ),
                    "same_index_physical_max_rotation_error_deg": math.degrees(
                        same["max_rotation_error_rad"]
                    ),
                    "same_index_physical_max_joint_error_deg": math.degrees(
                        same["max_joint_error_rad"]
                    ),
                    "same_index_physical_worst_position_object": same[
                        "worst_position_object"
                    ],
                    "same_index_physical_exact_match": same["exact_match"],
                    "nearest_physical_dataset_local_episode_index": nearest_index,
                    "nearest_physical_dataset_global_episode_index": global_episodes[
                        nearest_index
                    ],
                    "nearest_physical_max_position_error_mm": (
                        nearest["max_position_error_m"] * 1000.0
                    ),
                    "nearest_physical_rms_position_error_mm": (
                        nearest["rms_position_error_m"] * 1000.0
                    ),
                    "same_index_physical_is_nearest": nearest_index == local_index,
                    "exact_physical_dataset_local_episode_indices": exact_indices,
                    "any_physical_exact_task_episode_match": bool(exact_indices),
                }
            )
        np.savez_compressed(
            task_dir / "pairwise_100x100_comparison_matrices.npz",
            eval_seeds=np.asarray([row["eval_seed"] for row in rows], dtype=np.int64),
            dataset_local_episode_indices=np.arange(
                len(dataset_snapshots), dtype=np.int64
            ),
            dataset_global_episode_indices=np.asarray(global_episodes, dtype=np.int64),
            all_object_max_position_error_mm=all_max_position_matrix,
            all_object_rms_position_error_mm=all_rms_position_matrix,
            physical_max_position_error_mm=physical_max_position_matrix,
            physical_rms_position_error_mm=physical_rms_position_matrix,
            physical_max_rotation_error_deg=physical_max_rotation_matrix,
            physical_exact_match=physical_exact_matrix,
        )
        with open(
            task_dir / "pairwise_physical_max_position_error_mm.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ["eval_seed\\dataset_local_episode"]
                + list(range(len(dataset_snapshots)))
            )
            for row, errors in zip(rows, physical_max_position_matrix):
                writer.writerow([row["eval_seed"]] + errors.tolist())
        with open(task_dir / "episode_comparison.json", "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
        with open(task_dir / "episode_comparison.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                for key in (
                    "exact_dataset_local_episode_indices",
                    "physical_object_types",
                    "exact_physical_dataset_local_episode_indices",
                ):
                    serialized[key] = json.dumps(serialized[key], separators=(",", ":"))
                writer.writerow(serialized)
        all_rows.extend(rows)
        summaries.append(summarize_task_rows(task_name, rows))

    with open(output / "episode_comparison_all_1000.json", "w", encoding="utf-8") as file:
        json.dump(all_rows, file, indent=2)
    with open(output / "task_summary.json", "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    with open(output / "task_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    with open(output / "episode_comparison_all_1000.csv", "w", newline="", encoding="utf-8") as file:
        fieldnames = list(all_rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            serialized = dict(row)
            for key in (
                "exact_dataset_local_episode_indices",
                "physical_object_types",
                "exact_physical_dataset_local_episode_indices",
            ):
                serialized[key] = json.dumps(serialized[key], separators=(",", ":"))
            writer.writerow(serialized)

    overall = {
        "task_count": len(TASKS),
        "episode_count": len(all_rows),
        "seed_range": [args.seed_start, args.seed_end],
        "comparison_alignment": "eval seed i versus dataset task-local episode i",
        "same_index_exact_matches": sum(
            row["same_index_exact_match"] for row in all_rows
        ),
        "any_task_episode_exact_matches": sum(
            row["any_exact_task_episode_match"] for row in all_rows
        ),
        "same_index_is_nearest_count": sum(
            row["same_index_is_nearest"] for row in all_rows
        ),
        "same_index_physical_exact_matches": sum(
            row["same_index_physical_exact_match"] for row in all_rows
        ),
        "any_physical_exact_task_episode_matches": sum(
            row["any_physical_exact_task_episode_match"] for row in all_rows
        ),
        "same_index_physical_is_nearest_count": sum(
            row["same_index_physical_is_nearest"] for row in all_rows
        ),
        "exact_tolerances": {
            "position_m": POSITION_EXACT_TOLERANCE_M,
            "rotation_rad": ROTATION_EXACT_TOLERANCE_RAD,
            "joint_rad": JOINT_EXACT_TOLERANCE_RAD,
        },
    }
    with open(output / "overall.json", "w", encoding="utf-8") as file:
        json.dump(overall, file, indent=2)

    report_lines = [
        "# RLBench eval seed 0–99 vs. training scenes",
        "",
        "The comparison reproduces ordinary online evaluation initialization: "
        "`np.random.seed(i)` followed by `task_env.reset()` with variation 0, "
        "`static_positions=False`, and the evaluator's planning action-mode "
        "environment. Dataset alignment is task-local episode `i`.",
        "",
        "Physical objects are CoppeliaSim `SHAPE` and `JOINT` objects; waypoints, "
        "success sensors, dummies, and paths are excluded from the physical-object columns.",
        "",
        "| Task | Episodes | Same-index physical exact | Any exact physical scene | "
        "Same index physically nearest | Same-index physical max error median (mm) | "
        "Nearest physical max error median (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        report_lines.append(
            "| {task} | {episode_count} | {same_index_physical_exact_matches} | "
            "{any_physical_exact_task_episode_matches} | "
            "{same_index_physical_is_nearest_count} | "
            "{same_index_physical_max_position_error_mm_median:.3f} | "
            "{nearest_physical_max_position_error_mm_median:.3f} |".format(**summary)
        )
    report_lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Same-index exact matches: {overall['same_index_exact_matches']} / {len(all_rows)}",
            f"- Eval scenes with any exact match among the task's 100 dataset scenes: "
            f"{overall['any_task_episode_exact_matches']} / {len(all_rows)}",
            f"- Same-index dataset episode was the nearest scene: "
            f"{overall['same_index_is_nearest_count']} / {len(all_rows)}",
            f"- Same-index physical-object exact matches: "
            f"{overall['same_index_physical_exact_matches']} / {len(all_rows)}",
            f"- Eval scenes with any exact physical-object match among the task's "
            f"100 dataset scenes: {overall['any_physical_exact_task_episode_matches']} / {len(all_rows)}",
            "- Exact means every named task object pose and finite joint position "
            "matches within 1e-6 m/rad.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("[overall] " + json.dumps(overall, separators=(",", ":")), flush=True)


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
    else:
        run_task(args)


if __name__ == "__main__":
    main()
