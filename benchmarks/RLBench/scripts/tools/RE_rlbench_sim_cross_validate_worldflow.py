#!/usr/bin/env python3
"""Validate robot-base WorldFlow against live RLBench/PyRep scene objects."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve()
RLBENCH_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(RLBENCH_ROOT / "scripts"))
sys.path.insert(0, str(RLBENCH_ROOT))

from RE_rlbench_dataset_action_replay import (  # noqa: E402
    compare_initial_object_states,
    initial_object_state_path,
    load_parquet_episode,
    load_recorded_first_robot_state,
    make_environment,
    read_demo_reset_state,
    read_initial_task_state,
    restore_initial_object_states,
    restore_recorded_first_robot_state,
    task_class_from_name,
)
from pyrep.objects.dummy import Dummy  # noqa: E402


def pose9_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    first = value[3:6] / np.linalg.norm(value[3:6])
    second = value[6:9] - np.dot(first, value[6:9]) * first
    second = second / np.linalg.norm(second)
    out = np.eye(4, dtype=np.float64)
    out[:3, 0] = first
    out[:3, 1] = second
    out[:3, 2] = np.cross(first, second)
    out[:3, 3] = value[:3]
    return out


def matrix_errors(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    rotation = reference[:3, :3].T @ actual[:3, :3]
    return {
        "translation_mm": float(np.linalg.norm(reference[:3, 3] - actual[:3, 3]) * 1000.0),
        "rotation_deg": float(np.rad2deg(Rotation.from_matrix(rotation).magnitude())),
        "max_abs": float(np.max(np.abs(reference - actual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", default="phone_on_base")
    parser.add_argument("--episodes", default="0,25,50,75,99")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    episodes = [int(value) for value in args.episodes.split(",") if value.strip()]
    with open(root / "world_base_ee_poses" / "meta.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata_base = np.asarray(metadata["T_world_base"], dtype=np.float64)

    np.random.seed(20260828)
    random.seed(20260828)
    environment = make_environment("eef0_planning", args.image_size, include_point_cloud=False)
    records = []
    scene_names = []
    try:
        environment.launch()
        task = environment.get_task(task_class_from_name(args.task))
        task.set_variation(0)
        scene_objects = environment._pyrep.get_objects_in_tree()
        scene_names = sorted(obj.get_name() for obj in scene_objects if "panda" in obj.get_name().lower())
        link0_candidates = [obj for obj in scene_objects if "panda_link0" in obj.get_name().lower()]

        for local_episode in episodes:
            actions, states, global_episode = load_parquet_episode(root, args.task, local_episode)
            first_pose, first_width, pose_source = load_recorded_first_robot_state(root, global_episode, states)
            placeholder = root / "__no_artifact__.npz"
            initial_state, initial_state_source = read_initial_task_state(root, global_episode, None, placeholder)
            random_state, reset_attempts = read_demo_reset_state(root, global_episode, None, placeholder)
            if random_state is not None:
                np.random.set_state(random_state)

                class ReplayDemo:
                    num_reset_attempts = reset_attempts

                _, observation = task.reset(ReplayDemo())
            else:
                _, observation = task.reset()
            if initial_state is None:
                raise RuntimeError(f"episode {global_episode}: missing initial task state")
            task._task.restore_state(initial_state)
            snapshot = initial_object_state_path(root, global_episode)
            restore_initial_object_states(task._task, snapshot)
            object_validation = compare_initial_object_states(task._task, snapshot)
            observation, robot_validation = restore_recorded_first_robot_state(
                task, first_pose, first_width
            )

            # Independent live base: fixed joint-1 scene frame translated by the
            # Franka URDF d1=0.333 m to link0.  Do not call the sidecar helper.
            joint1 = task._robot.arm.joints[0]
            live_joint1 = np.asarray(joint1.get_matrix(), dtype=np.float64)
            joint1_to_link0 = np.eye(4, dtype=np.float64)
            joint1_to_link0[2, 3] = -0.333
            live_base = live_joint1 @ joint1_to_link0

            live_world_ee = pose9_to_matrix(first_pose)
            observation_world_ee = np.eye(4, dtype=np.float64)
            pose7 = np.asarray(observation.gripper_pose, dtype=np.float64)
            observation_world_ee[:3, :3] = Rotation.from_quat(pose7[3:7]).as_matrix()
            observation_world_ee[:3, 3] = pose7[:3]
            saved_base_ee = pose9_to_matrix(np.load(
                root / "world_base_ee_poses" / f"episode_{global_episode:06d}.npy"
            )[0])
            live_base_ee = np.linalg.inv(live_base) @ observation_world_ee

            direct_link0 = {}
            for candidate in link0_candidates:
                direct_link0[candidate.get_name()] = matrix_errors(
                    live_base, np.asarray(candidate.get_matrix(), dtype=np.float64)
                )

            # Let CoppeliaSim compose EEF0-relative action matrices.  This is an
            # independent convention check rather than NumPy multiplying the
            # same formula used by the generator.
            parent = Dummy.create(0.001)
            child = Dummy.create(0.001)
            try:
                parent.set_matrix(observation_world_ee)
                selected = sorted(set([0, len(actions) // 2, len(actions) - 1]))
                target_checks = []
                saved_targets = np.load(
                    root / "world_base_action_target_ee_poses" / f"episode_{global_episode:06d}.npy"
                )
                for frame in selected:
                    child.set_matrix(pose9_to_matrix(actions[frame, :9]), relative_to=parent)
                    pyrep_world_target = np.asarray(child.get_matrix(), dtype=np.float64)
                    pyrep_base_target = np.linalg.inv(live_base) @ pyrep_world_target
                    target_checks.append({
                        "frame": int(frame),
                        "saved_base_target_vs_pyrep_composition": matrix_errors(
                            pose9_to_matrix(saved_targets[frame]), pyrep_base_target
                        ),
                    })
            finally:
                child.remove()
                parent.remove()

            records.append({
                "local_episode": local_episode,
                "global_episode": global_episode,
                "initial_state_source": initial_state_source,
                "pose_source": pose_source,
                "object_restore": object_validation,
                "robot_restore": robot_validation,
                "metadata_base_vs_live_joint1_derived_base": matrix_errors(metadata_base, live_base),
                "recorded_world_ee_vs_live_observation_ee": matrix_errors(live_world_ee, observation_world_ee),
                "saved_base_ee_vs_live_base_ee": matrix_errors(saved_base_ee, live_base_ee),
                "direct_link0_candidates_vs_joint1_derived_base": direct_link0,
                "target_checks": target_checks,
            })
    finally:
        environment.shutdown()

    base_max = max(r["metadata_base_vs_live_joint1_derived_base"]["translation_mm"] for r in records)
    achieved_max = max(r["saved_base_ee_vs_live_base_ee"]["translation_mm"] for r in records)
    target_max = max(
        check["saved_base_target_vs_pyrep_composition"]["translation_mm"]
        for record in records for check in record["target_checks"]
    )
    result = {
        "dataset_root": str(root),
        "task": args.task,
        "display": os.environ.get("DISPLAY"),
        "episodes": records,
        "panda_scene_objects": scene_names,
        "summary": {
            "episodes_checked": len(records),
            "metadata_base_vs_live_max_translation_mm": base_max,
            "saved_achieved_vs_live_max_translation_mm": achieved_max,
            "saved_target_vs_pyrep_composition_max_translation_mm": target_max,
        },
        "pass": bool(base_max < 0.1 and achieved_max < 0.6 and target_max < 0.1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
