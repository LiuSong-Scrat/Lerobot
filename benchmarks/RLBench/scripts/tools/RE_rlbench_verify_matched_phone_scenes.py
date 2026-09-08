#!/usr/bin/env python
"""Restore matched phone scenes and save their actual pre-planning render/state error."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from RE_rlbench_collect_lerobot_pointcloud import (  # noqa: E402
    OBJECT_STATE_KEYS,
    capture_initial_object_states,
    make_observation_config,
    restore_task_environment_from_artifact_arrays,
    task_class_from_name,
)


def quaternion_angle(q1, q2):
    r1 = Rotation.from_quat(np.asarray(q1, dtype=np.float64))
    r2 = Rotation.from_quat(np.asarray(q2, dtype=np.float64))
    return float((r1.inv() * r2).magnitude())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-bank", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=make_observation_config(256),
        headless=True,
        static_positions=False,
    )
    env.launch()
    rows = []
    try:
        task_env = env.get_task(task_class_from_name("phone_on_base"))
        task_env.set_variation(0)
        for replay_index in range(args.count):
            source = (
                args.scene_bank
                / ("phone_on_base__episode_" + str(replay_index).zfill(5))
                / "arrays.npz"
            )
            with np.load(source, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
            _, observation, _, _ = restore_task_environment_from_artifact_arrays(
                task_env, arrays
            )
            actual = capture_initial_object_states(task_env._task)
            actual_by_name = {
                str(name): i for i, name in enumerate(actual["initial_object_names"])
            }
            position_errors = []
            rotation_errors = []
            for source_i, name in enumerate(arrays["initial_object_names"]):
                actual_i = actual_by_name[str(name)]
                source_pose = np.asarray(arrays["initial_object_poses"][source_i])
                actual_pose = np.asarray(actual["initial_object_poses"][actual_i])
                if np.isfinite(source_pose).all() and np.isfinite(actual_pose).all():
                    position_errors.append(
                        float(np.linalg.norm(source_pose[:3] - actual_pose[:3]))
                    )
                    rotation_errors.append(quaternion_angle(source_pose[3:7], actual_pose[3:7]))

            local_episode = int(arrays["source_phone_local_episode"])
            global_episode = int(arrays["source_global_episode"])
            old_pose = np.load(
                args.old_dataset
                / "world_ee_poses"
                / ("episode_" + str(global_episode).zfill(6) + ".npy")
            )[0]
            eef_position_error = float(
                np.linalg.norm(
                    np.asarray(observation.gripper_pose[:3], dtype=np.float64)
                    - np.asarray(old_pose[:3], dtype=np.float64)
                )
            )
            image = np.asarray(observation.front_rgb, dtype=np.uint8)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                "RESTORED BEFORE PLANNING | local %03d global %06d"
                % (local_episode, global_episode),
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            screenshot = args.output_dir / (
                "local_%03d_global_%06d_restored.png" % (local_episode, global_episode)
            )
            cv2.imwrite(str(screenshot), bgr)
            rows.append(
                {
                    "replay_index": replay_index,
                    "source_phone_local_episode": local_episode,
                    "source_global_episode": global_episode,
                    "max_object_position_error_m": max(position_errors, default=0.0),
                    "max_object_rotation_error_rad": max(rotation_errors, default=0.0),
                    "eef_position_error_vs_historical_frame0_m": eef_position_error,
                    "restored_initial_image": str(screenshot),
                    "object_count": len(arrays["initial_object_names"]),
                    "object_state_keys_present": all(key in arrays for key in OBJECT_STATE_KEYS),
                }
            )
            print("[verified]", json.dumps(rows[-1], separators=(",", ":")), flush=True)
    finally:
        env.shutdown()
    with open(args.output_dir / "verification.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


if __name__ == "__main__":
    main()
