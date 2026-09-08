#!/usr/bin/env python3
"""Replay saved RLBench evaluation actions and export selected video-frame point clouds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from _rlbench_tool_paths import SCRIPTS_DIR  # noqa: F401; initializes sibling-module paths
from RE_rlbench_collect_lerobot_pointcloud import make_observation_config, task_class_from_name
from RE_rlbench_official_eval import (
    apply_pointact_pyrep_compatibility_patch,
    live_model_observation,
    rgb_to_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated zero-based video frames.")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_frames(value: str) -> tuple[int, ...]:
    frames = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not frames or frames[0] < 0:
        raise ValueError("--frames must contain non-negative frame indices")
    return frames


def write_point_cloud_ply(path: Path, cloud: np.ndarray, comments: list[str]) -> None:
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 6:
        raise ValueError(f"Expected point cloud shape (N, >=6), got {cloud.shape}")
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:6]
    if rgb.size and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    rgb = np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    rgb = rgb[valid]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        for comment in comments:
            file.write(f"comment {comment}\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(xyz, rgb, strict=True):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def load_logged_after_poses(control_log: Path, episode: int) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    prefix = "[control-after] "
    if not control_log.is_file():
        return poses
    with control_log.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.startswith(prefix):
                continue
            try:
                record = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                continue
            if int(record.get("episode_index", -1)) != episode:
                continue
            pose = record.get("actual_state_world_pose7")
            if pose is not None:
                poses.append(np.asarray(pose, dtype=np.float64))
    return poses


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    frames = parse_frames(args.frames)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    actions_path = run_dir / f"episode_{args.episode:03d}_actions.npy"
    actions = np.asarray(np.load(actions_path), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Expected saved actions shape (T, 8), got {actions.shape}")
    if frames[-1] > len(actions):
        raise ValueError(f"Frame {frames[-1]} exceeds final video frame {len(actions)}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "replayed_frame_ply" / f"episode_{args.episode:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["RLBENCH_WATER_PLANT_COLLISION"] = str(
        config.get("water_plant_collision", "enabled")
    )
    os.environ["RLBENCH_WATER_DROP_COLLISION"] = str(
        config.get("water_drop_collision", "original")
    )

    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
    from rlbench.action_modes.gripper_action_modes import Discrete

    if bool(config.get("pointact_pyrep_compat", False)):
        patch_result = apply_pointact_pyrep_compatibility_patch()
    else:
        patch_result = {"requested": False, "applied": False}

    arm_mode = str(config.get("arm_action_mode", "planning"))
    if arm_mode != "planning":
        raise ValueError(f"This replay exporter currently requires planning mode, got {arm_mode!r}")
    action_mode = MoveArmThenGripper(
        EndEffectorPoseViaPlanning(
            absolute_mode=True,
            frame=RelativeFrame.WORLD,
            collision_checking=bool(config.get("collision_checking", False)),
        ),
        Discrete(),
    )
    env = Environment(
        action_mode,
        obs_config=make_observation_config(int(config.get("image_size", 256))),
        headless=True,
        static_positions=False,
    )

    expected_poses = load_logged_after_poses(run_dir / "control.log", args.episode)
    pose_errors: list[float] = []
    exported: list[dict[str, object]] = []
    try:
        env.launch()
        task_env = env.get_task(task_class_from_name(str(config["task"])))
        task_env.set_variation(int(config.get("variation", 0)))
        seed_episode_index = args.episode
        np.random.seed(int(config.get("seed", 0)) + seed_episode_index)
        _, observation = task_env.reset()
        if float(config.get("simulation_timestep", 0.0)) > 0.0:
            task_env._pyrep.set_simulation_timestep(float(config["simulation_timestep"]))

        cloud_args = SimpleNamespace(
            num_points=int(config.get("num_points", 20000)),
            add_gripper_cloud=bool(config.get("add_gripper_cloud", True)),
            gripper_template=str(config.get("gripper_template", "reap")),
            gripper_points=int(config.get("gripper_points", 500)),
        )
        for frame_index, action in enumerate(actions, start=1):
            observation, reward, termination = task_env.step(action)
            if frame_index <= len(expected_poses):
                pose_errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(observation.gripper_pose, dtype=np.float64)
                                - expected_poses[frame_index - 1]
                            )
                        )
                    )
                )
            if frame_index in frames:
                point_seed = (
                    int(config.get("seed", 0)) * 100000
                    + seed_episode_index * 1000
                    + frame_index
                )
                model_observation = live_model_observation(
                    observation,
                    cloud_args,
                    seed=point_seed,
                )
                ply_path = output_dir / f"frame_{frame_index:06d}_model_input_raw.ply"
                png_path = output_dir / f"frame_{frame_index:06d}_front.png"
                write_point_cloud_ply(
                    ply_path,
                    model_observation["point_cloud"],
                    [
                        "coordinate_frame current_eef",
                        "source deterministic_replay_of_saved_eval_actions",
                        f"episode_index {args.episode}",
                        f"video_frame_index {frame_index}",
                        f"point_sampling_seed {point_seed}",
                        "pointseg_probabilities unavailable_no_model_call_at_this_frame",
                    ],
                )
                Image.fromarray(rgb_to_uint8(observation.front_rgb)).save(png_path)
                exported.append(
                    {
                        "frame_index": frame_index,
                        "ply": str(ply_path),
                        "front_rgb": str(png_path),
                        "points": int(len(model_observation["point_cloud"])),
                        "gripper_pose_world7": np.asarray(
                            observation.gripper_pose, dtype=np.float32
                        ).tolist(),
                        "gripper_open": float(observation.gripper_open),
                        "reward": float(reward),
                        "termination": bool(termination),
                    }
                )
    finally:
        env.shutdown()

    summary = {
        "source_run": str(run_dir),
        "source_actions": str(actions_path),
        "episode_index": args.episode,
        "requested_frames": list(frames),
        "frame_numbering": "zero_based_video_frame; frame 0 is reset, frame N is after saved action N",
        "point_cloud_semantics": "replayed raw model input in current EEF coordinates; no PointSeg probability",
        "pyrep_patch": patch_result,
        "logged_control_after_poses": len(expected_poses),
        "pose_validation_frames": len(pose_errors),
        "max_abs_pose7_component_error": max(pose_errors) if pose_errors else None,
        "mean_abs_pose7_component_error": float(np.mean(pose_errors)) if pose_errors else None,
        "exports": exported,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
