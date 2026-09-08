#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-render point clouds from recorded RLBench artifact episodes.

The artifact collector already stores the exact demo reset random state and
the original absolute Panda joint targets.  This script replays those targets
in RLBench, captures a fresh front-camera cloud for every recorded LeRobot
frame, and packs the selected episodes with the same schema as the normal
collector.  It intentionally does not mutate the source artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SONG_SCRIPT_ROOT = REPO_ROOT / "benchmarks" / "song_real_libero" / "scripts"
COPPELIASIM_ROOT = REPO_ROOT / "benchmarks" / "CoppeliaSim"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SONG_SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero_setting.libero_pointcloud_utils import (
    RLBENCH_PANDA_MAX_WIDTH,
    add_world_gripper_clouds_to_episode,
    sample_or_repeat_points,
)
from rlbench_reap_gripper import (
    LIBERO_GRIPPER_TEMPLATE,
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    LIBERO_REAP_TEMPLATE_MAX_WIDTH,
    libero_reap_width_percent_from_physical,
)
from lerobot.policies.smolvla.song_pointseg import save_point_clouds_zarr
from RE_rlbench_collect_lerobot_pointcloud import (
    OBJECT_STATE_KEYS,
    POINT_DIR,
    RAW_ACTION_DIR,
    RAW_ACTION_FULL_DIR,
    TASK_STATE_DIR,
    OBJECT_STATE_DIR,
    POSE_DIR,
    RLBENCH_SCENE_BOUNDS,
    capture_initial_object_states,
    make_observation_config,
    observation_cloud,
    pose7_to_pose9,
    task_class_from_name,
    write_sidecar_meta,
)


FEATURE_NAMES = ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay RLBench artifacts and pack a fresh LeRobot dataset."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", default="close_laptop_lid")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=50000)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--gripper-max-width", type=float, default=RLBENCH_PANDA_MAX_WIDTH)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def episode_name(task: str, index: int) -> str:
    return f"{task}__episode_{int(index):05d}"


def source_episode(root: Path, task: str, index: int) -> Path:
    path = root.expanduser().resolve() / episode_name(task, index)
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact episode is missing: {path}")
    for required in ("arrays.npz", "record.json"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"Artifact episode is missing {required}: {path}")
    return path


def load_demo_random_state(arrays: dict[str, np.ndarray]) -> tuple[tuple, int]:
    required = {
        "demo_random_seed_state",
        "demo_random_seed_position",
        "demo_random_seed_has_gauss",
        "demo_random_seed_cached_gaussian",
        "demo_num_reset_attempts",
    }
    missing = sorted(required.difference(arrays.keys()))
    if missing:
        raise RuntimeError("Artifact does not contain exact demo reset state: " + ", ".join(missing))
    state = (
        "MT19937",
        np.asarray(arrays["demo_random_seed_state"], dtype=np.uint32),
        int(arrays["demo_random_seed_position"]),
        int(arrays["demo_random_seed_has_gauss"]),
        float(arrays["demo_random_seed_cached_gaussian"]),
    )
    return state, int(arrays["demo_num_reset_attempts"])


def make_environment(image_size: int):
    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete

    return Environment(
        MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete()),
        obs_config=make_observation_config(image_size),
        headless=True,
        static_positions=False,
    )


class ReplayDemo:
    def __init__(self, num_reset_attempts: int):
        self.num_reset_attempts = int(num_reset_attempts)


def render_episode(task_env, source: Path, args: argparse.Namespace, global_seed: int):
    with np.load(source / "arrays.npz", allow_pickle=False) as file:
        arrays = {key: file[key].copy() for key in file.files}
    recorded_frames = int(arrays["actions"].shape[0])
    full_raw = np.asarray(arrays["raw_expert_actions_full"], dtype=np.float32)
    if full_raw.shape != (recorded_frames + 1, 8):
        raise RuntimeError(
            f"Unexpected raw_expert_actions_full shape at {source}: {full_raw.shape}; "
            f"expected ({recorded_frames + 1}, 8)"
        )

    random_state, reset_attempts = load_demo_random_state(arrays)
    np.random.set_state(random_state)
    descriptions, observation = task_env.reset(ReplayDemo(reset_attempts))
    if descriptions and str(descriptions[0]) != str(json.loads((source / "record.json").read_text())["description"]):
        raise RuntimeError(f"RLBench description changed for {source}")

    rendered_images = []
    rendered_clouds = []
    rendered_poses = []
    rendered_grippers = []
    rendered_images.append(np.asarray(observation.front_rgb, dtype=np.uint8))

    def capture(current_observation, frame_index: int) -> None:
        pose = pose7_to_pose9(current_observation.gripper_pose)
        width = (
            float(args.gripper_max_width)
            if float(current_observation.gripper_open) > 0.5
            else 0.0
        )
        cloud = observation_cloud(current_observation, fallback_to_all_finite=True)
        cloud = sample_or_repeat_points(cloud, args.num_points, global_seed + frame_index)
        rendered_poses.append(pose)
        rendered_grippers.append(width)
        rendered_clouds.append(cloud)

    capture(observation, 0)
    for frame_index in range(1, recorded_frames):
        command = full_raw[frame_index]
        if not np.isfinite(command).all():
            raise RuntimeError(f"Missing finite raw action at frame {frame_index} in {source}")
        observation, _, _ = task_env.step(command.astype(np.float32))
        rendered_images.append(np.asarray(observation.front_rgb, dtype=np.uint8))
        capture(observation, frame_index)

    images = np.asarray(rendered_images, dtype=np.uint8)
    if images.shape != arrays["images"].shape:
        raise RuntimeError(
            f"Replayed image shape mismatch at {source}: {images.shape} vs {arrays['images'].shape}"
        )
    image_mae = float(np.mean(np.abs(images.astype(np.float32) - arrays["images"].astype(np.float32))))
    poses = np.asarray(rendered_poses, dtype=np.float32)
    widths = np.asarray(rendered_grippers, dtype=np.float32)
    clouds = np.asarray(rendered_clouds, dtype=np.float32)
    clouds = add_world_gripper_clouds_to_episode(
        clouds,
        poses,
        libero_reap_width_percent_from_physical(widths),
        total_points=args.num_points,
        gripper_points=args.gripper_points,
        gripper_template=LIBERO_GRIPPER_TEMPLATE,
        gripper_len=LIBERO_REAP_GRIPPER_LEN,
        seed=global_seed,
        drop_strategy="tail",
        shuffle_points=False,
        widths_are_normalized=True,
        gripper_max_width=None,
        gripper_opening_max_width=LIBERO_REAP_OPENING_MAX_WIDTH,
    )
    if clouds.shape != (recorded_frames, args.num_points, 6):
        raise RuntimeError(f"Unexpected rendered point cloud shape at {source}: {clouds.shape}")
    return arrays, images, clouds, poses, image_mae


def create_dataset(output_root: Path, repo_id: str, args: argparse.Namespace) -> LeRobotDataset:
    features = {
        "action": {"dtype": "float32", "shape": (10,), "names": FEATURE_NAMES},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": FEATURE_NAMES},
        "observation.images.front": {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=args.fps,
        features=features,
        robot_type="rlbench_panda",
        use_videos=False,
    )
    write_sidecar_meta(dataset.root)
    return dataset


def copy_episode_sidecars(output_root: Path, episode_index: int, arrays: dict[str, np.ndarray]) -> None:
    name = f"episode_{episode_index:06d}"
    np.save(output_root / POSE_DIR / f"{name}.npy", arrays["world_ee_poses"])
    np.save(output_root / RAW_ACTION_DIR / f"{name}.npy", arrays["raw_expert_actions"])
    np.save(output_root / RAW_ACTION_FULL_DIR / f"{name}.npy", arrays["raw_expert_actions_full"])
    np.savez(
        output_root / TASK_STATE_DIR / f"{name}.npz",
        configuration_bytes=arrays["initial_task_state_bytes"],
        object_count=arrays["initial_task_state_object_count"],
        demo_random_seed_state=arrays["demo_random_seed_state"],
        demo_random_seed_position=arrays["demo_random_seed_position"],
        demo_random_seed_has_gauss=arrays["demo_random_seed_has_gauss"],
        demo_random_seed_cached_gaussian=arrays["demo_random_seed_cached_gaussian"],
        demo_num_reset_attempts=arrays["demo_num_reset_attempts"],
    )
    object_state_arrays = {key: arrays[key] for key in OBJECT_STATE_KEYS if key in arrays}
    if object_state_arrays:
        np.savez_compressed(output_root / OBJECT_STATE_DIR / f"{name}.npz", **object_state_arrays)


def write_frame(dataset, task_description: str, arrays: dict[str, np.ndarray], index: int) -> None:
    dataset.add_frame(
        {
            "task": task_description,
            "action": arrays["actions"][index],
            "observation.state": arrays["states"][index],
            "observation.images.front": arrays["images"][index],
        }
    )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.num_points <= 0 or args.gripper_points < 0:
        raise ValueError("episodes and num-points must be positive; gripper-points cannot be negative")
    artifact_root = args.artifact_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        marker = output_root / "meta" / "rlbench_conversion_complete.json"
        if marker.is_file() and not args.overwrite:
            raise FileExistsError(f"Completed output already exists: {output_root}; use --overwrite")
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; use --overwrite")
        shutil.rmtree(output_root)

    episode_indices = list(range(args.episode_start, args.episode_start + args.episodes))
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]
    sources = [source_episode(artifact_root, args.task, index) for index in episode_indices]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(output_root, f"{output_root}", args)
    environment = make_environment(args.image_size)
    summaries = []
    try:
        environment.launch()
        task_env = environment.get_task(task_class_from_name(args.task))
        task_env.set_variation(args.variation)
        for output_index, (local_index, source) in enumerate(zip(episode_indices, sources)):
            print(f"[render-start] local_episode={local_index} output_episode={output_index}", flush=True)
            arrays, replay_images, clouds, poses, image_mae = render_episode(
                task_env, source, args, args.seed + output_index * 100003
            )
            # The requested change is limited to point-cloud rendering. Keep
            # the artifact's original RGB, action, state, and world-pose
            # arrays byte-for-byte; replayed RGB is only a reset diagnostic.
            episode_dir = output_root / POINT_DIR / f"episode_{output_index:06d}.zarr"
            save_point_clouds_zarr(episode_dir, clouds, compression_level=3)
            copy_episode_sidecars(output_root, output_index, arrays)
            for frame_index in range(len(arrays["actions"])):
                write_frame(dataset, args.task.replace("_", " "), arrays, frame_index)
            dataset.save_episode()
            summary = {
                "source_episode": int(local_index),
                "output_episode": int(output_index),
                "frames": int(len(arrays["actions"])),
                "points": int(clouds.shape[1]),
                "replay_image_mae": image_mae,
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    finally:
        try:
            environment.shutdown()
        finally:
            dataset.finalize()

    with open(output_root / "meta" / "rlbench_conversion.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "source": str(artifact_root),
                "source_mode": "exact_artifact_reset_and_raw_joint_replay",
                "task": args.task,
                "episode_count": len(summaries),
                "action_alignment": "transition",
                "action_label_mode": "expert_target",
                "observation_state_semantics": "copied from source artifact",
                "point_cloud_semantics": "fresh RLBench front-camera cloud plus canonical RLBench-aligned REAP template transformed to current EEF",
                "point_cloud_points": args.num_points,
                "gripper_points": args.gripper_points,
                "gripper_template": LIBERO_GRIPPER_TEMPLATE_VERSION,
                "gripper_template_name": LIBERO_GRIPPER_TEMPLATE,
                "gripper_template_version": LIBERO_GRIPPER_TEMPLATE_VERSION,
                "virtual_gripper_width_normalization_max_m": LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
                "virtual_gripper_geometry_max_width_m": LIBERO_REAP_TEMPLATE_MAX_WIDTH,
                "virtual_gripper_opening_max_width_m": LIBERO_REAP_OPENING_MAX_WIDTH,
                "virtual_gripper_local_offset_m": [0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN],
                "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
                "replay_image_policy": "re-rendered RGB frames",
                "episodes": summaries,
            },
            file,
            indent=2,
        )
    with open(output_root / "meta" / "rlbench_conversion_complete.json", "w", encoding="utf-8") as file:
        json.dump({"complete": True, "episode_count": len(summaries), "point_cloud_points": args.num_points}, file, indent=2)
    print(f"[done] dataset={output_root} episodes={len(summaries)} points={args.num_points}", flush=True)


if __name__ == "__main__":
    main()
