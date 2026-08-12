#!/usr/bin/env python3

"""Audit task sampling balance and camera provenance after global FPS fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.smolvla.song_pointseg import (
    compose_point_cloud_views,
    fps_sample_fused_point_cloud,
    open_episode_point_clouds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def voxel_keys(xyz: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    quantized = np.floor(xyz.astype(np.float64) / voxel_size).astype(np.int64)
    return {tuple(row) for row in quantized.tolist()}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    episodes = pd.read_parquet(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    tasks = pd.read_parquet(dataset_root / "meta/tasks.parquet").reset_index(names="task")
    task_name_to_index = dict(zip(tasks["task"], tasks["task_index"], strict=True))

    episode_rows: list[dict[str, int | str]] = []
    for row in episodes[["episode_index", "tasks", "length"]].itertuples(index=False):
        task_name = str(row.tasks[0])
        episode_rows.append(
            {
                "episode_index": int(row.episode_index),
                "task_index": int(task_name_to_index[task_name]),
                "task": task_name,
                "length": int(row.length),
            }
        )
    episode_frame = pd.DataFrame(episode_rows)
    total_frames = int(episode_frame["length"].sum())
    task_balance: list[dict[str, int | float | str]] = []
    for task_index, group in episode_frame.groupby("task_index", sort=True):
        lengths = group["length"].to_numpy(dtype=np.int64)
        task_frames = int(lengths.sum())
        task_balance.append(
            {
                "task_index": int(task_index),
                "task": str(group.iloc[0]["task"]),
                "episodes": int(len(group)),
                "frames": task_frames,
                "frame_share": task_frames / total_frames,
                "uniform_task_share": 1.0 / int(episode_frame["task_index"].nunique()),
                "episode_length_min": int(lengths.min()),
                "episode_length_mean": float(lengths.mean()),
                "episode_length_max": int(lengths.max()),
            }
        )

    sample_rows: list[dict[str, int | float]] = []
    device = torch.device(args.device)
    gripper_points = 500
    for task_index, group in episode_frame.groupby("task_index", sort=True):
        group = group.sort_values("episode_index").reset_index(drop=True)
        episode_positions = np.linspace(
            0, len(group) - 1, num=args.samples_per_task, dtype=np.int64
        )
        for sample_rank, episode_position in enumerate(episode_positions.tolist()):
            episode = group.iloc[episode_position]
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            frame_index = int(round((sample_rank + 1) * (length - 1) / (args.samples_per_task + 1)))
            primary_store = open_episode_point_clouds(
                dataset_root / "point_clouds", episode_index
            )
            wrist_store = open_episode_point_clouds(
                dataset_root / "point_clouds_robot0_eye_in_hand", episode_index
            )
            primary = np.asarray(primary_store[frame_index], dtype=np.float32)
            wrist = np.asarray(wrist_store[frame_index], dtype=np.float32)
            fused = compose_point_cloud_views(
                [primary, wrist], gripper_points=gripper_points, fusion="fps"
            )
            tensor = torch.from_numpy(fused).unsqueeze(0).to(device=device)
            _, _, indices = fps_sample_fused_point_cloud(
                tensor, target_points=primary.shape[0], gripper_points=gripper_points
            )
            selected = indices[0, :-gripper_points].cpu().numpy()
            scene_points = primary.shape[0] - gripper_points
            selected_primary = selected < scene_points
            primary_count = int(selected_primary.sum())
            wrist_count = int((~selected_primary).sum())

            primary_voxels = voxel_keys(primary[:-gripper_points, :3], args.voxel_size)
            selected_wrist_xyz = wrist[selected[~selected_primary] - scene_points, :3]
            selected_wrist_voxels = voxel_keys(selected_wrist_xyz, args.voxel_size)
            selected_wrist_novel = selected_wrist_voxels - primary_voxels
            all_wrist_voxels = voxel_keys(wrist[:-gripper_points, :3], args.voxel_size)
            all_wrist_novel = all_wrist_voxels - primary_voxels
            sample_rows.append(
                {
                    "task_index": int(task_index),
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "selected_primary_scene_points": primary_count,
                    "selected_wrist_scene_points": wrist_count,
                    "selected_primary_fraction": primary_count / len(selected),
                    "selected_wrist_fraction": wrist_count / len(selected),
                    "primary_voxels": len(primary_voxels),
                    "wrist_voxels": len(all_wrist_voxels),
                    "wrist_novel_voxels": len(all_wrist_novel),
                    "selected_wrist_voxels": len(selected_wrist_voxels),
                    "selected_wrist_novel_voxels": len(selected_wrist_novel),
                    "selected_wrist_novel_fraction": (
                        len(selected_wrist_novel) / max(len(selected_wrist_voxels), 1)
                    ),
                }
            )

    numeric_keys = [
        "selected_primary_scene_points",
        "selected_wrist_scene_points",
        "selected_primary_fraction",
        "selected_wrist_fraction",
        "primary_voxels",
        "wrist_voxels",
        "wrist_novel_voxels",
        "selected_wrist_voxels",
        "selected_wrist_novel_voxels",
        "selected_wrist_novel_fraction",
    ]
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in sample_rows])),
            "min": float(np.min([float(row[key]) for row in sample_rows])),
            "max": float(np.max([float(row[key]) for row in sample_rows])),
        }
        for key in numeric_keys
    }
    result = {
        "dataset_root": str(dataset_root),
        "total_episodes": int(len(episode_frame)),
        "total_frames": total_frames,
        "task_balance": task_balance,
        "fps_audit": {
            "fusion": "fps",
            "target_points": 10_000,
            "gripper_points": gripper_points,
            "voxel_size_m": args.voxel_size,
            "num_samples": len(sample_rows),
            "aggregate": aggregate,
            "samples": sample_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
