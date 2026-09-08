#!/usr/bin/env python3
"""Export one world-frame expert EEF trajectory diagnostic per RLBench task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to <dataset-root>/trajectory_ply.",
    )
    parser.add_argument(
        "--episode-per-task", type=int, default=0,
        help="Zero-based episode within each task (default: first episode).",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return name.strip("._") or "task"


def write_ply(path: Path, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    count = len(xyz)
    if count == 0:
        raise ValueError("Cannot write an empty trajectory")
    # Blue -> red encodes time while retaining the original world coordinates.
    if count == 1:
        colors = np.asarray([[255, 40, 40]], dtype=np.uint8)
    else:
        t = np.linspace(0.0, 1.0, count, dtype=np.float32)
        colors = np.stack(
            [np.rint(255.0 * t), np.full(count, 80.0), np.rint(255.0 * (1.0 - t))],
            axis=1,
        ).astype(np.uint8)
    edges = max(0, count - 1)
    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment coordinate_frame world\n")
        file.write("comment trajectory_color blue_to_red_time\n")
        file.write(f"element vertex {count}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write(f"element edge {edges}\nproperty int vertex1\nproperty int vertex2\n")
        file.write("end_header\n")
        for point, color in zip(xyz, colors, strict=True):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for index in range(edges):
            file.write(f"{index} {index + 1}\n")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = (args.output_dir or root / "trajectory_ply").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Import pyarrow only at runtime; the RLBench environment provides it.
    import pyarrow.parquet as pq

    task_table = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    task_names = {
        int(index): str(name)
        for index, name in zip(task_table["task_index"], task_table["__index_level_0__"])
    }
    task_indices_by_name = {name: index for index, name in task_names.items()}
    episode_table = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    episode_to_task = {}
    for episode_index, tasks in zip(episode_table["episode_index"], episode_table["tasks"]):
        if not tasks:
            continue
        task_name = str(tasks[0])
        if task_name not in task_indices_by_name:
            raise RuntimeError(f"Unknown task name in episode metadata: {task_name!r}")
        episode_to_task[int(episode_index)] = task_indices_by_name[task_name]

    selected = {}
    for episode_index in sorted(episode_to_task):
        task_index = episode_to_task[episode_index]
        if task_index not in selected:
            selected[task_index] = episode_index

    records = []
    for task_index in sorted(task_names):
        candidates = [
            episode for episode, mapped_task in episode_to_task.items()
            if mapped_task == task_index
        ]
        if len(candidates) <= args.episode_per_task:
            raise RuntimeError(
                f"Task {task_names[task_index]!r} has only {len(candidates)} episodes"
            )
        episode_index = sorted(candidates)[args.episode_per_task]
        pose_path = root / "world_ee_poses" / f"episode_{episode_index:06d}.npy"
        poses = np.load(pose_path, allow_pickle=False)
        xyz = np.asarray(poses[:, :3], dtype=np.float32)
        name = task_names[task_index]
        ply_path = output / f"{safe_name(name)}.ply"
        write_ply(ply_path, xyz)
        records.append({
            "task_index": task_index,
            "task": name,
            "episode_index": episode_index,
            "frame_count": int(len(xyz)),
            "ply": str(ply_path),
        })

    manifest = {
        "dataset_root": str(root),
        "selection": "first episode per task" if args.episode_per_task == 0 else f"episode offset {args.episode_per_task} per task",
        "coordinate_frame": "world",
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "tasks": len(records)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
