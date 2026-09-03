#!/usr/bin/env python3
"""Export selected packed RLBench point-cloud frames as task diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/task_pointcloud_ply.",
    )
    parser.add_argument(
        "--episode-per-task",
        type=int,
        default=0,
        help="Zero-based starting episode offset within each task (default: first episode).",
    )
    parser.add_argument(
        "--episode-stride",
        type=int,
        default=None,
        help=(
            "When set, export every Nth episode within each task, starting at "
            "--episode-per-task. Without it, export only the starting episode."
        ),
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=None,
        help=(
            "Maximum number of selected episodes per task. This is applied after "
            "--episode-per-task and --episode-stride."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional internal RLBench task names to export, e.g. water_plants.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Starting frame within each selected episode (default: first frame).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "When set, export every Nth frame through each selected episode, "
            "starting at --frame-index. Without it, export only the starting frame."
        ),
    )
    parser.add_argument(
        "--gripper-points",
        type=int,
        default=500,
        help="Number of virtual-gripper points appended to each packed frame.",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return name.strip("._") or "task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    task_name: str,
    task_description: str,
    episode_index: int,
    frame_index: int,
    gripper_points: int,
    gripper_template: str,
) -> int:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError(f"Invalid xyz/rgb shapes: {xyz.shape} and {rgb.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("Point cloud contains non-finite xyz values")
    if not 0 <= gripper_points <= len(xyz):
        raise ValueError(f"Invalid gripper point count: {gripper_points}")

    point_kind = np.zeros(len(xyz), dtype=np.uint8)
    if gripper_points:
        point_kind[-gripper_points:] = 3

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("point_kind", "u1"),
        ]
    )
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    vertices["point_kind"] = point_kind

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment coordinate_frame current_eef",
            f"comment task_name {task_name}",
            f"comment task_description {task_description}",
            f"comment episode_index {episode_index}",
            f"comment frame_index {frame_index}",
            f"comment scene_points {len(xyz) - gripper_points}",
            f"comment virtual_gripper_points {gripper_points}",
            f"comment gripper_template {gripper_template}",
            "comment point_kind 0=scene 3=virtual_gripper",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar point_kind",
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)

    expected_size = len(header) + len(vertices) * vertex_dtype.itemsize
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"PLY size verification failed for {path}: "
            f"{path.stat().st_size} != {expected_size}"
        )
    return expected_size


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = (args.output_dir or root / "task_pointcloud_ply").expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if args.episode_per_task < 0 or args.frame_index < 0:
        raise ValueError("Episode offset and frame index must be non-negative")
    if args.episode_stride is not None and args.episode_stride <= 0:
        raise ValueError("Episode stride must be positive")
    if args.episode_count is not None and args.episode_count <= 0:
        raise ValueError("Episode count must be positive")
    if args.frame_stride is not None and args.frame_stride <= 0:
        raise ValueError("Frame stride must be positive")

    # Runtime imports let callers choose the same environment used to pack Zarr v2.
    import pyarrow.parquet as pq
    import zarr

    conversion = json.loads((root / "meta" / "rlbench_conversion.json").read_text())
    internal_task_names = [str(name) for name in conversion["tasks"]]
    gripper_template = str(conversion["gripper_template"])
    requested_tasks = None if args.tasks is None else set(map(str, args.tasks))
    if requested_tasks is not None:
        unknown_tasks = sorted(requested_tasks.difference(internal_task_names))
        if unknown_tasks:
            raise ValueError(
                f"Unknown task names {unknown_tasks}; available tasks: {internal_task_names}"
            )

    task_table = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    task_descriptions = {
        int(index): str(description)
        for index, description in zip(
            task_table["task_index"], task_table["__index_level_0__"], strict=True
        )
    }
    if sorted(task_descriptions) != list(range(len(internal_task_names))):
        raise RuntimeError("Task indices do not match rlbench_conversion task order")

    description_to_task_index = {
        description: task_index for task_index, description in task_descriptions.items()
    }
    episodes_by_task = {task_index: [] for task_index in task_descriptions}
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError("No episode metadata parquet files found")
    for episode_file in episode_files:
        episode_table = pq.read_table(
            episode_file, columns=["episode_index", "tasks", "length"]
        ).to_pydict()
        for episode_index, descriptions, length in zip(
            episode_table["episode_index"],
            episode_table["tasks"],
            episode_table["length"],
            strict=True,
        ):
            if not descriptions:
                continue
            description = str(descriptions[0])
            task_index = description_to_task_index.get(description)
            if task_index is None:
                raise RuntimeError(f"Unknown task description: {description!r}")
            episodes_by_task[task_index].append((int(episode_index), int(length)))

    output.mkdir(parents=True)
    records = []
    for task_index in sorted(task_descriptions):
        task_name = internal_task_names[task_index]
        if requested_tasks is not None and task_name not in requested_tasks:
            continue
        candidates = sorted(episodes_by_task[task_index])
        if len(candidates) <= args.episode_per_task:
            raise RuntimeError(
                f"Task {task_index} has only {len(candidates)} episodes; "
                f"cannot select offset {args.episode_per_task}"
            )
        if args.episode_stride is None:
            episode_offsets = [args.episode_per_task]
        else:
            episode_offsets = range(
                args.episode_per_task, len(candidates), args.episode_stride
            )
        if args.episode_count is not None:
            episode_offsets = list(episode_offsets)[: args.episode_count]
        for episode_offset in episode_offsets:
            episode_index, episode_length = candidates[episode_offset]
            if args.frame_index >= episode_length:
                raise RuntimeError(
                    f"Frame {args.frame_index} is outside episode {episode_index} "
                    f"with length {episode_length}"
                )

            zarr_path = root / "point_clouds" / f"episode_{episode_index:06d}.zarr"
            group = zarr.open_group(str(zarr_path), mode="r")
            task_description = task_descriptions[task_index]
            if args.frame_stride is None:
                frame_indices = [args.frame_index]
            else:
                frame_indices = range(
                    args.frame_index, episode_length, args.frame_stride
                )
            for frame_index in frame_indices:
                xyz = np.asarray(group["xyz"][frame_index], dtype=np.float32)
                rgb = np.asarray(group["rgb"][frame_index], dtype=np.uint8)
                if args.frame_stride is not None:
                    task_dir = output / f"{task_index:02d}_{safe_name(task_name)}"
                    filename = (
                        f"episode_{episode_index:06d}_frame_{frame_index:06d}.ply"
                    )
                    ply_path = task_dir / filename
                elif args.episode_stride is None:
                    ply_path = output / f"{task_index:02d}_{safe_name(task_name)}.ply"
                else:
                    filename = (
                        f"{task_index:02d}_{safe_name(task_name)}"
                        f"_traj_{episode_offset:03d}_episode_{episode_index:06d}.ply"
                    )
                    ply_path = output / filename
                file_size = write_binary_ply(
                    ply_path,
                    xyz,
                    rgb,
                    task_name=task_name,
                    task_description=task_description,
                    episode_index=episode_index,
                    frame_index=frame_index,
                    gripper_points=args.gripper_points,
                    gripper_template=gripper_template,
                )
                records.append(
                    {
                        "task_index": task_index,
                        "task_name": task_name,
                        "task_description": task_description,
                        "episode_offset_within_task": episode_offset,
                        "episode_index": episode_index,
                        "episode_length": episode_length,
                        "frame_index": frame_index,
                        "total_points": int(len(xyz)),
                        "scene_points": int(len(xyz) - args.gripper_points),
                        "virtual_gripper_points": args.gripper_points,
                        "ply": str(ply_path.relative_to(output)),
                        "bytes": file_size,
                        "sha256": sha256(ply_path),
                    }
                )
                print(
                    f"[{task_index:02d}] {task_name}: task_episode={episode_offset} "
                    f"episode={episode_index} frame={frame_index} "
                    f"points={len(xyz)} -> {ply_path.relative_to(output)}"
                )

    manifest = {
        "dataset_root": str(root),
        "selection": {
            "episode_start_offset_per_task": args.episode_per_task,
            "episode_stride": args.episode_stride,
            "episode_count_per_task": args.episode_count,
            "tasks": None if requested_tasks is None else sorted(requested_tasks),
            "frame_start_index": args.frame_index,
            "frame_stride": args.frame_stride,
        },
        "coordinate_frame": "current_eef",
        "point_layout": "scene points followed by virtual gripper points",
        "point_kind": {"0": "scene", "3": "virtual_gripper"},
        "gripper_template": gripper_template,
        "records": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(records)} task PLY files to {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
