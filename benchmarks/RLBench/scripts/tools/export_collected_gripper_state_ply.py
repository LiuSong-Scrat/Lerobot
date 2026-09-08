#!/usr/bin/env python3
"""Export real collected RLBench gripper-state point-cloud snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
import pyarrow.parquet as pq

from _rlbench_tool_paths import LEROBOT_ROOT as ROOT
DEFAULT_DATASET = ROOT / "benchmarks/RLBench/outputs/one_close_box_gripper_open_close_20260823"
import sys

sys.path.insert(0, str(ROOT / "benchmarks" / "song_real_libero" / "scripts"))

from rlbench_reap_gripper import (
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    LIBERO_REAP_TEMPLATE_MAX_WIDTH,
    create_rlbench_reap_points_from_physical_width,
)


def write_ply(
    path: Path,
    simulator: np.ndarray,
    virtual: np.ndarray,
    state: float,
    source_frame: int,
    template: str,
) -> None:
    simulator = np.asarray(simulator, dtype=np.float32)
    virtual = np.asarray(virtual, dtype=np.float32)
    sim_rgb = np.clip(np.rint(simulator[:, 3:6]), 0, 255).astype(np.uint8)
    # Match LIBERO's canonical REAP gripper color.
    virtual_rgb = np.tile(np.array([[204, 51, 51]], dtype=np.uint8), (len(virtual), 1))
    xyz = np.concatenate((simulator[:, :3], virtual[:, :3]), axis=0)
    rgb = np.concatenate((sim_rgb, virtual_rgb), axis=0)
    kinds = np.concatenate((np.zeros(len(simulator), dtype=np.uint8), np.full(len(virtual), 3, dtype=np.uint8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment point_kind=0 actual simulator front-camera cloud\n")
        file.write(f"comment point_kind=3 {template} virtual gripper at the same EEF pose\n")
        file.write(f"comment simulator_gripper_width_m={state:.6f}\n")
        file.write(f"comment source_frame={source_frame}\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property uchar point_kind\nend_header\n")
        for point, color, kind in zip(xyz, rgb, kinds, strict=True):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(kind)}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--source-gripper-points", type=int, default=500)
    parser.add_argument("--virtual-gripper-points", type=int, default=None)
    parser.add_argument("--virtual-template", choices=["captured", "reap"], default="captured")
    parser.add_argument("--simulator-points", type=int, default=None)
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated frame indices to export.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset_root / (
        "gripper_state_ply_reap_500k"
        if args.virtual_template == "reap" and args.virtual_gripper_points == 500000
        else "gripper_state_ply"
    )
    virtual_points_count = (
        args.source_gripper_points
        if args.virtual_gripper_points is None
        else int(args.virtual_gripper_points)
    )
    if virtual_points_count <= 0:
        raise ValueError("--virtual-gripper-points must be positive")
    actions = np.load(args.dataset_root / "raw_expert_actions_full" / f"episode_{args.episode:06d}.npy")
    state_files = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    saved_widths = None
    if state_files:
        state_rows = []
        for state_file in state_files:
            column = pq.read_table(state_file, columns=["observation.state"])["observation.state"]
            state_rows.extend(column.to_pylist())
        if state_rows:
            saved_widths = np.asarray(state_rows, dtype=np.float32)[:, 9]
    zarr_obj = zarr.open(str(args.dataset_root / "point_clouds" / f"episode_{args.episode:06d}.zarr"), mode="r")
    xyz = np.asarray(zarr_obj["xyz"])
    rgb = np.asarray(zarr_obj["rgb"])
    if xyz.ndim != 3 or rgb.shape[:2] != xyz.shape[:2]:
        raise ValueError(f"Unexpected stored point-cloud arrays: xyz={xyz.shape}, rgb={rgb.shape}")
    if xyz.shape[-1] != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected xyz and rgb arrays with three channels")
    if len(actions) < len(xyz):
        raise ValueError("Action trajectory is shorter than the stored point-cloud trajectory")
    gripper_widths = (
        saved_widths[: len(xyz)]
        if saved_widths is not None and len(saved_widths) >= len(xyz)
        else np.asarray(actions[: len(xyz), -1], dtype=np.float32) * 0.08
    )
    if not 0 < args.source_gripper_points < xyz.shape[1]:
        raise ValueError("Invalid --source-gripper-points for this dataset")
    simulator_points_count = (
        xyz.shape[1] - args.source_gripper_points
        if args.simulator_points is None
        else int(args.simulator_points)
    )
    if simulator_points_count <= 0:
        raise ValueError("--simulator-points must be positive")

    # Stored clouds are xyz/rgb arrays; the collector appends the virtual tail
    # after the simulator camera cloud, in the same current-EEF frame.
    simulator = np.concatenate((xyz, rgb), axis=-1).astype(np.float32)
    if args.frames:
        source_frames = sorted(set(int(value.strip()) for value in args.frames.split(",") if value.strip()))
    else:
        source_frames = [0]
        for frame in range(1, len(xyz)):
            if abs(float(gripper_widths[frame]) - float(gripper_widths[frame - 1])) > 1e-6:
                source_frames.extend([frame - 1, frame])
        source_frames.extend([len(xyz) - 1])
    source_frames = sorted(set(frame for frame in source_frames if 0 <= frame < len(xyz)))

    records = []
    for frame in source_frames:
        # The saved point cloud has the same tail convention as collection.
        full = simulator[frame]
        scene = full[:-args.source_gripper_points]
        if simulator_points_count != len(scene):
            rng_scene = np.random.default_rng(20260823 + frame * 17)
            scene_indices = rng_scene.choice(
                len(scene),
                size=simulator_points_count,
                replace=len(scene) < simulator_points_count,
            )
            scene = scene[scene_indices]
        state = float(gripper_widths[frame])
        width_percent = float(
            np.clip(state / LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX, 0.0, 1.0)
        )
        if args.virtual_template == "reap":
            rng = np.random.default_rng(20260823 + frame)
            virtual = create_rlbench_reap_points_from_physical_width(
                state,
                np.zeros(6, dtype=np.float32),
                virtual_points_count,
                rng,
            ).astype(np.float32)
        else:
            virtual = full[-args.source_gripper_points:]
        state_name = "open" if width_percent > 0.99 else ("closed" if width_percent < 0.01 else f"mid_{width_percent:.3f}")
        path = output_dir / f"episode_{args.episode:06d}_frame_{frame:06d}_{state_name}.ply"
        write_ply(path, scene, virtual, state, frame, args.virtual_template)
        records.append({"frame": frame, "state": state_name, "gripper_open": state, "file": str(path)})

    manifest = {
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "coordinate_frame": "current EEF frame",
        "simulator_points_per_frame": int(xyz.shape[1] - args.source_gripper_points),
        "simulator_output_points_per_frame": int(simulator_points_count),
        "simulator_output_sampling": (
            "original_unique_points"
            if simulator_points_count == int(xyz.shape[1] - args.source_gripper_points)
            else (
                "without_replacement_downsampling_from_collected_camera_cloud"
                if simulator_points_count < int(xyz.shape[1] - args.source_gripper_points)
                else "with_replacement_upsampling_from_collected_camera_cloud"
            )
        ),
        "virtual_template": args.virtual_template,
        "virtual_template_version": (
            LIBERO_GRIPPER_TEMPLATE_VERSION
            if args.virtual_template == "reap"
            else "captured_from_source_dataset"
        ),
        "virtual_gripper_points_per_frame": int(virtual_points_count),
        "virtual_width_normalization_max_m": LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
        "virtual_template_max_width_m": LIBERO_REAP_TEMPLATE_MAX_WIDTH,
        "virtual_opening_max_width_m": LIBERO_REAP_OPENING_MAX_WIDTH,
        "virtual_gripper_len_m": LIBERO_REAP_GRIPPER_LEN,
        "virtual_gripper_local_offset_m": [0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN],
        "virtual_gripper_backward_shift_from_libero_m": 0.03,
        "point_kind": {
            "0": "actual simulator front-camera cloud",
            "3": f"{args.virtual_template} virtual gripper",
        },
        "snapshots": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
