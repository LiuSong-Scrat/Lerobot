#!/usr/bin/env python3
"""Export simulator camera points beside Panda virtual-gripper points.

The simulator portion is the stored front-camera cloud before the virtual
gripper tail is appended.  The virtual Panda cloud is generated in the same
EEF frame, so the two geometries can be inspected directly in MeshLab or
CloudCompare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from _rlbench_tool_paths import LEROBOT_ROOT as ROOT
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "song_real_libero" / "scripts"))

from lerobot.policies.smolvla.song_pointseg import open_episode_point_clouds  # noqa: E402
from libero_setting.libero_pointcloud_utils import (  # noqa: E402
    RLBENCH_PANDA_MAX_WIDTH,
    create_rlbench_panda_gripper_points,
)
from rlbench_reap_gripper import (  # noqa: E402
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    LIBERO_REAP_TEMPLATE_MAX_WIDTH,
    create_rlbench_reap_points_from_physical_width,
)


DEFAULT_DATASET = ROOT / "benchmarks/RLBench/datasets/rlbench_10tasks_100traj_lerobot_raw_expert_target_20260822_214128"
DEFAULT_OUTPUT = ROOT / "benchmarks/RLBench/outputs/sim_gripper_vs_panda_opening_compare"


def write_ply(
    path: Path,
    simulator_xyzrgb: np.ndarray,
    virtual_xyz: np.ndarray,
    width_percent: float,
    template: str,
    offset_y: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sim = np.asarray(simulator_xyzrgb, dtype=np.float32)
    virtual = np.asarray(virtual_xyz, dtype=np.float32)
    sim_rgb = np.clip(np.rint(sim[:, 3:6]), 0, 255).astype(np.uint8)
    # Simulator camera points keep their captured RGB; the virtual Panda is cyan.
    panda_rgb = np.tile(np.array([[35, 190, 235]], dtype=np.uint8), (len(virtual), 1))
    xyz = np.concatenate((sim[:, :3], virtual[:, :3]), axis=0)
    rgb = np.concatenate((sim_rgb, panda_rgb), axis=0)
    kinds = np.concatenate(
        (
            np.zeros(len(sim), dtype=np.uint8),
            np.full(len(virtual), 3, dtype=np.uint8),
        )
    )

    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment point_kind=0 simulator front-camera cloud\n")
        file.write(f"comment point_kind=3 virtual {template} gripper\n")
        file.write(f"comment virtual_offset_y_m={offset_y:.6f}\n")
        opening_scale = (
            LIBERO_REAP_OPENING_MAX_WIDTH
            if template == "reap"
            else RLBENCH_PANDA_MAX_WIDTH
        )
        file.write(f"comment virtual_opening_width_m={width_percent * opening_scale:.6f}\n")
        file.write(f"comment virtual_opening_percent={width_percent:.3f}\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property uchar point_kind\nend_header\n")
        for point, color, kind in zip(xyz, rgb, kinds, strict=True):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(kind)}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--gripper-points",
        type=int,
        default=1000,
        help="Number of virtual-template points written to each PLY.",
    )
    parser.add_argument(
        "--source-gripper-points",
        type=int,
        default=1000,
        help="Number of virtual-gripper points at the tail of the source dataset cloud.",
    )
    parser.add_argument("--template", choices=["panda", "reap"], default="reap")
    parser.add_argument(
        "--virtual-offset-y",
        type=float,
        default=0.0,
        help="Translate only the virtual gripper along EEF Y before writing PLY.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT.parent / (
            "sim_gripper_vs_reap_opening_compare"
            if args.template == "reap"
            else DEFAULT_OUTPUT.name
        )
    source = open_episode_point_clouds(args.dataset_root / "point_clouds", args.episode)
    if source.ndim != 3 or source.shape[-1] < 6:
        raise ValueError(f"Expected episode point cloud shape (frames, points, 6+), got {source.shape}")
    if not 0 <= args.frame < source.shape[0]:
        raise IndexError(f"frame {args.frame} outside episode length {source.shape[0]}")
    if not 0 < args.source_gripper_points < source.shape[1]:
        raise ValueError(
            f"--source-gripper-points must be in [1, {source.shape[1] - 1}]"
        )

    # Collection appends the virtual gripper at the tail. The prefix is the
    # actual simulator front-camera cloud, including any visible real gripper.
    simulator_cloud = np.asarray(
        source[args.frame, :-args.source_gripper_points, :6], dtype=np.float32
    )
    widths = (0.0, 0.25, 0.50, 0.75, 1.0)
    max_width = (
        LIBERO_REAP_TEMPLATE_MAX_WIDTH
        if args.template == "reap"
        else RLBENCH_PANDA_MAX_WIDTH
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for width in widths:
        rng = np.random.default_rng(20260823)
        if args.template == "reap":
            virtual_points = create_rlbench_reap_points_from_physical_width(
                width * LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
                np.array([0.0, args.virtual_offset_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                args.gripper_points,
                rng,
            )
        else:
            virtual_points = create_rlbench_panda_gripper_points(
                width,
                np.array([0.0, args.virtual_offset_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                args.gripper_points,
                rng,
            )
        output = args.output_dir / f"episode_{args.episode:06d}_frame_{args.frame:06d}_{args.template}_{int(width * 100):03d}pct.ply"
        write_ply(output, simulator_cloud, virtual_points, width, args.template, args.virtual_offset_y)
        opening_scale = (
            LIBERO_REAP_OPENING_MAX_WIDTH
            if args.template == "reap"
            else RLBENCH_PANDA_MAX_WIDTH
        )
        records.append({"width_percent": width, "width_m": width * opening_scale, "file": str(output)})

    manifest = {
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "frame": args.frame,
        "coordinate_frame": "current EEF frame",
        "simulator_points": int(len(simulator_cloud)),
        "virtual_template_points": int(args.gripper_points),
        "virtual_offset_y_m": float(args.virtual_offset_y),
        "template": args.template,
        "template_version": (
            LIBERO_GRIPPER_TEMPLATE_VERSION
            if args.template == "reap"
            else "rlbench_panda_tip_ttm_v1"
        ),
        "virtual_max_width_m": float(max_width),
        "virtual_opening_max_width_m": (
            LIBERO_REAP_OPENING_MAX_WIDTH
            if args.template == "reap"
            else RLBENCH_PANDA_MAX_WIDTH
        ),
        "virtual_width_normalization_max_m": (
            LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX
            if args.template == "reap"
            else RLBENCH_PANDA_MAX_WIDTH
        ),
        "virtual_gripper_len_m": (
            LIBERO_REAP_GRIPPER_LEN if args.template == "reap" else 0.0
        ),
        "virtual_gripper_local_offset_m": (
            [0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN]
            if args.template == "reap"
            else [0.0, 0.0, 0.0]
        ),
        "point_kind": {"0": "simulator front-camera cloud", "3": f"virtual {args.template} gripper"},
        "files": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
