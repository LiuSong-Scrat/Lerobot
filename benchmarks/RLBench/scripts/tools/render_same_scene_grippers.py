#!/usr/bin/env python3
"""Create two PLYs with identical scene points and different gripper templates."""

import argparse
import sys
from pathlib import Path

import numpy as np

from _rlbench_tool_paths import LEROBOT_ROOT as ROOT
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "song_real_libero" / "scripts"))

from libero_setting.libero_pointcloud_utils import (  # noqa: E402
    create_rlbench_panda_gripper_points,
)
from rlbench_reap_gripper import (  # noqa: E402
    RLBENCH_PANDA_PHYSICAL_MAX_WIDTH,
    create_rlbench_reap_points_from_physical_width,
)


DEFAULT_SOURCE = ROOT / "benchmarks/RLBench/outputs/eval/eval_history/eval_20260809_183253_10tasks_100episodes/close_box/action_visualizations/episode_061/frame_000032_model_call_0003_prob.ply"
DEFAULT_OUTPUT = ROOT / "benchmarks/RLBench/outputs/gripper_same_scene_compare"


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    vertex_count = next(int(line.split()[2]) for line in lines[:end] if line.startswith("element vertex"))
    properties = [line.split()[2] for line in lines[:end] if line.startswith("property ")]
    point_kind_index = properties.index("point_kind")
    red_index, green_index, blue_index = (properties.index(name) for name in ("red", "green", "blue"))
    rows = []
    for line in lines[end + 1 : end + 1 + vertex_count]:
        values = line.split()
        rows.append([float(values[0]), float(values[1]), float(values[2]), int(values[red_index]), int(values[green_index]), int(values[blue_index]), int(values[point_kind_index])])
    data = np.asarray(rows, dtype=np.float64)
    return data[:, :3], data[:, 3:6].astype(np.uint8), data[:, 6].astype(np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, kinds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment same point_kind=0 scene used in both files\n")
        file.write("comment appended point_kind=3 gripper template\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property uchar point_kind\nend_header\n")
        for point, color, kind in zip(xyz, rgb, kinds, strict=True):
            file.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {int(color[0])} {int(color[1])} {int(color[2])} {int(kind)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source
    output = args.output
    source_xyz, source_rgb, source_kinds = read_ply(source)
    scene_mask = source_kinds == 0
    scene_xyz = source_xyz[scene_mask]
    scene_rgb = source_rgb[scene_mask]
    rng_a = np.random.default_rng(20260813)
    rng_b = np.random.default_rng(20260813)
    pose = np.zeros(6, dtype=np.float32)
    # Both are open at their respective maximum width and share the same EEF frame.
    rlbench = create_rlbench_panda_gripper_points(1.0, pose, 500, rng_a)
    reap = create_rlbench_reap_points_from_physical_width(
        RLBENCH_PANDA_PHYSICAL_MAX_WIDTH, pose, 500, rng_b
    )
    scene_kind = np.zeros(len(scene_xyz), dtype=np.uint8)
    gripper_kind = np.full(500, 3, dtype=np.uint8)
    write_ply(output / "same_scene_rlbench_panda.ply", np.vstack((scene_xyz, rlbench)), np.vstack((scene_rgb, np.tile([30, 120, 255], (500, 1)))), np.concatenate((scene_kind, gripper_kind)))
    write_ply(output / "same_scene_rlbench_reap_v4.ply", np.vstack((scene_xyz, reap)), np.vstack((scene_rgb, np.tile([240, 90, 40], (500, 1)))), np.concatenate((scene_kind, gripper_kind)))
    print(f"source={source}")
    print(f"scene_points={len(scene_xyz)} appended_gripper_points=500")
    print(f"rlbench={output / 'same_scene_rlbench_panda.ply'}")
    print(f"reap={output / 'same_scene_rlbench_reap_v4.ply'}")


if __name__ == "__main__":
    main()
