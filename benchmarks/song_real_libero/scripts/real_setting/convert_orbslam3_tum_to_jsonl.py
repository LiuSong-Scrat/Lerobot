#!/usr/bin/env python3
"""Convert ORB-SLAM3 CameraTrajectory.txt to BestMan camera-pose JSONL.

TUM rows are:
  timestamp tx ty tz qx qy qz qw

ORB-SLAM3 saves Twc: current camera -> ORB-SLAM world. The script maps each
ORB timestamp back to record_index using frame_map.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quaternion_xyzw_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid quaternion: {q.tolist()}")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_tum(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    poses: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split()
        if len(values) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 columns")
        t, tx, ty, tz, qx, qy, qz, qw = map(float, values)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quaternion_xyzw_to_rotation(qx, qy, qz, qw)
        matrix[:3, 3] = [tx, ty, tz]
        poses.append({"timestamp": t, "matrix": matrix})
    if not poses:
        raise RuntimeError(f"No poses found in {path}")
    poses.sort(key=lambda item: item["timestamp"])
    return poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_map", type=Path)
    parser.add_argument("trajectory_tum", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--timestamp-tolerance-s",
        type=float,
        default=5e-5,
        help="Nearest timestamp matching tolerance.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write only tracked frames instead of failing on missing poses.",
    )
    parser.add_argument(
        "--rebase-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make the first exported camera pose exactly identity.",
    )
    args = parser.parse_args()

    frame_map = load_jsonl(args.frame_map)
    tum_poses = load_tum(args.trajectory_tum)
    tum_times = np.asarray([pose["timestamp"] for pose in tum_poses], dtype=np.float64)

    matched: list[tuple[dict, np.ndarray, float]] = []
    missing: list[int] = []

    for frame in frame_map:
        timestamp = float(frame["orb_timestamp_s"])
        position = int(np.searchsorted(tum_times, timestamp))
        candidates = []
        if position < len(tum_times):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        if not candidates:
            missing.append(int(frame["record_index"]))
            continue
        best = min(candidates, key=lambda i: abs(float(tum_times[i]) - timestamp))
        error = abs(float(tum_times[best]) - timestamp)
        if error > args.timestamp_tolerance_s:
            missing.append(int(frame["record_index"]))
            continue
        matched.append((frame, tum_poses[best]["matrix"].copy(), error))

    if missing and not args.allow_missing:
        raise RuntimeError(
            f"ORB-SLAM3 did not save poses for {len(missing)} frame(s); "
            f"first missing record indices: {missing[:20]}. "
            "ORB-SLAM3 omits frames where tracking failed. Improve calibration, "
            "texture, motion speed, or process a shorter continuous segment."
        )

    if not matched:
        raise RuntimeError("No frame timestamps matched the ORB-SLAM3 trajectory")

    first_inverse = (
        np.linalg.inv(matched[0][1])
        if args.rebase_first
        else np.eye(4, dtype=np.float64)
    )

    output_lines = []
    for frame, matrix, error in matched:
        rebased = first_inverse @ matrix
        output_lines.append(
            json.dumps(
                {
                    "record_index": int(frame["record_index"]),
                    "timestamp_ms": frame.get("recorded_timestamp_ms"),
                    "camera_to_tracking": rebased.tolist(),
                    "tracking_source": "orb_slam3_rgbd_tum_optimized",
                    "valid": True,
                    "transform_direction": "camera_to_tracking",
                    "training_ground_truth": False,
                    "orb_timestamp_s": float(frame["orb_timestamp_s"]),
                    "orb_timestamp_match_error_s": float(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(output_lines), encoding="utf-8")
    print(
        f"Wrote {len(output_lines)} poses to {args.output_jsonl}; "
        f"missing={len(missing)}, rebase_first={args.rebase_first}"
    )


if __name__ == "__main__":
    main()
