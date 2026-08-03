#!/usr/bin/env python3
"""Prepare a BestMan RGB-D recording for ORB-SLAM3's rgbd_tum example.

Expected input:
  frames.jsonl, metadata.json, color_jpg/, depth_png/
Optional:
  segments.json

Output per selected segment:
  associations.txt, frame_map.jsonl, L515_RGBD.yaml, manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def infer_fps(frames: list[dict], metadata: dict, requested: float | None) -> float:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--fps must be positive")
        return float(requested)

    capture_stats = metadata.get("capture_stats") or {}
    for key in ("camera_timestamp_fps", "host_saved_fps"):
        value = capture_stats.get(key)
        if value is not None and math.isfinite(float(value)) and float(value) > 0:
            return float(value)

    timestamps = [
        float(record["timestamp_ms"])
        for record in frames
        if record.get("timestamp_ms") is not None
    ]
    deltas = [
        (b - a) * 1e-3
        for a, b in zip(timestamps, timestamps[1:])
        if 0 < (b - a) < 1000
    ]
    if not deltas:
        return 15.0
    return 1.0 / statistics.median(deltas)


def load_segments(root: Path, frames: list[dict]) -> list[dict]:
    segments_path = root / "segments.json"
    if segments_path.is_file():
        payload = load_json(segments_path)
        raw = payload.get("segments") if isinstance(payload, dict) else payload
        if isinstance(raw, list) and raw:
            segments = []
            for i, item in enumerate(raw):
                start = int(item["start"])
                end = int(item["end"])
                if start < 0 or end < start:
                    raise ValueError(f"Invalid segment {i}: {item}")
                segments.append({"id": i, "start": start, "end": end})
            return segments

    indices = [int(record.get("index", i)) for i, record in enumerate(frames)]
    return [{"id": 0, "start": min(indices), "end": max(indices)}]


def select_segments(segments: list[dict], selector: str) -> list[dict]:
    if selector == "all":
        return segments
    wanted = {int(value.strip()) for value in selector.split(",") if value.strip()}
    selected = [segment for segment in segments if int(segment["id"]) in wanted]
    missing = wanted - {int(segment["id"]) for segment in selected}
    if missing:
        raise ValueError(f"Unknown segment id(s): {sorted(missing)}")
    return selected


def intrinsics_from(frames: list[dict], metadata: dict) -> dict:
    raw = frames[0].get("intrinsics") or metadata.get("intrinsics")
    if not raw:
        raise KeyError("No intrinsics found in first frame or metadata.json")

    coeffs = list(raw.get("coeffs") or [])
    coeffs += [0.0] * max(0, 5 - len(coeffs))
    return {
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "fx": float(raw["fx"]),
        "fy": float(raw["fy"]),
        "cx": float(raw.get("ppx", raw.get("cx"))),
        "cy": float(raw.get("ppy", raw.get("cy"))),
        "k1": float(coeffs[0]),
        "k2": float(coeffs[1]),
        "p1": float(coeffs[2]),
        "p2": float(coeffs[3]),
        "k3": float(coeffs[4]),
        "model": raw.get("model"),
    }


def get_color_path(record: dict) -> str:
    value = record.get("color_path") or record.get("color_png_path")
    if not value:
        raise KeyError(f"Frame {record.get('index')} has no color path")
    return str(value)


def get_depth_path(record: dict) -> str:
    value = record.get("depth_png_path")
    if not value:
        raise KeyError(
            f"Frame {record.get('index')} has no depth_png_path. "
            "ORB-SLAM3 rgbd_tum expects image depth input."
        )
    return str(value)


def write_yaml(
    path: Path,
    intr: dict,
    fps: float,
    *,
    n_features: int,
    ini_fast: int,
    min_fast: int,
    pseudo_baseline_m: float,
    close_depth_m: float,
    depth_factor: float,
) -> None:
    if pseudo_baseline_m <= 0:
        raise ValueError("--pseudo-baseline-m must be positive")
    if close_depth_m <= 0:
        raise ValueError("--close-depth-m must be positive")
    th_depth = close_depth_m / pseudo_baseline_m

    # cv::imread returns BGR for the saved JPEG/PNG files.
    text = f'''%YAML:1.0
File.version: "1.0"

Camera.type: "PinHole"
Camera1.fx: {intr['fx']:.9f}
Camera1.fy: {intr['fy']:.9f}
Camera1.cx: {intr['cx']:.9f}
Camera1.cy: {intr['cy']:.9f}

Camera1.k1: {intr['k1']:.9f}
Camera1.k2: {intr['k2']:.9f}
Camera1.p1: {intr['p1']:.9f}
Camera1.p2: {intr['p2']:.9f}
Camera1.k3: {intr['k3']:.9f}

Camera.width: {intr['width']}
Camera.height: {intr['height']}
Camera.fps: {max(1, int(round(fps)))}
Camera.RGB: 0

# RGB-D uses these stereo-style values internally for close/far handling.
Stereo.b: {pseudo_baseline_m:.9f}
Stereo.ThDepth: {th_depth:.9f}

# depth_png is uint16 millimetres: raw / 1000 = metres.
RGBD.DepthMapFactor: {depth_factor:.9f}

ORBextractor.nFeatures: {n_features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: {ini_fast}
ORBextractor.minThFAST: {min_fast}

loopClosing: 1

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''
    path.write_text(text, encoding="utf-8")



def ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Cannot create symlink because path exists: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--segments",
        default="all",
        help='Segment IDs such as "0", "0,2", or "all".',
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("continuous", "recorded"),
        default="continuous",
        help=(
            "continuous avoids long waits/gaps; recorded preserves RealSense "
            "timing inside each selected segment."
        ),
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--n-features", type=int, default=1800)
    parser.add_argument("--ini-fast", type=int, default=15)
    parser.add_argument("--min-fast", type=int, default=5)
    parser.add_argument("--pseudo-baseline-m", type=float, default=0.05)
    parser.add_argument("--close-depth-m", type=float, default=2.0)
    parser.add_argument("--depth-factor", type=float, default=1000.0)
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "orbslam3"
    )
    frames = load_jsonl(root / "frames.jsonl")
    metadata = load_json(root / "metadata.json") if (root / "metadata.json").is_file() else {}
    fps = infer_fps(frames, metadata, args.fps)
    intr = intrinsics_from(frames, metadata)
    segments = select_segments(load_segments(root, frames), args.segments)

    by_index = {int(record.get("index", i)): record for i, record in enumerate(frames)}
    if len(by_index) != len(frames):
        raise ValueError("Duplicate frame indices in frames.jsonl")

    output_root.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        segment_id = int(segment["id"])
        segment_dir = output_root / f"segment_{segment_id:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)

        records = [
            by_index[index]
            for index in range(int(segment["start"]), int(segment["end"]) + 1)
            if index in by_index
        ]
        if not records:
            raise RuntimeError(f"Segment {segment_id} contains no saved frames")

        first_recorded_ms = float(records[0].get("timestamp_ms") or 0.0)
        association_lines: list[str] = []
        map_lines: list[str] = []

        for local_index, record in enumerate(records):
            record_index = int(record.get("index", local_index))
            if args.timestamp_mode == "continuous":
                orb_timestamp_s = local_index / fps
            else:
                raw_ms = record.get("timestamp_ms")
                if raw_ms is None:
                    raise KeyError(
                        f"Frame {record_index} has no timestamp_ms; "
                        "use --timestamp-mode continuous."
                    )
                orb_timestamp_s = (float(raw_ms) - first_recorded_ms) * 1e-3

            rgb_rel = get_color_path(record)
            depth_rel = get_depth_path(record)
            for rel in (rgb_rel, depth_rel):
                if not (root / rel).is_file():
                    raise FileNotFoundError(root / rel)

            association_lines.append(
                f"{orb_timestamp_s:.9f} {rgb_rel} "
                f"{orb_timestamp_s:.9f} {depth_rel}\n"
            )
            map_lines.append(
                json.dumps(
                    {
                        "local_index": local_index,
                        "record_index": record_index,
                        "orb_timestamp_s": orb_timestamp_s,
                        "recorded_timestamp_ms": record.get("timestamp_ms"),
                        "color_path": rgb_rel,
                        "depth_png_path": depth_rel,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        (segment_dir / "associations.txt").write_text(
            "".join(association_lines), encoding="utf-8"
        )
        (segment_dir / "frame_map.jsonl").write_text(
            "".join(map_lines), encoding="utf-8"
        )

        # Lightweight segment-specific dataset for playback and ORB-SLAM3.
        # Image directories are symlinked, so no image data is duplicated.
        dataset_view = segment_dir / "dataset"
        dataset_view.mkdir(parents=True, exist_ok=True)
        (dataset_view / "frames.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        if (root / "metadata.json").is_file():
            (dataset_view / "metadata.json").write_text(
                (root / "metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        required_top_dirs = {
            Path(get_color_path(record)).parts[0] for record in records
        } | {
            Path(get_depth_path(record)).parts[0] for record in records
        }
        for directory_name in sorted(required_top_dirs):
            ensure_symlink(dataset_view / directory_name, root / directory_name)
        write_yaml(
            segment_dir / "L515_RGBD.yaml",
            intr,
            fps,
            n_features=args.n_features,
            ini_fast=args.ini_fast,
            min_fast=args.min_fast,
            pseudo_baseline_m=args.pseudo_baseline_m,
            close_depth_m=args.close_depth_m,
            depth_factor=args.depth_factor,
        )
        manifest = {
            "dataset_root": str(root),
            "segment_id": segment_id,
            "segment_start": int(segment["start"]),
            "segment_end": int(segment["end"]),
            "frame_count": len(records),
            "fps": fps,
            "timestamp_mode": args.timestamp_mode,
            "intrinsics": intr,
            "association_file": str(segment_dir / "associations.txt"),
            "frame_map": str(segment_dir / "frame_map.jsonl"),
            "settings": str(segment_dir / "L515_RGBD.yaml"),
            "dataset_view": str(dataset_view),
        }
        (segment_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Prepared segment {segment_id}: {len(records)} frames\n"
            f"  association: {segment_dir / 'associations.txt'}\n"
            f"  settings:    {segment_dir / 'L515_RGBD.yaml'}\n"
            f"  frame map:   {segment_dir / 'frame_map.jsonl'}\n"
            f"  dataset:     {dataset_view}"
        )


if __name__ == "__main__":
    main()
