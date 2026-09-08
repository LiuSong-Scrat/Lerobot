#!/usr/bin/env python3
"""Audit/migrate mixture-setting virtual-gripper tails to the RH20T v3 contract.

The operation is deliberately narrow: only the final 500 XYZ/RGB entries of
each Zarr frame and point-cloud metadata are writable.  Scene points, Parquet,
images, poses and episode indices are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from virtual_gripper_geometry import (
    CanonicalGripperParameters,
    DEFAULT_GRIPPER_POINTS,
    GRIPPER_COLOR_RGB,
    GRIPPER_CONTRACT,
    allocate_surface_counts,
    sample_gripper_width,
)


DEFAULT_ROOT = Path(
    "/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting"
)
DATASETS = {
    "libero": Path("libero_data/lerobot_dataset"),
    "rh20t": Path("rh20t_data/rh20t_robot_lerobot_dataset"),
    "static_human": Path(
        "real_data/static_human_data/"
        "wepvla_v043_doubleflow_static_franka_fold_trash_data/lerobot_dataset"
    ),
}
PURE_RED = GRIPPER_COLOR_RGB.astype(np.uint8)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def point_cloud_directories(dataset: Path) -> list[Path]:
    result = []
    for child in sorted(dataset.iterdir()):
        if not child.is_dir() or not child.name.startswith("point_clouds"):
            continue
        if next(child.glob("episode_*.zarr"), None) is not None:
            result.append(child)
    if not result:
        raise FileNotFoundError(f"No point-cloud Zarr directories found in {dataset}")
    return result


def episode_records(dataset: Path) -> dict[int, dict]:
    records = {}
    for path in sorted((dataset / "meta/episodes").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            records[int(row["episode_index"])] = row
    return records


def episode_widths_m(dataset: Path, record: dict) -> np.ndarray:
    data_path = dataset / (
        f"data/chunk-{int(record['data/chunk_index']):03d}/"
        f"file-{int(record['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(
        data_path,
        columns=["episode_index", "observation.state"],
        filters=[("episode_index", "=", int(record["episode_index"]))],
        use_threads=False,
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 1 or len(states) != int(record["length"]):
        raise ValueError(f"Invalid state rows for episode {record['episode_index']}: {states.shape}")
    widths = states[:, -1]
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError(f"Invalid physical widths for episode {record['episode_index']}")
    return widths


def update_metadata(directory: Path, *, geometry_updated: bool | None) -> None:
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.update(
        contains_gripper_template=True,
        gripper_points=DEFAULT_GRIPPER_POINTS,
        gripper_at_tail=True,
        virtual_gripper_contract=GRIPPER_CONTRACT,
        virtual_gripper_template="rh20t_canonical_four_box_v3",
        virtual_gripper_color_rgb=PURE_RED.tolist(),
        virtual_gripper_width_unit="meter",
        virtual_gripper_width_semantics="clear_gap_between_inner_finger_faces",
        virtual_gripper_width_normalized_for_geometry=False,
        virtual_gripper_palm_width_m=0.10,
    )
    if geometry_updated is not None:
        meta["virtual_gripper_geometry_updated"] = bool(geometry_updated)
    write_json(meta_path, meta)


def recolor_directory(directory: Path, *, apply: bool, batch_frames: int) -> dict:
    episodes = sorted(directory.glob("episode_*.zarr"))
    frames = changed_frames = non_red_points = 0
    for ep_index, path in enumerate(episodes):
        group = zarr.open_group(str(path), mode="r+" if apply else "r")
        rgb = group["rgb"]
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape[1] < DEFAULT_GRIPPER_POINTS:
            raise ValueError(f"Invalid RGB cloud shape {rgb.shape}: {path}")
        for start in range(0, rgb.shape[0], batch_frames):
            end = min(start + batch_frames, rgb.shape[0])
            tail = np.asarray(rgb[start:end, -DEFAULT_GRIPPER_POINTS:, :])
            mismatch = np.any(tail != PURE_RED, axis=-1)
            changed_frames += int(np.count_nonzero(np.any(mismatch, axis=1)))
            non_red_points += int(np.count_nonzero(mismatch))
            if apply and np.any(mismatch):
                rgb[start:end, -DEFAULT_GRIPPER_POINTS:, :] = np.broadcast_to(
                    PURE_RED, tail.shape
                )
            frames += end - start
        if (ep_index + 1) % 100 == 0 or ep_index + 1 == len(episodes):
            print(
                f"[color] {directory.name}: {ep_index + 1}/{len(episodes)} episodes, "
                f"{frames} frames, {non_red_points} non-red tail points",
                flush=True,
            )
    if apply:
        update_metadata(directory, geometry_updated=None)
    return {
        "episodes": len(episodes),
        "frames": frames,
        "changed_frames": changed_frames,
        "non_red_points_before": non_red_points,
    }


def canonical_counts(parameters: CanonicalGripperParameters) -> np.ndarray:
    t = parameters.finger_thickness_m
    p = parameters.palm_depth_m
    sizes = [
        np.array([t, t, parameters.finger_length_m]),
        np.array([t, t, parameters.finger_length_m]),
        np.array([t, parameters.palm_width_m, p]),
        np.array([p, p, parameters.handle_length_m]),
    ]
    return allocate_surface_counts(DEFAULT_GRIPPER_POINTS, sizes)


def upgrade_static_geometry(
    dataset: Path,
    *,
    apply: bool,
    batch_frames: int,
    seed: int,
) -> dict:
    """Replace only the static-human tail with canonical Franka geometry."""

    records = episode_records(dataset)
    directories = point_cloud_directories(dataset)
    parameters = CanonicalGripperParameters(max_width_m=0.08)
    counts = canonical_counts(parameters)
    cuts = np.concatenate(([0], np.cumsum(counts)))
    frames = 0
    clipped_frames = 0
    max_gap_error_m = 0.0
    max_palm_error_m = 0.0
    for ep_number, episode_index in enumerate(sorted(records)):
        record = records[episode_index]
        widths = episode_widths_m(dataset, record)
        clipped_frames += int(np.count_nonzero(widths > parameters.max_width_m))
        for directory in directories:
            path = directory / f"episode_{episode_index:06d}.zarr"
            group = zarr.open_group(str(path), mode="r+" if apply else "r")
            xyz = group["xyz"]
            if xyz.shape[0] != len(widths) or xyz.shape[1] < DEFAULT_GRIPPER_POINTS:
                raise ValueError(f"Cloud/state length mismatch: {path}: {xyz.shape} vs {len(widths)}")
            for start in range(0, len(widths), batch_frames):
                end = min(start + batch_frames, len(widths))
                expected = np.stack(
                    [
                        sample_gripper_width(
                            float(widths[frame]),
                            DEFAULT_GRIPPER_POINTS,
                            np.random.default_rng(seed + episode_index * 1_000_003 + frame),
                            parameters=parameters,
                        )
                        for frame in range(start, end)
                    ]
                ).astype(xyz.dtype)
                if apply:
                    xyz[start:end, -DEFAULT_GRIPPER_POINTS:, :] = expected
                    persisted = np.asarray(xyz[start:end, -DEFAULT_GRIPPER_POINTS:, :])
                    if not np.array_equal(persisted, expected):
                        raise RuntimeError(f"Persisted canonical tail mismatch: {path}:{start}:{end}")
                else:
                    persisted = expected
                for local, width in enumerate(widths[start:end]):
                    tail = persisted[local].astype(np.float32)
                    left = tail[cuts[0] : cuts[1]]
                    right = tail[cuts[1] : cuts[2]]
                    palm = tail[cuts[2] : cuts[3]]
                    expected_width = min(float(width), parameters.max_width_m)
                    gap = float(right[:, 1].min() - left[:, 1].max())
                    palm_width = float(np.ptp(palm[:, 1]))
                    max_gap_error_m = max(max_gap_error_m, abs(gap - expected_width))
                    max_palm_error_m = max(max_palm_error_m, abs(palm_width - 0.10))
        frames += len(widths)
        if (ep_number + 1) % 25 == 0 or ep_number + 1 == len(records):
            print(
                f"[geometry] static_human: {ep_number + 1}/{len(records)} episodes, "
                f"{frames} frames",
                flush=True,
            )
    if apply:
        for directory in directories:
            update_metadata(directory, geometry_updated=True)
    return {
        "episodes": len(records),
        "frames": frames,
        "point_cloud_directories": [directory.name for directory in directories],
        "franka_max_width_m": parameters.max_width_m,
        "width_clipped_frames": clipped_frames,
        "surface_counts": counts.tolist(),
        "max_inner_gap_error_m": max_gap_error_m,
        "max_palm_width_error_m": max_palm_error_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true", help="Perform the tail-only migration.")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--color-datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names for exhaustive RGB-tail scanning.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.monotonic()

    dataset_paths = {name: root / relative for name, relative in DATASETS.items()}
    for name, path in dataset_paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name} dataset: {path}")

    geometry = upgrade_static_geometry(
        dataset_paths["static_human"],
        apply=bool(args.apply),
        batch_frames=max(1, int(args.batch_frames)),
        seed=int(args.seed),
    )
    selected_colors = {part.strip() for part in args.color_datasets.split(",") if part.strip()}
    unknown = selected_colors - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown --color-datasets values: {sorted(unknown)}")
    colors = {}
    for name, dataset in dataset_paths.items():
        if name not in selected_colors:
            audit_path = root / "rh20t_gripper_audit_20260907.json"
            if name == "rh20t" and audit_path.is_file():
                colors[name] = {
                    "status": "passed_by_source_receipts_and_stratified_current_zarr_audit",
                    "audit_report": audit_path.name,
                }
            else:
                colors[name] = {"status": "not_scanned_in_this_run"}
            continue
        colors[name] = {}
        for directory in point_cloud_directories(dataset):
            colors[name][directory.name] = recolor_directory(
                directory,
                apply=bool(args.apply),
                batch_frames=max(1, int(args.batch_frames)),
            )

    report = {
        "status": "complete" if args.apply else "dry_run_complete",
        "root": str(root),
        "virtual_gripper_contract": GRIPPER_CONTRACT,
        "tail_points": DEFAULT_GRIPPER_POINTS,
        "color_rgb": PURE_RED.tolist(),
        "geometry": {
            "opening": "physical clear gap in metres; inner faces at +/-width/2",
            "palm_width_m": 0.10,
            "normalization": False,
            "origin": "five eighths of the 0.08 m finger length from finger base",
        },
        "static_geometry_upgrade": geometry,
        "color_audit_and_migration": colors,
        "scene_points_modified": False,
        "parquet_modified": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = root / "virtual_gripper_v3_migration.json"
    if args.apply:
        write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
