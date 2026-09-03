#!/usr/bin/env python3
"""Backfill strict robot-base WorldFlow sidecars into an RLBench LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from rlbench_worldflow_sidecars import (  # noqa: E402
    WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
    WORLD_BASE_EE_POSE_DIR,
    build_robot_base_episode_sidecars,
    validate_rigid_transform,
    write_sidecar_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    transform = parser.add_mutually_exclusive_group(required=True)
    transform.add_argument(
        "--t-world-base",
        type=float,
        nargs=16,
        metavar=(
            "R00", "R01", "R02", "TX",
            "R10", "R11", "R12", "TY",
            "R20", "R21", "R22", "TZ",
            "B0", "B1", "B2", "B3",
        ),
        help="Row-major ^world T_base matrix for the selected fixed robot-base frame.",
    )
    transform.add_argument(
        "--t-world-base-json",
        type=Path,
        help="JSON file containing a 4x4 matrix or a T_world_base field.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--transform-source",
        default="explicit --t-world-base matrix supplied to backfill tool",
    )
    parser.add_argument(
        "--base-frame-definition",
        default="explicit_fixed_robot_base",
        help="Stable semantic/version label written into sidecar metadata.",
    )
    return parser.parse_args()


def load_transform(args: argparse.Namespace) -> np.ndarray:
    if args.t_world_base is not None:
        matrix = np.asarray(args.t_world_base, dtype=np.float64).reshape(4, 4)
    else:
        payload = json.loads(args.t_world_base_json.expanduser().resolve().read_text())
        if isinstance(payload, dict):
            payload = payload["T_world_base"]
        matrix = np.asarray(payload, dtype=np.float64)
    return validate_rigid_transform(matrix, "T_world_base")


def read_episode_table(dataset_root: Path) -> list[dict]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet under {dataset_root}")
    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=[
                    "episode_index",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                    "dataset_from_index",
                    "dataset_to_index",
                ],
            )
            for path in paths
        ]
    )
    records = table.to_pylist()
    records.sort(key=lambda row: int(row["episode_index"]))
    expected = list(range(len(records)))
    actual = [int(row["episode_index"]) for row in records]
    if actual != expected:
        raise ValueError(
            "Episode indices must be dense and zero-based for episode sidecars: "
            f"first={actual[:3]}, last={actual[-3:]}"
        )
    return records


def read_episode_actions(dataset_root: Path, record: dict) -> np.ndarray:
    episode_index = int(record["episode_index"])
    chunk_index = int(record["data/chunk_index"])
    file_index = int(record["data/file_index"])
    path = dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
    episode_values = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    mask = episode_values == episode_index
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[mask]
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[mask]
    order = np.argsort(frame_indices, kind="stable")
    frame_indices = frame_indices[order]
    actions = actions[order]
    expected_length = int(record["length"])
    if len(actions) != expected_length:
        raise ValueError(
            f"episode {episode_index}: parquet action rows={len(actions)}, expected={expected_length}"
        )
    if not np.array_equal(frame_indices, np.arange(expected_length, dtype=np.int64)):
        raise ValueError(f"episode {episode_index}: frame_index is not contiguous from zero")
    return actions


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {dataset_root}")
    t_world_base = load_transform(args)
    conversion_path = dataset_root / "meta" / "rlbench_conversion.json"
    conversion = json.loads(conversion_path.read_text()) if conversion_path.is_file() else {}
    action_alignment = str(conversion.get("action_alignment", "transition"))
    action_label_mode = str(conversion.get("action_label_mode", ""))
    if action_label_mode != "expert_target":
        raise ValueError(
            "Commanded-target backfill requires meta/rlbench_conversion.json "
            "action_label_mode=expert_target; got "
            + repr(action_label_mode)
            + ". An executed/achieved action column must not be labeled as a commanded target."
        )
    records = read_episode_table(dataset_root)
    world_pose_paths = sorted((dataset_root / "world_ee_poses").glob("episode_*.npy"))
    if len(world_pose_paths) != len(records):
        raise ValueError(
            f"world_ee_poses files={len(world_pose_paths)}, dataset episodes={len(records)}"
        )

    destinations = [
        dataset_root / WORLD_BASE_EE_POSE_DIR,
        dataset_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Robot-base sidecar directories already exist; validate them or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    temp_root = dataset_root / f".robot_base_worldflow_sidecars_tmp_{os.getpid()}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir()
    aggregate = {
        "episode_count": len(records),
        "frame_count": 0,
        "achieved_roundtrip_max_abs": 0.0,
        "action_target_roundtrip_max_abs": 0.0,
        "achieved_rotation_orthogonality_max_abs": 0.0,
        "target_rotation_orthogonality_max_abs": 0.0,
        "achieved_rotation_determinant_min": float("inf"),
        "achieved_rotation_determinant_max": float("-inf"),
        "target_rotation_determinant_min": float("inf"),
        "target_rotation_determinant_max": float("-inf"),
    }
    try:
        write_sidecar_metadata(
            temp_root,
            t_world_base,
            args.transform_source,
            action_alignment=action_alignment,
            base_frame_definition=args.base_frame_definition,
        )
        for offset, record in enumerate(records):
            episode_index = int(record["episode_index"])
            expected_length = int(record["length"])
            source_path = dataset_root / "world_ee_poses" / f"episode_{episode_index:06d}.npy"
            world_ee_poses = np.load(source_path)
            actions = read_episode_actions(dataset_root, record)
            if world_ee_poses.shape != (expected_length, 9):
                raise ValueError(
                    f"episode {episode_index}: world_ee_poses shape={world_ee_poses.shape}, "
                    f"expected=({expected_length}, 9)"
                )
            achieved, target, metrics = build_robot_base_episode_sidecars(
                world_ee_poses,
                actions,
                t_world_base,
            )
            filename = f"episode_{episode_index:06d}.npy"
            achieved_path = temp_root / WORLD_BASE_EE_POSE_DIR / filename
            target_path = temp_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR / filename
            np.save(achieved_path, np.ascontiguousarray(achieved, dtype=np.float32))
            np.save(target_path, np.ascontiguousarray(target, dtype=np.float32))
            for saved_path in (achieved_path, target_path):
                saved = np.load(saved_path, mmap_mode="r")
                if saved.shape != (expected_length, 9) or saved.dtype != np.float32:
                    raise ValueError(f"Saved sidecar validation failed: {saved_path} {saved.shape} {saved.dtype}")
                if not np.isfinite(saved).all():
                    raise ValueError(f"Saved sidecar contains NaN/Inf: {saved_path}")
            aggregate["frame_count"] += expected_length
            for key in (
                "achieved_roundtrip_max_abs",
                "action_target_roundtrip_max_abs",
                "achieved_rotation_orthogonality_max_abs",
                "target_rotation_orthogonality_max_abs",
                "achieved_rotation_determinant_max",
                "target_rotation_determinant_max",
            ):
                aggregate[key] = max(float(aggregate[key]), float(metrics[key]))
            for key in (
                "achieved_rotation_determinant_min",
                "target_rotation_determinant_min",
            ):
                aggregate[key] = min(float(aggregate[key]), float(metrics[key]))
            if offset % 50 == 0 or offset + 1 == len(records):
                print(f"[sidecar] {offset + 1}/{len(records)} episode={episode_index}", flush=True)

        if aggregate["frame_count"] != sum(int(record["length"]) for record in records):
            raise ValueError("Generated frame total does not match episode metadata")
        for destination in existing:
            shutil.rmtree(destination)
        for directory_name in (
            WORLD_BASE_EE_POSE_DIR,
            WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
        ):
            (temp_root / directory_name).rename(dataset_root / directory_name)
        temp_root.rmdir()
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise

    report = {
        "complete": True,
        "dataset_root": str(dataset_root),
        "action_alignment": action_alignment,
        "target_semantics": "commanded action target; never achieved next pose",
        "transform_source": str(args.transform_source),
        "base_frame_definition": str(args.base_frame_definition),
        "T_world_base": t_world_base.tolist(),
        "T_base_world": np.linalg.inv(t_world_base).tolist(),
        "validation": aggregate,
    }
    report_path = dataset_root / "meta" / "rlbench_worldflow_robot_base_sidecars.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    conversion.update(
        {
            "world_base_worldflow_sidecars": True,
            "world_base_frame_definition": str(args.base_frame_definition),
            "world_base_ee_pose_semantics": (
                "T_base_ee = inverse(T_world_base) @ T_world_ee"
            ),
            "world_base_action_target_semantics": (
                "commanded expert target: inverse(T_world_base) @ "
                "T_world_eef0 @ T_eef0_target"
            ),
            "T_world_base": t_world_base.tolist(),
            "T_base_world": np.linalg.inv(t_world_base).tolist(),
        }
    )
    conversion_tmp = conversion_path.with_suffix(conversion_path.suffix + ".tmp")
    conversion_tmp.write_text(json.dumps(conversion, indent=2) + "\n", encoding="utf-8")
    os.replace(conversion_tmp, conversion_path)

    complete_path = dataset_root / "meta" / "rlbench_conversion_complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text())
        complete["world_base_worldflow_sidecars"] = True
        complete["world_base_frame_definition"] = str(args.base_frame_definition)
        complete_tmp = complete_path.with_suffix(complete_path.suffix + ".tmp")
        complete_tmp.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
        os.replace(complete_tmp, complete_path)
    print(f"[done] {report_path}")


if __name__ == "__main__":
    main()
