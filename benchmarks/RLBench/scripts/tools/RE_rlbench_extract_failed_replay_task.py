#!/usr/bin/env python3
"""Extract one task from an existing RLBench failed-replay review bundle.

The source bundle is produced by RE_rlbench_export_failed_replays_with_actions.py.
This tool creates a standalone, human-readable review package without modifying
the source validation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


FILE_MAP = {
    "action_labels_pose9_gripper.npy": "action_labels.npy",
    "action_labels_pose9_gripper.csv": "action_labels.csv",
    "dataset_trajectory_full_with_frame_and_action_index.mp4": (
        "dataset_expert_trajectory__frames_actions.mp4"
    ),
    "replay_with_frame_and_action_index.mp4": "failed_replay__frames_actions.mp4",
    "replay_original.mp4": "failed_replay__original.mp4",
    "raw_joint_actions.npy": "raw_joint_actions.npy",
    "raw_joint_actions.csv": "raw_joint_actions.csv",
    "replay_result.json": "replay_result.json",
    "raw_joint_replay_result.json": "raw_joint_replay_result.json",
    "raw_joint_replay_original.mp4": "raw_joint_replay__original.mp4",
    "raw_joint_replay_with_frame_and_action_index.mp4": (
        "raw_joint_replay__frames_actions.mp4"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    validation_root = args.validation_root.expanduser().resolve()
    source_bundle = validation_root / "failed_trajectories_with_actions"
    output_dir = args.output_dir.expanduser().resolve()

    if not source_bundle.is_dir():
        raise FileNotFoundError(source_bundle)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing review package: {output_dir}"
        )

    prefix = f"{args.task}_episode_"
    source_episodes = sorted(
        path
        for path in source_bundle.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if not source_episodes:
        raise RuntimeError(f"No failed replay directories match {prefix!r}")

    output_dir.mkdir(parents=True)
    manifest_rows = []

    for source_episode in source_episodes:
        source_meta_path = source_episode / "metadata.json"
        if not source_meta_path.is_file():
            raise FileNotFoundError(source_meta_path)
        metadata = read_json(source_meta_path)
        local_episode = int(metadata["local_episode"])
        destination = output_dir / f"episode_{local_episode:03d}"
        destination.mkdir()

        copied_files = []
        for source_name, destination_name in FILE_MAP.items():
            source_path = source_episode / source_name
            if not source_path.is_file():
                continue
            destination_path = destination / destination_name
            shutil.copy2(source_path, destination_path)
            if sha256(source_path) != sha256(destination_path):
                raise RuntimeError(f"Copy verification failed: {destination_path}")
            copied_files.append(destination_name)

        action_path = destination / "action_labels.npy"
        if not action_path.is_file():
            raise FileNotFoundError(action_path)
        actions = np.load(action_path, allow_pickle=False)
        if actions.ndim != 2 or actions.shape[1] != 10:
            raise RuntimeError(
                f"Expected (T, 10) action labels in {action_path}, got {actions.shape}"
            )

        review_metadata = dict(metadata)
        review_metadata.update(
            {
                "review_package_source": str(source_episode),
                "review_action_file": "action_labels.npy",
                "review_action_file_sha256": sha256(action_path),
                "review_files": copied_files,
            }
        )
        with (destination / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(review_metadata, file, ensure_ascii=False, indent=2)

        manifest_rows.append(
            {
                "task": args.task,
                "local_episode": local_episode,
                "global_episode": int(metadata["global_episode"]),
                "classification": metadata["classification"],
                "action_rows": int(actions.shape[0]),
                "action_columns": int(actions.shape[1]),
                "failed_replay_frames": int(metadata["main_video"]["video_frame_count"]),
                "dataset_video_frames": int(
                    metadata["full_dataset_video"]["video_frame_count"]
                ),
                "raw_joint_control_success": metadata.get(
                    "raw_joint_control_success"
                ),
                "episode_dir": destination.name,
                "action_labels_sha256": sha256(action_path),
            }
        )

    manifest_columns = list(manifest_rows[0])
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_columns)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_validation_root": str(validation_root),
        "source_failed_bundle": str(source_bundle),
        "task": args.task,
        "failure_definition": "primary EEF-label replay reported success=false",
        "failed_episode_count": len(manifest_rows),
        "local_failed_episodes": [row["local_episode"] for row in manifest_rows],
        "action_label_semantics": (
            "EEF0-relative target pose9 (columns 0:9) plus gripper width in metres "
            "(column 9)"
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    readme = f"""# {args.task} failed action-replay review package

- Source validation: `{validation_root}`
- Failure definition: primary EEF-label replay returned `success=false`.
- Failed episodes: **{len(manifest_rows)}**
- Episode list: `{summary['local_failed_episodes']}`

## Files in each `episode_NNN/`

- `action_labels.npy`: exact dataset action labels, shape `(T, 10)`, `float32`.
  Columns `0:9` are the EEF0-relative target pose9; column `9` is gripper width in metres.
- `action_labels.csv`: the same labels with column names for inspection.
- `dataset_expert_trajectory__frames_actions.mp4`: complete stored expert trajectory;
  the overlay shows `dataset_frame`, `action_index`, XYZ, and gripper label.
- `failed_replay__frames_actions.mp4`: failed EEF replay with `video_frame` and the
  action index that produced the displayed observation.
- `failed_replay__original.mp4`: unchanged original replay video.
- `raw_joint_actions.npy/.csv`: original RLBench expert joint commands for comparison.
- `metadata.json` and replay result JSON files: alignment and failure details.

## Exact frame/action alignment

- Failed replay video frame `0` is the restored initial observation; no action has run.
- Failed replay video frame `n >= 1` is the observation after executing
  `action_labels[n - 1]`.
- In the full dataset video, row/frame `i` displays the stored observation and the
  command label `action_labels[i]` originating from that row.

Use `manifest.csv` for the episode list, lengths, classifications, and SHA-256 hashes.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"task={args.task}")
    print(f"failed_episodes={len(manifest_rows)}")
    print(f"episodes={summary['local_failed_episodes']}")


if __name__ == "__main__":
    main()
