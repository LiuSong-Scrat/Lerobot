#!/usr/bin/env python3
"""Export RLBench task, success, frame, and action[9] diagnostic fields."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as pads
import pyarrow.parquet as pq


TASK_ALIASES = {
    "put the phone on the base": "phone_on_base",
    "phone on base": "phone_on_base",
    "water plant": "water_plants",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_task(task: str) -> str:
    normalized = " ".join(task.strip().lower().replace("_", " ").split())
    return TASK_ALIASES.get(normalized, normalized.replace(" ", "_"))


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info_path = dataset_root / "meta/info.json"
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    data_dir = dataset_root / "data"

    if not info_path.is_file() or not episodes_path.is_file() or not data_dir.is_dir():
        raise FileNotFoundError(f"Not a complete LeRobot dataset: {dataset_root}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature_names = info.get("features", {}).get("action", {}).get("names", [])
    if len(feature_names) <= 9 or feature_names[9] != "gripper":
        raise ValueError(f"Expected action[9] to be gripper, got names={feature_names!r}")

    stored_features = set(info.get("features", {}))
    success_columns = sorted(
        name
        for name in stored_features
        if any(token in name.lower() for token in ("success", "done", "reward"))
    )
    if success_columns:
        raise ValueError(f"Unexpected stored success-like fields need explicit handling: {success_columns}")

    episode_table = pq.read_table(
        episodes_path,
        columns=["episode_index", "tasks", "length"],
    )
    episode_meta = {}
    task_episode_counts: Counter[str] = Counter()
    for episode_index, tasks, length in zip(
        episode_table["episode_index"].to_pylist(),
        episode_table["tasks"].to_pylist(),
        episode_table["length"].to_pylist(),
        strict=True,
    ):
        if not tasks:
            raise ValueError(f"Episode {episode_index} has no task description")
        task = canonical_task(tasks[0])
        task_local_episode_index = task_episode_counts[task]
        task_episode_counts[task] += 1
        episode_meta[int(episode_index)] = {
            "task": task,
            "task_local_episode_index": task_local_episode_index,
            "length": int(length),
        }

    table = pads.dataset(data_dir, format="parquet").to_table(
        columns=["action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    )
    records = []
    for action, timestamp, frame_index, episode_index, dataset_index, task_index in zip(
        table["action"].to_pylist(),
        table["timestamp"].to_pylist(),
        table["frame_index"].to_pylist(),
        table["episode_index"].to_pylist(),
        table["index"].to_pylist(),
        table["task_index"].to_pylist(),
        strict=True,
    ):
        meta = episode_meta[int(episode_index)]
        if len(action) <= 9:
            raise ValueError(f"Episode {episode_index} frame {frame_index} has short action")
        action_9 = float(action[9])
        records.append(
            {
                "task": meta["task"],
                "task_index": int(task_index),
                "task_episode_index": meta["task_local_episode_index"],
                "episode_index": int(episode_index),
                # This is inferred from successful RLBench expert-demo collection;
                # it is not a stored frame-level success signal.
                "expert_demo_success_inferred": 1,
                "frame_index": int(frame_index),
                "is_last_frame": int(int(frame_index) == meta["length"] - 1),
                "dataset_index": int(dataset_index),
                "timestamp_s": float(timestamp),
                "action[9]_gripper_target_width_m": action_9,
                "gripper_target_open_gt_0.04m": int(action_9 > 0.04),
            }
        )

    records.sort(key=lambda item: (item["episode_index"], item["frame_index"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    per_task_summary = {}
    for task in sorted(task_episode_counts):
        task_records = [record for record in records if record["task"] == task]
        output_path = output_dir / f"{task}_success_frame_action9.csv"
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(task_records)
        per_task_summary[task] = {
            "episodes": task_episode_counts[task],
            "frames": len(task_records),
            "inferred_success_episodes": task_episode_counts[task],
            "action_9_closed_frames": sum(
                record["gripper_target_open_gt_0.04m"] == 0 for record in task_records
            ),
            "action_9_open_frames": sum(
                record["gripper_target_open_gt_0.04m"] == 1 for record in task_records
            ),
            "output": str(output_path),
        }

    manifest = {
        "dataset_root": str(dataset_root),
        "stored_success_like_fields": success_columns,
        "success_semantics": (
            "expert_demo_success_inferred=1 because only successfully returned and verified "
            "RLBench live expert demonstrations were packed; this is not a stored success column"
        ),
        "action_9_semantics": (
            "next expert-command Panda gripper target width in meters; 0=closed, 0.08=open; "
            "the current observed width is observation.state[9]"
        ),
        "tasks": per_task_summary,
    }
    manifest_path = output_dir / "export_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
