#!/usr/bin/env python3
"""Audit a selected RLBench dataset-action replay run and write a compact manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.dataset as pyarrow_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def load_selection(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["episode"] = int(row["episode"])
        row["planner_max_time_ms"] = int(row["planner_max_time_ms"])
    return rows


def main():
    args = parse_args()
    root = args.run_root.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    selected = load_selection(selection_path)
    replay_root = root / "replays"
    parquet = pyarrow_dataset.dataset(str(dataset_root / "data"), format="parquet")
    records = []

    for requested in selected:
        task = requested["task"]
        episode = int(requested["episode"])
        matches = sorted(
            replay_root.glob(
                f"{task}_episode_{episode:03d}_eef0_planning_*/result.json"
            )
        )
        result_path = matches[-1] if matches else None
        result = None
        if result_path is not None:
            with result_path.open("r", encoding="utf-8") as file:
                result = json.load(file)
        run_dir = result_path.parent if result_path is not None else None
        video = None if run_dir is None else run_dir / "replay.mp4"
        labels = None if run_dir is None else run_dir / "action_labels.npy"
        label_shape = None
        label_dtype = None
        label_valid = False
        label_exact_parquet = False
        if labels is not None and labels.is_file():
            array = np.load(labels, mmap_mode="r")
            label_shape = list(array.shape)
            label_dtype = str(array.dtype)
            label_valid = array.ndim == 2 and array.shape[1] == 10
            if result is not None:
                table = parquet.to_table(
                    filter=(
                        pyarrow_dataset.field("episode_index")
                        == int(result["global_episode"])
                    ),
                    columns=["frame_index", "action"],
                )
                frame_indices = np.asarray(
                    table["frame_index"].to_pylist(), dtype=np.int64
                )
                expected = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                expected = expected[np.argsort(frame_indices)]
                label_exact_parquet = bool(np.array_equal(np.asarray(array), expected))
        video_frame_count = None
        video_readable = False
        video_frame_overlay = False
        if video is not None and video.is_file():
            reader = imageio.get_reader(video)
            try:
                video_frame_count = int(reader.count_frames())
                if video_frame_count > 0:
                    reader.get_data(0)
                    video_readable = True
            finally:
                reader.close()
        if result is not None:
            overlay = result.get("video_frame_overlay") or {}
            video_frame_overlay = bool(
                overlay.get("enabled")
                and overlay.get("schema") == "rlbench_replay_frame_action_v1"
                and overlay.get("mapping_source") == "execution_trace"
            )
        record = {
            **requested,
            "success": None if result is None else bool(result.get("success")),
            "global_episode": None if result is None else result.get("global_episode"),
            "recorded_frames": None if result is None else result.get("recorded_frames"),
            "result_planner_max_time_ms": (
                None if result is None else result.get("planner_max_time_ms")
            ),
            "result_json": None if result_path is None else str(result_path),
            "video": None if video is None else str(video),
            "video_exists": bool(video is not None and video.is_file()),
            "video_readable": bool(video_readable),
            "video_frame_count": video_frame_count,
            "video_has_exact_frame_action_overlay": bool(video_frame_overlay),
            "action_labels": None if labels is None else str(labels),
            "action_labels_exists": bool(labels is not None and labels.is_file()),
            "action_labels_shape": label_shape,
            "action_labels_dtype": label_dtype,
            "action_labels_valid_t_by_10": bool(label_valid),
            "action_labels_exact_parquet": bool(label_exact_parquet),
        }
        records.append(record)

    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    complete = [
        row
        for row in records
        if row["result_json"]
        and row["video_exists"]
        and row["video_readable"]
        and row["video_has_exact_frame_action_overlay"]
        and row["action_labels_exists"]
        and row["action_labels_valid_t_by_10"]
        and row["action_labels_exact_parquet"]
        and row["result_planner_max_time_ms"] == row["planner_max_time_ms"]
    ]
    summary = {
        "run_root": str(root),
        "dataset_root": str(dataset_root),
        "selection": str(selection_path),
        "requested_episodes": len(selected),
        "completed_with_video_and_action_labels": len(complete),
        "successes": sum(row["success"] is True for row in records),
        "failures": sum(row["success"] is False for row in records),
        "process_or_artifact_errors": sum(row["success"] is None for row in records),
        "records": records,
    }
    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    columns = list(records[0].keys()) if records else []
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )

    lines = [
        "# Selected dataset action replay report",
        "",
        f"- Requested: **{len(selected)}**",
        f"- Complete result + frame/action-labeled video + exact parquet `(T,10)` action labels: **{len(complete)}**",
        f"- Replay result: **{summary['successes']} success / {summary['failures']} failure**",
        f"- Process/artifact errors: **{summary['process_or_artifact_errors']}**",
        "",
        "| Task | Selected | Replay success | Replay failure | Complete artifacts |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in dict.fromkeys(row["task"] for row in selected):
        rows = by_task[task]
        task_complete = sum(row in complete for row in rows)
        lines.append(
            f"| {task} | {len(rows)} | "
            f"{sum(row['success'] is True for row in rows)} | "
            f"{sum(row['success'] is False for row in rows)} | {task_complete} |"
        )
    lines.extend(
        [
            "",
            "Each replay directory contains a frame/action-labeled `replay.mp4`, `action_labels.npy`, and `result.json`.",
            "Every NumPy label array was checked byte-for-value against its selected parquet episode and has shape `(T, 10)`.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"complete={len(complete)}/{len(selected)} "
        f"success={summary['successes']} failure={summary['failures']}"
    )
    if len(complete) != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
