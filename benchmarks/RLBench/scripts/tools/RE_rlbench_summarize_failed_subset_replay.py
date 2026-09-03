#!/usr/bin/env python3
"""Summarize a rerun of failed episodes selected from an earlier replay summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    source_summary_path = args.source_summary.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    source = read_json(source_summary_path)
    source_task_records = [
        row for row in source["records"] if str(row["task"]) == args.task
    ]
    candidates = [row for row in source_task_records if not bool(row["success"])]

    rows = []
    for source_row in sorted(candidates, key=lambda row: int(row["episode"])):
        episode = int(source_row["episode"])
        result_paths = sorted(
            (run_root / "episodes" / f"episode_{episode:03d}").glob(
                f"{args.task}_episode_{episode:03d}_eef0_planning_*/result.json"
            )
        )
        result_path = result_paths[-1] if result_paths else None
        result = read_json(result_path) if result_path is not None else None
        rows.append(
            {
                "task": args.task,
                "episode": episode,
                "global_episode": int(source_row["global_episode"]),
                "original_success": bool(source_row["success"]),
                "original_classification": source_row.get("classification"),
                "original_raw_joint_control_success": source_row.get(
                    "raw_joint_control_success"
                ),
                "rerun_completed": result is not None,
                "rerun_success": None if result is None else bool(result.get("success")),
                "rerun_error_count": None
                if result is None
                else len(result.get("errors", [])),
                "gripper_mode": None if result is None else result.get("gripper_mode"),
                "gripper_delta_threshold_m": None
                if result is None
                else result.get("gripper_delta_threshold_m"),
                "controller_profile": None
                if result is None
                else result.get("controller_profile"),
                "video": None if result is None else result.get("video"),
                "result_json": None if result_path is None else str(result_path),
            }
        )

    completed = sum(bool(row["rerun_completed"]) for row in rows)
    successes = sum(row["rerun_success"] is True for row in rows)
    original_task_successes = sum(bool(row["success"]) for row in source_task_records)
    breakdown_groups = defaultdict(list)
    for row in rows:
        breakdown_groups[str(row["original_classification"])].append(row)
    classification_breakdown = {
        classification: {
            "selected": len(group),
            "completed": sum(bool(row["rerun_completed"]) for row in group),
            "rerun_success": sum(row["rerun_success"] is True for row in group),
            "rerun_failure": sum(row["rerun_success"] is False for row in group),
        }
        for classification, group in sorted(breakdown_groups.items())
    }
    summary = {
        "source_summary": str(source_summary_path),
        "run_root": str(run_root),
        "task": args.task,
        "selection": "episodes with success=false in source summary",
        "selected_episode_count": len(rows),
        "selected_episodes": [row["episode"] for row in rows],
        "completed_episode_count": completed,
        "rerun_success_count": successes,
        "rerun_failure_count": completed - successes,
        "rerun_success_rate_on_original_failures": (
            None if completed == 0 else successes / completed
        ),
        "source_task_success_count": original_task_successes,
        "source_task_episode_count": len(source_task_records),
        "hybrid_projected_success_count": original_task_successes + successes,
        "hybrid_projection_warning": (
            "This is not a full delta=0.003 success rate: source-success episodes were "
            "not rerun and could change under the new threshold."
        ),
        "classification_breakdown": classification_breakdown,
        "records": rows,
    }
    with (run_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    columns = list(rows[0]) if rows else []
    with (run_root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    succeeded_episodes = [row["episode"] for row in rows if row["rerun_success"] is True]
    failed_episodes = [row["episode"] for row in rows if row["rerun_success"] is False]
    missing_episodes = [row["episode"] for row in rows if not row["rerun_completed"]]
    breakdown_lines = []
    for classification, counts in classification_breakdown.items():
        breakdown_lines.append(
            f"- `{classification}`: **{counts['rerun_success']}/{counts['completed']}** "
            "rerun success"
        )
    report = f"""# Failed water replay rerun: LIBERO delta 0.003 m

- Source: `{source_summary_path}`
- Selection: the **{len(rows)}** original `{args.task}` failures only.
- Completed: **{completed}/{len(rows)}**
- Rescued by this rerun: **{successes}/{completed if completed else len(rows)}**
- Still failed: **{completed - successes}/{completed if completed else len(rows)}**
- Original task result: **{original_task_successes}/{len(source_task_records)}**
- Hybrid projection: **{original_task_successes + successes}/{len(source_task_records)}**
  (not a full 0.003 evaluation because the original successful episodes were not rerun).

## Episodes

- Rerun success: `{succeeded_episodes}`
- Rerun failure: `{failed_episodes}`
- Missing/incomplete: `{missing_episodes}`

## Breakdown by original failure class

{chr(10).join(breakdown_lines)}

See `summary.csv` for per-episode result paths, original failure classifications,
raw-joint control outcomes, error counts, and replay video paths.
"""
    (run_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"completed={completed}/{len(rows)} success={successes} "
        f"failure={completed - successes} missing={len(rows) - completed}"
    )


if __name__ == "__main__":
    main()
