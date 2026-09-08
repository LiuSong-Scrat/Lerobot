#!/usr/bin/env python3
"""Compare two complete RLBench action-replay summaries episode by episode."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def record_map(summary):
    records = summary["records"]
    mapped = {(str(row["task"]), int(row["episode"])): row for row in records}
    if len(mapped) != len(records):
        raise RuntimeError("Duplicate task/episode record in replay summary")
    return mapped


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    baseline_records = record_map(baseline)
    candidate_records = record_map(candidate)
    if baseline_records.keys() != candidate_records.keys():
        missing = sorted(baseline_records.keys() - candidate_records.keys())
        extra = sorted(candidate_records.keys() - baseline_records.keys())
        raise RuntimeError(f"Episode sets differ: missing={missing}, extra={extra}")

    rows = []
    for task, episode in sorted(baseline_records):
        old = baseline_records[(task, episode)]
        new = candidate_records[(task, episode)]
        old_success = bool(old["success"])
        new_success = bool(new["success"])
        if old_success and new_success:
            transition = "success_to_success"
        elif not old_success and new_success:
            transition = "failure_to_success"
        elif old_success and not new_success:
            transition = "success_to_failure"
        else:
            transition = "failure_to_failure"
        rows.append(
            {
                "task": task,
                "episode": episode,
                "baseline_success": old_success,
                "candidate_success": new_success,
                "transition": transition,
                "baseline_classification": old.get("classification"),
                "candidate_classification": new.get("classification"),
                "baseline_planner_max_time_ms": old.get("planner_max_time_ms"),
                "candidate_planner_max_time_ms": new.get("planner_max_time_ms"),
                "candidate_video": new.get("video"),
                "candidate_result_json": new.get("result_json"),
            }
        )

    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    task_rows = []
    for task, group in sorted(by_task.items()):
        old_success = sum(row["baseline_success"] for row in group)
        new_success = sum(row["candidate_success"] for row in group)
        task_rows.append(
            {
                "task": task,
                "episodes": len(group),
                "baseline_success": old_success,
                "candidate_success": new_success,
                "success_delta": new_success - old_success,
                "rescued": sum(
                    row["transition"] == "failure_to_success" for row in group
                ),
                "regressed": sum(
                    row["transition"] == "success_to_failure" for row in group
                ),
                "still_failed": sum(
                    row["transition"] == "failure_to_failure" for row in group
                ),
            }
        )

    total_old = sum(row["baseline_success"] for row in rows)
    total_new = sum(row["candidate_success"] for row in rows)
    comparison = {
        "baseline_summary": str(baseline_path),
        "candidate_summary": str(candidate_path),
        "episode_count": len(rows),
        "baseline_success": total_old,
        "candidate_success": total_new,
        "success_delta": total_new - total_old,
        "rescued": sum(row["transition"] == "failure_to_success" for row in rows),
        "regressed": sum(row["transition"] == "success_to_failure" for row in rows),
        "still_failed": sum(row["transition"] == "failure_to_failure" for row in rows),
        "task_comparison": task_rows,
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
    with (output_dir / "episode_transitions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "task_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    lines = [
        "# Action replay threshold comparison",
        "",
        f"- Episodes: **{len(rows)}**",
        f"- Baseline success: **{total_old}/{len(rows)}**",
        f"- Candidate success: **{total_new}/{len(rows)}**",
        f"- Net change: **{total_new - total_old:+d}**",
        f"- Failure → success: **{comparison['rescued']}**",
        f"- Success → failure: **{comparison['regressed']}**",
        f"- Failure → failure: **{comparison['still_failed']}**",
        "",
        "| Task | Baseline | Candidate | Delta | Rescued | Regressed | Still failed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task']} | {row['baseline_success']}/{row['episodes']} | "
            f"{row['candidate_success']}/{row['episodes']} | "
            f"{row['success_delta']:+d} | {row['rescued']} | "
            f"{row['regressed']} | {row['still_failed']} |"
        )
    (output_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"episodes={len(rows)} baseline={total_old} candidate={total_new} "
        f"delta={total_new - total_old:+d}"
    )


if __name__ == "__main__":
    main()
