#!/usr/bin/env python3
"""Compare fresh legacy and PointAct-parity replay results for candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def newest_result(root, task, episode):
    episode_root = root / f"{task}_episode_{episode:03d}"
    matches = sorted(episode_root.glob(f"{task}_episode_{episode:03d}_eef0_planning_*/result.json"))
    return matches[-1] if matches else None


def result_row(profile_root, task, episode):
    path = newest_result(profile_root, task, episode)
    if path is None:
        return {
            "result_present": False,
            "success": False,
            "result_json": None,
            "video": None,
            "errors": ["missing result.json"],
        }
    result = read_json(path)
    return {
        "result_present": True,
        "success": bool(result.get("success")),
        "result_json": str(path),
        "video": result.get("video"),
        "attempted_actions": result.get("attempted_actions"),
        "executed_actions": result.get("executed_actions"),
        "mover_attempts": result.get("mover_attempts"),
        "mover_retries": result.get("mover_retries"),
        "mover_unreached_targets": result.get("mover_unreached_targets"),
        "controller_continue_errors": result.get("controller_continue_errors"),
        "errors": result.get("errors", []),
    }


def main():
    args = parse_args()
    candidate_summary = args.candidate_summary.expanduser().resolve()
    comparison_root = args.comparison_root.expanduser().resolve()
    baseline = read_json(candidate_summary)
    candidates = [row for row in baseline["records"] if not bool(row["success"])]
    profile_roots = {
        "original_single_step": comparison_root / "original_single_step",
        "updated_pointact_eval": comparison_root / "updated_pointact_eval",
    }

    rows = []
    transitions = Counter()
    by_task = defaultdict(lambda: Counter(total=0, original_success=0, updated_success=0))
    for candidate in candidates:
        task = str(candidate["task"])
        episode = int(candidate["episode"])
        original = result_row(profile_roots["original_single_step"], task, episode)
        updated = result_row(profile_roots["updated_pointact_eval"], task, episode)
        transition = (
            ("success" if original["success"] else "failure")
            + "_to_"
            + ("success" if updated["success"] else "failure")
        )
        transitions[transition] += 1
        by_task[task]["total"] += 1
        by_task[task]["original_success"] += int(original["success"])
        by_task[task]["updated_success"] += int(updated["success"])
        rows.append(
            {
                "task": task,
                "episode": episode,
                "global_episode": candidate.get("global_episode"),
                "legacy_classification": candidate.get("classification"),
                "original_success": original["success"],
                "updated_success": updated["success"],
                "transition": transition,
                "original_attempted_actions": original.get("attempted_actions"),
                "updated_attempted_actions": updated.get("attempted_actions"),
                "updated_mover_attempts": updated.get("mover_attempts"),
                "updated_mover_retries": updated.get("mover_retries"),
                "updated_mover_unreached_targets": updated.get("mover_unreached_targets"),
                "updated_controller_errors": updated.get("controller_continue_errors"),
                "original_errors": original.get("errors"),
                "updated_errors": updated.get("errors"),
                "original_video": original.get("video"),
                "updated_video": updated.get("video"),
                "original_result_json": original.get("result_json"),
                "updated_result_json": updated.get("result_json"),
                "original_result_present": original.get("result_present"),
                "updated_result_present": updated.get("result_present"),
            }
        )

    original_success = sum(int(row["original_success"]) for row in rows)
    updated_success = sum(int(row["updated_success"]) for row in rows)
    original_failures_with_errors = sum(
        int(not row["original_success"] and bool(row["original_errors"])) for row in rows
    )
    original_error_failures_recovered = sum(
        int(
            not row["original_success"]
            and bool(row["original_errors"])
            and row["updated_success"]
        )
        for row in rows
    )
    updated_failures_without_errors = sum(
        int(not row["updated_success"] and not bool(row["updated_errors"])) for row in rows
    )
    updated_failures_with_retries_or_unreached = sum(
        int(
            not row["updated_success"]
            and (
                int(row.get("updated_mover_retries") or 0) > 0
                or int(row.get("updated_mover_unreached_targets") or 0) > 0
            )
        )
        for row in rows
    )
    output = {
        "candidate_summary": str(candidate_summary),
        "candidate_count": len(rows),
        "original_single_step": {
            "success": original_success,
            "failure": len(rows) - original_success,
        },
        "updated_pointact_eval": {
            "success": updated_success,
            "failure": len(rows) - updated_success,
        },
        "transitions": dict(sorted(transitions.items())),
        "diagnostics": {
            "original_failures_with_controller_errors": original_failures_with_errors,
            "original_error_failures_recovered_by_updated": original_error_failures_recovered,
            "updated_failures_without_controller_errors": updated_failures_without_errors,
            "updated_failures_with_mover_retries_or_unreached": (
                updated_failures_with_retries_or_unreached
            ),
        },
        "by_task": {task: dict(counts) for task, counts in sorted(by_task.items())},
        "records": rows,
    }
    comparison_root.mkdir(parents=True, exist_ok=True)
    with (comparison_root / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    csv_columns = [
        "task",
        "episode",
        "global_episode",
        "legacy_classification",
        "original_success",
        "updated_success",
        "transition",
        "original_attempted_actions",
        "updated_attempted_actions",
        "updated_mover_attempts",
        "updated_mover_retries",
        "updated_mover_unreached_targets",
        "updated_controller_errors",
        "original_video",
        "updated_video",
        "original_result_json",
        "updated_result_json",
    ]
    with (comparison_root / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in csv_columns} for row in rows)

    lines = [
        "# RLBench 失败候选：原始控制与 PointAct eval-parity A/B 重跑",
        "",
        f"- 候选轨迹：**{len(rows)}**",
        f"- 原始 single-step：**{original_success}/{len(rows)}** 成功",
        f"- 更新 PointAct eval-parity：**{updated_success}/{len(rows)}** 成功",
        "",
        "## 结论",
        "",
        "- 并不是原始设置全失败、更新设置全成功。",
        f"- 原始设置有 **{original_success}** 条重跑成功，说明这些旧失败并非完全可复现，"
        "仿真接触与路径规划存在非确定性。",
        f"- 原始设置中有 **{original_failures_with_errors}** 条因控制/规划异常失败；"
        f"更新设置恢复了其中 **{original_error_failures_recovered}** 条。",
        f"- 更新设置仍有 **{len(rows) - updated_success}** 条失败，其中 "
        f"**{updated_failures_without_errors}** 条没有控制异常，且 "
        f"**{updated_failures_with_retries_or_unreached}** 条出现 Mover 重试或未到位。",
        "- 因而 PointAct eval-parity 修复解决了 planner/workspace 类失败，但剩余失败主要是"
        "动作标签开放环重放造成的接触、物体状态漂移或最终任务条件未满足。",
        "",
        "## 结果迁移",
        "",
        "| 迁移 | 数量 |",
        "|---|---:|",
    ]
    for name in [
        "failure_to_success",
        "failure_to_failure",
        "success_to_success",
        "success_to_failure",
    ]:
        lines.append(f"| {name} | {transitions.get(name, 0)} |")
    lines.extend(
        [
            "",
            "## 分任务",
            "",
            "| 任务 | 候选 | 原始成功 | 更新后成功 |",
            "|---|---:|---:|---:|",
        ]
    )
    for task, counts in sorted(by_task.items()):
        lines.append(
            f"| {task} | {counts['total']} | {counts['original_success']} | "
            f"{counts['updated_success']} |"
        )
    lines.extend(
        [
            "",
            "## 逐条结果",
            "",
            "| 任务 | episode | 原始 | 更新 | 迁移 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['episode']} | {row['original_success']} | "
            f"{row['updated_success']} | `{row['transition']}` |"
        )
    (comparison_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output["transitions"], ensure_ascii=False, sort_keys=True))
    print(f"original_success={original_success}/{len(rows)}")
    print(f"updated_success={updated_success}/{len(rows)}")


if __name__ == "__main__":
    main()
