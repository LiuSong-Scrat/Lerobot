#!/usr/bin/env python3
"""Summarize an explicit set of RLBench action-replay results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--expected",
        action="append",
        required=True,
        metavar="TASK=EPISODE,EPISODE",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    root = args.validation_root.expanduser().resolve()
    dataset = args.dataset_root.expanduser().resolve()
    expected: set[tuple[str, int]] = set()
    for value in args.expected:
        task, separator, episodes = value.partition("=")
        if not separator:
            raise ValueError("Expected TASK=EPISODE,EPISODE: " + value)
        for episode in episodes.split(","):
            expected.add((task, int(episode)))

    attempts: dict[tuple[str, int], list[dict]] = {}
    for path in sorted((root / "results").rglob("result.json")):
        result = load_json(path)
        key = (str(result["task"]), int(result["episode"]))
        if key not in expected:
            continue
        result["result_json"] = str(path)
        result["result_mtime_ns"] = path.stat().st_mtime_ns
        attempts.setdefault(key, []).append(result)
    missing = sorted(expected - set(attempts))
    if missing:
        raise RuntimeError("Missing targeted replay results: " + repr(missing))

    required_config = {
        "mode": "eef0_planning",
        "action_source": "parquet",
        "controller_profile": "pointact_eval",
        "mover_max_tries": 10,
        "clip_within_workspace": True,
        "gripper_after_reach": True,
        "pointact_pyrep_compat": True,
        "planner_max_time_ms": 1000,
        "gripper_mode": "delta_width_initial_sync",
        "gripper_delta_alignment": "current_minus_previous",
        "water_plant_collision": "enabled",
        "water_drop_collision": "original",
    }
    for key, result_attempts in attempts.items():
        for result in result_attempts:
            for config_key, value in required_config.items():
                if result.get(config_key) != value:
                    raise RuntimeError(
                        f"{key} has {config_key}={result.get(config_key)!r}, expected {value!r}"
                    )
            if abs(float(result.get("gripper_delta_threshold_m", -1.0)) - 0.003) > 1e-12:
                raise RuntimeError(f"{key} has wrong gripper threshold")

    # Parallel tail acceleration can race with a sequential worker that has
    # already spawned the same episode. Preserve every attempt in the audit,
    # and use the newest attempt as the canonical per-slot record.
    records = {
        key: max(result_attempts, key=lambda row: int(row["result_mtime_ns"]))
        for key, result_attempts in attempts.items()
    }
    ordered = [records[key] for key in sorted(records)]
    summary = {
        "schema": "rlbench_targeted_action_replay_summary_v1",
        "dataset_root": str(dataset),
        "expected_total": len(expected),
        "success": sum(bool(row["success"]) for row in ordered),
        "failure": sum(not bool(row["success"]) for row in ordered),
        "attempt_total": sum(len(values) for values in attempts.values()),
        "duplicate_attempts": {
            task + ":" + str(episode): [row["result_json"] for row in values]
            for (task, episode), values in sorted(attempts.items())
            if len(values) > 1
        },
        "gate": {**required_config, "gripper_delta_threshold_m": 0.003},
        "records": ordered,
    }
    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    columns = sorted({key for row in ordered for key in row})
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})

    lines = [
        "# 替换轨迹最终 Action Replay 复验",
        "",
        f"- 数据集：`{dataset}`",
        f"- 成功：**{summary['success']}/{len(expected)}**",
        "- 控制：PointAct eval，1000 ms，delta_width_initial_sync 0.003。",
        "",
        "| task | local episode | global episode | result |",
        "|---|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['task']} | {row['episode']} | {row['global_episode']} | "
            f"{'SUCCESS' if row['success'] else 'FAILURE'} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"success={summary['success']}/{len(expected)}")
    print(f"summary={root / 'summary.json'}")


if __name__ == "__main__":
    main()
