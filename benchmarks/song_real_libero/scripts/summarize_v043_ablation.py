#!/usr/bin/env python
"""Aggregate cumulative SmolVLA ablation checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


VARIANTS = (
    "smolvla_src",
    "smolvla_pointcloud",
    "smolvla_pointcloud_effseg",
    "smolvla_pointcloud_effseg_pointaction",
)
LOSS_PATTERN = re.compile(r"step:(?P<step>\d+).*?loss:(?P<loss>[0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    return parser.parse_args()


def training_losses(log_path: Path) -> dict[int, float]:
    losses: dict[int, float] = {}
    if not log_path.is_file():
        return losses
    for line in log_path.read_text(errors="replace").splitlines():
        match = LOSS_PATTERN.search(line)
        if match:
            losses[int(match.group("step"))] = float(match.group("loss"))
    return losses


def nearest_loss(losses: dict[int, float], step: int) -> float | None:
    candidates = [item for item in losses if item <= step]
    return losses[max(candidates)] if candidates else None


def main() -> None:
    args = parse_args()
    records = []
    for variant in VARIANTS:
        losses = training_losses(args.root / "logs" / f"train_{variant}.log")
        for summary_path in sorted((args.root / "eval" / variant).glob("step*/summary.json")):
            summary = json.loads(summary_path.read_text())
            step = int(summary_path.parent.name.removeprefix("step"))
            tasks = {}
            for task_id in (6, 8):
                matches = [
                    item
                    for item in summary["results"]
                    if item["suite"] == "libero_10" and int(item["task_id"]) == task_id
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one task {task_id} result in {summary_path}")
                episodes = matches[0]["episodes"]
                if len(episodes) != args.episodes_per_task:
                    raise RuntimeError(
                        f"Expected {args.episodes_per_task} task {task_id} episodes in {summary_path}, "
                        f"got {len(episodes)}"
                    )
                tasks[task_id] = sum(bool(item.get("success")) for item in episodes)
            records.append(
                {
                    "variant": variant,
                    "step": step,
                    "train_loss": nearest_loss(losses, step),
                    "task6_successes": tasks[6],
                    "task6_success_rate": tasks[6] / args.episodes_per_task,
                    "task8_successes": tasks[8],
                    "task8_success_rate": tasks[8] / args.episodes_per_task,
                    "overall_success_rate": (tasks[6] + tasks[8]) / (2 * args.episodes_per_task),
                    "summary_path": str(summary_path),
                }
            )

    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = args.root / "ablation_results.csv"
    fieldnames = list(records[0]) if records else [
        "variant", "step", "train_loss", "task6_successes", "task6_success_rate",
        "task8_successes", "task8_success_rate", "overall_success_rate", "summary_path",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    best = {}
    for variant in VARIANTS:
        candidates = [record for record in records if record["variant"] == variant]
        if candidates:
            best[variant] = max(candidates, key=lambda item: (item["overall_success_rate"], item["step"]))

    report = [
        "# SmolVLA cumulative ablation results",
        "",
        f"Each checkpoint uses fixed seeds and {args.episodes_per_task} episodes for each of LIBERO-10 tasks 6 and 8.",
        "",
        "| variant | best step | task 6 | task 8 | overall | train loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        record = best.get(variant)
        if record is None:
            report.append(f"| {variant} | pending | — | — | — | — |")
            continue
        loss = "—" if record["train_loss"] is None else f"{record['train_loss']:.6g}"
        report.append(
            f"| {variant} | {record['step']} | {record['task6_success_rate']:.1%} | "
            f"{record['task8_success_rate']:.1%} | {record['overall_success_rate']:.1%} | {loss} |"
        )
    report.extend(["", "## Incremental module effects", ""])
    for before, after, module in zip(VARIANTS, VARIANTS[1:], ("point cloud", "EffSeg", "PointAction Adapter")):
        if before in best and after in best:
            delta = best[after]["overall_success_rate"] - best[before]["overall_success_rate"]
            report.append(f"- {module}: best-checkpoint overall success change {delta:+.1%}.")
        else:
            report.append(f"- {module}: pending complete evaluations.")
    report.extend(
        [
            "",
            "Best-checkpoint deltas are descriptive. The full checkpoint curve in `ablation_results.csv` "
            "must be used to judge convergence and whether improvements persist rather than selecting a noisy peak.",
            "",
        ]
    )
    report_path = args.root / "ABlation_RESULTS.md"
    report_path.write_text("\n".join(report))
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
