#!/usr/bin/env python3
"""Summarize standalone RLBench offline action-error audit JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HORIZONS = ("first_1", "first_16", "full_32")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def contract_for(run_name, report):
    if run_name.startswith("wep_vla_v043-20000+20000"):
        return "same_current_2task_contract"
    if report.get("config", {}).get("image_key_alias"):
        return "ood_legacy_agentview_alias"
    return "ood_different_training_input"


def metric_row(run_name, checkpoint, contract, scope, horizon, metric):
    return {
        "run": run_name,
        "checkpoint": checkpoint,
        "contract": contract,
        "scope": scope,
        "horizon": horizon,
        "rows": metric["rows"],
        "translation_mean_mm": metric["position_error_mm_mean"],
        "translation_rmse_mm": metric["position_error_mm_rmse"],
        "translation_p95_mm": metric["position_error_mm_p95"],
        "translation_max_mm": metric["position_error_mm_max"],
        "rotation_mean_deg": metric["rotation_error_deg_mean"],
        "rotation_rmse_deg": metric["rotation_error_deg_rmse"],
        "rotation_p95_deg": metric["rotation_error_deg_p95"],
        "rotation_max_deg": metric["rotation_error_deg_max"],
        "gripper_mean_mm": metric["gripper_error_mm_mean"],
        "gripper_rmse_mm": metric["gripper_error_mm_rmse"],
        "gripper_p95_mm": metric["gripper_error_mm_p95"],
        "gripper_max_mm": metric["gripper_error_mm_max"],
        "gripper_binary_accuracy": metric["gripper_binary_accuracy"],
        "gripper_binary_mismatches": metric["gripper_binary_mismatches"],
    }


def main():
    args = parse_args()
    reports = []
    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem
        if "__" not in stem:
            continue
        run_name, checkpoint = stem.rsplit("__", 1)
        contract = contract_for(run_name, report)
        reports.append(
            {
                "run": run_name,
                "checkpoint": checkpoint,
                "contract": contract,
                "source": str(path.resolve()),
            }
        )
        for horizon in HORIZONS:
            rows.append(
                metric_row(
                    run_name,
                    checkpoint,
                    contract,
                    "overall",
                    horizon,
                    report["overall"][horizon],
                )
            )
            for task, aggregates in sorted(report["aggregates"].items()):
                rows.append(
                    metric_row(
                        run_name,
                        checkpoint,
                        contract,
                        task,
                        horizon,
                        aggregates[horizon],
                    )
                )

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_prefix.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "metric_definition": {
                    "dataset_root": str(args.dataset_root.resolve()),
                    "sampling": "12 evenly spaced episodes per task x 5 non-tail frames",
                    "noise_seed": "20260801 + deterministic sample index",
                    "translation": "Euclidean position distance in millimetres",
                    "rotation": "SO(3) geodesic angle in degrees",
                    "gripper": "absolute physical opening-width error in millimetres",
                    "primary_horizon": "first_16 (the configured online execution window)",
                    "target_space": "dataset label after the training UMI world-to-current-EEF rigid transform",
                },
                "models": reports,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    primary = [row for row in rows if row["scope"] == "overall" and row["horizon"] == "first_16"]
    markdown = [
        "# RLBench checkpoint action-error audit",
        "",
        "Primary metric: teacher-forced `first_16`, matching the configured online execution window.",
        "",
        "| Run | Step | Contract | Translation mean / P95 (mm) | Rotation mean / P95 (deg) | Gripper mean / P95 (mm) | Open/close acc. |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        markdown.append(
            "| {run} | {checkpoint} | {contract} | {translation_mean_mm:.3f} / {translation_p95_mm:.3f} "
            "| {rotation_mean_deg:.3f} / {rotation_p95_deg:.3f} "
            "| {gripper_mean_mm:.3f} / {gripper_p95_mm:.3f} | {accuracy:.2f}% |".format(
                accuracy=100.0 * row["gripper_binary_accuracy"], **row
            )
        )
    task_primary = [
        row for row in rows if row["scope"] != "overall" and row["horizon"] == "first_16"
    ]
    if task_primary:
        markdown.extend(
            [
                "",
                "## Per-task first_16",
                "",
                "| Run | Step | Task | Translation mean / P95 (mm) | Rotation mean / P95 (deg) | Gripper mean / P95 (mm) | Open/close acc. |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in task_primary:
            markdown.append(
                "| {run} | {checkpoint} | {scope} | {translation_mean_mm:.3f} / {translation_p95_mm:.3f} "
                "| {rotation_mean_deg:.3f} / {rotation_p95_deg:.3f} "
                "| {gripper_mean_mm:.3f} / {gripper_p95_mm:.3f} | {accuracy:.2f}% |".format(
                    accuracy=100.0 * row["gripper_binary_accuracy"], **row
                )
            )
    markdown_path = output_prefix.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(summary_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
