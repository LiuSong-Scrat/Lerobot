#!/usr/bin/env python3
"""Calibrate dual-view point budgets against singleton baseline repeat noise."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from probe_v043_dualview_action_drift import (
    PointCloudMemmapDataset,
    SmolVLA_ModelInference,
    _aggregate_records,
    _predict_item,
    _record_with_raw,
    _replace_only_point_cloud,
    action_drift_metrics,
)


def parse_candidates(value: str) -> list[str]:
    candidates = [part.strip() for part in value.split(";") if part.strip()]
    if not candidates:
        raise ValueError("At least one camera-view weight candidate is required.")
    for candidate in candidates:
        weights = [float(part.strip()) for part in candidate.split(",")]
        if len(weights) != 2 or any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError(f"Invalid two-view positive weight candidate: {candidate!r}.")
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--exec-steps", type=int, default=24)
    parser.add_argument("--camera-views", default="agentview,robot0_eye_in_hand")
    parser.add_argument(
        "--weight-candidates",
        default="9,1;19,1;49,1;99,1",
        help="Semicolon-separated primary,wrist ratios, ordered from most to least wrist coverage.",
    )
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.weight_candidates = parse_candidates(args.weight_candidates)
    if args.num_samples <= 0 or args.exec_steps <= 0:
        parser.error("--num-samples and --exec-steps must be positive")
    return args


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _relative_gate(candidate: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    excess = {
        "translation_mean_m": max(
            0.0,
            candidate["translation_error_m"]["mean"] - repeat["translation_error_m"]["mean"],
        ),
        "translation_p95_m": max(
            0.0,
            candidate["translation_error_m"]["p95"] - repeat["translation_error_m"]["p95"],
        ),
        "rotation_mean_deg": max(
            0.0,
            candidate["rotation_error_deg"]["mean"] - repeat["rotation_error_deg"]["mean"],
        ),
        "rotation_p95_deg": max(
            0.0,
            candidate["rotation_error_deg"]["p95"] - repeat["rotation_error_deg"]["p95"],
        ),
        "gripper_mean": max(
            0.0,
            candidate["gripper_abs_error"]["mean"] - repeat["gripper_abs_error"]["mean"],
        ),
    }
    limits = {
        "translation_mean_m": 0.002,
        "translation_p95_m": 0.005,
        "rotation_mean_deg": 1.0,
        "rotation_p95_deg": 3.0,
        "gripper_mean": 0.01,
    }
    checks = {key: excess[key] <= limit for key, limit in limits.items()}
    checks["gripper_sign"] = candidate["gripper_sign_mismatch_fraction"] == 0.0
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "excess_over_agentview_repeat": excess,
        "excess_limits": limits,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    for required in (
        policy_path / "config.json",
        policy_path / "model.safetensors",
        dataset_root / "meta" / "info.json",
        dataset_root / "point_clouds",
        dataset_root / "point_clouds_robot0_eye_in_hand",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    dry = {
        "status": "dry_run",
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "output_json": str(output_json),
        "weight_candidates": args.weight_candidates,
        "num_samples": args.num_samples,
        "exec_steps": args.exec_steps,
    }
    if args.dry_run:
        return dry

    started = time.time()
    infer = SmolVLA_ModelInference(
        policy_path=policy_path,
        policy_repo_id=None,
        device=args.device,
        camera_views="agentview",
        visualize_foreground=False,
    )
    try:
        baseline_dataset = infer.load_dataset(dataset_root)
        candidate_datasets = {
            weights: PointCloudMemmapDataset(
                baseline_dataset.dataset,
                point_cloud_dir=dataset_root / "point_clouds",
                camera_views=args.camera_views,
                camera_view_weights=weights,
                gripper_points=args.gripper_points,
            )
            for weights in args.weight_candidates
        }
        count = min(int(args.num_samples), len(baseline_dataset))
        indices = np.linspace(0, len(baseline_dataset) - 1, count, dtype=np.int64).tolist()
        repeat_records: list[dict[str, Any]] = []
        candidate_records: dict[str, list[dict[str, Any]]] = {
            weights: [] for weights in args.weight_candidates
        }
        sample_summaries: list[dict[str, Any]] = []

        for ordinal, index in enumerate(indices):
            seed = int(args.seed + index)
            baseline_item = baseline_dataset[index]
            baseline_first = _predict_item(infer, baseline_item, seed=seed)

            order = list(args.weight_candidates)
            order = order[ordinal % len(order) :] + order[: ordinal % len(order)]
            candidate_actions = {}
            for weights in order:
                dual_item = _replace_only_point_cloud(baseline_item, candidate_datasets[weights][index])
                candidate_actions[weights] = _predict_item(infer, dual_item, seed=seed)
            baseline_second = _predict_item(infer, baseline_item, seed=seed)

            repeat_metrics = action_drift_metrics(
                baseline_first,
                baseline_second,
                exec_steps=args.exec_steps,
            )
            repeat_records.append(
                _record_with_raw(
                    {"dataset_index": int(index), "noise_seed": seed, **repeat_metrics},
                    baseline_first,
                    baseline_second,
                )
            )
            per_candidate = {}
            for weights in args.weight_candidates:
                comparisons = []
                for baseline_action in (baseline_first, baseline_second):
                    metrics = action_drift_metrics(
                        baseline_action,
                        candidate_actions[weights],
                        exec_steps=args.exec_steps,
                    )
                    comparisons.append(metrics)
                    candidate_records[weights].append(
                        _record_with_raw(
                            {"dataset_index": int(index), "noise_seed": seed, **metrics},
                            baseline_action,
                            candidate_actions[weights],
                        )
                    )
                per_candidate[weights] = {
                    "translation_mean_m": float(
                        np.mean([item["translation_error_m"]["mean"] for item in comparisons])
                    ),
                    "rotation_mean_deg": float(
                        np.mean([item["rotation_error_deg"]["mean"] for item in comparisons])
                    ),
                }
            sample_summaries.append(
                {
                    "dataset_index": int(index),
                    "noise_seed": seed,
                    "agentview_repeat": repeat_metrics,
                    "candidates": per_candidate,
                }
            )
            print(
                f"[{ordinal + 1}/{len(indices)}] index={index} "
                f"repeat_translation_m={repeat_metrics['translation_error_m']['mean']:.6f} "
                + " ".join(
                    f"{weights}={per_candidate[weights]['translation_mean_m']:.6f}m"
                    for weights in args.weight_candidates
                ),
                flush=True,
            )

        repeat_aggregate = _aggregate_records(repeat_records)
        repeat_stability_checks = {
            "translation_mean": repeat_aggregate["translation_error_m"]["mean"] <= 0.010,
            "translation_p95": repeat_aggregate["translation_error_m"]["p95"] <= 0.030,
            "rotation_mean": repeat_aggregate["rotation_error_deg"]["mean"] <= 5.0,
            "rotation_p95": repeat_aggregate["rotation_error_deg"]["p95"] <= 15.0,
            "gripper_mean": repeat_aggregate["gripper_abs_error"]["mean"] <= 0.05,
            "gripper_sign": repeat_aggregate["gripper_sign_mismatch_fraction"] == 0.0,
        }
        repeat_stable = all(repeat_stability_checks.values())
        candidate_results = {}
        selected = None
        for weights in args.weight_candidates:
            aggregate = _aggregate_records(candidate_records[weights])
            gate = _relative_gate(aggregate, repeat_aggregate)
            gate["passed"] = bool(gate["passed"] and repeat_stable)
            candidate_results[weights] = {"aggregate": aggregate, "relative_gate": gate}
            if selected is None and gate["passed"]:
                selected = weights

        result = {
            "status": "selected" if selected is not None else "no_candidate_passed",
            "policy_path": str(policy_path),
            "dataset_root": str(dataset_root),
            "device": args.device,
            "camera_views": args.camera_views.split(","),
            "weight_candidates": args.weight_candidates,
            "sample_indices": indices,
            "exec_steps": args.exec_steps,
            "agentview_repeat": {
                "aggregate": repeat_aggregate,
                "stable": repeat_stable,
                "checks": repeat_stability_checks,
            },
            "candidates": candidate_results,
            "selected_camera_view_weights": selected,
            "samples": sample_summaries,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_write_json(output_json, result)
        return result
    finally:
        infer.close()


def main() -> None:
    args = parse_args()
    result = run(args)
    summary = result if args.dry_run else {
        "status": result["status"],
        "selected_camera_view_weights": result["selected_camera_view_weights"],
        "agentview_repeat": result["agentview_repeat"],
        "candidate_gates": {
            weights: payload["relative_gate"] for weights, payload in result["candidates"].items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
