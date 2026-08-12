#!/usr/bin/env python3
"""Compare seeded actions from a baseline and function-preserving candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from probe_v043_dualview_action_drift import _predict_item, action_drift_metrics
from smolvla_model_inference import PointCloudMemmapDataset, SmolVLA_ModelInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    # Cover all ten tasks when num_samples >= 10; otherwise span the dataset.
    total = 137_590
    indices = np.linspace(0, total - 1, args.num_samples, dtype=np.int64).tolist()
    infer = SmolVLA_ModelInference(
        policy_path=args.candidate.resolve(),
        policy_repo_id=None,
        device=args.device,
        visualize_foreground=False,
    )
    try:
        residual_dataset = infer.load_dataset(dataset_root)
        primary_dataset = PointCloudMemmapDataset(
            residual_dataset.dataset,
            point_cloud_dir=dataset_root / "point_clouds",
            camera_views="agentview",
            camera_view_fusion="legacy_budget",
            gripper_points=500,
        )
        baseline_chunks = []
        candidate_chunks = []
        for index in indices:
            noise_seed = int(args.seed + index)
            infer.policy.config.camera_view_fusion = "legacy_budget"
            infer.policy.model.config.camera_view_fusion = "legacy_budget"
            baseline_chunks.append(_predict_item(infer, primary_dataset[index], seed=noise_seed))
            infer.policy.config.camera_view_fusion = "primary_residual"
            infer.policy.model.config.camera_view_fusion = "primary_residual"
            candidate_chunks.append(_predict_item(infer, residual_dataset[index], seed=noise_seed))
        baseline = torch.cat(baseline_chunks, dim=0)
        candidate = torch.cat(candidate_chunks, dim=0)
    finally:
        infer.close()
    raw = (candidate - baseline).abs()
    result = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "dataset_root": str(dataset_root),
        "indices": indices,
        "raw_max_abs": float(raw.max().item()),
        "raw_mean_abs": float(raw.mean().item()),
        "bit_exact": bool(torch.equal(candidate, baseline)),
        "metrics": action_drift_metrics(baseline, candidate, exec_steps=24),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["bit_exact"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
