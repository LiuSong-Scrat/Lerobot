#!/usr/bin/env python3
"""Separate pretrained-path drift from the learned World-to-Ego correction.

The probe evaluates uniformly spaced dataset frames with fixed flow-noise seeds.
It loads the immutable Ego baseline first, releases it, then loads the WorldFlow
checkpoint and evaluates both its complete path and a runtime-only intervention
that removes all World-to-Ego feedback. No checkpoint or dataset is modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe_v043_dualview_action_drift import (
    _aggregate_records,
    _atomic_write_json,
    _predict_item,
    _record_with_raw,
    action_drift_metrics,
)
from smolvla_model_inference import SmolVLA_ModelInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-policy-path", type=Path, required=True)
    parser.add_argument("--worldflow-policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--exec-steps", type=int, default=24)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    return args


def _item_with_current_pose(dataset, dataset_root: Path, index: int, pose_cache: dict[int, np.ndarray]):
    item = dict(dataset[index])
    episode_index = int(torch.as_tensor(item["episode_index"]).reshape(-1)[0].item())
    frame_index = int(torch.as_tensor(item["frame_index"]).reshape(-1)[0].item())
    poses = pose_cache.get(episode_index)
    if poses is None:
        pose_path = dataset_root / "world_ee_poses" / f"episode_{episode_index:06d}.npy"
        poses = np.load(pose_path, mmap_mode="r")
        if poses.ndim != 2 or poses.shape[-1] != 9:
            raise ValueError(f"Expected WorldFlow pose sidecar (T,9), got {poses.shape}.")
        pose_cache[episode_index] = poses
    current_index = min(max(frame_index, 0), len(poses) - 1)
    item["worldflow.current_ee_pose"] = torch.from_numpy(
        np.array(poses[current_index], dtype=np.float32, copy=True)
    )
    return item


def _metric_record(reference: torch.Tensor, candidate: torch.Tensor, index: int, seed: int, steps: int):
    metrics = action_drift_metrics(reference, candidate, exec_steps=steps)
    return _record_with_raw(
        {"dataset_index": int(index), "noise_seed": int(seed), **metrics},
        reference,
        candidate,
    )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    baseline_path = args.baseline_policy_path.expanduser().resolve()
    worldflow_path = args.worldflow_policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()

    baseline_infer = SmolVLA_ModelInference(
        policy_path=baseline_path,
        policy_repo_id=None,
        device=args.device,
        visualize_foreground=False,
    )
    baseline_dataset = baseline_infer.load_dataset(dataset_root)
    sample_count = min(int(args.num_samples), len(baseline_dataset))
    indices = np.linspace(0, len(baseline_dataset) - 1, sample_count, dtype=np.int64).tolist()
    baseline_actions = {
        int(index): _predict_item(baseline_infer, baseline_dataset[index], seed=int(args.seed + index))
        for index in indices
    }
    baseline_infer.close()
    del baseline_infer, baseline_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    world_infer = SmolVLA_ModelInference(
        policy_path=worldflow_path,
        policy_repo_id=None,
        device=args.device,
        visualize_foreground=False,
    )
    try:
        dataset = world_infer.load_dataset(dataset_root)
        model = world_infer.policy.model
        pose_cache: dict[int, np.ndarray] = {}
        pretrained_drift_records = []
        world_correction_records = []
        combined_drift_records = []
        repeat_records = []

        for ordinal, index in enumerate(indices):
            index = int(index)
            seed = int(args.seed + index)
            item = _item_with_current_pose(dataset, dataset_root, index, pose_cache)
            old_ablations = model.inference_ablation_modalities
            model.inference_ablation_modalities = frozenset({*old_ablations, "world_to_ego"})
            try:
                no_world = _predict_item(world_infer, item, seed=seed)
            finally:
                model.inference_ablation_modalities = old_ablations
            complete = _predict_item(world_infer, item, seed=seed)
            repeated = _predict_item(world_infer, item, seed=seed)
            baseline = baseline_actions[index]

            pretrained_drift_records.append(
                _metric_record(baseline, no_world, index, seed, args.exec_steps)
            )
            world_correction_records.append(
                _metric_record(no_world, complete, index, seed, args.exec_steps)
            )
            combined_drift_records.append(
                _metric_record(baseline, complete, index, seed, args.exec_steps)
            )
            repeat_records.append(
                _metric_record(complete, repeated, index, seed, args.exec_steps)
            )
            print(f"[{ordinal + 1}/{sample_count}] index={index}", flush=True)

        payload = {
            "status": "complete",
            "task_specific_training_or_inference": False,
            "baseline_policy_path": str(baseline_path),
            "worldflow_policy_path": str(worldflow_path),
            "dataset_root": str(dataset_root),
            "sample_indices": indices,
            "exec_steps": int(args.exec_steps),
            "baseline_to_trained_ego_without_world": _aggregate_records(pretrained_drift_records),
            "trained_ego_to_complete_worldflow": _aggregate_records(world_correction_records),
            "baseline_to_complete_worldflow": _aggregate_records(combined_drift_records),
            "deterministic_repeat": _aggregate_records(repeat_records),
            "elapsed_seconds": time.time() - started,
        }
        _atomic_write_json(output_json, payload)
        return payload
    finally:
        world_infer.close()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
