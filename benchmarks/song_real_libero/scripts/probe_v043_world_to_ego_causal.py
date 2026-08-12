#!/usr/bin/env python3
"""Measure the causal action contribution of the learned World-to-Ego path.

The checkpoint is loaded once. For every fixed dataset item and flow-noise seed,
the probe compares normal inference with a runtime-only intervention that zeros
the full World-to-Ego attention output projection. The original parameters are
restored immediately; no checkpoint or dataset file is modified.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--exec-steps", type=int, default=24)
    parser.add_argument(
        "--intervention",
        choices=("cross_attention", "full"),
        default="cross_attention",
        help=(
            "cross_attention zeros only the World-to-Ego attention output projection; "
            "full also removes the conjugate-residual World twist correction."
        ),
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    return args


def run(args: argparse.Namespace) -> dict:
    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    started = time.time()

    infer = SmolVLA_ModelInference(
        policy_path=policy_path,
        policy_repo_id=None,
        device=args.device,
        visualize_foreground=False,
    )
    try:
        model = infer.policy.model
        attention = model.world_to_ego_cross_attn
        if attention is None:
            raise RuntimeError("Checkpoint does not contain a World-to-Ego attention path.")
        residual_head = getattr(model, "world_twist_residual_out_proj", None)
        if args.intervention == "full" and residual_head is None:
            raise RuntimeError("Full World-to-Ego intervention requires a World residual twist head.")
        dataset = infer.load_dataset(dataset_root)
        sample_count = min(int(args.num_samples), len(dataset))
        indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64).tolist()
        pose_cache: dict[int, np.ndarray] = {}

        def item_with_current_pose(index: int) -> dict:
            item = dict(dataset[index])
            episode_value = item["episode_index"]
            frame_value = item["frame_index"]
            episode_index = int(torch.as_tensor(episode_value).reshape(-1)[0].item())
            frame_index = int(torch.as_tensor(frame_value).reshape(-1)[0].item())
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

        weight = attention.out_proj.weight
        bias = attention.out_proj.bias
        saved_weight = weight.detach().clone()
        saved_bias = None if bias is None else bias.detach().clone()
        records = []
        repeat_records = []
        repeat_max_abs = 0.0

        for ordinal, index in enumerate(indices):
            item = item_with_current_pose(index)
            seed = int(args.seed + index)
            full_action = _predict_item(infer, item, seed=seed)
            repeated = _predict_item(infer, item, seed=seed)
            repeat_max_abs = max(
                repeat_max_abs,
                float((repeated - full_action).abs().max().item()),
            )
            repeat_metrics = action_drift_metrics(
                full_action,
                repeated,
                exec_steps=args.exec_steps,
            )
            repeat_records.append(
                _record_with_raw(
                    {
                        "dataset_index": int(index),
                        "noise_seed": seed,
                        **repeat_metrics,
                    },
                    full_action,
                    repeated,
                )
            )

            old_ablations = getattr(model, "inference_ablation_modalities", frozenset())
            if args.intervention == "full":
                model.inference_ablation_modalities = frozenset({*old_ablations, "world_to_ego"})
            else:
                with torch.no_grad():
                    weight.zero_()
                    if bias is not None:
                        bias.zero_()
            try:
                without_world_to_ego = _predict_item(infer, item, seed=seed)
            finally:
                if args.intervention == "full":
                    model.inference_ablation_modalities = old_ablations
                else:
                    with torch.no_grad():
                        weight.copy_(saved_weight)
                        if bias is not None and saved_bias is not None:
                            bias.copy_(saved_bias)

            metrics = action_drift_metrics(
                without_world_to_ego,
                full_action,
                exec_steps=args.exec_steps,
            )
            records.append(
                _record_with_raw(
                    {
                        "dataset_index": int(index),
                        "noise_seed": seed,
                        **metrics,
                    },
                    without_world_to_ego,
                    full_action,
                )
            )
            print(
                f"[{ordinal + 1}/{sample_count}] index={index} "
                f"translation_mean_m={metrics['translation_error_m']['mean']:.6f} "
                f"rotation_mean_deg={metrics['rotation_error_deg']['mean']:.4f} "
                f"gripper_mae={metrics['gripper_abs_error']['mean']:.6f}",
                flush=True,
            )

        aggregate = _aggregate_records(records)
        repeat_aggregate = _aggregate_records(repeat_records)
        payload = {
            "status": "complete",
            "intervention": (
                "disable_world_to_ego_cross_attention_and_residual_twist"
                if args.intervention == "full"
                else "zero_world_to_ego_cross_attention_out_projection"
            ),
            "policy_path": str(policy_path),
            "dataset_root": str(dataset_root),
            "device": args.device,
            "sample_indices": indices,
            "exec_steps": int(args.exec_steps),
            "deterministic_repeat_max_abs": repeat_max_abs,
            "repeat_control_action_effect": repeat_aggregate,
            "world_to_ego_out_projection": {
                "weight_frobenius_norm": float(saved_weight.float().norm().item()),
                "bias_l2_norm": 0.0 if saved_bias is None else float(saved_bias.float().norm().item()),
            },
            "world_twist_residual_projection": (
                None
                if residual_head is None
                else {
                    "weight_frobenius_norm": float(residual_head.weight.detach().float().norm().item()),
                    "bias_l2_norm": float(residual_head.bias.detach().float().norm().item()),
                }
            ),
            "causal_action_effect": aggregate,
            "samples": records,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_write_json(output_json, payload)
        return payload
    finally:
        infer.close()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"status": payload["status"], "causal_action_effect": payload["causal_action_effect"]}, indent=2))


if __name__ == "__main__":
    main()
