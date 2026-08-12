#!/usr/bin/env python3
"""Measure the action perturbation caused only by guarded dual-view composition.

The probe loads one immutable Ego-only checkpoint once, then evaluates the same
dataset frames and seeded flow noise with either the unchanged 10k agentview
cloud or a deterministic 9:1 agentview/wrist cloud.  It does not train or alter
the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.policies.smolvla.song_pointseg import pose9_to_matrix

from smolvla_model_inference import PointCloudMemmapDataset, SmolVLA_ModelInference


DEFAULT_TRANSLATION_MEAN_M = 0.005
DEFAULT_TRANSLATION_P95_M = 0.015
DEFAULT_ROTATION_MEAN_DEG = 3.0
DEFAULT_ROTATION_P95_DEG = 10.0
DEFAULT_GRIPPER_MAE = 0.05
DEFAULT_GRIPPER_SIGN_MISMATCH = 0.0
DEFAULT_REPEAT_MAX_ABS = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=False)
    parser.add_argument("--dataset-root", type=Path, required=False)
    parser.add_argument("--output-json", type=Path, required=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--exec-steps", type=int, default=24)
    parser.add_argument("--camera-views", default="agentview,robot0_eye_in_hand")
    parser.add_argument("--camera-view-weights", default="9,1")
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--translation-mean-max-m", type=float, default=DEFAULT_TRANSLATION_MEAN_M)
    parser.add_argument("--translation-p95-max-m", type=float, default=DEFAULT_TRANSLATION_P95_M)
    parser.add_argument("--rotation-mean-max-deg", type=float, default=DEFAULT_ROTATION_MEAN_DEG)
    parser.add_argument("--rotation-p95-max-deg", type=float, default=DEFAULT_ROTATION_P95_DEG)
    parser.add_argument("--gripper-mae-max", type=float, default=DEFAULT_GRIPPER_MAE)
    parser.add_argument(
        "--gripper-sign-mismatch-max",
        type=float,
        default=DEFAULT_GRIPPER_SIGN_MISMATCH,
    )
    parser.add_argument("--repeat-max-abs", type=float, default=DEFAULT_REPEAT_MAX_ABS)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        for name in ("policy_path", "dataset_root", "output_json"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.exec_steps <= 0:
        parser.error("--exec-steps must be positive")
    return args


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty metric.")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def action_drift_metrics(
    baseline: torch.Tensor,
    dualview: torch.Tensor,
    *,
    exec_steps: int,
) -> dict[str, Any]:
    """Return physical action drift for two [B,T,D>=10] pose9 chunks."""

    baseline = baseline.detach().to(device="cpu", dtype=torch.float32)
    dualview = dualview.detach().to(device="cpu", dtype=torch.float32)
    if baseline.shape != dualview.shape or baseline.ndim != 3:
        raise ValueError(f"Expected equal [B,T,D] chunks, got {baseline.shape} and {dualview.shape}.")
    if baseline.shape[-1] < 10:
        raise ValueError(f"Pose9 + gripper drift requires action dim >=10, got {baseline.shape[-1]}.")
    steps = min(int(exec_steps), int(baseline.shape[1]))
    baseline = baseline[:, :steps]
    dualview = dualview[:, :steps]
    translation_error = torch.linalg.vector_norm(dualview[..., :3] - baseline[..., :3], dim=-1)

    baseline_transform = pose9_to_matrix(baseline[..., :9])
    dualview_transform = pose9_to_matrix(dualview[..., :9])
    relative_rotation = dualview_transform[..., :3, :3] @ baseline_transform[..., :3, :3].transpose(-1, -2)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_error_deg = torch.rad2deg(torch.acos(cosine))

    gripper_error = (dualview[..., 9] - baseline[..., 9]).abs()
    gripper_sign_mismatch = (dualview[..., 9] >= 0.0) != (baseline[..., 9] >= 0.0)
    raw_error = (dualview - baseline).abs()
    return {
        "steps": steps,
        "translation_error_m": _summary(translation_error.numpy()),
        "rotation_error_deg": _summary(rotation_error_deg.numpy()),
        "gripper_abs_error": _summary(gripper_error.numpy()),
        "gripper_sign_mismatch_fraction": float(gripper_sign_mismatch.float().mean().item()),
        "raw_abs_error": _summary(raw_error.numpy()),
        "first_action": {
            "translation_error_m": float(translation_error[:, 0].mean().item()),
            "rotation_error_deg": float(rotation_error_deg[:, 0].mean().item()),
            "gripper_abs_error": float(gripper_error[:, 0].mean().item()),
            "raw_max_abs": float(raw_error[:, 0].max().item()),
        },
    }


def evaluate_gate(
    aggregate: dict[str, Any],
    *,
    repeat_max_abs: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "deterministic_repeat": repeat_max_abs <= thresholds["repeat_max_abs"],
        "translation_mean": (
            aggregate["translation_error_m"]["mean"] <= thresholds["translation_mean_max_m"]
        ),
        "translation_p95": (
            aggregate["translation_error_m"]["p95"] <= thresholds["translation_p95_max_m"]
        ),
        "rotation_mean": (
            aggregate["rotation_error_deg"]["mean"] <= thresholds["rotation_mean_max_deg"]
        ),
        "rotation_p95": (
            aggregate["rotation_error_deg"]["p95"] <= thresholds["rotation_p95_max_deg"]
        ),
        "gripper_mae": aggregate["gripper_abs_error"]["mean"] <= thresholds["gripper_mae_max"],
        "gripper_sign": (
            aggregate["gripper_sign_mismatch_fraction"]
            <= thresholds["gripper_sign_mismatch_max"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    translations: list[float] = []
    rotations: list[float] = []
    grippers: list[float] = []
    raw_errors: list[float] = []
    sign_mismatch_numerator = 0.0
    sign_mismatch_denominator = 0
    for record in records:
        raw = record.pop("_raw")
        translations.extend(raw["translation"])
        rotations.extend(raw["rotation"])
        grippers.extend(raw["gripper"])
        raw_errors.extend(raw["raw"])
        sign_mismatch_numerator += float(raw["sign_mismatch_sum"])
        sign_mismatch_denominator += int(raw["sign_mismatch_count"])
    return {
        "translation_error_m": _summary(np.asarray(translations)),
        "rotation_error_deg": _summary(np.asarray(rotations)),
        "gripper_abs_error": _summary(np.asarray(grippers)),
        "gripper_sign_mismatch_fraction": sign_mismatch_numerator / max(1, sign_mismatch_denominator),
        "raw_abs_error": _summary(np.asarray(raw_errors)),
    }


def _record_with_raw(metrics: dict[str, Any], baseline: torch.Tensor, dualview: torch.Tensor) -> dict[str, Any]:
    steps = int(metrics["steps"])
    baseline = baseline.detach().cpu().float()[:, :steps]
    dualview = dualview.detach().cpu().float()[:, :steps]
    baseline_transform = pose9_to_matrix(baseline[..., :9])
    dualview_transform = pose9_to_matrix(dualview[..., :9])
    relative_rotation = dualview_transform[..., :3, :3] @ baseline_transform[..., :3, :3].transpose(-1, -2)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation = torch.rad2deg(torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)))
    translation = torch.linalg.vector_norm(dualview[..., :3] - baseline[..., :3], dim=-1)
    gripper = (dualview[..., 9] - baseline[..., 9]).abs()
    sign = (dualview[..., 9] >= 0.0) != (baseline[..., 9] >= 0.0)
    return {
        **metrics,
        "_raw": {
            "translation": translation.reshape(-1).tolist(),
            "rotation": rotation.reshape(-1).tolist(),
            "gripper": gripper.reshape(-1).tolist(),
            "raw": (dualview - baseline).abs().reshape(-1).tolist(),
            "sign_mismatch_sum": float(sign.sum().item()),
            "sign_mismatch_count": int(sign.numel()),
        },
    }


def _predict_item(infer: SmolVLA_ModelInference, item: dict[str, Any], *, seed: int) -> torch.Tensor:
    batch = default_collate([dict(item)])
    model_batch = infer.preprocessor(batch)
    noise = infer._make_seeded_action_noise(model_batch, seed)
    policy_device = next(infer.policy.parameters()).device
    cuda_devices: list[int] = []
    if policy_device.type == "cuda":
        cuda_devices = [policy_device.index if policy_device.index is not None else torch.cuda.current_device()]
    infer.policy.reset()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if policy_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        action = infer.policy.predict_action_chunk(model_batch, noise=noise)
    return infer.postprocessor(action).detach().cpu().float()


def _replace_only_point_cloud(
    baseline_item: dict[str, Any],
    dualview_item: dict[str, Any],
) -> dict[str, Any]:
    """Keep RGB/state/language byte-identical and replace only point geometry."""

    key = "observation.point_cloud"
    if key not in baseline_item or key not in dualview_item:
        raise KeyError(f"Both probe items must contain {key!r}.")
    result = dict(baseline_item)
    result[key] = dualview_item[key]
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def run_self_test() -> None:
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    baseline = identity.view(1, 1, 10).repeat(1, 2, 1)
    dualview = baseline.clone()
    dualview[..., 0] += 0.01
    angle = math.pi / 2.0
    dualview[..., 3:9] = torch.tensor([math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0])
    dualview[..., 9] = 0.2
    metrics = action_drift_metrics(baseline, dualview, exec_steps=2)
    assert abs(metrics["translation_error_m"]["mean"] - 0.01) < 1e-6
    assert abs(metrics["rotation_error_deg"]["mean"] - 90.0) < 1e-4
    assert abs(metrics["gripper_abs_error"]["mean"] - 0.2) < 1e-6
    rgb = torch.ones(3)
    base_item = {"observation.point_cloud": torch.zeros(2, 6), "observation.image": rgb}
    dual_item = {"observation.point_cloud": torch.ones(2, 6), "observation.image": torch.zeros(3)}
    replaced = _replace_only_point_cloud(base_item, dual_item)
    assert replaced["observation.image"] is rgb
    assert torch.equal(replaced["observation.point_cloud"], dual_item["observation.point_cloud"])
    print("self-test passed")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
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

    dry_payload = {
        "status": "dry_run",
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "output_json": str(output_json),
        "device": args.device,
        "num_samples": args.num_samples,
        "exec_steps": args.exec_steps,
        "camera_views": args.camera_views,
        "camera_view_weights": args.camera_view_weights,
    }
    if args.dry_run:
        return dry_payload
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This checkpoint's LitePT/spconv action probe requires an available CUDA device.")

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
        dualview_dataset = PointCloudMemmapDataset(
            baseline_dataset.dataset,
            point_cloud_dir=dataset_root / "point_clouds",
            camera_views=args.camera_views,
            camera_view_weights=args.camera_view_weights,
            gripper_points=args.gripper_points,
        )
        if len(baseline_dataset) != len(dualview_dataset):
            raise RuntimeError("Baseline and dual-view dataset lengths differ.")
        sample_count = min(int(args.num_samples), len(baseline_dataset))
        indices = np.linspace(0, len(baseline_dataset) - 1, sample_count, dtype=np.int64).tolist()

        first_seed = int(args.seed + indices[0])
        first_item = baseline_dataset[indices[0]]
        repeat_a = _predict_item(infer, first_item, seed=first_seed)
        repeat_b = _predict_item(infer, first_item, seed=first_seed)
        repeat_max_abs = float((repeat_b - repeat_a).abs().max().item())

        records: list[dict[str, Any]] = []
        for ordinal, index in enumerate(indices):
            seed = int(args.seed + index)
            baseline_item = first_item if ordinal == 0 else baseline_dataset[index]
            dualview_item = _replace_only_point_cloud(baseline_item, dualview_dataset[index])
            baseline_action = repeat_a if ordinal == 0 else _predict_item(infer, baseline_item, seed=seed)
            dualview_action = _predict_item(infer, dualview_item, seed=seed)
            metrics = action_drift_metrics(baseline_action, dualview_action, exec_steps=args.exec_steps)
            records.append(
                _record_with_raw(
                    {"dataset_index": int(index), "noise_seed": seed, **metrics},
                    baseline_action,
                    dualview_action,
                )
            )
            print(
                f"[{ordinal + 1}/{len(indices)}] index={index} "
                f"translation_mean_m={metrics['translation_error_m']['mean']:.6f} "
                f"rotation_mean_deg={metrics['rotation_error_deg']['mean']:.4f} "
                f"gripper_mae={metrics['gripper_abs_error']['mean']:.6f}",
                flush=True,
            )

        aggregate = _aggregate_records(records)
        thresholds = {
            "translation_mean_max_m": float(args.translation_mean_max_m),
            "translation_p95_max_m": float(args.translation_p95_max_m),
            "rotation_mean_max_deg": float(args.rotation_mean_max_deg),
            "rotation_p95_max_deg": float(args.rotation_p95_max_deg),
            "gripper_mae_max": float(args.gripper_mae_max),
            "gripper_sign_mismatch_max": float(args.gripper_sign_mismatch_max),
            "repeat_max_abs": float(args.repeat_max_abs),
        }
        gate = evaluate_gate(aggregate, repeat_max_abs=repeat_max_abs, thresholds=thresholds)
        result = {
            "status": "passed" if gate["passed"] else "failed",
            "policy_path": str(policy_path),
            "dataset_root": str(dataset_root),
            "device": str(args.device),
            "camera_views": args.camera_views.split(","),
            "camera_view_weights": [float(value) for value in args.camera_view_weights.split(",")],
            "gripper_points": int(args.gripper_points),
            "dataset_length": len(baseline_dataset),
            "sample_indices": indices,
            "exec_steps": int(args.exec_steps),
            "deterministic_repeat_max_abs": repeat_max_abs,
            "aggregate": aggregate,
            "gate": gate,
            "samples": records,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_write_json(output_json, result)
        return result
    finally:
        infer.close()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    result = run_probe(args)
    print(json.dumps(result if args.dry_run else {"status": result["status"], "gate": result["gate"]}, indent=2))
    if args.require_pass and not args.dry_run and not result["gate"]["passed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
