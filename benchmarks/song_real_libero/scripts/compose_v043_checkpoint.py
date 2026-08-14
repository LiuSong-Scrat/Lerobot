#!/usr/bin/env python3
"""Compose a baseline-compatible FPS checkpoint by linear model interpolation.

The output directory must not already exist.  Processor state is copied from
the baseline checkpoint because the baseline behavior is the preservation
anchor.  SmolVLA currently uses IDENTITY normalization for this experiment,
but keeping the baseline processor files also makes the alpha=0 control exact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROCESSOR_FILES = (
    "policy_preprocessor.json",
    "policy_preprocessor_step_6_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--fps-target-points",
        type=int,
        default=10_000,
        help="Inference/cache point budget after multi-view FPS fusion.",
    )
    parser.add_argument(
        "--camera-view-fusion",
        choices=(
            "fps",
            "voxel_fps",
            "voxel_cover_fps",
            "novelty_union",
            "multiscale_novelty_union",
            "transport_novelty_union",
            "uniform_union",
            "primary_residual",
        ),
        default="fps",
    )
    parser.add_argument("--camera-view-voxel-size", type=float, default=0.005)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if not 500 < args.fps_target_points <= 19_500:
        parser.error("--fps-target-points must be in [501, 19500]")
    return args


def require_checkpoint(path: Path) -> None:
    for name in ("config.json", "model.safetensors", *PROCESSOR_FILES):
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)


def tensor_schema(path: Path) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            key: (tuple(handle.get_tensor(key).shape), handle.get_tensor(key).dtype)
            for key in handle.keys()
        }


def interpolate_weights(baseline: Path, finetuned: Path, output: Path, alpha: float) -> None:
    if alpha == 0.0:
        # Relative symlink keeps the control bit-identical without duplicating
        # a 1.4 GB checkpoint.  The manifest records the resolved source.
        output.symlink_to(os.path.relpath(baseline, start=output.parent))
        return
    if alpha == 1.0:
        # The method/config screen often needs several input-only contracts for
        # one immutable trained model.  Keep those controls bit-identical and
        # avoid duplicating the 1.4 GB model for every contract.
        output.symlink_to(os.path.relpath(finetuned, start=output.parent))
        return

    baseline_schema = tensor_schema(baseline)
    finetuned_schema = tensor_schema(finetuned)
    if baseline_schema != finetuned_schema:
        missing = sorted(set(baseline_schema) - set(finetuned_schema))
        extra = sorted(set(finetuned_schema) - set(baseline_schema))
        mismatched = sorted(
            key
            for key in set(baseline_schema) & set(finetuned_schema)
            if baseline_schema[key] != finetuned_schema[key]
        )
        raise RuntimeError(
            f"Checkpoint schemas differ: missing={missing[:8]}, extra={extra[:8]}, "
            f"mismatched={mismatched[:8]}"
        )

    tensors: dict[str, torch.Tensor] = {}
    with (
        safe_open(baseline, framework="pt", device="cpu") as base_handle,
        safe_open(finetuned, framework="pt", device="cpu") as tuned_handle,
    ):
        for key in sorted(baseline_schema):
            base = base_handle.get_tensor(key)
            tuned = tuned_handle.get_tensor(key)
            if base.is_floating_point():
                # Blend in fp32, then restore the checkpoint dtype.
                value = torch.lerp(base.float(), tuned.float(), alpha).to(base.dtype)
            else:
                if not torch.equal(base, tuned):
                    raise RuntimeError(f"Non-floating tensor differs: {key}")
                value = base
            tensors[key] = value.contiguous()
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    save_file(tensors, temporary)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    baseline = args.baseline.expanduser().resolve()
    finetuned = args.finetuned.expanduser().resolve()
    output = args.output.expanduser().absolute()
    require_checkpoint(baseline)
    require_checkpoint(finetuned)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    output.mkdir(parents=True)
    try:
        config = json.loads((finetuned / "config.json").read_text())
        source_contract = {
            "camera_views": "agentview,robot0_eye_in_hand",
            "rgb_camera_views": "agentview",
            "camera_view_weights": None,
            "camera_view_fps_target_points": 10000,
            "camera_view_fps_gripper_points": 500,
            "worldflow_enable": False,
            "se3_enable": False,
        }
        for key, value in source_contract.items():
            if config.get(key) != value:
                raise RuntimeError(f"Finetuned config {key}={config.get(key)!r}, expected {value!r}")
        source_fusion = str(config.get("camera_view_fusion", "legacy_budget"))
        supported_source_fusions = {
            "fps",
            "voxel_fps",
            "voxel_cover_fps",
            "novelty_union",
            "multiscale_novelty_union",
            "transport_novelty_union",
            "uniform_union",
        }
        if source_fusion not in supported_source_fusions:
            raise RuntimeError(
                f"Finetuned camera_view_fusion={source_fusion!r} is not an input-only fusion "
                f"checkpoint in {sorted(supported_source_fusions)}."
            )
        source_contract["camera_view_fusion"] = source_fusion
        config["camera_view_fusion"] = args.camera_view_fusion
        config["camera_view_fps_target_points"] = int(args.fps_target_points)
        config["camera_view_voxel_size"] = float(args.camera_view_voxel_size)
        (output / "config.json").write_text(json.dumps(config, indent=4) + "\n")
        for name in PROCESSOR_FILES:
            shutil.copy2(baseline / name, output / name)
        interpolate_weights(
            baseline / "model.safetensors",
            finetuned / "model.safetensors",
            output / "model.safetensors",
            float(args.alpha),
        )
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": "linear_weight_interpolation_baseline_anchor",
            "alpha_finetuned": float(args.alpha),
            "alpha_baseline": float(1.0 - args.alpha),
            "baseline": str(baseline),
            "finetuned": str(finetuned),
            "config_source": str(finetuned / "config.json"),
            "processor_source": str(baseline),
            "normalization_contract": "IDENTITY",
            "camera_contract_source": source_contract,
            "camera_contract": {
                **source_contract,
                "camera_view_fusion": args.camera_view_fusion,
                "camera_view_fps_target_points": int(args.fps_target_points),
                "camera_view_voxel_size": float(args.camera_view_voxel_size),
            },
            "learned_gate": False,
        }
        (output / "composition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        # Never delete an experiment directory.  Leave any partial output in
        # place as forensic evidence and fail loudly.
        raise

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
