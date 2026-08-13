#!/usr/bin/env python3
"""Rebase a jointly trained WorldFlow checkpoint onto an immutable Ego anchor.

Every tensor that exists with the same name and shape in the baseline is copied
from the baseline. Tensors introduced by WorldFlow and bidirectional fusion are
kept from the jointly trained checkpoint. This is a task-agnostic post-training
parameter merge: it does not add a gate, alter the model topology, or inspect
rollout/task outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


COPY_FILES = (
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_6_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--worldflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_checkpoint(path: Path) -> None:
    for name in ("config.json", "model.safetensors"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)


def main() -> None:
    args = parse_args()
    baseline = args.baseline.expanduser().resolve()
    worldflow = args.worldflow.expanduser().resolve()
    output = args.output.expanduser().absolute()
    require_checkpoint(baseline)
    require_checkpoint(worldflow)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    config = json.loads((worldflow / "config.json").read_text())
    if not config.get("worldflow_enable", False):
        raise RuntimeError("The source checkpoint must have worldflow_enable=True.")
    if config.get("worldflow_ego_residual_gate_init") is not None:
        raise RuntimeError("WorldFlow gates are unsupported.")

    output.mkdir(parents=True)
    baseline_weights = baseline / "model.safetensors"
    worldflow_weights = worldflow / "model.safetensors"
    output_weights = output / "model.safetensors"
    temporary = output / f".model.safetensors.tmp.{os.getpid()}"
    restored = []
    retained = []
    tensors = {}
    with (
        safe_open(baseline_weights, framework="pt", device="cpu") as base_handle,
        safe_open(worldflow_weights, framework="pt", device="cpu") as world_handle,
    ):
        base_keys = set(base_handle.keys())
        world_keys = set(world_handle.keys())
        missing_shared = sorted(base_keys - world_keys)
        if missing_shared:
            raise RuntimeError(f"WorldFlow checkpoint dropped baseline tensors: {missing_shared[:8]}")
        for key in sorted(world_keys):
            world_tensor = world_handle.get_tensor(key)
            if key in base_keys:
                base_tensor = base_handle.get_tensor(key)
                if base_tensor.shape != world_tensor.shape or base_tensor.dtype != world_tensor.dtype:
                    raise RuntimeError(
                        f"Shared tensor schema mismatch for {key}: "
                        f"baseline={base_tensor.shape}/{base_tensor.dtype}, "
                        f"worldflow={world_tensor.shape}/{world_tensor.dtype}"
                    )
                tensors[key] = base_tensor.contiguous()
                restored.append(key)
            else:
                tensors[key] = world_tensor.contiguous()
                retained.append(key)
    save_file(tensors, temporary)
    os.replace(temporary, output_weights)

    for name in COPY_FILES:
        source = worldflow / name
        if source.is_file():
            shutil.copy2(source, output / name)
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "worldflow_new_parameters_on_exact_baseline_anchor",
        "task_specific_training_or_inference": False,
        "learned_gate": False,
        "baseline": str(baseline),
        "jointly_trained_worldflow": str(worldflow),
        "restored_baseline_tensor_count": len(restored),
        "retained_worldflow_tensor_count": len(retained),
        "retained_worldflow_tensors": retained,
        "contract": (
            "all same-name same-schema tensors come exactly from baseline; "
            "only WorldFlow/bidirectional tensors absent from baseline are retained"
        ),
    }
    (output / "rebase_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
