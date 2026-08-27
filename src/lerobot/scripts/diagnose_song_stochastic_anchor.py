#!/usr/bin/env python

"""Measure native-objective RNG variance on one fixed, preprocessed Song batch.

This is a strictly read-only checkpoint diagnostic.  It loads the same wrapped
dataset and preprocessor as :mod:`lerobot.scripts.eval_song`, fetches the fixed
anchor exactly once, and then evaluates that *same tensor batch* under a list of
deterministically derived forward RNG seeds.  The policy stays in eval mode;
only the non-persistent PointSeg flag needed to reconstruct the training
objective is enabled.  No optimizer is created and neither ``backward`` nor
``autograd.grad`` is called.
"""

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.scripts.eval_song import (
    EvalFrameSubset,
    _changed_tensor_fingerprints,
    _fixed_anchor_deterministic_algorithms,
    _fixed_anchor_phase_seed,
    _fixed_anchor_pointseg_aux_loss,
    _fixed_anchor_rng,
    _load_fixed_anchor_manifest,
    _make_eval_dataloader,
    _make_eval_dataset,
    _make_eval_preprocessor,
    _preprocessor_assets_fingerprint,
    _tensor_state_fingerprints,
    _update_batch_fingerprint,
    _validate_fixed_anchor_loss_contract,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class SongStochasticAnchorConfig(TrainPipelineConfig):
    """Arguments unique to the stochastic fixed-anchor diagnostic."""

    fixed_anchor_indices_path: str | None = None
    fixed_anchor_preprocessor_path: str | None = None
    fixed_anchor_seed: int = 20260827
    fixed_anchor_deterministic_algorithms: bool = True
    # Required by the shared eval dataloader.  This diagnostic always operates
    # on complete-dataset global frame indices and therefore keeps it false.
    libero_dataset_domain_action_mse: bool = False
    rng_repeats: int = 128
    rng_seed: int = 20260828
    output: str = "song_stochastic_anchor.json"


def derive_forward_seeds(base_seed: int, repeats: int) -> list[int]:
    """Derive distinct, stable forward seeds using the fixed-anchor seed ABI."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError(f"rng_seed must be a non-negative integer, got {base_seed!r}.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"rng_repeats must be a positive integer, got {repeats!r}.")
    seeds = [
        _fixed_anchor_phase_seed(base_seed, "forward", repeat_index + 1) for repeat_index in range(repeats)
    ]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Derived forward RNG seeds unexpectedly contain duplicates.")
    return seeds


def summarize_scalar_samples(values: list[float]) -> dict[str, float]:
    """Return population statistics for repeated stochastic forward scalars."""

    if not values:
        raise ValueError("At least one scalar sample is required.")
    samples = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("Scalar samples must all be finite.")
    return {
        "mean": float(samples.mean().item()),
        "std": float(samples.std(unbiased=False).item()),
        "median": float(torch.quantile(samples, 0.50).item()),
        "p10": float(torch.quantile(samples, 0.10).item()),
        "p90": float(torch.quantile(samples, 0.90).item()),
        "min": float(samples.min().item()),
        "max": float(samples.max().item()),
    }


def native_objective_terms(
    policy: torch.nn.Module,
    batch: dict[str, Any],
    *,
    pointseg_weight: float,
) -> dict[str, float]:
    """Read the four additive terms from one native policy forward."""

    total, output = policy(batch)
    output = dict(output or {})
    action = output.get("loss_action")
    if torch.is_tensor(action):
        action = action.detach().item()
    if action is None:
        raise RuntimeError("Native forward did not report loss_action.")

    model = getattr(policy, "model", None)
    raw_pointseg = getattr(model, "last_pointseg_aux_loss", None)
    if not torch.is_tensor(raw_pointseg):
        raise RuntimeError("Native forward did not produce the forced PointSeg auxiliary loss.")
    worldflow = model.compute_worldflow_aux_loss() if model is not None else None
    if not isinstance(worldflow, dict) or not torch.is_tensor(worldflow.get("per_sample_loss")):
        raise RuntimeError("Native forward did not produce a WorldFlow per-sample loss.")

    result = {
        "total": float(total.detach().item()),
        "action": float(action),
        "weighted_pointseg": float((float(pointseg_weight) * raw_pointseg).detach().item()),
        "worldflow": float(worldflow["per_sample_loss"].mean().detach().item()),
    }
    reconstructed = result["action"] + result["weighted_pointseg"] + result["worldflow"]
    gap = abs(result["total"] - reconstructed)
    if gap > 1e-7:
        raise RuntimeError(
            "Native objective does not equal action + weighted PointSeg + WorldFlow: "
            f"total={result['total']:.9g}, reconstructed={reconstructed:.9g}, gap={gap:.9g}."
        )
    return result


def _state_digest(fingerprints: dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for name, digest in sorted(fingerprints.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(digest.encode("ascii"))
    return hasher.hexdigest()


def _batch_digest(batch: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    _update_batch_fingerprint(hasher, 1, batch)
    return hasher.hexdigest()


@parser.wrap()
def diagnose(cfg: SongStochasticAnchorConfig) -> dict[str, Any]:
    # TrainPipelineConfig.validate resolves --policy.path into pretrained_path.
    cfg.validate()
    if cfg.fixed_anchor_indices_path is None:
        raise ValueError("fixed_anchor_indices_path is required.")
    if cfg.fixed_anchor_preprocessor_path is None:
        raise ValueError("fixed_anchor_preprocessor_path is required.")
    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("A trained checkpoint is required through --policy.path.")
    if cfg.resume:
        raise ValueError("This read-only diagnostic never resumes optimizer state; use --policy.path.")
    if cfg.dataset.streaming or cfg.dataset.episodes is not None:
        raise ValueError("Fixed global frame indices require a finite complete dataset.")
    if cfg.libero_dataset_domain_action_mse:
        raise ValueError("Dataset-domain subsetting cannot be combined with fixed global indices.")
    forward_seeds = derive_forward_seeds(int(cfg.rng_seed), int(cfg.rng_repeats))
    if cfg.fixed_anchor_deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    init_logging()
    device = torch.device(cfg.policy.device)
    dataset = _make_eval_dataset(cfg, None)
    manifest = _load_fixed_anchor_manifest(
        cfg.fixed_anchor_indices_path,
        dataset_repo_id=str(cfg.dataset.repo_id),
        dataset_length=len(dataset),
        strict=True,
    )
    subset = EvalFrameSubset(dataset, manifest["indices"])

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    training_modules = [name for name, module in policy.named_modules() if module.training]
    if training_modules:
        raise RuntimeError("Policy modules remain in training mode: " + ", ".join(training_modules[:10]))
    loss_contract = _validate_fixed_anchor_loss_contract(policy.config, manifest["loss_contract"])
    pointseg_weight = float(loss_contract["pointseg_aux_loss_weight"])
    if pointseg_weight <= 0.0:
        raise ValueError("Stochastic objective diagnosis requires positive pointseg_aux_loss_weight.")
    if not bool(getattr(policy.config, "worldflow_enable", False)):
        raise ValueError("Stochastic objective diagnosis requires worldflow_enable=true.")
    if bool(getattr(policy.config, "adapt_to_pi_aloha", False)):
        raise ValueError("This diagnostic requires a native forward that does not mutate its batch.")

    preprocessor_assets = _preprocessor_assets_fingerprint(cfg.fixed_anchor_preprocessor_path)
    preprocessor = _make_eval_preprocessor(
        cfg,
        policy,
        dataset,
        device,
        pretrained_path=cfg.fixed_anchor_preprocessor_path,
    )

    # Fetch and preprocess exactly once.  The whole manifest is one batch, so
    # every forward below consumes identical tensor objects and values.
    original_batch_size = cfg.batch_size
    cfg.batch_size = manifest["count"]
    try:
        dataloader = _make_eval_dataloader(cfg, subset, device, fixed_anchor=True)
    finally:
        cfg.batch_size = original_batch_size
    iterator_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "dataloader")
    with _fixed_anchor_rng(iterator_seed):
        iterator = iter(dataloader)
    fetch_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "fetch", 1)
    with _fixed_anchor_rng(fetch_seed):
        raw_batch = next(iterator)
    raw_indices = raw_batch.get("index")
    if not torch.is_tensor(raw_indices):
        raise KeyError("Fixed-anchor raw batch is missing tensor key 'index'.")
    observed_indices = [int(index) for index in raw_indices.reshape(-1).tolist()]
    if observed_indices != manifest["indices"]:
        raise RuntimeError(
            f"Fixed-anchor order drifted: expected {manifest['indices']}, observed {observed_indices}."
        )
    raw_input_sha256 = _batch_digest(raw_batch)
    preprocess_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "preprocess", 1)
    with _fixed_anchor_rng(preprocess_seed):
        batch = preprocessor(raw_batch)
    if not torch.is_tensor(batch.get(ACTION)):
        raise KeyError(f"Preprocessed fixed-anchor batch is missing tensor key {ACTION!r}.")
    preprocessed_input_sha256 = _batch_digest(batch)

    parameter_hashes_before = _tensor_state_fingerprints(policy, buffers=False)
    buffer_hashes_before = _tensor_state_fingerprints(policy, buffers=True)
    parameter_versions_before = {name: parameter._version for name, parameter in policy.named_parameters()}
    preexisting_grads = [name for name, parameter in policy.named_parameters() if parameter.grad is not None]
    if preexisting_grads:
        raise RuntimeError(
            f"Fresh policy unexpectedly has populated .grad tensors: {preexisting_grads[:10]}."
        )

    values: dict[str, list[float]] = {
        name: [] for name in ("total", "action", "weighted_pointseg", "worldflow")
    }
    logging.info(
        "Stochastic fixed anchor: samples=%s repeats=%s base_rng_seed=%s",
        manifest["count"],
        cfg.rng_repeats,
        cfg.rng_seed,
    )
    with _fixed_anchor_deterministic_algorithms(bool(cfg.fixed_anchor_deterministic_algorithms)):
        for forward_seed in forward_seeds:
            with (
                _fixed_anchor_rng(forward_seed),
                _fixed_anchor_pointseg_aux_loss(policy, True),
                torch.inference_mode(),
            ):
                terms = native_objective_terms(
                    policy,
                    batch,
                    pointseg_weight=pointseg_weight,
                )
            for name, value in terms.items():
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {name} at forward seed {forward_seed}: {value}.")
                values[name].append(value)

    parameter_hashes_after = _tensor_state_fingerprints(policy, buffers=False)
    buffer_hashes_after = _tensor_state_fingerprints(policy, buffers=True)
    changed_parameters = _changed_tensor_fingerprints(parameter_hashes_before, parameter_hashes_after)
    changed_buffers = _changed_tensor_fingerprints(buffer_hashes_before, buffer_hashes_after)
    changed_versions = [
        name
        for name, parameter in policy.named_parameters()
        if parameter._version != parameter_versions_before[name]
    ]
    populated_grads = [name for name, parameter in policy.named_parameters() if parameter.grad is not None]
    batch_sha256_after = _batch_digest(batch)
    if changed_parameters or changed_buffers or changed_versions or populated_grads:
        raise RuntimeError(
            "Read-only verification failed: "
            f"changed_parameters={changed_parameters[:10]}, changed_buffers={changed_buffers[:10]}, "
            f"changed_versions={changed_versions[:10]}, populated_grads={populated_grads[:10]}."
        )
    if batch_sha256_after != preprocessed_input_sha256:
        raise RuntimeError("Native forward mutated the preprocessed fixed-anchor batch.")

    report = {
        "schema_version": 1,
        "diagnostic": "song_stochastic_fixed_anchor_native_objective",
        "checkpoint": str(cfg.policy.pretrained_path),
        "dataset": str(cfg.dataset.repo_id),
        "pointseg_sample_cache_dir": str(cfg.pointseg_sample_cache_dir),
        "manifest": manifest,
        "preprocessor": preprocessor_assets,
        "fixed_anchor_seed": int(cfg.fixed_anchor_seed),
        "deterministic_algorithms": bool(cfg.fixed_anchor_deterministic_algorithms),
        "raw_input_sha256": raw_input_sha256,
        "preprocessed_input_sha256": preprocessed_input_sha256,
        "rng": {
            "base_seed": int(cfg.rng_seed),
            "repeats": int(cfg.rng_repeats),
            "derivation": "_fixed_anchor_phase_seed(base_seed, 'forward', repeat_index + 1)",
            "forward_seeds": forward_seeds,
        },
        "loss_contract": loss_contract,
        "statistics": {name: summarize_scalar_samples(samples) for name, samples in values.items()},
        "samples_by_repeat": [
            {"repeat": repeat_index, "forward_seed": forward_seed}
            | {name: values[name][repeat_index] for name in values}
            for repeat_index, forward_seed in enumerate(forward_seeds)
        ],
        "read_only_verification": {
            "parameters_sha256_before": _state_digest(parameter_hashes_before),
            "parameters_sha256_after": _state_digest(parameter_hashes_after),
            "buffers_sha256_before": _state_digest(buffer_hashes_before),
            "buffers_sha256_after": _state_digest(buffer_hashes_after),
            "preprocessed_batch_sha256_before": preprocessed_input_sha256,
            "preprocessed_batch_sha256_after": batch_sha256_after,
            "parameters_unchanged": True,
            "buffers_unchanged": True,
            "parameter_versions_unchanged": True,
            "parameter_grads_unpopulated": True,
            "preprocessed_batch_unchanged": True,
            "policy_eval": True,
            "optimizer_created": False,
            "backward_called": False,
        },
    }
    output = Path(cfg.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    logging.info("Saved stochastic fixed-anchor report to %s", output)
    return report


def main() -> None:
    register_third_party_plugins()
    diagnose()


if __name__ == "__main__":
    main()
