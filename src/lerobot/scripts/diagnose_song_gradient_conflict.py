#!/usr/bin/env python

"""Read-only Task 6/Task 8 gradient-conflict diagnosis for Song checkpoints.

The command intentionally reuses ``eval_song`` fixed-anchor manifests, dataset
wrappers, collate/preprocessing, phase-separated RNG, and eval-mode PointSeg
auxiliary loss.  It creates no optimizer, never calls backward/step, and checks
parameter and buffer hashes after the diagnosis.

For every microbatch it repeats the native forward with an identical RNG seed:

* A: action loss only;
* AP: action + configured-weight PointSeg loss;
* APW: native total, including the configured WorldFlow loss.

``torch.autograd.grad`` yields the three objective gradients.  The reported
components are ``g_action = g_A``, ``g_point = g_AP - g_A``, and
``g_world = g_APW - g_AP``.
"""

import fnmatch
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

DEFAULT_LAYER_SPECS = ",".join(
    (
        "scene_projection=model.pointseg_object_proj.*",
        "action_input=model.action_in_proj.*",
        "action_output=model.action_out_proj.*",
    )
)
OBJECTIVES = ("A", "AP", "APW")
COMPONENTS = ("action", "point", "world")
GRADIENT_KINDS = OBJECTIVES + COMPONENTS


@dataclass
class SongGradientConflictConfig(TrainPipelineConfig):
    """Inputs specific to the read-only gradient diagnostic."""

    task6_fixed_manifest_path: str | None = None
    task8_fixed_manifest_path: str | None = None
    fixed_anchor_preprocessor_path: str | None = None
    fixed_anchor_seed: int = 20260827
    fixed_anchor_deterministic_algorithms: bool = True
    libero_dataset_domain_action_mse: bool = False
    k: int = 32
    microbatch: int = 1
    layers: str = DEFAULT_LAYER_SPECS
    output: str = "song_gradient_conflict.json"


def parse_layer_specs(value: str) -> dict[str, str]:
    """Parse ``label=parameter_glob`` entries from a comma-separated string."""

    result: dict[str, str] = {}
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid layer specification {entry!r}; expected label=parameter_glob.")
        label, pattern = (part.strip() for part in entry.split("=", 1))
        if not label or not pattern:
            raise ValueError(f"Invalid layer specification {entry!r}; label and glob must be non-empty.")
        if label in result:
            raise ValueError(f"Duplicate layer label {label!r}.")
        result[label] = pattern
    if not result:
        raise ValueError("At least one layer specification is required.")
    return result


def summarize_gradient_moments(
    weighted_sum: torch.Tensor,
    weighted_squared_norm: float,
    total_weight: float,
) -> dict[str, float | None]:
    """Return mean norm, RMS microbatch noise, and noise-to-signal ratio."""

    if total_weight <= 0:
        raise ValueError("total_weight must be positive.")
    mean = weighted_sum / float(total_weight)
    mean_squared_norm = float(torch.dot(mean.double(), mean.double()).item())
    variance = max(float(weighted_squared_norm) / float(total_weight) - mean_squared_norm, 0.0)
    mean_norm = math.sqrt(mean_squared_norm)
    noise_rms = math.sqrt(variance)
    noise_to_signal = noise_rms / mean_norm if mean_norm > 0.0 else None
    return {
        "mean_gradient_norm": mean_norm,
        "microbatch_noise_rms": noise_rms,
        "noise_to_signal": noise_to_signal,
    }


def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float | None:
    """Cosine with an explicit null result for a zero gradient."""

    first64 = first.double()
    second64 = second.double()
    denominator = float(first64.norm().item() * second64.norm().item())
    if denominator == 0.0:
        return None
    cosine = float(torch.dot(first64, second64).item()) / denominator
    return min(1.0, max(-1.0, cosine))


def decompose_objective_gradients(
    action: torch.Tensor,
    action_point: torch.Tensor,
    action_point_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose nested-objective gradients without changing their scale."""

    return action, action_point - action, action_point_world - action_point


class _GradientMoments:
    def __init__(self, size: int) -> None:
        self.weighted_sum = torch.zeros(size, dtype=torch.float64)
        self.weighted_squared_norm = 0.0
        self.total_weight = 0.0
        self.microbatch_count = 0

    def add(self, gradient: torch.Tensor, weight: int) -> None:
        gradient64 = gradient.detach().to(device="cpu", dtype=torch.float64)
        self.weighted_sum.add_(gradient64, alpha=float(weight))
        self.weighted_squared_norm += float(torch.dot(gradient64, gradient64).item()) * float(weight)
        self.total_weight += float(weight)
        self.microbatch_count += 1

    @property
    def mean(self) -> torch.Tensor:
        if self.total_weight <= 0:
            raise RuntimeError("Cannot read an empty gradient accumulator.")
        return self.weighted_sum / self.total_weight

    def summary(self) -> dict[str, float | int | None]:
        return {
            **summarize_gradient_moments(
                self.weighted_sum,
                self.weighted_squared_norm,
                self.total_weight,
            ),
            "samples": int(self.total_weight),
            "microbatches": self.microbatch_count,
        }


def _resolve_layer_parameters(
    policy: torch.nn.Module,
    specs: dict[str, str],
) -> tuple[list[str], list[torch.nn.Parameter], dict[str, list[int]], dict[str, Any]]:
    trainable = [
        (name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]
    selected_names: list[str] = []
    selected_parameters: list[torch.nn.Parameter] = []
    name_to_index: dict[str, int] = {}
    layer_indices: dict[str, list[int]] = {}
    layer_report: dict[str, Any] = {}
    all_names = [name for name, _parameter in policy.named_parameters()]

    for label, pattern in specs.items():
        matched_all = [name for name in all_names if fnmatch.fnmatchcase(name, pattern)]
        matched = [(name, parameter) for name, parameter in trainable if fnmatch.fnmatchcase(name, pattern)]
        if not matched:
            suffix = " (matches only frozen parameters)" if matched_all else ""
            raise ValueError(f"Layer {label!r} glob {pattern!r} matched no trainable parameters{suffix}.")
        indices = []
        for name, parameter in matched:
            index = name_to_index.get(name)
            if index is None:
                index = len(selected_parameters)
                name_to_index[name] = index
                selected_names.append(name)
                selected_parameters.append(parameter)
            indices.append(index)
        layer_indices[label] = indices
        layer_report[label] = {
            "glob": pattern,
            "parameters": [name for name, _parameter in matched],
            "numel": sum(parameter.numel() for _name, parameter in matched),
        }

    layer_indices["all_selected"] = list(range(len(selected_parameters)))
    layer_report["all_selected"] = {
        "glob": None,
        "parameters": selected_names,
        "numel": sum(parameter.numel() for parameter in selected_parameters),
    }
    return selected_names, selected_parameters, layer_indices, layer_report


def _flatten_layer_gradient(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
    indices: list[int],
) -> torch.Tensor:
    chunks = []
    for index in indices:
        gradient = gradients[index]
        if gradient is None:
            chunks.append(torch.zeros(parameters[index].numel(), dtype=torch.float64))
        else:
            chunks.append(gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    return torch.cat(chunks)


def _native_loss_terms(
    policy: torch.nn.Module,
    batch: dict[str, Any],
    pointseg_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    total, output = policy(batch)
    model = getattr(policy, "model", None)
    pointseg = getattr(model, "last_pointseg_aux_loss", None)
    if not torch.is_tensor(pointseg):
        raise RuntimeError("The native forward did not produce a differentiable PointSeg auxiliary loss.")
    worldflow = model.compute_worldflow_aux_loss() if model is not None else None
    if not isinstance(worldflow, dict) or not torch.is_tensor(worldflow.get("per_sample_loss")):
        raise RuntimeError("The native forward did not produce a differentiable WorldFlow loss.")
    weighted_pointseg = float(pointseg_weight) * pointseg
    weighted_world = worldflow["per_sample_loss"].mean()
    action = total - weighted_pointseg - weighted_world
    values = {
        "action": float(action.detach().item()),
        "weighted_point": float(weighted_pointseg.detach().item()),
        "weighted_world": float(weighted_world.detach().item()),
        "native_total": float(total.detach().item()),
        "reported_action": float((output or {}).get("loss_action", math.nan)),
    }
    return action, weighted_pointseg, weighted_world, values


def _component_gradients(
    policy: torch.nn.Module,
    batch: dict[str, Any],
    parameters: list[torch.nn.Parameter],
    *,
    seed: int,
    pointseg_weight: float,
) -> tuple[dict[str, tuple[torch.Tensor | None, ...]], dict[str, float]]:
    """Measure all loss gradients on one prediction/graph.

    Repeating the full Molmo/point-cloud forward, even under the same RNG state,
    can differ by a few ulps because the CUDA attention path is not bitwise
    replayable.  Such differences are large relative to the tiny late-training
    gradients under investigation.  A single forward followed by three
    ``autograd.grad`` calls gives an exact, common prediction for every loss.
    """

    with _fixed_anchor_rng(seed), _fixed_anchor_pointseg_aux_loss(policy, True):
        action, point, world, values = _native_loss_terms(policy, batch, pointseg_weight)
        reported_action = values["reported_action"]
        if not math.isfinite(reported_action) or not math.isclose(
            values["action"], reported_action, rel_tol=1e-5, abs_tol=1e-8
        ):
            raise RuntimeError(
                "Native loss decomposition disagrees with output['loss_action']: "
                f"decomposed={values['action']} reported={reported_action}."
            )
        components = (action, point, world)
        gradients = {
            name: torch.autograd.grad(
                scalar,
                parameters,
                retain_graph=index < len(components) - 1,
                allow_unused=True,
                materialize_grads=False,
            )
            for index, (name, scalar) in enumerate(zip(COMPONENTS, components, strict=True))
        }
    return gradients, values


def _state_digest(fingerprints: dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for name, digest in sorted(fingerprints.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(digest.encode("ascii"))
    return hasher.hexdigest()


def _run_task(
    *,
    task_name: str,
    cfg: SongGradientConflictConfig,
    dataset: torch.utils.data.Dataset,
    manifest: dict[str, Any],
    policy: torch.nn.Module,
    preprocessor: Any,
    parameters: list[torch.nn.Parameter],
    layer_indices: dict[str, list[int]],
    pointseg_weight: float,
) -> tuple[dict[str, dict[str, _GradientMoments]], dict[str, Any]]:
    indices = manifest["indices"][: int(cfg.k)]
    subset = EvalFrameSubset(dataset, indices)
    original_batch_size = cfg.batch_size
    cfg.batch_size = int(cfg.microbatch)
    try:
        dataloader = _make_eval_dataloader(cfg, subset, torch.device(cfg.policy.device), fixed_anchor=True)
    finally:
        cfg.batch_size = original_batch_size

    dimensions = {
        label: sum(parameters[index].numel() for index in selected)
        for label, selected in layer_indices.items()
    }
    moments = {
        label: {kind: _GradientMoments(dimensions[label]) for kind in GRADIENT_KINDS}
        for label in layer_indices
    }
    loss_sums = dict.fromkeys(("action", "weighted_point", "weighted_world", "native_total"), 0.0)
    observed_indices: list[int] = []
    raw_hasher = hashlib.sha256()
    processed_hasher = hashlib.sha256()

    iterator_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "dataloader")
    with _fixed_anchor_rng(iterator_seed):
        iterator = iter(dataloader)
    batch_count = math.ceil(len(indices) / int(cfg.microbatch))
    for batch_number in range(1, batch_count + 1):
        fetch_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "fetch", batch_number)
        with _fixed_anchor_rng(fetch_seed):
            raw_batch = next(iterator)
        raw_indices = raw_batch.get("index")
        if not torch.is_tensor(raw_indices):
            raise KeyError(f"{task_name} fixed-anchor batch is missing tensor key 'index'.")
        observed_indices.extend(int(index) for index in raw_indices.reshape(-1).tolist())
        _update_batch_fingerprint(raw_hasher, batch_number, raw_batch)

        preprocess_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "preprocess", batch_number)
        with _fixed_anchor_rng(preprocess_seed):
            batch = preprocessor(raw_batch)
        _update_batch_fingerprint(processed_hasher, batch_number, batch)
        batch_size = int(batch[ACTION].shape[0])
        forward_seed = _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "forward", batch_number)

        component_gradients, reference = _component_gradients(
            policy,
            batch,
            parameters,
            seed=forward_seed,
            pointseg_weight=pointseg_weight,
        )
        for term in loss_sums:
            loss_sums[term] += reference[term] * batch_size

        objective_gradients: dict[str, tuple[torch.Tensor, ...]] = {
            objective: tuple(
                sum(
                    (
                        component_gradients[kind][parameter_index]
                        if component_gradients[kind][parameter_index] is not None
                        else torch.zeros_like(parameter)
                    )
                    for kind in COMPONENTS[: objective_index + 1]
                )
                for parameter_index, parameter in enumerate(parameters)
            )
            for objective_index, objective in enumerate(OBJECTIVES)
        }
        all_gradients = {**objective_gradients, **component_gradients}
        for label, selected in layer_indices.items():
            for kind, gradients in all_gradients.items():
                moments[label][kind].add(
                    _flatten_layer_gradient(gradients, parameters, selected),
                    batch_size,
                )

        del all_gradients, component_gradients, objective_gradients

    if observed_indices != indices:
        raise RuntimeError(
            f"{task_name} anchor order drifted: expected {indices}, observed {observed_indices}."
        )
    task_report = {
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "indices": indices,
        "samples": len(indices),
        "microbatch": int(cfg.microbatch),
        "raw_input_sha256": raw_hasher.hexdigest(),
        "preprocessed_input_sha256": processed_hasher.hexdigest(),
        "mean_losses": {name: value / len(indices) for name, value in loss_sums.items()},
        "layers": {
            label: {kind: accumulator.summary() for kind, accumulator in kinds.items()}
            for label, kinds in moments.items()
        },
    }
    return moments, task_report


def _comparison_report(
    task6: dict[str, dict[str, _GradientMoments]],
    task8: dict[str, dict[str, _GradientMoments]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for label in task6:
        cross_task = {}
        for kind in GRADIENT_KINDS:
            first = task6[label][kind]
            second = task8[label][kind]
            cross_task[kind] = {
                "cosine": cosine_similarity(first.mean, second.mean),
                "task6_norm": first.summary()["mean_gradient_norm"],
                "task8_norm": second.summary()["mean_gradient_norm"],
            }
        report[label] = {
            "cross_task": cross_task,
            "action_vs_world_cosine": {
                "task6": cosine_similarity(task6[label]["action"].mean, task6[label]["world"].mean),
                "task8": cosine_similarity(task8[label]["action"].mean, task8[label]["world"].mean),
            },
        }
    return report


@parser.wrap()
def diagnose(cfg: SongGradientConflictConfig) -> dict[str, Any]:
    # Resolve ``--policy.path`` into ``cfg.policy.pretrained_path`` before
    # validating diagnostic-only inputs.  This mirrors eval_song.py; checking
    # pretrained_path before TrainPipelineConfig.validate() rejects a valid
    # checkpoint CLI argument because the parser has not loaded its config yet.
    cfg.validate()
    if cfg.task6_fixed_manifest_path is None or cfg.task8_fixed_manifest_path is None:
        raise ValueError("Both task6_fixed_manifest_path and task8_fixed_manifest_path are required.")
    if cfg.fixed_anchor_preprocessor_path is None:
        raise ValueError("fixed_anchor_preprocessor_path is required to pin normalization/tokenization.")
    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("A trained checkpoint is required through --policy.path.")
    if cfg.resume:
        raise ValueError("This diagnostic never resumes optimizer state; use --policy.path.")
    if int(cfg.k) < 1 or int(cfg.microbatch) < 1:
        raise ValueError(f"k and microbatch must be positive, got k={cfg.k}, microbatch={cfg.microbatch}.")
    if cfg.dataset.streaming or cfg.dataset.episodes is not None:
        raise ValueError("Fixed manifests require a finite complete dataset with dataset.episodes unset.")
    if cfg.fixed_anchor_deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    init_logging()

    specs = parse_layer_specs(cfg.layers)
    dataset = _make_eval_dataset(cfg, None)
    manifests = {
        "task6": _load_fixed_anchor_manifest(
            cfg.task6_fixed_manifest_path,
            dataset_repo_id=str(cfg.dataset.repo_id),
            dataset_length=len(dataset),
            strict=True,
        ),
        "task8": _load_fixed_anchor_manifest(
            cfg.task8_fixed_manifest_path,
            dataset_repo_id=str(cfg.dataset.repo_id),
            dataset_length=len(dataset),
            strict=True,
        ),
    }
    for task_name, manifest in manifests.items():
        if int(cfg.k) > manifest["count"]:
            raise ValueError(f"k={cfg.k} exceeds {task_name} manifest count {manifest['count']}.")
    if manifests["task6"]["loss_contract"] != manifests["task8"]["loss_contract"]:
        raise ValueError("Task 6 and Task 8 manifests declare different loss contracts.")

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    training_modules = [name for name, module in policy.named_modules() if module.training]
    if training_modules:
        raise RuntimeError("Policy modules remain in training mode: " + ", ".join(training_modules[:10]))
    loss_contract = _validate_fixed_anchor_loss_contract(policy.config, manifests["task6"]["loss_contract"])
    if not bool(getattr(policy.config, "worldflow_enable", False)):
        raise ValueError("Gradient decomposition requires a checkpoint with worldflow_enable=true.")
    if bool(getattr(policy.config, "adapt_to_pi_aloha", False)):
        raise ValueError("This diagnostic requires a forward path that does not mutate the input batch.")
    if float(loss_contract["pointseg_aux_loss_weight"]) <= 0.0:
        raise ValueError("Gradient decomposition requires a positive PointSeg auxiliary loss weight.")

    preprocessor_assets = _preprocessor_assets_fingerprint(cfg.fixed_anchor_preprocessor_path)
    preprocessor = _make_eval_preprocessor(
        cfg,
        policy,
        dataset,
        torch.device(cfg.policy.device),
        pretrained_path=cfg.fixed_anchor_preprocessor_path,
    )
    selected_names, parameters, layer_indices, layer_report = _resolve_layer_parameters(policy, specs)
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("Freshly loaded policy unexpectedly has populated .grad tensors.")

    logging.info(
        "Gradient diagnostic: K=%s/task microbatch=%s selected_parameters=%s selected_numel=%s",
        cfg.k,
        cfg.microbatch,
        len(parameters),
        sum(parameter.numel() for parameter in parameters),
    )
    parameter_hashes_before = _tensor_state_fingerprints(policy, buffers=False)
    buffer_hashes_before = _tensor_state_fingerprints(policy, buffers=True)
    parameter_versions_before = {name: parameter._version for name, parameter in policy.named_parameters()}

    task_moments: dict[str, dict[str, dict[str, _GradientMoments]]] = {}
    task_reports: dict[str, Any] = {}
    with _fixed_anchor_deterministic_algorithms(bool(cfg.fixed_anchor_deterministic_algorithms)):
        for task_name in ("task6", "task8"):
            logging.info("Diagnosing %s", task_name)
            task_moments[task_name], task_reports[task_name] = _run_task(
                task_name=task_name,
                cfg=cfg,
                dataset=dataset,
                manifest=manifests[task_name],
                policy=policy,
                preprocessor=preprocessor,
                parameters=parameters,
                layer_indices=layer_indices,
                pointseg_weight=loss_contract["pointseg_aux_loss_weight"],
            )

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
    if changed_parameters or changed_buffers or changed_versions or populated_grads:
        raise RuntimeError(
            "Read-only verification failed: "
            f"changed_parameters={changed_parameters[:10]}, changed_buffers={changed_buffers[:10]}, "
            f"changed_versions={changed_versions[:10]}, populated_grads={populated_grads[:10]}."
        )

    report = {
        "schema_version": 1,
        "diagnostic": "song_task6_task8_gradient_conflict",
        "checkpoint": str(cfg.policy.pretrained_path),
        "dataset": str(cfg.dataset.repo_id),
        "pointseg_sample_cache_dir": str(cfg.pointseg_sample_cache_dir),
        "preprocessor": preprocessor_assets,
        "fixed_anchor_seed": int(cfg.fixed_anchor_seed),
        "deterministic_algorithms": bool(cfg.fixed_anchor_deterministic_algorithms),
        "k_per_task": int(cfg.k),
        "microbatch": int(cfg.microbatch),
        "loss_contract": loss_contract,
        "gradient_decomposition": {
            "A": "action",
            "AP": "action + configured_weight * pointseg",
            "APW": "action + configured_weight * pointseg + configured WorldFlow total",
            "measurement": "one native forward/graph; direct autograd.grad per component",
            "g_A": "g_action",
            "g_AP": "g_action + g_point",
            "g_APW": "g_action + g_point + g_world",
        },
        "selected_parameter_names": selected_names,
        "layers": layer_report,
        "tasks": task_reports,
        "comparisons": _comparison_report(task_moments["task6"], task_moments["task8"]),
        "read_only_verification": {
            "parameters_sha256_before": _state_digest(parameter_hashes_before),
            "parameters_sha256_after": _state_digest(parameter_hashes_after),
            "buffers_sha256_before": _state_digest(buffer_hashes_before),
            "buffers_sha256_after": _state_digest(buffer_hashes_after),
            "parameters_unchanged": True,
            "buffers_unchanged": True,
            "parameter_versions_unchanged": True,
            "parameter_grads_unpopulated": True,
            "optimizer_created": False,
            "backward_called": False,
        },
    }
    output = Path(cfg.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    logging.info("Saved gradient-conflict report to %s", output)
    return report


def main() -> None:
    register_third_party_plugins()
    diagnose()


if __name__ == "__main__":
    main()
