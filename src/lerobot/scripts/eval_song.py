#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate Song/SmolVLA checkpoints without changing model parameters.

This entry point deliberately reuses the dataset, PointSeg cache, collate, policy,
and preprocessing path from ``train_song.py``.  Unlike the training entry point it:

* keeps the policy in evaluation mode;
* runs every forward pass under ``torch.inference_mode()``;
* never creates an optimizer or learning-rate scheduler;
* never calls backward or saves a model checkpoint;
* traverses the evaluation data at most once.

For CLI compatibility, ``--steps`` is interpreted as the maximum number of
evaluation batches.  If it exceeds the DataLoader length, only one complete pass
over the dataset is evaluated.
"""

import hashlib
import json
import logging
import math
import os
import random
import sys
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.train_song import (
    OnlinePointSegBatchCollator,
    ensure_ddp_parameters_initialized,
    make_song_train_collate_fn,
    maybe_wrap_point_cloud_memmap_dataset,
    maybe_wrap_pointseg_cache_dataset,
    maybe_wrap_worldflow_dataset,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, inside_slurm

EVAL_METRIC_KEYS = (
    "loss",
    "loss_action",
    "loss_action_translation",
    "loss_action_rotation6d",
    "loss_action_gripper",
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
    "loss_pointseg_aux",
    "loss_se3_twist",
    "loss_se3_endpoint",
    "loss_se3_gripper",
    "loss_se3_equivariance",
    "se3_action_trans_err",
    "se3_action_rot_err_deg",
    "loss_worldflow_flow",
    "loss_worldflow_geo",
    "loss_worldflow_bridge",
    "loss_worldflow_equiv",
    "worldflow_trans_err",
    "worldflow_rot_err_deg",
    "worldflow_valid_ratio",
    "worldflow_foreground_points",
    "worldflow_noise_conjugacy_error",
    "worldflow_path_conjugacy_error",
    "pointseg_foreground_ratio",
    "pointseg_operation_prob_mean",
    "pointseg_selection_score_mean",
    "pred_operation_prob",
    "loss_soft_bce",
    "loss_smoothness",
    "pseudo_valid_ratio",
    "pseudo_foreground_ratio",
    "pseudo_soft_foreground_mean",
    "pred_foreground_ratio",
    "sample_action_mse",
    "sample_action_translation_l2_m",
    "sample_action_rot6d_mse",
    "sample_action_gripper_mae_m",
)

FLOW_TIME_SWEEP_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999)
FLOW_TIME_SWEEP_METRICS = (
    "loss_action",
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
)
EVAL_METRIC_KEYS = EVAL_METRIC_KEYS + tuple(
    f"flow_t{round(time_value * 1000):03d}_{metric_name}"
    for time_value in FLOW_TIME_SWEEP_VALUES
    for metric_name in FLOW_TIME_SWEEP_METRICS
)

# Keep the legacy metric tuple untouched unless fixed-anchor mode is enabled.
# Besides preserving ordinary eval_song output, this prevents an opt-in
# comparison tool from subtly changing distributed gather payloads elsewhere.
FIXED_ANCHOR_EXTRA_METRIC_KEYS = (
    "loss_anchor_total",
    "loss_anchor_total_gap",
    "worldflow_flow_translation_err_m",
    "worldflow_flow_rotation6d_rmse",
    "worldflow_endpoint_translation_err_m",
    "worldflow_endpoint_rotation_probe_err_m",
    "loss_worldflow_gripper",
    "worldflow_endpoint_gripper_err",
)


@dataclass
class SongEvalPipelineConfig(TrainPipelineConfig):
    """Read-only evaluation options that do not belong to the training config."""

    libero_dataset_domain_action_mse: bool = False
    libero_suite: str | None = None
    libero_task_id: int | None = None
    sample_action_mse: bool = False
    sample_action_noise_mode: str = "policy"
    flow_time_sweep: bool = False
    # Strict, checkpoint-independent loss comparison.  The JSON manifest fixes
    # exact *global* dataset frame indices and is applied only after the same
    # PointSeg/point-cloud/WorldFlow wrappers used by training have been built.
    # This mode deliberately remains opt-in so ordinary eval_song behavior is
    # backward compatible.
    fixed_anchor_indices_path: str | None = None
    fixed_anchor_seed: int = 20260827
    # Supplying one common processor directory for every checkpoint guarantees
    # that normalization/tokenization does not drift between comparisons.
    fixed_anchor_preprocessor_path: str | None = None
    fixed_anchor_strict: bool = True
    fixed_anchor_hash_inputs: bool = True
    # Optional second assertion in addition to the manifest's declared loss
    # contract.  For the current experiment pass 0.0005 explicitly.
    fixed_anchor_expected_pointseg_weight: float | None = None
    # Custom CUDA extensions may not advertise determinism to PyTorch.  Keep
    # strict RNG/index/processor checks while allowing this one kernel gate to
    # be explicitly disabled and recorded in the comparison contract.
    fixed_anchor_deterministic_algorithms: bool = True


FIXED_ANCHOR_SCHEMA_VERSION = 1
FIXED_ANCHOR_REQUIRED_METRICS = (
    "loss_action",
    "loss_worldflow_flow",
    "loss_pointseg_aux",
)
_FIXED_ANCHOR_PHASE_IDS = {
    "dataloader": 11,
    "fetch": 23,
    "preprocess": 37,
    "forward": 53,
    "sample": 71,
    "sweep": 89,
}


class EvalFrameSubset(torch.utils.data.Dataset):
    """Frame subset that preserves access to LeRobot metadata and wrapper attributes."""

    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


def _fixed_anchor_phase_seed(
    base_seed: int,
    phase: str,
    batch_index: int = 0,
    process_index: int = 0,
) -> int:
    """Derive a stable seed without depending on Python's randomized hash()."""

    if phase not in _FIXED_ANCHOR_PHASE_IDS:
        raise ValueError(f"Unknown fixed-anchor RNG phase {phase!r}.")
    payload = (
        f"song-fixed-anchor-v{FIXED_ANCHOR_SCHEMA_VERSION}:"
        f"{int(base_seed)}:{_FIXED_ANCHOR_PHASE_IDS[phase]}:"
        f"{int(batch_index)}:{int(process_index)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


@contextmanager
def _fixed_anchor_rng(seed: int) -> Generator[None, None, None]:
    """Seed and then restore Python, NumPy, CPU Torch, and every CUDA RNG.

    Separate contexts are used for fetch/collate, preprocessing, the native
    flow loss, the optional sampler, and the flow-time sweep.  Consequently an
    implementation detail in one phase cannot silently perturb another phase.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def _fixed_anchor_pointseg_aux_loss(
    policy: torch.nn.Module,
    enabled: bool,
) -> Generator[None, None, None]:
    """Request the training PointSeg scalar without switching modules to train mode.

    ``SongPointCloudConditioner`` normally omits its auxiliary loss in eval mode.
    A fixed-anchor comparison must reconstruct the exact training objective, but
    calling ``policy.train()`` would also update BatchNorm buffers and enable
    stochastic layers.  Toggle only the conditioner's non-persistent evaluation
    flag and restore it even if the forward pass fails.
    """

    if not enabled:
        yield
        return
    conditioners = [
        module for module in policy.modules() if hasattr(module, "_force_pointseg_aux_loss")
    ]
    if not conditioners:
        raise RuntimeError(
            "Fixed-anchor loss contract includes PointSeg, but no compatible "
            "SongPointCloudConditioner was found in the policy."
        )
    previous = [bool(module._force_pointseg_aux_loss) for module in conditioners]
    try:
        for module in conditioners:
            module._force_pointseg_aux_loss = True
        yield
    finally:
        for module, value in zip(conditioners, previous, strict=True):
            module._force_pointseg_aux_loss = value


@contextmanager
def _fixed_anchor_deterministic_algorithms(enabled: bool) -> Generator[None, None, None]:
    """Temporarily require deterministic kernels for strict anchor evaluation."""

    if not enabled:
        yield
        return
    algorithms_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark_before = torch.backends.cudnn.benchmark
    deterministic_before = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(algorithms_before, warn_only=warn_only_before)
        torch.backends.cudnn.benchmark = benchmark_before
        torch.backends.cudnn.deterministic = deterministic_before


def _seed_fixed_anchor_worker(_worker_id: int) -> None:
    """Seed non-Torch RNGs from DataLoader's deterministic worker seed."""

    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _load_fixed_anchor_manifest(
    path_value: str | Path,
    *,
    dataset_repo_id: str,
    dataset_length: int,
    strict: bool,
) -> dict[str, Any]:
    """Load and validate a checkpoint-independent global-frame manifest."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fixed-anchor indices manifest not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        if strict:
            raise ValueError(
                "Strict fixed-anchor evaluation requires an object manifest with "
                "schema_version, dataset_repo_id, dataset_length, and indices."
            )
        payload = {"indices": payload}
    if not isinstance(payload, dict):
        raise TypeError("Fixed-anchor manifest must be a JSON object or index list.")

    schema_version = payload.get("schema_version")
    if strict and schema_version != FIXED_ANCHOR_SCHEMA_VERSION:
        raise ValueError(
            "Fixed-anchor manifest schema mismatch: expected "
            f"{FIXED_ANCHOR_SCHEMA_VERSION}, got {schema_version!r}."
        )
    manifest_repo_id = payload.get("dataset_repo_id")
    if strict and str(manifest_repo_id) != str(dataset_repo_id):
        raise ValueError(
            "Fixed-anchor dataset mismatch: manifest has "
            f"{manifest_repo_id!r}, evaluator has {dataset_repo_id!r}."
        )
    manifest_length = payload.get("dataset_length")
    if strict and manifest_length != int(dataset_length):
        raise ValueError(
            "Fixed-anchor dataset length mismatch: manifest has "
            f"{manifest_length!r}, evaluator has {dataset_length}."
        )

    raw_loss_contract = payload.get("loss_contract")
    required_loss_contract_keys = (
        "pointseg_aux_loss_weight",
        "worldflow_loss_weight",
        "worldflow_geo_loss_weight",
        "worldflow_bridge_loss_weight",
        "worldflow_equiv_loss_weight",
    )
    if strict and not isinstance(raw_loss_contract, dict):
        raise ValueError("Strict fixed-anchor manifest requires a 'loss_contract' object.")
    raw_loss_contract = dict(raw_loss_contract or {})
    missing_loss_contract = [
        key for key in required_loss_contract_keys if key not in raw_loss_contract
    ]
    if strict and missing_loss_contract:
        raise ValueError(
            "Fixed-anchor manifest loss_contract is missing: "
            f"{missing_loss_contract}."
        )
    loss_contract: dict[str, float] = {}
    for key in required_loss_contract_keys:
        if key not in raw_loss_contract:
            continue
        value = float(raw_loss_contract[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Fixed-anchor loss_contract {key} must be finite and non-negative, got {value}."
            )
        loss_contract[key] = value

    raw_indices = payload.get("indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Fixed-anchor manifest requires a non-empty 'indices' list.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices):
        raise TypeError("Every fixed-anchor index must be an integer (booleans are invalid).")
    indices = [int(index) for index in raw_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("Fixed-anchor indices must be unique.")
    out_of_range = [index for index in indices if index < 0 or index >= int(dataset_length)]
    if out_of_range:
        raise IndexError(
            f"Fixed-anchor indices are outside [0, {dataset_length}): {out_of_range[:10]}"
        )

    canonical = {
        "schema_version": FIXED_ANCHOR_SCHEMA_VERSION,
        "dataset_repo_id": str(dataset_repo_id),
        "dataset_length": int(dataset_length),
        "indices": indices,
        "loss_contract": loss_contract,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return {
        **canonical,
        "path": str(path),
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        "count": len(indices),
    }


def _hash_tensor_into(hasher: Any, key: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().contiguous()
    hasher.update(key.encode("utf-8"))
    hasher.update(str(detached.dtype).encode("ascii"))
    hasher.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    # Viewing bytes before copying also supports bfloat16, which NumPy cannot
    # convert directly on all supported versions.
    byte_view = detached.reshape(-1).view(torch.uint8).cpu().numpy()
    hasher.update(memoryview(byte_view))


def _hash_nested_value_into(hasher: Any, key: str, value: Any) -> None:
    if torch.is_tensor(value):
        _hash_tensor_into(hasher, key, value)
        return
    if isinstance(value, dict):
        for child_key in sorted(value):
            _hash_nested_value_into(hasher, f"{key}.{child_key}", value[child_key])
        return
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _hash_nested_value_into(hasher, f"{key}[{index}]", child)
        return
    hasher.update(key.encode("utf-8"))
    hasher.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def _update_batch_fingerprint(hasher: Any, batch_index: int, batch: dict[str, Any]) -> None:
    hasher.update(f"batch:{int(batch_index)}".encode("ascii"))
    _hash_nested_value_into(hasher, "batch", batch)


def _tensor_state_fingerprints(module: torch.nn.Module, *, buffers: bool) -> dict[str, str]:
    named_tensors = module.named_buffers() if buffers else module.named_parameters()
    fingerprints: dict[str, str] = {}
    for name, tensor in named_tensors:
        hasher = hashlib.sha256()
        _hash_tensor_into(hasher, name, tensor)
        fingerprints[name] = hasher.hexdigest()
    return fingerprints


def _changed_tensor_fingerprints(
    before: dict[str, str],
    after: dict[str, str],
) -> list[str]:
    names = sorted(set(before) | set(after))
    return [name for name in names if before.get(name) != after.get(name)]


def _preprocessor_assets_fingerprint(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Fixed-anchor preprocessor path is not a directory: {path}")
    files = sorted(path.glob("policy_preprocessor*"))
    if not files:
        raise FileNotFoundError(f"No policy_preprocessor assets found in {path}")
    aggregate = hashlib.sha256()
    entries: dict[str, str] = {}
    for file in files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        entries[file.name] = digest
        aggregate.update(file.name.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    return {
        "path": str(path),
        "sha256": aggregate.hexdigest(),
        "files": entries,
    }


def _anchor_total_from_metrics(
    metrics: dict[str, Any],
    *,
    pointseg_weight: float,
) -> tuple[float, float]:
    missing = [key for key in FIXED_ANCHOR_REQUIRED_METRICS if _to_scalar(metrics.get(key)) is None]
    if missing:
        raise ValueError(f"Fixed-anchor forward is missing required loss metrics: {missing}")
    action = float(_to_scalar(metrics["loss_action"]))
    worldflow = float(_to_scalar(metrics["loss_worldflow_flow"]))
    pointseg = float(_to_scalar(metrics["loss_pointseg_aux"]))
    total = action + worldflow + float(pointseg_weight) * pointseg
    native = _to_scalar(metrics.get("loss"))
    gap = abs(float(native) - total) if native is not None else math.nan
    return total, gap


def _validate_fixed_anchor_loss_contract(
    policy_config: Any,
    expected: dict[str, float],
    *,
    cli_expected_pointseg_weight: float | None = None,
) -> dict[str, float]:
    expected = {name: float(value) for name, value in expected.items()}
    if cli_expected_pointseg_weight is not None:
        cli_weight = float(cli_expected_pointseg_weight)
        manifest_weight = expected.get("pointseg_aux_loss_weight")
        if manifest_weight is None or not math.isclose(
            cli_weight,
            manifest_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "CLI fixed_anchor_expected_pointseg_weight disagrees with manifest: "
                f"cli={cli_weight}, manifest={manifest_weight}."
            )
    actual = {name: float(getattr(policy_config, name)) for name in expected}
    drift = {
        name: (expected[name], actual[name])
        for name in expected
        if not math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1e-12)
    }
    if drift:
        details = ", ".join(
            f"{name}: expected={wanted}, actual={got}"
            for name, (wanted, got) in drift.items()
        )
        raise ValueError(f"Fixed-anchor loss contract drifted ({details}).")
    return actual


def _resolve_libero_dataset_domain_selection(
    cfg: SongEvalPipelineConfig,
) -> dict[str, Any] | None:
    """Resolve one LIBERO suite/task to the exact episodes stored in the training dataset."""

    if not cfg.libero_dataset_domain_action_mse:
        if cfg.libero_suite is not None or cfg.libero_task_id is not None:
            raise ValueError(
                "--libero_suite/--libero_task_id require "
                "--libero_dataset_domain_action_mse=true."
            )
        return None
    if cfg.libero_suite is None or cfg.libero_task_id is None:
        raise ValueError(
            "--libero_dataset_domain_action_mse=true requires both "
            "--libero_suite and --libero_task_id."
        )
    if cfg.dataset.streaming:
        raise ValueError("LIBERO dataset-domain action MSE does not support streaming datasets.")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        from libero.libero import benchmark
    except Exception as exc:
        raise RuntimeError(
            "LIBERO must be importable to resolve --libero_suite and --libero_task_id."
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_name = str(cfg.libero_suite)
    if suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO suite {suite_name!r}; available suites are "
            f"{sorted(benchmark_dict)}."
        )
    suite = benchmark_dict[suite_name]()
    task_id = int(cfg.libero_task_id)
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(
            f"Invalid task id {task_id} for {suite_name}; expected 0..{len(suite.tasks) - 1}."
        )
    task = suite.get_task(task_id)
    task_language = str(task.language)

    metadata = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    matching_episode_indices = [
        int(episode["episode_index"])
        for episode in metadata.episodes
        if task_language in list(episode["tasks"])
    ]
    if cfg.dataset.episodes is not None:
        requested = {int(index) for index in cfg.dataset.episodes}
        matching_episode_indices = [
            index for index in matching_episode_indices if index in requested
        ]
    if not matching_episode_indices:
        available_tasks = sorted(str(task_name) for task_name in metadata.tasks.index)
        preview = available_tasks[:20]
        raise ValueError(
            f"No episodes for {suite_name} task {task_id} ({task_language!r}) were found "
            f"in dataset {cfg.dataset.repo_id!r}. Available task labels include: {preview}"
        )

    frame_indices: list[int] = []
    for episode_index in matching_episode_indices:
        episode = metadata.episodes[episode_index]
        frame_indices.extend(
            range(
                int(episode["dataset_from_index"]),
                int(episode["dataset_to_index"]),
            )
        )
    return {
        "mode": "libero_training_dataset_action_mse",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": str(task.name),
        "task_language": task_language,
        "episode_indices": matching_episode_indices,
        "frame_indices": frame_indices,
        "episode_count": len(matching_episode_indices),
        "frame_count": len(frame_indices),
        "environment_source": "converted_training_dataset",
        "benchmark_rollout": False,
    }


def _make_eval_dataset(
    cfg: SongEvalPipelineConfig,
    dataset_domain_selection: dict[str, Any] | None,
):
    # Build the complete dataset first. Applying the frame subset after the
    # PointSeg/cache wrappers preserves their global dataset-index alignment.
    requested_episodes = cfg.dataset.episodes
    if dataset_domain_selection is not None:
        cfg.dataset.episodes = None
    dataset = make_dataset(cfg)
    dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
    dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
    dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)
    if dataset_domain_selection is not None:
        dataset = EvalFrameSubset(dataset, dataset_domain_selection["frame_indices"])
        cfg.dataset.episodes = requested_episodes
    return dataset


def _make_eval_preprocessor(
    cfg: TrainPipelineConfig,
    policy,
    dataset,
    device: torch.device,
    *,
    pretrained_path: str | Path | None = None,
):
    processor_kwargs: dict[str, Any] = {}
    resolved_pretrained_path = (
        pretrained_path if pretrained_path is not None else cfg.policy.pretrained_path
    )
    if resolved_pretrained_path is not None:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=resolved_pretrained_path,
        **processor_kwargs,
    )
    return preprocessor


def _make_eval_dataloader(
    cfg: SongEvalPipelineConfig,
    dataset,
    device: torch.device,
    *,
    fixed_anchor: bool = False,
):
    dataset_domain_action_mse = bool(cfg.libero_dataset_domain_action_mse)
    if (
        hasattr(cfg.policy, "drop_n_last_frames")
        and not dataset_domain_action_mse
        and not fixed_anchor
    ):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        sampler = None

    collate_fn = make_song_train_collate_fn(dataset)
    dataloader_num_workers = int(cfg.num_workers)
    if isinstance(collate_fn, OnlinePointSegBatchCollator) and collate_fn.device.type == "cuda":
        if dataloader_num_workers > 0:
            logging.warning(
                "Online PointSeg pseudo labels use CUDA; setting num_workers=0 for safe evaluation."
            )
        dataloader_num_workers = 0

    generator = None
    worker_init_fn = None
    if fixed_anchor:
        generator = torch.Generator()
        generator.manual_seed(
            _fixed_anchor_phase_seed(int(cfg.fixed_anchor_seed), "dataloader")
        )
        worker_init_fn = _seed_fixed_anchor_worker

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=dataloader_num_workers,
        batch_size=cfg.batch_size,
        shuffle=(
            sampler is None
            and not cfg.dataset.streaming
            and not dataset_domain_action_mse
            and not fixed_anchor
        ),
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,
        persistent_workers=dataloader_num_workers > 0,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def _to_scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _gather_metric_sums(
    accelerator: Accelerator,
    metrics: dict[str, Any],
    local_batch_size: int,
    *,
    metric_keys: tuple[str, ...] = EVAL_METRIC_KEYS,
) -> tuple[dict[str, float], dict[str, float]]:
    """Gather weighted metric sums/counts from every process."""

    payload: list[float] = []
    for key in metric_keys:
        value = _to_scalar(metrics.get(key))
        if value is None:
            payload.extend((0.0, 0.0))
        else:
            payload.extend((value * local_batch_size, float(local_batch_size)))

    local = torch.tensor(payload, device=accelerator.device, dtype=torch.float64).unsqueeze(0)
    gathered = accelerator.gather(local)
    if gathered.ndim == 1:
        gathered = gathered.unsqueeze(0)

    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for index, key in enumerate(metric_keys):
        sums[key] = float(gathered[:, 2 * index].sum().item())
        counts[key] = float(gathered[:, 2 * index + 1].sum().item())
    return sums, counts


def _format_metrics(
    metrics: dict[str, float],
    *,
    metric_keys: tuple[str, ...] = EVAL_METRIC_KEYS,
) -> str:
    parts = []
    for key in metric_keys:
        if key in metrics:
            parts.append(f"{key}:{metrics[key]:.4g}")
    return " ".join(parts)


def _sampled_action_metrics(
    policy,
    batch: dict[str, torch.Tensor],
    noise_mode: str = "policy",
) -> dict[str, torch.Tensor]:
    """Compare one complete flow-matching sample with its ground-truth chunk.

    The normal ``loss_action`` is a velocity-field objective at one random
    diffusion time.  It is useful for optimization, but it does not directly
    measure the trajectory returned by all denoising steps.  These optional
    metrics expose that distinction without changing parameters or gradients.
    """

    policy.reset()
    if noise_mode not in {"policy", "zero", "pose9_identity"}:
        raise ValueError(
            "sample_action_noise_mode must be one of policy, zero, pose9_identity; "
            f"got {noise_mode!r}."
        )
    predict_kwargs: dict[str, torch.Tensor] = {}
    if noise_mode != "policy":
        batch_size = int(batch[ACTION].shape[0])
        chunk_size = int(policy.config.chunk_size)
        action_dim = int(policy.config.max_action_dim)
        noise = torch.zeros(
            batch_size,
            chunk_size,
            action_dim,
            device=batch[ACTION].device,
            dtype=torch.float32,
        )
        if noise_mode == "pose9_identity":
            if action_dim < 10:
                raise ValueError("pose9_identity noise requires max_action_dim >= 10.")
            noise[..., 3] = 1.0
            noise[..., 7] = 1.0
        predict_kwargs["noise"] = noise
        if policy.config.worldflow_enable:
            world_noise = torch.zeros(
                batch_size,
                chunk_size,
                9,
                device=batch[ACTION].device,
                dtype=torch.float32,
            )
            world_noise[..., 3] = 1.0
            world_noise[..., 7] = 1.0
            predict_kwargs["worldflow_noise"] = world_noise
    predicted = policy.predict_action_chunk(batch, **predict_kwargs)
    target = batch[ACTION][..., : predicted.shape[-1]]
    if predicted.shape != target.shape:
        raise ValueError(
            f"Sampled/target action chunk shapes differ: {predicted.shape} != {target.shape}."
        )
    actions_is_pad = batch.get(f"{ACTION}_is_pad")
    if torch.is_tensor(actions_is_pad):
        valid = ~actions_is_pad.to(device=predicted.device, dtype=torch.bool)
    else:
        valid = torch.ones(predicted.shape[:2], device=predicted.device, dtype=torch.bool)
    valid_f = valid.to(dtype=predicted.dtype)
    valid_count = valid_f.sum().clamp_min(1.0)
    error = predicted - target
    metrics = {
        "sample_action_mse": (
            error.square().mean(dim=-1) * valid_f
        ).sum()
        / valid_count,
        "sample_action_translation_l2_m": (
            torch.linalg.norm(error[..., :3], dim=-1) * valid_f
        ).sum()
        / valid_count,
        "sample_action_rot6d_mse": (
            error[..., 3:9].square().mean(dim=-1) * valid_f
        ).sum()
        / valid_count,
    }
    if predicted.shape[-1] >= 10:
        metrics["sample_action_gripper_mae_m"] = (
            error[..., 9].abs() * valid_f
        ).sum() / valid_count
    return metrics


def _flow_time_sweep_metrics(
    policy,
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Measure velocity/endpoint error across the complete integration interval.

    Standard SmolVLA samples training time from ``Beta(1.5, 1)``. On very small
    datasets this can leave the beginning of the inference ODE under-trained.
    Reusing the same Ego/World noise and point-operation RNG state at every
    requested time isolates that coverage issue from ordinary sampling noise.
    """

    action = batch[ACTION]
    if not torch.is_tensor(action):
        raise TypeError("Flow-time sweep requires a tensor action chunk.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use the exact training/inference prior saved in the checkpoint.  A
    # hard-coded Gaussian makes the sweep an out-of-distribution diagnostic
    # whenever pose9_action_noise_enable is active, and even the legacy prior
    # is unit Gaussian rather than N(0, 0.1).
    model = getattr(policy, "model", None)
    if model is None:
        raise AttributeError("Flow-time sweep requires policy.model.")
    if bool(getattr(policy.config, "se3_enable", False)):
        _noise_transform, _gripper_noise, fixed_noise = model.sample_se3_action_noise(action)
    elif bool(getattr(policy.config, "pose9_action_noise_enable", False)):
        fixed_noise = model.sample_pose9_action_noise(tuple(action.shape), action.device)
    else:
        fixed_noise = model.sample_noise(action.shape, action.device)
    fixed_noise = fixed_noise.to(dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}
    for time_value in FLOW_TIME_SWEEP_VALUES:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        time = torch.full(
            (action.shape[0],),
            float(time_value),
            device=action.device,
            dtype=torch.float32,
        )
        loss, output = policy(batch, noise=fixed_noise, time=time)
        output = dict(output or {})
        output.setdefault("loss_action", loss)
        prefix = f"flow_t{round(time_value * 1000):03d}_"
        for metric_name in FLOW_TIME_SWEEP_METRICS:
            value = output.get(metric_name)
            if torch.is_tensor(value):
                metrics[prefix + metric_name] = value.detach()
            else:
                scalar = _to_scalar(value)
                if scalar is not None:
                    metrics[prefix + metric_name] = action.new_tensor(scalar)
    return metrics


@parser.wrap()
def evaluate(cfg: SongEvalPipelineConfig, accelerator: Accelerator | None = None) -> dict[str, Any]:
    # TrainPipelineConfig normally rejects an output directory containing prior
    # artifacts. Evaluation only replaces eval_metrics.json and never writes a
    # checkpoint, so reusing an evaluation directory is safe and convenient.
    requested_output_dir = cfg.output_dir
    if requested_output_dir is not None and Path(requested_output_dir).is_dir():
        cfg.output_dir = None
    fixed_anchor_enabled = cfg.fixed_anchor_indices_path is not None
    if fixed_anchor_enabled and cfg.fixed_anchor_deterministic_algorithms:
        # PyTorch deterministic CUDA GEMMs require this to be set before the
        # first cuBLAS workspace is created in this process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not cfg.resume:
        # These are training-only checkpoint metadata.  Saved restarted-phase
        # configs can keep them enabled, but a read-only evaluation neither
        # restores nor mutates optimizer state.
        cfg.resume_restart_scheduler = False
        cfg.resume_reset_optimizer_moments = False
        cfg.resume_optimizer_moments_reset_step = None
    cfg.validate()
    if requested_output_dir is not None:
        cfg.output_dir = Path(requested_output_dir)
    if cfg.resume:
        raise ValueError(
            "eval_song.py does not resume optimizer state; use --policy.path instead of --resume."
        )
    if cfg.policy.pretrained_path is None:
        raise ValueError("eval_song.py requires a trained checkpoint provided through --policy.path.")

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            cpu=cfg.policy.device == "cpu",
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    if fixed_anchor_enabled and cfg.libero_dataset_domain_action_mse:
        raise ValueError(
            "fixed_anchor_indices_path and libero_dataset_domain_action_mse cannot be combined; "
            "anchor indices are defined in the complete wrapped training dataset."
        )
    if fixed_anchor_enabled and cfg.dataset.streaming:
        raise ValueError("Fixed-anchor evaluation requires a finite non-streaming dataset.")
    if fixed_anchor_enabled and cfg.dataset.episodes is not None:
        raise ValueError(
            "Fixed-anchor indices are global dataset indices; dataset.episodes must remain unset."
        )
    if fixed_anchor_enabled and cfg.fixed_anchor_strict and accelerator.num_processes != 1:
        raise ValueError(
            "Strict fixed-anchor evaluation requires one process. Run four checkpoints on four "
            "separate GPUs instead; this avoids distributed tail duplication and makes hashes exact."
        )
    if fixed_anchor_enabled and cfg.fixed_anchor_strict and cfg.fixed_anchor_preprocessor_path is None:
        raise ValueError(
            "Strict fixed-anchor evaluation requires fixed_anchor_preprocessor_path so every "
            "checkpoint uses exactly the same normalization/tokenization assets."
        )
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = not fixed_anchor_enabled
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset_domain_selection = _resolve_libero_dataset_domain_selection(cfg)

    if is_main_process:
        logging.info("Creating evaluation dataset")
        if dataset_domain_selection is not None:
            logging.info(
                "LIBERO dataset-domain action MSE: suite=%s task=%s episodes=%s frames=%s",
                dataset_domain_selection["suite"],
                dataset_domain_selection["task_id"],
                dataset_domain_selection["episode_count"],
                dataset_domain_selection["frame_count"],
            )
        dataset = _make_eval_dataset(cfg, dataset_domain_selection)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = _make_eval_dataset(cfg, dataset_domain_selection)

    fixed_anchor_manifest = None
    if fixed_anchor_enabled:
        fixed_anchor_manifest = _load_fixed_anchor_manifest(
            cfg.fixed_anchor_indices_path,
            dataset_repo_id=str(cfg.dataset.repo_id),
            dataset_length=len(dataset),
            strict=bool(cfg.fixed_anchor_strict),
        )
        dataset = EvalFrameSubset(dataset, fixed_anchor_manifest["indices"])
        if is_main_process:
            logging.info(
                "Fixed anchor loaded: samples=%s sha256=%s path=%s",
                fixed_anchor_manifest["count"],
                fixed_anchor_manifest["sha256"],
                fixed_anchor_manifest["path"],
            )

    if is_main_process:
        logging.info("Loading evaluation policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if is_main_process and hasattr(policy.config, "flow_contract_summary"):
        logging.info("Resolved flow contract: %s", policy.config.flow_contract_summary())
    fixed_anchor_loss_contract = None
    if fixed_anchor_enabled:
        fixed_anchor_loss_contract = _validate_fixed_anchor_loss_contract(
            policy.config,
            fixed_anchor_manifest["loss_contract"],
            cli_expected_pointseg_weight=cfg.fixed_anchor_expected_pointseg_weight,
        )
    ensure_ddp_parameters_initialized(policy, accelerator)
    preprocessor_source = (
        cfg.fixed_anchor_preprocessor_path
        if fixed_anchor_enabled and cfg.fixed_anchor_preprocessor_path is not None
        else cfg.policy.pretrained_path
    )
    preprocessor_assets = (
        _preprocessor_assets_fingerprint(preprocessor_source)
        if fixed_anchor_enabled
        else None
    )
    preprocessor = _make_eval_preprocessor(
        cfg,
        policy,
        dataset,
        device,
        pretrained_path=preprocessor_source,
    )
    dataloader = _make_eval_dataloader(
        cfg,
        dataset,
        device,
        fixed_anchor=fixed_anchor_enabled,
    )
    effective_num_workers = int(dataloader.num_workers)

    policy, dataloader = accelerator.prepare(policy, dataloader)
    policy.eval()
    raw_policy = accelerator.unwrap_model(policy)
    parameter_versions_before = {
        name: int(parameter._version) for name, parameter in raw_policy.named_parameters()
    }
    buffer_fingerprints_before = None
    if fixed_anchor_enabled:
        buffer_fingerprints_before = _tensor_state_fingerprints(raw_policy, buffers=True)
        training_modules = [name for name, module in raw_policy.named_modules() if module.training]
        if training_modules:
            raise RuntimeError(
                "Evaluation policy contains modules left in training mode: "
                + ", ".join(training_modules[:10])
            )
    available_batches = len(dataloader)
    max_batches = available_batches if fixed_anchor_enabled else min(int(cfg.steps), available_batches)
    if max_batches <= 0:
        raise ValueError(
            f"No evaluation batches requested: steps={cfg.steps}, available={available_batches}."
        )

    output_dir = Path(cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(colored("Evaluation output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
        logging.info(
            "Evaluation is read-only: batches=%s/%s, batch_size=%s, processes=%s, gradients=disabled",
            max_batches,
            available_batches,
            cfg.batch_size,
            accelerator.num_processes,
        )
        if fixed_anchor_enabled:
            logging.info(
                "Strict fixed anchor: seed=%s deterministic_algorithms=%s preprocessor_sha256=%s",
                cfg.fixed_anchor_seed,
                bool(cfg.fixed_anchor_deterministic_algorithms),
                preprocessor_assets["sha256"],
            )
    accelerator.wait_for_everyone()

    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project and is_main_process else None

    metric_keys = (
        EVAL_METRIC_KEYS + FIXED_ANCHOR_EXTRA_METRIC_KEYS
        if fixed_anchor_enabled
        else EVAL_METRIC_KEYS
    )
    running_sums = dict.fromkeys(metric_keys, 0.0)
    running_counts = dict.fromkeys(metric_keys, 0.0)
    raw_input_hasher = hashlib.sha256()
    preprocessed_input_hasher = hashlib.sha256()
    observed_anchor_indices: list[int] = []

    progress = None
    if is_main_process:
        progress = tqdm(
            total=max_batches,
            desc="Evaluating",
            unit="batch",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )

    iterator_seed = _fixed_anchor_phase_seed(
        int(cfg.fixed_anchor_seed), "dataloader", process_index=accelerator.process_index
    )
    with (_fixed_anchor_rng(iterator_seed) if fixed_anchor_enabled else nullcontext()):
        data_iterator = iter(dataloader)

    with _fixed_anchor_deterministic_algorithms(
        fixed_anchor_enabled and bool(cfg.fixed_anchor_deterministic_algorithms)
    ):
        for eval_step in range(1, max_batches + 1):
            fetch_seed = _fixed_anchor_phase_seed(
                int(cfg.fixed_anchor_seed),
                "fetch",
                eval_step,
                accelerator.process_index,
            )
            if fixed_anchor_enabled:
                with _fixed_anchor_rng(fetch_seed):
                    raw_batch = next(data_iterator)
            else:
                raw_batch = next(data_iterator)

            if fixed_anchor_enabled:
                raw_indices = raw_batch.get("index")
                if not torch.is_tensor(raw_indices):
                    raise KeyError("Fixed-anchor raw batch is missing tensor key 'index'.")
                observed_anchor_indices.extend(int(index) for index in raw_indices.reshape(-1).tolist())
                if cfg.fixed_anchor_hash_inputs:
                    _update_batch_fingerprint(raw_input_hasher, eval_step, raw_batch)

            preprocess_seed = _fixed_anchor_phase_seed(
                int(cfg.fixed_anchor_seed),
                "preprocess",
                eval_step,
                accelerator.process_index,
            )
            if fixed_anchor_enabled:
                with _fixed_anchor_rng(preprocess_seed):
                    batch = preprocessor(raw_batch)
            else:
                batch = preprocessor(raw_batch)
            local_batch_size = int(batch[ACTION].shape[0])
            if fixed_anchor_enabled and cfg.fixed_anchor_hash_inputs:
                _update_batch_fingerprint(preprocessed_input_hasher, eval_step, batch)

            # Make flow time/noise, point sampling, and every other native RNG
            # draw reproducible without replacing the model's training prior.
            eval_seed = (
                _fixed_anchor_phase_seed(
                    int(cfg.fixed_anchor_seed),
                    "forward",
                    eval_step,
                    accelerator.process_index,
                )
                if fixed_anchor_enabled
                else int(cfg.seed or 0) + eval_step
            )
            if not fixed_anchor_enabled:
                torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)
            forward_rng = (
                _fixed_anchor_rng(eval_seed) if fixed_anchor_enabled else nullcontext()
            )
            force_pointseg_aux = (
                fixed_anchor_enabled
                and float(fixed_anchor_loss_contract["pointseg_aux_loss_weight"]) > 0.0
            )
            with (
                forward_rng,
                _fixed_anchor_pointseg_aux_loss(policy, force_pointseg_aux),
                torch.inference_mode(),
                accelerator.autocast(),
            ):
                loss, output_dict = policy(batch)

            local_metrics = dict(output_dict or {})
            local_metrics["loss"] = loss.detach()
            if fixed_anchor_enabled:
                anchor_total, anchor_gap = _anchor_total_from_metrics(
                    local_metrics,
                    pointseg_weight=fixed_anchor_loss_contract["pointseg_aux_loss_weight"],
                )
                local_metrics["loss_anchor_total"] = anchor_total
                local_metrics["loss_anchor_total_gap"] = anchor_gap
                if cfg.fixed_anchor_strict and anchor_gap > 1e-7:
                    raise RuntimeError(
                        "Native loss disagrees with fixed anchor formula "
                        "action + worldflow + configured_weight * pointseg: "
                        f"gap={anchor_gap:.9g} at batch {eval_step}."
                    )
            if cfg.sample_action_mse:
                sample_seed = (
                    _fixed_anchor_phase_seed(
                        int(cfg.fixed_anchor_seed),
                        "sample",
                        eval_step,
                        accelerator.process_index,
                    )
                    if fixed_anchor_enabled
                    else int(cfg.seed or 0) + 1_000_000 + eval_step
                )
                if not fixed_anchor_enabled:
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                sample_rng = (
                    _fixed_anchor_rng(sample_seed) if fixed_anchor_enabled else nullcontext()
                )
                with sample_rng, torch.inference_mode(), accelerator.autocast():
                    local_metrics.update(
                        _sampled_action_metrics(
                            accelerator.unwrap_model(policy),
                            batch,
                            noise_mode=cfg.sample_action_noise_mode,
                        )
                    )
            if cfg.flow_time_sweep:
                sweep_seed = (
                    _fixed_anchor_phase_seed(
                        int(cfg.fixed_anchor_seed),
                        "sweep",
                        eval_step,
                        accelerator.process_index,
                    )
                    if fixed_anchor_enabled
                    else int(cfg.seed or 0) + 2_000_000 + eval_step
                )
                sweep_rng = (
                    _fixed_anchor_rng(sweep_seed) if fixed_anchor_enabled else nullcontext()
                )
                with sweep_rng, torch.inference_mode(), accelerator.autocast():
                    local_metrics.update(
                        _flow_time_sweep_metrics(
                            policy,
                            batch,
                            seed=sweep_seed,
                        )
                    )
            batch_sums, batch_counts = _gather_metric_sums(
                accelerator,
                local_metrics,
                local_batch_size,
                metric_keys=metric_keys,
            )
            for key in metric_keys:
                running_sums[key] += batch_sums[key]
                running_counts[key] += batch_counts[key]

            batch_metrics = {
                key: batch_sums[key] / batch_counts[key]
                for key in metric_keys
                if batch_counts[key] > 0
            }

            if is_main_process and progress is not None:
                progress.update(1)
            if is_main_process and cfg.log_freq > 0 and eval_step % cfg.log_freq == 0:
                logging.info(
                    "eval_step:%s %s",
                    eval_step,
                    _format_metrics(batch_metrics, metric_keys=metric_keys),
                )
                if wandb_logger is not None:
                    wandb_logger.log_dict(
                        {f"eval/{key}": value for key, value in batch_metrics.items()},
                        eval_step,
                    )

    if progress is not None:
        progress.close()

    summary_metrics = {
        key: running_sums[key] / running_counts[key] for key in metric_keys if running_counts[key] > 0
    }
    fixed_anchor_report = None
    if fixed_anchor_enabled:
        expected_indices = fixed_anchor_manifest["indices"]
        if observed_anchor_indices != expected_indices:
            mismatch_index = next(
                (
                    index
                    for index, (observed, expected) in enumerate(
                        zip(observed_anchor_indices, expected_indices, strict=False)
                    )
                    if observed != expected
                ),
                min(len(observed_anchor_indices), len(expected_indices)),
            )
            raise RuntimeError(
                "Fixed-anchor DataLoader changed index order/content: "
                f"first mismatch={mismatch_index}, observed_count={len(observed_anchor_indices)}, "
                f"expected_count={len(expected_indices)}."
            )
        raw_sha256 = raw_input_hasher.hexdigest() if cfg.fixed_anchor_hash_inputs else None
        preprocessed_sha256 = (
            preprocessed_input_hasher.hexdigest() if cfg.fixed_anchor_hash_inputs else None
        )
        comparison_contract = {
            "schema_version": FIXED_ANCHOR_SCHEMA_VERSION,
            "manifest_sha256": fixed_anchor_manifest["sha256"],
            "fixed_anchor_seed": int(cfg.fixed_anchor_seed),
            "batch_size": int(cfg.batch_size),
            "num_processes": int(accelerator.num_processes),
            "num_workers": effective_num_workers,
            "preprocessor_sha256": preprocessor_assets["sha256"],
            "raw_input_sha256": raw_sha256,
            "preprocessed_input_sha256": preprocessed_sha256,
            "pointseg_sample_cache_dir": str(cfg.pointseg_sample_cache_dir),
            "deterministic_algorithms": bool(cfg.fixed_anchor_deterministic_algorithms),
            "loss_formula": (
                "loss_action + loss_worldflow_flow + "
                f"{fixed_anchor_loss_contract['pointseg_aux_loss_weight']:.17g} * "
                "loss_pointseg_aux"
            ),
        }
        contract_json = json.dumps(
            comparison_contract,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        fixed_anchor_report = {
            "enabled": True,
            "strict": bool(cfg.fixed_anchor_strict),
            "manifest": fixed_anchor_manifest,
            "rng_phases": dict(_FIXED_ANCHOR_PHASE_IDS),
            "preprocessor_assets": preprocessor_assets,
            "loss_contract": fixed_anchor_loss_contract,
            "raw_input_sha256": raw_sha256,
            "preprocessed_input_sha256": preprocessed_sha256,
            "comparison_contract": comparison_contract,
            "comparison_contract_sha256": hashlib.sha256(
                contract_json.encode("utf-8")
            ).hexdigest(),
        }
    evaluated_samples = int(running_counts.get("loss_action", 0.0))
    dataset_domain_report = None
    if dataset_domain_selection is not None:
        dataset_domain_report = {
            key: value
            for key, value in dataset_domain_selection.items()
            if key != "frame_indices"
        }
    summary = {
        "checkpoint": str(cfg.policy.pretrained_path),
        "dataset": str(cfg.dataset.repo_id),
        "pointseg_sample_cache_dir": str(cfg.pointseg_sample_cache_dir),
        "evaluated_batches": max_batches,
        "evaluated_samples": evaluated_samples,
        "batch_size_per_process": int(cfg.batch_size),
        "num_processes": int(accelerator.num_processes),
        "seed": cfg.seed,
        "metrics": summary_metrics,
        "dataset_domain_selection": dataset_domain_report,
        "metric_semantics": {
            "loss_action": (
                "The checkpoint's native flow-matching action velocity MSE, "
                "computed by policy(batch) with the same preprocessing, action chunks, "
                "padding masks, and PointSeg cache path as training."
            ),
            "sample_action_mse": (
                "Optional MSE between the fully denoised action chunk returned by "
                "predict_action_chunk and the ground-truth action chunk. Noise mode: "
                f"{cfg.sample_action_noise_mode}."
            ),
            "flow_time_sweep": (
                "Optional fixed-noise velocity and one-step endpoint diagnostics at "
                f"times {FLOW_TIME_SWEEP_VALUES}; this does not update parameters."
            ),
        },
    }
    if fixed_anchor_enabled:
        summary["fixed_anchor"] = fixed_anchor_report
        summary["metric_semantics"]["loss_anchor_total"] = (
            "Checkpoint-comparison objective reconstructed exactly as loss_action + "
            "loss_worldflow_flow + the manifest/config-verified pointseg weight * "
            "loss_pointseg_aux."
        )

    gradients_created = [
        name for name, parameter in raw_policy.named_parameters() if parameter.grad is not None
    ]
    if gradients_created:
        raise RuntimeError(
            "Evaluation unexpectedly created parameter gradients: " + ", ".join(gradients_created[:10])
        )
    parameters_modified = [
        name
        for name, parameter in raw_policy.named_parameters()
        if int(parameter._version) != parameter_versions_before[name]
    ]
    if parameters_modified:
        raise RuntimeError(
            "Evaluation unexpectedly modified model parameters in place: "
            + ", ".join(parameters_modified[:10])
        )
    summary["gradients_created"] = False
    summary["model_parameters_unchanged"] = True
    if fixed_anchor_enabled:
        buffer_fingerprints_after = _tensor_state_fingerprints(raw_policy, buffers=True)
        buffers_modified = _changed_tensor_fingerprints(
            buffer_fingerprints_before,
            buffer_fingerprints_after,
        )
        if buffers_modified:
            raise RuntimeError(
                "Evaluation unexpectedly modified model buffers (including possible BatchNorm stats): "
                + ", ".join(buffers_modified[:10])
            )
        summary["model_buffers_unchanged"] = True
        summary["all_modules_eval_mode"] = True

    if is_main_process:
        logging.info(
            "Evaluation summary: %s",
            _format_metrics(summary_metrics, metric_keys=metric_keys),
        )
        summary_path = output_dir / "eval_metrics.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        logging.info("Saved evaluation metrics to %s", summary_path)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return summary


def main() -> None:
    register_third_party_plugins()
    evaluate()


def _apply_song_eval_debug_defaults() -> None:
    if len(sys.argv) > 1 and os.environ.get("SONG_EVAL_DEBUG_DEFAULTS") != "1":
        return
    sys.argv = [
        "eval_song.py",
        "--policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/train/ep_vla/checkpoints/last/pretrained_model",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset",
        "--pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache",
        "--policy.vla_adapter_enable=true",
        "--policy.vla_adapter_freeze_vlm=true",
        "--policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct",
        "--policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base",
        "--policy.load_vlm_weights=true",
        "--batch_size=8",
        # In eval_song, --steps is a maximum batch count. Evaluation still
        # traverses the finite DataLoader at most once, even when this matches
        # the much larger training update count.
        "--steps=80000",
        "--log_freq=1",
        "--output_dir=/home/liusong/temp/eval_song_test",
        "--job_name=temp_eval",
        "--policy.device=cuda",
        "--wandb.enable=false",
        "--wandb.disable_artifact=true",
        "--num_workers=8",
        "--policy.pointseg_enable=true",
        "--policy.pointseg_backbone_type=litept",
        "--policy.pointseg_grid_size=0.01",
        "--policy.pointseg_feature_dim=64",
        "--policy.pointseg_aux_loss_weight=0.001",
        "--policy.pointseg_foreground_ratio=0.08",
        "--policy.pointseg_background_ratio=0.08",
        "--policy.pointseg_min_foreground_points=4000",
        "--policy.pointseg_min_background_points=0",
        "--policy.pointseg_use_temporal_priors_as_input=false",
        "--policy.pointseg_use_pseudo_selection=false",
        "--policy.worldflow_enable=false",
        "--policy.worldflow_se3_head_enable=false",
        "--policy.se3_enable=false",
        "--policy.se3_final_correction_enable=false",
    ]


if __name__ == "__main__":
    _apply_song_eval_debug_defaults()
    main()
