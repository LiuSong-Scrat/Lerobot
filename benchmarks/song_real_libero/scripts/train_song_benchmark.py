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
import dataclasses
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler, TaskBalancedFrameSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.optimizers import FP32MasterAdamW, optimizer_model_parameters
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import (
    FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
)
from lerobot.policies.smolvla.modeling_smolvla import (
    expected_full_molmo2er_parameter_budget,
    full_molmo2er_trainable_parameter_prefixes,
    is_full_molmo2er_lora_policy_parameter_name,
)
from lerobot.policies.smolvla.processor_smolvla import validate_smolvla_worldflow_preprocessor
from lerobot.policies.smolvla.song_pointseg import (
    DEFAULT_FUTURE_OFFSETS,
    ROLE_FOREGROUND,
    PseudoLabelConfig,
    SongPointSegCachedDataset,
    SongTemporalPointCloudDataset,
    compose_point_cloud_views,
    generate_pseudo_labels,
    open_episode_point_clouds,
    parse_camera_views,
    point_cloud_dir_for_view,
    song_pointseg_collate,
    write_role_ply,
)
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state_for_resume,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

CUDA_ALLOCATOR_LEASE_ENABLE_ENV = "MOLMO_FULL_CUDA_LEASE_ENABLE"
CUDA_ALLOCATOR_LEASE_TARGET_GIB_ENV = "MOLMO_FULL_CUDA_LEASE_TARGET_GIB"
CUDA_ALLOCATOR_LEASE_CHUNK_MIB_ENV = "MOLMO_FULL_CUDA_LEASE_CHUNK_MIB"
CUDA_ALLOCATOR_LEASE_HEADROOM_MIB_ENV = "MOLMO_FULL_CUDA_LEASE_HEADROOM_MIB"
CHECKPOINT_RETENTION_ENV = "MOLMO_CHECKPOINTS_TO_KEEP"
_MIB = 1024**2
_GIB = 1024**3
_MAX_CUDA_ALLOCATOR_LEASE_CHUNKS = 4096
_CUDA_FREE_AUDIT_TOLERANCE_BYTES = 8 * _MIB
EXACT_GLOBAL_BATCH_MANIFEST_NAME = "exact_global_batch_manifest.json"
FULL_MOLMO2ER_FIRST_STEP_GRADIENT_AUDIT_NAME = (
    "full_molmo2er_first_optimizer_step_gradient_audit.json"
)


@contextmanager
def fixed_overfit_rng(seed: int):
    """Seed then restore all RNGs used by the fixed-batch capacity diagnostic."""

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


def clone_fixed_overfit_batch(value: Any) -> Any:
    """Clone a cached collated batch so preprocessing cannot mutate the source."""

    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_fixed_overfit_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_fixed_overfit_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_fixed_overfit_batch(item) for item in value)
    return value


@dataclasses.dataclass(frozen=True)
class ExactGlobalBatchPlan:
    """A fixed-rank schedule whose DDP gradient is an exact global sample mean."""

    global_batch_size: int
    world_size: int
    micro_batch_size_per_rank: int
    gradient_accumulation_steps: int
    full_micro_steps: int
    partial_active_ranks: int
    physical_forward_samples_per_optimizer_step: int
    discarded_samples_per_optimizer_step: int
    valid_loss_scale: float

    @property
    def partial_micro_step_index(self) -> int | None:
        if self.partial_active_ranks == 0:
            return None
        return self.full_micro_steps


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def resolve_exact_global_batch_plan(
    *,
    global_batch_size: int | None,
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> ExactGlobalBatchPlan | None:
    """Resolve an exact DDP batch schedule, or preserve legacy accumulation.

    A partial micro-step is representable only as whole rank-local batches.
    Every rank still executes forward/backward on that micro-step; ranks outside
    the deterministic active set receive a zero loss scale.
    """

    if global_batch_size is None:
        return None

    global_batch_size = _require_positive_int("global_batch_size", global_batch_size)
    batch_size = _require_positive_int("batch_size", batch_size)
    gradient_accumulation_steps = _require_positive_int(
        "gradient_accumulation_steps", gradient_accumulation_steps
    )
    world_size = _require_positive_int("world_size", world_size)

    samples_per_full_micro_step = world_size * batch_size
    full_micro_steps, partial_samples = divmod(global_batch_size, samples_per_full_micro_step)
    required_accumulation_steps = full_micro_steps + int(partial_samples > 0)
    if gradient_accumulation_steps != required_accumulation_steps:
        raise ValueError(
            "Exact global_batch_size requires gradient_accumulation_steps="
            f"{required_accumulation_steps}, got {gradient_accumulation_steps}: "
            f"global_batch_size={global_batch_size}, world_size={world_size}, "
            f"batch_size={batch_size}."
        )
    if partial_samples % batch_size != 0:
        raise ValueError(
            "Exact global_batch_size cannot split a rank-local micro-batch: "
            f"partial_samples={partial_samples}, batch_size={batch_size}."
        )

    partial_active_ranks = partial_samples // batch_size
    physical_forward_samples = samples_per_full_micro_step * gradient_accumulation_steps
    return ExactGlobalBatchPlan(
        global_batch_size=global_batch_size,
        world_size=world_size,
        micro_batch_size_per_rank=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        full_micro_steps=full_micro_steps,
        partial_active_ranks=partial_active_ranks,
        physical_forward_samples_per_optimizer_step=physical_forward_samples,
        discarded_samples_per_optimizer_step=physical_forward_samples - global_batch_size,
        # DDP averages gradients over ranks, while each local loss is already
        # averaged over its B samples: (W * B / G) * (1/W) * sum(local means)
        # is exactly the mean over the G active samples.
        valid_loss_scale=world_size * batch_size / global_batch_size,
    )


def exact_global_batch_active_ranks(
    plan: ExactGlobalBatchPlan,
    *,
    optimizer_step: int,
    micro_step: int,
) -> tuple[int, ...]:
    """Return gradient-contributing ranks for one deterministic micro-step."""

    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int) or optimizer_step < 0:
        raise ValueError(f"optimizer_step must be a non-negative integer, got {optimizer_step!r}.")
    if isinstance(micro_step, bool) or not isinstance(micro_step, int):
        raise ValueError(f"micro_step must be an integer, got {micro_step!r}.")
    if not 0 <= micro_step < plan.gradient_accumulation_steps:
        raise ValueError(f"micro_step must be in [0, {plan.gradient_accumulation_steps}), got {micro_step}.")
    if micro_step < plan.full_micro_steps or plan.partial_active_ranks == 0:
        return tuple(range(plan.world_size))

    start_rank = optimizer_step % plan.world_size
    return tuple((start_rank + offset) % plan.world_size for offset in range(plan.partial_active_ranks))


def exact_global_batch_rank_loss_scale(
    plan: ExactGlobalBatchPlan,
    *,
    optimizer_step: int,
    micro_step: int,
    rank: int,
) -> float:
    """Return W*B/G for an active rank and zero for a discarded sample."""

    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < plan.world_size:
        raise ValueError(f"rank must be in [0, {plan.world_size}), got {rank!r}.")
    active_ranks = exact_global_batch_active_ranks(
        plan,
        optimizer_step=optimizer_step,
        micro_step=micro_step,
    )
    return plan.valid_loss_scale if rank in active_ranks else 0.0


def exact_global_batch_manifest(plan: ExactGlobalBatchPlan) -> dict[str, Any]:
    """Serialize the exact-gradient contract without host- or time-dependent data."""

    return {
        "version": 1,
        "mode": "exact_global_batch",
        "global_batch_size": plan.global_batch_size,
        "world_size": plan.world_size,
        "micro_batch_size_per_rank": plan.micro_batch_size_per_rank,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "full_micro_steps": plan.full_micro_steps,
        "partial_micro_step_index": plan.partial_micro_step_index,
        "partial_active_ranks": plan.partial_active_ranks,
        "physical_forward_samples_per_optimizer_step": (plan.physical_forward_samples_per_optimizer_step),
        "discarded_for_gradient_samples_per_optimizer_step": (plan.discarded_samples_per_optimizer_step),
        "valid_loss_scale": plan.valid_loss_scale,
        "valid_loss_scale_fraction": (
            f"{plan.world_size * plan.micro_batch_size_per_rank}/{plan.global_batch_size}"
        ),
        "ddp_gradient_reduction": "mean",
        "partial_rank_rotation": "consecutive ranks from optimizer_step % world_size",
        "partial_rank_rotation_period_optimizer_steps": plan.world_size,
        "all_ranks_forward_backward_every_micro_step": True,
        "sample_counter_increment_per_optimizer_step": plan.global_batch_size,
        "scheduler_steps_per_optimizer_step": 1,
        "logged_loss_reduction": "global mean over gradient-contributing samples, once per optimizer step",
    }


def write_exact_global_batch_manifest(
    output_dir: str | Path,
    plan: ExactGlobalBatchPlan,
) -> Path:
    """Atomically persist the contract and reject an incompatible resume."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / EXACT_GLOBAL_BATCH_MANIFEST_NAME
    payload = exact_global_batch_manifest(plan)
    if path.is_file():
        with open(path, encoding="utf-8") as manifest_file:
            existing_payload = json.load(manifest_file)
        if existing_payload != payload:
            raise RuntimeError(f"Existing exact global-batch manifest is incompatible: {path}")
        return path

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, ensure_ascii=False)
        manifest_file.write("\n")
    temporary_path.replace(path)
    return path


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(dict(payload), output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    temporary_path.replace(path)
    return path


def resolve_logical_physical_cuda_mapping(
    *,
    logical_cuda_index: int,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve a process-visible CUDA index back to its launcher-visible device."""

    if (
        isinstance(logical_cuda_index, bool)
        or not isinstance(logical_cuda_index, int)
        or logical_cuda_index < 0
    ):
        raise ValueError(f"logical_cuda_index must be a non-negative integer, got {logical_cuda_index!r}.")
    raw_visible_devices = environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible_devices is None or not raw_visible_devices.strip():
        visible_devices = None
        physical_cuda_device = str(logical_cuda_index)
    else:
        parsed_devices = tuple(part.strip() for part in raw_visible_devices.split(","))
        if any(not part for part in parsed_devices):
            raise ValueError(f"Malformed CUDA_VISIBLE_DEVICES={raw_visible_devices!r}.")
        if logical_cuda_index >= len(parsed_devices):
            raise ValueError(
                "Logical CUDA index is outside CUDA_VISIBLE_DEVICES: "
                f"logical={logical_cuda_index}, visible={parsed_devices}."
            )
        visible_devices = list(parsed_devices)
        physical_cuda_device = parsed_devices[logical_cuda_index]

    return {
        "logical_cuda_index": logical_cuda_index,
        "physical_cuda_device": physical_cuda_device,
        "cuda_visible_devices": visible_devices,
    }


def build_exact_global_batch_cuda_memory_rank_record(
    *,
    plan: ExactGlobalBatchPlan,
    completed_optimizer_step: int,
    global_rank: int,
    local_rank: int,
    logical_cuda_index: int,
    accelerator_device: str,
    memory_snapshot: Mapping[str, int],
    optimizer_state_tensor_count: int,
    optimizer_state_total_numel: int,
    environ: Mapping[str, str],
    hostname: str,
) -> dict[str, Any]:
    """Build one rank's post-first-Adam-step CUDA allocation record."""

    for name, value in (
        ("completed_optimizer_step", completed_optimizer_step),
        ("global_rank", global_rank),
        ("local_rank", local_rank),
        ("optimizer_state_tensor_count", optimizer_state_tensor_count),
        ("optimizer_state_total_numel", optimizer_state_total_numel),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    if completed_optimizer_step < 1:
        raise ValueError("completed_optimizer_step must be at least one.")
    if not 0 <= global_rank < plan.world_size:
        raise ValueError(f"global_rank must be in [0, {plan.world_size}), got {global_rank}.")
    if optimizer_state_tensor_count < 1 or optimizer_state_total_numel < 1:
        raise RuntimeError("Optimizer state tensors were not materialized by the completed first step.")

    required_memory_fields = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    normalized_memory: dict[str, int] = {}
    for field in required_memory_fields:
        value = memory_snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"memory_snapshot[{field!r}] must be a non-negative integer, got {value!r}.")
        normalized_memory[field] = value
    if normalized_memory["allocated_bytes"] > normalized_memory["reserved_bytes"]:
        raise ValueError("CUDA allocated_bytes cannot exceed reserved_bytes.")
    if normalized_memory["peak_allocated_bytes"] < normalized_memory["allocated_bytes"]:
        raise ValueError("CUDA peak_allocated_bytes cannot be below allocated_bytes.")
    if normalized_memory["peak_reserved_bytes"] < normalized_memory["reserved_bytes"]:
        raise ValueError("CUDA peak_reserved_bytes cannot be below reserved_bytes.")

    mapping = resolve_logical_physical_cuda_mapping(
        logical_cuda_index=logical_cuda_index,
        environ=environ,
    )
    gib = float(_GIB)
    return {
        "version": 1,
        "mode": "exact_global_batch_post_first_optimizer_step_cuda_memory",
        "completed_optimizer_step": completed_optimizer_step,
        "global_rank": global_rank,
        "local_rank": local_rank,
        "world_size": plan.world_size,
        "hostname": str(hostname),
        "accelerator_device": str(accelerator_device),
        **mapping,
        "global_batch_size": plan.global_batch_size,
        "physical_forward_samples_per_optimizer_step": (plan.physical_forward_samples_per_optimizer_step),
        "discarded_for_gradient_samples_per_optimizer_step": (plan.discarded_samples_per_optimizer_step),
        **normalized_memory,
        "allocated_gib": normalized_memory["allocated_bytes"] / gib,
        "reserved_gib": normalized_memory["reserved_bytes"] / gib,
        "peak_allocated_gib": normalized_memory["peak_allocated_bytes"] / gib,
        "peak_reserved_gib": normalized_memory["peak_reserved_bytes"] / gib,
        "optimizer_state_tensor_count": optimizer_state_tensor_count,
        "optimizer_state_total_numel": optimizer_state_total_numel,
    }


def exact_global_batch_cuda_memory_rank_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "exact_global_batch_post_adam_cuda_memory_ranks"


def exact_global_batch_cuda_memory_audit_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "exact_global_batch_post_adam_cuda_memory_audit.json"


def write_exact_global_batch_cuda_memory_rank_record(
    output_dir: str | Path,
    record: Mapping[str, Any],
) -> Path:
    global_rank = record.get("global_rank")
    if isinstance(global_rank, bool) or not isinstance(global_rank, int) or global_rank < 0:
        raise ValueError(f"Rank memory record has invalid global_rank={global_rank!r}.")
    path = exact_global_batch_cuda_memory_rank_dir(output_dir) / f"rank_{global_rank:03d}.json"
    return _write_json_atomically(path, record)


def aggregate_exact_global_batch_cuda_memory_records(
    output_dir: str | Path,
    *,
    plan: ExactGlobalBatchPlan,
    expected_completed_optimizer_step: int,
) -> Path:
    """Validate all independent rank records and atomically write rank-0's audit."""

    rank_dir = exact_global_batch_cuda_memory_rank_dir(output_dir)
    expected_paths = tuple(rank_dir / f"rank_{rank:03d}.json" for rank in range(plan.world_size))
    missing_paths = [str(path) for path in expected_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(f"Missing post-Adam CUDA memory rank records: {missing_paths}")

    records: list[dict[str, Any]] = []
    for expected_rank, path in enumerate(expected_paths):
        with open(path, encoding="utf-8") as rank_file:
            record = json.load(rank_file)
        expected_fields = {
            "global_rank": expected_rank,
            "world_size": plan.world_size,
            "completed_optimizer_step": expected_completed_optimizer_step,
            "global_batch_size": plan.global_batch_size,
            "physical_forward_samples_per_optimizer_step": (plan.physical_forward_samples_per_optimizer_step),
            "discarded_for_gradient_samples_per_optimizer_step": (plan.discarded_samples_per_optimizer_step),
        }
        mismatches = {
            field: {"expected": expected, "actual": record.get(field)}
            for field, expected in expected_fields.items()
            if record.get(field) != expected
        }
        if mismatches:
            raise RuntimeError(f"Invalid post-Adam CUDA memory record {path}: {mismatches}")
        records.append(record)

    payload = {
        "version": 1,
        "mode": "exact_global_batch_post_first_optimizer_step_cuda_memory",
        "completed_optimizer_step": expected_completed_optimizer_step,
        "world_size": plan.world_size,
        "global_batch_size": plan.global_batch_size,
        "rank_count": len(records),
        "max_allocated_bytes": max(record["allocated_bytes"] for record in records),
        "max_reserved_bytes": max(record["reserved_bytes"] for record in records),
        "max_peak_allocated_bytes": max(record["peak_allocated_bytes"] for record in records),
        "max_peak_reserved_bytes": max(record["peak_reserved_bytes"] for record in records),
        "logical_to_physical_cuda_mapping": [
            {
                "global_rank": record["global_rank"],
                "local_rank": record["local_rank"],
                "logical_cuda_index": record["logical_cuda_index"],
                "physical_cuda_device": record["physical_cuda_device"],
            }
            for record in records
        ],
        "ranks": records,
    }
    return _write_json_atomically(exact_global_batch_cuda_memory_audit_path(output_dir), payload)


def optimizer_state_tensor_summary(optimizer: Optimizer) -> tuple[int, int]:
    tensor_count = 0
    total_numel = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensor_count += 1
                total_numel += value.numel()
    return tensor_count, total_numel


def audit_first_optimizer_step_trainable_gradients(
    module: torch.nn.Module,
    output_dir: str | Path,
) -> Path:
    """Collectively prove that every registered trainable tensor received a gradient.

    DDP with ``find_unused_parameters=False`` otherwise reports a missing
    hook only when the next forward starts. Run this gate after the first
    synchronized backward and before Adam mutates or clears gradients. Every
    rank gathers the same compact missing-gradient mask, writes one
    deterministic named audit, and makes the same pass/fail decision.

    A materialized all-zero gradient is valid (for example, a zero-scaled rank
    in the exact-192 tail); only ``grad is None`` means the tensor was absent
    from that rank's accumulated autograd graph.
    """

    named_trainable = tuple(
        (name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad
    )
    if not named_trainable:
        raise RuntimeError("First-step gradient audit found no trainable parameters.")

    names = tuple(name for name, _ in named_trainable)
    numels = tuple(int(parameter.numel()) for _, parameter in named_trainable)
    device = named_trainable[0][1].device
    local_missing = torch.tensor(
        [int(parameter.grad is None) for _, parameter in named_trainable],
        dtype=torch.int32,
        device=device,
    )

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        gathered_missing = [torch.empty_like(local_missing) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_missing, local_missing)
    else:
        world_size = 1
        rank = 0
        gathered_missing = [local_missing]

    rank_records: list[dict[str, Any]] = []
    missing_name_sets: list[tuple[str, ...]] = []
    for global_rank, missing_mask in enumerate(gathered_missing):
        missing_indices = torch.nonzero(missing_mask, as_tuple=False).flatten().cpu().tolist()
        missing_names = tuple(names[index] for index in missing_indices)
        missing_name_sets.append(missing_names)
        rank_records.append(
            {
                "global_rank": global_rank,
                "missing_tensor_count": len(missing_indices),
                "missing_parameter_numel": sum(numels[index] for index in missing_indices),
                "missing_parameter_names": list(missing_names),
            }
        )

    union_missing_names = sorted({name for missing_names in missing_name_sets for name in missing_names})
    comparison_pass = not union_missing_names
    payload = {
        "version": 1,
        "mode": "full_molmo2er_first_optimizer_step_gradient_coverage",
        "comparison_pass": comparison_pass,
        "world_size": world_size,
        "trainable_parameter_tensor_count": len(named_trainable),
        "trainable_parameter_numel": sum(numels),
        "all_ranks_same_missing_set": all(
            missing_names == missing_name_sets[0] for missing_names in missing_name_sets[1:]
        ),
        "union_missing_tensor_count": len(union_missing_names),
        "union_missing_parameter_names": union_missing_names,
        "ranks": rank_records,
    }
    path = Path(output_dir) / FULL_MOLMO2ER_FIRST_STEP_GRADIENT_AUDIT_NAME
    if rank == 0:
        _write_json_atomically(path, payload)
    if distributed:
        torch.distributed.barrier()

    if not comparison_pass:
        preview = ", ".join(union_missing_names[:24])
        if len(union_missing_names) > 24:
            preview += f", ... (+{len(union_missing_names) - 24} more)"
        raise RuntimeError(
            "Full-Molmo2-ER first-step gradient coverage failed before optimizer.step: "
            f"{len(union_missing_names)} trainable tensors were absent on at least one rank "
            f"({preview}). Complete rank-specific names: {path}"
        )
    return path


def _update_sha256_with_length_prefixed_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def hash_full_molmo2er_frozen_parameters(
    policy: torch.nn.Module,
    *,
    chunk_bytes: int = 16 * _MIB,
) -> dict[str, Any]:
    """Hash frozen live parameters with bounded CPU memory and no full-tensor copy."""

    chunk_bytes = _require_positive_int("chunk_bytes", chunk_bytes)
    backend = policy.model.vlm_with_expert
    named_parameters = [
        (f"backend.{name}", parameter)
        for name, parameter in backend.named_frozen_molmo_parameters()
    ]
    named_parameters.sort(key=lambda item: item[0])
    if not named_parameters:
        raise RuntimeError("Full-Molmo2-ER frozen hash found no VLM/vision parameters.")

    digest = hashlib.sha256()
    total_numel = 0
    total_bytes = 0
    seen_names: set[str] = set()
    with torch.no_grad():
        for name, parameter in named_parameters:
            if name in seen_names:
                raise RuntimeError(f"Duplicate frozen parameter name in hash contract: {name}")
            seen_names.add(name)
            if parameter.requires_grad:
                raise RuntimeError(f"Trainable parameter entered frozen hash contract: {name}")
            if not parameter.is_contiguous():
                raise RuntimeError(
                    f"Frozen hash requires contiguous parameter storage to avoid a full copy: {name}"
                )

            element_size = parameter.element_size()
            parameter_numel = parameter.numel()
            parameter_bytes = parameter_numel * element_size
            metadata = json.dumps(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "numel": parameter_numel,
                    "nbytes": parameter_bytes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _update_sha256_with_length_prefixed_bytes(digest, metadata)

            elements_per_chunk = max(1, chunk_bytes // element_size)
            flattened = parameter.detach().view(-1)
            for start in range(0, parameter_numel, elements_per_chunk):
                chunk = flattened[start : start + elements_per_chunk]
                cpu_byte_chunk = chunk.view(torch.uint8).to(
                    device="cpu",
                    non_blocking=False,
                    copy=True,
                )
                digest.update(memoryview(cpu_byte_chunk.numpy()))
                del cpu_byte_chunk

            total_numel += parameter_numel
            total_bytes += parameter_bytes

    return {
        "algorithm": "sha256",
        "hash_scheme": "length-prefixed name/shape/dtype/numel/nbytes metadata then raw tensor bytes",
        "parameter_order": "lexicographic fully-qualified name",
        "chunk_bytes": chunk_bytes,
        "parameter_count": len(named_parameters),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def write_full_molmo2er_frozen_parameter_hash_audit(
    output_dir: str | Path,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Path:
    comparison_pass = dict(before) == dict(after)
    payload = {
        "version": 1,
        "mode": "full_molmo2er_frozen_live_parameter_before_after_hash",
        "comparison_pass": comparison_pass,
        "before": dict(before),
        "after": dict(after),
    }
    path = _write_json_atomically(
        Path(output_dir) / "full_molmo2er_frozen_parameter_hash_audit.json",
        payload,
    )
    if not comparison_pass:
        raise RuntimeError(f"Full-Molmo2-ER frozen live parameters changed during training; see {path}.")
    return path


def resolve_checkpoint_retention(environ: Mapping[str, str]) -> int | None:
    """Return an explicitly configured positive rolling-checkpoint limit."""

    raw_value = environ.get(CHECKPOINT_RETENTION_ENV)
    if raw_value is None or not str(raw_value).strip():
        return None
    normalized = str(raw_value).strip()
    if not normalized.isdigit() or int(normalized) < 1:
        raise ValueError(f"{CHECKPOINT_RETENTION_ENV} must be a positive integer, got {raw_value!r}.")
    return int(normalized)


def prune_committed_training_checkpoints(
    *,
    committed_checkpoint: Path,
    keep: int | None,
) -> tuple[Path, ...]:
    """Delete older numeric checkpoints only after ``last`` commits a new one."""

    if keep is None:
        return ()
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError(f"keep must be a positive integer or None, got {keep!r}.")

    committed_checkpoint = Path(committed_checkpoint)
    checkpoints_dir = committed_checkpoint.parent
    last_link = checkpoints_dir / "last"
    if not committed_checkpoint.is_dir() or committed_checkpoint.is_symlink():
        raise RuntimeError(f"Committed checkpoint is not a real directory: {committed_checkpoint}")
    if not last_link.is_symlink() or last_link.resolve(strict=True) != committed_checkpoint.resolve(strict=True):
        raise RuntimeError(f"Checkpoint retention requires last to commit {committed_checkpoint}.")

    numeric_checkpoints = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.name.isdigit() and len(path.name) == 6 and path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: int(path.name),
        reverse=True,
    )
    if committed_checkpoint.resolve(strict=True) not in {
        path.resolve(strict=True) for path in numeric_checkpoints[:keep]
    }:
        raise RuntimeError("The committed checkpoint is not inside the newest retained checkpoint set.")

    deleted: list[Path] = []
    checkpoints_root = checkpoints_dir.resolve(strict=True)
    for checkpoint_path in numeric_checkpoints[keep:]:
        resolved = checkpoint_path.resolve(strict=True)
        if resolved.parent != checkpoints_root or resolved == committed_checkpoint.resolve(strict=True):
            raise RuntimeError(f"Refusing unsafe checkpoint retention target: {checkpoint_path}")
        shutil.rmtree(resolved)
        deleted.append(checkpoint_path)
    return tuple(deleted)


@dataclasses.dataclass(frozen=True)
class CudaAllocatorLeaseConfig:
    """Early rank-local caching-allocator lease for the full Molmo control."""

    target_bytes: int
    chunk_bytes: int
    headroom_bytes: int


def _parse_cuda_lease_size(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
    unit_bytes: int,
    allow_zero: bool = False,
) -> int:
    raw_value = str(environ.get(name, default)).strip()
    try:
        value = Decimal(raw_value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite numeric value, got {raw_value!r}.") from exc
    if not value.is_finite() or value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} value, got {raw_value!r}.")
    byte_value = value * unit_bytes
    if byte_value != byte_value.to_integral_value():
        raise ValueError(f"{name}={raw_value!r} does not resolve to a whole number of bytes.")
    return int(byte_value)


def resolve_cuda_allocator_lease_config(
    *,
    environ: Mapping[str, str],
    vlm_backend: str | None,
    num_processes: int,
    device_type: str,
) -> CudaAllocatorLeaseConfig | None:
    """Resolve the opt-in lease without touching CUDA.

    The backend/process checks intentionally precede environment parsing so a
    malformed lease variable cannot alter any legacy or single-process run.
    """

    if vlm_backend != "molmo2_full" or num_processes <= 1:
        return None

    raw_enabled = environ.get(CUDA_ALLOCATOR_LEASE_ENABLE_ENV)
    if raw_enabled is None:
        return None
    normalized_enabled = str(raw_enabled).strip().lower()
    if normalized_enabled in {"0", "false", "no", "off"}:
        return None
    if normalized_enabled not in {"1", "true", "yes", "on"}:
        raise ValueError(
            f"{CUDA_ALLOCATOR_LEASE_ENABLE_ENV} must be an explicit boolean, got {raw_enabled!r}."
        )
    if device_type != "cuda":
        raise RuntimeError(
            "The Full-Molmo2-ER CUDA allocator lease was enabled, but Accelerator selected "
            f"device_type={device_type!r}."
        )

    config = CudaAllocatorLeaseConfig(
        target_bytes=_parse_cuda_lease_size(
            environ,
            CUDA_ALLOCATOR_LEASE_TARGET_GIB_ENV,
            default="30",
            unit_bytes=_GIB,
        ),
        chunk_bytes=_parse_cuda_lease_size(
            environ,
            CUDA_ALLOCATOR_LEASE_CHUNK_MIB_ENV,
            default="1024",
            unit_bytes=_MIB,
        ),
        headroom_bytes=_parse_cuda_lease_size(
            environ,
            CUDA_ALLOCATOR_LEASE_HEADROOM_MIB_ENV,
            default="512",
            unit_bytes=_MIB,
            allow_zero=True,
        ),
    )
    maximum_chunks = (config.target_bytes + config.chunk_bytes - 1) // config.chunk_bytes
    if maximum_chunks > _MAX_CUDA_ALLOCATOR_LEASE_CHUNKS:
        raise ValueError(
            "CUDA allocator lease would require too many allocation chunks: "
            f"target={config.target_bytes} bytes, chunk={config.chunk_bytes} bytes, "
            f"chunks={maximum_chunks}, limit={_MAX_CUDA_ALLOCATOR_LEASE_CHUNKS}."
        )
    return config


def plan_cuda_allocator_lease_chunks(
    *,
    allocated_bytes: int,
    reserved_bytes: int,
    free_bytes: int,
    target_bytes: int,
    chunk_bytes: int,
    headroom_bytes: int,
) -> tuple[int, ...]:
    """Return allocation sizes that force reserved memory to ``target_bytes``.

    Existing unused cache must be filled before the allocator asks CUDA for
    more memory. Therefore the tensors held temporarily must bring *allocated*
    memory to the target, while the driver-free feasibility check only needs
    the missing *reserved* bytes plus the configured safety headroom.
    """

    values = {
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "free_bytes": free_bytes,
        "target_bytes": target_bytes,
        "chunk_bytes": chunk_bytes,
        "headroom_bytes": headroom_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer number of bytes, got {value!r}.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if target_bytes == 0 or chunk_bytes == 0:
        raise ValueError("target_bytes and chunk_bytes must both be positive.")
    if allocated_bytes > reserved_bytes:
        raise ValueError(
            f"allocated_bytes ({allocated_bytes}) cannot exceed reserved_bytes ({reserved_bytes})."
        )
    if reserved_bytes >= target_bytes:
        return ()

    missing_driver_bytes = target_bytes - reserved_bytes
    required_free_bytes = missing_driver_bytes + headroom_bytes
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            "Insufficient rank-local CUDA memory for allocator lease: "
            f"free={free_bytes} bytes, missing_reserved={missing_driver_bytes} bytes, "
            f"required_headroom={headroom_bytes} bytes, required_free={required_free_bytes} bytes."
        )

    temporary_allocation_bytes = target_bytes - allocated_bytes
    full_chunks, remainder = divmod(temporary_allocation_bytes, chunk_bytes)
    number_of_chunks = full_chunks + int(remainder > 0)
    if number_of_chunks > _MAX_CUDA_ALLOCATOR_LEASE_CHUNKS:
        raise ValueError(
            f"CUDA allocator lease needs {number_of_chunks} chunks; "
            f"limit is {_MAX_CUDA_ALLOCATOR_LEASE_CHUNKS}."
        )
    chunks = [chunk_bytes] * full_chunks
    if remainder:
        chunks.append(remainder)
    return tuple(chunks)


def _cuda_allocator_snapshot(device: torch.device, *, label: str) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    snapshot = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    if snapshot["allocated_bytes"] > snapshot["reserved_bytes"]:
        raise RuntimeError(f"CUDA allocator audit {label}: allocated exceeds reserved: {snapshot}.")
    if snapshot["free_bytes"] > snapshot["total_bytes"]:
        raise RuntimeError(f"CUDA allocator audit {label}: free exceeds total: {snapshot}.")
    return snapshot


def _format_cuda_allocator_snapshot(snapshot: Mapping[str, int]) -> str:
    return " ".join(
        f"{name.removesuffix('_bytes')}={int(value) / _GIB:.3f}GiB"
        for name, value in snapshot.items()
    )


def reserve_full_molmo2_cuda_allocator_lease(
    *,
    accelerator: Accelerator,
    vlm_backend: str | None,
    environ: Mapping[str, str],
) -> dict[str, Any] | None:
    """Reserve rank-local CUDA cache early, then release all live tensors.

    No random operation is used and peak counters are reset after the lease, so
    this changes neither training mathematics nor reported model peaks. The
    cached blocks remain owned by this process until explicitly emptied or the
    process exits, allowing subsequent model allocations to reuse them.
    """

    device = torch.device(accelerator.device)
    config = resolve_cuda_allocator_lease_config(
        environ=environ,
        vlm_backend=vlm_backend,
        num_processes=int(accelerator.num_processes),
        device_type=device.type,
    )
    if config is None:
        return None

    rank = int(accelerator.process_index)
    local_rank = int(accelerator.local_process_index)
    prefix = f"[cuda-allocator-lease rank={rank} local_rank={local_rank} device={device}]"

    def audit_log(message: str, *args: object) -> None:
        # init_logging intentionally suppresses INFO on non-main ranks. Lease
        # provenance is rank-local, so preserve one visible line per GPU.
        if accelerator.is_main_process:
            logging.info(message, *args)
        else:
            print(message % args, flush=True)

    before = _cuda_allocator_snapshot(device, label="before")
    audit_log(
        "%s starting target=%.3fGiB chunk=%.3fGiB headroom=%.3fGiB; %s",
        prefix,
        config.target_bytes / _GIB,
        config.chunk_bytes / _GIB,
        config.headroom_bytes / _GIB,
        _format_cuda_allocator_snapshot(before),
    )
    lease_tensors: list[torch.Tensor] = []
    try:
        chunks = plan_cuda_allocator_lease_chunks(
            allocated_bytes=before["allocated_bytes"],
            reserved_bytes=before["reserved_bytes"],
            free_bytes=before["free_bytes"],
            target_bytes=config.target_bytes,
            chunk_bytes=config.chunk_bytes,
            headroom_bytes=config.headroom_bytes,
        )
        for allocation_bytes in chunks:
            lease_tensors.append(torch.empty(allocation_bytes, dtype=torch.uint8, device=device))
        held = _cuda_allocator_snapshot(device, label="held")
        expected_held_allocated = before["allocated_bytes"] + sum(chunks)
        if held["allocated_bytes"] != expected_held_allocated:
            raise RuntimeError(
                f"CUDA allocator lease live-allocation audit failed: expected "
                f"allocated={expected_held_allocated}, actual={held['allocated_bytes']}."
            )
        if held["reserved_bytes"] < config.target_bytes:
            raise RuntimeError(
                f"CUDA allocator lease missed target: reserved={held['reserved_bytes']} bytes, "
                f"target={config.target_bytes} bytes."
            )
        if held["free_bytes"] < config.headroom_bytes:
            raise RuntimeError(
                f"CUDA allocator lease consumed configured headroom: free={held['free_bytes']} bytes, "
                f"headroom={config.headroom_bytes} bytes."
            )
        audit_log("%s tensors-held audit passed; %s", prefix, _format_cuda_allocator_snapshot(held))

        lease_tensors.clear()
        retained = _cuda_allocator_snapshot(device, label="retained")
        if retained["allocated_bytes"] != before["allocated_bytes"]:
            raise RuntimeError(
                f"CUDA allocator lease leaked live tensors: baseline allocated={before['allocated_bytes']}, "
                f"retained allocated={retained['allocated_bytes']}."
            )
        if retained["reserved_bytes"] != held["reserved_bytes"]:
            raise RuntimeError(
                f"CUDA allocator released cache unexpectedly: held reserved={held['reserved_bytes']}, "
                f"retained reserved={retained['reserved_bytes']}."
            )
        if retained["reserved_bytes"] < config.target_bytes:
            raise RuntimeError(
                f"CUDA allocator retained only {retained['reserved_bytes']} bytes below "
                f"target {config.target_bytes}."
            )
        if abs(retained["free_bytes"] - held["free_bytes"]) > _CUDA_FREE_AUDIT_TOLERANCE_BYTES:
            raise RuntimeError(
                "CUDA driver-free memory changed when lease tensors were returned to cache: "
                f"held={held['free_bytes']} bytes, retained={retained['free_bytes']} bytes, "
                f"tolerance={_CUDA_FREE_AUDIT_TOLERANCE_BYTES} bytes."
            )
        audit_log("%s cache-retained audit passed; %s", prefix, _format_cuda_allocator_snapshot(retained))

        torch.cuda.reset_peak_memory_stats(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_allocated != retained["allocated_bytes"] or peak_reserved != retained["reserved_bytes"]:
            raise RuntimeError(
                f"CUDA allocator peak reset audit failed: peak_allocated={peak_allocated}, "
                f"peak_reserved={peak_reserved}, retained={retained}."
            )
        audit_log(
            "%s complete; live tensors deleted, cache retained, peaks reset "
            "(peak_allocated=%.3fGiB peak_reserved=%.3fGiB).",
            prefix,
            peak_allocated / _GIB,
            peak_reserved / _GIB,
        )
        return {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "config": dataclasses.asdict(config),
            "before": before,
            "held": held,
            "retained": retained,
        }
    except Exception as exc:
        # Dropping partial live allocations lets process teardown reclaim them;
        # never call empty_cache here because a successful partial lease must
        # remain observable in the failure audit until the fail-fast exit.
        lease_tensors.clear()
        try:
            failed = _cuda_allocator_snapshot(device, label="failure")
            failure_snapshot = _format_cuda_allocator_snapshot(failed)
        except Exception as audit_exc:
            failure_snapshot = f"snapshot-unavailable: {audit_exc!r}"
        logging.error("%s failed fast: %s; %s", prefix, exc, failure_snapshot)
        raise RuntimeError(f"{prefix} failed before model creation: {exc}") from exc


def validate_policy_camera_cli_overrides(cfg: TrainPipelineConfig) -> dict[str, Any]:
    """Fail before training if an explicit camera CLI override was not applied.

    ``--policy.path`` is handled by a second configuration parse in
    ``TrainPipelineConfig.validate``.  Camera selection is too consequential to
    trust silently: a cache may contain several views while the dataset wrapper
    consumes only the views stored in the resolved policy configuration.
    """

    provenance: dict[str, Any] = {}
    for field_name in ("camera_views", "rgb_camera_views"):
        raw_value = parser.parse_arg(f"policy.{field_name}")
        if raw_value is None:
            continue
        expected = tuple(parse_camera_views(raw_value))
        resolved_value = getattr(cfg.policy, field_name, None)
        resolved = tuple(parse_camera_views(resolved_value))
        provenance[field_name] = {
            "cli_raw": raw_value,
            "cli_parsed": list(expected),
            "resolved_raw": resolved_value,
            "resolved_parsed": list(resolved),
        }
        if resolved != expected:
            raise RuntimeError(
                f"Explicit --policy.{field_name}={raw_value!r} was not applied: "
                f"resolved cfg.policy.{field_name}={resolved_value!r}. "
                "Refusing to train with an unintended camera modality."
            )
    return provenance


def validate_policy_camera_config_matches_training_config(
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
) -> None:
    """Require the model config saved as config.json to match train_config.json."""

    for field_name in ("camera_views", "rgb_camera_views"):
        train_value = getattr(cfg.policy, field_name, None)
        model_value = getattr(policy.config, field_name, None)
        train_views = tuple(parse_camera_views(train_value))
        model_views = tuple(parse_camera_views(model_value))
        if train_views != model_views:
            raise RuntimeError(
                f"Camera configuration diverged while loading the policy: "
                f"cfg.policy.{field_name}={train_value!r}, "
                f"policy.config.{field_name}={model_value!r}. "
                "Refusing to create a checkpoint with contradictory metadata."
            )


def canonical_rgb_camera_name(name: str) -> str:
    """Map dataset/checkpoint RGB aliases to one semantic camera name.

    This is used only for compatibility validation. It intentionally does not
    rename dataset features or copy image tensors: the policy continues to read
    the exact serialized feature key, such as ``observation.images.overhead``.
    """

    name = str(name).removeprefix("observation.images.")
    aliases = {
        "overhead": "agentview",
        "overview": "agentview",
        "external": "agentview",
        "wrist": "robot0_eye_in_hand",
        "hand": "robot0_eye_in_hand",
    }
    return aliases.get(name, name)


def write_training_camera_provenance(
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    cli_provenance: dict[str, Any],
) -> Path:
    """Record the exact launch and resolved modalities beside the run."""

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_summary: dict[str, Any] | None = None
    # The active memmap wrapper reads point clouds from dataset.root. Retain a
    # compatibility fallback for older launch configs that exposed a separate
    # point_cloud_memmap_dir field.
    cache_dir = getattr(cfg, "point_cloud_memmap_dir", None) or getattr(
        cfg.dataset, "root", None
    )
    manifest_path = Path(cache_dir) / "manifest.json" if cache_dir else None
    if manifest_path is not None and manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifest_summary = {
            key: manifest.get(key)
            for key in (
                "camera_views",
                "gripper_points",
                "current_points",
                "future_points",
            )
        }

    payload = {
        "created_unix_s": time.time(),
        "hostname": os.uname().nodename,
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "policy_path": str(getattr(cfg.policy, "pretrained_path", None)),
        "cli_camera_overrides": cli_provenance,
        "resolved_train_config": {
            "camera_views": getattr(cfg.policy, "camera_views", None),
            "rgb_camera_views": getattr(cfg.policy, "rgb_camera_views", None),
        },
        "resolved_policy_config": {
            "camera_views": getattr(policy.config, "camera_views", None),
            "rgb_camera_views": getattr(policy.config, "rgb_camera_views", None),
            "effective_rgb_camera_views": list(
                getattr(policy.config, "selected_rgb_camera_views", ())
            ),
            "vlm_backend": getattr(policy.config, "vlm_backend", None),
            "image_features": sorted(getattr(policy.config, "image_features", {})),
        },
        "point_cloud_cache_manifest_path": (
            None if manifest_path is None else str(manifest_path)
        ),
        "point_cloud_cache_manifest": manifest_summary,
    }
    path = output_dir / "camera_training_provenance.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as provenance_file:
        json.dump(payload, provenance_file, indent=2, ensure_ascii=False, default=str)
    temporary_path.replace(path)
    return path


class PointCloudMemmapDataset(torch.utils.data.Dataset):
    """Inject point clouds from per-episode zarr/npy arrays into a LeRobotDataset item."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        point_cloud_dir: str | Path,
        key: str = "observation.point_cloud",
        mmap_mode: str = "r",
        camera_views: str | tuple[str, ...] | list[str] = "agentview",
        gripper_points: int = 500,
    ):
        self.dataset = dataset
        self.point_cloud_dir = Path(point_cloud_dir)
        self.dataset_root = self.point_cloud_dir.parent
        self.camera_views = parse_camera_views(camera_views)
        self.point_cloud_dirs = {
            view: (
                self.point_cloud_dir
                if view == "agentview"
                else point_cloud_dir_for_view(self.dataset_root, view)
            )
            for view in self.camera_views
        }
        self.gripper_points = int(gripper_points)
        self.key = key
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[tuple[str, int], np.ndarray] = {}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, view: str, episode_index: int) -> np.ndarray:
        cache_key = (str(view), int(episode_index))
        point_clouds = self._point_cloud_cache.get(cache_key)
        if point_clouds is None:
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dirs[str(view)],
                episode_index,
                mmap_mode=self.mmap_mode,
            )
            self._point_cloud_cache[cache_key] = point_clouds
        return point_clouds

    def _point_cloud_frame(self, episode_index: int, frame_index: int) -> np.ndarray:
        clouds = [
            np.asarray(self._episode_point_clouds(view, episode_index)[frame_index], dtype=np.float32)
            for view in self.camera_views
        ]
        seed = 1000 + int(episode_index) * 1_000_003 + int(frame_index) * 97
        return compose_point_cloud_views(
            clouds,
            gripper_points=self.gripper_points,
            seed=seed,
        )

    def __getitem__(self, idx):
        item = self.dataset[idx]
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        point_cloud = self._point_cloud_frame(episode_index, frame_index).copy()
        item[self.key] = torch.from_numpy(point_cloud).unsqueeze(0)
        return item


class WorldFlowMemmapDataset(torch.utils.data.Dataset):
    """Inject strict fixed-reference supervision for the selected World target.

    ``worldflow.current_ee_pose`` is the achieved pose at the observation
    frame. In ``world_eef_trajectory`` mode both it and the commanded future
    EEF targets are expressed directly in the complete robot-base frame.
    Legacy camera-frame datasets retain their historical behavior.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        root: str | Path,
        *,
        chunk_size: int,
        target_type: str = "legacy_eef",
        action_start_offset: int = 0,
        require_action_target_sidecar: bool = False,
        mmap_mode: str = "r",
    ):
        self.dataset = dataset
        self.root = Path(root)
        self.target_type = str(target_type)
        if self.target_type not in {"legacy_eef", "world_eef_trajectory"}:
            raise ValueError(f"Unsupported WorldFlow target_type={self.target_type!r}.")
        self.pose_dir = self.root / (
            "world_base_ee_poses"
            if self.target_type == "world_eef_trajectory"
            else "world_ee_poses"
        )
        command_target_dir = self.root / (
            "world_base_action_target_ee_poses"
            if self.target_type == "world_eef_trajectory"
            else "action_target_ee_poses"
        )
        if (
            self.target_type == "legacy_eef"
            and require_action_target_sidecar
            and not command_target_dir.is_dir()
        ):
            raise FileNotFoundError(
                "WorldFlow requires commanded action targets but the sidecar directory is missing: "
                f"{command_target_dir}. Regenerate the dataset with action_target_ee_poses or set "
                "worldflow_require_action_target_sidecar=False only for an explicitly achieved-trajectory dataset."
            )
        self.target_pose_dir = (
            command_target_dir if command_target_dir.is_dir() else self.pose_dir
        )
        self.chunk_size = int(chunk_size)
        self.action_start_offset = int(action_start_offset)
        if self.action_start_offset < 0:
            raise ValueError("WorldFlow action_start_offset must be non-negative.")
        self.mmap_mode = mmap_mode
        self._pose_cache: dict[int, np.ndarray] = {}
        self._target_pose_cache: dict[int, np.ndarray] = {}

        if not self.pose_dir.is_dir():
            raise FileNotFoundError(
                f"WorldFlow is enabled but reference-frame ee pose directory is missing: {self.pose_dir}"
            )
        if self.target_type == "world_eef_trajectory" and not command_target_dir.is_dir():
            raise FileNotFoundError(
                "Robot-base WorldFlow requires the commanded EEF trajectory sidecar: "
                f"{command_target_dir}. Camera-frame or achieved-pose fallbacks are forbidden."
            )
        if self.target_type == "world_eef_trajectory":
            base_meta_path = self.pose_dir / "meta.json"
            target_meta_path = command_target_dir / "meta.json"
            if not base_meta_path.is_file() or not target_meta_path.is_file():
                raise FileNotFoundError(
                    "Robot-base WorldFlow requires explicit coordinate metadata at "
                    f"{base_meta_path} and {target_meta_path}."
                )
            with open(base_meta_path, encoding="utf-8") as f:
                base_meta = json.load(f)
            with open(target_meta_path, encoding="utf-8") as f:
                target_meta = json.load(f)
            if (
                base_meta.get("coordinate_frame") != "robot_base"
                or target_meta.get("coordinate_frame") != "robot_base"
                or target_meta.get("target_semantics") != "commanded_eef_pose"
            ):
                raise ValueError(
                    "Robot-base WorldFlow metadata must declare robot_base coordinates and "
                    "target_semantics='commanded_eef_pose'."
                )
        if self.target_type == "legacy_eef" and self.target_pose_dir == self.pose_dir:
            logging.warning(
                "WorldFlow command-target sidecar is absent at %s; falling back to achieved future poses. "
                "The World--Ego bridge is exactly label-consistent only when action_target_ee_poses is present.",
                command_target_dir,
            )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pose_cache"] = {}
        state["_target_pose_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _load_episode_poses(
        self,
        episode_index: int,
        *,
        directory: Path,
        cache: dict[int, np.ndarray],
        description: str,
    ) -> np.ndarray:
        poses = cache.get(episode_index)
        if poses is None:
            path = directory / f"episode_{episode_index:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(f"WorldFlow {description} pose memmap file is missing: {path}")
            poses = np.load(path, mmap_mode=self.mmap_mode)
            if poses.ndim != 2 or poses.shape[-1] != 9:
                raise ValueError(f"Expected WorldFlow {description} poses shape (T,9), got {poses.shape}.")
            cache[episode_index] = poses
        return poses

    def _episode_poses(self, episode_index: int) -> np.ndarray:
        return self._load_episode_poses(
            episode_index,
            directory=self.pose_dir,
            cache=self._pose_cache,
            description="achieved current",
        )

    def _episode_target_poses(self, episode_index: int) -> np.ndarray:
        return self._load_episode_poses(
            episode_index,
            directory=self.target_pose_dir,
            cache=self._target_pose_cache,
            description="command target",
        )

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        poses = self._episode_poses(episode_index)
        episode_len = int(len(poses))
        if episode_len <= 0:
            raise ValueError(f"Worldflow episode {episode_index} is empty.")
        target_poses = self._episode_target_poses(episode_index)
        if len(target_poses) != episode_len:
            raise ValueError(
                f"WorldFlow episode {episode_index} achieved/target lengths differ: "
                f"{episode_len} != {len(target_poses)}."
            )

        current_index = min(max(frame_index, 0), episode_len - 1)
        current_pose = np.array(poses[current_index], dtype=np.float32, copy=True)
        item["worldflow.current_ee_pose"] = torch.from_numpy(
            current_pose
        )

        action = item.get("action")
        if (torch.is_tensor(action) or isinstance(action, np.ndarray)) and action.ndim >= 2:
            chunk_size = int(action.shape[0])
        else:
            chunk_size = self.chunk_size
        frame_indices = (
            frame_index
            + self.action_start_offset
            + np.arange(chunk_size, dtype=np.int64)
        )
        clamped_indices = np.clip(frame_indices, 0, episode_len - 1)
        target_key = (
            "worldflow.eef_trajectory"
            if self.target_type == "world_eef_trajectory"
            else "worldflow.ee_poses"
        )
        item[target_key] = torch.from_numpy(
            np.array(target_poses[clamped_indices], dtype=np.float32, copy=True)
        )
        target_frame_indices = frame_indices
        item["worldflow.step_is_pad"] = torch.from_numpy(
            target_frame_indices >= episode_len
        )
        return item


def _paired_pointseg_cache_contract_mismatches(
    all_view_manifest: dict[str, Any],
    primary_manifest: dict[str, Any],
    *,
    camera_view_fusion: str,
    num_views: int,
    gripper_points: int,
) -> dict[str, tuple[Any, Any]]:
    """Compare semantic cache contracts for exact primary/all-view pairing.

    ``full_union`` deliberately has a variable input length: every view keeps
    all of its scene points while only the primary gripper tail is retained.
    The primary replay therefore remains the checkpoint-native 10k cloud and
    is padded by ``song_pointseg_collate`` next to the 19.5k two-view cloud.

    ``nn_chunk_size`` only tiles exact nearest-neighbour computation.  It does
    not change the pseudo-label definition and may differ between caches made
    for different point counts.
    """

    # Cache schema versions describe the on-disk reader contract, not the
    # pseudo-label semantics that must match between paired views.  Each
    # SongPointSegCachedDataset has already rejected unsupported versions;
    # allow a compatible immutable primary cache (v11) to pair with a v12
    # multiscale cache that only adds the coarse-novelty input metadata.
    matching_fields = (
        "num_samples",
        "future_offsets",
        "temporal_offsets",
        "trajectory_mode",
        "trajectory_offset_filtering",
        "gripper_points",
        "pseudo_label_policy",
    )
    mismatches = {
        key: (all_view_manifest.get(key), primary_manifest.get(key))
        for key in matching_fields
        if all_view_manifest.get(key) != primary_manifest.get(key)
    }

    def semantic_pseudo_config(manifest: dict[str, Any]) -> Any:
        config = manifest.get("pseudo_label_config")
        if not isinstance(config, dict):
            return config
        return {key: value for key, value in config.items() if key != "nn_chunk_size"}

    all_pseudo_config = semantic_pseudo_config(all_view_manifest)
    primary_pseudo_config = semantic_pseudo_config(primary_manifest)
    if all_pseudo_config != primary_pseudo_config:
        mismatches["pseudo_label_config"] = (all_pseudo_config, primary_pseudo_config)

    for field in ("current_points", "future_points"):
        all_points = all_view_manifest.get(field)
        primary_points = primary_manifest.get(field)
        if camera_view_fusion == "full_union":
            try:
                expected_all_points = (
                    int(num_views) * (int(primary_points) - int(gripper_points))
                    + int(gripper_points)
                )
            except (TypeError, ValueError):
                expected_all_points = None
            if int(num_views) < 1 or expected_all_points != all_points:
                mismatches[field] = (all_points, primary_points)
        elif all_points != primary_points:
            mismatches[field] = (all_points, primary_points)
    return mismatches


class PointSegCacheInjectedDataset(torch.utils.data.Dataset):
    """Inject offline temporal pointseg samples into the action-training dataset."""

    pointseg_keys = (
        "observation.point_cloud",
        "pointseg.priors",
        "pointseg.labels",
        "pointseg.weights",
        "pointseg.class_scores",
        "pointseg.role_scores",
        "pointseg.foreground_score",
    )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        cache_dir: str | Path,
        *,
        point_cloud_dir: str | Path | None = None,
        strict: bool = True,
        mmap_mode: str = "r",
        camera_views: str | tuple[str, ...] | list[str] = "agentview",
        gripper_points: int = 500,
    ):
        self.dataset = dataset
        self.cache = SongPointSegCachedDataset(cache_dir)
        root_value = getattr(dataset, "root", None)
        if root_value is None:
            root_value = dataset.meta.root
        root = Path(root_value)
        self.point_cloud_dir = Path(point_cloud_dir) if point_cloud_dir is not None else root / "point_clouds"
        self.camera_views = parse_camera_views(camera_views)
        self.point_cloud_dirs = {
            view: (
                self.point_cloud_dir
                if view == "agentview"
                else point_cloud_dir_for_view(root, view)
            )
            for view in self.camera_views
        }
        self.gripper_points = int(gripper_points)
        cached_views = self.cache.manifest.get("camera_views")
        if cached_views is not None and tuple(cached_views) != self.camera_views:
            raise ValueError(
                f"PointSeg cache views {tuple(cached_views)} do not match training views {self.camera_views}."
            )
        self.strict = strict
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[tuple[str, int], np.ndarray] = {}
        if self.strict and len(self.cache) < len(self.dataset):
            raise ValueError(
                f"Song pointseg cache has {len(self.cache)} samples but action dataset has {len(self.dataset)}. "
                "Rebuild the cache without --max-samples, or set SONG_POINTSEG_CACHE_STRICT=0 for debugging."
            )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, view: str, episode_index: int) -> np.ndarray:
        cache_key = (str(view), int(episode_index))
        point_clouds = self._point_cloud_cache.get(cache_key)
        if point_clouds is None:
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dirs[str(view)],
                episode_index,
                mmap_mode=self.mmap_mode,
            )
            self._point_cloud_cache[cache_key] = point_clouds
        return point_clouds

    def _point_cloud_frame(self, episode_index: int, frame_index: int) -> np.ndarray:
        clouds = [
            np.asarray(self._episode_point_clouds(view, episode_index)[frame_index], dtype=np.float32)
            for view in self.camera_views
        ]
        seed = 1000 + int(episode_index) * 1_000_003 + int(frame_index) * 97
        return compose_point_cloud_views(
            clouds,
            gripper_points=self.gripper_points,
            seed=seed,
        )

    def _check_alignment(self, item: dict[str, Any], cache_item: dict[str, torch.Tensor], idx: int) -> None:
        if "episode_index" in item and "episode_index" in cache_item:
            item_episode = self._to_int(item["episode_index"])
            cache_episode = self._to_int(cache_item["episode_index"])
            if item_episode != cache_episode:
                raise ValueError(
                    f"Song pointseg cache is not aligned at dataset index {idx}: "
                    f"episode_index {cache_episode} != {item_episode}."
                )
        if "frame_index" in item and "frame_index" in cache_item:
            item_frame = self._to_int(item["frame_index"])
            cache_frame = self._to_int(cache_item["frame_index"])
            if item_frame != cache_frame:
                raise ValueError(
                    f"Song pointseg cache is not aligned at dataset index {idx}: "
                    f"frame_index {cache_frame} != {item_frame}."
                )

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if idx >= len(self.cache):
            if self.strict:
                raise IndexError(f"Song pointseg cache is missing dataset index {idx}.")
            return item

        cache_item = self.cache[idx]
        self._check_alignment(item, cache_item, idx)
        if "observation.point_cloud" not in cache_item and "observation.point_cloud_indices" in cache_item:
            episode_index = self._to_int(cache_item["episode_index"])
            frame_index = self._to_int(cache_item["frame_index"])
            indices = cache_item["observation.point_cloud_indices"].detach().cpu().numpy().astype(np.int64)
            point_cloud = np.asarray(
                self._point_cloud_frame(episode_index, frame_index)[indices],
                dtype=np.float32,
            ).copy()
            cache_item["observation.point_cloud"] = torch.from_numpy(point_cloud)
        for key in self.pointseg_keys:
            if key in cache_item:
                item[key] = cache_item[key]
        if "observation.point_cloud_indices" in cache_item:
            item["observation.point_cloud_indices"] = cache_item["observation.point_cloud_indices"]
        return item


class OnlinePointSegPseudoDataset(torch.utils.data.Dataset):
    """Adds temporal point-cloud fields for batch-level online pseudo-label generation."""

    pointseg_keys = (
        "pointseg.labels",
        "pointseg.weights",
        "pointseg.class_scores",
        "pointseg.foreground_score",
    )
    transient_keys = (
        "observation.point_cloud_future",
        "observation.point_cloud_future_is_pad",
        "future_is_pad",
        "future_offsets",
        "future_ee_poses",
        "pointseg_trajectory_ee_poses",
        "pointseg_trajectory_offsets",
    )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        point_cloud_dir: str | Path,
        policy_cfg: Any,
        mmap_mode: str = "r",
    ):
        self.dataset = SongTemporalPointCloudDataset(
            dataset,
            point_cloud_dir=point_cloud_dir,
            future_offsets=self._future_offsets(policy_cfg),
            current_points=self._env_int("SONG_POINTSEG_ONLINE_CURRENT_POINTS", 10_000),
            future_points=self._env_int("SONG_POINTSEG_ONLINE_FUTURE_POINTS", 10_000),
            seed=self._env_int("SONG_POINTSEG_ONLINE_SEED", 1000),
            camera_views=getattr(policy_cfg, "camera_views", "agentview"),
            gripper_points=self._env_int("SONG_POINTCLOUD_GRIPPER_POINTS", 500),
            mmap_mode=mmap_mode,
        )
        self.current_points = int(self.dataset.current_points)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(os.environ.get("SONG_POINTSEG_ONLINE_DEVICE", default_device))
        self.pseudo_config = PseudoLabelConfig(
            nn_chunk_size=self._env_int("SONG_POINTSEG_ONLINE_NN_CHUNK_SIZE", 512)
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["device"] = torch.device("cpu")
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)

    @staticmethod
    def _future_offsets(policy_cfg: Any) -> tuple[int, ...]:
        env_value = os.environ.get("SONG_POINTSEG_ONLINE_FUTURE_OFFSETS")
        if env_value:
            offsets = tuple(int(part) for part in env_value.replace(";", ",").split(",") if part.strip())
        else:
            offsets = DEFAULT_FUTURE_OFFSETS
        chunk_size = int(getattr(policy_cfg, "chunk_size", max(offsets) + 1))
        offsets = tuple(offset for offset in offsets if 0 < int(offset) < chunk_size)
        if not offsets:
            offsets = (1,)
        return offsets

    def __getitem__(self, idx):
        return dict(self.dataset[idx])

    def make_collate_fn(self):
        return OnlinePointSegBatchCollator(
            current_points=self.current_points,
            device=self.device,
            pseudo_config=self.pseudo_config,
        )


class OnlinePointSegBatchCollator:
    """Collate samples, compute Song pseudo labels once for the whole batch, and drop future fields."""

    transient_keys = OnlinePointSegPseudoDataset.transient_keys

    def __init__(self, *, current_points: int, device: torch.device, pseudo_config: PseudoLabelConfig):
        self.current_points = int(current_points)
        self.device = torch.device(device)
        self.pseudo_config = pseudo_config
        self.profile_freq = int(os.environ.get("SONG_POINTSEG_PROFILE_FREQ", "0") or 0)
        self._calls = 0

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        self._calls += 1
        t0 = time.perf_counter()
        batch = song_pointseg_collate(samples)
        t1 = time.perf_counter()
        current_pc = batch["observation.point_cloud"].to(device=self.device, dtype=torch.float32)
        future_pc = batch["observation.point_cloud_future"].to(device=self.device, dtype=torch.float32)
        future_poses = batch["future_ee_poses"].to(device=self.device, dtype=torch.float32)
        future_is_pad = batch["future_is_pad"].to(device=self.device, dtype=torch.bool)
        current_is_pad = batch.get("observation.point_cloud_is_pad")
        if torch.is_tensor(current_is_pad):
            current_is_pad = current_is_pad.to(device=self.device, dtype=torch.bool)
        future_point_is_pad = batch.get("observation.point_cloud_future_is_pad")
        if torch.is_tensor(future_point_is_pad):
            # Fixed-size zarr point clouds produce an all-False mask. Dropping it avoids a
            # per-KNN CUDA synchronization in the fast pointops path.
            future_point_is_pad = (
                future_point_is_pad.to(device=self.device, dtype=torch.bool)
                if bool(future_point_is_pad.any().item())
                else None
            )
        t2 = time.perf_counter()
        with torch.inference_mode():
            pseudo = generate_pseudo_labels(
                current_pc,
                future_pc,
                future_poses,
                future_is_pad,
                current_is_pad=current_is_pad,
                future_point_is_pad=future_point_is_pad,
                trajectory_poses=batch["pointseg_trajectory_ee_poses"].to(
                    device=self.device, dtype=torch.float32
                ),
                config=self.pseudo_config,
            )
        t3 = time.perf_counter()
        for source_key, dest_key in (
            ("labels", "pointseg.labels"),
            ("weights", "pointseg.weights"),
            ("class_scores", "pointseg.class_scores"),
            ("role_scores", "pointseg.role_scores"),
            ("foreground_score", "pointseg.foreground_score"),
        ):
            if source_key in pseudo:
                batch[dest_key] = pseudo[source_key].detach().cpu()
        for key in self.transient_keys:
            batch.pop(key, None)
        t4 = time.perf_counter()
        if self.profile_freq > 0 and self._calls % self.profile_freq == 0:
            logging.info(
                "Song pointseg online profile call=%s device=%s collate_s=%.3f to_device_s=%.3f "
                "pseudo_s=%.3f cpu_copy_s=%.3f future_mask=%s",
                self._calls,
                self.device,
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                future_point_is_pad is not None,
            )
        return batch


def maybe_wrap_pointseg_cache_dataset(dataset, cache_dir_value: str | Path | None = None, policy_cfg=None):
    def maybe_online_fallback(reason: str):
        if not bool(getattr(policy_cfg, "pointseg_enable", False)):
            logging.info(f"{reason}; pointseg is disabled, so no online pseudo labels are needed.")
            return dataset
        if os.environ.get("SONG_POINTSEG_ONLINE", "1").lower() in {"0", "false", "no"}:
            logging.info(f"{reason}; online pointseg pseudo labels are disabled by SONG_POINTSEG_ONLINE=0.")
            return dataset
        root = Path(getattr(dataset, "root", dataset.meta.root))
        point_cloud_dir = root / "point_clouds"
        if not point_cloud_dir.is_dir():
            logging.info(f"{reason}; point cloud dir not found at {point_cloud_dir}, using fallback point cloud loader.")
            return dataset
        mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
        logging.info(
            f"{reason}; computing bidirectional Song pointseg soft labels online from {point_cloud_dir}. "
            "This matches the offline cache supervision but is much slower; temporal context is supervision-only."
        )
        return OnlinePointSegPseudoDataset(
            dataset,
            point_cloud_dir=point_cloud_dir,
            policy_cfg=policy_cfg,
            mmap_mode=mmap_mode,
        )

    if cache_dir_value is None:
        cache_dir_value = ""
    cache_dir_value = str(cache_dir_value).strip()
    if not cache_dir_value or cache_dir_value.lower() in {"0", "false", "none"}:
        return maybe_online_fallback("Song pointseg cache is disabled")

    cache_dir = Path(cache_dir_value)
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return maybe_online_fallback(f"Song pointseg cache not found at {cache_dir}")

    strict = os.environ.get("SONG_POINTSEG_CACHE_STRICT", "1") != "0"
    root = Path(getattr(dataset, "root", dataset.meta.root))
    point_cloud_dir = root / "point_clouds"
    mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
    logging.info(f"Injecting Song pointseg temporal cache from {cache_dir}")
    return PointSegCacheInjectedDataset(
        dataset,
        cache_dir=cache_dir,
        point_cloud_dir=point_cloud_dir,
        strict=strict,
        mmap_mode=mmap_mode,
        camera_views=getattr(policy_cfg, "camera_views", "agentview"),
        gripper_points=int(os.environ.get("SONG_POINTCLOUD_GRIPPER_POINTS", "500")),
    )


def maybe_wrap_point_cloud_memmap_dataset(dataset, policy_cfg=None):
    if isinstance(dataset, (PointSegCacheInjectedDataset, OnlinePointSegPseudoDataset)):
        return dataset
    root = Path(getattr(dataset, "root", dataset.meta.root))
    point_cloud_dir = root / "point_clouds"
    if not point_cloud_dir.is_dir():
        return dataset
    camera_views = parse_camera_views(getattr(policy_cfg, "camera_views", "agentview"))
    for view in camera_views:
        view_dir = point_cloud_dir if view == "agentview" else point_cloud_dir_for_view(root, view)
        if not view_dir.is_dir():
            raise FileNotFoundError(f"Selected point-cloud view {view!r} is missing: {view_dir}")
    logging.info(
        "Loading point clouds with camera_views=%s and fixed model point count from %s",
        camera_views,
        root,
    )
    mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
    return PointCloudMemmapDataset(
        dataset,
        point_cloud_dir=point_cloud_dir,
        mmap_mode=mmap_mode,
        camera_views=camera_views,
        gripper_points=int(os.environ.get("SONG_POINTCLOUD_GRIPPER_POINTS", "500")),
    )


def _find_wrapped_dataset(dataset, cls):
    current = dataset
    while current is not None:
        if isinstance(current, cls):
            return current
        next_dataset = getattr(current, "dataset", None)
        if next_dataset is None or next_dataset is current:
            return None
        current = next_dataset
    return None


def make_song_train_collate_fn(dataset):
    online = _find_wrapped_dataset(dataset, OnlinePointSegPseudoDataset)
    if online is not None:
        return online.make_collate_fn()
    return song_pointseg_collate


def maybe_wrap_worldflow_dataset(dataset, policy_cfg):
    if not bool(getattr(policy_cfg, "worldflow_enable", False)):
        return dataset
    root = Path(getattr(dataset, "root", dataset.meta.root))
    mmap_mode = os.environ.get("SONG_WORLDFLOW_MMAP_MODE", os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r"))
    logging.info(f"Injecting worldflow supervision from {root}")
    return WorldFlowMemmapDataset(
        dataset,
        root=root,
        chunk_size=int(getattr(policy_cfg, "chunk_size", 32)),
        target_type=str(getattr(policy_cfg, "worldflow_target_type", "legacy_eef")),
        action_start_offset=int(getattr(policy_cfg, "action_chunk_start_offset", 0)),
        require_action_target_sidecar=bool(
            getattr(policy_cfg, "worldflow_require_action_target_sidecar", False)
        ),
        mmap_mode=mmap_mode,
    )


def visualize_res(batch, result, batch_idx=0, ood_test_sno=0, step=0, output_dir: str | Path | None = None):
    import numpy as np
    import open3d as o3d

    # ===== rot6d → rotation matrix =====
    def rot6d_to_matrix(rot6d):
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)

        return np.stack([b1, b2, b3], axis=-1)

    def create_frame(position, rot_matrix, scale=0.03):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale,
            origin=[0, 0, 0]
        )
        frame.rotate(rot_matrix, center=np.zeros(3))
        frame.translate(position)
        return frame

    geometries = []

    # ================= GT =================
    gt_action = batch['action'][batch_idx].cpu().numpy()
    gt_xyz = gt_action[:, :3]
    gt_rot6d = gt_action[:, 3:9]
    gt_rotmat = rot6d_to_matrix(gt_rot6d)

    for i in range(len(gt_xyz)):
        frame = create_frame(gt_xyz[i], gt_rotmat[i], scale=0.05)
        geometries.append(frame)

    # ================= Pred =================
    pred_action = result[batch_idx].cpu().numpy()
    pred_xyz = pred_action[:, :3]
    pred_rot6d = pred_action[:, 3:9]
    pred_rotmat = rot6d_to_matrix(pred_rot6d)

    for i in range(len(pred_xyz)):
        frame = create_frame(pred_xyz[i], pred_rotmat[i], scale=0.03)
        geometries.append(frame)

    # ================= Scene Point Cloud =================
    point_cloud_value = batch["observation.point_cloud"]
    if point_cloud_value.ndim == 4:
        cloud = point_cloud_value[batch_idx][0].cpu().numpy()
    else:
        cloud = point_cloud_value[batch_idx].cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] / 255)
    geometries.append(pcd)

    # ================= 转换并合并 =================
    all_points = []
    all_colors = []

    # 1. 添加场景点云的点和颜色
    scene_points = np.asarray(pcd.points)
    scene_colors = np.asarray(pcd.colors)
    all_points.append(scene_points)
    all_colors.append(scene_colors)

    # 2. 将每个坐标轴网格采样为点云
    for frame in geometries:
        if isinstance(frame, o3d.geometry.TriangleMesh):
            # 从网格中采样点
            frame_pcd = frame.sample_points_poisson_disk(number_of_points=100) # 每个坐标轴采样100个点
            all_points.append(np.asarray(frame_pcd.points))
            all_colors.append(np.asarray(frame_pcd.colors))

    # 3. 合并所有点和颜色
    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)

    # 4. 创建最终的点云对象
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_points)
    final_pcd.colors = o3d.utility.Vector3dVector(final_colors)


    # 5. 保存
    vis_dir = Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song")
    vis_dir = vis_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ply_save_path = vis_dir / f"step{step}_{ood_test_sno}.ply"
    if o3d.io.write_point_cloud(str(ply_save_path), final_pcd):
        print(f"合并后的点云已保存为 {ply_save_path}")
    else:
        logging.warning(f"合并后的点云保存失败: {ply_save_path}")

    # o3d.visualization.draw_geometries(geometries)


def _unwrap_policy_module(policy: PreTrainedPolicy) -> PreTrainedPolicy:
    while hasattr(policy, "module"):
        policy = policy.module
    return policy


def _is_visualization_process() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("ACCELERATE_PROCESS_INDEX") or os.environ.get("LOCAL_RANK")
    if rank is None or str(rank).strip() == "":
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


@torch.no_grad()
def save_joint_pointseg_visualization(
    policy: PreTrainedPolicy,
    batch: dict[str, torch.Tensor],
    *,
    step: int,
    output_dir: str | Path | None = None,
    tag: str = "train",
    threshold: float = 0.5,
    max_items: int = 2,
) -> None:
    """Save foreground/background masks produced by the joint pointseg branch."""
    if not _is_visualization_process():
        return

    raw_policy = _unwrap_policy_module(policy)
    model = getattr(raw_policy, "model", None)
    conditioner = getattr(model, "pointseg_conditioner", None)
    if conditioner is None:
        return

    point_cloud_payloads, _ = raw_policy.prepare_point_clouds(batch)
    payload = point_cloud_payloads[0]
    if not isinstance(payload, dict):
        return

    conditioned = conditioner(payload)
    point_cloud = payload["point_cloud"].detach().float().cpu().numpy()
    operation_prob = conditioned["operation_prob"].detach().float().cpu().numpy()
    selection_scores = conditioned["pointseg_selection_scores"].detach().float().cpu().numpy()
    point_is_pad = payload.get("point_is_pad")
    if torch.is_tensor(point_is_pad):
        point_is_pad = point_is_pad.detach().bool().cpu().numpy()
    vis_dir = Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song")
    vis_dir = vis_dir / "visualizations" / "pointseg"
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_without_split = 0
    for batch_idx in range(point_cloud.shape[0]):
        valid = ~point_is_pad[batch_idx] if point_is_pad is not None else np.ones(
            point_cloud.shape[1], dtype=bool
        )
        if not np.any(valid):
            skipped_without_split += 1
            continue

        current_point_cloud = point_cloud[batch_idx][valid]
        probs = operation_prob[batch_idx][valid]
        scores = selection_scores[batch_idx][valid]
        n_points = probs.shape[0]
        labels_threshold = (probs >= threshold).astype(np.int64)
        threshold_foreground_count = int(np.count_nonzero(labels_threshold == ROLE_FOREGROUND))
        if threshold_foreground_count == 0 or threshold_foreground_count == n_points:
            skipped_without_split += 1
            continue

        foreground_count = min(
            n_points,
            conditioner._target_count(n_points, conditioner.foreground_ratio, conditioner.min_foreground_points),
        )
        if foreground_count >= n_points:
            skipped_without_split += 1
            continue

        labels_topk = np.zeros(n_points, dtype=np.int64)
        topk_idx = np.argpartition(-scores, foreground_count - 1)[:foreground_count]
        labels_topk[topk_idx] = ROLE_FOREGROUND

        stem = f"{tag}_step{step}_b{batch_idx}"
        write_role_ply(vis_dir / f"{stem}_thr{threshold:.2f}.ply", current_point_cloud, labels_threshold, probs)
        write_role_ply(vis_dir / f"{stem}_topk.ply", current_point_cloud, labels_topk, probs)
        np.savez_compressed(
            vis_dir / f"{stem}.npz",
            point_cloud=current_point_cloud,
            operation_prob=probs,
            selection_scores=scores,
            labels_threshold=labels_threshold,
            labels_topk=labels_topk,
            foreground_count=np.asarray(foreground_count, dtype=np.int64),
            threshold=np.asarray(threshold, dtype=np.float32),
        )
        saved += 1
        if saved >= max_items:
            break

    if saved:
        logging.info(f"Joint pointseg visualization saved to {vis_dir} ({tag}, step {step}, {saved} item(s))")
    elif skipped_without_split:
        logging.info(
            "Skipped joint pointseg visualization (%s, step %s): no prediction contained both foreground and background.",
            tag,
            step,
        )


def ood_case_inference(
    policy,
    preprocessor,
    postprocessor,
    batch,
    step,
    output_dir: str | Path | None = None,
    ood_num_points: int = 10000,
    ood_tasks: dict[int, str] | list[str] | tuple[str, ...] | None = None,
):
    ######OOD task may differ from the training batch, so rebuild language tokens with processor.
    if not _is_visualization_process():
        return []

    import open3d as o3d
    ood_num_points = int(os.environ.get("SONG_OOD_NUM_POINTS", str(ood_num_points)))
    if ood_num_points <= 0:
        raise ValueError(f"ood_num_points should be positive, got {ood_num_points}.")

    def clone_first_batch_item(src: dict[str, Any]) -> dict[str, Any]:
        cloned = {}
        pc = src.get("observation.point_cloud")
        batch_size = int(pc.shape[0]) if torch.is_tensor(pc) and pc.ndim >= 3 else 1
        for key, value in src.items():
            if torch.is_tensor(value):
                if value.ndim > 0 and int(value.shape[0]) == batch_size:
                    cloned[key] = value[:1].clone()
                else:
                    cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = [value[0]] if len(value) == batch_size and batch_size > 0 else list(value)
            elif isinstance(value, tuple):
                cloned[key] = (value[0],) if len(value) == batch_size and batch_size > 0 else tuple(value)
            elif isinstance(value, dict):
                cloned[key] = dict(value)
            else:
                cloned[key] = value
        return cloned

    def random_repeat_sample_points(xyzrgb: np.ndarray, M: int, rng: np.random.Generator):
        N = xyzrgb.shape[0]
        if N == 0:
            return xyzrgb
        if N >= M:
            idx = rng.choice(N, M, replace=False)
            return xyzrgb[idx]
        extra = rng.choice(N, M - N, replace=True)
        return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)

    def load_ply_xyzrgb(path: Path) -> np.ndarray | None:
        if not path.exists():
            logging.warning(f"OOD ply file is missing: {path}")
            return None
        scene_pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(scene_pcd.points, dtype=np.float32)
        if points.size == 0:
            logging.warning(f"OOD ply file has no points: {path}")
            return None
        colors = np.asarray(scene_pcd.colors, dtype=np.float32)
        if colors.shape != points.shape:
            colors = np.zeros_like(points, dtype=np.float32)
        elif colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0.0, 255.0)
        return np.concatenate((points, colors), axis=1).astype(np.float32, copy=False)

    def load_ood_task(sno: int) -> str:
        if isinstance(ood_tasks, dict) and sno in ood_tasks:
            task = ood_tasks[sno]
        elif isinstance(ood_tasks, (list, tuple)) and 0 <= sno - 1 < len(ood_tasks):
            task = ood_tasks[sno - 1]
        else:
            task_path = Path(f"/home/liusong/temp/ood_test_new{sno}.txt")
            if task_path.exists():
                task = task_path.read_text(encoding="utf-8").strip()
            else:
                task = os.environ.get("SONG_OOD_TASK", 'Place the Red Cube on the Blue Cube\n')
        task = str(task).strip()
        return task if task.endswith("\n") else f"{task}\n"

    def set_ood_point_cloud(dst_batch: dict[str, Any], scene_tensor: torch.Tensor) -> None:
        point_cloud_value = dst_batch["observation.point_cloud"]
        if point_cloud_value.ndim == 4:
            dst_batch["observation.point_cloud"] = scene_tensor.unsqueeze(0).unsqueeze(0)
            dst_batch["observation.point_cloud_is_pad"] = torch.zeros(
                1, 1, scene_tensor.shape[0], dtype=torch.bool, device=scene_tensor.device
            )
        elif point_cloud_value.ndim == 3:
            dst_batch["observation.point_cloud"] = scene_tensor.unsqueeze(0)
            dst_batch["observation.point_cloud_is_pad"] = torch.zeros(
                1, scene_tensor.shape[0], dtype=torch.bool, device=scene_tensor.device
            )
        elif point_cloud_value.ndim == 2:
            dst_batch["observation.point_cloud"] = scene_tensor
            dst_batch["observation.point_cloud_is_pad"] = torch.zeros(
                scene_tensor.shape[0], dtype=torch.bool, device=scene_tensor.device
            )
        else:
            raise ValueError(f"Expected observation.point_cloud ndim 2/3/4, got {point_cloud_value.shape}")

    def remove_stale_pointseg_fields(dst_batch: dict[str, Any]) -> None:
        for key in list(dst_batch):
            if key.startswith("pointseg."):
                del dst_batch[key]

    def remove_language_token_fields(dst_batch: dict[str, Any]) -> None:
        dst_batch.pop("observation.language.tokens", None)
        dst_batch.pop("observation.language.attention_mask", None)

    def make_identity_pose_action_like(action: torch.Tensor) -> torch.Tensor:
        identity = torch.zeros_like(action)
        if identity.shape[-1] < 9:
            raise ValueError(f"Expected pose9 action with last dim >= 9, got {tuple(identity.shape)}")
        identity[..., 3] = 1.0
        identity[..., 7] = 1.0
        if action.shape[-1] > 9:
            identity[..., 9:] = action[..., 9:]
        return identity

    result_dir = Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song")
    result_dir = result_dir / "visualizations" / "ood"
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_policy = _unwrap_policy_module(policy)
    was_training = raw_policy.training
    results = []
    try:
        ood_test_sno = list(range(1,7))
        for sno in ood_test_sno:
            ply_path = Path(f"/home/liusong/temp/ood_test_new{sno}.ply")
            scene_point_cloud = load_ply_xyzrgb(ply_path)
            if scene_point_cloud is None:
                continue

            ood_batch = clone_first_batch_item(batch)
            if torch.is_tensor(ood_batch.get("action")):
                ood_batch["action"] = make_identity_pose_action_like(ood_batch["action"])
            ood_batch["task"] = [load_ood_task(sno)]

            point_cloud_value = ood_batch["observation.point_cloud"]
            rng = np.random.default_rng(1000 + int(step) * 31 + sno)
            scene_point_cloud = random_repeat_sample_points(scene_point_cloud, int(ood_num_points), rng)
            scene_tensor = torch.tensor(scene_point_cloud, device=point_cloud_value.device, dtype=point_cloud_value.dtype)
            set_ood_point_cloud(ood_batch, scene_tensor)
            remove_stale_pointseg_fields(ood_batch)
            remove_language_token_fields(ood_batch)

            model_batch = preprocessor(ood_batch)
            remove_stale_pointseg_fields(model_batch)

            action_chunk = raw_policy.predict_action_chunk(model_batch)
            action_chunk = postprocessor(action_chunk)
            visualize_res(ood_batch, action_chunk, ood_test_sno=sno, step=step, output_dir=output_dir)
            save_joint_pointseg_visualization(
                raw_policy,
                model_batch,
                step=step,
                output_dir=output_dir,
                tag=f"ood{sno}",
                max_items=1,
            )
            npz_path = result_dir / f"step{step}_ood{sno}.npz"
            np.savez_compressed(
                npz_path,
                source_ply=np.asarray(str(ply_path)),
                task=np.asarray(ood_batch["task"][0]),
                point_cloud=scene_point_cloud,
                action=action_chunk[0].detach().cpu().numpy(),
            )
            results.append(
                {
                    "sno": sno,
                    "source_ply": str(ply_path),
                    "result_npz": str(npz_path),
                    "merged_ply": str(Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song") / "visualizations" / f"step{step}_{sno}.ply"),
                }
            )
    finally:
        raw_policy.train(was_training)
    return results



def random_repeat_sample_points(xyzrgb: np.ndarray, M: int):
    N = xyzrgb.shape[0]
    if N == 0:
        return xyzrgb
    if N >= M:
        idx = np.random.choice(N, M, replace=False)
        return xyzrgb[idx]
    else:
        extra = np.random.choice(N, M - N, replace=True)
        return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)  
def count_parameters(module: torch.nn.Module, only_trainable: bool = False) -> int:
    skipped = 0
    total = 0
    for p in module.parameters():
        if only_trainable and not p.requires_grad:
            continue
        if getattr(p, "is_uninitialized", False):
            skipped += 1
            continue
        try:
            total += p.numel()
        except (ValueError, RuntimeError):
            skipped += 1
    if skipped > 0:
        logging.warning(
            f"Skipped {skipped} uninitialized parameters while counting {module.__class__.__name__}. "
            "Lazy modules will initialize on first forward."
        )
    return total


def audit_molmo2er_pointonly_policy(policy: torch.nn.Module) -> dict[str, int]:
    """Fail before optimizer/DDP if the registered 3B control has drifted."""

    config = getattr(policy, "config", None)
    if getattr(config, "vlm_backend", None) != "molmo2_text":
        return {}

    expected_total = 3_137_049_278
    expected_trainable = 930_980_030
    total = count_parameters(policy)
    trainable = count_parameters(policy, only_trainable=True)
    if (total, trainable) != (expected_total, expected_trainable):
        raise RuntimeError(
            "Molmo2-ER point-only parameter contract drifted before training: "
            f"expected total/trainable={expected_total}/{expected_trainable}, "
            f"actual={total}/{trainable}."
        )

    trainable_prefixes = (
        "model.vlm_with_expert.lm_expert.",
        "model.pointseg_conditioner.",
        "model.pointseg_object_proj.",
        "model.pointseg_background_proj.",
        "model.point_action_fusion.",
        "model.action_in_proj.",
        "model.action_out_proj.",
        "model.action_time_mlp_in.",
        "model.action_time_mlp_out.",
    )
    unexpected_trainable = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and not name.startswith(trainable_prefixes)
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Molmo2-ER point-only found trainable parameters outside the v0.4.3 allowlist: "
            f"{unexpected_trainable[:12]}"
        )

    backend = policy.model.vlm_with_expert
    if any(parameter.requires_grad for parameter in backend.vlm.parameters()):
        raise RuntimeError("The retained Molmo text backbone is not completely frozen.")
    if any(not parameter.requires_grad for parameter in backend.lm_expert.parameters()):
        raise RuntimeError("The Molmo Action Expert contains unexpectedly frozen parameters.")
    if getattr(backend, "scale_input_embeddings", None) is not False:
        raise RuntimeError("Native Molmo embeddings must not be multiplied by sqrt(hidden_size).")
    if any(
        marker in name
        for name, _ in backend.named_parameters()
        for marker in ("vision_backbone", "vision_model", "lm_head")
    ):
        raise RuntimeError("The point-only Molmo backend instantiated a forbidden vision/LM-head parameter.")

    return {"total_parameters": total, "trainable_parameters": trainable}


def audit_full_molmo2er_policy(policy: torch.nn.Module) -> dict[str, int]:
    """Fail before optimizer/DDP if the frozen Full-Molmo2-ER contract drifts."""

    config = getattr(policy, "config", None)
    if getattr(config, "vlm_backend", None) != "molmo2_full":
        return {}

    if getattr(config, "molmo_inference_only", None) is not False:
        raise RuntimeError(
            "Native-readout Full-Molmo2-ER requires molmo_inference_only=false so FG/BG "
            "input gradients can traverse the frozen decoder."
        )

    worldflow_enabled = bool(getattr(config, "worldflow_enable", False))
    lora_enabled = bool(getattr(config, "molmo_lora_enable", False))
    lora_rank = int(getattr(config, "molmo_lora_rank", 8))
    expected_budget = expected_full_molmo2er_parameter_budget(
        worldflow_enable=worldflow_enabled,
        molmo_lora_enable=lora_enabled,
        molmo_lora_rank=lora_rank,
    )
    expected_total = expected_budget["total"]
    expected_trainable = expected_budget["trainable"]
    total = count_parameters(policy)
    trainable = count_parameters(policy, only_trainable=True)
    if (total, trainable) != (expected_total, expected_trainable):
        raise RuntimeError(
            "Full-Molmo2-ER parameter contract drifted before training: "
            f"expected total/trainable={expected_total}/{expected_trainable}, "
            f"actual={total}/{trainable}."
        )

    trainable_prefixes = full_molmo2er_trainable_parameter_prefixes(
        worldflow_enable=worldflow_enabled
    )
    backend = policy.model.vlm_with_expert
    lora_named_parameters = dict(backend.named_molmo_lora_parameters())
    lora_parameter_ids = {id(parameter) for parameter in lora_named_parameters.values()}
    unexpected_trainable = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
        and not name.startswith(trainable_prefixes)
        and id(parameter) not in lora_parameter_ids
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Full-Molmo2-ER found trainable parameters outside the WEP-VLA allowlist: "
            f"{unexpected_trainable[:12]}"
        )

    architecture = backend.architecture_contract
    native_readout_contract = {
        "full_molmo_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
        "vlm_execution": (
            "frozen_base_with_lora_and_prefix_input_autograd"
            if lora_enabled
            else "frozen_parameters_with_prefix_input_autograd"
        ),
        "per_layer_memory": "official_native_stream_plus_evolving_fg_bg_readouts",
        "fg_bg_location": "post_native_trainable_prefix_readout_block",
        "action_location": "expert_suffix_only",
        "text_prefix_autograd_preserved": True,
        "native_attention": "official_causal_plus_bidirectional_image",
        "native_rope_positions": "official_unchanged",
        "native_reads_scene": False,
        "scene_reads_native": True,
        "scene_attention": "bidirectional_readout_block",
        "action_even_reads": "native_scene_action",
        "action_odd_reads": "native_scene",
        "gradient_checkpointing": bool(
            getattr(config, "molmo_gradient_checkpointing", True)
        ),
        "gradient_checkpointing_layers_per_segment": int(
            getattr(config, "molmo_gradient_checkpointing_layers_per_segment", 1)
        ),
        "molmo_lora_enable": lora_enabled,
        "molmo_lora_rank": lora_rank,
        "molmo_lora_alpha": float(getattr(config, "molmo_lora_alpha", 8.0)),
        "molmo_lora_dropout": float(getattr(config, "molmo_lora_dropout", 0.0)),
        "molmo_lora_target_modules": ("att_proj", "attn_out"),
        "molmo_lora_module_count": 36 + 35 if lora_enabled else 0,
        "molmo_lora_parameter_count": int(expected_budget.get("molmo_lora", 0)),
    }
    drift = {
        key: {"expected": expected, "actual": architecture.get(key)}
        for key, expected in native_readout_contract.items()
        if architecture.get(key) != expected
    }
    if drift or getattr(backend, "inference_only_vlm", None) is not False:
        raise RuntimeError(f"Full-Molmo2-ER native-readout execution contract drifted: {drift}.")
    if len(backend.vlm.blocks) != 36 or len(backend.lm_expert.layers) != 36:
        raise RuntimeError("Full-Molmo2-ER requires exactly 36 VLM and 36 Expert layers.")
    if not hasattr(backend, "vision_backbone"):
        raise RuntimeError("Full-Molmo2-ER did not instantiate its native vision backbone.")
    frozen_named_parameters = dict(backend.named_frozen_molmo_parameters())
    if any(parameter.requires_grad for parameter in frozen_named_parameters.values()):
        raise RuntimeError("Molmo ViT/connector/text base parameters must all be frozen.")
    # Two tensors (A/B) for all 36 att_proj modules and the 35 reachable
    # attn_out modules. The final VLM output is not consumed by the V3 loss.
    expected_lora_tensors = 2 * (36 + 35) if lora_enabled else 0
    if len(lora_named_parameters) != expected_lora_tensors:
        raise RuntimeError(
            "Full-Molmo2-ER LoRA tensor contract drifted: "
            f"expected={expected_lora_tensors}, actual={len(lora_named_parameters)}."
        )
    if any(
        not is_full_molmo2er_lora_policy_parameter_name(f"model.vlm_with_expert.{name}")
        for name in lora_named_parameters
    ):
        raise RuntimeError("Molmo V3 LoRA was registered outside the exact attention allowlist.")
    if any(not parameter.requires_grad for parameter in lora_named_parameters.values()):
        raise RuntimeError("Every registered Molmo V3 LoRA parameter must remain trainable.")
    if any(not parameter.requires_grad for parameter in backend.lm_expert.parameters()):
        raise RuntimeError("The Full-Molmo2-ER Action Expert contains unexpectedly frozen parameters.")
    if hasattr(backend, "lm_head") or any("lm_head" in name for name, _ in backend.named_parameters()):
        raise RuntimeError("The unused Molmo lm_head must be physically absent.")
    if getattr(backend, "scale_input_embeddings", None) is not False:
        raise RuntimeError("Native Molmo embeddings must not be multiplied by sqrt(hidden_size).")

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "molmo_lora_parameters": int(expected_budget.get("molmo_lora", 0)),
        "molmo_lora_parameter_tensors": len(lora_named_parameters),
    }


def write_full_molmo2er_parameter_audit(
    output_dir: str | Path,
    policy: torch.nn.Module,
    audit: Mapping[str, int],
) -> Path:
    """Atomically persist the actual pre-optimizer/pre-DDP parameter audit."""

    backend = policy.model.vlm_with_expert
    total = int(audit["total_parameters"])
    trainable = int(audit["trainable_parameters"])
    vision_backbone_present = hasattr(backend, "vision_backbone")
    molmo_base_frozen = vision_backbone_present and not any(
        parameter.requires_grad for _, parameter in backend.named_frozen_molmo_parameters()
    )
    lora_named_parameters = dict(backend.named_molmo_lora_parameters())
    lora_enabled = bool(getattr(policy.config, "molmo_lora_enable", False))
    molmo_lora_trainable = bool(lora_named_parameters) and all(
        parameter.requires_grad for parameter in lora_named_parameters.values()
    )
    payload = {
        "version": 6 if lora_enabled else 5,
        "backend": "molmo2_full",
        "full_molmo_topology": FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "worldflow_enabled": bool(getattr(policy.config, "worldflow_enable", False)),
        "vlm_layers": len(backend.vlm.blocks),
        "molmo_inference_only": bool(getattr(policy.config, "molmo_inference_only", False)),
        "vlm_execution": backend.architecture_contract.get("vlm_execution"),
        "per_layer_memory": backend.architecture_contract.get("per_layer_memory"),
        "fg_bg_location": backend.architecture_contract.get("fg_bg_location"),
        "gradient_checkpointing": backend.architecture_contract.get(
            "gradient_checkpointing"
        ),
        "gradient_checkpointing_layers_per_segment": backend.architecture_contract.get(
            "gradient_checkpointing_layers_per_segment"
        ),
        "distributed_strategy": "DDP",
        "expert_layers": len(backend.lm_expert.layers),
        "vision_backbone_present": vision_backbone_present,
        # Preserve the V3 payload byte-for-byte when LoRA is disabled, so an
        # existing V3 output directory remains resumable on this branch.
        "molmo_frozen": molmo_base_frozen and not lora_enabled,
        "trainable_allowlist_pass": True,
    }
    if lora_enabled:
        payload.update(
            {
                "molmo_base_frozen": molmo_base_frozen,
                "molmo_lora_enable": True,
                "molmo_lora_rank": int(getattr(policy.config, "molmo_lora_rank", 8)),
                "molmo_lora_alpha": float(getattr(policy.config, "molmo_lora_alpha", 8.0)),
                "molmo_lora_dropout": float(getattr(policy.config, "molmo_lora_dropout", 0.0)),
                "molmo_lora_lr_multiplier": float(
                    getattr(policy.config, "molmo_lora_lr_multiplier", 0.1)
                ),
                "molmo_lora_parameters": int(audit.get("molmo_lora_parameters", 0)),
                "molmo_lora_parameter_tensors": int(
                    audit.get("molmo_lora_parameter_tensors", 0)
                ),
                "molmo_lora_trainable": molmo_lora_trainable,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "full_molmo2er_parameter_audit.json"
    if path.is_file():
        with open(path, encoding="utf-8") as audit_file:
            existing_payload = json.load(audit_file)
        if existing_payload != payload:
            raise RuntimeError(f"Existing Full-Molmo2-ER parameter audit is incompatible: {path}")
        return path

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as audit_file:
        json.dump(payload, audit_file, indent=2, ensure_ascii=False)
        audit_file.write("\n")
    temporary_path.replace(path)
    return path


def ensure_ddp_parameters_initialized(module: torch.nn.Module, accelerator: Accelerator) -> None:
    """Fail before Accelerator/DDP wrapping and report the exact lazy parameters."""

    uninitialized = []
    for name, parameter in module.named_parameters():
        if isinstance(parameter, torch.nn.parameter.UninitializedParameter) or getattr(
            parameter, "is_uninitialized", False
        ):
            uninitialized.append(name)
    if not uninitialized:
        return

    details = ", ".join(uninitialized)
    message = (
        "Policy still contains uninitialized parameters before distributed wrapping: "
        f"{details}. Replace Lazy modules with explicit input dimensions or initialize them before "
        "creating the optimizer and calling accelerator.prepare()."
    )
    if accelerator.num_processes > 1:
        raise RuntimeError(message)
    logging.warning(message)


def make_song_training_ddp_kwargs(vlm_backend: str | None) -> DistributedDataParallelKwargs:
    """Build DDP kwargs while keeping legacy-policy behavior unchanged.

    Full-Molmo2-ER has about 3.47 GiB of trainable gradients per rank.  Making
    those gradients views into the existing all-reduce buckets avoids a second
    allocation of the same size after DDP's first iteration. The Full graph is
    fixed, so disabling unused-parameter discovery avoids traversing roughly
    2.1B trainable parameters every micro-step. ``static_graph=True`` is not
    used because PyTorch 2.8's reducer is incompatible with the ``no_sync``
    gradient-accumulation pattern used here. Other backends retain the previous
    dynamic-graph behavior.
    """

    full_molmo2er = vlm_backend == "molmo2_full"
    return DistributedDataParallelKwargs(
        find_unused_parameters=not full_molmo2er,
        gradient_as_bucket_view=full_molmo2er,
        static_graph=False,
    )


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
    loss_scale: float = 1.0,
    perform_optimizer_step: bool = True,
    record_loss: bool = True,
    require_per_sample_mean: bool = False,
    audit_first_step_gradients: bool = False,
    gradient_audit_output_dir: str | Path | None = None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        elif require_per_sample_mean:
            # Exact global-batch scaling is defined over equal-weight samples,
            # even when episode-tail action padding differs within B>1.
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            if per_sample_loss.ndim != 1:
                raise ValueError(
                    f"Expected per-sample loss shape [B], got {per_sample_loss.shape}."
                )
            loss = per_sample_loss.mean()
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Normalize the logging contract across policies (including RA-BC): this
    # is the exact scalar whose scaled gradient is accumulated below.
    output_dict["loss"] = loss.detach().item()

    finite_loss = torch.isfinite(loss.detach()).to(device=loss.device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Every rank must make the same backward/no-backward decision.  A
        # rank-local early return would leave its peers blocked forever in a
        # DDP gradient collective.
        torch.distributed.all_reduce(finite_loss, op=torch.distributed.ReduceOp.MIN)
    if not bool(finite_loss.item()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite loss detected on at least one distributed rank.")

    # Use accelerator's backward method
    # Scale each micro-batch contribution so accumulated gradients equal the
    # mean over the effective batch.  The unscaled loss is retained for logs.
    accelerator.backward(loss * float(loss_scale))

    if not perform_optimizer_step:
        if record_loss:
            train_metrics.loss = loss.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
        return train_metrics, output_dict

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    finite_grad = torch.isfinite(torch.as_tensor(grad_norm, device=loss.device)).to(torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite_grad, op=torch.distributed.ReduceOp.MIN)
    if not bool(finite_grad.item()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradient norm detected on at least one distributed rank.")

    if audit_first_step_gradients:
        if gradient_audit_output_dir is None:
            raise ValueError("gradient_audit_output_dir is required for the first-step gradient audit.")
        gradient_audit_path = audit_first_optimizer_step_trainable_gradients(
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
            gradient_audit_output_dir,
        )
        if accelerator.is_main_process:
            logging.info(
                "Full-Molmo2-ER first-step trainable gradient audit passed: %s",
                gradient_audit_path,
            )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    if record_loss:
        train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict




@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()
    camera_cli_provenance = validate_policy_camera_cli_overrides(cfg)

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        ddp_kwargs = make_song_training_ddp_kwargs(getattr(cfg.policy, "vlm_backend", None))
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    if cfg.diagnostic_repeat_first_batch and accelerator.num_processes != 1:
        raise ValueError(
            "diagnostic_repeat_first_batch is an exact single-process capacity test; "
            f"got {accelerator.num_processes} processes. Run independent LR arms on separate GPUs."
        )

    if getattr(cfg.policy, "vlm_backend", None) in {"molmo2_text", "molmo2_full"}:
        # safetensors treats the ambiguous string "cuda" as cuda:0 even when
        # Accelerate has selected another local rank.  A full policy warm-start
        # would therefore make all ranks stage the 3B checkpoint on GPU 0.
        # Pin checkpoint construction/loading to the rank-local device before
        # make_policy calls from_pretrained(strict=True).
        cfg.policy.device = str(accelerator.device)

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process
    # Prevent rank 0 from creating output artifacts while peers are still in cfg.validate().
    accelerator.wait_for_everyone()

    exact_global_batch_plan = resolve_exact_global_batch_plan(
        global_batch_size=cfg.global_batch_size,
        batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        world_size=accelerator.num_processes,
    )
    if exact_global_batch_plan is not None and is_main_process:
        exact_batch_manifest_path = write_exact_global_batch_manifest(
            cfg.output_dir,
            exact_global_batch_plan,
        )
        logging.info(
            "Exact global-batch contract: gradients=%s, physical_forwards=%s, "
            "discarded_for_gradient=%s, loss_scale=%s, manifest=%s",
            exact_global_batch_plan.global_batch_size,
            exact_global_batch_plan.physical_forward_samples_per_optimizer_step,
            exact_global_batch_plan.discarded_samples_per_optimizer_step,
            exact_global_batch_plan.valid_loss_scale,
            exact_batch_manifest_path,
        )
    accelerator.wait_for_everyone()

    if getattr(cfg.policy, "vlm_backend", None) in {"molmo2_text", "molmo2_full"}:
        if accelerator.mixed_precision != "no":
            raise ValueError(
                "The locked Molmo2-ER control requires Accelerate mixed_precision='no'; "
                f"got {accelerator.mixed_precision!r}."
            )
        if cfg.peft is not None:
            raise ValueError("The locked Molmo2-ER control does not permit a PEFT wrapper.")

    # Claim each rank-local GPU before dataset/model construction. This is an
    # explicit full-Molmo multi-GPU opt-in and is a no-op for every legacy
    # backend. Temporary tensors are deleted inside the helper; only reusable
    # caching-allocator reservations remain.
    reserve_full_molmo2_cuda_allocator_lease(
        accelerator=accelerator,
        vlm_backend=getattr(cfg.policy, "vlm_backend", None),
        environ=os.environ,
    )

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
        dataset = maybe_wrap_point_cloud_memmap_dataset(dataset, cfg.policy)
        dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)
        dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
        dataset = maybe_wrap_point_cloud_memmap_dataset(dataset, cfg.policy)
        dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    validate_policy_camera_config_matches_training_config(cfg, policy)
    molmo_audit = audit_molmo2er_pointonly_policy(policy)
    if molmo_audit and is_main_process:
        logging.info(
            "Molmo2-ER point-only pre-DDP audit passed: total=%s, trainable=%s",
            molmo_audit["total_parameters"],
            molmo_audit["trainable_parameters"],
        )
    full_molmo_audit = audit_full_molmo2er_policy(policy)
    if full_molmo_audit and is_main_process:
        full_molmo_audit_path = write_full_molmo2er_parameter_audit(
            cfg.output_dir,
            policy,
            full_molmo_audit,
        )
        logging.info(
            "Full-Molmo2-ER pre-DDP audit passed: total=%s, trainable=%s, frozen=%s, manifest=%s",
            full_molmo_audit["total_parameters"],
            full_molmo_audit["trainable_parameters"],
            full_molmo_audit["total_parameters"] - full_molmo_audit["trainable_parameters"],
            full_molmo_audit_path,
        )
    if is_main_process and hasattr(policy.config, "flow_contract_summary"):
        logging.info("Resolved flow contract: %s", policy.config.flow_contract_summary())

    selected_views = parse_camera_views(getattr(cfg.policy, "camera_views", "agentview"))
    selected_rgb_views = tuple(
        getattr(
            policy.config,
            "selected_rgb_camera_views",
            parse_camera_views(
                selected_views
                if getattr(cfg.policy, "rgb_camera_views", None) is None
                else getattr(cfg.policy, "rgb_camera_views")
            ),
        )
    )
    actual_image_keys = set(getattr(policy.config, "image_features", {}))
    expected_rgb_cameras = {
        canonical_rgb_camera_name(view)
        for view in selected_rgb_views
    }
    actual_rgb_cameras = {
        canonical_rgb_camera_name(key)
        for key in actual_image_keys
    }
    missing_rgb_cameras = sorted(expected_rgb_cameras - actual_rgb_cameras)
    if missing_rgb_cameras and bool(getattr(cfg.policy, "requires_rgb", False)):
        raise ValueError(
            f"Selected RGB camera views {selected_rgb_views} require semantic cameras "
            f"{missing_rgb_cameras}, "
            f"but policy image features are {sorted(actual_image_keys)}."
        )
    if is_main_process:
        logging.info(
            "Training point-cloud camera_views=%s; rgb_camera_views=%s; image_features=%s",
            selected_views,
            selected_rgb_views,
            sorted(actual_image_keys),
        )
        provenance_path = write_training_camera_provenance(
            cfg,
            policy,
            camera_cli_provenance,
        )
        logging.info("Saved camera training provenance to %s", provenance_path)

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    ensure_ddp_parameters_initialized(policy, accelerator)

    # All ranks hold the same initialized policy before rank 0 streams the
    # frozen VLM/vision bytes through a bounded-memory hash.
    accelerator.wait_for_everyone()
    frozen_molmo_hash_before = None
    if full_molmo_audit and exact_global_batch_plan is not None and is_main_process:
        logging.info("Hashing frozen Full-Molmo2-ER VLM/vision parameters before training")
        frozen_molmo_hash_before = hash_full_molmo2er_frozen_parameters(policy)
        _write_json_atomically(
            Path(cfg.output_dir) / "full_molmo2er_frozen_parameter_hash_before.json",
            {"version": 1, "before": frozen_molmo_hash_before},
        )
        logging.info(
            "Frozen Full-Molmo2-ER pre-training hash: sha256=%s bytes=%s",
            frozen_molmo_hash_before["sha256"],
            frozen_molmo_hash_before["total_bytes"],
        )
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}


    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    if bool(getattr(policy.config, "worldflow_enable", False)):
        validate_smolvla_worldflow_preprocessor(preprocessor)

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if getattr(policy.config, "vlm_backend", None) == "molmo2_full":
        fp32_master_config_enabled = bool(
            getattr(cfg.optimizer, "fp32_master_weights", False)
        )
        fp32_master_runtime_enabled = isinstance(optimizer, FP32MasterAdamW)
        if fp32_master_config_enabled != fp32_master_runtime_enabled:
            raise RuntimeError(
                "Full-Molmo2-ER FP32-master optimizer config/runtime mismatch: "
                f"config={fp32_master_config_enabled}, runtime={type(optimizer).__name__}."
            )
        policy_fp32_master_requested = bool(
            getattr(policy.config, "optimizer_fp32_master_weights", False)
        )
        if policy_fp32_master_requested and not fp32_master_runtime_enabled:
            raise RuntimeError(
                "policy.optimizer_fp32_master_weights=true did not enable the runtime optimizer. "
                "On config_path resume, optimizer settings are restored independently from the "
                "policy preset; also pass --optimizer.fp32_master_weights=true."
            )
        if fp32_master_runtime_enabled and not policy_fp32_master_requested:
            # Preserve the actual optimizer mode in the policy/config artifacts
            # written by this run, including resumes from an older config that
            # predates the FP32-master option.
            policy.config.optimizer_fp32_master_weights = True
            cfg.policy.optimizer_fp32_master_weights = True
        if is_main_process:
            master_parameters = (
                tuple(optimizer.master_parameters()) if fp32_master_runtime_enabled else ()
            )
            master_numel = sum(parameter.numel() for parameter in master_parameters)
            master_bytes = sum(
                parameter.numel() * parameter.element_size() for parameter in master_parameters
            )
            logging.info(
                "Full-Molmo2-ER optimizer=%s fp32_master_weights=%s "
                "master_tensors=%s master_parameters=%s master_bytes=%s",
                type(optimizer).__name__,
                fp32_master_runtime_enabled,
                len(master_parameters),
                master_numel,
                master_bytes,
            )
        # CUDA's automatic foreach AdamW path materializes a temporary tensor
        # list roughly as large as all 1.8B trainable parameters during the
        # first step.  The scalar-tensor path has identical AdamW
        # hyperparameters/state semantics but updates one tensor at a time,
        # avoiding that multi-GiB transient on 32-GiB RTX 5090s.
        for parameter_group in optimizer.param_groups:
            parameter_group["foreach"] = False
        if is_main_process:
            logging.info(
                "Full-Molmo2-ER AdamW uses foreach=False to bound optimizer-step peak memory; "
                "lr/betas/eps/weight_decay are unchanged."
            )
        optimizer_parameter_ids = {id(parameter) for parameter in optimizer_model_parameters(optimizer)}
        backend = policy.model.vlm_with_expert
        frozen_parameter_ids = {
            id(parameter)
            for _, parameter in backend.named_frozen_molmo_parameters()
        }
        overlap = optimizer_parameter_ids & frozen_parameter_ids
        if overlap:
            raise RuntimeError(
                f"Full-Molmo2-ER optimizer contains {len(overlap)} frozen Molmo parameter tensors."
            )
        lora_parameter_ids = {
            id(parameter) for _, parameter in backend.named_molmo_lora_parameters()
        }
        if not lora_parameter_ids.issubset(optimizer_parameter_ids):
            missing_lora = len(lora_parameter_ids - optimizer_parameter_ids)
            raise RuntimeError(
                f"Full-Molmo2-ER optimizer omitted {missing_lora} Molmo V3 LoRA parameter tensors."
            )
        if any(group.get("foreach") is not False for group in optimizer.param_groups):
            raise RuntimeError("Full-Molmo2-ER requires low-peak AdamW foreach=False on every group.")

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state_for_resume(
            cfg,
            optimizer,
            lr_scheduler,
        )
        if cfg.resume_reset_optimizer_moments and is_main_process:
            logging.info(
                "Resume Adam-moment reset: restored_step=%s reset_origin_step=%s applied_now=%s "
                "remaining_state_entries=%s",
                step,
                cfg.resume_optimizer_moments_reset_step,
                bool(getattr(cfg, "resume_optimizer_moments_reset_applied", False)),
                len(optimizer.state),
            )
        if cfg.resume_restart_scheduler and is_main_process:
            logging.info(
                "Resumed Adam/RNG/global step with a phase-relative scheduler restart: "
                "global_step=%s, phase_start_step=%s, phase_step=%s, start_lr=%s, "
                "end_lr=%s, decay_steps=%s, current_lrs=%s",
                step,
                cfg.resume_scheduler_phase_start_step,
                step - int(cfg.resume_scheduler_phase_start_step),
                cfg.resume_scheduler_start_lr,
                cfg.resume_scheduler_end_lr,
                cfg.resume_scheduler_decay_steps,
                [group["lr"] for group in optimizer.param_groups],
            )
    if exact_global_batch_plan is not None and is_main_process:
        logging.info(
            "Exact global-batch partial-rank rotation at optimizer step %s: active ranks=%s",
            step,
            exact_global_batch_active_ranks(
                exact_global_batch_plan,
                optimizer_step=step,
                micro_step=exact_global_batch_plan.gradient_accumulation_steps - 1,
            ),
        )

    num_learnable_params = count_parameters(policy, only_trainable=True)
    num_total_params = count_parameters(policy, only_trainable=False)


    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        accumulation_steps = int(cfg.gradient_accumulation_steps)
        if exact_global_batch_plan is None:
            effective_bs = cfg.batch_size * accumulation_steps * num_processes
            logging.info(
                "Effective batch size: "
                f"{cfg.batch_size} x {accumulation_steps} accumulation x "
                f"{num_processes} process(es) = {effective_bs}"
            )
        else:
            effective_bs = exact_global_batch_plan.global_batch_size
            logging.info(
                "Exact effective batch size: %s active samples from %s physical forwards "
                "(%s discarded only for gradient; every rank still runs forward/backward)",
                effective_bs,
                exact_global_batch_plan.physical_forward_samples_per_optimizer_step,
                exact_global_batch_plan.discarded_samples_per_optimizer_step,
            )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames") or cfg.task_balanced_sampling:
        shuffle = False
        sampler_kwargs = {
            "dataset_from_indices": dataset.meta.episodes["dataset_from_index"],
            "dataset_to_indices": dataset.meta.episodes["dataset_to_index"],
            "episode_indices_to_use": dataset.episodes,
            "rebase_selected_episodes": dataset.episodes is not None,
            "drop_n_last_frames": int(getattr(cfg.policy, "drop_n_last_frames", 0)),
            "shuffle": True,
        }
        if cfg.task_balanced_sampling:
            episode_tasks = dataset.meta.episodes["tasks"]
            invalid = [index for index, tasks in enumerate(episode_tasks) if len(tasks) != 1]
            if invalid:
                raise ValueError(
                    "task_balanced_sampling requires exactly one task per episode; "
                    f"invalid episode indices include {invalid[:10]}."
                )
            sampler = TaskBalancedFrameSampler(
                episode_group_ids=[str(tasks[0]) for tasks in episode_tasks],
                **sampler_kwargs,
            )
            if is_main_process:
                source_counts = {
                    str(group_id): len(indices)
                    for group_id, indices in sampler.grouped_indices.items()
                }
                logging.info(
                    "Task-balanced frame sampling enabled: %d tasks, %d samples/epoch, "
                    "source frame counts=%s",
                    len(source_counts),
                    len(sampler),
                    source_counts,
                )
        else:
            sampler = EpisodeAwareSampler(**sampler_kwargs)
    else:
        shuffle = True
        sampler = None

    collate_fn = make_song_train_collate_fn(dataset)
    dataloader_num_workers = int(cfg.num_workers)
    if cfg.diagnostic_repeat_first_batch and dataloader_num_workers != 0:
        if is_main_process:
            logging.info(
                "Fixed-batch capacity diagnostic sets DataLoader num_workers=0 so the cached "
                "batch is selected synchronously under diagnostic_fixed_batch_seed."
            )
        dataloader_num_workers = 0
    if is_main_process and isinstance(collate_fn, OnlinePointSegBatchCollator):
        logging.info("Song pointseg online pseudo labels will be computed once per DataLoader batch.")
    if isinstance(collate_fn, OnlinePointSegBatchCollator) and collate_fn.device.type == "cuda" and dataloader_num_workers > 0:
        if is_main_process:
            logging.warning(
                "Song pointseg online pseudo labels use CUDA; setting DataLoader num_workers=0 "
                "to avoid CUDA initialization in forked worker processes."
            )
        dataloader_num_workers = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=dataloader_num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=exact_global_batch_plan is not None,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,
        persistent_workers=dataloader_num_workers > 0,
        collate_fn=collate_fn,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    prepared_optimizer = getattr(optimizer, "optimizer", optimizer)
    if isinstance(prepared_optimizer, FP32MasterAdamW):
        prepared_optimizer.synchronize_master_parameters(src=0)
        if is_main_process:
            logging.info(
                "Full-Molmo2-ER FP32 master parameters synchronized from rank 0 after "
                "Accelerate/DDP preparation."
            )
    if getattr(cfg.policy, "vlm_backend", None) == "molmo2_full":
        if accelerator.num_processes > 1:
            if not isinstance(policy, torch.nn.parallel.DistributedDataParallel):
                raise RuntimeError(
                    "Full-Molmo2-ER expected Accelerate to return DistributedDataParallel "
                    f"for {accelerator.num_processes} processes, got {type(policy).__name__}."
                )
            if policy.gradient_as_bucket_view is not True:
                raise RuntimeError(
                    "Full-Molmo2-ER requires DDP gradient_as_bucket_view=True to avoid a "
                    "second 3.47-GiB gradient allocation per rank."
                )
            if policy.find_unused_parameters is not False:
                raise RuntimeError("Full-Molmo2-ER requires DDP find_unused_parameters=False.")
            if policy.static_graph is not False:
                raise RuntimeError(
                    "Full-Molmo2-ER requires DDP static_graph=False because its no_sync "
                    "accumulation is incompatible with PyTorch 2.8 static-graph DDP."
                )
            if is_main_process:
                logging.info(
                    "Full-Molmo2-ER DDP memory audit passed: "
                    "gradient_as_bucket_view=%s (effective after the first iteration), "
                    "find_unused_parameters=%s, static_graph=%s",
                    policy.gradient_as_bucket_view,
                    policy.find_unused_parameters,
                    policy.static_graph,
                )
        elif is_main_process:
            logging.info(
                "Full-Molmo2-ER DDP memory audit skipped for single-process execution; "
                "gradient_as_bucket_view is only applicable to multi-process DDP."
            )
    fixed_overfit_batch = None
    if cfg.diagnostic_repeat_first_batch:
        with fixed_overfit_rng(cfg.diagnostic_fixed_batch_seed):
            fixed_overfit_batch = next(iter(dataloader))
        if is_main_process:
            fixed_overfit_indices = fixed_overfit_batch.get("index")
            if not torch.is_tensor(fixed_overfit_indices):
                raise KeyError("Fixed-batch diagnostic requires the cached batch key 'index'.")
            fixed_overfit_contract = {
                "version": 1,
                "repeat_first_batch": True,
                "batch_seed": int(cfg.diagnostic_fixed_batch_seed),
                "forward_seed": int(cfg.diagnostic_fixed_forward_seed),
                "repeat_forward_rng": bool(cfg.diagnostic_repeat_forward_rng),
                "dataset_indices": [
                    int(index) for index in fixed_overfit_indices.detach().cpu().reshape(-1).tolist()
                ],
                "batch_size": int(cfg.batch_size),
                "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                "num_processes": int(accelerator.num_processes),
            }
            _write_json_atomically(
                Path(cfg.output_dir) / "fixed_batch_overfit_contract.json",
                fixed_overfit_contract,
            )
            logging.warning(
                "Fixed-batch capacity diagnostic is active: one cached batch will be reused for "
                "every optimizer update; repeat_forward_rng=%s. This run is not a "
                "generalization run.",
                bool(cfg.diagnostic_repeat_forward_rng),
            )
        dl_iter = None
    else:
        dl_iter = cycle(dataloader)

    policy.train()
    post_adam_cuda_memory_audit_done = True
    if exact_global_batch_plan is not None and device.type == "cuda":
        aggregate_memory_audit_path = exact_global_batch_cuda_memory_audit_path(cfg.output_dir)
        audit_exists_flag = torch.tensor(
            int(is_main_process and aggregate_memory_audit_path.is_file()),
            device=device,
            dtype=torch.int32,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(audit_exists_flag, src=0)
        elif accelerator.num_processes != 1:
            raise RuntimeError(
                "Exact post-Adam memory audit requires an initialized distributed process group."
            )
        post_adam_cuda_memory_audit_done = bool(audit_exists_flag.item())
        if post_adam_cuda_memory_audit_done and is_main_process:
            logging.info("Reusing completed post-Adam CUDA memory audit: %s", aggregate_memory_audit_path)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    accumulation_steps = int(cfg.gradient_accumulation_steps)
    if exact_global_batch_plan is None:
        effective_batch_size = cfg.batch_size * accumulation_steps * accelerator.num_processes
        tracker_batch_size = cfg.batch_size * accumulation_steps
        tracker_accelerator = accelerator
    else:
        effective_batch_size = exact_global_batch_plan.global_batch_size
        # Count only gradient-contributing samples. Supplying the already-global
        # value without an accelerator prevents MetricsTracker multiplying by W.
        tracker_batch_size = exact_global_batch_plan.global_batch_size
        tracker_accelerator = None
    train_tracker = MetricsTracker(
        tracker_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=tracker_accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        output_dict = {}
        exact_local_loss_sum = 0.0
        for micro_step in range(accumulation_steps):
            if exact_global_batch_plan is None:
                rank_loss_scale = 1.0 / accumulation_steps
                rank_contributes_gradient = True
            else:
                rank_loss_scale = exact_global_batch_rank_loss_scale(
                    exact_global_batch_plan,
                    optimizer_step=step,
                    micro_step=micro_step,
                    rank=accelerator.process_index,
                )
                rank_contributes_gradient = rank_loss_scale > 0.0
            start_time = time.perf_counter()
            overfit_rng = (
                fixed_overfit_rng(cfg.diagnostic_fixed_forward_seed)
                if cfg.diagnostic_repeat_first_batch and cfg.diagnostic_repeat_forward_rng
                else nullcontext()
            )
            with overfit_rng:
                if fixed_overfit_batch is not None:
                    batch = clone_fixed_overfit_batch(fixed_overfit_batch)
                else:
                    if dl_iter is None:
                        raise RuntimeError("Training DataLoader iterator was not initialized.")
                    batch = next(dl_iter)
                batch = preprocessor(batch)
                train_tracker.dataloading_s = time.perf_counter() - start_time
                is_last_micro_step = micro_step + 1 == accumulation_steps
                sync_context = nullcontext() if is_last_micro_step else accelerator.no_sync(policy)
                with sync_context:
                    train_tracker, micro_output_dict = update_policy(
                        train_tracker,
                        policy,
                        batch,
                        optimizer,
                        cfg.optimizer.grad_clip_norm,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        rabc_weights_provider=rabc_weights,
                        loss_scale=rank_loss_scale,
                        perform_optimizer_step=is_last_micro_step,
                        require_per_sample_mean=exact_global_batch_plan is not None,
                        audit_first_step_gradients=bool(
                            full_molmo_audit and step == 0 and is_last_micro_step
                        ),
                        gradient_audit_output_dir=cfg.output_dir,
                        # Exact mode records one globally reduced mean below,
                        # rather than treating every micro-step as a full batch.
                        record_loss=exact_global_batch_plan is None,
                    )
            if rank_contributes_gradient:
                output_dict = micro_output_dict
                if exact_global_batch_plan is not None:
                    exact_local_loss_sum += (
                        float(micro_output_dict["loss"])
                        * exact_global_batch_plan.micro_batch_size_per_rank
                    )

        if exact_global_batch_plan is not None:
            exact_loss_sum = torch.tensor(exact_local_loss_sum, device=device, dtype=torch.float64)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(exact_loss_sum, op=torch.distributed.ReduceOp.SUM)
            elif accelerator.num_processes != 1:
                raise RuntimeError("Exact loss logging requires an initialized process group.")
            exact_global_loss = float(exact_loss_sum.item()) / exact_global_batch_plan.global_batch_size
            train_tracker.loss = exact_global_loss
            output_dict["loss"] = exact_global_loss

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if device.type == "cuda":
            gib = float(1024**3)
            output_dict["cuda_allocated_gib"] = torch.cuda.memory_allocated(device) / gib
            output_dict["cuda_reserved_gib"] = torch.cuda.memory_reserved(device) / gib
            output_dict["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / gib
        if not post_adam_cuda_memory_audit_done:
            # update_policy has completed optimizer.step(), so Adam's lazy state
            # tensors exist before this synchronized memory snapshot.
            torch.cuda.synchronize(device)
            optimizer_state_count, optimizer_state_numel = optimizer_state_tensor_summary(optimizer)
            memory_record = build_exact_global_batch_cuda_memory_rank_record(
                plan=exact_global_batch_plan,
                completed_optimizer_step=step,
                global_rank=accelerator.process_index,
                local_rank=accelerator.local_process_index,
                logical_cuda_index=torch.cuda.current_device(),
                accelerator_device=str(device),
                memory_snapshot={
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
                optimizer_state_tensor_count=optimizer_state_count,
                optimizer_state_total_numel=optimizer_state_numel,
                environ=os.environ,
                hostname=os.uname().nodename,
            )
            rank_memory_path = write_exact_global_batch_cuda_memory_rank_record(
                cfg.output_dir,
                memory_record,
            )
            logging.info(
                "Rank %s saved post-Adam CUDA memory audit to %s",
                accelerator.process_index,
                rank_memory_path,
            )
            accelerator.wait_for_everyone()
            if is_main_process:
                aggregate_memory_audit_path = aggregate_exact_global_batch_cuda_memory_records(
                    cfg.output_dir,
                    plan=exact_global_batch_plan,
                    expected_completed_optimizer_step=step,
                )
                logging.info(
                    "Aggregated %s-rank post-Adam CUDA memory audit: %s",
                    accelerator.num_processes,
                    aggregate_memory_audit_path,
                )
            accelerator.wait_for_everyone()
            post_adam_cuda_memory_audit_done = True
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if output_dict:
                debug_keys = (
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
                    "point_prefix_rms",
                    "language_prefix_rms",
                    "point_language_rms_ratio",
                    "cuda_allocated_gib",
                    "cuda_reserved_gib",
                    "cuda_peak_allocated_gib",
                    "skipped_nonfinite_loss",
                    "skipped_nonfinite_grad",
                    "pred_operation_prob",
                    "loss_soft_bce",
                    "loss_smoothness",
                    "pseudo_valid_ratio",
                    "pseudo_foreground_ratio",
                    "pseudo_soft_foreground_mean",
                    "pred_foreground_ratio",
                )
                debug_items = []
                for key in debug_keys:
                    value = output_dict.get(key)
                    if value is not None:
                        debug_items.append(f"{key}:{float(value):.4g}")
                if debug_items:
                    logging.info(" ".join(debug_items))
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            with torch.no_grad(), accelerator.autocast():
                save_joint_pointseg_visualization(
                    policy,
                    batch,
                    step=step,
                    output_dir=cfg.output_dir,
                    tag="train",
                    max_items=2,
                )
                try:
                    ood_case_inference(policy, preprocessor, postprocessor, batch, step, output_dir=cfg.output_dir,ood_num_points=50000)
                except Exception:
                    logging.exception("OOD case inference failed at step %s; continuing training/checkpoint save.", step)

            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                deleted_checkpoints = prune_committed_training_checkpoints(
                    committed_checkpoint=checkpoint_dir,
                    keep=resolve_checkpoint_retention(os.environ),
                )
                if deleted_checkpoints:
                    logging.info(
                        "Checkpoint retention kept the newest committed checkpoint and deleted: %s",
                        ", ".join(str(path) for path in deleted_checkpoints),
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if frozen_molmo_hash_before is not None:
        logging.info("Hashing frozen Full-Molmo2-ER VLM/vision parameters after training")
        frozen_molmo_hash_after = hash_full_molmo2er_frozen_parameters(
            accelerator.unwrap_model(policy)
        )
        frozen_hash_audit_path = write_full_molmo2er_frozen_parameter_hash_audit(
            cfg.output_dir,
            before=frozen_molmo_hash_before,
            after=frozen_molmo_hash_after,
        )
        logging.info(
            "Frozen Full-Molmo2-ER before/after hash audit passed: sha256=%s artifact=%s",
            frozen_molmo_hash_after["sha256"],
            frozen_hash_audit_path,
        )
    accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()


# def random_repeat_sample_points(xyzrgb: np.ndarray, M: int):
#     N = xyzrgb.shape[0]
#     if N == 0:
#         return xyzrgb
#     if N >= M:
#         idx = np.random.choice(N, M, replace=False)
#         return xyzrgb[idx]
#     else:
#         extra = np.random.choice(N, M - N, replace=True)
#         return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)   
# batch['task'][0] = "Place the Red Cube on the Blue Cube"
# scene_pcd = o3d.io.read_point_cloud(f"/home/liusong/temp/ood_test_new4.ply",)
# scene_point_cloud = np.concatenate((np.asarray(scene_pcd.points[:]),np.asarray(scene_pcd.colors[:])*255), axis=1)
# scene_point_cloud = random_repeat_sample_points(scene_point_cloud, 10000)
# batch['observation.point_cloud'][0] = torch.tensor(scene_point_cloud).to("cuda")
# action_pred = self.predict_action_chunk(batch)
# vis_umi_data(action_pred.cpu().numpy()[0],batch['observation.point_cloud'].cpu().numpy()[0])
# print(action_pred[0][:,-1])
