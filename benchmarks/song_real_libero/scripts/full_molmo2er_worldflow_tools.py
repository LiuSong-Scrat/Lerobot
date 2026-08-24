#!/usr/bin/env python3
"""Auditable helpers for the formal eight-GPU Full-Molmo2-ER run.

This module intentionally uses only Python for JSON processing, CUDA discovery,
checkpoint validation, provenance hashing, and LIBERO report aggregation.  The
shell launchers therefore do not depend on jq or nvidia-smi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PHYSICAL_GPU_IDS = tuple(range(8))
CUDA_VISIBLE_DEVICES_CONTRACT = ",".join(str(item) for item in PHYSICAL_GPU_IDS)
GLOBAL_BATCH_SIZE = 192
DEFAULT_BATCH_PROFILE = "b24"
# These are the only performance profiles accepted by the eight-A800 launcher.
# Accumulation is kept explicit in this whitelist and then checked against the
# exact-global-batch arithmetic below; callers cannot supply arbitrary values.
BATCH_PROFILE_PARAMETERS: dict[str, tuple[int, int]] = {
    "b4": (4, 6),
    "b8": (8, 3),
    "b16": (16, 2),
    "b24": (24, 1),
}


def global_batch_contract(batch_profile: str = DEFAULT_BATCH_PROFILE) -> dict[str, Any]:
    """Return the exact-gradient global-192 contract for one safe profile."""

    try:
        microbatch_per_rank, microsteps = BATCH_PROFILE_PARAMETERS[batch_profile]
    except KeyError as exc:
        choices = ", ".join(BATCH_PROFILE_PARAMETERS)
        raise ValueError(f"Unknown batch profile {batch_profile!r}; expected one of: {choices}.") from exc

    world_size = len(PHYSICAL_GPU_IDS)
    samples_per_microstep = world_size * microbatch_per_rank
    expected_microsteps = (GLOBAL_BATCH_SIZE + samples_per_microstep - 1) // samples_per_microstep
    if microsteps != expected_microsteps:
        raise RuntimeError(f"Internal accumulation mismatch for batch profile {batch_profile!r}.")
    full_microsteps, partial_samples = divmod(GLOBAL_BATCH_SIZE, samples_per_microstep)
    if partial_samples % microbatch_per_rank:
        raise RuntimeError(f"Batch profile {batch_profile!r} cannot represent exact global batch 192.")
    partial_active_ranks = partial_samples // microbatch_per_rank
    physical_forward_samples = samples_per_microstep * microsteps
    return {
        "profile": batch_profile,
        "world_size": world_size,
        "microbatch_per_rank": microbatch_per_rank,
        "microsteps_per_optimizer_step": microsteps,
        "full_microsteps": full_microsteps,
        "partial_microstep_active_ranks": partial_active_ranks,
        "physical_forward_samples": physical_forward_samples,
        "discarded_for_gradient": physical_forward_samples - GLOBAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "sample_counter_increment_per_optimizer_step": GLOBAL_BATCH_SIZE,
        "scheduler_steps_per_optimizer_step": 1,
        "logged_loss_reduction": "global mean over gradient-contributing samples, once per optimizer step",
        "partial_microstep_valid_loss_scale": f"{world_size * microbatch_per_rank}/{GLOBAL_BATCH_SIZE}",
        "partial_rank_rotation": True,
    }




def exact_global_batch_runtime_contract(
    batch_profile: str = DEFAULT_BATCH_PROFILE,
) -> dict[str, Any]:
    """Return fields emitted by train_song_benchmark's exact-batch manifest."""

    contract = global_batch_contract(batch_profile)
    partial_active_ranks = contract["partial_microstep_active_ranks"]
    return {
        "global_batch_size": contract["global_batch_size"],
        "world_size": contract["world_size"],
        "micro_batch_size_per_rank": contract["microbatch_per_rank"],
        "gradient_accumulation_steps": contract["microsteps_per_optimizer_step"],
        "full_micro_steps": contract["full_microsteps"],
        "partial_micro_step_index": contract["full_microsteps"] if partial_active_ranks else None,
        "partial_active_ranks": partial_active_ranks,
        "physical_forward_samples_per_optimizer_step": contract["physical_forward_samples"],
        "discarded_for_gradient_samples_per_optimizer_step": contract["discarded_for_gradient"],
        "valid_loss_scale": (
            contract["world_size"] * contract["microbatch_per_rank"] / contract["global_batch_size"]
        ),
        "valid_loss_scale_fraction": contract["partial_microstep_valid_loss_scale"],
        "ddp_gradient_reduction": "mean",
        "partial_rank_rotation": "consecutive ranks from optimizer_step % world_size",
        "partial_rank_rotation_period_optimizer_steps": contract["world_size"],
        "all_ranks_forward_backward_every_micro_step": True,
        "sample_counter_increment_per_optimizer_step": contract["sample_counter_increment_per_optimizer_step"],
        "scheduler_steps_per_optimizer_step": contract["scheduler_steps_per_optimizer_step"],
        "logged_loss_reduction": contract["logged_loss_reduction"],
    }


@dataclass(frozen=True)
class EvalShard:
    name: str
    physical_gpu: int
    suite: str
    task_ids: tuple[int, ...]


# Eight persistent workers own one five-task shard each on physical GPUs 0--7.
EVAL_SHARDS = (
    EvalShard("gpu0_libero_spatial_tasks0-4", 0, "libero_spatial", tuple(range(0, 5))),
    EvalShard("gpu1_libero_spatial_tasks5-9", 1, "libero_spatial", tuple(range(5, 10))),
    EvalShard("gpu2_libero_object_tasks0-4", 2, "libero_object", tuple(range(0, 5))),
    EvalShard("gpu3_libero_object_tasks5-9", 3, "libero_object", tuple(range(5, 10))),
    EvalShard("gpu4_libero_goal_tasks0-4", 4, "libero_goal", tuple(range(0, 5))),
    EvalShard("gpu5_libero_goal_tasks5-9", 5, "libero_goal", tuple(range(5, 10))),
    EvalShard("gpu6_libero_10_tasks0-4", 6, "libero_10", tuple(range(0, 5))),
    EvalShard("gpu7_libero_10_tasks5-9", 7, "libero_10", tuple(range(5, 10))),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty {description}: {path}")


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _source_manifest(root: Path) -> dict[str, str]:
    selected_roots = (root / "src", root / "benchmarks" / "song_real_libero" / "scripts")
    files: list[Path] = []
    for source_root in selected_roots:
        if not source_root.is_dir():
            continue
        files.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts
        )
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(set(files))}


def _assert_eval_plan() -> None:
    if {shard.physical_gpu for shard in EVAL_SHARDS} != set(PHYSICAL_GPU_IDS):
        raise RuntimeError("Evaluation plan does not use exactly physical GPUs 0--7.")
    expected = {(suite, task_id) for suite in SUITES for task_id in range(10)}
    actual = {(shard.suite, task_id) for shard in EVAL_SHARDS for task_id in shard.task_ids}
    if actual != expected or sum(len(shard.task_ids) for shard in EVAL_SHARDS) != 40:
        raise RuntimeError("Evaluation plan is not an exact one-time partition of all 40 LIBERO tasks.")


def command_contract(args: argparse.Namespace) -> int:
    _assert_eval_plan()
    batch_contract = global_batch_contract(args.batch_profile)
    print(
        json.dumps(
            {
                "cuda_visible_devices": CUDA_VISIBLE_DEVICES_CONTRACT,
                "global_batch": batch_contract,
                "evaluation_shards": [asdict(item) for item in EVAL_SHARDS],
            },
            sort_keys=True,
        )
    )
    return 0


def command_eval_plan(args: argparse.Namespace) -> int:
    _assert_eval_plan()
    if args.json:
        print(json.dumps([asdict(item) for item in EVAL_SHARDS], sort_keys=True))
    else:
        for shard in EVAL_SHARDS:
            print(
                "\t".join(
                    (
                        shard.name,
                        str(shard.physical_gpu),
                        shard.suite,
                        ",".join(str(task_id) for task_id in shard.task_ids),
                    )
                )
            )
    return 0


def command_gpu_audit(args: argparse.Namespace) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != CUDA_VISIBLE_DEVICES_CONTRACT:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be exactly "
            f"{CUDA_VISIBLE_DEVICES_CONTRACT!r}; got {visible!r}."
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the selected training/evaluation Python.")
    if torch.cuda.device_count() != len(PHYSICAL_GPU_IDS):
        raise RuntimeError(
            f"Expected eight visible CUDA devices, found {torch.cuda.device_count()} under {visible}."
        )
    devices: list[dict[str, Any]] = []
    for logical_index, physical_index in enumerate(PHYSICAL_GPU_IDS):
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
        free_mib = free_bytes // (1024 * 1024)
        if free_mib < args.min_free_mib:
            raise RuntimeError(
                f"Physical GPU {physical_index} has only {free_mib} MiB free; "
                f"the launch requires at least {args.min_free_mib} MiB."
            )
        properties = torch.cuda.get_device_properties(logical_index)
        devices.append(
            {
                "logical_index": logical_index,
                "physical_index": physical_index,
                "name": properties.name,
                "total_mib": total_bytes // (1024 * 1024),
                "free_mib": free_mib,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    payload = {
        "audited_at": utc_now(),
        "cuda_visible_devices": visible,
        "minimum_free_mib": args.min_free_mib,
        "devices": devices,
    }
    if args.output:
        atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_model_audit(args: argparse.Namespace) -> int:
    """Fail before data hashing unless the complete local Molmo source is usable."""

    model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
    _require_file(model_dir / "config.json", "Molmo2-ER config.json")
    project_root = Path(__file__).resolve().parents[3]
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from transformers import AutoConfig, AutoProcessor

        from lerobot.policies.smolvla.molmo2_full_with_expert import (
            _audit_source_checkpoint,
            _local_vision_backbone_class,
            validate_molmo2_er_vision_contract,
        )
        from lerobot.policies.smolvla.molmo2_with_expert import (
            Molmo2TextSpec,
            _checkpoint_weight_map,
            validate_molmo2_er_text_contract,
        )

        text_spec = Molmo2TextSpec.from_model_directory(model_dir)
        validate_molmo2_er_text_contract(text_spec)
        native_config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        validate_molmo2_er_vision_contract(native_config)
        vision_class = _local_vision_backbone_class(model_dir, native_config)
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None or (
            getattr(tokenizer, "bos_token_id", None) is None
            and getattr(tokenizer, "eos_token_id", None) is None
        ):
            raise ValueError("Native Molmo2-ER processor has no usable tokenizer BOS/EOS token.")
        weight_map = _checkpoint_weight_map(model_dir)
        source_report = _audit_source_checkpoint(model_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Local Molmo2-ER config/processor/remote-code/weights are not loadable offline: {exc}"
        ) from exc

    weight_files = sorted(set(weight_map.values()))
    for path in weight_files:
        _require_file(path, "Molmo2-ER safetensors shard")
    payload = {
        "audited_at": utc_now(),
        "model_dir": str(model_dir),
        "offline_only": True,
        "native_processor_loaded": True,
        "native_remote_code_loaded": True,
        "vision_backbone_class": vision_class.__name__,
        "text_layers": text_spec.num_hidden_layers,
        "hidden_size": text_spec.hidden_size,
        "weight_file_count": len(weight_files),
        "weight_files": {
            path.name: {"size_bytes": path.stat().st_size} for path in weight_files
        },
        "source_checkpoint": source_report,
    }
    if args.output:
        atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _episode_files(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise ValueError(f"Required WorldFlow sidecar directory is missing: {directory}")
    result: dict[int, Path] = {}
    pattern = re.compile(r"episode_(\d{6})\.npy$")
    for path in sorted(directory.glob("episode_*.npy")):
        match = pattern.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected WorldFlow sidecar filename: {path}")
        index = int(match.group(1))
        if index in result:
            raise ValueError(f"Duplicate WorldFlow episode sidecar index {index} in {directory}")
        result[index] = path
    if not result:
        raise ValueError(f"WorldFlow sidecar directory is empty: {directory}")
    return result


def _load_episode_metadata(
    dataset: Path, *, total_episodes: int, total_frames: int
) -> list[dict[str, int]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for formal episode-boundary validation.") from exc

    paths = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError("Dataset has no authoritative meta/episodes parquet files.")
    rows: list[dict[str, int]] = []
    columns = (
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
    )
    for path in paths:
        _require_file(path, "episode metadata parquet")
        table = parquet.read_table(path, columns=list(columns))
        for raw in table.to_pylist():
            rows.append({name: int(raw[name]) for name in columns})
    rows.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(total_episodes)):
        raise ValueError("Episode metadata does not contain exactly one row for every episode index.")
    cursor = 0
    for row in rows:
        length = row["length"]
        start = row["dataset_from_index"]
        end = row["dataset_to_index"]
        if length <= 0 or start != cursor or end != start + length:
            raise ValueError(f"Episode metadata boundary drift at episode {row['episode_index']}: {row}")
        cursor = end
    if cursor != total_frames:
        raise ValueError(f"Episode metadata ends at dataset index {cursor}, expected {total_frames}.")
    return rows


def _audit_pointseg_cache(
    cache: Path,
    cache_manifest: dict[str, Any],
    *,
    total_frames: int,
    episode_rows: Sequence[dict[str, int]],
) -> list[dict[str, Any]]:
    import numpy as np

    fields = cache_manifest.get("fields")
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) for field in fields):
        raise ValueError("PointSeg cache manifest fields must be a non-empty string list.")
    required_sample_fields = {"dataset_index", "episode_index", "frame_index"}
    if not required_sample_fields.issubset(fields) or "point_indices" not in fields:
        raise ValueError(
            "PointSeg cache fields must include dataset/episode/frame indices and point_indices."
        )
    shards = sorted(cache_manifest.get("shards", []), key=lambda item: int(item["start"]))
    if not shards:
        raise ValueError("PointSeg cache manifest contains no shards.")
    episode_starts = np.asarray(
        [row["dataset_from_index"] for row in episode_rows], dtype=np.int64
    )
    episode_ends = np.asarray(
        [row["dataset_to_index"] for row in episode_rows], dtype=np.int64
    )
    cursor = 0
    audited: list[dict[str, Any]] = []
    for shard in shards:
        start, length = int(shard["start"]), int(shard["length"])
        num_points = int(shard.get("num_points", -1))
        if start != cursor or length <= 0 or num_points <= 0:
            raise ValueError(f"PointSeg shard coverage/count drift at sample {cursor}: {shard}")
        shard_dir = cache / str(shard["path"])
        if not shard_dir.is_dir():
            raise ValueError(f"PointSeg manifest references a missing shard directory: {shard['path']}")

        offsets_path = shard_dir / "sample_offsets.npy"
        _require_file(offsets_path, "PointSeg sample_offsets.npy")
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if (
            offsets.ndim != 1
            or offsets.shape[0] != length + 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != num_points
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError(f"PointSeg sample offsets are invalid in {shard_dir}.")

        field_shapes: dict[str, list[int]] = {}
        for field in fields:
            field_path = shard_dir / f"{field}.npy"
            _require_file(field_path, f"PointSeg field {field}")
            array = np.load(field_path, mmap_mode="r", allow_pickle=False)
            expected_first_dim = length if field in required_sample_fields else num_points
            if array.ndim < 1 or int(array.shape[0]) != expected_first_dim:
                raise ValueError(
                    f"PointSeg field {field} in {shard_dir} has shape {array.shape}; "
                    f"expected first dimension {expected_first_dim}."
                )
            field_shapes[field] = list(array.shape)

        dataset_indices = np.load(
            shard_dir / "dataset_index.npy", mmap_mode="r", allow_pickle=False
        )
        expected_indices = np.arange(start, start + length, dtype=np.int64)
        if not np.array_equal(dataset_indices, expected_indices):
            raise ValueError(f"PointSeg dataset_index is not contiguous/aligned in {shard_dir}.")
        expected_episode_positions = np.searchsorted(
            episode_ends, expected_indices, side="right"
        )
        if np.any(expected_episode_positions >= len(episode_rows)):
            raise ValueError(f"PointSeg dataset_index crosses episode metadata bounds in {shard_dir}.")
        expected_episode_indices = expected_episode_positions.astype(np.int64)
        expected_frame_indices = expected_indices - episode_starts[expected_episode_positions]
        cached_episode_indices = np.load(
            shard_dir / "episode_index.npy", mmap_mode="r", allow_pickle=False
        )
        cached_frame_indices = np.load(
            shard_dir / "frame_index.npy", mmap_mode="r", allow_pickle=False
        )
        if not np.array_equal(cached_episode_indices, expected_episode_indices):
            raise ValueError(f"PointSeg episode_index is misaligned in {shard_dir}.")
        if not np.array_equal(cached_frame_indices, expected_frame_indices):
            raise ValueError(f"PointSeg frame_index is misaligned in {shard_dir}.")

        audited.append(
            {
                "path": str(shard["path"]),
                "start": start,
                "length": length,
                "num_points": num_points,
                "sample_offsets_size_bytes": offsets_path.stat().st_size,
                "field_shapes": field_shapes,
            }
        )
        cursor += length
    if cursor != total_frames:
        raise ValueError(f"PointSeg shard coverage ends at {cursor}, expected {total_frames}.")
    return audited


def command_data_audit(args: argparse.Namespace) -> int:
    import numpy as np

    dataset = Path(args.dataset).expanduser().resolve(strict=True)
    cache = Path(args.cache).expanduser().resolve(strict=True)
    info_path = dataset / "meta" / "info.json"
    cache_manifest_path = cache / "manifest.json"
    _require_file(info_path, "dataset metadata")
    _require_file(cache_manifest_path, "PointSeg cache manifest")
    info = load_json(info_path)
    cache_manifest = load_json(cache_manifest_path)
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    total_tasks = int(info.get("total_tasks", -1))
    if total_episodes <= 0 or total_frames <= 0 or total_tasks != 40:
        raise ValueError(
            "Dataset metadata must describe positive episode/frame counts and exactly 40 tasks; "
            f"got episodes={total_episodes}, frames={total_frames}, tasks={total_tasks}."
        )
    episode_rows = _load_episode_metadata(dataset, total_episodes=total_episodes, total_frames=total_frames)
    if int(cache_manifest.get("num_samples", -1)) != total_frames:
        raise ValueError("PointSeg cache num_samples does not equal dataset total_frames.")
    if int(cache_manifest.get("current_points", -1)) != 10_000:
        raise ValueError("PointSeg cache must contain the single-view 10,000-point contract.")
    cache_dataset = cache_manifest.get("args", {}).get("dataset_repo_id")
    if not cache_dataset or Path(cache_dataset).expanduser().resolve() != dataset:
        raise ValueError(
            f"PointSeg cache was built for {cache_dataset!r}, not the requested dataset {dataset}."
        )
    cache_shards = _audit_pointseg_cache(
        cache,
        cache_manifest,
        total_frames=total_frames,
        episode_rows=episode_rows,
    )

    achieved = _episode_files(dataset / "world_ee_poses")
    commanded = _episode_files(dataset / "action_target_ee_poses")
    expected_indices = set(range(total_episodes))
    if set(achieved) != expected_indices or set(commanded) != expected_indices:
        raise ValueError(
            "WorldFlow achieved/commanded sidecars must each contain exactly episode indices "
            f"0..{total_episodes - 1}."
        )

    sidecars: list[dict[str, Any]] = []
    summed_frames = 0
    for episode_index in range(total_episodes):
        achieved_array = np.load(
            achieved[episode_index], mmap_mode="r", allow_pickle=False
        )
        commanded_array = np.load(
            commanded[episode_index], mmap_mode="r", allow_pickle=False
        )
        if achieved_array.ndim != 2 or achieved_array.shape[1] != 9:
            raise ValueError(f"Invalid achieved pose9 sidecar shape: {achieved[episode_index]} {achieved_array.shape}")
        if commanded_array.shape != achieved_array.shape:
            raise ValueError(
                f"Achieved/commanded sidecar shape mismatch for episode {episode_index}: "
                f"{achieved_array.shape} != {commanded_array.shape}."
            )
        expected_length = episode_rows[episode_index]["length"]
        if int(achieved_array.shape[0]) != expected_length:
            raise ValueError(
                f"WorldFlow sidecar episode {episode_index} has {achieved_array.shape[0]} frames; "
                f"episode metadata requires {expected_length}."
            )
        for role, array, path in (
            ("achieved", achieved_array, achieved[episode_index]),
            ("commanded", commanded_array, commanded[episode_index]),
        ):
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError(
                    f"WorldFlow {role} pose9 sidecar contains non-finite/non-numeric values: {path}"
                )
        summed_frames += int(achieved_array.shape[0])
        sidecars.append(
            {
                "episode_index": episode_index,
                "shape": list(achieved_array.shape),
                "achieved_sha256": sha256_file(achieved[episode_index]),
                "commanded_sha256": sha256_file(commanded[episode_index]),
            }
        )
    if summed_frames != total_frames:
        raise ValueError(f"WorldFlow sidecars contain {summed_frames} frames, expected {total_frames}.")

    metadata_hashes = {
        str(path.relative_to(dataset)): sha256_file(path)
        for path in sorted((dataset / "meta").rglob("*"))
        if path.is_file()
    }
    semantic_payload = {
        "dataset": str(dataset),
        "cache": str(cache),
        "dataset_info": info,
        "episode_boundaries": episode_rows,
        "dataset_metadata_sha256": metadata_hashes,
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "cache_shards": cache_shards,
        "sidecars": sidecars,
    }
    semantic_hash = canonical_sha256(semantic_payload)
    if args.expected_hash and semantic_hash != args.expected_hash:
        raise ValueError(
            f"Dataset/cache semantic hash drifted: expected {args.expected_hash}, got {semantic_hash}."
        )
    payload = {
        "audited_at": utc_now(),
        "semantic_hash": semantic_hash,
        "dataset": str(dataset),
        "cache": str(cache),
        "total_tasks": total_tasks,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "point_count": 10_000,
        "cache_mode": cache_manifest.get("cache_mode"),
        "commanded_target_sidecar_required": True,
        "commanded_target_sidecar_verified": True,
        "all_pose9_values_finite": True,
        "episode_boundaries_verified": True,
        "sidecar_episode_count": len(sidecars),
        "cache_shard_count": len(cache_shards),
        "cache_dataset_index_alignment_verified": True,
        "cache_manifest_sha256": semantic_payload["cache_manifest_sha256"],
        "dataset_metadata_sha256": metadata_hashes,
    }
    if args.output:
        atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


POLICY_CONTRACT: dict[str, Any] = {
    "vlm_backend": "molmo2_full",
    "full_molmo_topology": "wepvla_scene_in_vlm_prefix_v3",
    "num_vlm_layers": 36,
    "num_expert_layers": 36,
    "train_expert_only": True,
    "freeze_vision_encoder": True,
    "molmo_inference_only": False,
    "molmo_gradient_checkpointing": True,
    "molmo_gradient_checkpointing_layers_per_segment": 2,
    "camera_views": "agentview",
    "rgb_camera_views": "agentview",
    "n_obs_steps": 1,
    "chunk_size": 32,
    "n_action_steps": 16,
    "max_state_dim": 10,
    "max_action_dim": 10,
    "encode_robot_state": False,
    "pointseg_enable": True,
    "pointseg_freeze_batchnorm_stats": True,
    "point_action_fusion_enable": True,
    "pose9_action_noise_enable": False,
    "worldflow_enable": True,
    "worldflow_target_type": "world_eef_trajectory",
    "worldflow_world_eef_velocity_mode": "base_pose9_euclidean",
    "worldflow_reference_frame": "robot_base",
    "worldflow_frame_origin": "global",
    "worldflow_scene_frame_origin": "global",
    "worldflow_action_fusion": "point_action_expert_conjugate_bridge",
    "worldflow_action_expert_mode": "shared",
    "worldflow_current_ee_pose_token": False,
    "worldflow_freeze_pretrained_ego": False,
    "worldflow_training_coordinate_frame_augmentation": False,
    "worldflow_pretrained_lr_multiplier": 1.0,
    "worldflow_new_lr_multiplier": 1.0,
    "worldflow_eef_probe_radius_m": 0.10,
    "worldflow_bootstrap_from_ego": False,
    "worldflow_ego_residual_gate_init": None,
    "worldflow_noise_coupling": "left_compose_ego",
    "worldflow_require_action_target_sidecar": True,
    "worldflow_feature_dim": 64,
    "worldflow_grid_size": 0.01,
    "worldflow_max_points": 2048,
    "worldflow_loss_weight": 1.0,
    "worldflow_geo_loss_weight": 0.0,
    "worldflow_bridge_loss_weight": 0.0,
    "worldflow_equiv_loss_weight": 0.0,
    "worldflow_trans_weight": 1.0,
    "worldflow_rot_weight": 1.0,
    "worldflow_noise_trans_scale": 0.15,
    "worldflow_noise_rot_scale": 0.20,
    "worldflow_augmentation_trans_scale": 0.20,
    "worldflow_augmentation_rot_scale": 0.75,
    "worldflow_action_expert_layers": -1,
    "worldflow_action_expert_dropout": 0.0,
    "worldflow_min_transport_points": 3,
    "worldflow_transport_score_threshold": 0.05,
    "worldflow_se3_head_enable": False,
    "se3_enable": False,
    "se3_final_correction_enable": False,
}


def _same_contract_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def audit_checkpoint(checkpoint: Path, *, expected_step: int | None, training_state: bool) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        _require_file(checkpoint / name, f"checkpoint {name}")
    config = load_json(checkpoint / "config.json")
    drift = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in POLICY_CONTRACT.items()
        if not _same_contract_value(config.get(key), expected)
    }
    if drift:
        raise ValueError(f"Checkpoint policy contract drifted: {json.dumps(drift, sort_keys=True)}")
    if expected_step is not None:
        step_name = checkpoint.parent.name
        if not re.fullmatch(r"\d{6}", step_name) or int(step_name) != expected_step:
            raise ValueError(f"Checkpoint path step {step_name!r} does not equal expected {expected_step:06d}.")
    state_payload: dict[str, Any] | None = None
    if training_state:
        training_state_dir = checkpoint.parent / "training_state"
        required = (
            "optimizer_param_groups.json",
            "optimizer_state.safetensors",
            "rng_state.safetensors",
            "scheduler_state.json",
            "training_step.json",
        )
        for name in required:
            _require_file(training_state_dir / name, f"training state {name}")
        state_payload = load_json(training_state_dir / "training_step.json")
        state_step = int(state_payload.get("step", -1))
        path_step = int(checkpoint.parent.name)
        if state_step != path_step:
            raise ValueError(f"Training-state step {state_step} does not match checkpoint path step {path_step}.")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint.parent.name),
        "model_size_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "training_state_verified": training_state,
        "training_state": state_payload,
    }


def command_checkpoint_audit(args: argparse.Namespace) -> int:
    payload = audit_checkpoint(
        Path(args.checkpoint),
        expected_step=args.expected_step,
        training_state=args.training_state,
    )
    if args.output:
        atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_find_resume(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=True)
    candidates: list[tuple[int, Path]] = []
    for checkpoint in (output_dir / "checkpoints").glob("[0-9][0-9][0-9][0-9][0-9][0-9]/pretrained_model"):
        step = int(checkpoint.parent.name)
        if step <= 0 or step >= args.target_step:
            continue
        try:
            audit_checkpoint(checkpoint, expected_step=step, training_state=True)
            _require_file(checkpoint / "train_config.json", "resume train_config.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((step, checkpoint / "train_config.json"))
    if not candidates:
        return 4
    step, config_path = max(candidates)
    if args.json:
        print(json.dumps({"step": step, "config_path": str(config_path)}, sort_keys=True))
    else:
        print(config_path)
    return 0


def command_publish_alias(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve(strict=True)
    audit_checkpoint(source, expected_step=args.source_step, training_state=False)
    alias = Path(args.alias).expanduser().absolute()
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        try:
            existing = alias.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"Existing cumulative checkpoint alias is broken: {alias}") from exc
        if existing != source:
            raise ValueError(f"Cumulative checkpoint alias collision: {alias} -> {existing}, expected {source}")
    else:
        temporary = alias.with_name(f".{alias.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(source, target_is_directory=True)
        os.rename(temporary, alias)
    payload = {
        "published_at": utc_now(),
        "cumulative_step": args.cumulative_step,
        "source_stage_step": args.source_step,
        "alias": str(alias),
        "checkpoint": str(source),
    }
    if args.output:
        atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_write_stage_manifest(args: argparse.Namespace) -> int:
    root = Path(args.experiment_root).expanduser().resolve(strict=True)
    model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
    data_audit = load_json(Path(args.data_audit))
    source_checkpoint = None
    if args.policy_path:
        source_checkpoint = audit_checkpoint(Path(args.policy_path), expected_step=None, training_state=False)
    model_files: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.glob("model-*.safetensors")):
        model_files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if args.hash_model_shards else None,
        }
    if not model_files:
        single_model = model_dir / "model.safetensors"
        _require_file(single_model, "Molmo2-ER source weights")
        model_files[single_model.name] = {
            "size_bytes": single_model.stat().st_size,
            "sha256": sha256_file(single_model) if args.hash_model_shards else None,
        }
    payload = {
        "created_at": utc_now(),
        "architecture": "Frozen Full-Molmo2-ER World-Ego WEP-VLA",
        "stage": {
            "label": args.stage_label,
            "kind": args.stage_kind,
            "steps": args.steps,
            "cumulative_start_step": args.cumulative_start_step,
            "cumulative_end_step": args.cumulative_start_step + args.steps,
            "resume": bool(args.resume_config),
            "resume_config": args.resume_config or None,
            "source_checkpoint": source_checkpoint,
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "grad_clip_norm": 10.0,
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "warmup_steps": 100,
            "decay_steps": 30_000,
            "decay_lr": args.decay_lr,
        },
        "seed": 1000,
        "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES_CONTRACT,
        "global_batch_contract": global_batch_contract(args.batch_profile),
        "policy_contract": POLICY_CONTRACT,
        "model_dir": str(model_dir),
        "model_source_files": model_files,
        "data_audit": data_audit,
        "code": {
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "status": _git_value(root, "status", "--short"),
            "source_sha256": _source_manifest(root),
        },
        "command": args.command,
    }
    atomic_write_json(Path(args.output), payload)
    print(json.dumps({"manifest": str(Path(args.output)), "stage": args.stage_label}, sort_keys=True))
    return 0


def command_pipeline_event(args: argparse.Namespace) -> int:
    output = Path(args.output)
    payload: dict[str, Any]
    if output.is_file():
        payload = load_json(output)
    else:
        payload = {"created_at": utc_now(), "events": []}
    event = {
        "at": utc_now(),
        "status": args.status,
        "stage": args.stage,
        "message": args.message,
        "checkpoint": args.checkpoint or None,
    }
    payload["updated_at"] = event["at"]
    payload["status"] = args.status
    payload["current_stage"] = args.stage
    payload.setdefault("events", []).append(event)
    atomic_write_json(output, payload)
    return 0


def _validate_post_adam_memory_audit(
    audit: dict[str, Any], *, batch_profile: str = DEFAULT_BATCH_PROFILE
) -> None:
    batch_contract = global_batch_contract(batch_profile)
    required = {
        "version": 1,
        "completed_optimizer_step": 1,
        "world_size": batch_contract["world_size"],
        "global_batch_size": batch_contract["global_batch_size"],
        "rank_count": batch_contract["world_size"],
    }
    drift = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in required.items()
        if audit.get(key) != expected
    }
    if drift or audit.get("mode") != "exact_global_batch_post_first_optimizer_step_cuda_memory":
        raise ValueError(f"Eight-rank post-Adam memory audit drifted: {json.dumps(drift, sort_keys=True)}")
    ranks = audit.get("ranks")
    mapping = audit.get("logical_to_physical_cuda_mapping")
    if not isinstance(ranks, list) or len(ranks) != 8 or not isinstance(mapping, list) or len(mapping) != 8:
        raise ValueError("Post-Adam CUDA memory audit must aggregate exactly eight ranks.")
    ranks_by_id = {int(record.get("global_rank", -1)): record for record in ranks}
    if set(ranks_by_id) != set(range(8)):
        raise ValueError("Post-Adam CUDA memory audit rank coverage is not exactly 0..7.")
    mapping_by_id = {int(record.get("global_rank", -1)): record for record in mapping}
    if set(mapping_by_id) != set(range(8)):
        raise ValueError("Post-Adam CUDA logical-to-physical mapping is incomplete.")
    byte_fields = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    for rank in range(8):
        record = ranks_by_id[rank]
        mapped = mapping_by_id[rank]
        physical = mapped.get("physical_cuda_device")
        if str(physical) != str(PHYSICAL_GPU_IDS[rank]):
            raise ValueError(f"Smoke rank {rank} mapped to physical CUDA device {physical}, not {PHYSICAL_GPU_IDS[rank]}.")
        if mapped.get("local_rank") != rank or mapped.get("logical_cuda_index") != rank:
            raise ValueError(f"Smoke rank {rank} has an invalid local/logical CUDA mapping.")
        if record.get("local_rank") != rank or record.get("logical_cuda_index") != rank:
            raise ValueError(f"Smoke rank {rank} record disagrees with its rank mapping.")
        if str(record.get("physical_cuda_device")) != str(PHYSICAL_GPU_IDS[rank]):
            raise ValueError(f"Smoke rank {rank} record has an invalid physical CUDA device.")
        if record.get("cuda_visible_devices") != [str(item) for item in PHYSICAL_GPU_IDS]:
            raise ValueError(f"Smoke rank {rank} did not inherit the physical GPU 0--7 visibility contract.")
        allocated, reserved, peak_allocated, peak_reserved = (
            int(record.get(field, 0)) for field in byte_fields
        )
        if allocated <= 0 or reserved < allocated or peak_allocated < allocated or peak_reserved < reserved:
            raise ValueError(f"Smoke rank {rank} contains invalid post-Adam CUDA memory values.")
        if int(record.get("optimizer_state_tensor_count", 0)) <= 0 or int(
            record.get("optimizer_state_total_numel", 0)
        ) <= 0:
            raise ValueError(f"Smoke rank {rank} did not materialize Adam optimizer state.")
        for key, expected in (
            ("global_batch_size", batch_contract["global_batch_size"]),
            ("physical_forward_samples_per_optimizer_step", batch_contract["physical_forward_samples"]),
            ("discarded_for_gradient_samples_per_optimizer_step", batch_contract["discarded_for_gradient"]),
        ):
            if record.get(key) != expected:
                raise ValueError(f"Smoke rank {rank} exact-batch field {key} drifted.")
    for aggregate_key, rank_key in (
        ("max_allocated_bytes", "allocated_bytes"),
        ("max_reserved_bytes", "reserved_bytes"),
        ("max_peak_allocated_bytes", "peak_allocated_bytes"),
        ("max_peak_reserved_bytes", "peak_reserved_bytes"),
    ):
        if int(audit.get(aggregate_key, -1)) != max(int(record[rank_key]) for record in ranks):
            raise ValueError(f"Post-Adam CUDA aggregate {aggregate_key} does not match rank records.")


def _validate_frozen_hash_audit(audit: dict[str, Any]) -> None:
    before = audit.get("before", {})
    after = audit.get("after", {})
    if (
        audit.get("version") != 1
        or audit.get("mode") != "full_molmo2er_frozen_live_parameter_before_after_hash"
        or audit.get("comparison_pass") is not True
    ):
        raise ValueError("Smoke did not preserve every frozen Molmo parameter bit-for-bit.")
    for snapshot_name, snapshot in (("before", before), ("after", after)):
        if (
            snapshot.get("algorithm") != "sha256"
            or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("sha256", "")))
            or int(snapshot.get("parameter_count", 0)) <= 0
            or int(snapshot.get("total_numel", 0)) <= 0
            or int(snapshot.get("total_bytes", 0)) <= 0
        ):
            raise ValueError(f"Frozen parameter hash {snapshot_name} snapshot is incomplete.")
    comparable = (
        "algorithm",
        "hash_scheme",
        "parameter_order",
        "chunk_bytes",
        "parameter_count",
        "total_numel",
        "total_bytes",
        "sha256",
    )
    if any(before.get(key) != after.get(key) for key in comparable):
        raise ValueError("Frozen Molmo before/after parameter hashes or coverage differ.")


def _validate_smoke_manifest_payload(
    payload: dict[str, Any],
    *,
    expected_data_hash: str | None,
    expected_batch_profile: str = DEFAULT_BATCH_PROFILE,
) -> None:
    expected_batch_contract = global_batch_contract(expected_batch_profile)
    if payload.get("status") != "complete" or int(payload.get("optimizer_steps", -1)) != 2:
        raise ValueError("Smoke marker is not a completed two-optimizer-step run.")
    if payload.get("physical_gpu_ids") != list(PHYSICAL_GPU_IDS):
        raise ValueError("Smoke marker did not use exactly physical GPUs 0--7.")
    if payload.get("global_batch_contract") != expected_batch_contract:
        raise ValueError("Smoke marker exact-gradient global-batch contract drifted.")
    if expected_data_hash and payload.get("data_semantic_hash") != expected_data_hash:
        raise ValueError("Smoke marker belongs to a different dataset/cache semantic hash.")
    parameter_audit = payload.get("parameter_audit", {})
    required_parameters = {
        "backend": "molmo2_full",
        "worldflow_enabled": True,
        "vlm_layers": 36,
        "expert_layers": 36,
        "vision_backbone_present": True,
        "molmo_frozen": True,
        "trainable_allowlist_pass": True,
    }
    if any(parameter_audit.get(key) != value for key, value in required_parameters.items()):
        raise ValueError("Smoke parameter audit is not Full+native-vision+WorldFlow 36/36.")
    _validate_post_adam_memory_audit(
        payload.get("post_adam_cuda_memory_audit", {}), batch_profile=expected_batch_profile
    )
    _validate_frozen_hash_audit(payload.get("frozen_parameter_hash_audit", {}))
    artifacts = payload.get("artifacts", {})
    expected_artifacts = {
        "launch_manifest",
        "training_log",
        "exact_global_batch_manifest",
        "parameter_audit",
        "post_adam_cuda_memory_audit",
        "frozen_parameter_hash_audit",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Smoke marker does not contain the complete evidence artifact set.")
    for artifact in artifacts.values():
        path = Path(str(artifact.get("path", "")))
        _require_file(path, "smoke evidence artifact")
        if sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"Smoke evidence artifact hash drifted: {path}")


def command_write_smoke_manifest(args: argparse.Namespace) -> int:
    batch_profile = getattr(args, "batch_profile", DEFAULT_BATCH_PROFILE)
    batch_contract = global_batch_contract(batch_profile)
    smoke_output = Path(args.smoke_output).expanduser().resolve(strict=True)
    launch_manifest_path = Path(args.launch_manifest).expanduser().resolve(strict=True)
    log_path = Path(args.log).expanduser().resolve(strict=True)
    exact_path = smoke_output / "exact_global_batch_manifest.json"
    parameter_path = smoke_output / "full_molmo2er_parameter_audit.json"
    memory_audit_path = smoke_output / "exact_global_batch_post_adam_cuda_memory_audit.json"
    frozen_hash_path = smoke_output / "full_molmo2er_frozen_parameter_hash_audit.json"
    for path, description in (
        (launch_manifest_path, "smoke launch manifest"),
        (log_path, "smoke training log"),
        (exact_path, "runtime exact-global-batch manifest"),
        (parameter_path, "runtime Full-Molmo parameter audit"),
        (memory_audit_path, "runtime eight-rank post-Adam CUDA memory audit"),
        (frozen_hash_path, "runtime frozen-Molmo before/after hash audit"),
    ):
        _require_file(path, description)
    launch_manifest = load_json(launch_manifest_path)
    exact = load_json(exact_path)
    parameter_audit = load_json(parameter_path)
    memory_audit = load_json(memory_audit_path)
    frozen_hash_audit = load_json(frozen_hash_path)
    _validate_post_adam_memory_audit(memory_audit, batch_profile=batch_profile)
    _validate_frozen_hash_audit(frozen_hash_audit)
    expected_exact = exact_global_batch_runtime_contract(batch_profile)
    drift = {
        key: {"expected": value, "actual": exact.get(key)}
        for key, value in expected_exact.items()
        if exact.get(key) != value
    }
    if drift:
        raise ValueError(f"Runtime smoke exact-global-batch drifted: {json.dumps(drift, sort_keys=True)}")
    if launch_manifest.get("policy_contract") != POLICY_CONTRACT:
        raise ValueError("Smoke launch policy contract drifted from formal Full+WorldFlow.")
    if launch_manifest.get("global_batch_contract") != batch_contract:
        raise ValueError("Smoke launch exact-gradient global-batch profile drifted.")
    data_hash = launch_manifest.get("data_audit", {}).get("semantic_hash")
    if data_hash != args.expected_data_hash:
        raise ValueError("Smoke launch used a different dataset/cache semantic hash.")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"\bstep:2\b", log_text):
        raise ValueError("Smoke log does not prove completion of optimizer step 2.")
    fatal_pattern = re.compile(
        r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|"
        r"ChildFailedError|NCCL[^\n]*(?:error|timeout)|Segmentation fault|Killed(?:\r?$)",
        re.IGNORECASE | re.MULTILINE,
    )
    fatal = fatal_pattern.search(log_text)
    if fatal:
        raise ValueError(f"Smoke log contains a fatal signature: {fatal.group(0)}")
    for metric_name in ("skipped_nonfinite_loss", "skipped_nonfinite_grad"):
        values = [float(value) for value in re.findall(rf"{metric_name}:([0-9.eE+-]+)", log_text)]
        if any(value != 0.0 for value in values):
            raise ValueError(f"Smoke log reports {metric_name} > 0.")
    payload = {
        "status": "complete",
        "completed_at": utc_now(),
        "architecture": "Frozen Full-Molmo2-ER World-Ego WEP-VLA",
        "optimizer": "AdamW",
        "optimizer_steps": 2,
        "adam_state_materialized_before_measurement": True,
        "memory_observed_after_optimizer_step": 1,
        "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
        "global_batch_contract": batch_contract,
        "data_semantic_hash": data_hash,
        "parameter_audit": parameter_audit,
        "post_adam_cuda_memory_audit": memory_audit,
        "frozen_parameter_hash_audit": frozen_hash_audit,
        "artifacts": {
            "launch_manifest": {
                "path": str(launch_manifest_path),
                "sha256": sha256_file(launch_manifest_path),
            },
            "training_log": {"path": str(log_path), "sha256": sha256_file(log_path)},
            "exact_global_batch_manifest": {
                "path": str(exact_path),
                "sha256": sha256_file(exact_path),
            },
            "parameter_audit": {
                "path": str(parameter_path),
                "sha256": sha256_file(parameter_path),
            },
            "post_adam_cuda_memory_audit": {
                "path": str(memory_audit_path),
                "sha256": sha256_file(memory_audit_path),
            },
            "frozen_parameter_hash_audit": {
                "path": str(frozen_hash_path),
                "sha256": sha256_file(frozen_hash_path),
            },
        },
    }
    _validate_smoke_manifest_payload(
        payload,
        expected_data_hash=args.expected_data_hash,
        expected_batch_profile=batch_profile,
    )
    atomic_write_json(Path(args.output), payload)
    print(
        json.dumps(
            {
                "rank_count": memory_audit["rank_count"],
                "max_peak_reserved_bytes": memory_audit["max_peak_reserved_bytes"],
                "frozen_hash_comparison_pass": frozen_hash_audit["comparison_pass"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_validate_smoke(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.manifest))
    _validate_smoke_manifest_payload(
        payload,
        expected_data_hash=args.expected_data_hash,
        expected_batch_profile=args.batch_profile,
    )
    return 0


def command_write_eval_manifest(args: argparse.Namespace) -> int:
    _assert_eval_plan()
    root = Path(args.experiment_root).expanduser().resolve(strict=True)
    checkpoint_66 = audit_checkpoint(
        Path(args.checkpoint_066000), expected_step=None, training_state=False
    )
    checkpoint_96 = audit_checkpoint(
        Path(args.checkpoint_096000), expected_step=None, training_state=False
    )
    payload = {
        "created_at": utc_now(),
        "architecture": "Frozen Full-Molmo2-ER World-Ego WEP-VLA",
        "checkpoint_066000": checkpoint_66,
        "checkpoint_096000": checkpoint_96,
        "protocol": {
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "episodes_per_task": 50,
            "episodes_per_checkpoint": 2000,
            "exec_action_steps": 16,
            "adaptive_exec_max_steps": 16,
            "grasp_exec_steps": 16,
            "strict_official_init": True,
            "save_video": False,
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
            "static_shards": [asdict(item) for item in EVAL_SHARDS],
        },
        "code": {
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "status": _git_value(root, "status", "--short"),
            "source_sha256": _source_manifest(root),
        },
    }
    atomic_write_json(Path(args.output), payload)
    return 0


def _episodes_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [episode for suite in report.get("suites", []) for task in suite.get("tasks", []) for episode in task.get("episodes", [])]


def validate_shard_report(
    report: dict[str, Any], *, suite_name: str, task_ids: Sequence[int], episodes_per_task: int
) -> None:
    suites = report.get("suites", [])
    if len(suites) != 1 or suites[0].get("suite") != suite_name:
        raise ValueError(f"Shard report does not contain exactly suite {suite_name}.")
    tasks = suites[0].get("tasks", [])
    if sorted(int(task["task_id"]) for task in tasks) != sorted(task_ids):
        raise ValueError(f"Shard task IDs drifted for {suite_name}: {[task.get('task_id') for task in tasks]}")
    for task in tasks:
        episodes = task.get("episodes", [])
        if len(episodes) != episodes_per_task:
            raise ValueError(f"Task {suite_name}:{task['task_id']} has {len(episodes)} episodes.")
        if sorted(int(episode["episode_index"]) for episode in episodes) != list(range(episodes_per_task)):
            raise ValueError(f"Task {suite_name}:{task['task_id']} episode indices are incomplete.")
        for episode in episodes:
            if episode.get("error") is not None or not isinstance(episode.get("success"), bool):
                raise ValueError(f"Task {suite_name}:{task['task_id']} contains an invalid episode result.")
            if int(episode.get("model_call_count", 0)) <= 0 or int(episode.get("policy_forward_call_count", 0)) <= 0:
                raise ValueError("An evaluation episode did not execute the policy.")
            if episode.get("strict_official_init") is not True:
                raise ValueError("Evaluation did not use strict official initialization.")
            if int(episode.get("adaptive_exec_max_steps", -1)) != 16:
                raise ValueError("Evaluation adaptive execution horizon is not 16.")
            if int(episode.get("grasp_exec_steps", -1)) != 16:
                raise ValueError("Evaluation grasp execution horizon is not 16.")
    expected_episodes = len(task_ids) * episodes_per_task
    episodes = _episodes_from_report(report)
    if len(episodes) != expected_episodes:
        raise ValueError(f"Shard contains {len(episodes)} episodes, expected {expected_episodes}.")
    success_count = sum(bool(episode["success"]) for episode in episodes)
    overall = report.get("overall", {})
    if int(overall.get("episode_count", -1)) != expected_episodes or int(overall.get("success_count", -1)) != success_count:
        raise ValueError("Shard overall counts do not agree with its episode records.")


def command_validate_shard(args: argparse.Namespace) -> int:
    task_ids = tuple(int(item) for item in args.task_ids.split(",") if item)
    validate_shard_report(
        load_json(Path(args.report)),
        suite_name=args.suite,
        task_ids=task_ids,
        episodes_per_task=args.episodes,
    )
    return 0


def _task_summary(task: dict[str, Any]) -> dict[str, Any]:
    result = dict(task)
    episodes = result["episodes"]
    successes = sum(bool(episode["success"]) for episode in episodes)
    result["episode_count"] = len(episodes)
    result["success_count"] = successes
    result["success_rate"] = successes / len(episodes)
    return result


def merge_eval_reports(paths: Iterable[Path]) -> dict[str, Any]:
    _assert_eval_plan()
    tasks_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        report = load_json(path)
        for suite in report.get("suites", []):
            for task in suite.get("tasks", []):
                key = (str(task["suite"]), int(task["task_id"]))
                if key in tasks_by_key:
                    raise ValueError(f"Duplicate evaluation task across shards: {key}")
                tasks_by_key[key] = _task_summary(task)
    expected_keys = {(suite, task_id) for suite in SUITES for task_id in range(10)}
    if set(tasks_by_key) != expected_keys:
        missing = sorted(expected_keys - set(tasks_by_key))
        extra = sorted(set(tasks_by_key) - expected_keys)
        raise ValueError(f"Evaluation shard union is incomplete: missing={missing}, extra={extra}")
    suites: list[dict[str, Any]] = []
    for suite_name in SUITES:
        tasks = [tasks_by_key[(suite_name, task_id)] for task_id in range(10)]
        episodes = [episode for task in tasks for episode in task["episodes"]]
        successes = sum(bool(episode["success"]) for episode in episodes)
        suites.append(
            {
                "suite": suite_name,
                "task_count": 10,
                "episode_count": len(episodes),
                "success_count": successes,
                "success_rate": successes / len(episodes),
                "task_success_rate_mean": sum(float(task["success_rate"]) for task in tasks) / 10,
                "tasks": tasks,
            }
        )
    all_tasks = [task for suite in suites for task in suite["tasks"]]
    all_episodes = [episode for task in all_tasks for episode in task["episodes"]]
    if len(all_episodes) != 2000:
        raise ValueError(f"Merged evaluation has {len(all_episodes)} episodes, expected 2000.")
    successes = sum(bool(episode["success"]) for episode in all_episodes)
    return {
        "overall": {
            "task_count": 40,
            "episode_count": 2000,
            "success_count": successes,
            "success_rate": successes / 2000,
            "task_success_rate_mean": sum(float(task["success_rate"]) for task in all_tasks) / 40,
        },
        "suites": suites,
    }


def command_merge_eval(args: argparse.Namespace) -> int:
    report = merge_eval_reports(Path(path) for path in args.shard)
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


def _validate_overall(report: dict[str, Any]) -> None:
    if report.get("overall", {}).get("task_count") != 40 or report.get("overall", {}).get("episode_count") != 2000:
        raise ValueError("Overall report is not the required 40-task/2000-episode evaluation.")
    expected = {(suite, task_id) for suite in SUITES for task_id in range(10)}
    actual = {
        (str(task["suite"]), int(task["task_id"]))
        for suite in report.get("suites", [])
        for task in suite.get("tasks", [])
    }
    if actual != expected or len(_episodes_from_report(report)) != 2000:
        raise ValueError("Overall report task/episode coverage is incomplete.")


def command_validate_overall(args: argparse.Namespace) -> int:
    _validate_overall(load_json(Path(args.report)))
    return 0


def command_write_comparison(args: argparse.Namespace) -> int:
    report_66 = load_json(Path(args.report_066000))
    report_96 = load_json(Path(args.report_096000))
    _validate_overall(report_66)
    _validate_overall(report_96)
    checkpoint_66 = Path(args.checkpoint_066000).expanduser().resolve(strict=True)
    checkpoint_96 = Path(args.checkpoint_096000).expanduser().resolve(strict=True)
    payload = {
        "generated_at": utc_now(),
        "protocol": {
            "suites": list(SUITES),
            "task_count": 40,
            "episodes_per_task": 50,
            "episode_count_per_checkpoint": 2000,
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
            "static_shards": [asdict(item) for item in EVAL_SHARDS],
            "exec_action_steps": 16,
            "adaptive_exec_max_steps": 16,
            "grasp_exec_steps": 16,
            "strict_official_init": True,
        },
        "checkpoint_066000": dict(report_66["overall"], path=str(checkpoint_66)),
        "checkpoint_096000": dict(report_96["overall"], path=str(checkpoint_96)),
    }
    atomic_write_json(Path(args.output), payload)
    tsv_path = Path(args.tsv_output)
    rows = ["label\tcheckpoint\tepisodes\tsuccesses\tsuccess_rate"]
    for label in ("checkpoint_066000", "checkpoint_096000"):
        item = payload[label]
        rows.append(
            f"{label}\t{item['path']}\t{item['episode_count']}\t{item['success_count']}\t{item['success_rate']:.9f}"
        )
    _atomic_write_text(tsv_path, "\n".join(rows) + "\n")
    print(json.dumps({key: payload[key] for key in ("checkpoint_066000", "checkpoint_096000")}, sort_keys=True))
    return 0


def command_json_get(args: argparse.Namespace) -> int:
    value: Any = load_json(Path(args.file))
    for component in args.path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"JSON path {args.path!r} is absent from {args.file}.")
        value = value[component]
    if isinstance(value, (dict, list)):
        print(json.dumps(value, sort_keys=True))
    elif value is None:
        print("null")
    elif isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument(
        "--batch-profile",
        choices=tuple(BATCH_PROFILE_PARAMETERS),
        default=DEFAULT_BATCH_PROFILE,
    )
    contract_parser.set_defaults(func=command_contract)

    eval_plan_parser = subparsers.add_parser("eval-plan")
    eval_plan_parser.add_argument("--json", action="store_true")
    eval_plan_parser.set_defaults(func=command_eval_plan)

    gpu_parser = subparsers.add_parser("gpu-audit")
    gpu_parser.add_argument("--min-free-mib", type=int, required=True)
    gpu_parser.add_argument("--output")
    gpu_parser.set_defaults(func=command_gpu_audit)

    model_parser = subparsers.add_parser("model-audit")
    model_parser.add_argument("--model-dir", required=True)
    model_parser.add_argument("--output")
    model_parser.set_defaults(func=command_model_audit)

    data_parser = subparsers.add_parser("data-audit")
    data_parser.add_argument("--dataset", required=True)
    data_parser.add_argument("--cache", required=True)
    data_parser.add_argument("--expected-hash")
    data_parser.add_argument("--output")
    data_parser.set_defaults(func=command_data_audit)

    checkpoint_parser = subparsers.add_parser("checkpoint-audit")
    checkpoint_parser.add_argument("--checkpoint", required=True)
    checkpoint_parser.add_argument("--expected-step", type=int)
    checkpoint_parser.add_argument("--training-state", action="store_true")
    checkpoint_parser.add_argument("--output")
    checkpoint_parser.set_defaults(func=command_checkpoint_audit)

    resume_parser = subparsers.add_parser("find-resume")
    resume_parser.add_argument("--output-dir", required=True)
    resume_parser.add_argument("--target-step", type=int, required=True)
    resume_parser.add_argument("--json", action="store_true")
    resume_parser.set_defaults(func=command_find_resume)

    alias_parser = subparsers.add_parser("publish-alias")
    alias_parser.add_argument("--source", required=True)
    alias_parser.add_argument("--source-step", type=int, required=True)
    alias_parser.add_argument("--cumulative-step", type=int, required=True)
    alias_parser.add_argument("--alias", required=True)
    alias_parser.add_argument("--output")
    alias_parser.set_defaults(func=command_publish_alias)

    manifest_parser = subparsers.add_parser("write-stage-manifest")
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--experiment-root", required=True)
    manifest_parser.add_argument("--model-dir", required=True)
    manifest_parser.add_argument("--data-audit", required=True)
    manifest_parser.add_argument("--stage-label", required=True)
    manifest_parser.add_argument("--stage-kind", choices=("fresh", "finetune"), required=True)
    manifest_parser.add_argument("--steps", type=int, required=True)
    manifest_parser.add_argument("--cumulative-start-step", type=int, required=True)
    manifest_parser.add_argument("--decay-lr", type=float, required=True)
    manifest_parser.add_argument("--policy-path")
    manifest_parser.add_argument("--resume-config")
    manifest_parser.add_argument(
        "--batch-profile",
        choices=tuple(BATCH_PROFILE_PARAMETERS),
        default=DEFAULT_BATCH_PROFILE,
    )
    hash_group = manifest_parser.add_mutually_exclusive_group()
    hash_group.add_argument("--hash-model-shards", dest="hash_model_shards", action="store_true")
    hash_group.add_argument("--no-hash-model-shards", dest="hash_model_shards", action="store_false")
    manifest_parser.set_defaults(hash_model_shards=True)
    manifest_parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    manifest_parser.set_defaults(func=command_write_stage_manifest)

    event_parser = subparsers.add_parser("pipeline-event")
    event_parser.add_argument("--output", required=True)
    event_parser.add_argument("--status", required=True)
    event_parser.add_argument("--stage", required=True)
    event_parser.add_argument("--message", required=True)
    event_parser.add_argument("--checkpoint")
    event_parser.set_defaults(func=command_pipeline_event)

    smoke_parser = subparsers.add_parser("write-smoke-manifest")
    smoke_parser.add_argument("--output", required=True)
    smoke_parser.add_argument("--smoke-output", required=True)
    smoke_parser.add_argument("--launch-manifest", required=True)
    smoke_parser.add_argument("--log", required=True)
    smoke_parser.add_argument("--expected-data-hash", required=True)
    smoke_parser.add_argument(
        "--batch-profile",
        choices=tuple(BATCH_PROFILE_PARAMETERS),
        default=DEFAULT_BATCH_PROFILE,
    )
    smoke_parser.set_defaults(func=command_write_smoke_manifest)

    validate_smoke_parser = subparsers.add_parser("validate-smoke")
    validate_smoke_parser.add_argument("--manifest", required=True)
    validate_smoke_parser.add_argument("--expected-data-hash")
    validate_smoke_parser.add_argument(
        "--batch-profile",
        choices=tuple(BATCH_PROFILE_PARAMETERS),
        default=DEFAULT_BATCH_PROFILE,
    )
    validate_smoke_parser.set_defaults(func=command_validate_smoke)

    eval_manifest_parser = subparsers.add_parser("write-eval-manifest")
    eval_manifest_parser.add_argument("--output", required=True)
    eval_manifest_parser.add_argument("--experiment-root", required=True)
    eval_manifest_parser.add_argument("--checkpoint-066000", required=True)
    eval_manifest_parser.add_argument("--checkpoint-096000", required=True)
    eval_manifest_parser.set_defaults(func=command_write_eval_manifest)

    shard_parser = subparsers.add_parser("validate-shard")
    shard_parser.add_argument("--report", required=True)
    shard_parser.add_argument("--suite", choices=SUITES, required=True)
    shard_parser.add_argument("--task-ids", required=True)
    shard_parser.add_argument("--episodes", type=int, default=50)
    shard_parser.set_defaults(func=command_validate_shard)

    merge_parser = subparsers.add_parser("merge-eval")
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--shard", action="append", required=True)
    merge_parser.set_defaults(func=command_merge_eval)

    overall_parser = subparsers.add_parser("validate-overall")
    overall_parser.add_argument("--report", required=True)
    overall_parser.set_defaults(func=command_validate_overall)

    comparison_parser = subparsers.add_parser("write-comparison")
    comparison_parser.add_argument("--output", required=True)
    comparison_parser.add_argument("--tsv-output", required=True)
    comparison_parser.add_argument("--checkpoint-066000", required=True)
    comparison_parser.add_argument("--checkpoint-096000", required=True)
    comparison_parser.add_argument("--report-066000", required=True)
    comparison_parser.add_argument("--report-096000", required=True)
    comparison_parser.set_defaults(func=command_write_comparison)

    json_parser = subparsers.add_parser("json-get")
    json_parser.add_argument("--file", required=True)
    json_parser.add_argument("--path", required=True)
    json_parser.set_defaults(func=command_json_get)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[full-molmo2er-tools] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
