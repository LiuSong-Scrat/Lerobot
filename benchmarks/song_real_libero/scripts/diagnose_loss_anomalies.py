#!/usr/bin/env python
"""Find task/frame groups that produce unusually large SmolVLA losses.

This is an evaluation-only diagnostic. It does not create an optimizer, update
the checkpoint, or start a W&B run. It uses the same dataset wrappers,
pointseg cache, collate function, and processors as the benchmark training
script, then records per-sample and per-batch loss statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Keep imports identical to the benchmark checkout when this file is run
# directly rather than through the package entry point.
SCRIPT_DIR = Path(__file__).resolve().parent
LEROBOT_ROOT = SCRIPT_DIR.parents[2]
if str(LEROBOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LEROBOT_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.import_utils import register_third_party_plugins

from train_song_benchmark import (
    make_song_train_collate_fn,
    maybe_wrap_point_cloud_memmap_dataset,
    maybe_wrap_pointseg_cache_dataset,
)


DEFAULT_DATASET = "/opt/data/private/liusong/rlbench/home/rlbench_all_tasks_100traj_lerobot_raw_20260807_123258"
DEFAULT_POINTSEG_CACHE = (
    "/opt/data/private/liusong/rlbench/home/rlbench_all_tasks_100traj_lerobot_raw_20260807_123258_pointseg_cache"
)
DEFAULT_CHECKPOINT = (
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/output/checkpoints/002000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--pointseg-cache", default=DEFAULT_POINTSEG_CACHE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-task", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def tensor_column(batch: dict[str, Any], key: str, default: int = -1) -> np.ndarray:
    value = batch.get(key)
    if value is None:
        return np.full(0, default, dtype=np.int64)
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.samples_per_task <= 0 or args.batch_size <= 0:
        raise ValueError("--samples-per-task and --batch-size must be positive")

    os.environ.setdefault("SONG_POINTSEG_CACHE_STRICT", "1")
    os.environ.setdefault("SONG_POINTCLOUD_MMAP_MODE", "r")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset).expanduser().resolve()
    pointseg_cache = Path(args.pointseg_cache).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset not found: {dataset_root}")
    if not pointseg_cache.is_dir():
        raise FileNotFoundError(f"pointseg cache not found: {pointseg_cache}")

    output_dir = Path(args.output_dir or (SCRIPT_DIR.parent.parent / "output" / "loss_diagnostics"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    logging.info("Loading checkpoint=%s on device=%s", checkpoint, device)

    metadata = LeRobotDatasetMetadata("loss-diagnostic", root=dataset_root)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint), local_files_only=True)
    policy.to(device)
    # Match the training forward path. RNG is reset before each batch so task
    # comparisons use the same deterministic flow-matching noise schedule.
    policy.train()

    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        "loss-diagnostic",
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    dataset = maybe_wrap_pointseg_cache_dataset(dataset, pointseg_cache, policy.config)
    dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
    collate_fn = make_song_train_collate_fn(dataset)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    # Load only the task_index column. The raw dataset index remains the cache
    # key, so the cache alignment checks stay active for every diagnostic item.
    dataset._ensure_hf_dataset_loaded()
    task_indices = np.asarray(dataset.hf_dataset["task_index"], dtype=np.int64)
    task_names = {
        task_id: str(dataset.meta.tasks.iloc[task_id].name)
        for task_id in sorted(np.unique(task_indices).tolist())
    }
    rng = np.random.default_rng(args.seed)
    selected_indices: list[int] = []
    selected_task_ids: list[int] = []
    for task_id in task_names:
        candidates = np.flatnonzero(task_indices == task_id)
        sample_count = min(args.samples_per_task, len(candidates))
        chosen = np.sort(rng.choice(candidates, size=sample_count, replace=False))
        selected_indices.extend(int(index) for index in chosen)
        selected_task_ids.extend([task_id] * sample_count)
        logging.info(
            "task=%d name=%s dataset_frames=%d sampled=%d",
            task_id,
            task_names[task_id],
            len(candidates),
            sample_count,
        )

    subset = Subset(dataset, selected_indices)
    worker_count = max(0, int(args.num_workers))
    loader_kwargs: dict[str, Any] = {
        "dataset": subset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "collate_fn": collate_fn,
    }
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(**loader_kwargs)

    batch_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    pointseg_weight = float(getattr(policy.config, "pointseg_aux_loss_weight", 0.0))
    total_samples = 0
    logging.info(
        "Starting diagnostic: tasks=%d samples=%d batches=%d batch_size=%d workers=%d",
        len(task_names),
        len(selected_indices),
        (len(selected_indices) + args.batch_size - 1) // args.batch_size,
        args.batch_size,
        worker_count,
    )

    with torch.inference_mode():
        for batch_no, raw_batch in enumerate(dataloader):
            torch.manual_seed(args.seed + batch_no)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + batch_no)
            batch = preprocessor(raw_batch)
            per_sample_loss, output = policy.forward(batch, reduction="none")
            per_sample_loss = per_sample_loss.detach().float().cpu().numpy()
            aux_loss = scalar(output.get("loss_pointseg_aux", 0.0))
            aux_contribution = pointseg_weight * aux_loss
            action_loss = per_sample_loss - aux_contribution

            task_ids = tensor_column(raw_batch, "task_index")
            episode_ids = tensor_column(raw_batch, "episode_index")
            frame_ids = tensor_column(raw_batch, "frame_index")
            dataset_ids = tensor_column(raw_batch, "index")
            if len(task_ids) != len(per_sample_loss):
                raise RuntimeError(
                    f"metadata/loss size mismatch at batch {batch_no}: "
                    f"metadata={len(task_ids)} loss={len(per_sample_loss)}"
                )

            unique_tasks, counts = np.unique(task_ids, return_counts=True)
            task_mix = ",".join(f"{task_names.get(int(t), t)}:{int(c)}" for t, c in zip(unique_tasks, counts))
            batch_rows.append(
                {
                    "batch": batch_no,
                    "task_mix": task_mix,
                    "samples": len(per_sample_loss),
                    "loss_mean": float(per_sample_loss.mean()),
                    "loss_std": float(per_sample_loss.std()),
                    "loss_p90": float(np.percentile(per_sample_loss, 90)),
                    "loss_max": float(per_sample_loss.max()),
                    "action_loss_mean": float(action_loss.mean()),
                    "pointseg_aux": aux_loss,
                    "pointseg_aux_contribution": aux_contribution,
                }
            )
            for row_idx, loss_value in enumerate(per_sample_loss):
                task_id = int(task_ids[row_idx])
                sample_rows.append(
                    {
                        "batch": batch_no,
                        "sample_in_batch": row_idx,
                        "task_id": task_id,
                        "task": task_names.get(task_id, str(task_id)),
                        "dataset_index": int(dataset_ids[row_idx]) if len(dataset_ids) else -1,
                        "episode_index": int(episode_ids[row_idx]) if len(episode_ids) else -1,
                        "frame_index": int(frame_ids[row_idx]) if len(frame_ids) else -1,
                        "loss": float(loss_value),
                        "action_loss": float(action_loss[row_idx]),
                        "pointseg_aux": aux_loss,
                        "pointseg_aux_contribution": aux_contribution,
                    }
                )
            total_samples += len(per_sample_loss)
            logging.info(
                "batch=%d/%d tasks=%s loss_mean=%.6f p90=%.6f max=%.6f action_mean=%.6f aux=%.6f",
                batch_no + 1,
                (len(selected_indices) + args.batch_size - 1) // args.batch_size,
                task_mix,
                float(per_sample_loss.mean()),
                float(np.percentile(per_sample_loss, 90)),
                float(per_sample_loss.max()),
                float(action_loss.mean()),
                aux_loss,
            )

    task_rows: list[dict[str, Any]] = []
    for task_id, task_name in task_names.items():
        rows = [row for row in sample_rows if row["task_id"] == task_id]
        values = np.asarray([row["loss"] for row in rows], dtype=np.float64)
        action_values = np.asarray([row["action_loss"] for row in rows], dtype=np.float64)
        task_rows.append(
            {
                "task_id": task_id,
                "task": task_name,
                "dataset_frames": int(np.sum(task_indices == task_id)),
                "sampled": len(rows),
                "loss_mean": float(values.mean()),
                "loss_std": float(values.std()),
                "loss_p90": float(np.percentile(values, 90)),
                "loss_max": float(values.max()),
                "action_loss_mean": float(action_values.mean()),
                "action_loss_p90": float(np.percentile(action_values, 90)),
                "action_loss_max": float(action_values.max()),
            }
        )
    task_rows.sort(key=lambda row: row["loss_mean"], reverse=True)
    batch_rows_by_loss = sorted(batch_rows, key=lambda row: row["loss_mean"], reverse=True)
    sample_rows_by_loss = sorted(sample_rows, key=lambda row: row["loss"], reverse=True)

    write_csv(
        run_dir / "task_summary.csv",
        task_rows,
        [
            "task_id",
            "task",
            "dataset_frames",
            "sampled",
            "loss_mean",
            "loss_std",
            "loss_p90",
            "loss_max",
            "action_loss_mean",
            "action_loss_p90",
            "action_loss_max",
        ],
    )
    write_csv(
        run_dir / "batch_anomalies.csv",
        batch_rows_by_loss,
        list(batch_rows[0].keys()) if batch_rows else [],
    )
    write_csv(
        run_dir / "sample_anomalies.csv",
        sample_rows_by_loss[: args.top_k],
        list(sample_rows[0].keys()) if sample_rows else [],
    )
    summary = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "pointseg_cache": str(pointseg_cache),
        "device": str(device),
        "samples_per_task": args.samples_per_task,
        "batch_size": args.batch_size,
        "num_workers": worker_count,
        "total_samples": total_samples,
        "pointseg_aux_weight": pointseg_weight,
        "top_tasks": task_rows[:5],
        "top_batches": batch_rows_by_loss[:10],
        "top_samples": sample_rows_by_loss[: min(args.top_k, 20)],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    logging.info("Finished. Results: %s", run_dir)
    logging.info("Top tasks by mean loss:")
    for row in task_rows:
        logging.info(
            "  %s | mean=%.6f p90=%.6f max=%.6f action_mean=%.6f",
            row["task"],
            row["loss_mean"],
            row["loss_p90"],
            row["loss_max"],
            row["action_loss_mean"],
        )


if __name__ == "__main__":
    main()
