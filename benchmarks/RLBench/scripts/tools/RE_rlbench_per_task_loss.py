#!/usr/bin/env python3
"""Measure the trained policy loss separately for selected RLBench tasks.

This is a diagnostic script. It does not modify the training script, dataset, or
checkpoint. The forward pass is the same policy.forward() used during training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


from _rlbench_tool_paths import LEROBOT_ROOT as REPO_ROOT, SONG_SCRIPTS_DIR


SONG_SCRIPTS = SONG_SCRIPTS_DIR
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SONG_SCRIPTS))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from train_song_benchmark import (
    make_song_train_collate_fn,
    maybe_wrap_point_cloud_memmap_dataset,
    maybe_wrap_pointseg_cache_dataset,
    maybe_wrap_worldflow_dataset,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "benchmarks/RLBench/outputs/wep_vla_v041_rlbench_08051103/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_DATASET = REPO_ROOT / "benchmarks/RLBench/datasets/rlbench_all_tasks_lerobot_20260804_211804"
DEFAULT_CACHE = REPO_ROOT / "benchmarks/RLBench/datasets/rlbench_all_tasks_lerobot_20260804_211804_pointseg_cache"

TASK_ALIASES = {
    "water_plants": "water plant",
    "sweep_to_dustpan": "sweep dirt to dustpan",
    "toilet_seat_down": "toilet seat down",
    "close_fridge": "close fridge",
    "close_box": "close box",
    "close_laptop_lid": "close laptop lid",
    "phone_on_base": "phone on base",
    "put_the_phone_on_the_base": "phone on base",
    "stack_wine": "stack wine",
    "take_frame_off_hanger": "take frame off hanger",
    "take_umbrella_out_of_umbrella_stand": "take umbrella out of umbrella stand",
}


def canonical_task(value: str) -> str:
    value = " ".join(str(value).strip().lower().replace("_", " ").split())
    return TASK_ALIASES.get(value.replace(" ", "_"), value)


def parse_tasks(value: str) -> list[str]:
    tasks = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task name")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Task names separated by spaces or commas")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples-per-task", type=int, default=0, help="0 means all samples")
    parser.add_argument(
        "--sample-strategy",
        choices=("head", "uniform"),
        default="head",
        help="When limiting samples, take the first frames or spread samples uniformly over all episodes",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-offload-vlm",
        action="store_true",
        help="Keep the VLM/action transformer stack on CPU and stream its layers to CUDA",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("train_loss", "eval_loss"),
        default="train_loss",
        help="train_loss includes the PointSeg auxiliary term like training; eval_loss measures inference mode",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def select_task_indices(
    dataset,
    requested_tasks: list[str],
    max_samples: int,
    sample_strategy: str,
) -> dict[str, list[int]]:
    wanted = {canonical_task(task): task for task in requested_tasks}
    selected = {task: [] for task in requested_tasks}
    episodes = dataset.meta.episodes
    for task_values, start, end in zip(
        episodes["tasks"], episodes["dataset_from_index"], episodes["dataset_to_index"]
    ):
        recorded_task = task_values[0] if task_values else ""
        requested_task = wanted.get(canonical_task(recorded_task))
        if requested_task is None:
            continue
        episode_indices = list(range(int(start), int(end)))
        selected[requested_task].extend(episode_indices)

    if max_samples > 0:
        for task, indices in selected.items():
            if len(indices) <= max_samples:
                continue
            if sample_strategy == "uniform":
                positions = np.linspace(0, len(indices) - 1, max_samples, dtype=np.int64)
                selected[task] = [indices[int(position)] for position in positions]
            else:
                selected[task] = indices[:max_samples]

    missing = [task for task, indices in selected.items() if not indices]
    if missing:
        available = sorted({canonical_task(values[0]) for values in episodes["tasks"] if values})
        raise ValueError(f"No samples found for {missing}; available tasks: {available}")
    return selected


def build_policy(cfg, checkpoint: Path, dataset, device: str, mode: str, cpu_offload_vlm: bool):
    cfg.policy.pretrained_path = checkpoint
    cfg.policy.device = "cpu" if cpu_offload_vlm else device
    cfg.policy.use_amp = False
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if cpu_offload_vlm:
        if not device.startswith("cuda"):
            raise ValueError("--cpu-offload-vlm requires a CUDA --device")
        from accelerate import cpu_offload

        transformer_stack = policy.model.vlm_with_expert
        policy.model.vlm_with_expert = None
        policy = policy.to(device)
        policy.model.vlm_with_expert = transformer_stack
        cpu_offload(
            transformer_stack,
            execution_device=torch.device(device),
            offload_buffers=True,
        )
        policy.config.device = device
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=checkpoint,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    if mode == "train_loss":
        # The model only computes PointSeg auxiliary supervision while training.
        policy.train()
    else:
        policy.eval()
    return policy, preprocessor


def summarize_task(
    policy,
    preprocessor,
    dataset,
    indices,
    collate_fn,
    batch_size,
    num_workers,
):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    total_values = []
    action_sum = 0.0
    pointseg_sum = 0.0
    sample_count = 0
    for batch in loader:
        batch = preprocessor(batch)
        with torch.inference_mode():
            per_sample_loss, output = policy.forward(batch, reduction="none")
        values = per_sample_loss.detach().float().cpu().numpy()
        count = int(values.size)
        total_values.extend(values.tolist())
        action_sum += float(output.get("loss_action", 0.0)) * count
        pointseg_sum += float(output.get("loss_pointseg_aux", 0.0)) * count
        sample_count += count

    values = np.asarray(total_values, dtype=np.float64)
    trim = int(values.size * 0.1)
    trimmed_values = np.sort(values)[trim:-trim] if trim > 0 else values
    return {
        "samples": sample_count,
        "loss_total_mean": float(values.mean()),
        "loss_total_std": float(values.std()),
        "loss_total_min": float(values.min()),
        "loss_total_q10": float(np.quantile(values, 0.10)),
        "loss_total_median": float(np.median(values)),
        "loss_total_q90": float(np.quantile(values, 0.90)),
        "loss_total_max": float(values.max()),
        "loss_total_trimmed_mean_10pct": float(trimmed_values.mean()),
        "loss_action_mean": action_sum / sample_count,
        "loss_pointseg_aux_mean": pointseg_sum / sample_count,
        "pointseg_aux_weight": float(policy.config.pointseg_aux_loss_weight),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples_per_task < 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive; sample limit and workers cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu only for policies without LitePT")

    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    set_seed(args.seed)
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_root}")

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.dataset.repo_id = str(dataset_root)
    cfg.dataset.root = str(dataset_root)
    cfg.pointseg_sample_cache_dir = str(cache_root) if cache_root.is_dir() else ""
    dataset = make_dataset(cfg)
    dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
    dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
    dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)

    requested_tasks = parse_tasks(args.tasks)
    selected = select_task_indices(
        dataset,
        requested_tasks,
        args.max_samples_per_task,
        args.sample_strategy,
    )
    policy, preprocessor = build_policy(
        cfg,
        checkpoint,
        dataset,
        args.device,
        args.mode,
        args.cpu_offload_vlm,
    )
    collate_fn = make_song_train_collate_fn(dataset)
    results = {}
    for task in requested_tasks:
        results[task] = summarize_task(
            policy,
            preprocessor,
            dataset,
            selected[task],
            collate_fn,
            args.batch_size,
            args.num_workers,
        )
        print(f"{task}: {json.dumps(results[task], ensure_ascii=True)}", flush=True)

    output = args.output.expanduser().resolve() if args.output else dataset_root.parent / f"{dataset_root.name}_per_task_loss.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "dataset_root": str(dataset_root),
                "cache_root": str(cache_root) if cache_root.is_dir() else None,
                "seed": args.seed,
                "mode": args.mode,
                "cpu_offload_vlm": args.cpu_offload_vlm,
                "sample_strategy": args.sample_strategy,
                "max_samples_per_task": args.max_samples_per_task,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
