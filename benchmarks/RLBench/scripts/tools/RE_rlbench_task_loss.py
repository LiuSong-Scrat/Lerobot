#!/usr/bin/env python3
"""Report checkpoint loss separately for selected RLBench tasks."""

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


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "benchmarks/RLBench/outputs/wep_vla_v041_rlbench_08051103/checkpoints/last/pretrained_model"
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
        raise ValueError("--tasks must contain at least one task name.")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Task names, separated by spaces or commas.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples-per-task", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def select_task_indices(dataset, requested_tasks: list[str], max_samples: int) -> dict[str, list[int]]:
    wanted = {canonical_task(task): task for task in requested_tasks}
    selected = {task: [] for task in requested_tasks}
    episodes = dataset.meta.episodes
    for task_values, start, end in zip(
        episodes["tasks"],
        episodes["dataset_from_index"],
        episodes["dataset_to_index"],
    ):
        recorded_task = task_values[0] if task_values else ""
        canonical = canonical_task(recorded_task)
        if canonical not in wanted:
            continue
        requested = wanted[canonical]
        indices = list(range(int(start), int(end)))
        if max_samples > 0:
            remaining = max_samples - len(selected[requested])
            indices = indices[: max(remaining, 0)]
        selected[requested].extend(indices)
    missing = [task for task, indices in selected.items() if not indices]
    if missing:
        available = sorted({canonical_task(values[0]) for values in episodes["tasks"] if values})
        raise ValueError(f"No samples found for {missing}; available tasks: {available}")
    return selected


def make_eval_pipeline(cfg, checkpoint: Path, dataset, device: str):
    cfg.policy.pretrained_path = checkpoint
    cfg.policy.device = device
    cfg.policy.use_amp = False
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
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
    policy.eval()
    return policy, preprocessor


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu explicitly.")

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
    cfg.dataset.root = str(dataset_root)
    cfg.pointseg_sample_cache_dir = str(cache_root) if cache_root.is_dir() else ""
    dataset = make_dataset(cfg)
    dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
    dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
    dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)

    requested_tasks = parse_tasks(args.tasks)
    selected = select_task_indices(dataset, requested_tasks, args.max_samples_per_task)
    policy, preprocessor = make_eval_pipeline(cfg, checkpoint, dataset, args.device)
    collate_fn = make_song_train_collate_fn(dataset)
    results = {}

    for requested_task in requested_tasks:
        subset = torch.utils.data.Subset(dataset, selected[requested_task])
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        losses = []
        for batch in loader:
            batch = preprocessor(batch)
            with torch.inference_mode():
                per_sample_loss, _ = policy.forward(batch, reduction="none")
            losses.extend(per_sample_loss.detach().float().cpu().tolist())
        values = np.asarray(losses, dtype=np.float64)
        results[requested_task] = {
            "samples": int(values.size),
            "loss_mean": float(values.mean()),
            "loss_std": float(values.std()),
            "loss_min": float(values.min()),
            "loss_max": float(values.max()),
        }
        print(f"{requested_task}: {json.dumps(results[requested_task], ensure_ascii=True)}", flush=True)

    payload = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root) if cache_root.is_dir() else None,
        "results": results,
    }
    if args.output is None:
        output = dataset_root.parent / (dataset_root.name + "_task_loss.json")
    else:
        output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
