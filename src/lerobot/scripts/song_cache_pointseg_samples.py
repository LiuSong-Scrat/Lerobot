#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.song_pointseg import (
    DEFAULT_FUTURE_OFFSETS,
    POINTSEG_CACHE_LABEL_FIELDS,
    POINTSEG_CACHE_VERSION,
    ROLE_NAMES,
    PseudoLabelConfig,
    SongTemporalPointCloudDataset,
    force_small_current_clouds_foreground,
    generate_pseudo_labels,
    move_batch_to_device,
    parse_future_offsets,
    song_pointseg_collate,
    write_role_ply,
)
from lerobot.utils.random_utils import set_seed

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "SONG_POINTSEG_DATASET",
        "/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/temp",
    )
)
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SONG_POINTSEG_SAMPLE_CACHE",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/song_pointseg_sample_cache",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Song pointseg sampled points and pseudo labels offline.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset.root", dest="dataset_root", type=str, default=None)
    parser.add_argument("--point-cloud-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--future-offsets", type=parse_future_offsets, default=DEFAULT_FUTURE_OFFSETS)
    parser.add_argument("--current-points", type=int, default=10000)
    parser.add_argument("--future-points", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--nn-chunk-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--vis-count", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _make_lerobot_dataset(args: argparse.Namespace) -> LeRobotDataset:
    repo_id = args.dataset_repo_id
    root = Path(args.dataset_root) if args.dataset_root else None
    max_offset = max(args.future_offsets)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = int(metadata.fps)
    return LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            "action": [i / fps for i in range(max_offset + 1)],
            "observation.state": [0.0],
        },
    )


def make_dataset(args: argparse.Namespace) -> SongTemporalPointCloudDataset:
    dataset = _make_lerobot_dataset(args)
    dataset_root = Path(getattr(dataset, "root", args.dataset_repo_id))
    point_cloud_dir = args.point_cloud_dir or dataset_root / "point_clouds"
    return SongTemporalPointCloudDataset(
        dataset,
        point_cloud_dir=point_cloud_dir,
        future_offsets=args.future_offsets,
        current_points=args.current_points,
        future_points=args.future_points,
        seed=args.seed,
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Cache output dir is not empty: {output_dir}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _make_shard_manifest(total_samples: int, shard_size: int) -> list[dict[str, Any]]:
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    shards = []
    for start in range(0, total_samples, shard_size):
        length = min(shard_size, total_samples - start)
        shard_index = len(shards)
        shards.append(
            {
                "path": f"shard_{shard_index:06d}",
                "start": start,
                "length": length,
            }
        )
    return shards


def _slice_batch_to_size(batch: dict[str, Any], batch_size: int) -> dict[str, Any]:
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= batch_size:
            sliced[key] = value[:batch_size]
        else:
            sliced[key] = value
    return sliced


def _save_variable_shard(
    output_dir: Path,
    shard: dict[str, Any],
    samples: list[dict[str, np.ndarray | int]],
    *,
    storage_dtype: np.dtype,
) -> dict[str, Any]:
    shard_dir = output_dir / shard["path"]
    shard_dir.mkdir(parents=True, exist_ok=True)
    lengths = [int(sample["point_indices"].shape[0]) for sample in samples]
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    np.save(shard_dir / "sample_offsets.npy", offsets)
    np.save(shard_dir / "point_indices.npy", np.concatenate([sample["point_indices"] for sample in samples], axis=0).astype(np.int64, copy=False))
    np.save(shard_dir / "labels.npy", np.concatenate([sample["labels"] for sample in samples], axis=0).astype(np.int16, copy=False))
    np.save(shard_dir / "weights.npy", np.concatenate([sample["weights"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "class_scores.npy", np.concatenate([sample["class_scores"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "foreground_score.npy", np.concatenate([sample["foreground_score"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "episode_index.npy", np.asarray([sample["episode_index"] for sample in samples], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.asarray([sample["frame_index"] for sample in samples], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.asarray([sample["dataset_index"] for sample in samples], dtype=np.int64))
    shard["num_points"] = int(offsets[-1])
    return shard


def _sample_from_batch(
    current_pc: torch.Tensor,
    pseudo: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    batch_index: int,
    dataset_index: int,
) -> dict[str, np.ndarray | int]:
    is_pad = batch.get("observation.point_cloud_is_pad")
    valid = ~is_pad[batch_index].bool().detach().cpu() if is_pad is not None else torch.ones(
        current_pc.shape[1], dtype=torch.bool
    )
    point_indices = batch.get("observation.point_cloud_indices")
    if point_indices is None:
        point_indices_np = torch.arange(current_pc.shape[1], dtype=torch.long)[valid].numpy()
    else:
        point_indices_np = point_indices[batch_index].detach().cpu()[valid].to(dtype=torch.long).numpy()
    return {
        "point_indices": point_indices_np,
        "labels": pseudo["labels"][batch_index].detach().cpu()[valid].numpy(),
        "weights": pseudo["weights"][batch_index].detach().cpu()[valid].numpy(),
        "class_scores": pseudo["class_scores"][batch_index].detach().cpu()[valid].numpy(),
        "foreground_score": pseudo["foreground_score"][batch_index].detach().cpu()[valid].numpy(),
        "episode_index": int(batch["episode_index"][batch_index].detach().cpu().reshape(-1)[0].item()),
        "frame_index": int(batch["frame_index"][batch_index].detach().cpu().reshape(-1)[0].item()),
        "dataset_index": int(dataset_index),
    }


def _save_preview(
    output_dir: Path,
    sample_index: int,
    current_pc: torch.Tensor,
    pseudo: dict[str, torch.Tensor],
) -> None:
    write_role_ply(
        output_dir / "visualizations" / f"sample_{sample_index:06d}_pseudo.ply",
        current_pc.detach().cpu().numpy(),
        pseudo["labels"].detach().cpu().numpy(),
    )


def cache_samples(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.current_points = min(args.current_points, 256)
        args.future_points = min(args.future_points, 512)
        args.batch_size = min(args.batch_size, 2)
        args.shard_size = min(args.shard_size, 4)
        args.max_samples = 4 if args.max_samples is None else min(args.max_samples, 4)
        args.vis_count = min(args.vis_count, 2)

    set_seed(args.seed)
    device = torch.device(args.device)
    storage_dtype = np.dtype(args.storage_dtype)
    _prepare_output_dir(args.output_dir, args.overwrite)

    pseudo_cfg = replace(PseudoLabelConfig(), nn_chunk_size=args.nn_chunk_size)
    dataset = make_dataset(args)
    total_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    if total_samples <= 0:
        raise ValueError("Song pointseg cache needs at least one sample.")
    shards = _make_shard_manifest(total_samples, args.shard_size)
    manifest = {
        "version": POINTSEG_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role_names": list(ROLE_NAMES),
        "fields": list(POINTSEG_CACHE_LABEL_FIELDS),
        "cache_mode": "indices",
        "num_samples": total_samples,
        "future_offsets": list(args.future_offsets),
        "current_points": args.current_points,
        "future_points": args.future_points,
        "variable_num_points": True,
        "point_count_policy": "cap_without_repeat",
        "small_cloud_label_policy": "all_valid_current_points_are_foreground_when_count_lt_current_points",
        "storage_dtype": args.storage_dtype,
        "pseudo_label_config": asdict(pseudo_cfg),
        "args": _jsonable(vars(args)),
        "shards": shards,
    }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=song_pointseg_collate,
    )

    current_shard_index = 0
    shard_samples: list[dict[str, np.ndarray | int]] = []
    written = 0
    previews = 0
    progress = tqdm(total=total_samples, desc="Cache Song pointseg", unit="sample")

    with torch.inference_mode():
        for batch in dataloader:
            if written >= total_samples:
                break
            batch_size = min(int(batch["observation.point_cloud"].shape[0]), total_samples - written)
            batch = _slice_batch_to_size(batch, batch_size)
            batch = move_batch_to_device(batch, device)
            current_pc = batch["observation.point_cloud"]
            pseudo = generate_pseudo_labels(
                current_pc,
                batch["observation.point_cloud_future"],
                batch["future_ee_poses"],
                batch["future_is_pad"],
                current_is_pad=batch.get("observation.point_cloud_is_pad"),
                future_point_is_pad=batch.get("observation.point_cloud_future_is_pad"),
                config=pseudo_cfg,
            )
            pseudo = force_small_current_clouds_foreground(
                pseudo,
                current_pc,
                args.current_points,
                batch.get("observation.point_cloud_is_pad"),
            )

            if previews < args.vis_count:
                preview_count = min(args.vis_count - previews, batch_size)
                for batch_index in range(preview_count):
                    current_is_pad = batch.get("observation.point_cloud_is_pad")
                    valid = ~current_is_pad[batch_index].bool() if current_is_pad is not None else torch.ones(
                        current_pc.shape[1], dtype=torch.bool, device=current_pc.device
                    )
                    _save_preview(
                        args.output_dir,
                        written + batch_index,
                        current_pc[batch_index][valid],
                        {key: value[batch_index][valid] for key, value in pseudo.items() if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == current_pc.shape[:2]},
                    )
                previews += preview_count

            for batch_index in range(batch_size):
                shard_samples.append(
                    _sample_from_batch(current_pc, pseudo, batch, batch_index, written)
                )
                written += 1
                progress.update(1)
                if len(shard_samples) == int(shards[current_shard_index]["length"]):
                    _save_variable_shard(
                        args.output_dir,
                        shards[current_shard_index],
                        shard_samples,
                        storage_dtype=storage_dtype,
                    )
                    shard_samples = []
                    current_shard_index += 1

    progress.close()
    if shard_samples:
        _save_variable_shard(
            args.output_dir,
            shards[current_shard_index],
            shard_samples,
            storage_dtype=storage_dtype,
        )
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Cached {written} Song pointseg samples to {args.output_dir}")


def main() -> None:
    cache_samples(parse_args())


if __name__ == "__main__":
    main()
