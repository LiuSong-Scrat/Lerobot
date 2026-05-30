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
    MOTION_PRIOR_DIM,
    POINTSEG_CACHE_FIELDS,
    POINTSEG_CACHE_VERSION,
    ROLE_NAMES,
    PseudoLabelConfig,
    SongTemporalPointCloudDataset,
    generate_pseudo_labels,
    move_batch_to_device,
    parse_future_offsets,
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
    parser.add_argument("--current-points", type=int, default=8192)
    parser.add_argument("--future-points", type=int, default=16384)
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


def _open_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _open_shard_arrays(
    output_dir: Path,
    shard: dict[str, Any],
    *,
    current_points: int,
    storage_dtype: np.dtype,
) -> dict[str, np.memmap]:
    shard_dir = output_dir / shard["path"]
    length = int(shard["length"])
    return {
        "point_cloud": _open_memmap(
            shard_dir / "point_cloud.npy", (length, current_points, 6), storage_dtype
        ),
        "priors": _open_memmap(
            shard_dir / "priors.npy", (length, current_points, MOTION_PRIOR_DIM), storage_dtype
        ),
        "labels": _open_memmap(shard_dir / "labels.npy", (length, current_points), np.int16),
        "weights": _open_memmap(shard_dir / "weights.npy", (length, current_points), storage_dtype),
        "class_scores": _open_memmap(
            shard_dir / "class_scores.npy", (length, current_points, len(ROLE_NAMES)), storage_dtype
        ),
        "role_scores": _open_memmap(shard_dir / "role_scores.npy", (length, current_points, 3), storage_dtype),
        "foreground_score": _open_memmap(
            shard_dir / "foreground_score.npy", (length, current_points), storage_dtype
        ),
        "episode_index": _open_memmap(shard_dir / "episode_index.npy", (length,), np.int64),
        "frame_index": _open_memmap(shard_dir / "frame_index.npy", (length,), np.int64),
        "dataset_index": _open_memmap(shard_dir / "dataset_index.npy", (length,), np.int64),
    }


def _flush_arrays(arrays: dict[str, np.memmap] | None) -> None:
    if arrays is None:
        return
    for array in arrays.values():
        array.flush()


def _to_numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


def _slice_batch_to_size(batch: dict[str, Any], batch_size: int) -> dict[str, Any]:
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= batch_size:
            sliced[key] = value[:batch_size]
        else:
            sliced[key] = value
    return sliced


def _write_batch_slice(
    arrays: dict[str, np.memmap],
    dst: slice,
    src: slice,
    *,
    current_pc: torch.Tensor,
    pseudo: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    dataset_indices: np.ndarray,
    storage_dtype: np.dtype,
) -> None:
    arrays["point_cloud"][dst] = _to_numpy(current_pc[src], storage_dtype)
    arrays["priors"][dst] = _to_numpy(pseudo["priors"][src], storage_dtype)
    arrays["labels"][dst] = _to_numpy(pseudo["labels"][src], np.int16)
    arrays["weights"][dst] = _to_numpy(pseudo["weights"][src], storage_dtype)
    arrays["class_scores"][dst] = _to_numpy(pseudo["class_scores"][src], storage_dtype)
    arrays["role_scores"][dst] = _to_numpy(pseudo["role_scores"][src], storage_dtype)
    arrays["foreground_score"][dst] = _to_numpy(pseudo["foreground_score"][src], storage_dtype)
    arrays["episode_index"][dst] = _to_numpy(batch["episode_index"][src], np.int64)
    arrays["frame_index"][dst] = _to_numpy(batch["frame_index"][src], np.int64)
    arrays["dataset_index"][dst] = dataset_indices


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
        "fields": list(POINTSEG_CACHE_FIELDS),
        "num_samples": total_samples,
        "future_offsets": list(args.future_offsets),
        "current_points": args.current_points,
        "future_points": args.future_points,
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
    )

    current_shard_index = 0
    shard_cursor = 0
    arrays: dict[str, np.memmap] | None = _open_shard_arrays(
        args.output_dir,
        shards[current_shard_index],
        current_points=args.current_points,
        storage_dtype=storage_dtype,
    )
    written = 0
    previews = 0
    progress = tqdm(total=total_samples, desc="Cache Song pointseg", unit="sample")

    with torch.no_grad():
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
                config=pseudo_cfg,
            )

            if previews < args.vis_count:
                preview_count = min(args.vis_count - previews, batch_size)
                for batch_index in range(preview_count):
                    _save_preview(
                        args.output_dir,
                        written + batch_index,
                        current_pc[batch_index],
                        {key: value[batch_index] for key, value in pseudo.items() if torch.is_tensor(value)},
                    )
                previews += preview_count

            src_start = 0
            while src_start < batch_size:
                if arrays is None:
                    arrays = _open_shard_arrays(
                        args.output_dir,
                        shards[current_shard_index],
                        current_points=args.current_points,
                        storage_dtype=storage_dtype,
                    )
                shard_length = int(shards[current_shard_index]["length"])
                take = min(batch_size - src_start, shard_length - shard_cursor)
                dst = slice(shard_cursor, shard_cursor + take)
                src = slice(src_start, src_start + take)
                dataset_indices = np.arange(written, written + take, dtype=np.int64)
                _write_batch_slice(
                    arrays,
                    dst,
                    src,
                    current_pc=current_pc,
                    pseudo=pseudo,
                    batch=batch,
                    dataset_indices=dataset_indices,
                    storage_dtype=storage_dtype,
                )
                src_start += take
                shard_cursor += take
                written += take
                progress.update(take)

                if shard_cursor == shard_length:
                    _flush_arrays(arrays)
                    arrays = None
                    shard_cursor = 0
                    current_shard_index += 1

    progress.close()
    _flush_arrays(arrays)
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Cached {written} Song pointseg samples to {args.output_dir}")


def main() -> None:
    cache_samples(parse_args())


if __name__ == "__main__":
    main()
