#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")
os.environ.setdefault("SONG_POINTSEG_REQUIRE_POINTOPS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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

if __package__:
    from ._paths import REAL_DATA_ROOT
else:
    from _paths import REAL_DATA_ROOT

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "SONG_POINTSEG_DATASET",
        str(REAL_DATA_ROOT / "lerobot_dataset"),
    )
)
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SONG_POINTSEG_SAMPLE_CACHE",
        str(REAL_DATA_ROOT / "pointseg_cache"),
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
    parser.add_argument("--rank-wait-timeout-sec", type=int, default=0, help="Timeout while rank0 waits for rank done marker files. 0 means wait forever.")
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


def _episode_preview_targets(
    dataset: SongTemporalPointCloudDataset,
    total_samples: int,
) -> dict[int, list[tuple[int, str, int]]]:
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError("Episode metadata is required to save first/middle/last pseudo-label previews.")

    targets: dict[int, list[tuple[int, str, int]]] = {}
    for episode_position in range(len(episodes)):
        episode = episodes[episode_position]
        episode_index = int(episode.get("episode_index", episode_position))
        start_index = int(episode["dataset_from_index"])
        end_index = int(episode["dataset_to_index"])
        episode_length = end_index - start_index
        if episode_length <= 0:
            continue

        frame_targets = (
            ("first", 0),
            ("middle", (episode_length - 1) // 2),
            ("last", episode_length - 1),
        )
        for position, frame_index in frame_targets:
            dataset_index = start_index + frame_index
            if 0 <= dataset_index < total_samples:
                targets.setdefault(dataset_index, []).append((episode_index, position, frame_index))
    return targets


def _save_episode_preview(
    output_dir: Path,
    episode_index: int,
    position: str,
    frame_index: int,
    current_pc: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    write_role_ply(
        output_dir
        / "visualizations"
        / f"episode_{episode_index:06d}"
        / f"{position}_frame_{frame_index:06d}_pseudo.ply",
        current_pc.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
    )



def _is_torchrun_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _init_multiprocess(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    """Read torchrun rank env vars and choose one GPU per rank.

    This cache job does not need gradient collectives, so it intentionally avoids
    torch.distributed/NCCL. Synchronization is done with marker files to prevent
    NCCL watchdog timeouts when some ranks finish much earlier than others.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    wants_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if wants_cuda:
        visible_count = torch.cuda.device_count()
        if visible_count <= 0:
            raise RuntimeError("args.device requests CUDA but torch.cuda.device_count() is 0")
        device_index = local_rank % visible_count
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device(args.device)

    return rank, local_rank, world_size, device


def _sync_dir(output_dir: Path) -> Path:
    return output_dir / "_dist_sync"


def _write_marker(path: Path, text: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _wait_for_marker(path: Path, *, timeout_sec: int = 0, poll_sec: float = 2.0) -> None:
    start = time.time()
    while not path.exists():
        if timeout_sec > 0 and (time.time() - start) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for marker: {path}")
        time.sleep(poll_sec)


def _wait_for_all_rank_done(output_dir: Path, world_size: int, *, timeout_sec: int = 0) -> None:
    sync = _sync_dir(output_dir)
    start = time.time()
    missing_report_t = 0.0
    while True:
        missing = [r for r in range(world_size) if not (sync / f"rank_{r:03d}.done").exists()]
        failed = sorted(sync.glob("rank_*.failed"))
        if failed:
            details = []
            for path in failed:
                try:
                    details.append(f"{path.name}: {path.read_text()[:2000]}")
                except Exception:
                    details.append(str(path))
            raise RuntimeError("One or more ranks failed:\n" + "\n".join(details))
        if not missing:
            return
        now = time.time()
        if now - missing_report_t > 60:
            print(f"[rank 0] waiting for done markers from ranks: {missing}", flush=True)
            missing_report_t = now
        if timeout_sec > 0 and (now - start) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for ranks {missing} to finish")
        time.sleep(2.0)


def _rank_bounds(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous split, preserving global dataset/cache order."""
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    length = base + (1 if rank < rem else 0)
    return start, start + length


def _make_rank_shards(total_samples: int, shard_size: int, world_size: int, rank: int) -> list[dict[str, Any]]:
    start_index, end_index = _rank_bounds(total_samples, world_size, rank)
    local_samples = end_index - start_index
    local_shards = _make_shard_manifest(local_samples, shard_size)
    for shard in local_shards:
        shard["path"] = f"rank_{rank:03d}/{shard['path']}"
        shard["start"] = start_index + int(shard["start"])
    return local_shards


def _build_all_shards_from_disk(output_dir: Path, total_samples: int, shard_size: int, world_size: int) -> list[dict[str, Any]]:
    """Rank0 rebuilds manifest shard list after all ranks have written shard arrays."""
    all_shards: list[dict[str, Any]] = []
    for rank in range(world_size):
        for shard in _make_rank_shards(total_samples, shard_size, world_size, rank):
            offsets_path = output_dir / shard["path"] / "sample_offsets.npy"
            if not offsets_path.exists():
                raise FileNotFoundError(f"Missing shard offsets written by rank {rank}: {offsets_path}")
            offsets = np.load(offsets_path, mmap_mode="r")
            shard["num_points"] = int(offsets[-1])
            all_shards.append(shard)
    return all_shards


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def cache_samples(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.current_points = min(args.current_points, 256)
        args.future_points = min(args.future_points, 512)
        args.batch_size = min(args.batch_size, 2)
        args.shard_size = min(args.shard_size, 4)
        args.max_samples = 4 if args.max_samples is None else min(args.max_samples, 4)
        args.vis_count = min(args.vis_count, 2)

    rank, local_rank, world_size, device = _init_multiprocess(args)
    is_main = rank == 0

    # Make each rank deterministic but distinct for DataLoader workers.
    set_seed(args.seed + rank)
    storage_dtype = np.dtype(args.storage_dtype)

    sync = _sync_dir(args.output_dir)
    if is_main:
        _prepare_output_dir(args.output_dir, args.overwrite)
        sync.mkdir(parents=True, exist_ok=True)
        for marker in sync.glob("rank_*.done"):
            marker.unlink(missing_ok=True)
        for marker in sync.glob("rank_*.failed"):
            marker.unlink(missing_ok=True)
        _write_marker(sync / "ready", f"ready pid={os.getpid()} time={time.time()}\n")
    else:
        _wait_for_marker(sync / "ready", timeout_sec=args.rank_wait_timeout_sec)

    pseudo_cfg = replace(PseudoLabelConfig(), nn_chunk_size=args.nn_chunk_size)
    full_dataset = make_dataset(args)
    total_samples = len(full_dataset) if args.max_samples is None else min(len(full_dataset), args.max_samples)
    if total_samples <= 0:
        raise ValueError("Song pointseg cache needs at least one sample.")
    preview_targets = _episode_preview_targets(full_dataset, total_samples)

    start_index, end_index = _rank_bounds(total_samples, world_size, rank)
    local_samples = end_index - start_index
    local_indices = range(start_index, end_index)
    dataset = Subset(full_dataset, local_indices)
    shards = _make_rank_shards(total_samples, args.shard_size, world_size, rank)

    if is_main:
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
            "distributed": {
                "world_size": world_size,
                "launcher": "torchrun",
                "split": "contiguous_by_dataset_index",
            },
            "shards": [],
        }
    else:
        manifest = None

    if is_main:
        print(
            f"[cache] total_samples={total_samples} world_size={world_size} "
            f"batch_size_per_gpu={args.batch_size} device={device}",
            flush=True,
        )
    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} device={device} "
        f"range=[{start_index}, {end_index}) local_samples={local_samples} shards={len(shards)}",
        flush=True,
    )

    if local_samples == 0:
        _write_marker(sync / f"rank_{rank:03d}.done", "empty\n")
        if is_main and manifest is not None:
            _wait_for_all_rank_done(args.output_dir, world_size, timeout_sec=args.rank_wait_timeout_sec)
            manifest["shards"] = _build_all_shards_from_disk(
                args.output_dir, total_samples, args.shard_size, world_size
            )
            _atomic_write_json(args.output_dir / "manifest.json", manifest)
        return

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
    previews_saved = 0
    t0 = time.time()
    progress = tqdm(
        total=local_samples,
        desc=f"Rank {rank} cache",
        unit="sample",
        position=rank,
        disable=not is_main,
    )

    try:
        with torch.inference_mode():
            for batch in dataloader:
                if written >= local_samples:
                    break

                batch_size = min(int(batch["observation.point_cloud"].shape[0]), local_samples - written)
                batch = _slice_batch_to_size(batch, batch_size)
                batch = move_batch_to_device(batch, device)
                current_pc = batch["observation.point_cloud"]

                geometric_pseudo = generate_pseudo_labels(
                    current_pc,
                    batch["observation.point_cloud_future"],
                    batch["future_ee_poses"],
                    batch["future_is_pad"],
                    current_is_pad=batch.get("observation.point_cloud_is_pad"),
                    future_point_is_pad=batch.get("observation.point_cloud_future_is_pad"),
                    config=pseudo_cfg,
                )
                pseudo = force_small_current_clouds_foreground(
                    geometric_pseudo,
                    current_pc,
                    args.current_points,
                    batch.get("observation.point_cloud_is_pad"),
                )

                current_is_pad = batch.get("observation.point_cloud_is_pad")
                for batch_index in range(batch_size):
                    dataset_index = start_index + written + batch_index
                    targets = preview_targets.get(dataset_index, ())
                    if not targets:
                        continue
                    valid = (
                        ~current_is_pad[batch_index].bool()
                        if current_is_pad is not None
                        else torch.ones(current_pc.shape[1], dtype=torch.bool, device=current_pc.device)
                    )
                    preview_pc = current_pc[batch_index][valid]
                    preview_labels = pseudo["labels"][batch_index][valid]
                    for episode_index, position, frame_index in targets:
                        _save_episode_preview(
                            args.output_dir,
                            episode_index,
                            position,
                            frame_index,
                            preview_pc,
                            preview_labels,
                        )
                        previews_saved += 1

                for batch_index in range(batch_size):
                    global_dataset_index = start_index + written
                    shard_samples.append(
                        _sample_from_batch(current_pc, pseudo, batch, batch_index, global_dataset_index)
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
    finally:
        progress.close()

    if shard_samples:
        _save_variable_shard(
            args.output_dir,
            shards[current_shard_index],
            shard_samples,
            storage_dtype=storage_dtype,
        )

    elapsed = time.time() - t0
    speed = written / max(elapsed, 1e-6)
    print(
        f"[rank {rank}/{world_size}] wrote {written} samples in {elapsed:.1f}s "
        f"({speed:.2f} samples/s); episode previews saved={previews_saved}",
        flush=True,
    )

    _write_marker(
        sync / f"rank_{rank:03d}.done",
        f"written={written} elapsed={elapsed:.3f} speed={speed:.3f} pid={os.getpid()}\n",
    )

    if is_main:
        assert manifest is not None
        _wait_for_all_rank_done(args.output_dir, world_size, timeout_sec=args.rank_wait_timeout_sec)
        manifest["shards"] = _build_all_shards_from_disk(
            args.output_dir, total_samples, args.shard_size, world_size
        )
        _atomic_write_json(args.output_dir / "manifest.json", manifest)
        print(f"Cached {total_samples} Song pointseg samples to {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    try:
        cache_samples(args)
    except Exception as exc:
        try:
            _write_marker(_sync_dir(args.output_dir) / f"rank_{rank:03d}.failed", repr(exc))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
