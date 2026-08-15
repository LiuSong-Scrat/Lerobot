#!/usr/bin/env python3
"""Verify that every sampled cache shard uses the exact online V46 point indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla.song_pointseg import (
    compose_point_cloud_views,
    multiscale_novelty_union_sample_fused_point_cloud,
    open_episode_point_clouds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--coarse-novelty-scale", type=float, default=3.0)
    parser.add_argument("--expected-version", type=int)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-shards", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.expected_version is not None:
        assert int(manifest["version"]) == int(args.expected_version)
    shards = manifest["shards"]
    if args.expected_samples is not None:
        assert int(manifest["num_samples"]) == int(args.expected_samples)
    if args.expected_shards is not None:
        assert len(shards) == int(args.expected_shards)
    assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"]
    assert manifest["camera_view_fusion"] == "multiscale_novelty_union"
    assert float(manifest["camera_view_voxel_size"]) == float(args.voxel_size)
    assert float(manifest.get("camera_view_coarse_novelty_scale", 3.0)) == float(
        args.coarse_novelty_scale
    )
    assert int(manifest["current_points"]) == 10_000
    assert int(manifest["future_points"]) == 10_000
    assert int(manifest["gripper_points"]) == 500

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for shard in shards:
        shard_dir = cache_dir / shard["path"]
        length = int(shard["length"])
        assert length > 0
        local_index = length // 2
        global_index = int(shard["start"]) + local_index
        offsets = np.load(shard_dir / "sample_offsets.npy", mmap_mode="r")
        cached_all = np.load(shard_dir / "point_indices.npy", mmap_mode="r")
        dataset_indices = np.load(shard_dir / "dataset_index.npy", mmap_mode="r")
        episode_indices = np.load(shard_dir / "episode_index.npy", mmap_mode="r")
        frame_indices = np.load(shard_dir / "frame_index.npy", mmap_mode="r")
        assert offsets.shape == (length + 1,)
        assert dataset_indices.shape == episode_indices.shape == frame_indices.shape == (length,)
        assert int(dataset_indices[local_index]) == global_index
        start, stop = int(offsets[local_index]), int(offsets[local_index + 1])
        cached = np.asarray(cached_all[start:stop], dtype=np.int64)
        episode_index = int(episode_indices[local_index])
        frame_index = int(frame_indices[local_index])

        primary = np.asarray(
            open_episode_point_clouds(dataset_root / "point_clouds", episode_index)[frame_index],
            dtype=np.float32,
        )
        secondary = np.asarray(
            open_episode_point_clouds(
                dataset_root / "point_clouds_robot0_eye_in_hand", episode_index
            )[frame_index],
            dtype=np.float32,
        )
        union = compose_point_cloud_views(
            [primary, secondary],
            gripper_points=500,
            fusion="multiscale_novelty_union",
        )
        assert union.shape == (19_500, 6)
        _sampled, _pad, expected_tensor = multiscale_novelty_union_sample_fused_point_cloud(
            torch.from_numpy(union).unsqueeze(0).to(device=device),
            target_points=10_000,
            gripper_points=500,
            voxel_size=float(args.voxel_size),
            coarse_novelty_scale=float(args.coarse_novelty_scale),
        )
        expected = expected_tensor[0].cpu().numpy().astype(np.int64, copy=False)
        exact = bool(np.array_equal(cached, expected))
        assert exact, (shard["path"], global_index, cached.shape, expected.shape)
        assert cached.shape == (10_000,)
        assert np.unique(cached).size == 10_000
        assert np.array_equal(cached[-500:], np.arange(19_000, 19_500, dtype=np.int64))
        rows.append(
            {
                "shard": shard["path"],
                "global_dataset_index": global_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "exact_online_cache_index_match": exact,
                "secondary_insertions": int(np.count_nonzero(cached[:9_500] >= 9_500)),
                "point_count": int(cached.size),
                "unique_index_count": int(np.unique(cached).size),
                "gripper_exact": True,
            }
        )

    payload = {
        "schema": "v46_multiscale_novelty_cache_online_exact_index_audit_v1",
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "manifest_num_samples": int(manifest["num_samples"]),
        "manifest_version": int(manifest["version"]),
        "coarse_novelty_scale": float(args.coarse_novelty_scale),
        "manifest_shard_count": len(shards),
        "audited_samples": len(rows),
        "selection": "one midpoint sample from every cache shard",
        "all_exact_online_cache_index_match": all(
            bool(row["exact_online_cache_index_match"]) for row in rows
        ),
        "all_exact_10000": all(int(row["point_count"]) == 10_000 for row in rows),
        "all_unique_indices": all(int(row["unique_index_count"]) == 10_000 for row in rows),
        "all_gripper_exact": all(bool(row["gripper_exact"]) for row in rows),
        "secondary_insertions": {
            "min": min(int(row["secondary_insertions"]) for row in rows),
            "max": max(int(row["secondary_insertions"]) for row in rows),
            "mean": float(np.mean([int(row["secondary_insertions"]) for row in rows])),
        },
        "rows": rows,
    }
    assert payload["all_exact_online_cache_index_match"]
    assert payload["all_exact_10000"]
    assert payload["all_unique_indices"]
    assert payload["all_gripper_exact"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
