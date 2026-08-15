#!/usr/bin/env python3
"""Verify every V52 cache shard against the exact online consensus sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla.song_pointseg import (
    compose_point_cloud_views,
    consensus_multiscale_novelty_union_sample_fused_point_cloud,
    open_episode_point_clouds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--coarse-novelty-scale", type=float, default=4.0)
    parser.add_argument("--expected-version", type=int, default=12)
    parser.add_argument("--expected-samples", type=int, default=137_590)
    parser.add_argument("--expected-shards", type=int, default=36)
    return parser.parse_args()


def voxel_rows(xyz: np.ndarray, size: float) -> set[tuple[int, int, int]]:
    return {tuple(row) for row in np.floor(xyz / size).astype(np.int64)}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    shards = manifest["shards"]
    assert int(manifest["version"]) == args.expected_version
    assert int(manifest["num_samples"]) == args.expected_samples
    assert len(shards) == args.expected_shards
    assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"]
    assert manifest["camera_view_fusion"] == "consensus_multiscale_novelty_union"
    assert float(manifest["camera_view_voxel_size"]) == args.voxel_size
    assert float(manifest["camera_view_coarse_novelty_scale"]) == args.coarse_novelty_scale
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
            fusion="consensus_multiscale_novelty_union",
        )
        assert union.shape == (19_500, 6)
        _sampled, _pad, expected_tensor = (
            consensus_multiscale_novelty_union_sample_fused_point_cloud(
                torch.from_numpy(union).unsqueeze(0).to(device=device),
                target_points=10_000,
                gripper_points=500,
                voxel_size=args.voxel_size,
                coarse_novelty_scale=args.coarse_novelty_scale,
            )
        )
        expected = expected_tensor[0].cpu().numpy().astype(np.int64, copy=False)
        exact = bool(np.array_equal(cached, expected))
        assert exact, (shard["path"], global_index, cached.shape, expected.shape)
        assert cached.shape == (10_000,)
        assert np.unique(cached).size == 10_000
        assert np.array_equal(cached[-500:], np.arange(19_000, 19_500, dtype=np.int64))

        scene_indices = cached[:9_500]
        selected_secondary = scene_indices[
            (scene_indices >= 9_500) & (scene_indices < 19_000)
        ]
        primary_fine = voxel_rows(primary[:9_500, :3], args.voxel_size)
        primary_coarse = voxel_rows(
            primary[:9_500, :3], args.voxel_size * args.coarse_novelty_scale
        )
        selected_fine = voxel_rows(union[scene_indices, :3], args.voxel_size)
        assert primary_fine.issubset(selected_fine)
        overlap_medoid_count = 0
        coarse_novel_count = 0
        for index in selected_secondary:
            xyz = union[int(index), :3]
            fine = tuple(np.floor(xyz / args.voxel_size).astype(np.int64))
            coarse = tuple(
                np.floor(xyz / (args.voxel_size * args.coarse_novelty_scale)).astype(np.int64)
            )
            if fine in primary_fine:
                overlap_medoid_count += 1
            else:
                assert coarse not in primary_coarse
                coarse_novel_count += 1

        rows.append(
            {
                "shard": shard["path"],
                "global_dataset_index": global_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "exact_online_cache_index_match": exact,
                "secondary_insertions": int(selected_secondary.size),
                "overlap_consensus_medoids": overlap_medoid_count,
                "coarse_novel_secondary_medoids": coarse_novel_count,
                "point_count": int(cached.size),
                "unique_index_count": int(np.unique(cached).size),
                "primary_fine_voxels_all_covered": True,
                "secondary_geometry_contract": True,
                "gripper_exact": True,
            }
        )

    payload = {
        "schema": "v52_consensus_multiscale_cache_online_exact_index_audit_v1",
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "manifest_num_samples": int(manifest["num_samples"]),
        "manifest_version": int(manifest["version"]),
        "fine_voxel_size_m": args.voxel_size,
        "coarse_novelty_scale": args.coarse_novelty_scale,
        "manifest_shard_count": len(shards),
        "audited_samples": len(rows),
        "selection": "one midpoint sample from every cache shard",
        "all_exact_online_cache_index_match": all(
            bool(row["exact_online_cache_index_match"]) for row in rows
        ),
        "all_exact_10000": all(int(row["point_count"]) == 10_000 for row in rows),
        "all_unique_indices": all(int(row["unique_index_count"]) == 10_000 for row in rows),
        "all_primary_fine_voxels_covered": all(
            bool(row["primary_fine_voxels_all_covered"]) for row in rows
        ),
        "all_secondary_geometry_contract": all(
            bool(row["secondary_geometry_contract"]) for row in rows
        ),
        "all_gripper_exact": all(bool(row["gripper_exact"]) for row in rows),
        "secondary_insertions": {
            "min": min(int(row["secondary_insertions"]) for row in rows),
            "max": max(int(row["secondary_insertions"]) for row in rows),
            "mean": float(np.mean([int(row["secondary_insertions"]) for row in rows])),
        },
        "overlap_consensus_medoids": {
            "min": min(int(row["overlap_consensus_medoids"]) for row in rows),
            "max": max(int(row["overlap_consensus_medoids"]) for row in rows),
            "mean": float(np.mean([int(row["overlap_consensus_medoids"]) for row in rows])),
        },
        "coarse_novel_secondary_medoids": {
            "min": min(int(row["coarse_novel_secondary_medoids"]) for row in rows),
            "max": max(int(row["coarse_novel_secondary_medoids"]) for row in rows),
            "mean": float(
                np.mean([int(row["coarse_novel_secondary_medoids"]) for row in rows])
            ),
        },
        "rows": rows,
    }
    required = [
        "all_exact_online_cache_index_match",
        "all_exact_10000",
        "all_unique_indices",
        "all_primary_fine_voxels_covered",
        "all_secondary_geometry_contract",
        "all_gripper_exact",
    ]
    assert all(payload[key] is True for key in required)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
