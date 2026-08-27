#!/usr/bin/env python

"""Create a deterministic global-frame manifest for ``eval_song`` anchor loss.

This command reads only LeRobot metadata.  It does not load images, point clouds,
sidecars, a policy, or a GPU.  Uniform sampling from all valid frames mirrors the
default offline DataLoader's frame distribution; the resulting order is retained
so every checkpoint receives the same batches as well as the same samples.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.scripts.eval_song import FIXED_ANCHOR_SCHEMA_VERSION


def build_manifest(
    dataset_repo_id: str,
    *,
    count: int,
    seed: int,
    pointseg_aux_loss_weight: float,
    drop_n_last_frames: int = 0,
    root: str | Path | None = None,
) -> dict:
    if count < 1:
        raise ValueError(f"count must be positive, got {count}.")
    if drop_n_last_frames < 0:
        raise ValueError(
            f"drop_n_last_frames must be non-negative, got {drop_n_last_frames}."
        )
    if not np.isfinite(pointseg_aux_loss_weight) or pointseg_aux_loss_weight < 0.0:
        raise ValueError(
            "pointseg_aux_loss_weight must be finite and non-negative, got "
            f"{pointseg_aux_loss_weight}."
        )
    metadata = LeRobotDatasetMetadata(
        dataset_repo_id,
        root=Path(root) if root is not None else None,
    )
    valid_ranges: list[np.ndarray] = []
    for episode in metadata.episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"]) - int(drop_n_last_frames)
        if stop > start:
            valid_ranges.append(np.arange(start, stop, dtype=np.int64))
    if not valid_ranges:
        raise ValueError("Dataset contains no valid frames after endpoint dropping.")
    valid_indices = np.concatenate(valid_ranges)
    if count > int(valid_indices.size):
        raise ValueError(
            f"Requested {count} unique anchors from only {valid_indices.size} valid frames."
        )
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(valid_indices, size=int(count), replace=False)
    return {
        "schema_version": FIXED_ANCHOR_SCHEMA_VERSION,
        "dataset_repo_id": str(dataset_repo_id),
        "dataset_length": int(metadata.total_frames),
        "indices": [int(index) for index in selected],
        "loss_contract": {
            "pointseg_aux_loss_weight": float(pointseg_aux_loss_weight),
            "worldflow_loss_weight": 1.0,
            "worldflow_geo_loss_weight": 0.0,
            "worldflow_bridge_loss_weight": 0.0,
            "worldflow_equiv_loss_weight": 0.0,
        },
        "selection": {
            "method": "uniform_without_replacement_over_valid_frames",
            "seed": int(seed),
            "count": int(count),
            "drop_n_last_frames": int(drop_n_last_frames),
            "valid_frame_count": int(valid_indices.size),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--drop-n-last-frames", type=int, default=0)
    parser.add_argument("--pointseg-aux-loss-weight", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    manifest = build_manifest(
        args.dataset_repo_id,
        count=args.count,
        seed=args.seed,
        pointseg_aux_loss_weight=args.pointseg_aux_loss_weight,
        drop_n_last_frames=args.drop_n_last_frames,
        root=args.root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(manifest['indices'])} fixed anchors to {output}")


if __name__ == "__main__":
    main()
