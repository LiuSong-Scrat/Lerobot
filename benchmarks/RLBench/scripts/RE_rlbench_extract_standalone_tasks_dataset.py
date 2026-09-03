#!/usr/bin/env python3
"""Build a standalone, reindexed RLBench LeRobot task subset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/rlbench_subset_hf_cache")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import write_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--expected-episodes-per-task", type=int, default=None)
    return parser.parse_args()


def _compact_data_files(root: Path, data_metadata: dict[int, dict]) -> None:
    old_keys = sorted(
        {
            (int(meta["data/chunk_index"]), int(meta["data/file_index"]))
            for meta in data_metadata.values()
        }
    )
    temporary_paths = []
    for compact_index, (chunk_index, file_index) in enumerate(old_keys):
        old_path = root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        temporary_path = old_path.parent / f".compact-{compact_index:03d}.parquet"
        old_path.rename(temporary_path)
        temporary_paths.append(temporary_path)
    for compact_index, temporary_path in enumerate(temporary_paths):
        final_path = root / "data" / "chunk-000" / f"file-{compact_index:03d}.parquet"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.rename(final_path)

    compact_mapping = {old_key: index for index, old_key in enumerate(old_keys)}
    for meta in data_metadata.values():
        old_key = (int(meta["data/chunk_index"]), int(meta["data/file_index"]))
        meta["data/chunk_index"] = 0
        meta["data/file_index"] = compact_mapping[old_key]


def _set_scalar_list(columns: dict[str, list], key: str, row: int, value: float | int) -> None:
    current = columns[key][row]
    if not isinstance(current, list) or len(current) != 1:
        raise RuntimeError(f"Expected one-element list for {key}, got {current!r}")
    if isinstance(current[0], int):
        columns[key][row] = [int(value)]
    else:
        columns[key][row] = [float(value)]


def _patch_reindexed_episode_stats(
    root: Path,
    episode_mapping: dict[int, int],
    source_episodes,
    new_task_by_episode: dict[int, int],
    feature_names: set[str],
) -> None:
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    columns = table.to_pydict()
    row_by_episode = {
        int(episode_index): row for row, episode_index in enumerate(columns["episode_index"])
    }
    reverse_mapping = {new: old for old, new in episode_mapping.items()}

    constant_stats = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
    shifted_index_stats = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
    for new_episode, old_episode in sorted(reverse_mapping.items()):
        row = row_by_episode[new_episode]
        old_from = int(source_episodes[old_episode]["dataset_from_index"])
        new_from = int(columns["dataset_from_index"][row])
        index_shift = new_from - old_from
        for stat in constant_stats:
            _set_scalar_list(columns, f"stats/episode_index/{stat}", row, new_episode)
            _set_scalar_list(
                columns,
                f"stats/task_index/{stat}",
                row,
                new_task_by_episode[new_episode],
            )
        for stat in shifted_index_stats:
            key = f"stats/index/{stat}"
            _set_scalar_list(columns, key, row, columns[key][row][0] + index_shift)
        _set_scalar_list(columns, "stats/episode_index/std", row, 0.0)
        _set_scalar_list(columns, "stats/task_index/std", row, 0.0)

    patched = pa.Table.from_pydict(columns, schema=table.schema)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(patched, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)

    all_episode_stats = []
    for episode_row in patched.to_pylist():
        episode_stats: dict[str, dict[str, np.ndarray]] = {}
        for key, value in episode_row.items():
            if not key.startswith("stats/"):
                continue
            feature_name, stat_name = key[len("stats/") :].rsplit("/", 1)
            episode_stats.setdefault(feature_name, {})[stat_name] = np.asarray(value)
        all_episode_stats.append(episode_stats)
    aggregated = aggregate_stats(all_episode_stats)
    write_stats(
        {key: value for key, value in aggregated.items() if key in feature_names}, root
    )


def _hardlink_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, copy_function=os.link)


def _hardlink_episode_sidecars(
    source_root: Path,
    build_root: Path,
    episode_mapping: dict[int, int],
) -> None:
    sidecars = {
        "world_ee_poses": ".npy",
        "raw_expert_actions": ".npy",
        "raw_expert_actions_full": ".npy",
        "initial_task_states": ".npz",
        "initial_object_states": ".npz",
    }
    for directory, suffix in sidecars.items():
        source_dir = source_root / directory
        target_dir = build_root / directory
        target_dir.mkdir(parents=True)
        os.link(source_dir / "meta.json", target_dir / "meta.json")
        for old_episode, new_episode in sorted(episode_mapping.items(), key=lambda item: item[1]):
            os.link(
                source_dir / f"episode_{old_episode:06d}{suffix}",
                target_dir / f"episode_{new_episode:06d}{suffix}",
            )

    point_cloud_dir = build_root / "point_clouds"
    point_cloud_dir.mkdir(parents=True)
    for old_episode, new_episode in sorted(episode_mapping.items(), key=lambda item: item[1]):
        _hardlink_tree(
            source_root / "point_clouds" / f"episode_{old_episode:06d}.zarr",
            point_cloud_dir / f"episode_{new_episode:06d}.zarr",
        )


def _write_rlbench_metadata(
    source_root: Path,
    build_root: Path,
    final_root: Path,
    canonical_tasks: list[str],
    episode_mapping: dict[int, int],
) -> None:
    source_meta = source_root / "meta"
    target_meta = build_root / "meta"
    conversion = json.loads((source_meta / "rlbench_conversion.json").read_text())
    conversion.update(
        {
            "episode_count": len(episode_mapping),
            "tasks": canonical_tasks,
            "subset_source_dataset": str(source_root),
            "subset_episode_indices_reindexed": True,
        }
    )
    (target_meta / "rlbench_conversion.json").write_text(
        json.dumps(conversion, indent=2) + "\n", encoding="utf-8"
    )

    complete = json.loads((source_meta / "rlbench_conversion_complete.json").read_text())
    complete["episode_count"] = len(episode_mapping)
    (target_meta / "rlbench_conversion_complete.json").write_text(
        json.dumps(complete, indent=2) + "\n", encoding="utf-8"
    )

    pipeline = json.loads((source_meta / "rlbench_collection_pipeline.json").read_text())
    pipeline.update(
        {
            "dataset_root": str(final_root),
            "pointseg_cache": str(final_root) + "_pointseg_cache",
            "subset_source_dataset": str(source_root),
            "subset_episode_indices_reindexed": True,
        }
    )
    (target_meta / "rlbench_collection_pipeline.json").write_text(
        json.dumps(pipeline, indent=2) + "\n", encoding="utf-8"
    )

    mapping = {
        "source_dataset": str(source_root),
        "output_dataset": str(final_root),
        "tasks": canonical_tasks,
        "old_to_new_episode_index": {
            str(old): new for old, new in sorted(episode_mapping.items(), key=lambda item: item[1])
        },
        "storage": {
            "data_parquet": "rewritten and reindexed",
            "point_clouds_and_sidecars": "hardlinked immutable payloads with reindexed filenames",
        },
    }
    (target_meta / "rlbench_subset_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )


def _validate_build(
    root: Path,
    expected_episodes: int,
    expected_frames: int,
    expected_tasks: int,
) -> None:
    dataset = LeRobotDataset(root.name, root=root)
    if dataset.meta.total_episodes != expected_episodes:
        raise RuntimeError("Unexpected total_episodes")
    if dataset.meta.total_frames != expected_frames or len(dataset) != expected_frames:
        raise RuntimeError("Unexpected total_frames")
    if dataset.meta.total_tasks != expected_tasks:
        raise RuntimeError("Unexpected total_tasks")

    episode_indices = []
    task_indices = []
    global_indices = []
    frame_count = 0
    for data_file in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            data_file, columns=["episode_index", "task_index", "index", "frame_index"]
        )
        episode_indices.extend(table["episode_index"].to_pylist())
        task_indices.extend(table["task_index"].to_pylist())
        global_indices.extend(table["index"].to_pylist())
        frame_count += len(table)
    if frame_count != expected_frames:
        raise RuntimeError("Parquet frame count mismatch")
    if set(episode_indices) != set(range(expected_episodes)):
        raise RuntimeError("Episode indices are not contiguous")
    if set(task_indices) != set(range(expected_tasks)):
        raise RuntimeError("Task indices are not contiguous")
    if global_indices != list(range(expected_frames)):
        raise RuntimeError("Global indices are not contiguous")

    for episode_index in range(expected_episodes):
        episode = dataset.meta.episodes[episode_index]
        zarr_attrs = json.loads(
            (root / "point_clouds" / f"episode_{episode_index:06d}.zarr" / ".zattrs").read_text()
        )
        if zarr_attrs["shape"] != [int(episode["length"]), 20000, 6]:
            raise RuntimeError(f"Point cloud shape mismatch for episode {episode_index}")
    for directory in (
        "world_ee_poses",
        "raw_expert_actions",
        "raw_expert_actions_full",
        "initial_task_states",
        "initial_object_states",
    ):
        if len(list((root / directory).glob("episode_*"))) != expected_episodes:
            raise RuntimeError(f"Sidecar count mismatch for {directory}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    final_root = args.output_root.expanduser().resolve()
    build_root = args.build_root.expanduser().resolve()
    if final_root.exists():
        raise FileExistsError(f"Output already exists: {final_root}")
    if build_root.exists():
        raise FileExistsError(f"Build directory already exists: {build_root}")
    build_root.parent.mkdir(parents=True, exist_ok=True)

    source_dataset = LeRobotDataset(str(source_root), root=source_root)
    conversion = json.loads((source_root / "meta" / "rlbench_conversion.json").read_text())
    source_canonical_tasks = [str(task) for task in conversion["tasks"]]
    canonical_tasks = list(dict.fromkeys(args.tasks))
    missing_tasks = sorted(set(canonical_tasks) - set(source_canonical_tasks))
    if missing_tasks:
        raise ValueError(f"Tasks are absent from source dataset: {missing_tasks}")

    old_task_indices = [source_canonical_tasks.index(task) for task in canonical_tasks]
    task_descriptions = [str(source_dataset.meta.tasks.iloc[index].name) for index in old_task_indices]
    episodes_by_description = {description: [] for description in task_descriptions}
    for old_episode in range(source_dataset.meta.total_episodes):
        descriptions = source_dataset.meta.episodes[old_episode]["tasks"]
        if descriptions and descriptions[0] in episodes_by_description:
            episodes_by_description[descriptions[0]].append(old_episode)
    if args.expected_episodes_per_task is not None:
        bad = {
            description: len(episodes)
            for description, episodes in episodes_by_description.items()
            if len(episodes) != args.expected_episodes_per_task
        }
        if bad:
            raise RuntimeError(f"Unexpected task episode counts: {bad}")

    ordered_old_episodes = [
        episode
        for description in task_descriptions
        for episode in episodes_by_description[description]
    ]
    episode_mapping = {
        old_episode: new_episode
        for new_episode, old_episode in enumerate(ordered_old_episodes)
    }
    expected_frames = sum(
        int(source_dataset.meta.episodes[old_episode]["length"])
        for old_episode in ordered_old_episodes
    )

    destination_meta = LeRobotDatasetMetadata.create(
        repo_id=final_root.name,
        fps=source_dataset.meta.fps,
        features=source_dataset.meta.features,
        robot_type=source_dataset.meta.robot_type,
        root=build_root,
        use_videos=False,
        chunks_size=source_dataset.meta.chunks_size,
        data_files_size_in_mb=source_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=source_dataset.meta.video_files_size_in_mb,
    )
    destination_meta.save_episode_tasks(task_descriptions)
    data_metadata = _copy_and_reindex_data(source_dataset, destination_meta, episode_mapping)
    _compact_data_files(build_root, data_metadata)
    _copy_and_reindex_episodes_metadata(
        source_dataset, destination_meta, episode_mapping, data_metadata
    )

    new_task_by_episode = {}
    for new_task_index, description in enumerate(task_descriptions):
        for old_episode in episodes_by_description[description]:
            new_task_by_episode[episode_mapping[old_episode]] = new_task_index
    _patch_reindexed_episode_stats(
        build_root,
        episode_mapping,
        source_dataset.meta.episodes,
        new_task_by_episode,
        set(destination_meta.features),
    )
    _hardlink_episode_sidecars(source_root, build_root, episode_mapping)
    _write_rlbench_metadata(
        source_root, build_root, final_root, canonical_tasks, episode_mapping
    )
    _validate_build(
        build_root,
        expected_episodes=len(episode_mapping),
        expected_frames=expected_frames,
        expected_tasks=len(canonical_tasks),
    )
    os.replace(build_root, final_root)
    _validate_build(
        final_root,
        expected_episodes=len(episode_mapping),
        expected_frames=expected_frames,
        expected_tasks=len(canonical_tasks),
    )
    print(
        json.dumps(
            {
                "output_root": str(final_root),
                "tasks": canonical_tasks,
                "episodes": len(episode_mapping),
                "frames": expected_frames,
                "episode_mapping": {
                    canonical_tasks[0]: [0, len(episodes_by_description[task_descriptions[0]]) - 1],
                    canonical_tasks[1]: [
                        len(episodes_by_description[task_descriptions[0]]),
                        len(episode_mapping) - 1,
                    ],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
