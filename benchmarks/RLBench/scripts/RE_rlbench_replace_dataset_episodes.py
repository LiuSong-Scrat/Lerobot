#!/usr/bin/env python3
"""Replace LeRobot episodes with replay-gated candidates without raw artifacts.

The data parquet stream is rebuilt in a sibling staging directory so replacement
episodes may have a different number of frames.  Point clouds and RLBench
sidecars are replaced per episode.  The original mutable files are moved to a
backup directory before the staged files are installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


STRUCTURAL_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
SIDECARS = {
    "point_clouds": ".zarr",
    "initial_task_states": ".npz",
    "initial_object_states": ".npz",
    "raw_expert_actions": ".npy",
    "raw_expert_actions_full": ".npy",
    "world_ee_poses": ".npy",
    "world_base_ee_poses": ".npy",
    "world_base_action_target_ee_poses": ".npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidate-dataset-root", type=Path, required=True)
    parser.add_argument("--target-replay-summary", type=Path, required=True)
    parser.add_argument("--candidate-replay-summary", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        required=True,
        metavar="TARGET_GLOBAL_EP=CANDIDATE_GLOBAL_EP",
    )
    parser.add_argument("--target-file-size-mb", type=int, default=100)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise RuntimeError("Expected JSON object: " + str(path))
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def parse_mappings(values: list[str]) -> dict[int, int]:
    mappings: dict[int, int] = {}
    candidates: set[int] = set()
    for value in values:
        left, separator, right = value.partition("=")
        if not separator:
            raise ValueError("Expected TARGET=CANDIDATE, got " + repr(value))
        target, candidate = int(left), int(right)
        if target in mappings or candidate in candidates:
            raise ValueError("Target and candidate episode indices must each be unique")
        mappings[target] = candidate
        candidates.add(candidate)
    return mappings


def read_episode_rows(root: Path) -> tuple[pa.Schema, list[dict]]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No episode metadata under " + str(root))
    tables = [pq.read_table(path) for path in files]
    table = pa.concat_tables(tables)
    rows = sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("Episode metadata is not contiguous under " + str(root))
    return tables[0].schema, rows


def indexed_replay_records(path: Path) -> dict[int, dict]:
    records = load_json(path).get("records")
    if not isinstance(records, list):
        raise RuntimeError("Replay summary has no records list: " + str(path))
    indexed = {}
    for summary_row in records:
        row = dict(summary_row)
        result_path = row.get("result_json")
        if result_path and Path(result_path).is_file():
            # The merged summary deliberately keeps only concise fields.  Load
            # the authoritative per-episode result to validate controller and
            # gripper settings rather than trusting a directory name.
            result = load_json(Path(result_path))
            for key in (
                "mode",
                "action_source",
                "controller_profile",
                "mover_max_tries",
                "clip_within_workspace",
                "gripper_after_reach",
                "pointact_pyrep_compat",
                "gripper_mode",
                "gripper_delta_alignment",
                "gripper_delta_threshold_m",
                "planner_max_time_ms",
                "water_plant_collision",
                "water_drop_collision",
            ):
                if key in result:
                    row[key] = result[key]
        indexed[int(row["global_episode"])] = row
    return indexed


def validate_replay_gate(row: dict, *, must_succeed: bool) -> None:
    if bool(row.get("success")) != must_succeed:
        raise RuntimeError("Replay success gate mismatch: " + repr(row))
    expected = {
        "mode": "eef0_planning",
        "action_source": "parquet",
        "controller_profile": "pointact_eval",
        "mover_max_tries": 10,
        "clip_within_workspace": True,
        "gripper_after_reach": True,
        "pointact_pyrep_compat": True,
        "gripper_mode": "delta_width_initial_sync",
        "gripper_delta_alignment": "current_minus_previous",
        "planner_max_time_ms": 1000,
        "water_plant_collision": "enabled",
        "water_drop_collision": "original",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"Replay gate requires {key}={value!r}, got {row.get(key)!r}")
    if abs(float(row.get("gripper_delta_threshold_m", -1.0)) - 0.003) > 1e-12:
        raise RuntimeError("Replay gate requires gripper delta threshold 0.003")


def compatible_info(target: dict, candidate: dict) -> None:
    for key in ("codebase_version", "robot_type", "fps"):
        if target.get(key) != candidate.get(key):
            raise RuntimeError(f"Dataset info differs for {key}: {target.get(key)!r} != {candidate.get(key)!r}")
    if target.get("features") != candidate.get("features"):
        raise RuntimeError("Target and candidate LeRobot feature schemas differ")


def data_path(root: Path, row: dict) -> Path:
    return root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"


class EpisodeReader:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.table: pa.Table | None = None

    def read(self, root: Path, row: dict) -> pa.Table:
        path = data_path(root, row)
        if self.path != path:
            self.table = pq.read_table(path)
            self.path = path
        assert self.table is not None
        episode = int(row["episode_index"])
        result = self.table.filter(pc.equal(self.table["episode_index"], episode))
        if result.num_rows != int(row["length"]):
            raise RuntimeError(f"Episode row mismatch for {root} episode {episode}")
        return result


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError("Missing data column " + name)
    field = table.schema.field(index)
    return table.set_column(index, field, values.cast(field.type))


def reindex_table(table: pa.Table, target_episode: int, target_task: int, global_from: int) -> pa.Table:
    length = table.num_rows
    table = replace_column(table, "timestamp", pa.array(np.arange(length, dtype=np.float32) / 20.0))
    table = replace_column(table, "frame_index", pa.array(np.arange(length, dtype=np.int64)))
    table = replace_column(table, "episode_index", pa.array(np.full(length, target_episode, dtype=np.int64)))
    table = replace_column(table, "index", pa.array(np.arange(global_from, global_from + length, dtype=np.int64)))
    table = replace_column(table, "task_index", pa.array(np.full(length, target_task, dtype=np.int64)))
    return table


def shift_stat(stats: dict, delta: int) -> None:
    if delta == 0:
        return
    for key in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        if key in stats:
            stats[key] = (np.asarray(stats[key], dtype=np.float64) + delta).tolist()


def stats_from_row(row: dict) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, feature, stat = parts
        array = np.asarray(value)
        if array.dtype == object:
            array = np.asarray(value, dtype=np.float64)
        result.setdefault(feature, {})[stat] = array
    return result


def row_with_shifted_stats(source: dict, *, target_episode: int, target_task: int, global_from: int) -> dict:
    row = json.loads(json.dumps(source))
    old_episode = int(source["episode_index"])
    old_from = int(source["dataset_from_index"])
    for feature, delta in {
        "episode_index": target_episode - old_episode,
        "task_index": target_task - int(np.asarray(source["stats/task_index/mean"])[0]),
        "index": global_from - old_from,
    }.items():
        prefix = "stats/" + feature + "/"
        stats = {key[len(prefix):]: value for key, value in row.items() if key.startswith(prefix)}
        shift_stat(stats, delta)
        for stat, value in stats.items():
            row[prefix + stat] = value
    return row


def serializable_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    return {feature: {key: np.asarray(value).tolist() for key, value in values.items()} for feature, values in stats.items()}


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def stage_sidecars(stage: Path, target_root: Path, candidate_root: Path, mappings: dict[int, int]) -> list[dict]:
    records = []
    for target, candidate in mappings.items():
        for directory, suffix in SIDECARS.items():
            source = candidate_root / directory / f"episode_{candidate:06d}{suffix}"
            current = target_root / directory / f"episode_{target:06d}{suffix}"
            destination = stage / "sidecars" / directory / current.name
            if not source.exists() or not current.exists():
                raise FileNotFoundError(f"Missing sidecar source/current: {source} / {current}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, copy_function=hardlink_or_copy)
            else:
                hardlink_or_copy(str(source), str(destination))
            records.append({"directory": directory, "target": target, "candidate": candidate})
    return records


def write_non_image_csv(data_root: Path, output: Path, features: dict) -> None:
    """Recreate the repository's flattened non-image inspection CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for path in sorted(data_root.rglob("*.parquet")):
        table = pq.read_table(path)
        columns: dict[str, object] = {}
        for name in table.column_names:
            if name.startswith("observation.images."):
                continue
            column = table[name].combine_chunks()
            if pa.types.is_fixed_size_list(column.type):
                values = np.asarray(column.values).reshape(len(column), column.type.list_size)
                component_names = features[name].get("names") or list(range(column.type.list_size))
                if len(component_names) != column.type.list_size:
                    raise RuntimeError("Feature names do not match fixed-list width for " + name)
                for index, component in enumerate(component_names):
                    columns[name + "." + str(component)] = values[:, index]
            else:
                columns[name] = column.to_numpy(zero_copy_only=False)
        pd.DataFrame(columns).to_csv(output, mode="w" if first else "a", header=first, index=False)
        first = False


def build_staging(
    stage: Path,
    target_root: Path,
    candidate_root: Path,
    target_rows: list[dict],
    candidate_rows: list[dict],
    episode_schema: pa.Schema,
    mappings: dict[int, int],
    target_file_size_mb: int,
    features: dict,
) -> tuple[list[dict], int]:
    from lerobot.datasets.compute_stats import aggregate_stats

    target_reader, candidate_reader = EpisodeReader(), EpisodeReader()
    staged_rows: list[dict] = []
    staged_stats: list[dict[str, dict[str, np.ndarray]]] = []
    file_tables: list[pa.Table] = []
    file_bytes = 0
    file_index = 0
    global_from = 0
    threshold = target_file_size_mb * 1024 * 1024

    def flush() -> None:
        nonlocal file_tables, file_bytes, file_index
        if not file_tables:
            return
        path = stage / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(file_tables), path, compression="snappy", use_dictionary=True)
        file_tables, file_bytes = [], 0
        file_index += 1

    for target_episode, target_row in enumerate(target_rows):
        candidate_episode = mappings.get(target_episode)
        if candidate_episode is None:
            source_root, source_row, reader = target_root, target_row, target_reader
        else:
            source_root, source_row, reader = candidate_root, candidate_rows[candidate_episode], candidate_reader
        table = reader.read(source_root, source_row)
        target_task = int(np.asarray(target_row["stats/task_index/mean"])[0])
        table = reindex_table(table, target_episode, target_task, global_from)
        if file_tables and file_bytes + table.nbytes > threshold:
            flush()
        assigned_file = file_index
        file_tables.append(table)
        file_bytes += table.nbytes

        row = row_with_shifted_stats(
            source_row,
            target_episode=target_episode,
            target_task=target_task,
            global_from=global_from,
        )
        length = table.num_rows
        row.update(
            {
                "episode_index": target_episode,
                "tasks": target_row["tasks"],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": assigned_file,
                "dataset_from_index": global_from,
                "dataset_to_index": global_from + length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        staged_rows.append(row)
        staged_stats.append(stats_from_row(row))
        global_from += length
    flush()

    episodes_path = stage / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(staged_rows, schema=episode_schema), episodes_path, compression="snappy")
    aggregated = aggregate_stats(staged_stats)
    atomic_json(stage / "meta" / "stats.json", serializable_stats(aggregated))
    write_non_image_csv(stage / "data", stage / "data_non_image_flat.csv", features)
    return staged_rows, global_from


def validate_staging(stage: Path, rows: list[dict], total_frames: int) -> None:
    files = sorted((stage / "data").rglob("*.parquet"))
    row_count = 0
    next_index = 0
    for path in files:
        table = pq.read_table(path, columns=["index", "episode_index", "frame_index"])
        indices = table["index"].to_numpy()
        if len(indices) and (indices[0] != next_index or not np.array_equal(indices, np.arange(next_index, next_index + len(indices)))):
            raise RuntimeError("Staged global indices are not contiguous at " + str(path))
        next_index += len(indices)
        row_count += len(indices)
    if row_count != total_frames or next_index != total_frames:
        raise RuntimeError(f"Staged row count mismatch: {row_count} != {total_frames}")
    if int(rows[-1]["dataset_to_index"]) != total_frames:
        raise RuntimeError("Staged episode metadata final index mismatch")


def main() -> None:
    args = parse_args()
    target_root = args.dataset_root.expanduser().resolve()
    candidate_root = args.candidate_dataset_root.expanduser().resolve()
    backup_root = args.backup_root.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    mappings = parse_mappings(args.replace)
    if backup_root.exists():
        raise FileExistsError("Backup root already exists: " + str(backup_root))

    target_info = load_json(target_root / "meta" / "info.json")
    candidate_info = load_json(candidate_root / "meta" / "info.json")
    compatible_info(target_info, candidate_info)
    episode_schema, target_rows = read_episode_rows(target_root)
    _, candidate_rows = read_episode_rows(candidate_root)
    target_replays = indexed_replay_records(args.target_replay_summary)
    candidate_replays = indexed_replay_records(args.candidate_replay_summary)
    checks = []
    for target, candidate in mappings.items():
        if target >= len(target_rows) or candidate >= len(candidate_rows):
            raise IndexError("Replacement episode index is outside dataset")
        target_replay = target_replays[target]
        candidate_replay = candidate_replays[candidate]
        validate_replay_gate(target_replay, must_succeed=False)
        validate_replay_gate(candidate_replay, must_succeed=True)
        if target_replay["task"] != candidate_replay["task"]:
            raise RuntimeError(f"Cross-task replacement forbidden: {target_replay['task']} != {candidate_replay['task']}")
        checks.append(
            {
                "target_global_episode": target,
                "candidate_global_episode": candidate,
                "task": target_replay["task"],
                "target_frames": int(target_rows[target]["length"]),
                "candidate_frames": int(candidate_rows[candidate]["length"]),
            }
        )

    stage = target_root.parent / ("." + target_root.name + ".replacement_staging_" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        staged_rows, total_frames = build_staging(
            stage,
            target_root,
            candidate_root,
            target_rows,
            candidate_rows,
            episode_schema,
            mappings,
            args.target_file_size_mb,
            target_info["features"],
        )
        sidecars = stage_sidecars(stage, target_root, candidate_root, mappings)
        info = dict(target_info)
        info["total_frames"] = total_frames
        atomic_json(stage / "meta" / "info.json", info)
        validate_staging(stage, staged_rows, total_frames)
        audit = {
            "schema": "rlbench_replay_gated_dataset_episode_replacement_v1",
            "created_unix_s": time.time(),
            "dataset_root": str(target_root),
            "candidate_dataset_root": str(candidate_root),
            "target_replay_summary": str(args.target_replay_summary.resolve()),
            "candidate_replay_summary": str(args.candidate_replay_summary.resolve()),
            "gate": {
                "mode": "eef0_planning",
                "action_source": "parquet",
                "controller_profile": "pointact_eval",
                "mover_max_tries": 10,
                "clip_within_workspace": True,
                "gripper_after_reach": True,
                "pointact_pyrep_compat": True,
                "planner_max_time_ms": 1000,
                "gripper_mode": "delta_width_initial_sync",
                "gripper_delta_threshold_m": 0.003,
                "gripper_delta_alignment": "current_minus_previous",
                "water_plant_collision": "enabled",
                "water_drop_collision": "original",
            },
            "old_total_frames": int(target_info["total_frames"]),
            "new_total_frames": total_frames,
            "replacements": checks,
            "sidecars": sidecars,
            "applied": bool(args.apply),
        }
        atomic_json(stage / "audit.json", audit)
        if not args.apply:
            atomic_json(audit_output, audit)
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return

        backup_root.mkdir(parents=True)
        moves: list[tuple[Path, Path, bool]] = []

        def install(current: Path, staged: Path, backup: Path) -> None:
            backup.parent.mkdir(parents=True, exist_ok=True)
            had_current = current.exists()
            if had_current:
                os.replace(current, backup)
            moves.append((current, backup, had_current))
            os.replace(staged, current)

        try:
            install(target_root / "data", stage / "data", backup_root / "data")
            install(
                target_root / "meta" / "episodes",
                stage / "meta" / "episodes",
                backup_root / "meta" / "episodes",
            )
            for relative in (Path("meta/info.json"), Path("meta/stats.json"), Path("data_non_image_flat.csv")):
                install(target_root / relative, stage / relative, backup_root / relative)
            for record in sidecars:
                directory = record["directory"]
                target = int(record["target"])
                suffix = SIDECARS[directory]
                relative = Path(directory) / f"episode_{target:06d}{suffix}"
                install(target_root / relative, stage / "sidecars" / relative, backup_root / relative)
        except Exception:
            for current, backup, had_current in reversed(moves):
                if current.exists():
                    failed = stage / "failed_install" / current.relative_to(target_root)
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(current, failed)
                if had_current and backup.exists():
                    current.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, current)
            raise

        atomic_json(audit_output, audit)
        atomic_json(backup_root / "replacement_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
