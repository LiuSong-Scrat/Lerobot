#!/usr/bin/env python3
"""Relabel LeRobot action gripper width from the same-row observation state.

This is the gripper-label convention used by the RLBench ``*_action9_from_
observation_state9_*`` datasets.  Data parquet files are staged before any
existing file is replaced; episode and global action statistics are updated by
copying dimension 9 from ``observation.state`` statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def replace_last(values: object, replacement: object) -> list:
    output = np.asarray(values).tolist()
    source = np.asarray(replacement).tolist()
    if not isinstance(output, list) or not isinstance(source, list) or not output or not source:
        raise RuntimeError("Expected non-empty vector statistics")
    output[-1] = source[-1]
    return output


def relabel_table(table: pa.Table) -> tuple[pa.Table, int]:
    action_index = table.schema.get_field_index("action")
    state_index = table.schema.get_field_index("observation.state")
    if action_index < 0 or state_index < 0:
        raise RuntimeError("Dataset must contain action and observation.state")
    action_column = table["action"].combine_chunks()
    state_column = table["observation.state"].combine_chunks()
    if not pa.types.is_fixed_size_list(action_column.type) or action_column.type.list_size != 10:
        raise RuntimeError("Expected action fixed-size list width 10")
    if not pa.types.is_fixed_size_list(state_column.type) or state_column.type.list_size != 10:
        raise RuntimeError("Expected observation.state fixed-size list width 10")
    actions = np.asarray(action_column.values).reshape(len(table), 10).copy()
    states = np.asarray(state_column.values).reshape(len(table), 10)
    changed = int(np.count_nonzero(~np.isclose(actions[:, 9], states[:, 9], rtol=0.0, atol=0.0)))
    actions[:, 9] = states[:, 9]
    values = pa.array(actions.reshape(-1), type=action_column.type.value_type)
    replacement = pa.FixedSizeListArray.from_arrays(values, 10)
    return table.set_column(action_index, table.schema.field(action_index), replacement), changed


def relabel_episode_metadata(source: Path, destination: Path) -> int:
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        for stat in STAT_NAMES:
            action_key = f"stats/action/{stat}"
            state_key = f"stats/observation.state/{stat}"
            row[action_key] = replace_last(row[action_key], row[state_key])
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), destination, compression="snappy")
    return len(rows)


def relabel_global_stats(source: Path, destination: Path) -> None:
    stats = json.loads(source.read_text(encoding="utf-8"))
    for stat in STAT_NAMES:
        stats["action"][stat] = replace_last(
            stats["action"][stat], stats["observation.state"][stat]
        )
    atomic_json(destination, stats)


def verify(root: Path) -> tuple[int, float]:
    rows = 0
    maximum = 0.0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = np.asarray(table["action"].combine_chunks().values).reshape(len(table), 10)
        states = np.asarray(table["observation.state"].combine_chunks().values).reshape(len(table), 10)
        if len(table):
            maximum = max(maximum, float(np.max(np.abs(actions[:, 9] - states[:, 9]))))
        rows += len(table)
    if maximum != 0.0:
        raise RuntimeError(f"Relabel verification failed: max difference={maximum}")
    return rows, maximum


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    backup = args.backup_root.expanduser().resolve()
    if backup.exists():
        raise FileExistsError("Backup root already exists: " + str(backup))
    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise FileNotFoundError("LeRobot parquet data or episode metadata is missing")
    stage = root.parent / ("." + root.name + ".gripper_relabel_staging")
    if stage.exists():
        raise FileExistsError("Staging directory already exists: " + str(stage))
    changed = 0
    episode_count = 0
    try:
        for path in data_files:
            destination = stage / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            table, file_changed = relabel_table(pq.read_table(path))
            pq.write_table(table, destination, compression="snappy", use_dictionary=True)
            changed += file_changed
        for path in episode_files:
            episode_count += relabel_episode_metadata(path, stage / path.relative_to(root))
        relabel_global_stats(root / "meta" / "stats.json", stage / "meta" / "stats.json")

        audit = {
            "schema": "rlbench_action9_from_same_row_observation_state9_v1",
            "created_unix_s": time.time(),
            "dataset_root": str(root),
            "rows_where_value_changed": changed,
            "episode_count": episode_count,
            "applied": bool(args.apply),
        }
        atomic_json(stage / "meta" / "action_gripper_relabel.json", audit)
        if not args.apply:
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return

        backup.mkdir(parents=True)
        installed: list[tuple[Path, Path]] = []
        try:
            for relative in (Path("data"), Path("meta/episodes"), Path("meta/stats.json")):
                current = root / relative
                staged = stage / relative
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(current, saved)
                installed.append((current, saved))
                os.replace(staged, current)
            existing_audit = root / "meta" / "action_gripper_relabel.json"
            if existing_audit.exists():
                saved = backup / "meta" / "action_gripper_relabel.json"
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(existing_audit, saved)
                installed.append((existing_audit, saved))
            os.replace(stage / "meta" / "action_gripper_relabel.json", existing_audit)
        except Exception:
            for current, saved in reversed(installed):
                if current.exists():
                    failed = stage / "failed_install" / current.relative_to(root)
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(current, failed)
                if saved.exists():
                    current.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(saved, current)
            raise
        rows, maximum = verify(root)
        audit.update({"rows_processed": rows, "max_abs_action9_state9_difference": maximum})
        atomic_json(root / "meta" / "action_gripper_relabel.json", audit)
        atomic_json(backup / "relabel_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
