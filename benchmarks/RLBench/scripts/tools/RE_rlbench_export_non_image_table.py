#!/usr/bin/env python3
"""Export every non-image column of an RLBench LeRobot dataset for diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.dataset as pads
import pyarrow.parquet as pq


TASK_ALIASES = {
    "put the phone on the base": "phone_on_base",
    "phone on base": "phone_on_base",
    "water plant": "water_plants",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_task(task: str) -> str:
    normalized = " ".join(task.strip().lower().replace("_", " ").split())
    return TASK_ALIASES.get(normalized, normalized.replace(" ", "_"))


def flatten_value(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, (list, tuple)):
        return {f"{name}[{index}]": item for index, item in enumerate(value)}
    return {name: value}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info_path = dataset_root / "meta/info.json"
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    data_dir = dataset_root / "data"

    if not info_path.is_file() or not episodes_path.is_file() or not data_dir.is_dir():
        raise FileNotFoundError(f"Not a complete LeRobot dataset: {dataset_root}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    image_columns = {
        name
        for name, spec in features.items()
        if spec.get("dtype") in {"image", "video"} or name.startswith("observation.images.")
    }

    dataset = pads.dataset(data_dir, format="parquet")
    source_columns = list(dataset.schema.names)
    non_image_columns = [name for name in source_columns if name not in image_columns]
    excluded_columns = [name for name in source_columns if name in image_columns]
    if not excluded_columns:
        raise ValueError(f"No image column was detected in {source_columns!r}")

    episode_table = pq.read_table(episodes_path, columns=["episode_index", "tasks", "length"])
    episode_meta: dict[int, dict[str, Any]] = {}
    task_episode_counts: Counter[str] = Counter()
    for episode_index, tasks, length in zip(
        episode_table["episode_index"].to_pylist(),
        episode_table["tasks"].to_pylist(),
        episode_table["length"].to_pylist(),
        strict=True,
    ):
        if not tasks:
            raise ValueError(f"Episode {episode_index} has no task description")
        task = canonical_task(tasks[0])
        task_episode_index = task_episode_counts[task]
        task_episode_counts[task] += 1
        episode_meta[int(episode_index)] = {
            "task_name": task,
            "task_episode_index": task_episode_index,
            "length": int(length),
        }

    table = dataset.to_table(columns=non_image_columns)
    records: list[dict[str, Any]] = []
    for source_row in table.to_pylist():
        episode_index = int(source_row["episode_index"])
        meta = episode_meta[episode_index]
        record: dict[str, Any] = {
            "task_name": meta["task_name"],
            "task_episode_index": meta["task_episode_index"],
        }
        for column in non_image_columns:
            record.update(flatten_value(column, source_row[column]))
        records.append(record)

    records.sort(key=lambda item: int(item["index"]))
    if not records:
        raise RuntimeError("Dataset contains no rows")
    expected_rows = int(info["total_frames"])
    if len(records) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, exported {len(records)}")
    if [int(record["index"]) for record in records] != list(range(expected_rows)):
        raise RuntimeError("Dataset index is not contiguous after sorting")

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    all_path = output_dir / "all_tasks_non_image_data.csv"
    write_csv(all_path, records)

    task_summary: dict[str, Any] = {}
    for task in sorted(task_episode_counts):
        task_records = [record for record in records if record["task_name"] == task]
        task_path = output_dir / f"{task}_non_image_data.csv"
        write_csv(task_path, task_records)
        task_summary[task] = {
            "episodes": int(task_episode_counts[task]),
            "frames": len(task_records),
            "output": str(task_path),
        }

    summary = {
        "dataset_root": str(dataset_root),
        "total_rows": len(records),
        "source_non_image_columns": non_image_columns,
        "excluded_image_columns": excluded_columns,
        "output_columns": fieldnames,
        "all_tasks_output": str(all_path),
        "tasks": task_summary,
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# RLBench 非图像数据表

数据集：`{dataset_root}`

- 总帧数：{len(records)}
- 排除的图像字段：{', '.join(f'`{name}`' for name in excluded_columns)}
- 完整表：`{all_path.name}`
- phone_on_base：`phone_on_base_non_image_data.csv`
- water_plants：`water_plants_non_image_data.csv`

`task_name` 和 `task_episode_index` 是根据 `meta/episodes` 添加的辅助列；其余列均来自
`data/**/*.parquet`。数组字段已经展开为 `action[0]` 至 `action[9]` 和
`observation.state[0]` 至 `observation.state[9]`。

其中 `action[9]` 是下一步专家夹爪目标宽度，`observation.state[9]` 是当前观测到的夹爪宽度。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
