#!/usr/bin/env python3
"""Export exact model rows aligned with actions that reached RLBench step().

The evaluator stores exact predicted chunks and exact world-frame controller
commands separately.  Its control log contains the integer model-call/chunk-row
mapping for every successful simulator step.  This tool joins those sources
without using the rounded floating-point values in the control log.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONTROL_AFTER_PREFIX = "[control-after] "
PHASE_CODES = {"move": 0, "gripper_after_reach": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed task directory containing actions/, model_chunks/, and control.log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory; defaults to RUN_DIR/executed_action_alignment.",
    )
    return parser.parse_args()


def load_control_after_records(path: Path) -> dict[int, list[dict]]:
    records: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.startswith(CONTROL_AFTER_PREFIX):
                continue
            payload = json.loads(line[len(CONTROL_AFTER_PREFIX) :])
            payload["_control_log_line"] = int(line_number)
            records[int(payload["episode_index"])].append(payload)
    return records


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "executed_action_alignment"
    )
    actions_dir = run_dir / "actions"
    chunks_dir = run_dir / "model_chunks"
    control_log = run_dir / "control.log"
    for required in (actions_dir, chunks_dir, control_log):
        if not required.exists():
            raise FileNotFoundError(f"Required evaluator output does not exist: {required}")

    records_by_episode = load_control_after_records(control_log)
    action_paths = sorted(actions_dir.glob("episode_*_actions.npy"))
    if not action_paths:
        raise RuntimeError(f"No episode action files found under {actions_dir}")

    all_model_actions: list[np.ndarray] = []
    all_controller_actions: list[np.ndarray] = []
    all_pairs: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    episode_summaries: list[dict] = []

    for action_path in action_paths:
        episode_index = int(action_path.stem.split("_")[1])
        chunk_path = chunks_dir / f"episode_{episode_index:03d}_model_chunks.npy"
        if not chunk_path.is_file():
            raise FileNotFoundError(f"Missing model chunks for episode {episode_index}: {chunk_path}")

        controller_actions = np.asarray(np.load(action_path), dtype=np.float32)
        model_chunks = np.asarray(np.load(chunk_path), dtype=np.float32)
        records = records_by_episode.get(episode_index, [])
        if controller_actions.ndim != 2 or controller_actions.shape[1] != 8:
            raise ValueError(
                f"Expected {action_path.name} to have shape (N, 8), got {controller_actions.shape}"
            )
        if model_chunks.ndim != 3 or model_chunks.shape[2] < 10:
            raise ValueError(
                f"Expected {chunk_path.name} to have shape (calls, rows, >=10), "
                f"got {model_chunks.shape}"
            )
        if len(records) != len(controller_actions):
            raise RuntimeError(
                f"Episode {episode_index}: {len(controller_actions)} controller actions but "
                f"{len(records)} successful control-after records"
            )

        model_actions = []
        execution_indices = []
        for record in records:
            model_call = int(record["model_call"])
            chunk_row = int(record["chunk_row_index"])
            if not 1 <= model_call <= model_chunks.shape[0]:
                raise IndexError(
                    f"Episode {episode_index}: model_call={model_call} outside "
                    f"1..{model_chunks.shape[0]}"
                )
            if not 0 <= chunk_row < model_chunks.shape[1]:
                raise IndexError(
                    f"Episode {episode_index}: chunk_row={chunk_row} outside "
                    f"0..{model_chunks.shape[1] - 1}"
                )
            phase = str(record.get("pointact_mover_phase", "move"))
            if phase not in PHASE_CODES:
                raise ValueError(f"Episode {episode_index}: unknown execution phase {phase!r}")
            model_actions.append(model_chunks[model_call - 1, chunk_row, :10])
            execution_indices.append(
                [
                    episode_index,
                    model_call,
                    chunk_row,
                    PHASE_CODES[phase],
                    int(record.get("pointact_mover_attempt", 0)),
                    int(record["_control_log_line"]),
                ]
            )

        model_actions_array = np.asarray(model_actions, dtype=np.float32)
        execution_indices_array = np.asarray(execution_indices, dtype=np.int64)
        paired_array = np.concatenate((model_actions_array, controller_actions), axis=1)

        prefix = output_dir / f"episode_{episode_index:03d}"
        save_array(
            prefix.with_name(prefix.name + "_executed_model_actions_relative10.npy"),
            model_actions_array,
        )

        all_model_actions.append(model_actions_array)
        all_controller_actions.append(controller_actions)
        all_pairs.append(paired_array)
        all_indices.append(execution_indices_array)
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "executed_environment_steps": int(len(controller_actions)),
                "model_calls": int(model_chunks.shape[0]),
                "move_steps": int(np.sum(execution_indices_array[:, 3] == 0)),
                "gripper_after_reach_steps": int(np.sum(execution_indices_array[:, 3] == 1)),
            }
        )

    combined_model = np.concatenate(all_model_actions, axis=0)
    combined_controller = np.concatenate(all_controller_actions, axis=0)
    combined_pairs = np.concatenate(all_pairs, axis=0)
    combined_indices = np.concatenate(all_indices, axis=0)
    manifest = {
        "source_run_dir": str(run_dir),
        "episodes": len(episode_summaries),
        "executed_environment_steps": int(len(combined_model)),
        "alignment": "one row per successful task_env.step(command)",
        "model_action_columns": [
            "relative_x",
            "relative_y",
            "relative_z",
            "rotation_column_1_x",
            "rotation_column_1_y",
            "rotation_column_1_z",
            "rotation_column_2_x",
            "rotation_column_2_y",
            "rotation_column_2_z",
            "predicted_gripper_width_m",
        ],
        "controller_action_columns": [
            "world_x",
            "world_y",
            "world_z",
            "qx",
            "qy",
            "qz",
            "qw",
            "gripper_open_discrete",
        ],
        "execution_index_columns": [
            "episode_index",
            "model_call_1based",
            "chunk_row_index_0based",
            "execution_phase_code",
            "mover_attempt_1based_or_0_for_gripper_phase",
            "control_log_line_1based",
        ],
        "execution_phase_codes": {"0": "move", "1": "gripper_after_reach"},
        "precision_note": (
            "The model10 values come from exact float32 model_chunks.npy values; "
            "the rounded values stored in control.log are not used as action data."
        ),
        "episode_summaries": episode_summaries,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
