#!/usr/bin/env python3
"""Audit phone release action[9] predictions around the second opening.

The phone trajectories start with an open gripper, close around the phone, and
then open again to release it.  This script calls the release onset the
"second opening": the first positive width delta after at least one negative
width delta.  Both deltas use the same threshold as strict
``delta_width_initial_sync`` evaluation.

One teacher-forced model call is made from a recorded observation before the
release.  Consequently, label and prediction rows on both sides of the
release belong to the same action chunk and are directly aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.dataset as pyarrow_dataset
import torch
from torch.utils.data._utils.collate import default_collate

from _rlbench_tool_paths import LEROBOT_ROOT, RLBENCH_ROOT, SCRIPTS_DIR


SONG_SCRIPT_DIR = LEROBOT_ROOT / "benchmarks" / "song_real_libero" / "scripts"
for path in (str(SONG_SCRIPT_DIR), str(LEROBOT_ROOT / "src"), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from smolvla_model_inference import SmolVLA_ModelInference


DEFAULT_DATASET = (
    RLBENCH_ROOT
    / "datasets/rlbench_10tasks_100traj_raw_expert_target_reap_v4_20k_20260823_"
    "action9_from_observation_state9_20260826"
)
DEFAULT_POLICY = (
    RLBENCH_ROOT
    / "outputs/wep_vla_v043-20000+20000_5000_2_fixed_gripper_2tasks_0826"
    / "checkpoints/014000/pretrained_model"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=3)
    parser.add_argument("--delta-threshold", type=float, default=0.003)
    parser.add_argument("--pre-context", type=int, default=8)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--seeds",
        default="20260827,20260828,20260829",
        help="Comma-separated deterministic flow-matching seeds.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.inference_mode()
def seeded_predict_preprocessed(infer, model_batch, seed):
    device = next(infer.policy.parameters()).device
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        noise = infer._make_seeded_action_noise(model_batch, int(seed))
        worldflow_noise = infer._make_seeded_worldflow_noise(model_batch, int(seed))
        action = infer.policy.predict_action_chunk(
            model_batch,
            noise=noise,
            worldflow_noise=worldflow_noise,
        )
    return infer.postprocessor(action).detach().cpu()


def event_name(delta, threshold):
    if float(delta) > float(threshold):
        return "open"
    if float(delta) < -float(threshold):
        return "close"
    return "keep"


def numeric_metrics(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    absolute = np.abs(array)
    return {
        "count": int(len(array)),
        "bias_m": float(np.mean(array)),
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(np.square(array)))),
        "median_abs_error_m": float(np.median(absolute)),
        "p90_abs_error_m": float(np.quantile(absolute, 0.90)),
        "max_abs_error_m": float(np.max(absolute)),
    }


def event_metrics(rows):
    pairs = Counter((row["label_event"], row["predicted_event"]) for row in rows)
    label_open = sum(count for (label, _), count in pairs.items() if label == "open")
    predicted_open = sum(count for (_, predicted), count in pairs.items() if predicted == "open")
    true_open = pairs[("open", "open")]
    correct = sum(count for (label, predicted), count in pairs.items() if label == predicted)
    return {
        "rows": int(len(rows)),
        "accuracy": float(correct / len(rows)) if rows else None,
        "label_open_rows": int(label_open),
        "predicted_open_rows": int(predicted_open),
        "open_true_positive_rows": int(true_open),
        "open_recall": float(true_open / label_open) if label_open else None,
        "open_precision": float(true_open / predicted_open) if predicted_open else None,
        "confusion": {
            f"label_{label}__pred_{predicted}": int(count)
            for (label, predicted), count in sorted(pairs.items())
        },
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_float(value, scale=1.0, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value) * scale:.{digits}f}"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if args.pre_context <= args.window:
        raise ValueError("pre-context must be greater than window so row deltas are aligned.")

    parquet = pyarrow_dataset.dataset(str(args.dataset_root / "data"), format="parquet")
    table = parquet.to_table(
        filter=pyarrow_dataset.field("task_index") == int(args.task_index),
        columns=["episode_index", "frame_index", "index", "action", "observation.state"],
    )
    episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_values = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    global_indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
    action_widths = np.asarray(table["action"].to_pylist(), dtype=np.float32)[:, 9]
    state_widths = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[:, 9]

    global_order = np.argsort(global_indices)
    global_to_local = {
        int(global_indices[source_row]): int(local_row)
        for local_row, source_row in enumerate(global_order)
    }
    episodes = []
    for episode in sorted(np.unique(episode_values).tolist()):
        selected = np.flatnonzero(episode_values == episode)
        selected = selected[np.argsort(frame_values[selected])]
        widths = action_widths[selected]
        deltas = np.diff(widths, prepend=widths[0])
        close_rows = np.flatnonzero(deltas < -args.delta_threshold)
        if not len(close_rows):
            continue
        open_rows = np.flatnonzero(
            (np.arange(len(widths)) > int(close_rows[0]))
            & (deltas > args.delta_threshold)
        )
        if not len(open_rows):
            continue
        release_row = int(open_rows[0])
        context_row = release_row - int(args.pre_context)
        # LeRobot pads the remainder of a 32-row action chunk at an episode
        # boundary.  The audit only consumes the explicitly requested window,
        # so require real rows through release + window rather than 32 rows.
        if context_row < 0 or release_row + int(args.window) >= len(selected):
            continue
        episodes.append(
            {
                "episode_index": int(episode),
                "selected": selected,
                "release_row": release_row,
                "context_row": context_row,
                "global_index": int(global_indices[selected[context_row]]),
                "local_dataset_index": global_to_local[int(global_indices[selected[context_row]])],
                "release_frame": int(frame_values[selected[release_row]]),
                "label_pre_width": float(widths[release_row - 1]),
                "label_release_width": float(widths[release_row]),
                "label_release_delta": float(deltas[release_row]),
            }
        )
    if args.max_episodes:
        episodes = episodes[: int(args.max_episodes)]
    if not episodes:
        raise RuntimeError("No valid second-opening episode was found.")

    selected_episode_ids = sorted(np.unique(episode_values).tolist())
    infer = SmolVLA_ModelInference(
        args.policy_path,
        device=args.device,
        visualize_foreground=False,
    )
    dataset = infer.load_dataset(args.dataset_root, episodes=selected_episode_ids)

    detail_rows = []
    trial_rows = []
    for episode_number, episode in enumerate(episodes, start=1):
        item = dict(dataset[episode["local_dataset_index"]])
        observed_global_index = int(to_numpy(item["index"]))
        if observed_global_index != episode["global_index"]:
            raise RuntimeError(
                "Dataset subset order mismatch: expected global index "
                f"{episode['global_index']}, got {observed_global_index}."
            )
        model_batch = infer.preprocessor(default_collate([item]))
        reference = to_numpy(model_batch["action"])[0, :, :10]
        if abs(float(reference[0, 9]) - float(action_widths[episode["selected"]][episode["context_row"]])) > 1e-6:
            raise RuntimeError("Teacher-forced action chunk is not aligned with parquet labels.")

        for seed in seeds:
            predicted = to_numpy(
                seeded_predict_preprocessed(
                    infer,
                    model_batch,
                    seed + int(episode["episode_index"]),
                )
            )[0, :, :10]
            predicted_deltas = np.diff(predicted[:, 9], prepend=predicted[0, 9])
            label_deltas = np.diff(reference[:, 9], prepend=reference[0, 9])
            first_predicted_open = next(
                (
                    row
                    for row in range(1, min(16, len(predicted_deltas)))
                    if predicted_deltas[row] > args.delta_threshold
                ),
                None,
            )
            timing = (
                int(first_predicted_open - args.pre_context)
                if first_predicted_open is not None
                else None
            )
            trial_rows.append(
                {
                    "episode_index": episode["episode_index"],
                    "seed": seed,
                    "release_frame": episode["release_frame"],
                    "label_pre_width_m": episode["label_pre_width"],
                    "label_release_width_m": episode["label_release_width"],
                    "label_release_delta_m": episode["label_release_delta"],
                    "predicted_release_width_m": float(predicted[args.pre_context, 9]),
                    "release_width_error_m": float(
                        predicted[args.pre_context, 9] - reference[args.pre_context, 9]
                    ),
                    "predicted_release_delta_m": float(predicted_deltas[args.pre_context]),
                    "predicted_release_event": event_name(
                        predicted_deltas[args.pre_context], args.delta_threshold
                    ),
                    "predicted_open_timing_offset_frames": timing,
                }
            )
            for offset in range(-args.window, args.window + 1):
                chunk_row = int(args.pre_context + offset)
                label_width = float(reference[chunk_row, 9])
                predicted_width = float(predicted[chunk_row, 9])
                label_delta = float(label_deltas[chunk_row])
                predicted_delta = float(predicted_deltas[chunk_row])
                detail_rows.append(
                    {
                        "episode_index": episode["episode_index"],
                        "seed": seed,
                        "release_frame": episode["release_frame"],
                        "relative_frame": offset,
                        "dataset_frame": episode["release_frame"] + offset,
                        "label_width_m": label_width,
                        "predicted_width_m": predicted_width,
                        "signed_error_m": predicted_width - label_width,
                        "abs_error_m": abs(predicted_width - label_width),
                        "label_delta_m": label_delta,
                        "predicted_delta_m": predicted_delta,
                        "label_event": event_name(label_delta, args.delta_threshold),
                        "predicted_event": event_name(predicted_delta, args.delta_threshold),
                    }
                )
        print(
            f"[phone-release-audit] episodes={episode_number}/{len(episodes)} "
            f"episode={episode['episode_index']} release_frame={episode['release_frame']}",
            flush=True,
        )

    offset_rows = []
    for offset in range(-args.window, args.window + 1):
        rows = [row for row in detail_rows if row["relative_frame"] == offset]
        errors = [row["signed_error_m"] for row in rows]
        metrics = numeric_metrics(errors)
        events = event_metrics(rows)
        offset_rows.append(
            {
                "relative_frame": offset,
                "samples": len(rows),
                "label_width_mean_m": float(np.mean([row["label_width_m"] for row in rows])),
                "predicted_width_mean_m": float(
                    np.mean([row["predicted_width_m"] for row in rows])
                ),
                **metrics,
                "event_accuracy": events["accuracy"],
                "open_recall": events["open_recall"],
                "open_precision": events["open_precision"],
                "label_open_rows": events["label_open_rows"],
                "predicted_open_rows": events["predicted_open_rows"],
            }
        )

    before_rows = [row for row in detail_rows if row["relative_frame"] < 0]
    release_rows = [row for row in detail_rows if row["relative_frame"] == 0]
    after_rows = [row for row in detail_rows if row["relative_frame"] > 0]
    timing_values = [
        row["predicted_open_timing_offset_frames"]
        for row in trial_rows
        if row["predicted_open_timing_offset_frames"] is not None
    ]
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "policy_path": str(args.policy_path.resolve()),
        "task_index": int(args.task_index),
        "episode_count": len(episodes),
        "seeds": seeds,
        "prediction_trials": len(trial_rows),
        "definition": {
            "second_opening": "first action[9] delta > threshold after a delta < -threshold",
            "delta_threshold_m": float(args.delta_threshold),
            "teacher_forced_context_frames_before_release": int(args.pre_context),
            "reported_window_frames": [-int(args.window), int(args.window)],
            "action9_label": "same-row observation.state[9] measured gripper width",
        },
        "label_release": {
            "frame_min": int(min(row["release_frame"] for row in trial_rows)),
            "frame_median": float(np.median([row["release_frame"] for row in trial_rows])),
            "frame_max": int(max(row["release_frame"] for row in trial_rows)),
            "pre_width_mean_m": float(np.mean([row["label_pre_width_m"] for row in trial_rows])),
            "release_delta_mean_m": float(
                np.mean([row["label_release_delta_m"] for row in trial_rows])
            ),
        },
        "continuous_width_error": {
            "before_release_minus5_to_minus1": numeric_metrics(
                [row["signed_error_m"] for row in before_rows]
            ),
            "release_frame": numeric_metrics([row["signed_error_m"] for row in release_rows]),
            "after_release_plus1_to_plus5": numeric_metrics(
                [row["signed_error_m"] for row in after_rows]
            ),
            "full_window": numeric_metrics([row["signed_error_m"] for row in detail_rows]),
        },
        "event_error": {
            "before_release_minus5_to_minus1": event_metrics(before_rows),
            "release_frame": event_metrics(release_rows),
            "after_release_plus1_to_plus5": event_metrics(after_rows),
            "full_window": event_metrics(detail_rows),
            "trials_with_any_predicted_open_in_executed_chunk": len(timing_values),
            "predicted_open_detection_rate": float(len(timing_values) / len(trial_rows)),
            "predicted_open_timing_offset_frames_median": (
                float(np.median(timing_values)) if timing_values else None
            ),
        },
        "offset_metrics": offset_rows,
    }

    detail_fields = list(detail_rows[0])
    trial_fields = list(trial_rows[0])
    offset_fields = list(offset_rows[0])
    write_csv(args.output_dir / "per_episode_seed_frame_errors.csv", detail_rows, detail_fields)
    write_csv(args.output_dir / "per_episode_seed_release_summary.csv", trial_rows, trial_fields)
    write_csv(args.output_dir / "aggregate_by_relative_frame.csv", offset_rows, offset_fields)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    release_metrics = summary["continuous_width_error"]["release_frame"]
    post_metrics = summary["continuous_width_error"]["after_release_plus1_to_plus5"]
    release_events = summary["event_error"]["release_frame"]
    post_events = summary["event_error"]["after_release_plus1_to_plus5"]
    report_lines = [
        "# Phone second-opening action[9] audit",
        "",
        f"- checkpoint: `{args.policy_path.resolve()}`",
        f"- dataset: `{args.dataset_root.resolve()}`",
        f"- episodes: {len(episodes)}; deterministic seeds: {', '.join(map(str, seeds))}",
        f"- strict delta threshold: {args.delta_threshold:.4f} m",
        f"- prediction context: {args.pre_context} recorded frames before the release onset",
        "- label semantics: `action[9] == observation.state[9]` on the same dataset row",
        "",
        "## Main result",
        "",
        f"- Release-frame width MAE: {release_metrics['mae_m'] * 1000:.3f} mm.",
        f"- Post-release (+1..+5) width MAE: {post_metrics['mae_m'] * 1000:.3f} mm.",
        f"- Release-frame open-event recall: {release_events['open_recall'] * 100:.1f}%.",
        f"- Post-release open-event recall: {post_events['open_recall'] * 100:.1f}%.",
        "- Any predicted opening in the 16 executed rows: "
        f"{summary['event_error']['trials_with_any_predicted_open_in_executed_chunk']}/"
        f"{len(trial_rows)} ({summary['event_error']['predicted_open_detection_rate'] * 100:.1f}%).",
        "",
        "## Aligned frame table",
        "",
        "| Relative frame | Label width mean (mm) | Predicted width mean (mm) | Bias (mm) | MAE (mm) | Open recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in offset_rows:
        open_recall = (
            "—"
            if row["open_recall"] is None
            else f"{markdown_float(row['open_recall'], 100, 1)}%"
        )
        report_lines.append(
            f"| {row['relative_frame']:+d} | "
            f"{markdown_float(row['label_width_mean_m'], 1000)} | "
            f"{markdown_float(row['predicted_width_mean_m'], 1000)} | "
            f"{markdown_float(row['bias_m'], 1000)} | "
            f"{markdown_float(row['mae_m'], 1000)} | "
            f"{open_recall} |"
        )
    report_lines.extend(
        [
            "",
            "Each seed is shifted by the global episode index, so trials are deterministic but do not reuse identical flow noise across episodes.",
            "The executed-chunk event calculation self-references predicted row 0, matching strict `delta_width_initial_sync`.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
