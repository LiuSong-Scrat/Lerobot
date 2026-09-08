#!/usr/bin/env python3
"""Compare two RLBench evaluations and prepare evidence for failure diagnosis.

The report deliberately separates directly measured rollout facts from manual
stage/root-cause labels.  Contact sheets sample the complete trajectory and add
frames around simulator gripper-command transitions so the earliest fatal event
can be reviewed without treating a downstream timeout as the cause.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


GRIPPER_REQUIRED_TASKS = {
    "close_box",
    "close_laptop_lid",
    "phone_on_base",
    "stack_wine",
    "sweep_to_dustpan",
    "take_frame_off_hanger",
    "take_umbrella_out_of_umbrella_stand",
    "toilet_seat_down",
    "water_plants",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-a", type=Path, required=True)
    parser.add_argument("--eval-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contact-sheets",
        choices=("none", "failures", "all"),
        default="none",
    )
    parser.add_argument("--sheet-frames", type=int, default=12)
    return parser.parse_args()


def task_summaries(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    # Supplemental reruns (for example, a task that originally failed before
    # episode 0 because of a launcher error) may live one level deeper inside
    # the canonical run root.  Discover those without treating root summary
    # files as task summaries.
    for path in sorted(root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        task = data.get("task")
        if task:
            found[str(task)] = (path.parent, data)
    return found


def as_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def relative_file(task_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else task_dir / path


def gripper_evidence(task_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    action_path = relative_file(task_dir, result.get("executed_actions"))
    aligned_path = relative_file(task_dir, result.get("executed_model_actions_relative10"))
    transitions: list[int] = []
    command_min = command_max = None
    if action_path and action_path.is_file():
        actions = np.load(action_path, mmap_mode="r")
        if actions.ndim == 2 and actions.shape[0] and actions.shape[1] >= 8:
            command = np.asarray(actions[:, 7], dtype=np.float64)
            command_min, command_max = float(np.nanmin(command)), float(np.nanmax(command))
            transitions = (np.flatnonzero(np.diff(command) != 0) + 1).astype(int).tolist()
    model_a9_min = model_a9_max = None
    if aligned_path and aligned_path.is_file():
        aligned = np.load(aligned_path, mmap_mode="r")
        if aligned.ndim == 2 and aligned.shape[0] and aligned.shape[1] >= 10:
            a9 = np.asarray(aligned[:, 9], dtype=np.float64)
            model_a9_min, model_a9_max = float(np.nanmin(a9)), float(np.nanmax(a9))
    return {
        "sim_gripper_transition_count": len(transitions),
        "sim_gripper_transition_frames": transitions,
        "sim_gripper_command_min": command_min,
        "sim_gripper_command_max": command_max,
        "executed_model_action9_min": model_a9_min,
        "executed_model_action9_max": model_a9_max,
    }


def initial_evidence_label(task: str, row: dict[str, Any]) -> tuple[str, str]:
    if row["success"]:
        return "SUCCESS", "measured"
    if row["error"]:
        return "EVAL_OR_ENVIRONMENT_ERROR", "measured"
    if row["controller_errors"]:
        return "CONTROLLER_INVOLVEMENT_REVIEW_REQUIRED", "measured_association_only"
    transitions = row["sim_gripper_transition_count"]
    if task in GRIPPER_REQUIRED_TASKS and transitions == 0:
        return "GRIPPER_EVENT_MISSING_CANDIDATE", "strong_measured_evidence"
    if transitions >= 4:
        return "GRIPPER_EVENT_CHATTER_CANDIDATE", "strong_measured_evidence"
    return "VISUAL_STAGE_REVIEW_REQUIRED", "unknown"


def episode_rows(label: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, (task_dir, summary) in task_summaries(root).items():
        for result in summary.get("results", []):
            evidence = gripper_evidence(task_dir, result)
            row = {
                "evaluation": label,
                "task": task,
                "episode": as_int(result.get("episode_index")),
                "seed_episode_index": as_int(result.get("seed_episode_index")),
                "success": bool(result.get("success")),
                "end_reason": str(result.get("end_reason", "")),
                "error": str(result.get("error") or ""),
                "model_calls": as_int(result.get("model_calls")),
                "environment_actions": as_int(result.get("environment_actions")),
                "physics_frames": as_int(result.get("physics_frames")),
                "video_frames": as_int(result.get("video_frames")),
                "summary_gripper_transitions": as_int(result.get("gripper_transitions")),
                "controller_errors": as_int(result.get("controller_continue_errors")),
                "control_error_reinferences": as_int(result.get("control_error_reinferences")),
                "mover_targets": as_int(result.get("mover_targets")),
                "mover_reached_targets": as_int(result.get("mover_reached_targets")),
                "mover_unreached_targets": as_int(result.get("mover_unreached_targets")),
                "mover_retries": as_int(result.get("mover_retries")),
                "workspace_clipped_actions": as_int(result.get("workspace_clipped_actions")),
                "discarded_chunk_rows": as_int(result.get("discarded_chunk_rows")),
                "task_dir": str(task_dir.resolve()),
                "video": str((relative_file(task_dir, result.get("video")) or Path()).resolve()),
                "executed_actions": str(
                    (relative_file(task_dir, result.get("executed_actions")) or Path()).resolve()
                ),
                "executed_model_actions_relative10": str(
                    (
                        relative_file(task_dir, result.get("executed_model_actions_relative10"))
                        or Path()
                    ).resolve()
                ),
                **evidence,
            }
            label_value, confidence = initial_evidence_label(task, row)
            row["automatic_evidence_label"] = label_value
            row["automatic_label_confidence"] = confidence
            rows.append(row)
    return rows


def wilson(successes: int, episodes: int) -> tuple[float, float]:
    if episodes == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return center - margin, center + margin


def task_rows(rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    tasks = sorted({row["task"] for row in rows})
    for row in rows:
        grouped[(row["evaluation"], row["task"])].append(row)
    output = []
    for task in tasks:
        per_label = {}
        for label in labels:
            items = grouped.get((label, task), [])
            successes = sum(item["success"] for item in items)
            low, high = wilson(successes, len(items))
            failures = [item for item in items if not item["success"]]
            per_label[label] = {
                "episodes": len(items),
                "successes": successes,
                "success_rate": successes / len(items) if items else None,
                "wilson95_low": low,
                "wilson95_high": high,
                "failure_controller_errors": sum(item["controller_errors"] for item in failures),
                "failure_missing_gripper_event": sum(
                    item["sim_gripper_transition_count"] == 0 for item in failures
                ),
                "failure_gripper_chatter": sum(
                    item["sim_gripper_transition_count"] >= 4 for item in failures
                ),
            }
        a, b = per_label[labels[0]], per_label[labels[1]]
        output.append(
            {
                "task": task,
                **{f"{labels[0]}_{key}": value for key, value in a.items()},
                **{f"{labels[1]}_{key}": value for key, value in b.items()},
                "success_rate_delta_b_minus_a": (
                    None
                    if a["success_rate"] is None or b["success_rate"] is None
                    else b["success_rate"] - a["success_rate"]
                ),
            }
        )
    return output


def paired_rows(rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    lookup = {(r["evaluation"], r["task"], r["seed_episode_index"]): r for r in rows}
    output = []
    keys = sorted({(r["task"], r["seed_episode_index"]) for r in rows})
    for task, seed_episode in keys:
        a = lookup.get((labels[0], task, seed_episode))
        b = lookup.get((labels[1], task, seed_episode))
        if not a or not b:
            continue
        if a["success"] and b["success"]:
            outcome = "both_success"
        elif a["success"]:
            outcome = f"{labels[0]}_only"
        elif b["success"]:
            outcome = f"{labels[1]}_only"
        else:
            outcome = "both_fail"
        output.append(
            {
                "task": task,
                "seed_episode_index": seed_episode,
                "paired_outcome": outcome,
                f"{labels[0]}_success": a["success"],
                f"{labels[1]}_success": b["success"],
                f"{labels[0]}_model_calls": a["model_calls"],
                f"{labels[1]}_model_calls": b["model_calls"],
                f"{labels[0]}_controller_errors": a["controller_errors"],
                f"{labels[1]}_controller_errors": b["controller_errors"],
                f"{labels[0]}_gripper_transitions": a["sim_gripper_transition_count"],
                f"{labels[1]}_gripper_transitions": b["sim_gripper_transition_count"],
            }
        )
    return output


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired binary outcomes."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) * (0.5**discordant)
        for value in range(min(a_only, b_only) + 1)
    )
    return min(1.0, 2.0 * tail)


def paired_task_rows(pairs: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        grouped[pair["task"]][pair["paired_outcome"]] += 1
    output = []
    for task, counts in sorted(grouped.items()):
        a_only = counts[f"{labels[0]}_only"]
        b_only = counts[f"{labels[1]}_only"]
        output.append(
            {
                "task": task,
                "both_success": counts["both_success"],
                f"{labels[0]}_only": a_only,
                f"{labels[1]}_only": b_only,
                "both_fail": counts["both_fail"],
                "discordant_pairs": a_only + b_only,
                "exact_mcnemar_p": exact_mcnemar_p(a_only, b_only),
            }
        )
    return output


def phone_chunk_event_rows(rows: list[dict[str, Any]], threshold: float = 0.003) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["task"] != "phone_on_base":
            continue
        path = (
            Path(row["task_dir"])
            / "model_chunks"
            / f"episode_{row['episode']:03d}_model_chunks.npy"
        )
        if not path.is_file():
            continue
        chunks = np.load(path, mmap_mode="r")
        if chunks.ndim != 3 or chunks.shape[1] < 32 or chunks.shape[2] < 10:
            continue
        widths = np.asarray(chunks[:, :, 9], dtype=np.float64)
        first = np.diff(widths[:, :16], axis=1)
        hidden = np.diff(widths[:, 16:32], axis=1)
        boundary = widths[1:, 0] - widths[:-1, 15]
        output.append(
            {
                "evaluation": row["evaluation"],
                "episode": row["episode"],
                "success": row["success"],
                "executed_window_open_candidates": int(np.count_nonzero(first > threshold)),
                "executed_window_close_candidates": int(np.count_nonzero(first < -threshold)),
                "unexecuted_rows16to31_open_candidates": int(np.count_nonzero(hidden > threshold)),
                "unexecuted_rows16to31_close_candidates": int(np.count_nonzero(hidden < -threshold)),
                "ignored_cross_chunk_open_jumps": int(np.count_nonzero(boundary > threshold)),
                "ignored_cross_chunk_close_jumps": int(np.count_nonzero(boundary < -threshold)),
                "sim_gripper_transition_count": row["sim_gripper_transition_count"],
            }
        )
    return output


def initial_frame_pair_rows(rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    lookup = {(r["evaluation"], r["task"], r["seed_episode_index"]): r for r in rows}
    output = []
    for task, seed_episode in sorted({(r["task"], r["seed_episode_index"]) for r in rows}):
        a = lookup.get((labels[0], task, seed_episode))
        b = lookup.get((labels[1], task, seed_episode))
        if not a or not b:
            continue
        frames = []
        for row in (a, b):
            capture = cv2.VideoCapture(row["video"])
            ok, frame = capture.read()
            capture.release()
            frames.append(frame if ok else None)
        difference = None
        if frames[0] is not None and frames[1] is not None and frames[0].shape == frames[1].shape:
            # Ignore the metadata overlay band and compare the rendered scene.
            difference = float(
                np.abs(frames[0][80:].astype(np.int16) - frames[1][80:].astype(np.int16)).mean()
            )
        output.append(
            {
                "task": task,
                "seed_episode_index": seed_episode,
                "mean_absolute_pixel_difference_below_overlay": difference,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_sheet_frames(frame_count: int, transition_frames: list[int], count: int) -> list[int]:
    candidates = {0, max(0, frame_count - 1)}
    if frame_count > 1:
        candidates.update(np.linspace(0, frame_count - 1, count, dtype=int).tolist())
    for transition in transition_frames:
        candidates.update((transition - 8, transition - 1, transition, transition + 8))
    candidates = sorted(max(0, min(frame_count - 1, value)) for value in candidates)
    candidates = sorted(set(candidates))
    if len(candidates) <= count:
        return candidates
    # Retain every transition frame, plus an evenly spread trajectory story.
    required = {max(0, min(frame_count - 1, value)) for value in transition_frames}
    required.update((0, frame_count - 1))
    optional = [value for value in candidates if value not in required]
    slots = max(0, count - len(required))
    if slots and optional:
        indexes = np.linspace(0, len(optional) - 1, slots, dtype=int)
        required.update(optional[index] for index in indexes)
    return sorted(required)[:count]


def make_contact_sheet(row: dict[str, Any], destination: Path, count: int) -> None:
    video = Path(row["video"])
    if not video.is_file():
        return
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return
    frame_indexes = select_sheet_frames(
        frame_count, list(row["sim_gripper_transition_frames"]), count
    )
    cells = []
    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        target_width = 320
        frame = cv2.resize(frame, (target_width, round(height * target_width / width)))
        marker = " G" if frame_index in row["sim_gripper_transition_frames"] else ""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 27), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"frame={frame_index}{marker}",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255) if marker else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(frame)
    capture.release()
    if not cells:
        return
    columns = 4
    rows = math.ceil(len(cells) / columns)
    cell_h, cell_w = cells[0].shape[:2]
    canvas = np.zeros((rows * cell_h + 38, columns * cell_w, 3), dtype=np.uint8)
    title = (
        f"{row['evaluation']} {row['task']} ep={row['episode']:03d} "
        f"success={int(row['success'])} transitions={row['sim_gripper_transition_count']}"
    )
    cv2.putText(canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    for index, cell in enumerate(cells):
        y = 38 + (index // columns) * cell_h
        x = (index % columns) * cell_w
        canvas[y : y + cell_h, x : x + cell_w] = cell
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def make_task_montages(output: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["evaluation"], row["task"])].append(row)
    for (evaluation, task), items in grouped.items():
        images = []
        for row in sorted(items, key=lambda item: item["episode"]):
            path = (
                output
                / "contact_sheets"
                / evaluation
                / task
                / f"episode_{row['episode']:03d}.jpg"
            )
            image = cv2.imread(str(path))
            if image is None:
                continue
            target_width = 600
            image = cv2.resize(
                image,
                (target_width, round(image.shape[0] * target_width / image.shape[1])),
            )
            images.append(image)
        if not images:
            continue
        columns = 2
        montage_rows = math.ceil(len(images) / columns)
        cell_h, cell_w = images[0].shape[:2]
        canvas = np.zeros((montage_rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
        for index, image in enumerate(images):
            y = (index // columns) * cell_h
            x = (index % columns) * cell_w
            canvas[y : y + cell_h, x : x + cell_w] = image
        destination = output / "task_montages" / evaluation / f"{task}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), canvas)


def markdown_report(
    path: Path,
    roots: list[Path],
    labels: list[str],
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    lines = [
        "# RLBench paired evaluation evidence report",
        "",
        "This file contains measured evidence only. Earliest-fatal-stage and root-cause labels "
        "must be completed after reviewing the generated contact sheets/videos; controller errors "
        "and gripper transitions are associations, not automatic causal proof.",
        "",
        f"- `{labels[0]}`: `{roots[0]}`",
        f"- `{labels[1]}`: `{roots[1]}`",
        "",
        "## Task results",
        "",
        f"| Task | {labels[0]} | {labels[1]} | Delta | Paired both/A/B/neither |",
        "|---|---:|---:|---:|---:|",
    ]
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        pair_counts[pair["task"]][pair["paired_outcome"]] += 1
    for item in summaries:
        task = item["task"]
        a_n = item[f"{labels[0]}_episodes"]
        b_n = item[f"{labels[1]}_episodes"]
        a_s = item[f"{labels[0]}_successes"]
        b_s = item[f"{labels[1]}_successes"]
        delta = item["success_rate_delta_b_minus_a"]
        counts = pair_counts[task]
        paired = "/".join(
            str(counts[key])
            for key in ("both_success", f"{labels[0]}_only", f"{labels[1]}_only", "both_fail")
        )
        delta_text = "n/a" if delta is None else f"{delta * 100:+.0f} pp"
        lines.append(f"| {task} | {a_s}/{a_n} | {b_s}/{b_n} | {delta_text} | {paired} |")
    lines.extend(
        [
            "",
            "## Automatic evidence flags on failed episodes",
            "",
            "These flags are triage hints. They do not replace stage review.",
            "",
            "| Evaluation | Failed | Missing gripper event | 4+ transitions | Controller errors present |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        failed = [row for row in rows if row["evaluation"] == label and not row["success"]]
        lines.append(
            f"| {label} | {len(failed)} | "
            f"{sum(row['sim_gripper_transition_count'] == 0 for row in failed)} | "
            f"{sum(row['sim_gripper_transition_count'] >= 4 for row in failed)} | "
            f"{sum(row['controller_errors'] > 0 for row in failed)} |"
        )
    lines.extend(
        [
            "",
            "## Review workflow",
            "",
            "1. Mark the last passed task stage and the earliest failed necessary predicate.",
            "2. At that frame, trace raw model action -> delta decoder -> simulator command -> actual state.",
            "3. Assign one primary layer only: data, model, input/perception, decoder, controller/planner, physics/contact, success condition, budget, or environment.",
            "4. Record downstream symptoms separately and state confidence/counterfactual evidence.",
            "",
            "See `episode_metrics.csv`, `paired_outcomes.csv`, and `task_comparison.csv` for the complete measured tables.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = [args.eval_a.expanduser().resolve(), args.eval_b.expanduser().resolve()]
    labels = [args.label_a, args.label_b]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = episode_rows(labels[0], roots[0]) + episode_rows(labels[1], roots[1])
    summaries = task_rows(rows, labels)
    pairs = paired_rows(rows, labels)
    paired_tasks = paired_task_rows(pairs, labels)
    phone_events = phone_chunk_event_rows(rows)
    initial_pairs = initial_frame_pair_rows(rows, labels)
    write_csv(output / "episode_metrics.csv", rows)
    write_csv(output / "task_comparison.csv", summaries)
    write_csv(output / "paired_outcomes.csv", pairs)
    write_csv(output / "paired_task_statistics.csv", paired_tasks)
    write_csv(output / "phone_chunk_gripper_events.csv", phone_events)
    write_csv(output / "initial_frame_pair_differences.csv", initial_pairs)
    (output / "measured_evidence.json").write_text(
        json.dumps(
            {"evaluations": dict(zip(labels, map(str, roots))), "episodes": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_report(output / "MEASURED_EVIDENCE.md", roots, labels, rows, summaries, pairs)
    if args.contact_sheets != "none":
        for row in rows:
            if args.contact_sheets == "failures" and row["success"]:
                continue
            destination = (
                output
                / "contact_sheets"
                / row["evaluation"]
                / row["task"]
                / f"episode_{row['episode']:03d}.jpg"
            )
            make_contact_sheet(row, destination, args.sheet_frames)
        make_task_montages(output, rows)
    print(f"episodes={len(rows)} tasks={len(summaries)} paired={len(pairs)} output={output}")


if __name__ == "__main__":
    main()
