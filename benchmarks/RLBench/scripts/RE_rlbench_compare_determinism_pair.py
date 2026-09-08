#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def vector(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.size else None


def position_error(pose_a, pose_b):
    a, b = vector(pose_a), vector(pose_b)
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return None
    return float(np.linalg.norm(a[:3] - b[:3]))


def rotation_error_degrees(pose_a, pose_b):
    a, b = vector(pose_a), vector(pose_b)
    if a is None or b is None or len(a) < 7 or len(b) < 7:
        return None
    qa, qb = a[3:7], b[3:7]
    qa = qa / max(float(np.linalg.norm(qa)), 1e-15)
    qb = qb / max(float(np.linalg.norm(qb)), 1e-15)
    dot = min(max(abs(float(np.dot(qa, qb))), -1.0), 1.0)
    return float(math.degrees(2.0 * math.acos(dot)))


def first_index(records_a, records_b, predicate):
    for index, (a, b) in enumerate(zip(records_a, records_b)):
        if predicate(a, b):
            return index
    if len(records_a) != len(records_b):
        return min(len(records_a), len(records_b))
    return None


def state_metrics(a, b):
    return {
        "eef_position_m": position_error(a.get("eef_pose7_world"), b.get("eef_pose7_world")),
        "eef_rotation_deg": rotation_error_degrees(a.get("eef_pose7_world"), b.get("eef_pose7_world")),
        "gripper_width_m": abs(float(a.get("gripper_width_m", 0.0)) - float(b.get("gripper_width_m", 0.0))),
        "waterer_position_m": position_error(a.get("waterer_pose7_world"), b.get("waterer_pose7_world")),
        "waterer_rotation_deg": rotation_error_degrees(a.get("waterer_pose7_world"), b.get("waterer_pose7_world")),
        "head_position_m": position_error(a.get("waterer_head_pose7_world"), b.get("waterer_head_pose7_world")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = [load_json(root / "manifest.json") for root in (args.run_a, args.run_b)]
    records = [load_jsonl(root / "executed_states.jsonl") for root in (args.run_a, args.run_b)]
    manifest_a, manifest_b = manifests
    records_a, records_b = records

    initial_keys = [
        "reset_front_rgb_raw_sha256",
        "model_front_rgb_sha256",
        "model_point_cloud_sha256",
        "model_state_sha256",
    ]
    initial_equal = {key: manifest_a.get(key) == manifest_b.get(key) for key in initial_keys}
    first_action_equal = (
        manifest_a.get("first_action_chunk_sha256")
        == manifest_b.get("first_action_chunk_sha256")
    )
    first_action_mismatch_step = first_index(
        records_a,
        records_b,
        lambda a, b: a.get("simulator_action_sha256") != b.get("simulator_action_sha256"),
    )
    first_exact_state_mismatch_step = first_index(
        records_a,
        records_b,
        lambda a, b: a.get("state_sha256") != b.get("state_sha256"),
    )

    meaningful_thresholds = {
        "eef_position_m": 1e-5,
        "eef_rotation_deg": 1e-3,
        "gripper_width_m": 1e-5,
        "waterer_position_m": 1e-5,
        "waterer_rotation_deg": 1e-3,
        "head_position_m": 1e-5,
    }

    def meaningful(a, b):
        metrics = state_metrics(a, b)
        return any(
            metrics[key] is not None and metrics[key] > threshold
            for key, threshold in meaningful_thresholds.items()
        )

    first_meaningful_state_mismatch_step = first_index(records_a, records_b, meaningful)
    first_metrics = None
    if (
        first_meaningful_state_mismatch_step is not None
        and first_meaningful_state_mismatch_step < min(len(records_a), len(records_b))
    ):
        first_metrics = state_metrics(
            records_a[first_meaningful_state_mismatch_step],
            records_b[first_meaningful_state_mismatch_step],
        )

    if not all(initial_equal.values()):
        classification = "initial_observation_or_point_sampling_diverged"
    elif not first_action_equal:
        classification = "model_inference_diverged_before_control"
    elif first_exact_state_mismatch_step is not None:
        classification = "planner_controller_or_simulator_diverged_after_identical_first_action"
    else:
        classification = "no_divergence_observed_in_recorded_prefix"

    report = {
        "classification": classification,
        "run_a": str(args.run_a.resolve()),
        "run_b": str(args.run_b.resolve()),
        "outcomes": {
            "run_a": {
                key: manifest_a.get(key)
                for key in ("success", "end_reason", "model_calls", "simulator_state_records")
            },
            "run_b": {
                key: manifest_b.get(key)
                for key in ("success", "end_reason", "model_calls", "simulator_state_records")
            },
        },
        "initial_hash_equal": initial_equal,
        "first_action_chunk_hash_equal": first_action_equal,
        "first_action_mismatch_simulator_step": first_action_mismatch_step,
        "first_exact_state_mismatch_simulator_step": first_exact_state_mismatch_step,
        "first_meaningful_state_mismatch_simulator_step": first_meaningful_state_mismatch_step,
        "meaningful_thresholds": meaningful_thresholds,
        "first_meaningful_state_error": first_metrics,
        "record_counts": {"run_a": len(records_a), "run_b": len(records_b)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    markdown_path = args.output.with_suffix(".md")
    lines = [
        "# RLBench water_plants paired determinism report",
        "",
        f"- Classification: `{classification}`",
        f"- Initial RGB equal: `{initial_equal['model_front_rgb_sha256']}`",
        f"- Initial point cloud equal: `{initial_equal['model_point_cloud_sha256']}`",
        f"- First action chunk equal: `{first_action_equal}`",
        f"- First action mismatch simulator step: `{first_action_mismatch_step}`",
        f"- First exact state mismatch simulator step: `{first_exact_state_mismatch_step}`",
        f"- First meaningful state mismatch simulator step: `{first_meaningful_state_mismatch_step}`",
        f"- Run A outcome: `{report['outcomes']['run_a']}`",
        f"- Run B outcome: `{report['outcomes']['run_b']}`",
        "",
        "## First meaningful state error",
        "",
        "```json",
        json.dumps(first_metrics, indent=2, ensure_ascii=False),
        "```",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
