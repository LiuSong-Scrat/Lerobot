#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare checkpoint predictions with real training chunks.

This is a diagnostic, not an RLBench success-rate test.  It feeds observations
from the training dataset through the saved checkpoint and reports geometric
action error against the corresponding ground-truth chunk.  Ground-truth
actions are never sent to the online evaluation environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


from _rlbench_tool_paths import LEROBOT_ROOT as REPO_ROOT, SONG_SCRIPTS_DIR


SONG_SCRIPTS = SONG_SCRIPTS_DIR
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SONG_SCRIPTS))

from smolvla_model_inference import SmolVLA_ModelInference, rot6d_to_matrix


class ImageKeyAliasDataset(torch.utils.data.Dataset):
    """Expose one existing RGB stream under a legacy checkpoint's key."""

    def __init__(self, dataset, source_key, target_key):
        self.dataset = dataset
        self.source_key = str(source_key)
        self.target_key = str(target_key)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item[self.target_key] = item[self.source_key]
        return item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=(
            "Inference-only flow-matching integration steps. The default keeps "
            "the value serialized in the checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-image-key-alias",
        action="store_true",
        help=(
            "Allow a single dataset RGB feature (for example front) to be exposed "
            "under a legacy checkpoint's single RGB key (for example agentview)."
        ),
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=12,
        help="Number of evenly spaced training episodes inspected for each task.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help=(
            "Optional comma-separated task names. The default evaluates every task "
            "present in the dataset."
        ),
    )
    parser.add_argument(
        "--frame-fractions",
        default="0.05,0.25,0.45,0.65,0.80",
        help="Comma-separated non-tail positions sampled from every selected episode.",
    )
    return parser.parse_args()


def rotation_errors_deg(predicted, target):
    pred_matrix = rot6d_to_matrix(torch.from_numpy(predicted[:, 3:9])).numpy()
    target_matrix = rot6d_to_matrix(torch.from_numpy(target[:, 3:9])).numpy()
    delta = np.swapaxes(target_matrix, 1, 2) @ pred_matrix
    return np.degrees(Rotation.from_matrix(delta).magnitude())


def metric_record(predicted, target, count):
    predicted = predicted[:count]
    target = target[:count]
    position_mm = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1) * 1000.0
    rotation_deg = rotation_errors_deg(predicted, target)
    gripper_error_mm = np.abs(predicted[:, 9] - target[:, 9]) * 1000.0
    predicted_open = predicted[:, 9] > 0.04
    target_open = target[:, 9] > 0.04
    return {
        "rows": int(len(predicted)),
        "position_error_mm_mean": float(np.mean(position_mm)),
        "position_error_mm_rmse": float(np.sqrt(np.mean(np.square(position_mm)))),
        "position_error_mm_median": float(np.median(position_mm)),
        "position_error_mm_p90": float(np.percentile(position_mm, 90)),
        "position_error_mm_p95": float(np.percentile(position_mm, 95)),
        "position_error_mm_max": float(np.max(position_mm)),
        "rotation_error_deg_mean": float(np.mean(rotation_deg)),
        "rotation_error_deg_rmse": float(np.sqrt(np.mean(np.square(rotation_deg)))),
        "rotation_error_deg_median": float(np.median(rotation_deg)),
        "rotation_error_deg_p90": float(np.percentile(rotation_deg, 90)),
        "rotation_error_deg_p95": float(np.percentile(rotation_deg, 95)),
        "rotation_error_deg_max": float(np.max(rotation_deg)),
        "gripper_error_mm_mean": float(np.mean(gripper_error_mm)),
        "gripper_error_mm_rmse": float(np.sqrt(np.mean(np.square(gripper_error_mm)))),
        "gripper_error_mm_median": float(np.median(gripper_error_mm)),
        "gripper_error_mm_p95": float(np.percentile(gripper_error_mm, 95)),
        "gripper_error_mm_max": float(np.max(gripper_error_mm)),
        "gripper_binary_accuracy": float(np.mean(predicted_open == target_open)),
        "gripper_binary_mismatches": int(np.count_nonzero(predicted_open != target_open)),
    }


def aggregate_metric_rows(predicted_rows, target_rows):
    """Compute distribution statistics from every action row, not averages of samples."""
    predicted = np.concatenate(predicted_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    return metric_record(predicted, target, len(predicted))


def main():
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive.")
    frame_fractions = [float(value) for value in args.frame_fractions.split(",") if value.strip()]
    if not frame_fractions or any(value < 0.0 or value > 1.0 for value in frame_fractions):
        raise ValueError("--frame-fractions must contain values between 0 and 1.")
    requested_tasks = {
        value.strip() for value in str(args.tasks).split(",") if value.strip()
    }
    os.environ.setdefault("HF_HOME", "/tmp/rlbench_offline_action_diagnostic_hf")

    inference = SmolVLA_ModelInference(
        policy_path=args.policy_path.expanduser().resolve(),
        device=args.device,
    )
    checkpoint_num_steps = int(inference.policy.config.num_steps)
    effective_num_steps = checkpoint_num_steps if args.num_steps is None else int(args.num_steps)
    if effective_num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    # The policy and its nested model normally share the same config object,
    # but assign both explicitly so the diagnostic remains correct if that
    # implementation detail changes.
    inference.policy.config.num_steps = effective_num_steps
    inference.policy.model.config.num_steps = effective_num_steps
    normalization_mapping = {
        str(key): str(value).split(".")[-1].upper()
        for key, value in inference.policy.config.normalization_mapping.items()
    }
    if normalization_mapping.get("ACTION") != "IDENTITY":
        raise ValueError(
            "Physical action-error metrics require ACTION identity normalization; got "
            + repr(normalization_mapping)
        )
    dataset = inference.load_dataset(args.dataset_root.expanduser().resolve())
    policy_image_keys = [
        key for key in inference.policy.config.input_features if key.startswith("observation.images.")
    ]
    dataset_image_keys = [
        key for key in dataset.meta.features if key.startswith("observation.images.")
    ]
    image_key_alias = None
    missing_policy_image_keys = [key for key in policy_image_keys if key not in dataset.meta.features]
    if missing_policy_image_keys:
        if not args.allow_image_key_alias:
            raise ValueError(
                "Checkpoint image keys are absent from the dataset: "
                + repr(missing_policy_image_keys)
                + "; dataset provides " + repr(dataset_image_keys)
            )
        if len(policy_image_keys) != 1 or len(dataset_image_keys) != 1:
            raise ValueError(
                "Automatic RGB aliasing requires exactly one checkpoint and one dataset image key; got "
                + repr(policy_image_keys) + " and " + repr(dataset_image_keys)
            )
        image_key_alias = {"source": dataset_image_keys[0], "target": policy_image_keys[0]}
        dataset = ImageKeyAliasDataset(dataset, dataset_image_keys[0], policy_image_keys[0])
        inference.dataset = dataset
    episodes = dataset.meta.episodes

    # Choose episodes evenly across each task rather than concentrating on the
    # first recorded demonstrations.  All sampled frames leave a full 32-row
    # future action chunk, so padded chunk tails are never evaluated.
    episode_indices_by_task = {}
    for episode_index, episode in enumerate(episodes):
        if int(episode["length"]) >= 33:
            task = str(episode["tasks"][0])
            if requested_tasks and task not in requested_tasks:
                continue
            episode_indices_by_task.setdefault(task, []).append(episode_index)

    missing_tasks = sorted(requested_tasks - set(episode_indices_by_task))
    if missing_tasks:
        raise ValueError(
            "Requested tasks are absent from the dataset or have no episode with a full "
            f"32-step target chunk: {missing_tasks}"
        )

    selected_episode_indices = []
    for task in sorted(episode_indices_by_task):
        candidates = episode_indices_by_task[task]
        selected_count = min(int(args.episodes_per_task), len(candidates))
        positions = np.linspace(0, len(candidates) - 1, selected_count, dtype=np.int64)
        selected_episode_indices.extend(candidates[int(position)] for position in positions)

    sample_results = []
    # Keep the actual action rows only in memory.  This lets the final P90/P95
    # describe all predictions, rather than an average of per-frame P95 values.
    rows_by_task = {}
    rows_by_step = {}

    try:
        sample_number = 0
        for episode_index in selected_episode_indices:
            episode = episodes[episode_index]
            start = int(episode["dataset_from_index"])
            length = int(episode["length"])
            task = str(episode["tasks"][0])
            for fraction in frame_fractions:
                frame_index = int(round((length - 33) * fraction))
                dataset_index = start + frame_index
                torch.manual_seed(args.seed + sample_number)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed + sample_number)
                result = inference.predict_from_dataset(dataset_index)
                predicted = np.asarray(result["action_chunk"][0], dtype=np.float32)
                # This target is the real dataset label after the same rigid
                # world->current-EEF UMI transform used during training.  A
                # rigid change of coordinates preserves translation distance
                # and rotation angle.  ACTION normalization is asserted to be
                # identity above, so the metrics remain physical mm/degrees.
                target = np.asarray(result["gt_action_chunk"][0], dtype=np.float32)
                task_rows = rows_by_task.setdefault(
                    task,
                    {
                        name: {"predicted": [], "target": []}
                        for name in ["first_1", "first_4", "first_8", "first_16", "full_32"]
                    },
                )
                for name, count in [
                    ("first_1", 1),
                    ("first_4", 4),
                    ("first_8", 8),
                    ("first_16", 16),
                    ("full_32", 32),
                ]:
                    task_rows[name]["predicted"].append(predicted[:count])
                    task_rows[name]["target"].append(target[:count])
                for step in range(32):
                    step_rows = rows_by_step.setdefault(task, {}).setdefault(
                        str(step), {"predicted": [], "target": []}
                    )
                    step_rows["predicted"].append(predicted[step : step + 1])
                    step_rows["target"].append(target[step : step + 1])
                sample_results.append(
                    {
                        "episode_index": int(episode_index),
                        "frame_index": int(frame_index),
                        "dataset_index": int(dataset_index),
                        "task": task,
                        "first_1": metric_record(predicted, target, 1),
                        "first_4": metric_record(predicted, target, 4),
                        "first_8": metric_record(predicted, target, 8),
                        "first_16": metric_record(predicted, target, 16),
                        "full_32": metric_record(predicted, target, 32),
                    }
                )
                print(
                    "[sample] task=" + task
                    + " episode=" + str(episode_index)
                    + " frame=" + str(frame_index)
                    + " first4_position_mm="
                    + format(sample_results[-1]["first_4"]["position_error_mm_mean"], ".3f"),
                    flush=True,
                )
                sample_number += 1
    finally:
        inference.close()

    summary = {
        "config": {
            "policy_path": str(args.policy_path.expanduser().resolve()),
            "dataset_root": str(args.dataset_root.expanduser().resolve()),
            "seed": int(args.seed),
            "checkpoint_num_steps": checkpoint_num_steps,
            "effective_num_steps": effective_num_steps,
            "episodes_per_task": int(args.episodes_per_task),
            "requested_tasks": sorted(requested_tasks),
            "frame_fractions": frame_fractions,
            "selected_episode_indices": selected_episode_indices,
            "target_coordinate_space": "current_eef_after_training_umi_transform",
            "action_normalization": normalization_mapping,
            "image_key_alias": image_key_alias,
            "physical_metric_invariance": (
                "rigid world-to-EEF transform preserves translation and rotation errors"
            ),
        },
        "samples": sample_results,
        "trajectory_aggregates": [],
        "aggregates": {},
    }
    for episode_index in selected_episode_indices:
        trajectory_items = [item for item in sample_results if item["episode_index"] == episode_index]
        if not trajectory_items:
            continue
        summary["trajectory_aggregates"].append(
            {
                "episode_index": int(episode_index),
                "task": trajectory_items[0]["task"],
                "sampled_frames": [item["frame_index"] for item in trajectory_items],
                "first_1_position_error_mm_mean": float(
                    np.mean([item["first_1"]["position_error_mm_mean"] for item in trajectory_items])
                ),
                "first_4_position_error_mm_mean": float(
                    np.mean([item["first_4"]["position_error_mm_mean"] for item in trajectory_items])
                ),
                "first_8_position_error_mm_mean": float(
                    np.mean([item["first_8"]["position_error_mm_mean"] for item in trajectory_items])
                ),
                "first_16_position_error_mm_mean": float(
                    np.mean([item["first_16"]["position_error_mm_mean"] for item in trajectory_items])
                ),
                "full_32_position_error_mm_mean": float(
                    np.mean([item["full_32"]["position_error_mm_mean"] for item in trajectory_items])
                ),
            }
        )
    for task in sorted(rows_by_task):
        summary["aggregates"][task] = {}
        for horizon in ["first_1", "first_4", "first_8", "first_16", "full_32"]:
            rows = rows_by_task[task][horizon]
            summary["aggregates"][task][horizon] = aggregate_metric_rows(
                rows["predicted"], rows["target"]
            )
        summary["aggregates"][task]["per_chunk_step"] = []
        for step in range(32):
            rows = rows_by_step[task][str(step)]
            step_metric = aggregate_metric_rows(rows["predicted"], rows["target"])
            step_metric["chunk_step"] = step
            summary["aggregates"][task]["per_chunk_step"].append(step_metric)

    summary["overall"] = {}
    for horizon in ["first_1", "first_4", "first_8", "first_16", "full_32"]:
        predicted = []
        target = []
        for task_rows in rows_by_task.values():
            predicted.extend(task_rows[horizon]["predicted"])
            target.extend(task_rows[horizon]["target"])
        summary["overall"][horizon] = aggregate_metric_rows(predicted, target)

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.output.expanduser().resolve(), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("[done] " + str(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
