#!/usr/bin/env python3
"""Diagnose where an RLBench training episode diverges under policy rollout.

The diagnostic deliberately separates three effects at every model refresh:

1. teacher-forced prediction from the exact stored LeRobot sample;
2. prediction from the freshly rendered simulator observation;
3. controller tracking after executing the live prediction.

All comparisons use the checkpoint's saved UMI pre/postprocessors.  Dataset
``action`` is used only as a reference label and is never executed by the model
rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pyarrow.dataset as pyarrow_dataset
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch.utils.data._utils.collate import default_collate


from _rlbench_tool_paths import LEROBOT_ROOT, RLBENCH_ROOT, SCRIPTS_DIR


SCRIPT_DIR = SCRIPTS_DIR
SONG_SCRIPT_DIR = LEROBOT_ROOT / "benchmarks" / "song_real_libero" / "scripts"
for path in (str(SCRIPT_DIR), str(SONG_SCRIPT_DIR), str(LEROBOT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import RE_rlbench_dataset_action_replay as replay
import RE_rlbench_official_eval as online
from rlbench_video_utils import annotate_final_task_result_frames
from RE_rlbench_gripper_control import (
    ABSOLUTE_WIDTH,
    DELTA_WIDTH_INITIAL_SYNC,
    GRIPPER_CONTROL_MODES,
    absolute_width_gripper_target,
    initial_delta_reference_for_chunk,
    libero_style_gripper_target,
    set_gripper_absolute_width_position_target,
)
from smolvla_model_inference import SmolVLA_ModelInference


DEFAULT_DATASET = (
    RLBENCH_ROOT
    / "datasets/2tasks/rlbench_2tasks_phone_on_base_water_plants_100traj_each_reap_v4_20k_20260823"
)
DEFAULT_POLICY = (
    RLBENCH_ROOT
    / "outputs/wep_vla_v043-20000+20000+2_gripper_2tasks_0823"
    / "checkpoints/020000/pretrained_model"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop model-vs-training-trajectory RLBench diagnostic."
    )
    parser.add_argument("--task", choices=["phone_on_base", "water_plants"], required=True)
    parser.add_argument("--episode", type=int, default=0, help="Task-local episode index.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exec-action-steps", type=int, default=16)
    parser.add_argument("--max-logical-steps", type=int, default=0, help="0 uses the complete episode.")
    parser.add_argument("--noise-seed", type=int, default=20260825)
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.04)
    parser.add_argument(
        "--gripper-mode",
        choices=GRIPPER_CONTROL_MODES,
        default=DELTA_WIDTH_INITIAL_SYNC,
    )
    parser.add_argument("--gripper-delta-threshold", type=float, default=0.002)
    parser.add_argument(
        "--gripper-delta-alignment",
        choices=["current_minus_previous", "next_minus_current"],
        default="current_minus_previous",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def image_uint8(value):
    image = to_numpy(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0 + 1e-6:
        image = image * 255.0
    return np.clip(np.nan_to_num(image), 0.0, 255.0).astype(np.uint8)[..., :3]


def unwrap_point_cloud(value):
    cloud = to_numpy(value)
    while cloud.ndim > 2 and cloud.shape[0] == 1:
        cloud = cloud[0]
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Expected point cloud (N,C>=3), got {cloud.shape}.")
    return np.asarray(cloud, dtype=np.float32)


def pose9_error(left, right):
    left_matrix = online.pose9_to_homo_np(np.asarray(left, dtype=np.float64)[:9])
    right_matrix = online.pose9_to_homo_np(np.asarray(right, dtype=np.float64)[:9])
    position = float(np.linalg.norm(left_matrix[:3, 3] - right_matrix[:3, 3]))
    rotation = float(
        Rotation.from_matrix(left_matrix[:3, :3].T @ right_matrix[:3, :3]).magnitude()
    )
    return position, rotation


def pose7_error(left, right):
    left_matrix = replay.pose7_to_matrix(left)
    right_matrix = replay.pose7_to_matrix(right)
    position = float(np.linalg.norm(left_matrix[:3, 3] - right_matrix[:3, 3]))
    rotation = float(
        Rotation.from_matrix(left_matrix[:3, :3].T @ right_matrix[:3, :3]).magnitude()
    )
    return position, rotation


def chunk_errors(predicted, reference, valid_rows):
    rows = min(int(valid_rows), len(predicted), len(reference))
    position = []
    rotation = []
    gripper_abs = []
    gripper_discrete = []
    for row in range(rows):
        pos, rot = pose9_error(predicted[row, :9], reference[row, :9])
        position.append(pos)
        rotation.append(rot)
        gripper_abs.append(abs(float(predicted[row, 9]) - float(reference[row, 9])))
        gripper_discrete.append(
            bool(float(predicted[row, 9]) > 0.04)
            != bool(float(reference[row, 9]) > 0.04)
        )
    return {
        "rows": rows,
        "position_error_m_mean": float(np.mean(position)) if position else None,
        "position_error_m_max": float(np.max(position)) if position else None,
        "rotation_error_rad_mean": float(np.mean(rotation)) if rotation else None,
        "rotation_error_rad_max": float(np.max(rotation)) if rotation else None,
        "gripper_width_abs_error_m_mean": float(np.mean(gripper_abs)) if gripper_abs else None,
        "gripper_width_abs_error_m_max": float(np.max(gripper_abs)) if gripper_abs else None,
        "gripper_discrete_mismatches": int(sum(gripper_discrete)),
    }


def model_batch_differences(left, right):
    """Compare common tensor inputs produced by the two preprocessing paths."""
    result = {}
    for key in sorted(set(left) & set(right)):
        if key == "action" or not torch.is_tensor(left[key]) or not torch.is_tensor(right[key]):
            continue
        first = left[key].detach().cpu()
        second = right[key].detach().cpu()
        entry = {
            "left_shape": list(first.shape),
            "right_shape": list(second.shape),
            "same_shape": bool(first.shape == second.shape),
        }
        if first.shape == second.shape:
            if first.dtype == torch.bool or second.dtype == torch.bool:
                entry["mismatch_count"] = int(torch.count_nonzero(first != second))
            else:
                delta = torch.abs(first.to(torch.float32) - second.to(torch.float32))
                entry["abs_mean"] = float(delta.mean())
                entry["abs_max"] = float(delta.max())
        result[key] = entry
    return result


def point_cloud_distance(live_cloud, recorded_cloud, max_points=20000, gripper_points=500):
    live = np.asarray(live_cloud, dtype=np.float32)
    recorded = np.asarray(recorded_cloud, dtype=np.float32)
    live = live[np.isfinite(live[:, :3]).all(axis=1)]
    recorded = recorded[np.isfinite(recorded[:, :3]).all(axis=1)]
    live = live[:max_points]
    recorded = recorded[:max_points]
    if not len(live) or not len(recorded):
        return {"live_points": len(live), "recorded_points": len(recorded)}
    recorded_tree = cKDTree(recorded[:, :3])
    live_tree = cKDTree(live[:, :3])
    live_to_recorded, live_nn = recorded_tree.query(live[:, :3], k=1, workers=-1)
    recorded_to_live, _ = live_tree.query(recorded[:, :3], k=1, workers=-1)
    result = {
        "live_points": int(len(live)),
        "recorded_points": int(len(recorded)),
        "geometry_live_to_recorded_mean_m": float(np.mean(live_to_recorded)),
        "geometry_live_to_recorded_p95_m": float(np.quantile(live_to_recorded, 0.95)),
        "geometry_recorded_to_live_mean_m": float(np.mean(recorded_to_live)),
        "geometry_symmetric_mean_m": float(
            0.5 * (np.mean(live_to_recorded) + np.mean(recorded_to_live))
        ),
    }
    if live.shape[1] >= 6 and recorded.shape[1] >= 6:
        color_delta = np.abs(live[:, 3:6] - recorded[live_nn, 3:6])
        result["nearest_color_abs_mean"] = float(np.mean(color_delta))
    if live.shape == recorded.shape:
        indexed_xyz = np.linalg.norm(live[:, :3] - recorded[:, :3], axis=1)
        scene_stop = max(len(live) - int(gripper_points), 0)
        result["indexed_geometry_mean_m"] = float(np.mean(indexed_xyz))
        result["indexed_geometry_p95_m"] = float(np.quantile(indexed_xyz, 0.95))
        if scene_stop:
            result["indexed_scene_geometry_mean_m"] = float(
                np.mean(indexed_xyz[:scene_stop])
            )
        if scene_stop < len(live):
            result["indexed_gripper_geometry_mean_m"] = float(
                np.mean(indexed_xyz[scene_stop:])
            )
        if live.shape[1] >= 6:
            result["indexed_color_abs_mean"] = float(
                np.mean(np.abs(live[:, 3:6] - recorded[:, 3:6]))
            )
    return result


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


def episode_dataset_indices(dataset_root, global_episode):
    parquet = pyarrow_dataset.dataset(str(dataset_root / "data"), format="parquet")
    table = parquet.to_table(
        filter=pyarrow_dataset.field("episode_index") == int(global_episode),
        columns=["frame_index", "index"],
    )
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
    return indices[np.argsort(frames)]


def dataset_action_tracking_summary(actions, states):
    position = []
    rotation = []
    gripper = []
    for frame in range(min(len(actions) - 1, len(states) - 1)):
        pos, rot = pose9_error(actions[frame, :9], states[frame + 1, :9])
        position.append(pos)
        rotation.append(rot)
        gripper.append(abs(float(actions[frame, 9]) - float(states[frame + 1, 9])))
    return {
        "pairs": len(position),
        "position_error_m_median": float(np.median(position)),
        "position_error_m_max": float(np.max(position)),
        "rotation_error_rad_median": float(np.median(rotation)),
        "rotation_error_rad_max": float(np.max(rotation)),
        "gripper_width_abs_error_m_median": float(np.median(gripper)),
        "gripper_width_abs_error_m_max": float(np.max(gripper)),
    }


def annotate(image, lines):
    frame = Image.fromarray(image_uint8(image))
    draw = ImageDraw.Draw(frame)
    text = "\n".join(lines)
    box = draw.multiline_textbbox((6, 6), text)
    draw.rectangle((2, 2, box[2] + 10, box[3] + 10), fill=(0, 0, 0))
    draw.multiline_text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(frame)


def first_threshold(rows, key, threshold):
    for row in rows:
        value = row.get(key)
        if value is not None and float(value) > float(threshold):
            return int(row["logical_step"])
    return None


def aggregate_call_errors(calls, key):
    metrics = [call[key] for call in calls if key in call and call[key]["rows"]]
    rows = sum(metric["rows"] for metric in metrics)
    if not rows:
        return {"rows": 0}
    return {
        "rows": int(rows),
        "position_error_m_mean": float(
            sum(metric["position_error_m_mean"] * metric["rows"] for metric in metrics) / rows
        ),
        "position_error_m_max": float(max(metric["position_error_m_max"] for metric in metrics)),
        "rotation_error_rad_mean": float(
            sum(metric["rotation_error_rad_mean"] * metric["rows"] for metric in metrics) / rows
        ),
        "rotation_error_rad_max": float(max(metric["rotation_error_rad_max"] for metric in metrics)),
        "gripper_width_abs_error_m_mean": float(
            sum(metric["gripper_width_abs_error_m_mean"] * metric["rows"] for metric in metrics)
            / rows
        ),
        "gripper_width_abs_error_m_max": float(
            max(metric["gripper_width_abs_error_m_max"] for metric in metrics)
        ),
        "gripper_discrete_mismatches": int(
            sum(metric["gripper_discrete_mismatches"] for metric in metrics)
        ),
    }


def main():
    args = parse_args()
    if (
        args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
        and args.gripper_delta_alignment != "current_minus_previous"
    ):
        raise ValueError(
            "delta_width_initial_sync requires current_minus_previous alignment."
        )
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.policy_path = args.policy_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RLBENCH_WATER_PLANT_COLLISION"] = "enabled"
    os.environ["RLBENCH_WATER_DROP_COLLISION"] = "original"

    actions, states, global_episode = replay.load_parquet_episode(
        args.dataset_root, args.task, args.episode
    )
    dataset_indices = episode_dataset_indices(args.dataset_root, global_episode)
    if len(dataset_indices) != len(actions):
        raise RuntimeError("Dataset index/action length mismatch.")
    reset_spec = online.load_dataset_reset_specs(
        args.dataset_root, args.task, str(args.episode), 1
    )[0]

    infer = SmolVLA_ModelInference(args.policy_path, device="cuda", visualize_foreground=False)
    dataset = infer.load_dataset(args.dataset_root)

    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
    from rlbench.action_modes.gripper_action_modes import Discrete

    environment = Environment(
        MoveArmThenGripper(
            EndEffectorPoseViaPlanning(
                absolute_mode=True,
                frame=RelativeFrame.WORLD,
                collision_checking=False,
            ),
            Discrete(),
        ),
        obs_config=online.make_observation_config(args.image_size),
        headless=True,
        static_positions=False,
    )

    rows = []
    calls = []
    video_frames = []
    success = False
    terminated = False
    controller_error = None
    initial_validation = None
    gripper_initial_sync_info = None
    language = args.task.replace("_", " ")
    try:
        environment.launch()
        task_env = environment.get_task(online.task_class_from_name(args.task))
        task_env.set_variation(0)
        descriptions, observation, reset_method, initial_validation = online.reset_task_for_episode(
            task_env,
            SimpleNamespace(seed=0),
            0,
            reset_spec,
        )
        language = descriptions[0] if descriptions else language
        episode_origin_world = online.pose9_to_homo_np(states[0, :9])
        recorded_first_world = online.pose9_to_homo_np(reset_spec["recorded_first_pose9"])
        # states are episode-origin relative; the recorded first world pose is
        # the actual world anchor used by conversion and replay.
        episode_origin_world = recorded_first_world @ np.linalg.inv(episode_origin_world)

        logical_step = 0
        gripper_command_open = float(observation.gripper_open) > 0.5
        label_gripper_command_open = bool(gripper_command_open)
        previous_predicted_width = online.observed_gripper_width(observation)
        previous_label_width = float(previous_predicted_width)
        max_steps = len(actions) if args.max_logical_steps <= 0 else min(len(actions), args.max_logical_steps)
        while logical_step < max_steps and not success and not terminated:
            dataset_item = dict(dataset[int(dataset_indices[logical_step])])
            model_batch = infer.preprocessor(default_collate([dataset_item]))
            reference_chunk = to_numpy(model_batch["action"])[0, :, :10]
            valid_rows = min(args.exec_action_steps, len(actions) - logical_step)
            call_seed = int(args.noise_seed + logical_step)
            teacher_chunk = to_numpy(
                seeded_predict_preprocessed(infer, model_batch, call_seed)
            )[0, :, :10]

            recorded_image = image_uint8(dataset_item["observation.images.front"])
            dataset_cloud = unwrap_point_cloud(dataset_item["observation.point_cloud"])
            stored_online_observation = {
                "point_cloud": dataset_cloud,
                "state": to_numpy(dataset_item["observation.state"]),
                "front": recorded_image,
                "agentview": recorded_image,
                "observation.images.agentview": recorded_image,
            }
            stored_online_batch = infer.build_model_batch(
                stored_online_observation,
                task=str(dataset_item.get("task", language)),
                state_pose_mode="identity",
            )
            stored_online_chunk = to_numpy(
                seeded_predict_preprocessed(infer, stored_online_batch, call_seed)
            )[0, :, :10]

            live_model_observation = online.live_model_observation(
                observation,
                SimpleNamespace(
                    num_points=args.num_points,
                    add_gripper_cloud=True,
                    gripper_template="reap",
                    gripper_points=args.gripper_points,
                ),
                seed=logical_step,
            )
            live_chunk = to_numpy(
                infer.predict_action_chunk_obs(
                    live_model_observation,
                    task=language,
                    postprocess=True,
                    state_pose_mode="identity",
                    noise_seed=call_seed,
                )
            )[0, :, :10]

            if args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC:
                if gripper_initial_sync_info is None:
                    gripper_initial_sync_info = (
                        set_gripper_absolute_width_position_target(
                            task_env, float(live_chunk[0, 9]), 0.08
                        )
                    )
                    observation = task_env.get_observation()
                    gripper_command_open = bool(
                        gripper_initial_sync_info["command_gripper_open"]
                    )
                    label_gripper_command_open = bool(gripper_command_open)
                previous_predicted_width = initial_delta_reference_for_chunk(
                    live_chunk, 0
                )
                previous_label_width = initial_delta_reference_for_chunk(
                    reference_chunk, 0
                )

            live_image = image_uint8(observation.front_rgb)
            if recorded_image.shape != live_image.shape:
                recorded_image = np.asarray(
                    Image.fromarray(recorded_image).resize(
                        (live_image.shape[1], live_image.shape[0]), Image.Resampling.BILINEAR
                    )
                )
            live_cloud = unwrap_point_cloud(live_model_observation["point_cloud"])
            point_metrics = point_cloud_distance(
                live_cloud, dataset_cloud, gripper_points=args.gripper_points
            )
            call_report = {
                "model_call": len(calls),
                "logical_step": logical_step,
                "noise_seed": call_seed,
                "valid_rows": valid_rows,
                "live_task_text": language,
                "dataset_task_value": str(dataset_item.get("task", "")),
                "teacher_vs_label": chunk_errors(teacher_chunk, reference_chunk, valid_rows),
                "stored_eval_path_vs_label": chunk_errors(
                    stored_online_chunk, reference_chunk, valid_rows
                ),
                "stored_eval_path_vs_teacher": chunk_errors(
                    stored_online_chunk, teacher_chunk, valid_rows
                ),
                "stored_preprocessor_batch_differences": model_batch_differences(
                    model_batch, stored_online_batch
                ),
                "live_vs_label": chunk_errors(live_chunk, reference_chunk, valid_rows),
                "live_vs_teacher": chunk_errors(live_chunk, teacher_chunk, valid_rows),
                "input_rgb_mae": float(
                    np.mean(np.abs(live_image.astype(np.float32) - recorded_image.astype(np.float32)))
                ),
                "input_point_cloud": point_metrics,
                "live_model_state10": np.asarray(live_model_observation["state"]).tolist(),
                "processed_dataset_state10": to_numpy(model_batch["observation.state"])[0].tolist(),
            }
            calls.append(call_report)

            chunk_anchor_world = online.pose9_to_homo_np(
                online.pose7_to_pose9(observation.gripper_pose)
            )
            for row_index in range(valid_rows):
                frame_index = logical_step + row_index
                before_expected_world = episode_origin_world @ online.pose9_to_homo_np(states[frame_index, :9])
                before_actual_world = replay.pose7_to_matrix(observation.gripper_pose)
                before_pos, before_rot = online.pose_error(before_actual_world, before_expected_world)

                predicted_relative = online.pose9_to_homo_np(live_chunk[row_index, :9])
                predicted_world = chunk_anchor_world @ predicted_relative
                label_world = episode_origin_world @ online.pose9_to_homo_np(actions[frame_index, :9])
                target_pos, target_rot = online.pose_error(predicted_world, label_world)
                if args.gripper_mode == ABSOLUTE_WIDTH:
                    predicted_open, predicted_gripper_event = (
                        absolute_width_gripper_target(
                            live_chunk[row_index, 9], args.gripper_open_threshold
                        )
                    )
                    label_open, label_gripper_event = absolute_width_gripper_target(
                        reference_chunk[row_index, 9], args.gripper_open_threshold
                    )
                    predicted_width_change = float(live_chunk[row_index, 9]) - float(
                        previous_predicted_width
                    )
                    label_width_change = float(reference_chunk[row_index, 9]) - float(
                        previous_label_width
                    )
                else:
                    predicted_open, predicted_gripper_event, predicted_width_change = (
                        libero_style_gripper_target(
                            previous_predicted_width,
                            live_chunk[row_index, 9],
                            gripper_command_open,
                            args.gripper_delta_threshold,
                            alignment=args.gripper_delta_alignment,
                            next_width=(
                                live_chunk[row_index + 1, 9]
                                if row_index + 1 < len(live_chunk)
                                else None
                            ),
                        )
                    )
                    label_open, label_gripper_event, label_width_change = (
                        libero_style_gripper_target(
                            previous_label_width,
                            reference_chunk[row_index, 9],
                            label_gripper_command_open,
                            args.gripper_delta_threshold,
                            alignment=args.gripper_delta_alignment,
                            next_width=(
                                reference_chunk[row_index + 1, 9]
                                if row_index + 1 < len(reference_chunk)
                                else None
                            ),
                        )
                    )
                previous_predicted_width = float(live_chunk[row_index, 9])
                previous_label_width = float(reference_chunk[row_index, 9])
                gripper_command_open = bool(predicted_open)
                label_gripper_command_open = bool(label_open)
                command = np.concatenate(
                    [online.matrix_to_pose7(predicted_world), [1.0 if predicted_open else 0.0]]
                ).astype(np.float32)
                try:
                    observation, reward, terminated = task_env.step(command)
                except Exception as error:
                    controller_error = {
                        "logical_step": frame_index,
                        "model_call": len(calls) - 1,
                        "chunk_row": row_index,
                        "error": repr(error),
                    }
                    break
                success = bool(float(reward) > 0.0)
                if not success:
                    success_value, _ = task_env._task.success()
                    success = bool(success_value)

                expected_next_index = min(frame_index + 1, len(states) - 1)
                expected_next_world = episode_origin_world @ online.pose9_to_homo_np(
                    states[expected_next_index, :9]
                )
                actual_next_world = replay.pose7_to_matrix(observation.gripper_pose)
                obs_pos, obs_rot = online.pose_error(actual_next_world, expected_next_world)
                actual_width = online.observed_gripper_width(observation)
                expected_width = float(states[expected_next_index, 9])
                row_report = {
                    "logical_step": frame_index,
                    "model_call": len(calls) - 1,
                    "chunk_row": row_index,
                    "before_observation_position_error_m": float(before_pos),
                    "before_observation_rotation_error_rad": float(before_rot),
                    "predicted_target_vs_label_position_error_m": float(target_pos),
                    "predicted_target_vs_label_rotation_error_rad": float(target_rot),
                    "predicted_width_m": float(live_chunk[row_index, 9]),
                    "label_width_m": float(actions[frame_index, 9]),
                    "predicted_gripper_event": predicted_gripper_event,
                    "label_gripper_event": label_gripper_event,
                    "predicted_width_change_m": float(predicted_width_change),
                    "label_width_change_m": float(label_width_change),
                    "predicted_label_gripper_discrete_mismatch": bool(predicted_open != label_open),
                    "after_observation_position_error_m": float(obs_pos),
                    "after_observation_rotation_error_rad": float(obs_rot),
                    "after_observation_gripper_width_error_m": abs(actual_width - expected_width),
                    "actual_gripper_width_m": float(actual_width),
                    "expected_gripper_width_m": float(expected_width),
                    "actual_gripper_open": bool(actual_width > 0.04),
                    "expected_gripper_open": bool(expected_width > 0.04),
                    "success": success,
                }
                rows.append(row_report)
                video_frames.append(
                    annotate(
                        observation.front_rgb,
                        [
                            f"step={frame_index} call={len(calls)-1} row={row_index}",
                            f"target err: {target_pos*1000:.1f}mm {np.degrees(target_rot):.1f}deg",
                            f"obs err: {obs_pos*1000:.1f}mm {np.degrees(obs_rot):.1f}deg",
                            f"grip pred/label/actual={int(predicted_open)}/{int(label_open)}/{int(actual_width>0.04)}",
                            f"success={int(success)}",
                        ],
                    )
                )
                if success or terminated:
                    break
            logical_step += valid_rows
            if controller_error is not None:
                break
    finally:
        environment.shutdown()
        infer.close()

    summary = {
        "task": args.task,
        "local_episode": args.episode,
        "global_episode": global_episode,
        "policy_path": str(args.policy_path),
        "dataset_root": str(args.dataset_root),
        "rollout_executes_dataset_actions": False,
        "rollout_action_source": "policy prediction only",
        "exec_action_steps": int(args.exec_action_steps),
        "gripper_mode": args.gripper_mode,
        "gripper_delta_threshold_m": float(args.gripper_delta_threshold),
        "gripper_delta_alignment": args.gripper_delta_alignment,
        "gripper_initial_sync": gripper_initial_sync_info,
        "model_noise_seed_base": int(args.noise_seed),
        "success": success,
        "terminated": bool(terminated),
        "controller_error": controller_error,
        "logical_steps_executed": len(rows),
        "model_calls": len(calls),
        "reset_validation": initial_validation,
        "dataset_expert_target_vs_next_observation": dataset_action_tracking_summary(actions, states),
        "aggregate_model_errors": {
            key: aggregate_call_errors(calls, key)
            for key in (
                "teacher_vs_label",
                "stored_eval_path_vs_label",
                "stored_eval_path_vs_teacher",
                "live_vs_label",
                "live_vs_teacher",
            )
        },
        "first_divergence": {
            "observation_position_gt_5mm": first_threshold(rows, "after_observation_position_error_m", 0.005),
            "observation_position_gt_20mm": first_threshold(rows, "after_observation_position_error_m", 0.02),
            "observation_rotation_gt_5deg": first_threshold(
                rows, "after_observation_rotation_error_rad", np.deg2rad(5.0)
            ),
            "predicted_target_position_gt_20mm": first_threshold(
                rows, "predicted_target_vs_label_position_error_m", 0.02
            ),
            "predicted_target_rotation_gt_5deg": first_threshold(
                rows, "predicted_target_vs_label_rotation_error_rad", np.deg2rad(5.0)
            ),
            "gripper_discrete_mismatch": next(
                (
                    int(row["logical_step"])
                    for row in rows
                    if row["predicted_label_gripper_discrete_mismatch"]
                ),
                None,
            ),
        },
        "model_calls_detail": calls,
        "steps": rows,
    }
    with open(args.output_dir / "diagnostic.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with open(args.output_dir / "steps.jsonl", "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    if video_frames:
        video_frames = annotate_final_task_result_frames(video_frames, success)
        imageio.mimsave(
            args.output_dir / "closed_loop.mp4",
            video_frames,
            fps=args.video_fps,
            macro_block_size=1,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
