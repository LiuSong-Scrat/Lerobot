#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize predicted and ground-truth action chunks from the RLBench training set.

The saved checkpoint preprocessor converts both the dataset action and the
current observation to the coordinate system used during training. Therefore
both trajectories plotted here are relative to the EEF pose at the sampled
frame. They can be compared directly without an RLBench environment reset.

Colors:
    blue: ground-truth future action trajectory
    red: model-predicted future action trajectory
    gray: line joining corresponding ground-truth and predicted targets
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


from _rlbench_tool_paths import LEROBOT_ROOT as REPO_ROOT
SONG_SCRIPTS = REPO_ROOT / "benchmarks" / "song_real_libero" / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SONG_SCRIPTS))

from smolvla_model_inference import SmolVLA_ModelInference, rot6d_to_matrix


DEFAULT_POLICY = (
    REPO_ROOT
    / "benchmarks/RLBench/outputs/wep_vla_v041_rlbench_no_ood/checkpoints/last/pretrained_model"
)
DEFAULT_DATASET = (
    REPO_ROOT
    / "benchmarks/RLBench/datasets/"
    "rlbench_water_plants_sweep_to_dustpan_150episodes_pointcloud_lerobot"
)
RUN_DATE = time.strftime("%Y%m%d%H%M%S")
DEFAULT_OUTPUT = REPO_ROOT / ("benchmarks/RLBench/outputs/action_pred_err_" + RUN_DATE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=10,
        help="Number of evenly spaced training episodes selected for each task.",
    )
    parser.add_argument(
        "--frame-fractions",
        default="0.10,0.50,0.80",
        help="Positions sampled within each selected episode; each must leave 32 future actions.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Optional exact language task, for example 'water plant'. Empty means every task.",
    )
    return parser.parse_args()


def rotation_errors_deg(predicted, target):
    predicted_matrix = rot6d_to_matrix(torch.from_numpy(predicted[:, 3:9])).numpy()
    target_matrix = rot6d_to_matrix(torch.from_numpy(target[:, 3:9])).numpy()
    relative = np.swapaxes(target_matrix, 1, 2) @ predicted_matrix
    return np.degrees(Rotation.from_matrix(relative).magnitude())


def compute_metrics(predicted, target):
    position_error_mm = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1) * 1000.0
    rotation_error_deg = rotation_errors_deg(predicted, target)
    gripper_error_mm = np.abs(predicted[:, 9] - target[:, 9]) * 1000.0
    gripper_matches = (predicted[:, 9] > 0.04) == (target[:, 9] > 0.04)
    return {
        "position_error_mm": position_error_mm.tolist(),
        "rotation_error_deg": rotation_error_deg.tolist(),
        "gripper_error_mm": gripper_error_mm.tolist(),
        "position_error_mm_mean": float(position_error_mm.mean()),
        "position_error_mm_median": float(np.median(position_error_mm)),
        "position_error_mm_max": float(position_error_mm.max()),
        "first_1_position_error_mm": float(position_error_mm[0]),
        "first_4_position_error_mm_mean": float(position_error_mm[:4].mean()),
        "rotation_error_deg_mean": float(rotation_error_deg.mean()),
        "gripper_error_mm_mean": float(gripper_error_mm.mean()),
        "gripper_binary_accuracy": float(gripper_matches.mean()),
    }


def set_axes_equal(axis, first_xyz, second_xyz):
    points = np.concatenate((first_xyz, second_xyz, np.zeros((1, 3), dtype=np.float32)), axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def trajectory_color_scale(count, start_rgb, end_rgb):
    """Create one RGB color per action step so time is visible in the trajectory."""
    if count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    start = np.asarray(start_rgb, dtype=np.float32)[None, :]
    end = np.asarray(end_rgb, dtype=np.float32)[None, :]
    return np.rint((1.0 - alpha) * start + alpha * end).astype(np.uint8)


def save_png(path, predicted, target, task, episode_index, frame_index, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted_xyz = predicted[:, :3]
    target_xyz = target[:, :3]
    steps = np.arange(len(predicted))
    position_error = np.asarray(metrics["position_error_mm"], dtype=np.float32)
    target_colors = trajectory_color_scale(len(target_xyz), [0, 40, 180], [0, 235, 255])
    predicted_colors = trajectory_color_scale(len(predicted_xyz), [255, 220, 0], [220, 0, 50])

    figure = plt.figure(figsize=(14, 6))
    trajectory_axis = figure.add_subplot(1, 2, 1, projection="3d")
    for index in range(len(target_xyz) - 1):
        trajectory_axis.plot(
            target_xyz[index : index + 2, 0],
            target_xyz[index : index + 2, 1],
            target_xyz[index : index + 2, 2],
            color=target_colors[index].astype(np.float32) / 255.0,
            linewidth=2.2,
        )
        trajectory_axis.plot(
            predicted_xyz[index : index + 2, 0],
            predicted_xyz[index : index + 2, 1],
            predicted_xyz[index : index + 2, 2],
            color=predicted_colors[index].astype(np.float32) / 255.0,
            linewidth=2.2,
        )
    trajectory_axis.scatter(
        target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2],
        color=target_colors.astype(np.float32) / 255.0, s=13,
    )
    trajectory_axis.scatter(
        predicted_xyz[:, 0], predicted_xyz[:, 1], predicted_xyz[:, 2],
        color=predicted_colors.astype(np.float32) / 255.0, s=13,
    )
    trajectory_axis.plot([], [], [], color="#0028b4", linewidth=2.2, label="Ground truth: blue -> cyan")
    trajectory_axis.plot([], [], [], color="#dc0032", linewidth=2.2, label="Prediction: yellow -> red")
    for index in range(len(predicted_xyz)):
        trajectory_axis.plot(
            [target_xyz[index, 0], predicted_xyz[index, 0]],
            [target_xyz[index, 1], predicted_xyz[index, 1]],
            [target_xyz[index, 2], predicted_xyz[index, 2]],
            color="#777777", linewidth=0.6, alpha=0.6,
        )
    for index in [0, 3, 7, 15, 23, 31]:
        if index < len(target_xyz):
            target_text_color = target_colors[index].astype(np.float32) / 255.0
            predicted_text_color = predicted_colors[index].astype(np.float32) / 255.0
            trajectory_axis.text(*target_xyz[index], str(index + 1), color=target_text_color, fontsize=8)
            trajectory_axis.text(
                *predicted_xyz[index], str(index + 1), color=predicted_text_color, fontsize=8
            )
    trajectory_axis.scatter([0.0], [0.0], [0.0], color="black", marker="x", s=55, label="Current EEF")
    trajectory_axis.set_xlabel("EEF X (m)")
    trajectory_axis.set_ylabel("EEF Y (m)")
    trajectory_axis.set_zlabel("EEF Z (m)")
    trajectory_axis.set_title("Future 32-target trajectory")
    trajectory_axis.legend(loc="upper left")
    set_axes_equal(trajectory_axis, predicted_xyz, target_xyz)

    error_axis = figure.add_subplot(1, 2, 2)
    error_axis.plot(steps + 1, position_error, color="#37474f", marker="o", markersize=3)
    error_axis.axhline(
        metrics["position_error_mm_mean"], color="#d32f2f", linestyle="--",
        label="Mean = %.2f mm" % metrics["position_error_mm_mean"],
    )
    error_axis.set_xlabel("Action chunk step")
    error_axis.set_ylabel("Position target error (mm)")
    error_axis.set_xlim(1, len(predicted))
    error_axis.grid(alpha=0.25)
    error_axis.legend()
    error_axis.set_title(
        "Position error\nrotation mean %.2f deg, gripper binary %.1f%%"
        % (metrics["rotation_error_deg_mean"], metrics["gripper_binary_accuracy"] * 100.0)
    )

    figure.suptitle(
        task + " | episode " + str(episode_index) + " frame " + str(frame_index),
        fontsize=13,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def sample_line(start, end, count=10):
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    return (1.0 - alpha) * start[None, :] + alpha * end[None, :]


def point_cloud_to_numpy(point_cloud):
    """Return the current-frame model point cloud as an (N, 6) NumPy array."""
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.detach().cpu().numpy()
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    while point_cloud.ndim > 2 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
        raise ValueError(
            "observation.point_cloud must have shape (N, >=3), got "
            + str(point_cloud.shape)
        )
    return point_cloud


def save_ply(path, predicted, target, point_cloud):
    scene = point_cloud_to_numpy(point_cloud)
    scene_valid = np.isfinite(scene[:, :3]).all(axis=1)
    scene = scene[scene_valid]
    scene_xyz = scene[:, :3].astype(np.float32)
    if scene.shape[1] >= 6:
        scene_rgb = scene[:, 3:6]
        if scene_rgb.size and float(scene_rgb.max()) <= 1.0:
            scene_rgb = scene_rgb * 255.0
        scene_rgb = np.clip(scene_rgb, 0.0, 255.0).astype(np.uint8)
    else:
        scene_rgb = np.full((len(scene_xyz), 3), 128, dtype=np.uint8)

    points = list(scene_xyz)
    colors = list(scene_rgb)
    point_kinds = [0] * len(scene_xyz)
    chunk_steps = [-1] * len(scene_xyz)
    source_indices = np.flatnonzero(scene_valid).astype(np.int32).tolist()

    trajectory_data = [
        (target[:, :3], trajectory_color_scale(len(target), [0, 40, 180], [0, 235, 255]), 1),
        (
            predicted[:, :3],
            trajectory_color_scale(len(predicted), [255, 220, 0], [220, 0, 50]),
            2,
        ),
    ]
    for trajectory, trajectory_colors, kind in trajectory_data:
        for index in range(len(trajectory) - 1):
            line = sample_line(trajectory[index], trajectory[index + 1])
            line_colors = trajectory_color_scale(
                len(line), trajectory_colors[index], trajectory_colors[index + 1]
            )
            points.extend(line)
            colors.extend(line_colors)
            point_kinds.extend([kind] * len(line))
            chunk_steps.extend([index] * len(line))
            source_indices.extend([-1] * len(line))
        points.extend(trajectory)
        colors.extend(trajectory_colors)
        point_kinds.extend([kind] * len(trajectory))
        chunk_steps.extend(range(len(trajectory)))
        source_indices.extend([-1] * len(trajectory))

    for index in range(len(predicted)):
        line = sample_line(target[index, :3], predicted[index, :3], count=6)
        points.extend(line)
        colors.extend([np.array([130, 130, 130], dtype=np.uint8)] * len(line))
        point_kinds.extend([3] * len(line))
        chunk_steps.extend([index] * len(line))
        source_indices.extend([-1] * len(line))

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment coordinate_frame current_eef_at_dataset_frame\n")
        file.write(
            "comment point_kind 0=scene 1=ground_truth 2=prediction 3=correspondence_error\n"
        )
        file.write("comment ground_truth_color step_1_blue_to_step_32_cyan\n")
        file.write("comment prediction_color step_1_yellow_to_step_32_red\n")
        file.write("comment scene_rgb original_training_point_cloud_rgb\n")
        file.write("comment scene_point_count " + str(len(scene_xyz)) + "\n")
        file.write("element vertex " + str(len(points)) + "\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property uchar point_kind\nproperty int chunk_step\n")
        file.write("property int source_point_index\nend_header\n")
        for point, color, kind, step, source_index in zip(
            points, colors, point_kinds, chunk_steps, source_indices
        ):
            file.write(
                "%.7f %.7f %.7f %d %d %d %d %d %d\n"
                % (
                    point[0], point[1], point[2],
                    color[0], color[1], color[2], kind, step, source_index,
                )
            )


def choose_samples(episodes, episodes_per_task, frame_fractions, requested_task):
    episodes_by_task = {}
    for episode_index, episode in enumerate(episodes):
        task = str(episode["tasks"][0])
        if requested_task and task != requested_task:
            continue
        if int(episode["length"]) >= 33:
            episodes_by_task.setdefault(task, []).append(episode_index)

    samples = []
    for task in sorted(episodes_by_task):
        candidates = episodes_by_task[task]
        count = min(episodes_per_task, len(candidates))
        episode_positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        for episode_position in episode_positions:
            episode_index = candidates[int(episode_position)]
            episode = episodes[episode_index]
            length = int(episode["length"])
            start = int(episode["dataset_from_index"])
            for fraction in frame_fractions:
                frame_index = int(round((length - 33) * fraction))
                samples.append((task, episode_index, frame_index, start + frame_index))
    return samples


def main():
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive.")
    fractions = [float(value) for value in args.frame_fractions.split(",") if value.strip()]
    if not fractions or any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("--frame-fractions values must be between 0 and 1.")

    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/tmp/rlbench_action_pred_err_hf")

    inference = SmolVLA_ModelInference(policy_path=policy_path, device=args.device)
    dataset = inference.load_dataset(dataset_root)
    samples = choose_samples(
        dataset.meta.episodes,
        args.episodes_per_task,
        fractions,
        args.task,
    )
    if not samples:
        inference.close()
        raise RuntimeError("No matching dataset samples were found.")

    records = []
    all_position_errors = []
    all_rotation_errors = []
    try:
        for sample_number, sample in enumerate(samples):
            task, episode_index, frame_index, dataset_index = sample
            dataset_item = dataset[dataset_index]
            point_cloud = dataset_item["observation.point_cloud"]
            torch.manual_seed(args.seed + sample_number)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + sample_number)
            result = inference.predict_from_dataset(dataset_index)
            predicted = np.asarray(result["action_chunk"][0], dtype=np.float32)
            target = np.asarray(result["gt_action_chunk"][0], dtype=np.float32)
            count = min(len(predicted), len(target), 32)
            predicted = predicted[:count, :10]
            target = target[:count, :10]
            metrics = compute_metrics(predicted, target)
            all_position_errors.extend(metrics["position_error_mm"])
            all_rotation_errors.extend(metrics["rotation_error_deg"])

            stem = (
                task.replace(" ", "_")
                + "_episode_" + str(episode_index).zfill(3)
                + "_frame_" + str(frame_index).zfill(3)
            )
            png_path = output_dir / (stem + ".png")
            ply_path = output_dir / (stem + ".ply")
            save_png(png_path, predicted, target, task, episode_index, frame_index, metrics)
            save_ply(ply_path, predicted, target, point_cloud)
            record = {
                "task": task,
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "dataset_index": int(dataset_index),
                "png": str(png_path),
                "ply": str(ply_path),
                "point_cloud_points": int(len(point_cloud_to_numpy(point_cloud))),
                "metrics": metrics,
            }
            records.append(record)
            print(
                "[sample] task=" + task
                + " episode=" + str(episode_index)
                + " frame=" + str(frame_index)
                + " first4_mm=" + format(metrics["first_4_position_error_mm_mean"], ".2f")
                + " full32_mm=" + format(metrics["position_error_mm_mean"], ".2f"),
                flush=True,
            )
    finally:
        inference.close()

    summary = {
        "run_date": RUN_DATE,
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "coordinate_frame": "current_eef_at_dataset_frame",
        "samples": records,
        "aggregate": {
            "sample_count": len(records),
            "action_target_count": len(all_position_errors),
            "position_error_mm_mean": float(np.mean(all_position_errors)),
            "position_error_mm_median": float(np.median(all_position_errors)),
            "position_error_mm_p90": float(np.percentile(all_position_errors, 90)),
            "rotation_error_deg_mean": float(np.mean(all_rotation_errors)),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("[done] " + str(output_dir), flush=True)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
