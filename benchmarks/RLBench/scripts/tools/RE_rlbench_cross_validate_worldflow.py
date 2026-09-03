#!/usr/bin/env python3
"""Cross-validate RLBench robot-base WorldFlow sidecars against independent data paths.

This intentionally does more than a matrix round trip.  It checks that the
saved base trajectories reproduce the LeRobot observation/action labels, that
the transition-aligned command is closer to the next achieved pose, that the
real training wrapper returns the intended frames, and that the stored REAP
points are geometrically expressed in the current EEF frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import torch
import zarr
from scipy.spatial.transform import Rotation


def pose9_to_matrix(value: np.ndarray) -> np.ndarray:
    """Independent rot6d decoder: xyz + the first two rotation columns."""
    value = np.asarray(value, dtype=np.float64)
    a = value[..., 3:6]
    b = value[..., 6:9]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c = np.cross(a, b)
    out = np.zeros(value.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, 0] = a
    out[..., :3, 1] = b
    out[..., :3, 2] = c
    out[..., :3, 3] = value[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_pose9(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.concatenate(
        [value[..., :3, 3], value[..., :3, 0], value[..., :3, 1]], axis=-1
    )


def pose_errors(reference: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = pose9_to_matrix(reference)
    actual = pose9_to_matrix(actual)
    translation = np.linalg.norm(reference[..., :3, 3] - actual[..., :3, 3], axis=-1)
    relative = np.swapaxes(reference[..., :3, :3], -1, -2) @ actual[..., :3, :3]
    angle = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(translation.shape)
    return translation, angle


def summarize(values: list[float] | np.ndarray, scale: float = 1.0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def fixed_list(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float64)


def point_to_box_surface_distance(points: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    outside = np.maximum(np.maximum(low - points, points - high), 0.0)
    outside_distance = np.linalg.norm(outside, axis=1)
    inside = np.all((points >= low) & (points <= high), axis=1)
    inside_distance = np.min(np.minimum(points - low, high - points), axis=1)
    return np.where(inside, inside_distance, outside_distance)


def reap_surface_errors(local_points: np.ndarray, physical_width: float) -> np.ndarray:
    """Distance to the exact four-box REAP surface, in the pre-rotation frame."""
    width = float(np.clip(physical_width / 0.1, 0.0, 1.0)) * 0.1
    max_width = 0.06
    boxes = [
        (np.array([width / 2.0, 0.01, 0.0]), np.array([0.01, 0.08, 0.01])),
        (np.array([-width / 2.0 - 0.01, 0.01, 0.0]), np.array([0.01, 0.08, 0.01])),
        (np.array([-max_width / 2.0 - 0.005, 0.0, 0.0]), np.array([0.07, 0.01, 0.01])),
        (np.array([-0.005, -0.05, 0.0]), np.array([0.01, 0.05, 0.01])),
    ]
    # Collector: q = p @ static_rot.T + [0, 0, -0.09].
    static_rot = Rotation.from_euler("zyx", [math.pi / 2.0, math.pi / 2.0, 0.0]).as_matrix()
    pre_rot = (np.asarray(local_points, dtype=np.float64) - np.array([0.0, 0.0, -0.09])) @ static_rot
    distances = [point_to_box_surface_distance(pre_rot, low, low + size) for low, size in boxes]
    return np.min(np.stack(distances, axis=1), axis=1)


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, episode: int, frame: int, chunk: int):
        self.episode = episode
        self.frame = frame
        self.chunk = chunk

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {
            "episode_index": torch.tensor(self.episode),
            "frame_index": torch.tensor(self.frame),
            "action": torch.zeros((self.chunk, 10), dtype=torch.float32),
        }


def load_training_wrapper(script: Path):
    spec = importlib.util.spec_from_file_location("worldflow_training_entry_for_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.WorldFlowMemmapDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--pointcloud-samples", type=int, default=20)
    parser.add_argument("--loader-samples", type=int, default=100)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    achieved_dir = root / "world_base_ee_poses"
    target_dir = root / "world_base_action_target_ee_poses"
    world_dir = root / "world_ee_poses"
    with open(achieved_dir / "meta.json", encoding="utf-8") as handle:
        achieved_meta = json.load(handle)
    with open(target_dir / "meta.json", encoding="utf-8") as handle:
        target_meta = json.load(handle)
    t_world_base = np.asarray(achieved_meta["T_world_base"], dtype=np.float64)
    t_base_world = np.linalg.inv(t_world_base)

    table = pads.dataset(str(root / "data"), format="parquet").to_table(
        columns=["action", "observation.state", "episode_index", "frame_index"]
    )
    actions_all = fixed_list(table, "action")
    states_all = fixed_list(table, "observation.state")
    episodes_all = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frames_all = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)

    state_position_errors: list[float] = []
    state_rotation_errors: list[float] = []
    action_position_errors: list[float] = []
    action_rotation_errors: list[float] = []
    world_position_errors: list[float] = []
    world_rotation_errors: list[float] = []
    next_position_errors: list[float] = []
    next_rotation_errors: list[float] = []
    same_position_errors: list[float] = []
    same_rotation_errors: list[float] = []
    lengths: dict[int, int] = {}
    episode_payload: dict[int, dict[str, np.ndarray]] = {}

    episode_ids = sorted(int(value) for value in np.unique(episodes_all))
    for episode in episode_ids:
        selection = np.flatnonzero(episodes_all == episode)
        selection = selection[np.argsort(frames_all[selection])]
        actions = actions_all[selection]
        states = states_all[selection]
        achieved = np.asarray(np.load(achieved_dir / f"episode_{episode:06d}.npy"), dtype=np.float64)
        targets = np.asarray(np.load(target_dir / f"episode_{episode:06d}.npy"), dtype=np.float64)
        world = np.asarray(np.load(world_dir / f"episode_{episode:06d}.npy"), dtype=np.float64)
        if not (len(actions) == len(states) == len(achieved) == len(targets) == len(world)):
            raise RuntimeError(f"episode {episode}: inconsistent lengths")
        lengths[episode] = len(actions)

        base_achieved_m = pose9_to_matrix(achieved)
        base_target_m = pose9_to_matrix(targets)
        state_from_base = matrix_to_pose9(np.linalg.inv(base_achieved_m[0])[None] @ base_achieved_m)
        action_from_base = matrix_to_pose9(np.linalg.inv(base_achieved_m[0])[None] @ base_target_m)
        pos, rot = pose_errors(states[:, :9], state_from_base)
        state_position_errors.extend(pos.tolist())
        state_rotation_errors.extend(rot.tolist())
        pos, rot = pose_errors(actions[:, :9], action_from_base)
        action_position_errors.extend(pos.tolist())
        action_rotation_errors.extend(rot.tolist())

        world_from_base = matrix_to_pose9(t_world_base[None] @ base_achieved_m)
        pos, rot = pose_errors(world, world_from_base)
        world_position_errors.extend(pos.tolist())
        world_rotation_errors.extend(rot.tolist())

        if len(actions) > 1:
            pos, rot = pose_errors(targets[:-1], achieved[1:])
            next_position_errors.extend(pos.tolist())
            next_rotation_errors.extend(rot.tolist())
            pos, rot = pose_errors(targets[:-1], achieved[:-1])
            same_position_errors.extend(pos.tolist())
            same_rotation_errors.extend(rot.tolist())
        episode_payload[episode] = {"actions": actions, "states": states, "achieved": achieved, "targets": targets}

    # Exercise the exact class used by the requested training entry point.
    training_script = root.parents[2] / "song_real_libero" / "scripts" / "train_song_benchmark.py"
    WorldFlowMemmapDataset = load_training_wrapper(training_script)
    rng = np.random.default_rng(20260828)
    loader_current_max = 0.0
    loader_target_max = 0.0
    loader_pad_mismatches = 0
    for _ in range(args.loader_samples):
        episode = int(rng.choice(episode_ids))
        length = lengths[episode]
        frame = int(rng.integers(0, length))
        chunk = int(rng.integers(1, 33))
        wrapped = WorldFlowMemmapDataset(
            FakeDataset(episode, frame, chunk), root, chunk_size=chunk,
            target_type="world_eef_trajectory", action_start_offset=0,
            require_action_target_sidecar=True,
        )
        item = wrapped[0]
        expected_indices = np.clip(frame + np.arange(chunk), 0, length - 1)
        expected_pad = frame + np.arange(chunk) >= length
        loader_current_max = max(loader_current_max, float(np.max(np.abs(
            item["worldflow.current_ee_pose"].numpy() - episode_payload[episode]["achieved"][frame]
        ))))
        loader_target_max = max(loader_target_max, float(np.max(np.abs(
            item["worldflow.eef_trajectory"].numpy() - episode_payload[episode]["targets"][expected_indices]
        ))))
        loader_pad_mismatches += int(np.count_nonzero(item["worldflow.step_is_pad"].numpy() != expected_pad))

    # Stored cloud must contain a red 500-point REAP surface in current-EEF coordinates.
    cloud_surface_errors: list[float] = []
    cloud_rgb_errors: list[float] = []
    sampled_frames: list[dict[str, int]] = []
    for _ in range(min(args.pointcloud_samples, len(episode_ids))):
        episode = int(rng.choice(episode_ids))
        frame = int(rng.integers(0, lengths[episode]))
        group = zarr.open_group(str(root / "point_clouds" / f"episode_{episode:06d}.zarr"), mode="r")
        xyz = np.asarray(group["xyz"][frame, -args.gripper_points:], dtype=np.float64)
        rgb = np.asarray(group["rgb"][frame, -args.gripper_points:], dtype=np.float64)
        surface = reap_surface_errors(xyz, episode_payload[episode]["states"][frame, 9])
        cloud_surface_errors.extend(surface.tolist())
        cloud_rgb_errors.extend(np.max(np.abs(rgb - np.array([204.0, 51.0, 51.0])), axis=1).tolist())
        sampled_frames.append({"episode": episode, "frame": frame})

    next_pos = np.asarray(next_position_errors)
    same_pos = np.asarray(same_position_errors)
    result = {
        "dataset_root": str(root),
        "episodes": len(episode_ids),
        "frames": int(len(actions_all)),
        "metadata_contract": {
            "achieved_coordinate_frame": achieved_meta.get("coordinate_frame"),
            "target_coordinate_frame": target_meta.get("coordinate_frame"),
            "target_semantics": target_meta.get("target_semantics"),
            "alignment": target_meta.get("alignment"),
            "T_world_base": t_world_base.tolist(),
            "T_base_world": t_base_world.tolist(),
        },
        "independent_parquet_consistency": {
            "state_from_base_achieved_translation_mm": summarize(state_position_errors, 1000.0),
            "state_from_base_achieved_rotation_deg": summarize(state_rotation_errors, 180.0 / math.pi),
            "action_from_base_target_translation_mm": summarize(action_position_errors, 1000.0),
            "action_from_base_target_rotation_deg": summarize(action_rotation_errors, 180.0 / math.pi),
            "world_from_base_achieved_translation_mm": summarize(world_position_errors, 1000.0),
            "world_from_base_achieved_rotation_deg": summarize(world_rotation_errors, 180.0 / math.pi),
        },
        "transition_semantics": {
            "target_vs_next_achieved_translation_mm": summarize(next_position_errors, 1000.0),
            "target_vs_next_achieved_rotation_deg": summarize(next_rotation_errors, 180.0 / math.pi),
            "target_vs_same_achieved_translation_mm": summarize(same_position_errors, 1000.0),
            "target_vs_same_achieved_rotation_deg": summarize(same_rotation_errors, 180.0 / math.pi),
            "fraction_next_translation_closer_than_same": float(np.mean(next_pos < same_pos)),
        },
        "actual_training_loader": {
            "script": str(training_script),
            "class": "WorldFlowMemmapDataset",
            "samples": int(args.loader_samples),
            "action_start_offset": 0,
            "current_pose_max_abs": loader_current_max,
            "target_pose_max_abs": loader_target_max,
            "padding_mismatch_count": loader_pad_mismatches,
        },
        "pointcloud_current_eef_reap": {
            "sampled_frames": sampled_frames,
            "points": len(cloud_surface_errors),
            "surface_distance_mm": summarize(cloud_surface_errors, 1000.0),
            "red_rgb_max_channel_error": summarize(cloud_rgb_errors),
        },
    }
    result["pass"] = bool(
        result["independent_parquet_consistency"]["state_from_base_achieved_translation_mm"]["max"] < 0.1
        and result["independent_parquet_consistency"]["action_from_base_target_translation_mm"]["max"] < 0.1
        and result["independent_parquet_consistency"]["world_from_base_achieved_translation_mm"]["max"] < 0.1
        and loader_current_max < 1e-6
        and loader_target_max < 1e-6
        and loader_pad_mismatches == 0
        and result["pointcloud_current_eef_reap"]["surface_distance_mm"]["max"] < 0.2
        and result["pointcloud_current_eef_reap"]["red_rgb_max_channel_error"]["max"] == 0.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
