# Example: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py --config benchmarks/song_real_libero/configs/libero.json
#!/usr/bin/env python
"""Minimal LIBERO point-cloud evaluation with absolute-pose chunk execution.

Kept pipeline:
  observation -> point cloud -> model action chunk -> absolute OSC pose actions
  -> execute selected chunk rows -> optional videos + JSON summaries.

Removed from the original evaluator:
  pose/gripper wait state machines, fast-physics executor, rim correction,
  keyboard visualization, heavy timing diagnostics,
  and delta-pose execution branches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import BENCHMARK_ROOT, DEFAULT_LIBERO_CONFIG, load_json_config
    from ..smolvla_model_inference import SmolVLA_ModelInference, identity_pose9_gripper
    from .libero_hdf5_to_dataset import (
        append_video_frames,
        export_episode_videos,
        resolve_suite_names,
        resolve_task_ids_for_suite,
    )
    from .libero_pointcloud_utils import (
        add_world_gripper_cloud_to_point_cloud,
        attach_mujoco_3d_viewer,
        ensure_libero_config,
        eef_pose9_gripper_from_obs,
        fast_inverse_homogeneous,
        get_task_init_states,
        gripper_scalar,
        gripper_width_percent_from_scalar,
        make_libero_env,
        normalize_render_camera_name,
        observation_to_point_clouds,
        pointcloud_camera_names_from_config,
        pose9_to_homo_np,
        render_camera_names_from_config,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import BENCHMARK_ROOT, DEFAULT_LIBERO_CONFIG, load_json_config
    from smolvla_model_inference import SmolVLA_ModelInference, identity_pose9_gripper
    from libero_setting.libero_hdf5_to_dataset import (
        append_video_frames,
        export_episode_videos,
        resolve_suite_names,
        resolve_task_ids_for_suite,
    )
    from libero_setting.libero_pointcloud_utils import (
        add_world_gripper_cloud_to_point_cloud,
        attach_mujoco_3d_viewer,
        ensure_libero_config,
        eef_pose9_gripper_from_obs,
        fast_inverse_homogeneous,
        get_task_init_states,
        gripper_scalar,
        gripper_width_percent_from_scalar,
        make_libero_env,
        normalize_render_camera_name,
        observation_to_point_clouds,
        pointcloud_camera_names_from_config,
        pose9_to_homo_np,
        render_camera_names_from_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean LIBERO eval: point-cloud observation -> action chunk -> absolute pose execution."
    )
    parser.add_argument("--settle-steps", type=int, default=5, help="MuJoCo physics ticks after env.reset() before the first policy inference, so free objects can settle.")
    parser.add_argument("--settle-keep-robot-fixed", action=argparse.BooleanOptionalAction, default=True, help="During initial settling, restore arm/gripper qpos each sim tick so only free objects settle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", default=None)
    parser.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--observation-height", type=int, default=None)
    parser.add_argument("--observation-width", type=int, default=None)
    parser.add_argument("--render-mode", choices=("offscreen", "onscreen", "viewer3d"), default=None)
    parser.add_argument("--render-camera", default=None)
    parser.add_argument("--render-every-n-steps", type=int, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--control-freq", type=float, default=None)

    parser.add_argument("--action-index", type=int, default=None)
    parser.add_argument("--exec-action-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)

    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Normalized gripper-width threshold. width_pct >= threshold => open(-1), else close(+1).",
    )
    parser.add_argument("--gripper-qpos-max-width", type=float, default=None)
    parser.add_argument("--add-gripper-cloud", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gripper-points", type=int, default=None)

    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recreate-env-per-episode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return load_json_config(
        path,
        path_keys=("policy_path", "policy_repo_id", "libero_config_path", "demo_root", "output_dir", "vis_dir"),
    )


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg.get(key, default)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    rot = matrix[..., :3, :3]
    return np.concatenate([matrix[..., :3, 3], rot[..., :, 0], rot[..., :, 1]], axis=-1).astype(np.float32)


def world_pose_to_libero_absolute_action(target_world: np.ndarray) -> np.ndarray:
    """Convert a 4x4 world-frame target pose to robosuite OSC absolute-pose action[0:6]."""
    target_world = np.asarray(target_world, dtype=np.float32)
    rotvec = R.from_matrix(target_world[:3, :3]).as_rotvec().astype(np.float32)
    return np.concatenate([target_world[:3, 3], rotvec]).astype(np.float32)


def current_controller_eef_world(env: Any) -> np.ndarray:
    """Read the exact end-effector site pose controlled by robosuite OSC."""
    controller = env.robots[0].controller
    controller.update(force=True)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.asarray(controller.ee_ori_mat, dtype=np.float32)
    out[:3, 3] = np.asarray(controller.ee_pos, dtype=np.float32)
    return out


def iter_env_chain(env: Any, max_depth: int = 32) -> list[Any]:
    """Return env plus wrapped inner envs, outermost first."""
    chain: list[Any] = []
    current = env
    for _ in range(max_depth):
        if current is None or any(current is old for old in chain):
            break
        chain.append(current)
        next_env = getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return chain


def find_attached_viewer(env: Any) -> Any | None:
    """Find a viewer handle on any LIBERO / robosuite wrapper layer."""
    for obj in reversed(iter_env_chain(env)):
        for attr in (
            "viewer",
            "_viewer",
            "mujoco_viewer",
            "_mujoco_viewer",
            "viewer3d",
            "_viewer3d",
            "mujoco_3d_viewer",
            "_mujoco_3d_viewer",
            "passive_viewer",
            "_passive_viewer",
        ):
            viewer = getattr(obj, attr, None)
            if viewer is not None:
                return viewer
    return None


def sync_viewer(viewer: Any) -> bool:
    """Advance a mujoco passive viewer or robosuite viewer wrapper once."""
    if viewer is None:
        return False
    for name in ("sync", "render", "update"):
        fn = getattr(viewer, name, None)
        if callable(fn):
            fn()
            return True
    wrapped = getattr(viewer, "viewer", None)
    if wrapped is not None and hasattr(wrapped, "sync"):
        wrapped.sync()
        return True
    return False


def render_viewer3d(env: Any, cfg: dict[str, Any], step: int, *, force: bool = False) -> None:
    """Minimal Viewer3D attach/sync; intentionally no complex lifecycle logic."""
    render_mode = str(cfg.get("render_mode", "offscreen")).lower()
    if render_mode != "viewer3d":
        return

    every_n = max(1, int(cfg.get("render_every_n_steps", 1) or 1))
    if not force and int(step) % every_n != 0:
        return

    viewer = find_attached_viewer(env)
    if viewer is None:
        render_camera = normalize_render_camera_name(str(cfg.get("render_camera", "agentview")))
        viewer = attach_mujoco_3d_viewer(env, render_camera=render_camera)

    try:
        sync_viewer(viewer)
        time.sleep(1.0 / 60.0)
    except Exception as exc:
        print(f"[WARN] Viewer3D sync failed: {exc!r}", flush=True)


def gripper_threshold_command(
    predicted_width: float,
    *,
    threshold: float,
    max_physical_width: float,
) -> float:
    """LIBERO / robosuite PandaGripper convention: -1 opens, +1 closes."""
    width_pct = gripper_width_percent_from_scalar(
        float(predicted_width),
        max_physical_width=float(max_physical_width),
    )
    return -1.0 if width_pct >= float(threshold) else 1.0


def action_chunk_to_absolute_libero_actions(
    *,
    env: Any,
    current_eef_pose9_gripper: np.ndarray,
    action_chunk: np.ndarray,
    gripper_threshold: float,
    gripper_max_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one model chunk to directly executable absolute OSC actions.

    The model chunk is treated like the original UMI-style trajectory: each row's
    pose9 is relative to the current observation EEF frame.  We first turn the
    whole chunk into fixed model-world targets, then map model EEF frame to the
    robosuite OSC controller EEF site frame, and finally emit 7D LIBERO actions:
        [abs_x, abs_y, abs_z, abs_rx, abs_ry, abs_rz, gripper_open_close]
    """
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < 10:
        raise ValueError(f"Expected action chunk shape (T, >=10), got {chunk.shape}.")

    current_model_world = pose9_to_homo_np(np.asarray(current_eef_pose9_gripper, dtype=np.float32)[..., :9])
    current_controller_world = current_controller_eef_world(env)
    model_to_controller = fast_inverse_homogeneous(current_model_world) @ current_controller_world

    relative_targets = pose9_to_homo_np(chunk[:, :9])
    target_model_worlds = current_model_world @ relative_targets

    libero_actions: list[np.ndarray] = []
    target_controller_pose9: list[np.ndarray] = []
    for idx, (row, target_model_world) in enumerate(zip(chunk, target_model_worlds, strict=True)):
        target_controller_world = target_model_world @ model_to_controller
        arm_action = world_pose_to_libero_absolute_action(target_controller_world)
        delta_pre_width = chunk[idx+1,-1]-chunk[idx,-1] if idx< chunk.shape[0]-1 else 0
        gripper_action = -1.0 if delta_pre_width > 0.004 else (1.0 if delta_pre_width < -0.004 else 0.0)
        libero_actions.append(np.concatenate([arm_action, np.asarray([gripper_action], dtype=np.float32)]))
        target_controller_pose9.append(matrix_to_pose9(target_controller_world))

    return (
        np.asarray(libero_actions, dtype=np.float32),
        target_model_worlds.astype(np.float32),
        np.asarray(target_controller_pose9, dtype=np.float32),
    )


def task_success_rate(episodes: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(item.get("success", False)) for item in episodes])) if episodes else 0.0


def make_task_summary(
    *,
    suite_name: str,
    task_id: int,
    task_name: str,
    task_language: str,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "task_id": int(task_id),
        "task_name": task_name,
        "task_language": task_language,
        "episode_count": int(len(episodes)),
        "success_rate": task_success_rate(episodes),
        "episodes": episodes,
    }


def aggregate_task_results(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = int(sum(len(task.get("episodes", [])) for task in tasks))
    success_count = int(
        sum(
            1
            for task in tasks
            for episode in task.get("episodes", [])
            if bool(episode.get("success", False))
        )
    )
    task_success_rates = [float(task.get("success_rate", 0.0)) for task in tasks]
    return {
        "task_count": int(len(tasks)),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": float(success_count / episode_count) if episode_count else 0.0,
        "task_success_rate_mean": float(np.mean(task_success_rates)) if task_success_rates else 0.0,
    }


def write_eval_reports(output_dir: Path, cfg: dict[str, Any], suite_names: list[str], tasks: list[dict[str, Any]]) -> None:
    suite_reports = []
    for suite_name in suite_names:
        suite_tasks = [task for task in tasks if task.get("suite") == suite_name]
        suite_reports.append({"suite": suite_name, **aggregate_task_results(suite_tasks), "tasks": suite_tasks})

    summary = {
        "created_unix_s": time.time(),
        "policy_path": cfg.get("policy_path"),
        "suites": suite_names,
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": pointcloud_camera_names_from_config(cfg),
        "render_mode": str(cfg.get("render_mode", "offscreen")),
        "control": {
            "controller_mode": "OSC_POSE absolute pose",
            "action_dim": 7,
            "action_format": ["abs_x", "abs_y", "abs_z", "abs_rx", "abs_ry", "abs_rz", "gripper"],
            "gripper_convention": "-1=open, +1=close",
            "gripper_threshold": float(cfg["control"]["gripper_threshold"]),
            "exec_action_steps": int(cfg["control"]["exec_action_steps"]),
            "action_index": int(cfg["control"]["action_index"]),
            "control_freq": float(cfg["control"].get("control_freq", cfg.get("control_freq", 20))),
        },
        "overall": aggregate_task_results(tasks),
        "suite_reports": suite_reports,
        "results": tasks,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "overall_report.json", {"overall": summary["overall"], "suites": suite_reports})
    for suite in suite_reports:
        write_json_atomic(output_dir / str(suite["suite"]) / "suite_report.json", suite)


def compact_episode_record(result: dict[str, Any], episode_idx: int, action_npz: str | None) -> dict[str, Any]:
    drop_keys = {
        "video_frames",
        "libero_actions",
        "model_action_rows",
        "target_controller_pose9",
        "target_model_worlds",
    }
    record = {k: v for k, v in result.items() if k not in drop_keys}
    record["episode_index"] = int(episode_idx)
    if action_npz is not None:
        record["action_npz"] = action_npz
    return json_safe(record)


def save_episode_actions(result: dict[str, Any], episode_dir: Path) -> str | None:
    arrays = {
        "libero_actions": np.asarray(result.get("libero_actions", []), dtype=np.float32),
        "model_action_rows": np.asarray(result.get("model_action_rows", []), dtype=np.float32),
        "target_controller_pose9": np.asarray(result.get("target_controller_pose9", []), dtype=np.float32),
        "target_model_worlds": np.asarray(result.get("target_model_worlds", []), dtype=np.float32),
    }
    if arrays["libero_actions"].size == 0:
        return None
    path = episode_dir / "actions.npz"
    np.savez_compressed(path, **arrays)
    return str(path)

def build_point_cloud_observation(env: Any, raw_obs: dict[str, Any], cfg: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the exact online UMI observation used for model inference.

    Returns:
      pc_eff:      xyzrgb in the current model EEF frame, with optional local gripper cloud.
      world_pose9: model EEF pose in world frame from LIBERO raw_obs.
      gripper:     physical gripper width scalar saved/trained in the dataset.

    This mirrors the non-instant LIBERO evaluator: first back-project to world,
    then express the cloud in the model EEF frame, then pass an identity UMI state
    to the inference wrapper.  The current world_pose9 is kept only for converting
    the predicted UMI trajectory back to world/controller targets.
    """
    pc_eff, pc_world, pose9_gripper = observation_to_point_clouds(
        env,
        raw_obs,
        pointcloud_camera_names_from_config(cfg),
        int(cfg["observation_height"]),
        int(cfg["observation_width"]),
        int(cfg["num_points"]),
        seed=int(seed),
    )
    world_pose9 = np.asarray(pose9_gripper[:9], dtype=np.float32)
    gripper = float(pose9_gripper[-1])
    if bool(cfg.get("add_gripper_cloud", True)):
        gripper_max_width = float(cfg.get("gripper_qpos_max_width", 0.08))
        pc_eff = add_world_gripper_cloud_to_point_cloud(
            pc_world,
            pose9_gripper,
            gripper_width_percent_from_scalar(gripper, max_physical_width=gripper_max_width),
            total_points=int(cfg["num_points"]),
            gripper_points=int(cfg.get("gripper_points", 500)),
            gripper_len=float(cfg.get("gripper_len", 0.06)),
            gripper_template=str(cfg.get("gripper_template", "reap")),
            seed=int(seed),
            drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
            shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
        )
    return np.ascontiguousarray(pc_eff, dtype=np.float32), world_pose9, gripper


def _axis_points(origin: np.ndarray, rot: np.ndarray, *, scale: float = 0.04, samples: int = 12) -> tuple[np.ndarray, np.ndarray]:
    axes = [
        (rot[:, 0], np.asarray([255, 0, 0], dtype=np.uint8)),
        (rot[:, 1], np.asarray([0, 255, 0], dtype=np.uint8)),
        (rot[:, 2], np.asarray([0, 80, 255], dtype=np.uint8)),
    ]
    points = []
    colors = []
    alpha = np.linspace(0.0, scale, samples, dtype=np.float32)
    for direction, color in axes:
        points.append(origin[None, :] + alpha[:, None] * direction[None, :])
        colors.append(np.tile(color[None, :], (samples, 1)))
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def _rgb_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return colors.reshape(-1, 3).astype(np.uint8)
    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)

def write_umi_debug_ply(path: Path, action_chunk: np.ndarray, point_cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    scene_points = point_cloud[:, :3]
    scene_colors = _rgb_uint8(point_cloud[:, 3:6])

    frame_points = []
    frame_colors = []
    poses = pose9_to_homo_np(action_chunk[:, :9])
    for pose in poses:
        pts, cols = _axis_points(pose[:3, 3], pose[:3, :3])
        frame_points.append(pts)
        frame_colors.append(cols)

    if frame_points:
        all_points = np.concatenate([scene_points, *frame_points], axis=0)
        all_colors = np.concatenate([scene_colors, *frame_colors], axis=0)
    else:
        all_points = scene_points
        all_colors = scene_colors

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(all_points, all_colors):
            f.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

def get_sim(env: Any):
    current = env
    for _ in range(5):
        if hasattr(current, "sim"):
            return current.sim
        if hasattr(current, "env"):
            current = current.env
        else:
            break
    raise AttributeError("Could not find env.sim")


def _refresh_robot_and_observables(env: Any) -> None:
    """Refresh robosuite caches after direct qpos writes / IK without env.step().

    Directly editing sim.data.qpos bypasses robosuite's normal env.step() path.
    Without forcing observable updates, _get_observations() can return cached
    images / EEF pose from the first frame, which makes the next model call use
    stale point clouds even though the MuJoCo state has moved.
    """
    try:
        sim = get_sim(env)
        sim.forward()
    except Exception:
        pass

    for obj in _iter_env_chain(env):
        # Make controller.ee_pos / ee_ori_mat consistent with the just-written qpos.
        for robot in getattr(obj, "robots", []) or []:
            controller = getattr(robot, "controller", None)
            if controller is not None:
                update = getattr(controller, "update", None)
                if callable(update):
                    try:
                        update(force=True)
                    except TypeError:
                        try:
                            update()
                        except Exception:
                            pass
                    except Exception:
                        pass
            # Some robosuite versions expose explicit robot observable updates.
            for name in ("update", "update_observables"):
                fn = getattr(robot, name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        try:
                            fn(force=True)
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Clear common observation caches defensively; force_update below is the
        # main fix, but this makes older robosuite / wrapper versions behave too.
        for cache_name in ("_obs_cache", "obs_cache", "_observations_cache"):
            cache = getattr(obj, cache_name, None)
            if isinstance(cache, dict):
                cache.clear()
        observables = getattr(obj, "_observables", None) or getattr(obj, "observables", None)
        if isinstance(observables, dict):
            for observable in observables.values():
                # Reset cached observed value when the API exposes it.
                reset = getattr(observable, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass
                for attr in ("_current_observed_value", "_obs", "_cached_value"):
                    if hasattr(observable, attr):
                        try:
                            setattr(observable, attr, None)
                        except Exception:
                            pass



def _iter_env_chain(env: Any, max_depth: int = 20) -> list[Any]:
    """Return env plus wrapped inner envs, outermost first."""
    out: list[Any] = []
    current = env
    for _ in range(int(max_depth)):
        if current is None or any(current is item for item in out):
            break
        out.append(current)
        next_env = getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return out


def get_raw_obs(env: Any, *, force_update: bool = True) -> dict[str, Any]:
    """Fetch a fresh LIBERO/robosuite observation after direct simulator edits.

    Important: the fast runner does not use env.step() for arm execution.  Robosuite
    observables are cached unless _get_observations(force_update=True) is used.
    Calling _get_observations() without force_update can keep returning the reset
    frame, causing observation_to_point_clouds() to see the initial point cloud
    forever.
    """
    _refresh_robot_and_observables(env)
    last_exc: Exception | None = None
    for obj in _iter_env_chain(env):
        fn = getattr(obj, "_get_observations", None)
        if not callable(fn):
            continue
        if force_update:
            try:
                return fn(force_update=True)
            except TypeError as exc:
                last_exc = exc
            except Exception as exc:
                last_exc = exc
                # Try inner envs before giving up.
                continue
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            continue
    # Fallback: a zero step is avoided by default; reset obs is passed around by caller.
    raise AttributeError("Environment does not expose _get_observations().") from last_exc
def name_to_id(model: Any, kind: str, name: str) -> int:
    for method in (f"{kind}_name2id", "name2id"):
        fn = getattr(model, method, None)
        if callable(fn):
            try:
                if method == "name2id":
                    return int(fn(name, kind))
                return int(fn(name))
            except Exception:
                pass
    names = model_names(model, kind)
    if name not in names:
        raise KeyError(f"Unknown {kind} name: {name}")
    return names.index(name)

def model_names(model: Any, kind: str) -> list[str]:
    # robosuite MjModel exposes body_names / joint_names / site_names; mujoco-py
    # and dm-control wrappers differ slightly. Keep this very defensive.
    attr = f"{kind}_names"
    names = getattr(model, attr, None)
    if names is not None:
        return [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    n = int(getattr(model, f"n{kind[:3]}", 0) or getattr(model, f"n{kind}", 0) or 0)
    out = []
    for i in range(n):
        try:
            out.append(getattr(model, f"{kind}_id2name")(i))
        except Exception:
            pass
    return out


def _joint_type_name(model: Any, joint_id: int) -> str:
    joint_type = int(model.jnt_type[int(joint_id)])
    return {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(joint_type, str(joint_type))


def robot_arm_indices(env: Any) -> tuple[list[int], list[int]]:
    sim = get_sim(env)
    model = sim.model
    robot = getattr(env, "robots", [None])[0]
    for pos_attr, vel_attr in (
        ("_ref_joint_pos_indexes", "_ref_joint_vel_indexes"),
        ("ref_joint_pos_indexes", "ref_joint_vel_indexes"),
    ):
        if robot is not None and hasattr(robot, pos_attr) and hasattr(robot, vel_attr):
            qpos = list(map(int, getattr(robot, pos_attr)))
            qvel = list(map(int, getattr(robot, vel_attr)))
            if qpos and qvel:
                return qpos, qvel
    joint_names = []
    if robot is not None:
        for attr in ("robot_joints", "joints"):
            value = getattr(robot, attr, None)
            if value:
                joint_names.extend(list(value))
    if not joint_names:
        joint_names = [n for n in model_names(model, "joint") if re.search(r"robot|panda|joint", n, re.I)]
    qpos: list[int] = []
    qvel: list[int] = []
    for name in joint_names:
        if re.search(r"gripper|finger|knuckle", name, re.I):
            continue
        try:
            jid = name_to_id(model, "joint", name)
            if int(model.jnt_type[jid]) != 3:  # hinge joints only; free=0, ball=1, slide=2, hinge=3
                continue
            qpos.append(int(model.jnt_qposadr[jid]))
            qvel.append(int(model.jnt_dofadr[jid]))
        except Exception:
            continue
    if not qpos:
        raise RuntimeError("Could not infer robot arm joint indexes for IK.")
    return qpos, qvel

# -----------------------------------------------------------------------------
# Gripper and object attach logic
# -----------------------------------------------------------------------------

_GRIPPER_JOINT_CACHE: dict[int, list[dict[str, Any]]] = {}

def _joint_record(model: Any, name: str) -> dict[str, Any] | None:
    try:
        jid = name_to_id(model, "joint", str(name))
        return {
            "name": str(name),
            "joint_id": int(jid),
            "type": _joint_type_name(model, int(jid)),
            "qpos_adr": int(model.jnt_qposadr[jid]),
            "qvel_adr": int(model.jnt_dofadr[jid]),
        }
    except Exception:
        return None


def _unique_joint_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        qpos_adr = int(rec["qpos_adr"])
        if qpos_adr in seen:
            continue
        seen.add(qpos_adr)
        out.append(rec)
    return out


def gripper_joint_records(env: Any) -> list[dict[str, Any]]:
    """Return the actual left/right finger slide joints for direct qpos writing.

    The previous implementation trusted robosuite's ref gripper indexes first.
    In some wrappers that list can contain only one actuated side, so directly
    writing it makes only one visual finger move.  For instant simulation we want
    a physical width, so we explicitly prefer the two Panda finger slide joints
    from the MuJoCo model, then fall back to robosuite metadata.
    """
    cache_key = id(env)
    cached = _GRIPPER_JOINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    sim = get_sim(env)
    model = sim.model
    joint_names = model_names(model, "joint")
    robot = getattr(env, "robots", [None])[0]

    # Best path for Panda / LIBERO: exact slide joints for the two fingers.
    exact_name_patterns = [
        r"(^|_)finger_joint1$",
        r"(^|_)finger_joint2$",
        r"left.*finger.*joint",
        r"right.*finger.*joint",
        r"finger.*left.*joint",
        r"finger.*right.*joint",
    ]
    records: list[dict[str, Any]] = []
    for pattern in exact_name_patterns:
        for name in joint_names:
            if not re.search(pattern, str(name), flags=re.I):
                continue
            rec = _joint_record(model, str(name))
            if rec is not None and rec["type"] == "slide":
                records.append(rec)
    records = _unique_joint_records(records)
    if len(records) >= 2:
        records = sorted(records, key=lambda r: str(r["name"]))[:2]
        _GRIPPER_JOINT_CACHE[cache_key] = records
        return records

    # Next: any slide joint whose name is finger-ish, avoiding passive pads/knuckles.
    records = []
    for name in joint_names:
        lname = str(name).lower()
        if "finger" not in lname:
            continue
        if re.search(r"knuckle|hinge|pad|tip|visual", lname):
            continue
        rec = _joint_record(model, str(name))
        if rec is not None and rec["type"] == "slide":
            records.append(rec)
    records = _unique_joint_records(records)
    if len(records) >= 2:
        records = sorted(records, key=lambda r: str(r["name"]))[:2]
        _GRIPPER_JOINT_CACHE[cache_key] = records
        return records

    # Fallback to robosuite robot metadata, then augment with model slide joints.
    metadata_records: list[dict[str, Any]] = []
    if robot is not None:
        for attr in (
            "_ref_gripper_joint_pos_indexes",
            "ref_gripper_joint_pos_indexes",
            "gripper_joint_pos_indexes",
            "_gripper_joint_pos_indexes",
        ):
            value = getattr(robot, attr, None)
            if value is None:
                continue
            qpos_set = {int(v) for v in np.asarray(value).reshape(-1).tolist()}
            for name in joint_names:
                rec = _joint_record(model, str(name))
                if rec is not None and int(rec["qpos_adr"]) in qpos_set:
                    metadata_records.append(rec)
            if metadata_records:
                break

    augmented = _unique_joint_records(metadata_records + records)
    if augmented:
        _GRIPPER_JOINT_CACHE[cache_key] = augmented
        return augmented

    _GRIPPER_JOINT_CACHE[cache_key] = []
    return []


# -----------------------------------------------------------------------------
# MuJoCo viewer sync
# -----------------------------------------------------------------------------


def unwrap_base_env(env: Any) -> Any:
    base_env = env
    for _ in range(20):
        next_env = getattr(base_env, "env", None)
        if next_env is None or next_env is base_env:
            break
        base_env = next_env
    return base_env



def render_mujoco_viewer(env: Any, cfg: dict[str, Any], step: int, *, force: bool = False) -> None:
    mode = str(cfg.get("render_mode", "none")).lower()
    if mode not in {"viewer3d", "mujoco", "onscreen", "headed", "human"}:
        return
    every = int(cfg.get("render_every_n_steps", 1) or 1)
    if every <= 0:
        return
    if not force and int(step) % every != 0:
        return

    if mode in {"viewer3d", "mujoco"}:
        try:
            attach_mujoco_3d_viewer(env, render_camera=cfg.get("render_camera", "agentview"))
        except Exception as exc:
            print(f"[warn] attach_mujoco_3d_viewer failed: {exc!r}")

    base_env = unwrap_base_env(env)
    viewer = getattr(base_env, "viewer", None)
    if viewer is not None and hasattr(viewer, "render"):
        try:
            viewer.render()
            time.sleep(1.0 / 60.0)
            return
        except Exception as exc:
            print(f"[warn] viewer.render failed: {exc!r}")

    render_fn = getattr(env, "render", None)
    if callable(render_fn):
        try:
            render_fn()
            time.sleep(1.0 / 60.0)
        except Exception as exc:
            print(f"[warn] env.render failed: {exc!r}")


def settle_scene_after_reset(
    env: Any,
    *,
    steps: int,
    keep_robot_fixed: bool = True,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Let objects settle before the first policy inference without env.step().

    The fast runner moves the robot by direct IK/qpos writes instead of robosuite
    env.step().  After env.reset(), some LIBERO objects may still be slightly
    above the table or have transient velocity.  If the policy starts immediately,
    the first point cloud can represent a scene that has not settled yet.  This
    helper advances only MuJoCo physics for a few ticks, while optionally restoring
    the robot arm and gripper qpos so the robot itself does not drift before
    policy execution starts.
    """
    sim = get_sim(env)
    steps = max(0, int(steps))
    if steps <= 0:
        return get_raw_obs(env, force_update=True)

    arm_qpos_idx: list[int] = []
    arm_qvel_idx: list[int] = []
    gripper_records: list[dict[str, Any]] = []
    if bool(keep_robot_fixed):
        try:
            arm_qpos_idx, arm_qvel_idx = robot_arm_indices(env)
        except Exception:
            arm_qpos_idx, arm_qvel_idx = [], []
        try:
            gripper_records = gripper_joint_records(env)
        except Exception:
            gripper_records = []

    fixed_arm_qpos = np.asarray(sim.data.qpos[arm_qpos_idx], dtype=np.float64).copy() if arm_qpos_idx else None
    fixed_gripper_qpos = {int(rec["qpos_adr"]): float(sim.data.qpos[int(rec["qpos_adr"])]) for rec in gripper_records}

    def _restore_robot() -> None:
        if fixed_arm_qpos is not None and arm_qpos_idx:
            sim.data.qpos[arm_qpos_idx] = fixed_arm_qpos
        if arm_qvel_idx and hasattr(sim.data, "qvel"):
            sim.data.qvel[arm_qvel_idx] = 0.0
        if fixed_gripper_qpos:
            for qpos_adr, value in fixed_gripper_qpos.items():
                sim.data.qpos[int(qpos_adr)] = float(value)
        if gripper_records and hasattr(sim.data, "qvel"):
            for rec in gripper_records:
                qvel_adr = int(rec.get("qvel_adr", -1))
                if 0 <= qvel_adr < len(sim.data.qvel):
                    sim.data.qvel[qvel_adr] = 0.0

    sim.forward()
    if bool(keep_robot_fixed):
        _restore_robot()
    for settle_step in range(steps):
        if hasattr(sim.data, "ctrl"):
            try:
                sim.data.ctrl[:] = 0.0
            except Exception:
                pass
        if bool(keep_robot_fixed):
            _restore_robot()
        try:
            sim.step()
        except Exception:
            break
        if bool(keep_robot_fixed):
            _restore_robot()
    sim.forward()
    raw_obs = get_raw_obs(env, force_update=True)
    return raw_obs


def run_episode(
    *,
    infer: SmolVLA_ModelInference,
    env: Any,
    task_language: str,
    init_state: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    control = cfg["control"]
    max_steps = int(control.get("max_steps", getattr(env, "horizon", 500)))
    action_index = max(0, int(control.get("action_index", 0)))
    exec_action_steps = int(control.get("exec_action_steps", 16))
    warmup_steps = max(0, int(control.get("warmup_steps", 0)))
    gripper_threshold = float(control.get("gripper_threshold", 0.5))
    gripper_max_width = float(cfg.get("gripper_qpos_max_width", 0.08))


    raw_obs = env.reset()
    raw_obs = env.set_init_state(init_state)

    raw_obs = settle_scene_after_reset(
        env,
        steps=int(cfg['settle_steps']),
        keep_robot_fixed=bool(cfg['settle_keep_robot_fixed']),
        cfg=cfg,
    )


    # Absolute pose execution only.
    for robot in env.robots:
        robot.controller.use_delta = False


    for _ in range(warmup_steps):
        raw_obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    infer.policy.reset()
    infer.policy_reset()

    pc_camera_names = pointcloud_camera_names_from_config(cfg)
    save_video = bool(cfg.get("save_video", True))
    video_frames: dict[str, list[np.ndarray]] = {}
    if save_video:
        append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))

    rewards: list[float] = []
    libero_actions: list[np.ndarray] = []
    model_action_rows: list[np.ndarray] = []
    target_controller_pose9: list[np.ndarray] = []
    target_model_worlds: list[np.ndarray] = []
    gripper_commands: list[float] = []
    gripper_raw_widths: list[float] = []
    gripper_width_pcts: list[float] = []

    success_ever = False
    done = False
    model_call_count = 0
    steps = 0
    start_s = time.perf_counter()

    while steps < max_steps and not done and not success_ever:
        point_cloud, point_cloud_world, eef_pose = observation_to_point_clouds(
            env,
            raw_obs,
            pc_camera_names,
            int(cfg["observation_height"]),
            int(cfg["observation_width"]),
            int(cfg["num_points"]),
            seed=steps,
        )
        if bool(cfg.get("add_gripper_cloud", True)):
            point_cloud = add_world_gripper_cloud_to_point_cloud(
                point_cloud_world,
                eef_pose,
                gripper_width_percent_from_scalar(float(eef_pose[-1]), max_physical_width=gripper_max_width),
                total_points=int(cfg["num_points"]),
                gripper_points=int(cfg.get("gripper_points", 500)),
                gripper_len=float(cfg.get("gripper_len", 0.06)),
                gripper_template=str(cfg.get("gripper_template", "reap")),
                seed=steps,
                drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
                shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
            )

        chunk = infer.predict_action_chunk_obs(
            {"point_cloud": point_cloud, "state": identity_pose9_gripper(float(eef_pose[-1]))},
            task=task_language,
            postprocess=True,
            state_pose_mode="identity",
        )[0].detach().cpu().numpy()
        model_call_count += 1

        start_idx = min(action_index, max(0, len(chunk) - 1))
        end_idx = len(chunk) if exec_action_steps <= 0 else min(len(chunk), start_idx + exec_action_steps)
        selected_chunk = np.asarray(chunk[start_idx:end_idx], dtype=np.float32)

        actions, model_worlds, controller_pose9 = action_chunk_to_absolute_libero_actions(
            env=env,
            current_eef_pose9_gripper=eef_pose,
            action_chunk=selected_chunk,
            gripper_threshold=gripper_threshold,
            gripper_max_width=gripper_max_width,
        )

        for row, action, model_world, controller_pose in zip(
            selected_chunk,
            actions,
            model_worlds,
            controller_pose9,
            strict=True,
        ):
            if steps >= max_steps or done or success_ever:
                break
            try:
                raw_obs, reward, done, _ = env.step(action)
            except ValueError as exc:
                if "terminated episode" in str(exc):
                    done = True
                    break
                raise

            steps += 1
            render_viewer3d(env, cfg, steps)
            reward = float(reward)
            rewards.append(reward)
            libero_actions.append(np.asarray(action, dtype=np.float32))
            model_action_rows.append(np.asarray(row, dtype=np.float32))
            target_model_worlds.append(np.asarray(model_world, dtype=np.float32))
            target_controller_pose9.append(np.asarray(controller_pose, dtype=np.float32))
            gripper_commands.append(float(action[-1]))
            gripper_raw_widths.append(float(row[-1]))
            gripper_width_pcts.append(
                gripper_width_percent_from_scalar(float(row[-1]), max_physical_width=gripper_max_width)
            )

            if save_video:
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))

            try:
                success_ever = success_ever or bool(env.check_success())
            except Exception:
                success_ever = success_ever or bool(reward > 0.0)

    final_eef_pose = eef_pose9_gripper_from_obs(raw_obs)
    return {
        "success": bool(success_ever),
        "done": bool(done),
        "steps": int(steps),
        "action_rows_executed": int(len(libero_actions)),
        "model_call_count": int(model_call_count),
        "sum_reward": float(np.sum(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "wall_s": float(time.perf_counter() - start_s),
        "gripper_threshold": float(gripper_threshold),
        "gripper_open_steps": int(np.sum(np.asarray(gripper_commands) < 0.0)),
        "gripper_close_steps": int(np.sum(np.asarray(gripper_commands) > 0.0)),
        "final_gripper_qpos_sum": float(gripper_scalar(raw_obs)),
        "final_eef_pose9_gripper": final_eef_pose,
        "libero_actions": np.asarray(libero_actions, dtype=np.float32),
        "model_action_rows": np.asarray(model_action_rows, dtype=np.float32),
        "target_model_worlds": np.asarray(target_model_worlds, dtype=np.float32),
        "target_controller_pose9": np.asarray(target_controller_pose9, dtype=np.float32),
        #"gripper_commands": np.asarray(gripper_commands, dtype=np.float32),
        #"gripper_raw_widths": np.asarray(gripper_raw_widths, dtype=np.float32),
        #"gripper_width_pcts": np.asarray(gripper_width_pcts, dtype=np.float32),
        "video_frames": video_frames,
    }


def prepare_config(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], Path]:
    cfg = load_config(args.config)
    cfg.setdefault("control", {})

    cfg["policy_path"] = cfg_get(cfg, args.policy_path, "policy_path")
    cfg["policy_repo_id"] = cfg_get(cfg, args.policy_repo_id, "policy_repo_id")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes", 1))
    cfg["device"] = cfg_get(cfg, args.device, "device", "cuda")
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points", 4096))
    cfg["observation_height"] = int(cfg_get(cfg, args.observation_height, "observation_height", 128))
    cfg["observation_width"] = int(cfg_get(cfg, args.observation_width, "observation_width", 128))
    cfg["render_mode"] = cfg_get(cfg, args.render_mode, "render_mode", "offscreen")
    cfg["render_camera"] = cfg_get(cfg, args.render_camera, "render_camera", "agentview")
    cfg["render_every_n_steps"] = int(cfg_get(cfg, args.render_every_n_steps, "render_every_n_steps", 1))
    cfg["render_gpu_device_id"] = int(cfg_get(cfg, args.render_gpu_device_id, "render_gpu_device_id", -1))
    cfg["save_video"] = bool(cfg_get(cfg, args.save_video, "save_video", True))
    cfg["add_gripper_cloud"] = bool(cfg_get(cfg, args.add_gripper_cloud, "add_gripper_cloud", True))
    if args.gripper_points is not None:
        cfg["gripper_points"] = int(args.gripper_points)
    cfg.setdefault("camera_names", ["agentview", "robot0_eye_in_hand"])

    cfg["control"]["control_freq"] = float(
        cfg_get(cfg["control"], args.control_freq, "control_freq", cfg.get("control_freq", 20.0))
    )
    cfg["control"]["action_index"] = int(cfg_get(cfg["control"], args.action_index, "action_index", 0))
    cfg["control"]["exec_action_steps"] = int(cfg_get(cfg["control"], args.exec_action_steps, "exec_action_steps", 16))
    cfg["control"]["max_steps"] = int(cfg_get(cfg["control"], args.max_steps, "max_steps", 500))
    cfg["control"]["warmup_steps"] = int(cfg_get(cfg["control"], args.warmup_steps, "warmup_steps", 0))
    cfg["control"]["gripper_threshold"] = float(
        cfg_get(cfg["control"], args.gripper_threshold, "gripper_threshold", 0.5)
    )
    if args.gripper_qpos_max_width is not None:
        cfg["gripper_qpos_max_width"] = float(args.gripper_qpos_max_width)
    cfg.setdefault("gripper_qpos_max_width", 0.08)

    suite_names = resolve_suite_names(args.suite, cfg)
    cfg["suites"] = suite_names
    cfg["all_tasks"] = bool(args.all_tasks if args.all_tasks is not None else cfg.get("all_tasks", False))
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")

    output_dir = Path(cfg_get(cfg, args.output_dir, "output_dir", BENCHMARK_ROOT / "outputs" / "libero_setting" / "eval")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    cfg["recreate_env_per_episode"] = bool(
        cfg_get(cfg, args.recreate_env_per_episode, "recreate_env_per_episode", False)
    )
    cfg["settle_steps"] = args.settle_steps
    cfg["settle_keep_robot_fixed"] = args.settle_keep_robot_fixed

    return cfg, suite_names, output_dir


def main() -> None:
    args = parse_args()
    cfg, suite_names, output_dir = prepare_config(args)
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark

    infer = SmolVLA_ModelInference(
        policy_path=cfg["policy_path"],
        policy_repo_id=cfg.get("policy_repo_id"),
        device=cfg["device"],
    )

    print(
        "[info] clean absolute-pose eval: "
        f"suites={suite_names}, episodes={cfg['episodes']}, "
        f"exec_action_steps={cfg['control']['exec_action_steps']}, "
        f"gripper_threshold={cfg['control']['gripper_threshold']}, "
        f"save_video={cfg['save_video']}, "
        f"render_mode={cfg.get('render_mode')}, "
        f"render_every_n_steps={cfg.get('render_every_n_steps')}"
    )

    all_task_summaries: list[dict[str, Any]] = []
    benchmark_dict = benchmark.get_benchmark_dict()

    for suite_name in suite_names:
        suite = benchmark_dict[suite_name]()
        task_ids = resolve_task_ids_for_suite(
            suite_name=suite_name,
            task_count=len(suite.tasks),
            cli_task_ids=args.task_id,
            cfg=cfg,
        )

        for task_id in task_ids:
            init_states = get_task_init_states(suite, int(task_id))
            task_results: list[dict[str, Any]] = []
            task_name = f"task_{int(task_id):03d}"
            task_language = f"{suite_name}:{int(task_id)}"

            shared_env = None
            shared_task = None
            try:
                if not bool(cfg.get("recreate_env_per_episode", False)):
                    shared_env, shared_task = make_libero_env(
                        suite,
                        int(task_id),
                        int(cfg["observation_height"]),
                        int(cfg["observation_width"]),
                        render_camera_names_from_config(cfg),
                        render_mode=str(cfg.get("render_mode", "offscreen")),
                        render_camera=str(cfg.get("render_camera", "agentview")),
                        render_gpu_device_id=int(cfg.get("render_gpu_device_id", -1)),
                        control_delta=False,
                        control_freq=float(cfg["control"].get("control_freq", 20.0)),
                    )
                    task_name = str(getattr(shared_task, "name", task_name))
                    task_language = str(getattr(shared_task, "language", task_language))

                for episode_idx in range(int(cfg["episodes"])):
                    episode_dir = output_dir / suite_name / f"task_{int(task_id):03d}" / f"episode_{episode_idx:03d}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[eval] start suite={suite_name} task={task_id} episode={episode_idx}", flush=True)

                    env = shared_env
                    task = shared_task
                    if bool(cfg.get("recreate_env_per_episode", False)):
                        env, task = make_libero_env(
                            suite,
                            int(task_id),
                            int(cfg["observation_height"]),
                            int(cfg["observation_width"]),
                            render_camera_names_from_config(cfg),
                            render_mode=str(cfg.get("render_mode", "offscreen")),
                            render_camera=str(cfg.get("render_camera", "agentview")),
                            render_gpu_device_id=int(cfg.get("render_gpu_device_id", -1)),
                            control_delta=False,
                            control_freq=float(cfg["control"].get("control_freq", 20.0)),
                        )
                        task_name = str(getattr(task, "name", task_name))
                        task_language = str(getattr(task, "language", task_language))

                    try:
                        assert env is not None
                        result = run_episode(
                            infer=infer,
                            env=env,
                            task_language=task_language,
                            init_state=init_states[episode_idx % len(init_states)],
                            cfg=cfg,
                        )


                        action_npz = save_episode_actions(result, episode_dir)
                        episode_record = compact_episode_record(result, episode_idx, action_npz)

                        if bool(cfg.get("save_video", True)):
                            video_record = {
                                "episode_index": int(episode_idx),
                                "demo_name": "rollout",
                                "video_dir_name": episode_dir.name,
                            }
                            video_paths = export_episode_videos(result, episode_dir.parent, video_record, cfg)
                            if video_paths:
                                episode_record["videos"] = video_paths

                        write_json_atomic(episode_dir / "result.json", episode_record)
                        task_results.append(episode_record)
                        print(
                            f"[eval] done suite={suite_name} task={task_id} episode={episode_idx} "
                            f"success={episode_record['success']} steps={episode_record['steps']} "
                            f"model_calls={episode_record['model_call_count']} "
                            f"sum_reward={episode_record['sum_reward']:.3f}",
                            flush=True,
                        )
                    except Exception as exc:
                        failure = {
                            "episode_index": int(episode_idx),
                            "success": False,
                            "steps": 0,
                            "model_call_count": 0,
                            "sum_reward": 0.0,
                            "max_reward": 0.0,
                            "error": repr(exc),
                        }
                        write_json_atomic(episode_dir / "result.json", failure)
                        write_json_atomic(episode_dir / "error.json", failure)
                        task_results.append(failure)
                        print(
                            f"[warn] failed suite={suite_name} task={task_id} episode={episode_idx}: {exc!r}",
                            flush=True,
                        )
                    finally:
                        if bool(cfg.get("recreate_env_per_episode", False)) and env is not None:
                            try:
                                env.close()
                            except Exception:
                                pass

                    current_task_summary = make_task_summary(
                        suite_name=suite_name,
                        task_id=int(task_id),
                        task_name=task_name,
                        task_language=task_language,
                        episodes=task_results,
                    )
                    write_eval_reports(output_dir, cfg, suite_names, [*all_task_summaries, current_task_summary])

            finally:
                if shared_env is not None:
                    try:
                        shared_env.close()
                    except Exception:
                        pass

            all_task_summaries.append(
                make_task_summary(
                    suite_name=suite_name,
                    task_id=int(task_id),
                    task_name=task_name,
                    task_language=task_language,
                    episodes=task_results,
                )
            )

    write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
    print(json.dumps(aggregate_task_results(all_task_summaries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
