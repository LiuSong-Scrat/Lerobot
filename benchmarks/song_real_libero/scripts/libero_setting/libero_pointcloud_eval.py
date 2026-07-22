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
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import queue
import re
import select
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

_ENV_CREATION_LOCK = threading.Lock()
_ENV_WORKER_BOOTSTRAP = os.environ.get("SONG_LIBERO_ENV_WORKER", "0") == "1"
_ISOLATED_POLICY_WORKER_BOOTSTRAP = (
    os.environ.get("SONG_LIBERO_ISOLATED_POLICY_WORKER", "0") == "1"
)
_SUITE_LAUNCHER_BOOTSTRAP = any(
    arg == "--suite-gpu-ids" or arg.startswith("--suite-gpu-ids=") for arg in sys.argv[1:]
)
# This must be set before the policy import performs the first CUDA operation.
# PyTorch uses it when deterministic cuBLAS execution is requested below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LIBERO_STANDARD_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _identity_pose9_gripper(gripper: float = 0.0) -> np.ndarray:
    state = np.zeros(10, dtype=np.float32)
    state[3] = 1.0
    state[7] = 1.0
    state[-1] = float(gripper)
    return state

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import BENCHMARK_ROOT, DEFAULT_LIBERO_CONFIG, load_json_config
    if (
        not _ENV_WORKER_BOOTSTRAP
        and not _ISOLATED_POLICY_WORKER_BOOTSTRAP
        and not _SUITE_LAUNCHER_BOOTSTRAP
    ):
        from ..smolvla_model_inference import SmolVLA_ModelInference, identity_pose9_gripper
    else:
        SmolVLA_ModelInference = Any
        identity_pose9_gripper = _identity_pose9_gripper
    from .libero_pointcloud_utils import (
        add_world_gripper_cloud_to_point_cloud,
        attach_mujoco_3d_viewer,
        eef_pose9_gripper_from_obs,
        ensure_libero_config,
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
    from libero_setting.libero_pointcloud_utils import (
        add_world_gripper_cloud_to_point_cloud,
        attach_mujoco_3d_viewer,
        eef_pose9_gripper_from_obs,
        ensure_libero_config,
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
    if (
        not _ENV_WORKER_BOOTSTRAP
        and not _ISOLATED_POLICY_WORKER_BOOTSTRAP
        and not _SUITE_LAUNCHER_BOOTSTRAP
    ):
        from smolvla_model_inference import SmolVLA_ModelInference, identity_pose9_gripper
    else:
        SmolVLA_ModelInference = Any
        identity_pose9_gripper = _identity_pose9_gripper


def resolve_suite_names(cli_suites: list[str] | None, cfg: dict[str, Any]) -> list[str]:
    suites = cli_suites if cli_suites is not None else cfg.get("suites", cfg.get("suite", "libero_object"))
    if isinstance(suites, str):
        suites = [suites]
    suites = [str(suite) for suite in suites]
    if not suites:
        raise ValueError("At least one LIBERO suite must be configured.")
    return suites


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_evaluation_identity(policy_path: str | Path | None) -> dict[str, Any]:
    """Capture enough immutable identity to compare local and server runs."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]
    policy = Path(policy_path).expanduser() if policy_path is not None else None
    resolved_policy = policy.resolve() if policy is not None and policy.exists() else policy
    model_path = resolved_policy / "model.safetensors" if resolved_policy is not None else None
    config_path = resolved_policy / "config.json" if resolved_policy is not None else None

    identity: dict[str, Any] = {
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "policy_path_requested": None if policy is None else str(policy),
        "policy_path_resolved": None if resolved_policy is None else str(resolved_policy),
        "eval_script_sha256": _sha256_file(script_path),
        "inference_wrapper_sha256": _sha256_file(script_path.parent.parent / "smolvla_model_inference.py"),
        "pointcloud_utils_sha256": _sha256_file(script_path.with_name("libero_pointcloud_utils.py")),
        "modeling_smolvla_sha256": _sha256_file(
            repo_root / "src" / "lerobot" / "policies" / "smolvla" / "modeling_smolvla.py"
        ),
        "song_pointseg_sha256": _sha256_file(
            repo_root / "src" / "lerobot" / "policies" / "smolvla" / "song_pointseg.py"
        ),
        "smolvlm_with_expert_sha256": _sha256_file(
            repo_root / "src" / "lerobot" / "policies" / "smolvla" / "smolvlm_with_expert.py"
        ),
        "policy_config_sha256": _sha256_file(config_path) if config_path is not None else None,
    }
    package_versions: dict[str, str] = {}
    for distribution in ("torch", "spconv-cu118", "spconv-cu120", "mujoco", "robosuite", "libero"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    identity["package_versions"] = package_versions
    try:
        identity["gpu_inventory"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if model_path is not None and model_path.is_file():
        stat = model_path.stat()
        identity.update(
            {
                "model_size_bytes": int(stat.st_size),
                "model_mtime_ns": int(stat.st_mtime_ns),
            }
        )
        # Suite children do not write a global summary; avoid hashing the same
        # 1.4 GB checkpoint once per GPU child.  The launcher or single process
        # records the full digest exactly once.
        if os.environ.get("SONG_LIBERO_SUITE_WORKER", "0") != "1":
            identity["model_sha256"] = _sha256_file(model_path)
    try:
        identity["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        identity["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        identity["git_commit"] = None
        identity["git_dirty"] = None
    return identity


def resolve_task_ids_for_suite(
    *,
    suite_name: str,
    task_count: int,
    cli_task_ids: list[int] | None,
    cfg: dict[str, Any],
) -> list[int]:
    if bool(cfg.get("all_tasks", False)):
        return list(range(task_count))
    if cli_task_ids is not None:
        task_ids = cli_task_ids
    else:
        suite_task_ids = cfg.get("suite_task_ids", {})
        if isinstance(suite_task_ids, dict) and suite_name in suite_task_ids:
            task_ids = suite_task_ids[suite_name]
        else:
            task_ids = cfg.get("task_ids")
    if task_ids is None:
        return list(range(task_count))
    resolved = [int(task_id) for task_id in task_ids]
    invalid = [task_id for task_id in resolved if task_id < 0 or task_id >= task_count]
    if invalid:
        raise ValueError(
            f"Invalid task id(s) for {suite_name}: {invalid}; valid range is [0, {task_count - 1}]"
        )
    return resolved


def _camera_image_keys(camera_names: list[str]) -> list[str]:
    return [
        str(camera_name)
        if str(camera_name).endswith("_image")
        else f"{camera_name}_image"
        for camera_name in camera_names
    ]


def _image_from_raw_obs(raw_obs: dict[str, Any], image_key: str) -> np.ndarray | None:
    if image_key not in raw_obs:
        return None
    image = np.asarray(raw_obs[image_key])
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dataset_image_from_raw_obs(raw_obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    image_key = str(camera_name) if str(camera_name).endswith("_image") else f"{camera_name}_image"
    image = _image_from_raw_obs(raw_obs, image_key)
    if image is None:
        return None
    if image_key == "agentview_image":
        image = np.ascontiguousarray(image[:, ::-1])
    return image.copy()


def append_video_frames(
    video_frames: dict[str, list[np.ndarray]],
    raw_obs: dict[str, Any],
    camera_names: list[str],
) -> None:
    for image_key in _camera_image_keys(camera_names):
        image = _image_from_raw_obs(raw_obs, image_key)
        if image is not None:
            if image_key == "agentview_image":
                image = np.ascontiguousarray(image[:, ::-1])
            video_frames.setdefault(image_key, []).append(image.copy())


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps)
        return
    except Exception:
        pass

    import cv2

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def export_episode_videos(
    episode: dict[str, Any],
    video_dir: Path,
    record: dict[str, Any],
    cfg: dict[str, Any],
) -> list[str]:
    if not cfg.get("save_video", False):
        return []
    video_frames = episode.get("video_frames") or {}
    if not video_frames:
        return []
    video_dir_name = record.get("video_dir_name")
    if video_dir_name is None:
        video_dir_name = f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"
    episode_dir = video_dir / video_dir_name
    written: list[str] = []
    for image_key, frames in video_frames.items():
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_key)
        path = episode_dir / f"{safe_key}.mp4"
        _write_video(path, frames, int(cfg["fps"]))
        written.append(str(path))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean LIBERO eval: point-cloud observation -> action chunk -> absolute pose execution."
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=None,
        help="Minimum MuJoCo physics ticks after reset before accepting a stable scene (legacy override).",
    )
    parser.add_argument(
        "--settle-min-seconds",
        type=float,
        default=None,
        help="Minimum simulated time before scene stability can be accepted.",
    )
    parser.add_argument(
        "--settle-stable-seconds",
        type=float,
        default=None,
        help="Required continuous low-velocity duration before the first policy inference.",
    )
    parser.add_argument(
        "--settle-max-seconds",
        type=float,
        default=None,
        help="Time before warning that settling is slow; strict mode keeps waiting instead of skipping the episode.",
    )
    parser.add_argument(
        "--settle-linear-velocity-threshold",
        type=float,
        default=None,
        help="Maximum free-object linear speed in m/s for a stable physics tick.",
    )
    parser.add_argument(
        "--settle-angular-velocity-threshold",
        type=float,
        default=None,
        help="Maximum free-object angular speed in rad/s for a stable physics tick.",
    )
    parser.add_argument(
        "--settle-other-dof-velocity-threshold",
        type=float,
        default=None,
        help="Fallback scalar joint-speed threshold when a scene has no free joints.",
    )
    parser.add_argument(
        "--settle-require-stable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep waiting after the settling warning until the scene is stable (default: true).",
    )
    parser.add_argument(
        "--settle-keep-robot-fixed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="During initial settling, restore arm/gripper qpos each sim tick so only scene objects settle.",
    )
    parser.add_argument(
        "--initial-gripper-open",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start every evaluation episode with the gripper at its fully open joint limits.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", default=None)
    parser.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument(
        "--suite-gpu-ids",
        default=None,
        help=(
            "Comma-separated GPU ids for one-process-per-suite evaluation, for example 0,1,2,3. "
            "The number of ids must match the number of suites."
        ),
    )
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--observation-height", type=int, default=None)
    parser.add_argument("--observation-width", type=int, default=None)
    parser.add_argument("--render-mode", choices=("offscreen", "onscreen", "viewer3d"), default="viewer3d")
    parser.add_argument("--render-camera", default=None)
    parser.add_argument("--render-every-n-steps", type=int, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--control-freq", type=float, default=None)

    parser.add_argument("--action-index", type=int, default=None)
    parser.add_argument("--exec-action-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--use-suite-max-steps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the standard LIBERO per-suite horizons (220/280/300/520/400).",
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument(
        "--policy-noise-seed",
        type=int,
        default=None,
        help=(
            "Base seed for per-suite/task/episode/model-call flow-matching noise. "
            "The default is 0 for scheduling-invariant evaluation; use a negative value to restore stochastic noise."
        ),
    )
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use deterministic PyTorch/cuDNN/cuBLAS inference settings (default: true). "
            "Disable only for throughput experiments whose scores will not be compared to official runs."
        ),
    )

    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Physical gripper-width threshold in metres used by absolute_width control.",
    )
    parser.add_argument(
        "--gripper-control-mode",
        choices=("delta_width", "absolute_width", "target_width"),
        default=None,
        help=(
            "Convert predicted widths either from their temporal derivative (legacy evaluator behavior) "
            "by thresholding the absolute physical width, or by tracking the predicted physical width "
            "against the measured gripper width."
        ),
    )
    parser.add_argument(
        "--gripper-delta-threshold",
        type=float,
        default=None,
        help="Physical width-change threshold in metres used by delta_width control.",
    )
    parser.add_argument(
        "--gripper-target-tolerance",
        type=float,
        default=None,
        help="Physical width deadband in metres used by target_width closed-loop control.",
    )
    parser.add_argument("--gripper-qpos-max-width", type=float, default=None)
    parser.add_argument("--add-gripper-cloud", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gripper-points", type=int, default=None)

    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--visualize-foreground",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Continuously show PointSeg foreground probability in a non-blocking Open3D window.",
    )
    parser.add_argument("--foreground-vis-max-points", type=int, default=None)
    parser.add_argument(
        "--task-workers",
        type=int,
        default=None,
        help="Maximum number of distinct tasks evaluated concurrently by this GPU model service.",
    )
    parser.add_argument(
        "--isolated-policy-workers",
        type=int,
        default=None,
        help=(
            "Independent model processes per visible GPU. Each process owns one checkpoint copy and "
            "serially evaluates a disjoint task shard with inference batch size 1. On a 24 GB GPU, "
            "2 workers is the recommended upper bound for this checkpoint."
        ),
    )
    parser.add_argument(
        "--episode-workers-per-task",
        type=int,
        default=None,
        help=(
            "Independent environment processes used to split the episodes of each active task. "
            "For example, 2 assigns even and odd episode indices to separate environments."
        ),
    )
    parser.add_argument(
        "--task-worker-backend",
        choices=("process", "thread"),
        default=None,
        help=(
            "Parallel environment backend. 'process' gives real MuJoCo task parallelism and is the default "
            "when task-workers > 1; 'thread' is retained only for compatibility."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Maximum number of concurrent task requests combined into one policy forward.",
    )
    parser.add_argument(
        "--inference-batch-wait-ms",
        type=float,
        default=None,
        help="Short collection window used by the shared-GPU dynamic batcher.",
    )
    parser.add_argument(
        "--recreate-env-per-episode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rebuild a task environment before every episode (disabled by default for parallel evaluation).",
    )
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


@contextmanager
def interprocess_file_lock(path: Path):
    """Serialize report updates from independent MuJoCo worker processes."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    try:
        with open(path, encoding="utf-8") as json_file:
            value = json.load(json_file)
        return value if isinstance(value, dict) else default
    except (OSError, json.JSONDecodeError):
        return default


def _suite_progress_summary(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": str(progress["suite"]),
        "status": str(progress.get("status", "running")),
        "expected_task_count": int(progress.get("expected_task_count", 0)),
        "expected_episode_count": int(progress.get("expected_episode_count", 0)),
        "completed_episode_count": int(progress.get("completed_episode_count", 0)),
        "success_count": int(progress.get("success_count", 0)),
        "failure_count": int(progress.get("failure_count", 0)),
        "completion_rate": float(progress.get("completion_rate", 0.0)),
        "success_rate": float(progress.get("success_rate", 0.0)),
        "progress_path": str(progress.get("progress_path", "")),
        "error": progress.get("error"),
    }


def _update_root_progress(output_dir: Path, suite_progress: dict[str, Any]) -> None:
    progress_path = output_dir / "progress.json"
    with interprocess_file_lock(output_dir / ".progress.lock"):
        progress = _read_json_or_default(
            progress_path,
            {
                "created_unix_s": time.time(),
                "status": "running",
                "suites": {},
            },
        )
        suites = progress.setdefault("suites", {})
        suites[str(suite_progress["suite"])] = _suite_progress_summary(suite_progress)
        suite_values = list(suites.values())
        expected = int(sum(int(item.get("expected_episode_count", 0)) for item in suite_values))
        completed = int(sum(int(item.get("completed_episode_count", 0)) for item in suite_values))
        success = int(sum(int(item.get("success_count", 0)) for item in suite_values))
        failure = int(sum(int(item.get("failure_count", 0)) for item in suite_values))
        statuses = [str(item.get("status", "running")) for item in suite_values]
        progress.update(
            {
                "updated_unix_s": time.time(),
                "status": (
                    "failed"
                    if "failed" in statuses
                    else "completed"
                    if statuses and all(status == "completed" for status in statuses)
                    else "running"
                ),
                "expected_episode_count": expected,
                "completed_episode_count": completed,
                "success_count": success,
                "failure_count": failure,
                "completion_rate": float(completed / expected) if expected else 0.0,
                "success_rate": float(success / completed) if completed else 0.0,
            }
        )
        write_json_atomic(progress_path, progress)


def initialize_realtime_suite_progress(
    *,
    output_dir: Path,
    suite_name: str,
    task_ids: list[int],
    episodes_per_task: int,
) -> None:
    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    progress_path = suite_dir / "progress.json"
    expected_episode_count = len(task_ids) * int(episodes_per_task)
    progress = {
        "created_unix_s": time.time(),
        "updated_unix_s": time.time(),
        "suite": suite_name,
        "status": "running",
        "expected_task_ids": [int(task_id) for task_id in task_ids],
        "expected_task_count": len(task_ids),
        "episodes_per_task": int(episodes_per_task),
        "expected_episode_count": expected_episode_count,
        "completed_episode_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "completion_rate": 0.0,
        "success_rate": 0.0,
        "progress_path": str(progress_path),
        "tasks": {
            str(int(task_id)): {
                "completed_episode_count": 0,
                "success_count": 0,
                "episodes": {},
            }
            for task_id in task_ids
        },
    }
    with interprocess_file_lock(suite_dir / ".progress.lock"):
        write_json_atomic(progress_path, progress)
        # This file is scoped to one suite and therefore can be reset safely
        # before its environment workers are started.
        with open(suite_dir / "evaluation_events.jsonl", "w", encoding="utf-8"):
            pass
    _update_root_progress(output_dir, progress)


def append_realtime_episode_event(
    *,
    output_dir: Path,
    suite_name: str,
    task_id: int,
    episode_index: int,
    event: str,
    episode_record: dict[str, Any] | None = None,
) -> None:
    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    event_record = {
        "unix_s": time.time(),
        "event": str(event),
        "suite": suite_name,
        "task_id": int(task_id),
        "episode_index": int(episode_index),
    }
    if episode_record is not None:
        event_record["result"] = json_safe(episode_record)

    with interprocess_file_lock(suite_dir / ".events.lock"):
        with open(suite_dir / "evaluation_events.jsonl", "a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(event_record, ensure_ascii=False) + "\n")
            events_file.flush()
            os.fsync(events_file.fileno())


def update_realtime_episode_progress(
    *,
    output_dir: Path,
    suite_name: str,
    task_id: int,
    episode_index: int,
    episode_record: dict[str, Any],
) -> None:
    suite_dir = output_dir / suite_name
    progress_path = suite_dir / "progress.json"
    with interprocess_file_lock(suite_dir / ".progress.lock"):
        progress = _read_json_or_default(
            progress_path,
            {
                "suite": suite_name,
                "status": "running",
                "expected_task_count": 0,
                "expected_episode_count": 0,
                "tasks": {},
            },
        )
        tasks = progress.setdefault("tasks", {})
        task_progress = tasks.setdefault(
            str(int(task_id)),
            {"completed_episode_count": 0, "success_count": 0, "episodes": {}},
        )
        task_progress.setdefault("episodes", {})[str(int(episode_index))] = json_safe(episode_record)

        completed = 0
        success = 0
        for value in tasks.values():
            episodes = list(value.get("episodes", {}).values())
            value["completed_episode_count"] = len(episodes)
            value["success_count"] = sum(bool(item.get("success", False)) for item in episodes)
            completed += int(value["completed_episode_count"])
            success += int(value["success_count"])
        expected = int(progress.get("expected_episode_count", 0))
        progress.update(
            {
                "updated_unix_s": time.time(),
                "status": "completed" if expected > 0 and completed >= expected else "running",
                "completed_episode_count": completed,
                "success_count": success,
                "failure_count": completed - success,
                "completion_rate": float(completed / expected) if expected else 0.0,
                "success_rate": float(success / completed) if completed else 0.0,
                "progress_path": str(progress_path),
            }
        )
        write_json_atomic(progress_path, progress)
    append_realtime_episode_event(
        output_dir=output_dir,
        suite_name=suite_name,
        task_id=task_id,
        episode_index=episode_index,
        event="episode_finished",
        episode_record=episode_record,
    )
    _update_root_progress(output_dir, progress)


def mark_realtime_suite_failed(
    *,
    output_dir: Path,
    suite_name: str,
    error: str,
) -> None:
    suite_dir = output_dir / suite_name
    progress_path = suite_dir / "progress.json"
    with interprocess_file_lock(suite_dir / ".progress.lock"):
        progress = _read_json_or_default(progress_path, {"suite": suite_name, "tasks": {}})
        progress.update({"updated_unix_s": time.time(), "status": "failed", "error": str(error)})
        write_json_atomic(progress_path, progress)
    _update_root_progress(output_dir, progress)


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


class EpisodeKeyboardControl:
    """Collect an immediate `n` request from the viewer or the terminal."""

    def __init__(self) -> None:
        self._manual_failure = threading.Event()
        self._stdin_fd: int | None = None
        self._stdin_attrs: list[Any] | None = None

    def _request_manual_failure(self, source: str) -> None:
        if self._manual_failure.is_set():
            return
        self._manual_failure.set()
        print(
            f"[eval] received 'n' from {source}: mark current episode as failed and continue",
            flush=True,
        )

    def viewer_key_callback(self, keycode: int) -> None:
        """MuJoCo passive-viewer callback (GLFW letter codes are ASCII-compatible)."""
        if int(keycode) in (ord("n"), ord("N")):
            self._request_manual_failure("viewer")

    def start_terminal(self) -> bool:
        """Enable single-key terminal input without requiring Enter."""
        if not sys.stdin.isatty():
            return False
        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            return True
        except (OSError, termios.error):
            self._stdin_fd = None
            self._stdin_attrs = None
            return False

    def poll(self) -> bool:
        """Poll terminal input and return whether this episode must stop."""
        if self._stdin_fd is not None:
            try:
                while select.select([self._stdin_fd], [], [], 0.0)[0]:
                    key = os.read(self._stdin_fd, 1)
                    if not key:
                        break
                    if key.lower() == b"n":
                        self._request_manual_failure("terminal")
            except OSError:
                pass
        return self._manual_failure.is_set()

    def close(self) -> None:
        if self._stdin_fd is not None and self._stdin_attrs is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_attrs)
            except (OSError, termios.error):
                pass
        self._stdin_fd = None
        self._stdin_attrs = None


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
    """Threshold a physical width using the LIBERO -1=open, +1=close convention."""
    width = float(np.clip(float(predicted_width), 0.0, float(max_physical_width)))
    return -1.0 if width >= float(threshold) else 1.0


def gripper_target_width_command(
    predicted_width: float,
    measured_width: float,
    *,
    tolerance: float,
    max_physical_width: float,
) -> float:
    """Track a predicted physical opening using LIBERO's directional command."""
    target = float(np.clip(float(predicted_width), 0.0, float(max_physical_width)))
    measured = float(np.clip(float(measured_width), 0.0, float(max_physical_width)))
    if target > measured + float(tolerance):
        return -1.0
    if target < measured - float(tolerance):
        return 1.0
    return 0.0


def rotation_error_radians(actual: np.ndarray, target: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two rotation matrices."""
    relative = np.asarray(actual, dtype=np.float64) @ np.asarray(target, dtype=np.float64).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def deterministic_policy_noise_seed(
    base_seed: int,
    *,
    suite_name: str,
    task_id: int,
    episode_index: int,
    model_call_index: int,
) -> int | None:
    """Derive a stable per-request seed independent of process and batch scheduling."""
    if int(base_seed) < 0:
        return None
    payload = (
        f"{int(base_seed)}\0{suite_name}\0{int(task_id)}\0"
        f"{int(episode_index)}\0{int(model_call_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def model_observation_fingerprints(observation: dict[str, Any]) -> dict[str, str]:
    """Return exact, transport-independent hashes for one policy observation."""
    fingerprints: dict[str, str] = {}
    overall = hashlib.sha256()
    for key in sorted(observation):
        array = np.ascontiguousarray(np.asarray(observation[key]))
        header = f"{key}\0{array.dtype.str}\0{tuple(array.shape)}\0".encode()
        digest = hashlib.sha256()
        digest.update(header)
        digest.update(array.view(np.uint8))
        fingerprints[str(key)] = digest.hexdigest()
        overall.update(header)
        overall.update(array.view(np.uint8))
    fingerprints["__all__"] = overall.hexdigest()
    return fingerprints


def configure_torch_determinism(enabled: bool) -> dict[str, Any]:
    """Configure repeatable policy inference without making env workers import torch."""
    if not bool(enabled):
        return {
            "enabled": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }

    import torch

    # Some CUDA cumsum versions only advertise a non-deterministic kernel even
    # though this evaluator's fixed masks repeat exactly.  warn_only retains the
    # strongest available deterministic kernels instead of aborting evaluation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "enabled": True,
        "warn_only": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def action_chunk_to_absolute_libero_actions(
    *,
    env: Any,
    current_eef_pose9_gripper: np.ndarray,
    action_chunk: np.ndarray,
    gripper_threshold: float,
    gripper_max_width: float,
    gripper_control_mode: str,
    gripper_delta_threshold: float,
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
    control_mode = str(gripper_control_mode)
    if control_mode not in {"delta_width", "absolute_width", "target_width"}:
        raise ValueError(f"Unsupported gripper_control_mode={control_mode!r}.")
    for idx, (row, target_model_world) in enumerate(zip(chunk, target_model_worlds, strict=True)):
        target_controller_world = target_model_world @ model_to_controller
        arm_action = world_pose_to_libero_absolute_action(target_controller_world)
        if control_mode == "absolute_width":
            gripper_action = gripper_threshold_command(
                float(row[-1]),
                threshold=float(gripper_threshold),
                max_physical_width=float(gripper_max_width),
            )
        elif control_mode == "delta_width":
            # The command applied while moving from row i to row i+1 is the
            # direction of the predicted width change over that interval.
            delta_width = float(chunk[idx + 1, -1] - row[-1]) if idx < chunk.shape[0] - 1 else 0.0
            threshold = float(gripper_delta_threshold)
            gripper_action = -1.0 if delta_width > threshold else 1.0 if delta_width < -threshold else 0.0
        else:
            # The measured width is available only after each environment
            # step, so run_episode replaces this placeholder online.
            gripper_action = 0.0
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
        "evaluation_identity": cfg.get("evaluation_identity", {}),
        "env_seed": int(cfg.get("env_seed", 7)),
        "use_suite_max_steps": bool(cfg.get("use_suite_max_steps", True)),
        "suite_max_steps": {
            suite_name: int(LIBERO_STANDARD_MAX_STEPS.get(suite_name, cfg["control"]["max_steps"]))
            for suite_name in suite_names
        },
        "execution": {
            "task_workers": int(cfg["task_workers"]),
            "isolated_policy_workers": int(cfg.get("isolated_policy_workers", 1)),
            "episode_workers_per_task": int(cfg["episode_workers_per_task"]),
            "task_worker_backend": str(cfg["task_worker_backend"]),
            "inference_batch_size": int(cfg["inference_batch_size"]),
            "inference_batch_wait_ms": float(cfg["inference_batch_wait_ms"]),
            "recreate_env_per_episode": bool(cfg["recreate_env_per_episode"]),
            "deterministic_torch": bool(cfg["deterministic_torch"]),
            "torch_determinism": cfg.get("torch_determinism", {}),
        },
        "initialization": {
            "settle_steps": int(cfg["settle_steps"]),
            "settle_min_seconds": float(cfg["settle_min_seconds"]),
            "settle_stable_seconds": float(cfg["settle_stable_seconds"]),
            "settle_max_seconds": float(cfg["settle_max_seconds"]),
            "settle_require_stable": bool(cfg["settle_require_stable"]),
            "settle_keep_robot_fixed": bool(cfg["settle_keep_robot_fixed"]),
            "initial_gripper_open": bool(cfg["initial_gripper_open"]),
        },
        "observation": {
            "num_points": int(cfg["num_points"]),
            "height": int(cfg["observation_height"]),
            "width": int(cfg["observation_width"]),
            "add_gripper_cloud": bool(cfg["add_gripper_cloud"]),
            "gripper_points": int(cfg.get("gripper_points", 0)),
        },
        "control": {
            "controller_mode": "OSC_POSE absolute pose",
            "action_dim": 7,
            "action_format": ["abs_x", "abs_y", "abs_z", "abs_rx", "abs_ry", "abs_rz", "gripper"],
            "gripper_convention": "-1=open, +1=close",
            "gripper_threshold": float(cfg["control"]["gripper_threshold"]),
            "gripper_control_mode": str(cfg["control"]["gripper_control_mode"]),
            "gripper_delta_threshold": float(cfg["control"]["gripper_delta_threshold"]),
            "gripper_delta_alignment": "next_width_minus_current_width",
            "gripper_target_tolerance": float(cfg["control"]["gripper_target_tolerance"]),
            "exec_action_steps": int(cfg["control"]["exec_action_steps"]),
            "action_index": int(cfg["control"]["action_index"]),
            "control_freq": float(cfg["control"].get("control_freq", cfg.get("control_freq", 20))),
            "policy_noise_seed": int(cfg["policy_noise_seed"]),
        },
        "overall": aggregate_task_results(tasks),
        "suite_reports": suite_reports,
        "results": tasks,
    }
    for suite in suite_reports:
        write_json_atomic(output_dir / str(suite["suite"]) / "suite_report.json", suite)
    # Multi-GPU suite children share one output root but own disjoint suite
    # directories.  They leave the two global reports to the launcher so no
    # cross-process file can overwrite another suite's results.
    if os.environ.get("SONG_LIBERO_SUITE_WORKER", "0") == "1":
        return
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "overall_report.json", {"overall": summary["overall"], "suites": suite_reports})


def compact_episode_record(result: dict[str, Any], episode_idx: int, action_npz: str | None) -> dict[str, Any]:
    drop_keys = {
        "video_frames",
        "libero_actions",
        "model_action_rows",
        "target_controller_pose9",
        "target_model_worlds",
        "achieved_model_worlds",
        "tracking_position_errors",
        "tracking_rotation_errors",
        "chunk_start_model_worlds",
        "chunk_previous_target_model_worlds",
        "chunk_boundary_position_errors",
        "chunk_boundary_rotation_errors",
        "gripper_commands",
        "gripper_raw_widths",
        "gripper_width_pcts",
        "gripper_actual_widths",
        "gripper_width_errors",
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
        "achieved_model_worlds": np.asarray(result.get("achieved_model_worlds", []), dtype=np.float32),
        "tracking_position_errors": np.asarray(result.get("tracking_position_errors", []), dtype=np.float32),
        "tracking_rotation_errors": np.asarray(result.get("tracking_rotation_errors", []), dtype=np.float32),
        "chunk_start_model_worlds": np.asarray(result.get("chunk_start_model_worlds", []), dtype=np.float32),
        "chunk_previous_target_model_worlds": np.asarray(
            result.get("chunk_previous_target_model_worlds", []), dtype=np.float32
        ),
        "chunk_boundary_position_errors": np.asarray(
            result.get("chunk_boundary_position_errors", []), dtype=np.float32
        ),
        "chunk_boundary_rotation_errors": np.asarray(
            result.get("chunk_boundary_rotation_errors", []), dtype=np.float32
        ),
        "gripper_commands": np.asarray(result.get("gripper_commands", []), dtype=np.float32),
        "gripper_raw_widths": np.asarray(result.get("gripper_raw_widths", []), dtype=np.float32),
        "gripper_width_pcts": np.asarray(result.get("gripper_width_pcts", []), dtype=np.float32),
        "gripper_actual_widths": np.asarray(result.get("gripper_actual_widths", []), dtype=np.float32),
        "gripper_width_errors": np.asarray(result.get("gripper_width_errors", []), dtype=np.float32),
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


def _normalized_camera_name(camera_name: Any) -> str:
    camera_name = str(camera_name).strip()
    return camera_name[: -len("_image")] if camera_name.endswith("_image") else camera_name


def _append_camera_candidate(candidates: list[str], camera_name: Any) -> None:
    if camera_name is None:
        return
    camera_name = _normalized_camera_name(camera_name)
    if camera_name and camera_name not in candidates:
        candidates.append(camera_name)


def _policy_image_camera_candidates(
    image_key: str,
    cfg: dict[str, Any],
    *,
    single_image_feature: bool,
) -> list[str]:
    """Map a dataset image feature name to possible LIBERO render cameras."""
    feature_camera = image_key.removeprefix("observation.images.")
    normalized_feature = feature_camera.lower().replace("-", "_")
    candidates: list[str] = []

    camera_map = cfg.get("policy_image_camera_map") or {}
    if not isinstance(camera_map, dict):
        raise TypeError("policy_image_camera_map must be a JSON object mapping policy image keys to LIBERO cameras.")
    _append_camera_candidate(candidates, camera_map.get(image_key, camera_map.get(feature_camera)))

    # image_camera is the source camera used by the LIBERO dataset converter.
    # It is therefore the strongest implicit mapping for a single-image checkpoint.
    if single_image_feature:
        _append_camera_candidate(candidates, cfg.get("image_camera"))
    _append_camera_candidate(candidates, feature_camera)

    overview_aliases = {
        "agentview",
        "external",
        "external_camera",
        "overhead",
        "overhead_camera",
        "overview",
        "overview_camera",
    }
    hand_aliases = {
        "eye_in_hand",
        "hand",
        "hand_camera",
        "robot0_eye_in_hand",
        "wrist",
        "wrist_camera",
    }
    configured_cameras = list(cfg.get("camera_names") or [])

    if normalized_feature in overview_aliases:
        _append_camera_candidate(candidates, cfg.get("pointcloud_reference_camera"))
        for camera_name in cfg.get("pointcloud_camera_names") or []:
            _append_camera_candidate(candidates, camera_name)
        for camera_name in configured_cameras:
            normalized = _normalized_camera_name(camera_name).lower()
            if "agentview" in normalized or "overhead" in normalized or "overview" in normalized:
                _append_camera_candidate(candidates, camera_name)
        _append_camera_candidate(candidates, "agentview")
    elif normalized_feature in hand_aliases:
        for camera_name in configured_cameras:
            normalized = _normalized_camera_name(camera_name).lower()
            if "eye_in_hand" in normalized or "wrist" in normalized or "hand" in normalized:
                _append_camera_candidate(candidates, camera_name)
        _append_camera_candidate(candidates, "robot0_eye_in_hand")

    return candidates


def resolve_policy_rgb_cameras(
    infer: SmolVLA_ModelInference,
    raw_obs: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, str]:
    """Resolve canonical adapter image features to concrete LIBERO raw-observation cameras."""
    image_keys = list(infer.policy.config.image_features)
    resolved: dict[str, str] = {}
    for image_key in image_keys:
        candidates = _policy_image_camera_candidates(
            image_key,
            cfg,
            single_image_feature=len(image_keys) == 1,
        )
        for camera_name in candidates:
            if dataset_image_from_raw_obs(raw_obs, camera_name) is not None:
                resolved[image_key] = camera_name
                break
        else:
            available = sorted(
                key for key, value in raw_obs.items() if key.endswith("_image") and np.asarray(value).ndim == 3
            )
            raise KeyError(
                f"Adapter checkpoint requires {image_key!r}, but no matching LIBERO image was found. "
                f"Tried cameras={candidates!r}; available image keys={available!r}. "
                "Set policy_image_camera_map in the eval config when using a custom camera alias."
            )
    return resolved


def build_policy_rgb_observation(
    camera_map: dict[str, str],
    raw_obs: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Read adapter RGB inputs under the exact feature keys declared by the checkpoint."""
    images: dict[str, np.ndarray] = {}
    for image_key, camera_name in camera_map.items():
        image = dataset_image_from_raw_obs(raw_obs, camera_name)
        if image is None:
            raise KeyError(
                f"Adapter image mapping {image_key!r} <- {camera_name!r} became unavailable in raw_obs."
            )
        images[image_key] = image
    return images


@dataclass(slots=True)
class _InferenceRequest:
    observation: dict[str, Any]
    task: str
    postprocess: bool
    state_pose_mode: str
    noise_seed: int | None
    future: Future[Any]


def _stack_model_observations(observations: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Stack homogeneous per-environment observations into one model batch."""
    if not observations:
        raise ValueError("Cannot stack an empty observation list.")
    expected_keys = set(observations[0])
    for index, observation in enumerate(observations[1:], start=1):
        if set(observation) != expected_keys:
            raise KeyError(
                f"Parallel inference observation {index} has keys {sorted(observation)}, "
                f"expected {sorted(expected_keys)}."
            )

    batch: dict[str, np.ndarray] = {}
    for key in sorted(expected_keys):
        values = [np.asarray(observation[key]) for observation in observations]
        try:
            batch[key] = np.stack(values, axis=0)
        except ValueError as exc:
            shapes = [tuple(value.shape) for value in values]
            raise ValueError(f"Cannot batch observation key {key!r} with shapes {shapes}.") from exc
    return batch


class BatchedInferenceScheduler:
    """One-policy dynamic batcher shared by concurrent task environment threads.

    MuJoCo work remains in the task threads.  Only this scheduler thread touches
    the policy, so a GPU process loads one checkpoint and never performs
    concurrent forwards on the same module.
    """

    shared_parallel_inference = True

    def __init__(
        self,
        infer: SmolVLA_ModelInference,
        *,
        max_batch_size: int,
        batch_wait_ms: float,
    ) -> None:
        self.infer = infer
        self.policy = infer.policy
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_s = max(0.0, float(batch_wait_ms)) / 1000.0
        self._queue: queue.Queue[_InferenceRequest | object] = queue.Queue()
        self._stop_token = object()
        self._closed = False
        self._batch_count = 0
        self._request_count = 0
        self._max_observed_batch = 0
        self._thread = threading.Thread(
            target=self._run,
            name="libero-gpu-inference-batcher",
            daemon=True,
        )
        self._thread.start()

    def policy_reset(self) -> None:
        # predict_action_chunk() does not consume the policy action queue.  A
        # shared reset from one environment would otherwise race other tasks.
        return None

    def predict_action_chunk_obs(
        self,
        observation: dict[str, Any],
        *,
        task: str | list[str] = "",
        postprocess: bool = True,
        state_pose_mode: str = "identity",
        noise_seed: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Parallel inference scheduler is already closed.")
        if not isinstance(task, str):
            if len(task) != 1:
                raise ValueError("A task worker must submit exactly one language instruction.")
            task = str(task[0])
        future: Future[Any] = Future()
        self._queue.put(
            _InferenceRequest(
                observation=observation,
                task=str(task),
                postprocess=bool(postprocess),
                state_pose_mode=str(state_pose_mode),
                noise_seed=None if noise_seed is None else int(noise_seed),
                future=future,
            )
        )
        return future.result()

    def _collect_batch(self, first: _InferenceRequest) -> tuple[list[_InferenceRequest], bool]:
        requests = [first]
        closing = False
        deadline = time.monotonic() + self.batch_wait_s
        while len(requests) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is self._stop_token:
                closing = True
                break
            requests.append(item)
        return requests, closing

    def _execute_batch(self, requests: list[_InferenceRequest]) -> None:
        try:
            postprocess_values = {request.postprocess for request in requests}
            state_modes = {request.state_pose_mode for request in requests}
            if len(postprocess_values) != 1 or len(state_modes) != 1:
                raise ValueError("All requests in one dynamic batch must use the same inference options.")
            if len(requests) == 1:
                # Preserve the exact serial preprocessing path.  Stacking a
                # singleton and wrapping task/seed in lists is numerically very
                # close, but the resulting sub-millimetre action difference can
                # bifurcate a contact-rich closed-loop rollout.
                request = requests[0]
                action_chunks = self.infer.predict_action_chunk_obs(
                    request.observation,
                    task=request.task,
                    postprocess=request.postprocess,
                    state_pose_mode=request.state_pose_mode,
                    noise_seed=request.noise_seed,
                )
            else:
                observation_batch = _stack_model_observations(
                    [request.observation for request in requests]
                )
                action_chunks = self.infer.predict_action_chunk_obs(
                    observation_batch,
                    task=[request.task for request in requests],
                    postprocess=requests[0].postprocess,
                    state_pose_mode=requests[0].state_pose_mode,
                    noise_seed=[request.noise_seed for request in requests]
                    if all(request.noise_seed is not None for request in requests)
                    else None,
                )
            if int(action_chunks.shape[0]) != len(requests):
                raise RuntimeError(
                    f"Policy returned batch {int(action_chunks.shape[0])}, expected {len(requests)}."
                )
            for index, request in enumerate(requests):
                request.future.set_result(action_chunks[index : index + 1])
            self._batch_count += 1
            self._request_count += len(requests)
            self._max_observed_batch = max(self._max_observed_batch, len(requests))
        except BaseException as exc:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(exc)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop_token:
                break
            requests, closing = self._collect_batch(item)
            self._execute_batch(requests)
            if closing:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop_token)
        self._thread.join()
        mean_batch = self._request_count / self._batch_count if self._batch_count else 0.0
        print(
            "[parallel] inference batches: "
            f"requests={self._request_count}, batches={self._batch_count}, "
            f"mean_batch={mean_batch:.2f}, max_batch={self._max_observed_batch}",
            flush=True,
        )


@dataclass(slots=True)
class _ProcessInferenceRequest:
    worker_id: int
    request_id: int
    observation: dict[str, Any]
    task: str
    postprocess: bool
    state_pose_mode: str
    noise_seed: int | None


class ProcessInferenceProxy:
    """Child-process policy facade backed by the parent GPU model service."""

    shared_parallel_inference = True

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
        vla_adapter_enable: bool,
        image_feature_keys: list[str],
    ) -> None:
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._next_request_id = 0
        self.policy = SimpleNamespace(
            config=SimpleNamespace(
                vla_adapter_enable=bool(vla_adapter_enable),
                image_features={str(key): None for key in image_feature_keys},
            )
        )

    def policy_reset(self) -> None:
        # Chunk inference does not consume the policy action queue.  Resetting a
        # shared parent policy from an individual environment would be racy.
        return None

    def predict_action_chunk_obs(
        self,
        observation: dict[str, Any],
        *,
        task: str | list[str] = "",
        postprocess: bool = True,
        state_pose_mode: str = "identity",
        noise_seed: int | None = None,
    ) -> np.ndarray:
        if not isinstance(task, str):
            if len(task) != 1:
                raise ValueError("An environment worker must submit exactly one language instruction.")
            task = str(task[0])
        request_id = self._next_request_id
        self._next_request_id += 1
        self.request_queue.put(
            _ProcessInferenceRequest(
                worker_id=self.worker_id,
                request_id=request_id,
                observation=observation,
                task=str(task),
                postprocess=bool(postprocess),
                state_pose_mode=str(state_pose_mode),
                noise_seed=None if noise_seed is None else int(noise_seed),
            )
        )
        status, response_request_id, payload = self.response_queue.get()
        if int(response_request_id) != request_id:
            raise RuntimeError(
                f"Inference IPC response id {response_request_id} does not match request id {request_id}."
            )
        if status != "ok":
            raise RuntimeError(f"Parent GPU inference failed:\n{payload}")
        return np.asarray(payload)


class _SingleTaskSuiteProxy:
    """Minimal suite interface needed by make_libero_env inside a worker."""

    def __init__(self, task_id: int, task_spec: dict[str, str]) -> None:
        self.task_id = int(task_id)
        self.task = SimpleNamespace(**{str(key): str(value) for key, value in task_spec.items()})

    def get_task(self, task_id: int) -> Any:
        if int(task_id) != self.task_id:
            raise KeyError(f"Single-task worker owns task {self.task_id}, requested {task_id}.")
        return self.task


def _process_task_worker_entry(
    *,
    worker_id: int,
    suite_name: str,
    task_id: int,
    shard_index: int,
    episode_indices: list[int],
    task_spec: dict[str, str],
    task_init_states: np.ndarray,
    cfg: dict[str, Any],
    output_dir: Path,
    request_queue: Any,
    response_queue: Any,
    ready_queue: Any,
    result_queue: Any,
    start_event: Any,
    vla_adapter_enable: bool,
    image_feature_keys: list[str],
) -> None:
    """Own one MuJoCo task environment in a process that never loads the policy."""
    try:
        ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
        suite = _SingleTaskSuiteProxy(task_id, task_spec)
        infer = ProcessInferenceProxy(
            worker_id=worker_id,
            request_queue=request_queue,
            response_queue=response_queue,
            vla_adapter_enable=vla_adapter_enable,
            image_feature_keys=image_feature_keys,
        )

        def _announce_ready_and_wait() -> None:
            ready_queue.put(int(worker_id))
            start_event.wait()

        summary = evaluate_task(
            infer=infer,
            suite=suite,
            suite_name=suite_name,
            task_id=int(task_id),
            cfg=cfg,
            output_dir=output_dir,
            episode_indices=episode_indices,
            task_init_states=task_init_states,
            on_environment_ready=_announce_ready_and_wait,
        )
        result_queue.put(("ok", int(worker_id), int(task_id), int(shard_index), summary))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                int(worker_id),
                int(task_id),
                int(shard_index),
                repr(exc),
                traceback.format_exc(),
            )
        )


def _execute_process_inference_batch(
    infer: SmolVLA_ModelInference,
    requests: list[_ProcessInferenceRequest],
    response_queues: dict[int, Any],
) -> None:
    try:
        postprocess_values = {request.postprocess for request in requests}
        state_modes = {request.state_pose_mode for request in requests}
        if len(postprocess_values) != 1 or len(state_modes) != 1:
            raise ValueError("All requests in one dynamic batch must use the same inference options.")
        if len(requests) == 1:
            request = requests[0]
            action_chunks = infer.predict_action_chunk_obs(
                request.observation,
                task=request.task,
                postprocess=request.postprocess,
                state_pose_mode=request.state_pose_mode,
                noise_seed=request.noise_seed,
            )
        else:
            observation_batch = _stack_model_observations([request.observation for request in requests])
            action_chunks = infer.predict_action_chunk_obs(
                observation_batch,
                task=[request.task for request in requests],
                postprocess=requests[0].postprocess,
                state_pose_mode=requests[0].state_pose_mode,
                noise_seed=[request.noise_seed for request in requests]
                if all(request.noise_seed is not None for request in requests)
                else None,
            )
        if hasattr(action_chunks, "detach"):
            action_chunks = action_chunks.detach().cpu().numpy()
        else:
            action_chunks = np.asarray(action_chunks)
        if int(action_chunks.shape[0]) != len(requests):
            raise RuntimeError(
                f"Policy returned batch {int(action_chunks.shape[0])}, expected {len(requests)}."
            )
        for index, request in enumerate(requests):
            response_queues[request.worker_id].put(
                ("ok", request.request_id, np.asarray(action_chunks[index : index + 1]))
            )
    except BaseException:
        error_text = traceback.format_exc()
        for request in requests:
            response_queues[request.worker_id].put(("error", request.request_id, error_text))


def _collect_process_request_batch(
    request_queue: Any,
    first: _ProcessInferenceRequest,
    *,
    max_batch_size: int,
    batch_wait_s: float,
) -> list[_ProcessInferenceRequest]:
    requests = [first]
    deadline = time.monotonic() + max(0.0, float(batch_wait_s))
    while len(requests) < max(1, int(max_batch_size)):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        try:
            requests.append(request_queue.get(timeout=remaining))
        except queue.Empty:
            break
    return requests


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
    steps: int = 0,
    min_seconds: float = 0.25,
    stable_seconds: float = 0.25,
    max_seconds: float = 8.0,
    linear_velocity_threshold: float = 0.01,
    angular_velocity_threshold: float = 0.05,
    other_dof_velocity_threshold: float = 0.02,
    require_stable: bool = True,
    keep_robot_fixed: bool = True,
    open_gripper: bool = True,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance physics until scene objects have remained stable for a time window.

    The fast runner moves the robot by direct IK/qpos writes instead of robosuite
    env.step().  After env.reset(), some LIBERO objects may still be slightly
    above the table or have transient velocity.  If the policy starts immediately,
    the first point cloud can represent a scene that has not settled yet.  This
    helper advances only MuJoCo physics and monitors free-object linear/angular
    velocities.  Stability must hold continuously, which avoids accepting the
    near-zero velocity at the apex of a falling object's trajectory.  The robot
    arm and gripper can be restored every tick so their motion does not affect the
    stability decision or the initial policy observation.
    """
    sim = get_sim(env)
    model = sim.model

    timestep = 0.002
    for candidate in (model, getattr(model, "_model", None)):
        opt = getattr(candidate, "opt", None)
        candidate_timestep = getattr(opt, "timestep", None)
        if candidate_timestep is not None and float(candidate_timestep) > 0.0:
            timestep = float(candidate_timestep)
            break

    minimum_steps = max(max(0, int(steps)), int(np.ceil(max(0.0, float(min_seconds)) / timestep)))
    stable_steps = max(1, int(np.ceil(max(0.0, float(stable_seconds)) / timestep)))
    warning_steps = max(
        minimum_steps + stable_steps,
        int(np.ceil(max(0.0, float(max_seconds)) / timestep)),
    )

    arm_qpos_idx: list[int] = []
    arm_qvel_idx: list[int] = []
    gripper_records: list[dict[str, Any]] = []
    if bool(keep_robot_fixed):
        try:
            arm_qpos_idx, arm_qvel_idx = robot_arm_indices(env)
        except Exception:
            arm_qpos_idx, arm_qvel_idx = [], []
    if bool(keep_robot_fixed) or bool(open_gripper):
        try:
            gripper_records = gripper_joint_records(env)
        except Exception:
            gripper_records = []

    if bool(open_gripper):
        if not gripper_records:
            raise RuntimeError("Could not identify gripper joints required to set the initial fully-open state.")
        for rec in gripper_records:
            joint_id = int(rec["joint_id"])
            qpos_adr = int(rec["qpos_adr"])
            joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64).reshape(-1)
            if joint_range.size < 2:
                raise RuntimeError(f"Gripper joint {rec['name']!r} does not have a valid joint range.")
            # Panda finger joints have opposite signs: [0, +0.04] and
            # [-0.04, 0].  The endpoint with the larger absolute displacement
            # is the open limit for each finger.
            low, high = float(joint_range[0]), float(joint_range[1])
            sim.data.qpos[qpos_adr] = low if abs(low) > abs(high) else high
            qvel_adr = int(rec.get("qvel_adr", -1))
            if 0 <= qvel_adr < len(sim.data.qvel):
                sim.data.qvel[qvel_adr] = 0.0

        sim.forward()

    fixed_arm_qpos = np.asarray(sim.data.qpos[arm_qpos_idx], dtype=np.float64).copy() if arm_qpos_idx else None
    fixed_gripper_qpos = {int(rec["qpos_adr"]): float(sim.data.qpos[int(rec["qpos_adr"])]) for rec in gripper_records}

    robot_qvel_idx = {int(index) for index in arm_qvel_idx}
    for rec in gripper_records:
        qvel_adr = int(rec.get("qvel_adr", -1))
        if qvel_adr >= 0:
            robot_qvel_idx.add(qvel_adr)
    robot = getattr(env, "robots", [None])[0]
    if robot is not None:
        for attr in (
            "_ref_joint_vel_indexes",
            "ref_joint_vel_indexes",
            "_ref_gripper_joint_vel_indexes",
            "ref_gripper_joint_vel_indexes",
            "gripper_joint_vel_indexes",
        ):
            value = getattr(robot, attr, None)
            if value is not None:
                robot_qvel_idx.update(int(index) for index in np.asarray(value).reshape(-1).tolist())

    # MuJoCo joint types: free=0, ball=1, slide=2, hinge=3.  LIBERO tabletop
    # objects use free joints, so these six-DoF velocity blocks are the most
    # direct, scene-independent signal that falling / bouncing has stopped.
    free_object_dof_addresses: list[int] = []
    all_non_robot_dofs: list[int] = []
    joint_type_to_dofs = {0: 6, 1: 3, 2: 1, 3: 1}
    try:
        joint_types = np.asarray(model.jnt_type).reshape(-1)
        joint_dof_addresses = np.asarray(model.jnt_dofadr).reshape(-1)
        for joint_id, raw_joint_type in enumerate(joint_types):
            joint_type = int(raw_joint_type)
            dof_address = int(joint_dof_addresses[joint_id])
            dof_count = int(joint_type_to_dofs.get(joint_type, 0))
            dofs = [dof_address + offset for offset in range(dof_count)]
            scene_dofs = [index for index in dofs if index not in robot_qvel_idx]
            all_non_robot_dofs.extend(scene_dofs)
            if joint_type == 0 and len(scene_dofs) == 6:
                free_object_dof_addresses.append(dof_address)
    except Exception:
        free_object_dof_addresses = []
        all_non_robot_dofs = [
            index for index in range(len(sim.data.qvel)) if index not in robot_qvel_idx
        ]

    # Prefer free objects whenever they exist.  This avoids passive robot finger
    # joints or tiny fixture-joint jitter blocking an otherwise settled scene.
    use_free_object_velocity = bool(free_object_dof_addresses)

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

    def _scene_velocity_metrics() -> tuple[float, float, float]:
        qvel = np.asarray(sim.data.qvel, dtype=np.float64)
        max_linear = 0.0
        max_angular = 0.0
        if use_free_object_velocity:
            for dof_address in free_object_dof_addresses:
                max_linear = max(max_linear, float(np.linalg.norm(qvel[dof_address : dof_address + 3])))
                max_angular = max(max_angular, float(np.linalg.norm(qvel[dof_address + 3 : dof_address + 6])))
            return max_linear, max_angular, 0.0
        max_other = (
            float(np.max(np.abs(qvel[np.asarray(all_non_robot_dofs, dtype=np.int64)])))
            if all_non_robot_dofs
            else 0.0
        )
        return 0.0, 0.0, max_other

    sim.forward()
    if bool(keep_robot_fixed) or bool(open_gripper):
        _restore_robot()
        sim.forward()

    consecutive_stable_steps = 0
    scene_is_stable = False
    executed_steps = 0
    max_linear_speed = 0.0
    max_angular_speed = 0.0
    max_other_speed = 0.0
    warned_slow_settling = False
    settle_step = 0
    while True:
        if hasattr(sim.data, "ctrl"):
            try:
                sim.data.ctrl[:] = 0.0
            except Exception:
                pass
        if bool(keep_robot_fixed) or bool(open_gripper):
            _restore_robot()
        try:
            sim.step()
        except Exception as exc:
            raise RuntimeError(f"MuJoCo failed while settling the scene at physics tick {settle_step}.") from exc
        if bool(keep_robot_fixed) or bool(open_gripper):
            _restore_robot()
            sim.forward()

        executed_steps = settle_step + 1
        max_linear_speed, max_angular_speed, max_other_speed = _scene_velocity_metrics()
        velocity_is_stable = (
            max_linear_speed <= float(linear_velocity_threshold)
            and max_angular_speed <= float(angular_velocity_threshold)
            and max_other_speed <= float(other_dof_velocity_threshold)
        )
        if executed_steps >= minimum_steps and velocity_is_stable:
            consecutive_stable_steps += 1
        else:
            consecutive_stable_steps = 0
        if consecutive_stable_steps >= stable_steps:
            scene_is_stable = True
            break
        if executed_steps >= warning_steps and not warned_slow_settling:
            monitored = (
                f"{len(free_object_dof_addresses)} free objects"
                if use_free_object_velocity
                else f"{len(all_non_robot_dofs)} scene DoFs"
            )
            message = (
                "Scene settling is slower than expected: "
                f"ticks={executed_steps}, sim_time={executed_steps * timestep:.3f}s, "
                f"monitored={monitored}, linear={max_linear_speed:.6f}m/s, "
                f"angular={max_angular_speed:.6f}rad/s, other={max_other_speed:.6f}."
            )
            if bool(require_stable):
                print(f"[warn] {message} Keeping this episode and waiting until it is stable.", flush=True)
            else:
                print(
                    f"[warn] {message} Continuing because --no-settle-require-stable was selected.",
                    flush=True,
                )
                break
            warned_slow_settling = True
        settle_step += 1

    if bool(keep_robot_fixed) or bool(open_gripper):
        # Restore once more so the arm exactly matches the episode init_state and
        # the gripper exactly matches its selected initial state (fully open by
        # default) in both simulator state and the first model observation.
        _restore_robot()
    sim.forward()

    robot_qpos_error = 0.0
    if bool(keep_robot_fixed) or bool(open_gripper):
        if fixed_arm_qpos is not None and arm_qpos_idx:
            robot_qpos_error = max(
                robot_qpos_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(sim.data.qpos[arm_qpos_idx], dtype=np.float64)
                            - fixed_arm_qpos
                        )
                    )
                ),
            )
        for qpos_adr, expected_value in fixed_gripper_qpos.items():
            robot_qpos_error = max(
                robot_qpos_error,
                abs(float(sim.data.qpos[int(qpos_adr)]) - float(expected_value)),
            )
        if robot_qpos_error > 1e-10:
            raise RuntimeError(
                "Robot initial qpos changed during scene settling "
                f"(maximum absolute error={robot_qpos_error:.3e})."
            )

    monitored = (
        f"{len(free_object_dof_addresses)} free objects"
        if use_free_object_velocity
        else f"{len(all_non_robot_dofs)} scene DoFs"
    )
    settle_summary = (
        f"ticks={executed_steps}, sim_time={executed_steps * timestep:.3f}s, monitored={monitored}, "
        f"linear={max_linear_speed:.6f}m/s, angular={max_angular_speed:.6f}rad/s, "
        f"other={max_other_speed:.6f}, robot_qpos_error={robot_qpos_error:.3e}, "
        f"gripper_width={sum(abs(value) for value in fixed_gripper_qpos.values()):.6f}m"
    )
    # if scene_is_stable:
    #     print(f"[settle] scene stable: {settle_summary}", flush=True)

    raw_obs = get_raw_obs(env, force_update=True)
    return raw_obs


def run_episode(
    *,
    infer: Any,
    env: Any,
    suite_name: str,
    task_id: int,
    episode_index: int,
    task_language: str,
    init_state: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    control = cfg["control"]
    configured_max_steps = int(control.get("max_steps", getattr(env, "horizon", 500)))
    max_steps = (
        int(LIBERO_STANDARD_MAX_STEPS[suite_name])
        if bool(cfg.get("use_suite_max_steps", True)) and suite_name in LIBERO_STANDARD_MAX_STEPS
        else configured_max_steps
    )
    action_index = max(0, int(control.get("action_index", 0)))
    exec_action_steps = int(control.get("exec_action_steps", 16))
    warmup_steps = max(0, int(control.get("warmup_steps", 0)))
    gripper_threshold = float(control.get("gripper_threshold", 0.5))
    gripper_max_width = float(cfg.get("gripper_qpos_max_width", 0.08))
    gripper_control_mode = str(control.get("gripper_control_mode", "delta_width"))
    gripper_delta_threshold = float(control.get("gripper_delta_threshold", 0.002))
    gripper_target_tolerance = float(control.get("gripper_target_tolerance", 0.004))
    policy_noise_seed_base = int(cfg.get("policy_noise_seed", 0))


    raw_obs = env.reset()
    raw_obs = env.set_init_state(init_state)

    raw_obs = settle_scene_after_reset(
        env,
        steps=int(cfg["settle_steps"]),
        min_seconds=float(cfg["settle_min_seconds"]),
        stable_seconds=float(cfg["settle_stable_seconds"]),
        max_seconds=float(cfg["settle_max_seconds"]),
        linear_velocity_threshold=float(cfg["settle_linear_velocity_threshold"]),
        angular_velocity_threshold=float(cfg["settle_angular_velocity_threshold"]),
        other_dof_velocity_threshold=float(cfg["settle_other_dof_velocity_threshold"]),
        require_stable=bool(cfg["settle_require_stable"]),
        keep_robot_fixed=bool(cfg["settle_keep_robot_fixed"]),
        open_gripper=bool(cfg["initial_gripper_open"]),
        cfg=cfg,
    )


    # Absolute pose execution only.
    for robot in env.robots:
        robot.controller.use_delta = False


    for _ in range(warmup_steps):
        raw_obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    if not bool(getattr(infer, "shared_parallel_inference", False)):
        infer.policy.reset()
        infer.policy_reset()

    policy_rgb_camera_map: dict[str, str] = {}
    if infer.policy.config.vla_adapter_enable:
        policy_rgb_camera_map = resolve_policy_rgb_cameras(infer, raw_obs, cfg)
        # print(f"[info] adapter RGB camera mapping: {policy_rgb_camera_map}", flush=True)

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
    gripper_actual_widths: list[float] = []
    gripper_width_errors: list[float] = []
    achieved_model_worlds: list[np.ndarray] = []
    tracking_position_errors: list[float] = []
    tracking_rotation_errors: list[float] = []
    model_input_hashes: list[str] = []
    first_model_input_component_hashes: dict[str, str] = {}
    chunk_start_model_worlds: list[np.ndarray] = []
    chunk_previous_target_model_worlds: list[np.ndarray] = []
    chunk_boundary_position_errors: list[float] = []
    chunk_boundary_rotation_errors: list[float] = []
    previous_issued_target_model_world: np.ndarray | None = None

    success_ever = False
    done = False
    model_call_count = 0
    steps = 0
    start_s = time.perf_counter()

    manual_failure = False
    keyboard = EpisodeKeyboardControl()
    if bool(cfg.get("keyboard_control_enabled", True)):
        keyboard.start_terminal()
    if str(cfg.get("render_mode", "offscreen")).lower() == "viewer3d":
        render_camera = normalize_render_camera_name(str(cfg.get("render_camera", "agentview")))
        attach_mujoco_3d_viewer(env, render_camera=render_camera, key_callback=keyboard.viewer_key_callback)
        render_viewer3d(env, cfg, steps, force=True)
    # if terminal_keys_enabled or str(cfg.get("render_mode", "offscreen")).lower() == "viewer3d":
        # print("[eval] press 'n' to mark the current episode as failed and continue", flush=True)

    try:
        while steps < max_steps and not done and not success_ever:
            if keyboard.poll():
                manual_failure = True
                break

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

            model_observation = {
                "point_cloud": point_cloud,
                "state": identity_pose9_gripper(float(eef_pose[-1])),
            }
            chunk_start_model_world = pose9_to_homo_np(np.asarray(eef_pose[:9], dtype=np.float32))
            chunk_start_model_worlds.append(np.asarray(chunk_start_model_world, dtype=np.float32))
            if previous_issued_target_model_world is not None:
                previous_target = np.asarray(previous_issued_target_model_world, dtype=np.float32)
                chunk_previous_target_model_worlds.append(previous_target)
                chunk_boundary_position_errors.append(
                    float(np.linalg.norm(chunk_start_model_world[:3, 3] - previous_target[:3, 3]))
                )
                chunk_boundary_rotation_errors.append(
                    rotation_error_radians(chunk_start_model_world[:3, :3], previous_target[:3, :3])
                )
            if infer.policy.config.vla_adapter_enable:
                model_observation.update(build_policy_rgb_observation(policy_rgb_camera_map, raw_obs))
            input_fingerprints = model_observation_fingerprints(model_observation)
            model_input_hashes.append(input_fingerprints["__all__"])
            if not first_model_input_component_hashes:
                first_model_input_component_hashes = input_fingerprints
            chunk_batch = infer.predict_action_chunk_obs(
                model_observation,
                task=task_language,
                postprocess=True,
                state_pose_mode="identity",
                noise_seed=deterministic_policy_noise_seed(
                    policy_noise_seed_base,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    episode_index=int(episode_index),
                    model_call_index=int(model_call_count),
                ),
            )
            if hasattr(chunk_batch, "detach"):
                chunk = chunk_batch[0].detach().cpu().numpy()
            else:
                chunk = np.asarray(chunk_batch)[0]
            model_call_count += 1

            if keyboard.poll():
                manual_failure = True
                break

            start_idx = min(action_index, max(0, len(chunk) - 1))
            end_idx = len(chunk) if exec_action_steps <= 0 else min(len(chunk), start_idx + exec_action_steps)
            selected_chunk = np.asarray(chunk[start_idx:end_idx], dtype=np.float32)
            actions, model_worlds, controller_pose9 = action_chunk_to_absolute_libero_actions(
                env=env,
                current_eef_pose9_gripper=eef_pose,
                action_chunk=selected_chunk,
                gripper_threshold=gripper_threshold,
                gripper_max_width=gripper_max_width,
                gripper_control_mode=gripper_control_mode,
                gripper_delta_threshold=gripper_delta_threshold,
            )

            for row, action, model_world, controller_pose in zip(
                selected_chunk,
                actions,
                model_worlds,
                controller_pose9,
                strict=True,
            ):
                if steps >= max_steps or done or success_ever or keyboard.poll():
                    manual_failure = keyboard.poll()
                    break
                action = np.asarray(action, dtype=np.float32).copy()
                if gripper_control_mode == "target_width":
                    action[-1] = gripper_target_width_command(
                        float(row[-1]),
                        gripper_scalar(raw_obs),
                        tolerance=gripper_target_tolerance,
                        max_physical_width=gripper_max_width,
                    )
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
                achieved_pose = eef_pose9_gripper_from_obs(raw_obs)
                gripper_actual_widths.append(float(achieved_pose[-1]))
                gripper_width_errors.append(float(achieved_pose[-1] - row[-1]))
                achieved_model_world = pose9_to_homo_np(np.asarray(achieved_pose[:9], dtype=np.float32))
                achieved_model_worlds.append(np.asarray(achieved_model_world, dtype=np.float32))
                tracking_position_errors.append(
                    float(np.linalg.norm(achieved_model_world[:3, 3] - model_world[:3, 3]))
                )
                tracking_rotation_errors.append(
                    rotation_error_radians(achieved_model_world[:3, :3], model_world[:3, :3])
                )
                previous_issued_target_model_world = np.asarray(model_world, dtype=np.float32)

                if save_video:
                    append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))

                if keyboard.poll():
                    manual_failure = True
                    break

                try:
                    success_ever = success_ever or bool(env.check_success())
                except Exception:
                    success_ever = success_ever or bool(reward > 0.0)

            if manual_failure:
                break
        manual_failure = manual_failure or keyboard.poll()
    finally:
        keyboard.close()

    if manual_failure:
        success_ever = False

    final_eef_pose = eef_pose9_gripper_from_obs(raw_obs)
    return {
        "success": bool(success_ever),
        "done": bool(done),
        "manual_failure": bool(manual_failure),
        "termination_reason": (
            "manual_failure_n"
            if manual_failure
            else "success"
            if success_ever
            else "env_done"
            if done
            else "max_steps"
        ),
        "steps": int(steps),
        "max_steps": int(max_steps),
        "action_rows_executed": int(len(libero_actions)),
        "model_call_count": int(model_call_count),
        "sum_reward": float(np.sum(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "wall_s": float(time.perf_counter() - start_s),
        "gripper_threshold": float(gripper_threshold),
        "gripper_control_mode": gripper_control_mode,
        "gripper_delta_threshold": float(gripper_delta_threshold),
        "gripper_delta_alignment": "next_width_minus_current_width",
        "gripper_target_tolerance": float(gripper_target_tolerance),
        "policy_noise_seed_base": int(policy_noise_seed_base),
        "model_input_hashes": model_input_hashes,
        "first_model_input_component_hashes": first_model_input_component_hashes,
        "gripper_open_steps": int(np.sum(np.asarray(gripper_commands) < 0.0)),
        "gripper_close_steps": int(np.sum(np.asarray(gripper_commands) > 0.0)),
        "final_gripper_qpos_sum": float(gripper_scalar(raw_obs)),
        "gripper_width_error_median_abs_m": (
            float(np.median(np.abs(gripper_width_errors))) if gripper_width_errors else 0.0
        ),
        "gripper_width_error_p95_abs_m": (
            float(np.quantile(np.abs(gripper_width_errors), 0.95)) if gripper_width_errors else 0.0
        ),
        "final_eef_pose9_gripper": final_eef_pose,
        "tracking_position_error_median_m": (
            float(np.median(tracking_position_errors)) if tracking_position_errors else 0.0
        ),
        "tracking_position_error_p95_m": (
            float(np.quantile(tracking_position_errors, 0.95)) if tracking_position_errors else 0.0
        ),
        "tracking_position_error_max_m": (
            float(np.max(tracking_position_errors)) if tracking_position_errors else 0.0
        ),
        "tracking_rotation_error_median_rad": (
            float(np.median(tracking_rotation_errors)) if tracking_rotation_errors else 0.0
        ),
        "tracking_rotation_error_p95_rad": (
            float(np.quantile(tracking_rotation_errors, 0.95)) if tracking_rotation_errors else 0.0
        ),
        "tracking_rotation_error_max_rad": (
            float(np.max(tracking_rotation_errors)) if tracking_rotation_errors else 0.0
        ),
        "chunk_boundary_position_error_median_m": (
            float(np.median(chunk_boundary_position_errors)) if chunk_boundary_position_errors else 0.0
        ),
        "chunk_boundary_position_error_max_m": (
            float(np.max(chunk_boundary_position_errors)) if chunk_boundary_position_errors else 0.0
        ),
        "chunk_boundary_rotation_error_median_rad": (
            float(np.median(chunk_boundary_rotation_errors)) if chunk_boundary_rotation_errors else 0.0
        ),
        "chunk_boundary_rotation_error_max_rad": (
            float(np.max(chunk_boundary_rotation_errors)) if chunk_boundary_rotation_errors else 0.0
        ),
        "libero_actions": np.asarray(libero_actions, dtype=np.float32),
        "model_action_rows": np.asarray(model_action_rows, dtype=np.float32),
        "target_model_worlds": np.asarray(target_model_worlds, dtype=np.float32),
        "target_controller_pose9": np.asarray(target_controller_pose9, dtype=np.float32),
        "achieved_model_worlds": np.asarray(achieved_model_worlds, dtype=np.float32),
        "tracking_position_errors": np.asarray(tracking_position_errors, dtype=np.float32),
        "tracking_rotation_errors": np.asarray(tracking_rotation_errors, dtype=np.float32),
        "chunk_start_model_worlds": np.asarray(chunk_start_model_worlds, dtype=np.float32),
        "chunk_previous_target_model_worlds": np.asarray(
            chunk_previous_target_model_worlds, dtype=np.float32
        ),
        "chunk_boundary_position_errors": np.asarray(chunk_boundary_position_errors, dtype=np.float32),
        "chunk_boundary_rotation_errors": np.asarray(chunk_boundary_rotation_errors, dtype=np.float32),
        "gripper_commands": np.asarray(gripper_commands, dtype=np.float32),
        "gripper_raw_widths": np.asarray(gripper_raw_widths, dtype=np.float32),
        "gripper_width_pcts": np.asarray(gripper_width_pcts, dtype=np.float32),
        "gripper_actual_widths": np.asarray(gripper_actual_widths, dtype=np.float32),
        "gripper_width_errors": np.asarray(gripper_width_errors, dtype=np.float32),
        "video_frames": video_frames,
    }


def evaluate_task(
    *,
    infer: Any,
    suite: Any,
    suite_name: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    episode_indices: list[int] | None = None,
    task_init_states: np.ndarray | None = None,
    on_environment_ready: Any | None = None,
) -> dict[str, Any]:
    """Evaluate all or one deterministic episode shard for a task."""
    init_states = (
        get_task_init_states(suite, int(task_id))
        if task_init_states is None
        else np.asarray(task_init_states)
    )
    if episode_indices is None:
        resolved_episode_indices = list(range(int(cfg["episodes"])))
    else:
        resolved_episode_indices = [int(episode_index) for episode_index in episode_indices]
        if len(set(resolved_episode_indices)) != len(resolved_episode_indices):
            raise ValueError(f"Duplicate episode indices for task {task_id}: {resolved_episode_indices}")
        invalid_episode_indices = [
            episode_index
            for episode_index in resolved_episode_indices
            if episode_index < 0 or episode_index >= int(cfg["episodes"])
        ]
        if invalid_episode_indices:
            raise ValueError(
                f"Invalid episode indices for task {task_id}: {invalid_episode_indices}; "
                f"configured episode count is {int(cfg['episodes'])}."
            )
    task_results: list[dict[str, Any]] = []
    task_name = f"task_{int(task_id):03d}"
    task_language = f"{suite_name}:{int(task_id)}"

    def _make_task_env() -> tuple[Any, Any]:
        # LIBERO / robosuite model construction and EGL context setup enter
        # process-global native registries.  Concurrent construction can
        # segfault even though already-created environments are independent.
        with _ENV_CREATION_LOCK:
            return make_libero_env(
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
                env_seed=int(cfg.get("env_seed", 7)),
            )

    shared_env = None
    try:
        if not bool(cfg.get("recreate_env_per_episode", False)):
            shared_env, shared_task = _make_task_env()
            task_name = str(getattr(shared_task, "name", task_name))
            task_language = str(getattr(shared_task, "language", task_language))
            if on_environment_ready is not None:
                on_environment_ready()
        elif on_environment_ready is not None:
            # All process workers cross the startup barrier together, then
            # construct their per-episode environments concurrently.
            on_environment_ready()

        for episode_idx in resolved_episode_indices:
            episode_dir = (
                output_dir
                / suite_name
                / f"task_{int(task_id):03d}"
                / f"episode_{episode_idx:03d}"
            )
            episode_dir.mkdir(parents=True, exist_ok=True)
            append_realtime_episode_event(
                output_dir=output_dir,
                suite_name=suite_name,
                task_id=int(task_id),
                episode_index=int(episode_idx),
                event="episode_started",
            )
            print(
                f"[eval] start suite={suite_name} task={task_id} episode={episode_idx}",
                flush=True,
            )

            env = shared_env
            try:
                if bool(cfg.get("recreate_env_per_episode", False)):
                    env, task = _make_task_env()
                    task_name = str(getattr(task, "name", task_name))
                    task_language = str(getattr(task, "language", task_language))

                assert env is not None
                result = run_episode(
                    infer=infer,
                    env=env,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    episode_index=int(episode_idx),
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
                    video_paths = export_episode_videos(
                        result,
                        episode_dir.parent,
                        video_record,
                        cfg,
                    )
                    if video_paths:
                        episode_record["videos"] = video_paths

                write_json_atomic(episode_dir / "result.json", episode_record)
                update_realtime_episode_progress(
                    output_dir=output_dir,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    episode_index=int(episode_idx),
                    episode_record=episode_record,
                )
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
                update_realtime_episode_progress(
                    output_dir=output_dir,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    episode_index=int(episode_idx),
                    episode_record=failure,
                )
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
    finally:
        if shared_env is not None:
            try:
                shared_env.close()
            except Exception:
                pass

    return make_task_summary(
        suite_name=suite_name,
        task_id=int(task_id),
        task_name=task_name,
        task_language=task_language,
        episodes=task_results,
    )


def failed_task_summary(
    *,
    suite_name: str,
    task_id: int,
    episode_count: int,
    exc: BaseException,
    episode_indices: list[int] | None = None,
) -> dict[str, Any]:
    resolved_episode_indices = (
        list(range(int(episode_count)))
        if episode_indices is None
        else [int(episode_index) for episode_index in episode_indices]
    )
    episodes = [
        {
            "episode_index": episode_idx,
            "success": False,
            "steps": 0,
            "model_call_count": 0,
            "sum_reward": 0.0,
            "max_reward": 0.0,
            "error": repr(exc),
        }
        for episode_idx in resolved_episode_indices
    ]
    return make_task_summary(
        suite_name=suite_name,
        task_id=int(task_id),
        task_name=f"task_{int(task_id):03d}",
        task_language=f"{suite_name}:{int(task_id)}",
        episodes=episodes,
    )


@dataclass(frozen=True, slots=True)
class _EpisodeWorkerJob:
    worker_id: int
    task_id: int
    shard_index: int
    episode_indices: tuple[int, ...]


def _build_episode_worker_jobs(
    task_ids: list[int],
    *,
    episode_count: int,
    episode_workers_per_task: int,
) -> list[_EpisodeWorkerJob]:
    shard_count = min(max(1, int(episode_workers_per_task)), max(1, int(episode_count)))
    jobs: list[_EpisodeWorkerJob] = []
    for task_id in task_ids:
        for shard_index in range(shard_count):
            episode_indices = tuple(range(shard_index, int(episode_count), shard_count))
            if not episode_indices:
                continue
            jobs.append(
                _EpisodeWorkerJob(
                    worker_id=len(jobs),
                    task_id=int(task_id),
                    shard_index=shard_index,
                    episode_indices=episode_indices,
                )
            )
    return jobs


def serialize_libero_task(task: Any) -> dict[str, str]:
    required_fields = ("name", "language", "problem_folder", "bddl_file")
    missing = [field for field in required_fields if not hasattr(task, field)]
    if missing:
        raise AttributeError(f"LIBERO task is missing required fields: {missing}")
    if hasattr(task, "_asdict"):
        values = task._asdict()
    else:
        values = {
            field: getattr(task, field)
            for field in (
                "name",
                "language",
                "problem",
                "problem_folder",
                "bddl_file",
                "init_states_file",
            )
            if hasattr(task, field)
        }
    return {str(key): str(value) for key, value in values.items()}


def init_states_as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def merge_task_episode_shards(
    *,
    suite_name: str,
    task_ids: list[int],
    episode_count: int,
    partial_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    partials_by_task: dict[int, list[dict[str, Any]]] = {int(task_id): [] for task_id in task_ids}
    for summary in partial_summaries:
        task_id = int(summary["task_id"])
        if task_id not in partials_by_task:
            raise RuntimeError(f"Received an unexpected task summary for task {task_id}.")
        partials_by_task[task_id].append(summary)

    merged: list[dict[str, Any]] = []
    expected_episode_indices = set(range(int(episode_count)))
    for task_id in task_ids:
        partials = partials_by_task[int(task_id)]
        if not partials:
            raise RuntimeError(f"No episode worker returned a summary for task {task_id}.")
        episodes_by_index: dict[int, dict[str, Any]] = {}
        for partial in partials:
            for episode in partial.get("episodes", []):
                episode_index = int(episode["episode_index"])
                if episode_index in episodes_by_index:
                    raise RuntimeError(
                        f"Duplicate result for suite={suite_name} task={task_id} episode={episode_index}."
                    )
                episodes_by_index[episode_index] = episode
        actual_episode_indices = set(episodes_by_index)
        if actual_episode_indices != expected_episode_indices:
            missing = sorted(expected_episode_indices - actual_episode_indices)
            unexpected = sorted(actual_episode_indices - expected_episode_indices)
            raise RuntimeError(
                f"Incomplete episode shards for suite={suite_name} task={task_id}: "
                f"missing={missing}, unexpected={unexpected}."
            )
        first = partials[0]
        merged.append(
            make_task_summary(
                suite_name=suite_name,
                task_id=int(task_id),
                task_name=str(first.get("task_name", f"task_{int(task_id):03d}")),
                task_language=str(first.get("task_language", f"{suite_name}:{int(task_id)}")),
                episodes=[episodes_by_index[index] for index in sorted(episodes_by_index)],
            )
        )
    return merged


def evaluate_suite_process_parallel(
    *,
    infer: SmolVLA_ModelInference,
    suite_name: str,
    task_ids: list[int],
    task_specs_by_id: dict[int, dict[str, str]],
    task_init_states_by_id: dict[int, np.ndarray],
    cfg: dict[str, Any],
    output_dir: Path,
    worker_count: int,
) -> list[dict[str, Any]]:
    """Run MuJoCo tasks in child processes and serve all policy calls in this process.

    Each child owns one persistent environment for one task/episode shard.  The
    parent owns the only policy copy on the GPU and dynamically batches requests
    arriving over IPC.  Tasks are processed in waves when task_ids exceeds
    worker_count, while episode shards of every active task run together.
    """
    context = mp.get_context("spawn")
    adapter_enabled = bool(infer.policy.config.vla_adapter_enable)
    image_feature_keys = [str(key) for key in infer.policy.config.image_features]
    max_batch_size = max(1, int(cfg["inference_batch_size"]))
    batch_wait_s = max(0.0, float(cfg["inference_batch_wait_ms"])) / 1000.0
    partial_summaries: list[dict[str, Any]] = []
    request_count = 0
    batch_count = 0
    max_observed_batch = 0
    infrastructure_failures: list[str] = []
    episode_workers_per_task = min(
        max(1, int(cfg.get("episode_workers_per_task", 1))),
        max(1, int(cfg["episodes"])),
    )

    for wave_start in range(0, len(task_ids), max(1, int(worker_count))):
        wave_task_ids = [int(task_id) for task_id in task_ids[wave_start : wave_start + worker_count]]
        jobs = _build_episode_worker_jobs(
            wave_task_ids,
            episode_count=int(cfg["episodes"]),
            episode_workers_per_task=episode_workers_per_task,
        )
        request_queue = context.Queue(maxsize=max(2, len(jobs) * 2))
        ready_queue = context.Queue()
        result_queue = context.Queue()
        start_event = context.Event()
        response_queues = {
            job.worker_id: context.Queue(maxsize=1) for job in jobs
        }
        processes: dict[int, Any] = {}
        job_by_worker = {job.worker_id: job for job in jobs}

        worker_environment = {
            "SONG_LIBERO_ENV_WORKER": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
        }
        previous_worker_environment = {
            key: os.environ.get(key) for key in worker_environment
        }
        os.environ.update(worker_environment)
        try:
            for job in jobs:
                process = context.Process(
                    target=_process_task_worker_entry,
                    kwargs={
                        "worker_id": job.worker_id,
                        "suite_name": suite_name,
                        "task_id": job.task_id,
                        "shard_index": job.shard_index,
                        "episode_indices": list(job.episode_indices),
                        "task_spec": task_specs_by_id[job.task_id],
                        "task_init_states": task_init_states_by_id[job.task_id],
                        "cfg": cfg,
                        "output_dir": output_dir,
                        "request_queue": request_queue,
                        "response_queue": response_queues[job.worker_id],
                        "ready_queue": ready_queue,
                        "result_queue": result_queue,
                        "start_event": start_event,
                        "vla_adapter_enable": adapter_enabled,
                        "image_feature_keys": image_feature_keys,
                    },
                    name=(
                        f"libero-{suite_name}-task-{job.task_id}-"
                        f"episode-shard-{job.shard_index}"
                    ),
                )
                process.start()
                processes[job.worker_id] = process
        finally:
            for key, previous_value in previous_worker_environment.items():
                if previous_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous_value

        ready_workers: set[int] = set()
        finished_workers: set[int] = set()
        worker_summaries: dict[int, dict[str, Any]] = {}

        def _record_result(message: tuple[Any, ...]) -> None:
            status = str(message[0])
            worker_id = int(message[1])
            task_id = int(message[2])
            shard_index = int(message[3])
            if worker_id in finished_workers:
                return
            job = job_by_worker[worker_id]
            if task_id != job.task_id or shard_index != job.shard_index:
                raise RuntimeError(
                    f"Worker {worker_id} returned task/shard ({task_id}, {shard_index}), "
                    f"expected ({job.task_id}, {job.shard_index})."
                )
            if status == "ok":
                summary = message[4]
            else:
                error_repr = str(message[4])
                error_traceback = str(message[5])
                infrastructure_failures.append(
                    f"suite={suite_name} task={task_id} shard={shard_index} "
                    f"episodes={list(job.episode_indices)}: {error_repr}"
                )
                print(
                    f"[warn] process worker failed suite={suite_name} task={task_id} "
                    f"shard={shard_index} episodes={list(job.episode_indices)}: "
                    f"{error_repr}\n{error_traceback}",
                    flush=True,
                )
                summary = failed_task_summary(
                    suite_name=suite_name,
                    task_id=task_id,
                    episode_count=int(cfg["episodes"]),
                    exc=RuntimeError(error_repr),
                    episode_indices=list(job.episode_indices),
                )
            worker_summaries[worker_id] = summary
            finished_workers.add(worker_id)

        def _drain_results() -> None:
            while True:
                try:
                    _record_result(result_queue.get_nowait())
                except queue.Empty:
                    break

        def _record_dead_workers() -> None:
            for worker_id, process in processes.items():
                if worker_id in finished_workers or process.is_alive() or process.exitcode is None:
                    continue
                job = job_by_worker[worker_id]
                task_id = job.task_id
                exc = RuntimeError(
                    f"Environment worker exited without a result (exitcode={process.exitcode})."
                )
                infrastructure_failures.append(
                    f"suite={suite_name} task={task_id} shard={job.shard_index} "
                    f"episodes={list(job.episode_indices)}: {exc}"
                )
                print(
                    f"[warn] process worker died suite={suite_name} task={task_id} "
                    f"shard={job.shard_index} episodes={list(job.episode_indices)}: {exc}",
                    flush=True,
                )
                worker_summaries[worker_id] = failed_task_summary(
                    suite_name=suite_name,
                    task_id=task_id,
                    episode_count=int(cfg["episodes"]),
                    exc=exc,
                    episode_indices=list(job.episode_indices),
                )
                finished_workers.add(worker_id)

        try:
            # No task starts policy rollout before every successfully-created
            # environment in this wave is ready.  This removes the old startup
            # artifact where task 0 could run several episodes while other
            # tasks were still serially constructing EGL contexts.
            while len(ready_workers | finished_workers) < len(jobs):
                try:
                    ready_workers.add(int(ready_queue.get(timeout=0.1)))
                except queue.Empty:
                    pass
                _drain_results()
                _record_dead_workers()
            print(
                f"[parallel] suite={suite_name}: environments ready "
                f"{len(ready_workers)}/{len(jobs)} for {len(wave_task_ids)} tasks; "
                "starting rollout wave",
                flush=True,
            )
            start_event.set()

            while len(finished_workers) < len(jobs):
                _drain_results()
                _record_dead_workers()
                if len(finished_workers) >= len(jobs):
                    break
                try:
                    first_request = request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                requests = _collect_process_request_batch(
                    request_queue,
                    first_request,
                    max_batch_size=max_batch_size,
                    batch_wait_s=batch_wait_s,
                )
                _execute_process_inference_batch(infer, requests, response_queues)
                request_count += len(requests)
                batch_count += 1
                max_observed_batch = max(max_observed_batch, len(requests))
        except BaseException:
            start_event.set()
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            start_event.set()
            for process in processes.values():
                process.join(timeout=30.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)

            for ipc_queue in (
                request_queue,
                ready_queue,
                result_queue,
                *response_queues.values(),
            ):
                try:
                    ipc_queue.close()
                    ipc_queue.join_thread()
                except Exception:
                    pass

        partial_summaries.extend(worker_summaries[index] for index in sorted(worker_summaries))

    mean_batch = request_count / batch_count if batch_count else 0.0
    print(
        "[parallel] process inference batches: "
        f"requests={request_count}, batches={batch_count}, "
        f"mean_batch={mean_batch:.2f}, max_batch={max_observed_batch}",
        flush=True,
    )
    if infrastructure_failures:
        preview = "\n".join(infrastructure_failures[:10])
        if len(infrastructure_failures) > 10:
            preview += f"\n... and {len(infrastructure_failures) - 10} more worker failures"
        error_message = (
            "Parallel LIBERO environment workers failed before completing evaluation:\n" + preview
        )
        mark_realtime_suite_failed(
            output_dir=output_dir,
            suite_name=suite_name,
            error=error_message,
        )
        raise RuntimeError(error_message)
    return merge_task_episode_shards(
        suite_name=suite_name,
        task_ids=task_ids,
        episode_count=int(cfg["episodes"]),
        partial_summaries=partial_summaries,
    )


def _isolated_policy_worker_entry(
    *,
    worker_id: int,
    suite_name: str,
    task_ids: list[int],
    cfg: dict[str, Any],
    output_dir: Path,
    result_queue: Any,
) -> None:
    """Evaluate a fixed task shard with a private policy and private RNG state."""
    try:
        # The spawn bootstrap intentionally skips the heavyweight policy import
        # at module import time. Import it only inside the process that owns it.
        if __package__ and __package__.startswith("benchmarks."):
            from ..smolvla_model_inference import SmolVLA_ModelInference as WorkerInference
        else:
            from smolvla_model_inference import SmolVLA_ModelInference as WorkerInference

        ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
        configure_torch_determinism(bool(cfg.get("deterministic_torch", True)))

        from libero.libero import benchmark

        infer = WorkerInference(
            policy_path=cfg["policy_path"],
            policy_repo_id=cfg.get("policy_repo_id"),
            device=cfg["device"],
            visualize_foreground=False,
            foreground_visualizer_max_points=int(cfg["foreground_vis_max_points"]),
        )
        suite = benchmark.get_benchmark_dict()[suite_name]()
        summaries: list[dict[str, Any]] = []
        for task_id in task_ids:
            summaries.append(
                evaluate_task(
                    infer=infer,
                    suite=suite,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    cfg=cfg,
                    output_dir=output_dir,
                )
            )
        result_queue.put(("ok", int(worker_id), summaries))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                int(worker_id),
                [int(task_id) for task_id in task_ids],
                repr(exc),
                traceback.format_exc(),
            )
        )


def evaluate_suite_isolated_policy_processes(
    *,
    suite_name: str,
    task_ids: list[int],
    cfg: dict[str, Any],
    output_dir: Path,
    worker_count: int,
) -> list[dict[str, Any]]:
    """Run disjoint task shards in processes that each own a policy copy.

    Unlike the dynamic-batch path, no observation from one task is combined
    with another task. A worker serially evaluates its assigned tasks, so each
    model has an independent flow-matching RNG and CUDA execution history.
    """
    worker_count = min(max(1, int(worker_count)), max(1, len(task_ids)))
    task_shards = [
        [int(task_id) for task_id in task_ids[worker_id::worker_count]]
        for worker_id in range(worker_count)
    ]
    task_shards = [shard for shard in task_shards if shard]
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes: dict[int, Any] = {}

    worker_environment = {
        "SONG_LIBERO_ISOLATED_POLICY_WORKER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }
    previous_environment = {key: os.environ.get(key) for key in worker_environment}
    os.environ.update(worker_environment)
    try:
        for worker_id, worker_task_ids in enumerate(task_shards):
            process = context.Process(
                target=_isolated_policy_worker_entry,
                kwargs={
                    "worker_id": int(worker_id),
                    "suite_name": suite_name,
                    "task_ids": worker_task_ids,
                    "cfg": cfg,
                    "output_dir": output_dir,
                    "result_queue": result_queue,
                },
                name=f"libero-{suite_name}-isolated-policy-{worker_id}",
            )
            process.start()
            processes[int(worker_id)] = process
    finally:
        for key, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value

    print(
        f"[isolated-policy] suite={suite_name}: started {len(processes)} independent model "
        f"processes with task shards={task_shards}",
        flush=True,
    )
    finished: set[int] = set()
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        while len(finished) < len(processes):
            try:
                message = result_queue.get(timeout=0.2)
            except queue.Empty:
                message = None
            if message is not None:
                status = str(message[0])
                worker_id = int(message[1])
                if worker_id in finished:
                    continue
                if status == "ok":
                    summaries.extend(message[2])
                else:
                    worker_task_ids = [int(task_id) for task_id in message[2]]
                    error_repr = str(message[3])
                    error_traceback = str(message[4])
                    failures.append(
                        f"worker={worker_id} tasks={worker_task_ids}: {error_repr}"
                    )
                    print(
                        f"[warn] isolated policy worker failed suite={suite_name} "
                        f"worker={worker_id} tasks={worker_task_ids}: "
                        f"{error_repr}\n{error_traceback}",
                        flush=True,
                    )
                    summaries.extend(
                        failed_task_summary(
                            suite_name=suite_name,
                            task_id=task_id,
                            episode_count=int(cfg["episodes"]),
                            exc=RuntimeError(error_repr),
                        )
                        for task_id in worker_task_ids
                    )
                finished.add(worker_id)

            for worker_id, process in processes.items():
                if worker_id in finished or process.is_alive() or process.exitcode is None:
                    continue
                worker_task_ids = task_shards[worker_id]
                error = (
                    f"worker={worker_id} tasks={worker_task_ids} exited without a result "
                    f"(exitcode={process.exitcode})"
                )
                failures.append(error)
                summaries.extend(
                    failed_task_summary(
                        suite_name=suite_name,
                        task_id=task_id,
                        episode_count=int(cfg["episodes"]),
                        exc=RuntimeError(error),
                    )
                    for task_id in worker_task_ids
                )
                finished.add(worker_id)
    except BaseException:
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes.values():
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    summaries.sort(key=lambda item: int(item["task_id"]))
    if failures:
        error_message = "Isolated policy workers failed:\n" + "\n".join(failures)
        mark_realtime_suite_failed(
            output_dir=output_dir,
            suite_name=suite_name,
            error=error_message,
        )
        raise RuntimeError(error_message)
    return summaries


def prepare_config(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], Path]:
    cfg = load_config(args.config)
    cfg.setdefault("control", {})

    cfg["policy_path"] = cfg_get(cfg, args.policy_path, "policy_path")
    cfg["policy_repo_id"] = cfg_get(cfg, args.policy_repo_id, "policy_repo_id")
    cfg["evaluation_identity"] = collect_evaluation_identity(cfg["policy_path"])
    config_path = Path(args.config).expanduser().resolve()
    cfg["evaluation_identity"]["eval_config_path"] = str(config_path)
    cfg["evaluation_identity"]["eval_config_sha256"] = _sha256_file(config_path)
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes", 1))
    cfg["env_seed"] = int(cfg_get(cfg, args.env_seed, "env_seed", 7))
    cfg["device"] = cfg_get(cfg, args.device, "device", "cuda")
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points", 4096))
    cfg["observation_height"] = int(cfg_get(cfg, args.observation_height, "observation_height", 128))
    cfg["observation_width"] = int(cfg_get(cfg, args.observation_width, "observation_width", 128))
    cfg["render_mode"] = cfg_get(cfg, args.render_mode, "render_mode", "offscreen")
    cfg["render_camera"] = cfg_get(cfg, args.render_camera, "render_camera", "agentview")
    cfg["render_every_n_steps"] = int(cfg_get(cfg, args.render_every_n_steps, "render_every_n_steps", 1))
    cfg["render_gpu_device_id"] = int(cfg_get(cfg, args.render_gpu_device_id, "render_gpu_device_id", -1))
    cfg["save_video"] = bool(cfg_get(cfg, args.save_video, "save_video", True))
    cfg["visualize_foreground"] = bool(
        cfg_get(cfg, args.visualize_foreground, "visualize_foreground", False)
    )
    cfg["foreground_vis_max_points"] = int(
        cfg_get(cfg, args.foreground_vis_max_points, "foreground_vis_max_points", 50000)
    )
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
    cfg["use_suite_max_steps"] = bool(
        cfg_get(cfg, args.use_suite_max_steps, "use_suite_max_steps", True)
    )
    cfg["control"]["warmup_steps"] = int(cfg_get(cfg["control"], args.warmup_steps, "warmup_steps", 0))
    cfg["policy_noise_seed"] = int(cfg_get(cfg, args.policy_noise_seed, "policy_noise_seed", 0))
    cfg["deterministic_torch"] = bool(
        cfg_get(cfg, args.deterministic_torch, "deterministic_torch", True)
    )
    cfg["torch_determinism"] = {
        "enabled": bool(cfg["deterministic_torch"]),
        "configured_in_policy_process": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    cfg["evaluation_identity"]["torch_determinism"] = cfg["torch_determinism"]
    cfg["control"]["gripper_threshold"] = float(
        cfg_get(cfg["control"], args.gripper_threshold, "gripper_threshold", 0.5)
    )
    cfg["control"]["gripper_control_mode"] = str(
        cfg_get(cfg["control"], args.gripper_control_mode, "gripper_control_mode", "delta_width")
    )
    cfg["control"]["gripper_delta_threshold"] = float(
        cfg_get(cfg["control"], args.gripper_delta_threshold, "gripper_delta_threshold", 0.003)
    )
    cfg["control"]["gripper_target_tolerance"] = float(
        cfg_get(cfg["control"], args.gripper_target_tolerance, "gripper_target_tolerance", 0.004)
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
    cfg["task_workers"] = max(1, int(cfg_get(cfg, args.task_workers, "task_workers", 1)))
    cfg["isolated_policy_workers"] = max(
        1,
        int(
            cfg_get(
                cfg,
                args.isolated_policy_workers,
                "isolated_policy_workers",
                1,
            )
        ),
    )
    cfg["episode_workers_per_task"] = max(
        1,
        int(
            cfg_get(
                cfg,
                args.episode_workers_per_task,
                "episode_workers_per_task",
                1,
            )
        ),
    )
    configured_environment_workers = (
        cfg["task_workers"]
        * min(cfg["episode_workers_per_task"], max(1, cfg["episodes"]))
    )
    cfg["task_worker_backend"] = str(
        cfg_get(
            cfg,
            args.task_worker_backend,
            "task_worker_backend",
            "process" if configured_environment_workers > 1 else "thread",
        )
    ).lower()
    cfg["recreate_env_per_episode"] = bool(
        cfg_get(cfg, args.recreate_env_per_episode, "recreate_env_per_episode", False)
    )
    cfg["settle_steps"] = int(cfg_get(cfg, args.settle_steps, "settle_steps", 0))
    cfg["settle_min_seconds"] = float(
        cfg_get(cfg, args.settle_min_seconds, "settle_min_seconds", 0.25)
    )
    cfg["settle_stable_seconds"] = float(
        cfg_get(cfg, args.settle_stable_seconds, "settle_stable_seconds", 0.25)
    )
    cfg["settle_max_seconds"] = float(
        cfg_get(cfg, args.settle_max_seconds, "settle_max_seconds", 8.0)
    )
    cfg["settle_linear_velocity_threshold"] = float(
        cfg_get(cfg, args.settle_linear_velocity_threshold, "settle_linear_velocity_threshold", 0.01)
    )
    cfg["settle_angular_velocity_threshold"] = float(
        cfg_get(cfg, args.settle_angular_velocity_threshold, "settle_angular_velocity_threshold", 0.05)
    )
    cfg["settle_other_dof_velocity_threshold"] = float(
        cfg_get(cfg, args.settle_other_dof_velocity_threshold, "settle_other_dof_velocity_threshold", 0.02)
    )
    cfg["settle_require_stable"] = bool(
        cfg_get(cfg, args.settle_require_stable, "settle_require_stable", True)
    )
    cfg["settle_keep_robot_fixed"] = bool(
        cfg_get(cfg, args.settle_keep_robot_fixed, "settle_keep_robot_fixed", True)
    )
    cfg["initial_gripper_open"] = bool(
        cfg_get(cfg, args.initial_gripper_open, "initial_gripper_open", True)
    )
    cfg["inference_batch_size"] = max(
        1,
        int(
            cfg_get(
                cfg,
                args.inference_batch_size,
                "inference_batch_size",
                configured_environment_workers,
            )
        ),
    )
    cfg["inference_batch_wait_ms"] = max(
        0.0,
        float(cfg_get(cfg, args.inference_batch_wait_ms, "inference_batch_wait_ms", 5.0)),
    )
    cfg["keyboard_control_enabled"] = configured_environment_workers == 1
    if configured_environment_workers > 1:
        if cfg["task_worker_backend"] not in {"process", "thread"}:
            raise ValueError(
                "task_worker_backend must be either 'process' or 'thread', got "
                f"{cfg['task_worker_backend']!r}."
            )
        if str(cfg.get("render_mode", "offscreen")).lower() != "offscreen":
            raise ValueError("Parallel environment workers require --render-mode offscreen.")
        if bool(cfg.get("visualize_foreground", False)):
            raise ValueError("Parallel environment workers require --no-visualize-foreground.")
    if cfg["episode_workers_per_task"] > 1 and cfg["task_worker_backend"] != "process":
        raise ValueError("--episode-workers-per-task > 1 requires --task-worker-backend process.")
    if cfg["isolated_policy_workers"] > 1 and cfg["episode_workers_per_task"] > 1:
        raise ValueError(
            "--isolated-policy-workers > 1 already provides independent rollout processes; "
            "combine it with --episode-workers-per-task 1."
        )
    if cfg["isolated_policy_workers"] > 1 and bool(cfg.get("visualize_foreground", False)):
        raise ValueError("Isolated policy workers require --no-visualize-foreground.")

    return cfg, suite_names, output_dir


def _strip_cli_options(argv: list[str], option_names: set[str]) -> list[str]:
    """Remove repeatable value-taking options from argv, including --name=value."""
    result: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        option = argument.split("=", 1)[0]
        if option not in option_names:
            result.append(argument)
            index += 1
            continue
        if "=" not in argument:
            index += 2
        else:
            index += 1
    return result


def run_multi_gpu_suite_launcher(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    suite_names: list[str],
    output_dir: Path,
) -> None:
    gpu_ids = [item.strip() for item in str(args.suite_gpu_ids).split(",") if item.strip()]
    if len(gpu_ids) != len(suite_names):
        raise ValueError(
            f"--suite-gpu-ids supplied {len(gpu_ids)} ids for {len(suite_names)} suites: "
            f"gpu_ids={gpu_ids}, suites={suite_names}."
        )
    if len(set(suite_names)) != len(suite_names):
        raise ValueError(f"Multi-GPU suite names must be unique, got {suite_names}.")

    base_argv = _strip_cli_options(
        list(sys.argv[1:]),
        {
            "--suite",
            "--suite-gpu-ids",
            "--output-dir",
            "--device",
            "--render-gpu-device-id",
        },
    )
    processes: list[tuple[str, str, subprocess.Popen[Any]]] = []
    print(
        "[multi-gpu] launching one model service per suite: "
        + ", ".join(f"{suite}->GPU{gpu_id}" for suite, gpu_id in zip(suite_names, gpu_ids, strict=True)),
        flush=True,
    )
    try:
        for suite_name, gpu_id in zip(suite_names, gpu_ids, strict=True):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *base_argv,
                "--suite",
                suite_name,
                "--output-dir",
                str(output_dir),
                "--device",
                "cuda",
                "--render-gpu-device-id",
                str(gpu_id),
            ]
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # This robosuite / MuJoCo stack interprets both variables as
            # physical EGL device ids and explicitly requires the EGL id to be
            # present in CUDA_VISIBLE_DEVICES.  Torch still sees this one
            # physical card as logical cuda:0 inside the child process.
            child_env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
            child_env["SONG_LIBERO_SUITE_WORKER"] = "1"
            process = subprocess.Popen(command, env=child_env, cwd=str(Path.cwd()))
            processes.append((suite_name, gpu_id, process))

        return_codes = {
            suite_name: process.wait() for suite_name, _gpu_id, process in processes
        }
    except BaseException:
        for _suite_name, _gpu_id, process in processes:
            if process.poll() is None:
                process.terminate()
        for _suite_name, _gpu_id, process in processes:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
        raise

    all_task_summaries: list[dict[str, Any]] = []
    for suite_name in suite_names:
        report_path = output_dir / suite_name / "suite_report.json"
        if return_codes[suite_name] == 0 and report_path.is_file():
            with open(report_path, encoding="utf-8") as report_file:
                suite_report = json.load(report_file)
            all_task_summaries.extend(suite_report.get("tasks", []))
    all_task_summaries.sort(
        key=lambda item: (
            suite_names.index(str(item["suite"])),
            int(item["task_id"]),
        )
    )
    write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
    print(json.dumps(aggregate_task_results(all_task_summaries), indent=2, ensure_ascii=False))

    failed = [
        f"{suite_name}(GPU{gpu_id}, exit={return_codes[suite_name]})"
        for suite_name, gpu_id, _process in processes
        if return_codes[suite_name] != 0
    ]
    if failed:
        raise RuntimeError("Multi-GPU suite evaluation failed: " + ", ".join(failed))


def main() -> None:
    args = parse_args()
    cfg, suite_names, output_dir = prepare_config(args)
    if args.suite_gpu_ids is not None:
        run_multi_gpu_suite_launcher(
            args=args,
            cfg=cfg,
            suite_names=suite_names,
            output_dir=output_dir,
        )
        return
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark
    episode_horizons = {
        suite_name: (
            int(LIBERO_STANDARD_MAX_STEPS[suite_name])
            if bool(cfg.get("use_suite_max_steps", True))
            and suite_name in LIBERO_STANDARD_MAX_STEPS
            else int(cfg["control"]["max_steps"])
        )
        for suite_name in suite_names
    }

    if int(cfg.get("isolated_policy_workers", 1)) > 1:
        benchmark_dict = benchmark.get_benchmark_dict()
        all_task_summaries: list[dict[str, Any]] = []
        print(
            "[info] isolated-policy evaluation: "
            f"suites={suite_names}, independent_models_per_gpu={cfg['isolated_policy_workers']}, "
            f"episodes={cfg['episodes']}, inference_batch_size=1, "
            f"exec_action_steps={cfg['control']['exec_action_steps']}, "
            f"policy_noise_seed={cfg['policy_noise_seed']}, "
            f"episode_horizons={episode_horizons}",
            flush=True,
        )
        for suite_name in suite_names:
            suite = benchmark_dict[suite_name]()
            task_ids = resolve_task_ids_for_suite(
                suite_name=suite_name,
                task_count=len(suite.tasks),
                cli_task_ids=args.task_id,
                cfg=cfg,
            )
            initialize_realtime_suite_progress(
                output_dir=output_dir,
                suite_name=suite_name,
                task_ids=[int(task_id) for task_id in task_ids],
                episodes_per_task=int(cfg["episodes"]),
            )
            suite_summaries = evaluate_suite_isolated_policy_processes(
                suite_name=suite_name,
                task_ids=[int(task_id) for task_id in task_ids],
                cfg=cfg,
                output_dir=output_dir,
                worker_count=int(cfg["isolated_policy_workers"]),
            )
            all_task_summaries.extend(suite_summaries)
            all_task_summaries.sort(
                key=lambda item: (
                    suite_names.index(str(item["suite"])),
                    int(item["task_id"]),
                )
            )
            write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
        print(json.dumps(aggregate_task_results(all_task_summaries), indent=2, ensure_ascii=False))
        return

    cfg["torch_determinism"] = configure_torch_determinism(bool(cfg["deterministic_torch"]))
    cfg["evaluation_identity"]["torch_determinism"] = cfg["torch_determinism"]
    infer = SmolVLA_ModelInference(
        policy_path=cfg["policy_path"],
        policy_repo_id=cfg.get("policy_repo_id"),
        device=cfg["device"],
        visualize_foreground=cfg["visualize_foreground"],
        foreground_visualizer_max_points=cfg["foreground_vis_max_points"],
    )

    print(
        "[info] clean absolute-pose eval: "
        f"suites={suite_names}, episodes={cfg['episodes']}, "
        f"task_workers={cfg['task_workers']}, "
        f"isolated_policy_workers={cfg['isolated_policy_workers']}, "
        f"episode_workers_per_task={cfg['episode_workers_per_task']}, "
        f"task_worker_backend={cfg['task_worker_backend']}, "
        f"inference_batch_size={cfg['inference_batch_size']}, "
        f"recreate_env_per_episode={cfg['recreate_env_per_episode']}, "
        f"exec_action_steps={cfg['control']['exec_action_steps']}, "
        f"gripper_threshold={cfg['control']['gripper_threshold']}, "
        f"gripper_control_mode={cfg['control']['gripper_control_mode']}, "
        f"gripper_delta_threshold={cfg['control']['gripper_delta_threshold']}, "
        f"gripper_target_tolerance={cfg['control']['gripper_target_tolerance']}, "
        f"policy_noise_seed={cfg['policy_noise_seed']}, "
        f"deterministic_torch={cfg['deterministic_torch']}, "
        f"env_seed={cfg['env_seed']}, "
        f"use_suite_max_steps={cfg['use_suite_max_steps']}, "
        f"episode_horizons={episode_horizons}, "
        f"save_video={cfg['save_video']}, "
        f"render_mode={cfg.get('render_mode')}, "
        f"render_every_n_steps={cfg.get('render_every_n_steps')}, "
        f"visualize_foreground={cfg.get('visualize_foreground')}"
    )

    all_task_summaries: list[dict[str, Any]] = []
    benchmark_dict = benchmark.get_benchmark_dict()
    scheduler: BatchedInferenceScheduler | None = None
    eval_infer: Any = infer
    if int(cfg["task_workers"]) > 1 and str(cfg["task_worker_backend"]) == "thread":
        scheduler = BatchedInferenceScheduler(
            infer,
            max_batch_size=int(cfg["inference_batch_size"]),
            batch_wait_ms=float(cfg["inference_batch_wait_ms"]),
        )
        eval_infer = scheduler

    try:
        for suite_name in suite_names:
            suite = benchmark_dict[suite_name]()
            task_ids = resolve_task_ids_for_suite(
                suite_name=suite_name,
                task_count=len(suite.tasks),
                cli_task_ids=args.task_id,
                cfg=cfg,
            )
            initialize_realtime_suite_progress(
                output_dir=output_dir,
                suite_name=suite_name,
                task_ids=[int(task_id) for task_id in task_ids],
                episodes_per_task=int(cfg["episodes"]),
            )
            worker_count = min(int(cfg["task_workers"]), max(1, len(task_ids)))
            episode_worker_count = min(
                int(cfg["episode_workers_per_task"]),
                max(1, int(cfg["episodes"])),
            )
            environment_worker_count = worker_count * episode_worker_count

            if environment_worker_count == 1:
                for task_id in task_ids:
                    summary = evaluate_task(
                        infer=eval_infer,
                        suite=suite,
                        suite_name=suite_name,
                        task_id=int(task_id),
                        cfg=cfg,
                        output_dir=output_dir,
                    )
                    all_task_summaries.append(summary)
                    write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
                continue

            if str(cfg["task_worker_backend"]) == "process":
                task_specs_by_id = {
                    int(task_id): serialize_libero_task(suite.get_task(int(task_id)))
                    for task_id in task_ids
                }
                task_init_states_by_id = {
                    int(task_id): init_states_as_numpy(get_task_init_states(suite, int(task_id)))
                    for task_id in task_ids
                }
                print(
                    f"[parallel] suite={suite_name}: starting up to {environment_worker_count} "
                    f"MuJoCo processes ({worker_count} tasks x {episode_worker_count} episode shards) "
                    f"for task_ids={list(map(int, task_ids))}; policy remains in the parent process",
                    flush=True,
                )
                suite_summaries = evaluate_suite_process_parallel(
                    infer=infer,
                    suite_name=suite_name,
                    task_ids=[int(task_id) for task_id in task_ids],
                    task_specs_by_id=task_specs_by_id,
                    task_init_states_by_id=task_init_states_by_id,
                    cfg=cfg,
                    output_dir=output_dir,
                    worker_count=worker_count,
                )
                all_task_summaries.extend(suite_summaries)
                all_task_summaries.sort(
                    key=lambda item: (
                        suite_names.index(str(item["suite"])),
                        int(item["task_id"]),
                    )
                )
                write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
                continue

            print(
                f"[parallel] suite={suite_name}: starting {worker_count} compatibility task threads "
                f"for task_ids={list(map(int, task_ids))}",
                flush=True,
            )
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=f"{suite_name}-task",
            ) as executor:
                futures = {
                    executor.submit(
                        evaluate_task,
                        infer=eval_infer,
                        suite=suite,
                        suite_name=suite_name,
                        task_id=int(task_id),
                        cfg=cfg,
                        output_dir=output_dir,
                    ): int(task_id)
                    for task_id in task_ids
                }
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        summary = future.result()
                    except BaseException as exc:
                        print(
                            f"[warn] task worker failed suite={suite_name} task={task_id}: {exc!r}",
                            flush=True,
                        )
                        summary = failed_task_summary(
                            suite_name=suite_name,
                            task_id=task_id,
                            episode_count=int(cfg["episodes"]),
                            exc=exc,
                        )
                    all_task_summaries.append(summary)
                    all_task_summaries.sort(
                        key=lambda item: (
                            suite_names.index(str(item["suite"])),
                            int(item["task_id"]),
                        )
                    )
                    write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
    finally:
        if scheduler is not None:
            scheduler.close()
        infer.close()

    write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
    print(json.dumps(aggregate_task_results(all_task_summaries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
