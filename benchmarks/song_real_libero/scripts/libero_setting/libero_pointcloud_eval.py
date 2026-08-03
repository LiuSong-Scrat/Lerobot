# Example: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py --config benchmarks/song_real_libero/configs/libero.json
#!/usr/bin/env python
"""Minimal LIBERO point-cloud evaluation with absolute-pose chunk execution.

Kept pipeline:
  observation -> point cloud -> model action chunk -> absolute OSC pose actions
  -> execute selected chunk rows -> optional videos + JSON summaries.

Removed from the original evaluator:
  pose/gripper wait state machines, fast-physics executor, rim correction,
  heavy timing diagnostics, and delta-pose execution branches.
"""
from __future__ import annotations

import argparse
import atexit
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
_EVALUATION_RUN_LOCK_FILE: Any | None = None
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

# A benchmark initial state is evaluated by exactly one uninterrupted rollout.
# Frequency and horizon may be configured, but failures are never reset and
# retried, and a model call never samples several chunks and selects among them.
FAIR_EVALUATION_PROTOCOL = {
    "name": "single_uninterrupted_rollout",
    "rollouts_per_initial_state": 1,
    "retry_failed_rollout": False,
    "action_samples_per_model_call": 1,
    "action_sample_selection": "none",
    "initial_state_source": "task_suite.get_task_init_states",
    "fixture_reset_sequence": "seeded_serial_episode_index",
    "benchmark_comparable": True,
}


def evaluation_protocol_for_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Describe whether this is a standard benchmark or a source-demo domain diagnostic."""

    if not bool(cfg.get("dataset_domain_env", False)):
        return dict(FAIR_EVALUATION_PROTOCOL)
    state_offset = int(cfg.get("dataset_domain_state_observation_offset", 1))
    oracle_actions = bool(cfg.get("dataset_domain_oracle_actions", False))
    initial_state_index = 0 if oracle_actions else state_offset
    return {
        **FAIR_EVALUATION_PROTOCOL,
        "name": "dataset_domain_source_demo_rollout",
        "initial_state_source": f"source_demo_hdf5.states[{initial_state_index}]",
        "fixture_reset_sequence": "source_demo_hdf5.model_file_per_episode",
        "environment_domain": "training_source_demo",
        "post_state_settling": "disabled_to_preserve_source_observation",
        "forced_initial_gripper_open": False,
        "pre_policy_warmup_steps": 0,
        "action_source": (
            "source_demo_raw_delta_to_source_anchored_absolute_osc"
            if oracle_actions
            else "policy_flow_matching_sample"
        ),
        "benchmark_comparable": False,
        "diagnostic_only": True,
    }


def acquire_evaluation_run_lock(output_dir: Path) -> None:
    """Prevent two top-level evaluators from corrupting one output directory.

    Multi-GPU suite children intentionally share their launcher's output root,
    so only the top-level launcher (or a normal single-process run) owns this
    lock. The file descriptor remains open until process exit.
    """
    global _EVALUATION_RUN_LOCK_FILE
    if os.environ.get("SONG_LIBERO_SUITE_WORKER", "0") == "1":
        return
    if _EVALUATION_RUN_LOCK_FILE is not None:
        return

    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".evaluation_run.lock"
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown owner"
        lock_file.close()
        raise RuntimeError(
            f"Evaluation output directory is already active: {output_dir}. "
            f"Lock owner: {owner}"
        ) from exc

    lock_file.seek(0)
    lock_file.truncate()
    json.dump(
        {
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "started_unix_s": time.time(),
            "argv": sys.argv,
        },
        lock_file,
        ensure_ascii=False,
    )
    lock_file.flush()
    os.fsync(lock_file.fileno())
    _EVALUATION_RUN_LOCK_FILE = lock_file

    def _release() -> None:
        global _EVALUATION_RUN_LOCK_FILE
        active_lock = _EVALUATION_RUN_LOCK_FILE
        if active_lock is None:
            return
        try:
            fcntl.flock(active_lock.fileno(), fcntl.LOCK_UN)
        finally:
            active_lock.close()
            _EVALUATION_RUN_LOCK_FILE = None

    atexit.register(_release)


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
    if cli_task_ids is not None:
        task_ids = cli_task_ids
    elif bool(cfg.get("all_tasks", False)):
        return list(range(task_count))
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


    ###################TrainDatasetTest##########
    parser.add_argument(
        "--dataset-domain-env",
        "--align-env-to-training-data",
        dest="dataset_domain_env",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic mode: map episode N to source demo N, restore that demo's HDF5 "
            "model_file, and initialize from the same state used by dataset conversion. "
            "Disabled by default and not comparable to the standard LIBERO benchmark."
        ),
    )
    parser.add_argument(
        "--dataset-domain-demo-root",
        type=Path,
        default=None,
        help=(
            "Root containing the official LIBERO HDF5 demonstrations used for training. "
            "Defaults to demo_root from the JSON config."
        ),
    )
    parser.add_argument(
        "--dataset-domain-oracle-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic mode: bypass policy inference, read the matched HDF5 demo's raw "
            "OSC delta actions, and combine each one with its matched source state to "
            "construct an absolute OSC setpoint while keeping controller.use_delta=False. Requires "
            "--dataset-domain-env and is not a benchmark score."
        ),
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
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        default=None,
        help="Evaluate only these LIBERO initial-state indices; repeat for multiple indices.",
    )
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
    parser.add_argument(
        "--adaptive-exec-max-steps",
        type=int,
        default=None,
        help=(
            "Maximum rows retained from a predicted chunk when the robot has not reached the final "
            "base waypoint. Values at or below --exec-action-steps disable adaptive continuation."
        ),
    )
    parser.add_argument(
        "--adaptive-exec-position-error-threshold",
        type=float,
        default=None,
        help=(
            "Continue beyond --exec-action-steps only while end-effector position tracking error "
            "exceeds this value in metres."
        ),
    )
    parser.add_argument(
        "--adaptive-exec-rotation-error-threshold",
        type=float,
        default=None,
        help=(
            "Continue beyond --exec-action-steps only while end-effector rotation tracking error "
            "exceeds this value in radians."
        ),
    )
    parser.add_argument(
        "--adaptive-exec-position-error-max",
        type=float,
        default=None,
        help=(
            "Do not continue an old chunk when position tracking error exceeds this safety bound "
            "in metres; replan immediately instead."
        ),
    )
    parser.add_argument(
        "--adaptive-exec-rotation-error-max",
        type=float,
        default=None,
        help=(
            "Do not continue an old chunk when rotation tracking error exceeds this safety bound "
            "in radians; replan immediately instead."
        ),
    )
    parser.add_argument(
        "--grasp-exec-steps",
        type=int,
        default=None,
        help=(
            "Rows executed from a chunk when the measured gripper width indicates a stable "
            "grasp. This lets placement reach deeper chunk rows without changing approach or "
            "empty-gripper replanning."
        ),
    )
    parser.add_argument("--grasp-width-min", type=float, default=None)
    parser.add_argument("--grasp-width-max", type=float, default=None)
    parser.add_argument(
        "--grasp-lift-threshold",
        type=float,
        default=None,
        help=(
            "Minimum upward end-effector displacement after an intermediate-width closure "
            "before using grasp_exec_steps. This distinguishes transported objects from "
            "fixed handles using robot state only."
        ),
    )
    parser.add_argument(
        "--release-event-exec-enable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When an intermediate-width grasp is present, preserve a significant predicted "
            "gripper-opening event that falls beyond the normal chunk cutoff."
        ),
    )
    parser.add_argument(
        "--release-event-exec-max-steps",
        type=int,
        default=None,
        help="Maximum predicted chunk rows that may be committed to complete a release event.",
    )
    parser.add_argument(
        "--release-event-min-width-change",
        type=float,
        default=None,
        help=(
            "Minimum predicted opening increase in metres required to preserve a release event."
        ),
    )
    parser.add_argument(
        "--waypoint-max-hold-steps",
        type=int,
        default=None,
        help=(
            "Maximum controller steps spent tracking each predicted waypoint before advancing. "
            "One preserves the original one-waypoint-per-env-step behavior."
        ),
    )
    parser.add_argument(
        "--waypoint-position-tolerance",
        type=float,
        default=None,
        help="Position error in metres below which a held waypoint is considered reached.",
    )
    parser.add_argument(
        "--waypoint-rotation-tolerance",
        type=float,
        default=None,
        help="Rotation error in radians below which a held waypoint is considered reached.",
    )
    parser.add_argument(
        "--waypoint-gripper-tolerance",
        type=float,
        default=None,
        help=(
            "Physical-width error in metres used when a held waypoint carries an open/close event."
        ),
    )
    parser.add_argument(
        "--rollback-chunks",
        type=int,
        default=None,
        help="Number of completed policy chunks rewound when non-blocking key 'r' is pressed.",
    )
    parser.add_argument(
        "--rollback-max-steps",
        type=int,
        default=None,
        help="Maximum LIBERO controller steps used to return to the rollback target pose.",
    )
    parser.add_argument(
        "--rollback-position-tolerance",
        type=float,
        default=None,
        help="Position tolerance in metres for completing a manual rollback.",
    )
    parser.add_argument(
        "--rollback-rotation-tolerance",
        type=float,
        default=None,
        help="Rotation tolerance in radians for completing a manual rollback.",
    )
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
        "--gripper-delta-alignment",
        choices=("current_minus_previous", "next_minus_current"),
        default=None,
        help=(
            "Align a predicted width transition with the waypoint being executed. "
            "current_minus_previous follows trajectory time; next_minus_current is the legacy one-row-early mode."
        ),
    )
    parser.add_argument(
        "--synchronize-gripper-controller-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Synchronize robosuite's integrated gripper.current_action with the physical finger qpos "
            "after reset/settling, so a zero command preserves the initialized opening."
        ),
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
        "--visualize-action-trajectory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Continuously show the exact model observation point cloud and predicted UMI pose9 "
            "trajectory in an isolated non-blocking viewer. In serial evaluation, press 'v' "
            "for a one-shot update even when this option is disabled."
        ),
    )
    parser.add_argument(
        "--trajectory-vis-max-points",
        type=int,
        default=None,
        help="Maximum combined scene/trajectory points sent to the online trajectory viewer.",
    )
    parser.add_argument(
        "--trajectory-vis-every-n-model-calls",
        type=int,
        default=None,
        help="Refresh continuous action-trajectory visualization every N policy calls.",
    )
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
        default=True,
        help="Rebuild a task environment before every episode (disabled by default for parallel evaluation).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return load_json_config(
        path,
        path_keys=(
            "policy_path",
            "policy_repo_id",
            "libero_config_path",
            "demo_root",
            "dataset_domain_demo_root",
            "output_dir",
            "vis_dir",
        ),
    )


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg.get(key, default)


def configured_demo_root(cfg: dict[str, Any]) -> Any:
    if bool(cfg.get("dataset_domain_env", False)):
        return cfg.get("dataset_domain_demo_root") or cfg.get("demo_root")
    return cfg.get("demo_root")


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


def libero_delta_action_to_absolute_action(
    env: Any,
    source_action: np.ndarray,
    *,
    force_controller_update: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one source OSC delta command to an equivalent absolute command.

    LIBERO demonstrations store normalized OSC deltas. A demonstrated state,
    however, is the pose reached *after* contact dynamics and is not the OSC
    setpoint encoded by the action that generated it. Replaying reached poses
    as controller commands therefore changes the source action semantics.

    This helper preserves an absolute-pose controller while reconstructing the
    setpoint that robosuite's delta branch encodes at the supplied controller
    state. No heuristic offset is added, and the source gripper command is
    copied unchanged.
    """

    raw_action = np.asarray(source_action, dtype=np.float64).reshape(-1)
    if raw_action.shape != (7,):
        raise ValueError(f"Expected one LIBERO source action with shape (7,), got {raw_action.shape}.")
    if not np.isfinite(raw_action).all():
        raise ValueError("LIBERO source action contains non-finite values.")
    if len(getattr(env, "robots", [])) != 1:
        raise ValueError("Source-action absolute conversion currently requires one LIBERO robot.")

    import robosuite.utils.transform_utils as T
    from robosuite.utils.control_utils import set_goal_orientation, set_goal_position

    controller = env.robots[0].controller
    if bool(getattr(controller, "use_delta", True)):
        raise RuntimeError(
            "libero_delta_action_to_absolute_action requires controller.use_delta=False."
        )
    # Normal online conversion mirrors OperationalSpaceController.set_goal().
    # Source-state reconstruction explicitly force-refreshes after setting each
    # saved MuJoCo state so the absolute target is anchored to that source pose.
    controller.update(force=bool(force_controller_update))
    scaled_delta = np.asarray(controller.scale_action(raw_action[:6]), dtype=np.float64)
    target_position = set_goal_position(
        scaled_delta[:3],
        np.asarray(controller.ee_pos, dtype=np.float64),
        position_limit=controller.position_limits,
    )
    target_orientation = set_goal_orientation(
        scaled_delta[3:],
        np.asarray(controller.ee_ori_mat, dtype=np.float64),
        orientation_limit=controller.orientation_limits,
    )

    target_controller_world = np.eye(4, dtype=np.float64)
    target_controller_world[:3, :3] = np.asarray(target_orientation, dtype=np.float64)
    target_controller_world[:3, 3] = np.asarray(target_position, dtype=np.float64)
    absolute_action = np.concatenate(
        [
            np.asarray(target_position, dtype=np.float64),
            # Use robosuite's own matrix -> quaternion -> axis-angle
            # convention. SciPy projects a slightly non-orthogonal MuJoCo
            # matrix onto SO(3); close to pi this changed one source target by
            # about 0.02 in matrix space and was enough to break contact replay.
            T.quat2axisangle(T.mat2quat(target_orientation)),
            raw_action[6:7],
        ]
    )
    return (
        absolute_action.astype(np.float64),
        target_controller_world.astype(np.float64),
        scaled_delta.astype(np.float64),
    )


def refresh_arm_controller_state(env: Any) -> None:
    """Refresh cached OSC kinematics after restoring a MuJoCo state."""

    for robot in env.robots:
        robot.controller.update(force=True)


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
    """Collect immediate non-blocking episode-control and debug requests."""

    def __init__(self) -> None:
        self._manual_failure = threading.Event()
        self._rollback_request = threading.Event()
        self._trajectory_visualization_request = threading.Event()
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

    def _request_rollback(self, source: str) -> None:
        if self._rollback_request.is_set():
            return
        self._rollback_request.set()
        print(
            f"[eval] received 'r' from {source}: rollback recent policy chunk(s)",
            flush=True,
        )

    def _request_trajectory_visualization(self, source: str) -> None:
        if self._trajectory_visualization_request.is_set():
            return
        self._trajectory_visualization_request.set()
        print(
            f"[eval] received 'v' from {source}: visualize the next predicted UMI trajectory",
            flush=True,
        )

    def viewer_key_callback(self, keycode: int) -> None:
        """MuJoCo passive-viewer callback (GLFW letter codes are ASCII-compatible)."""
        keycode = int(keycode)
        if keycode in (ord("n"), ord("N")):
            self._request_manual_failure("viewer")
        elif keycode in (ord("r"), ord("R")):
            self._request_rollback("viewer")
        elif keycode in (ord("v"), ord("V")):
            self._request_trajectory_visualization("viewer")

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
                    key = key.lower()
                    if key == b"n":
                        self._request_manual_failure("terminal")
                    elif key == b"r":
                        self._request_rollback("terminal")
                    elif key == b"v":
                        self._request_trajectory_visualization("terminal")
            except OSError:
                pass
        return self._manual_failure.is_set()

    def pop_rollback_request(self) -> bool:
        """Consume one latched rollback request without blocking."""
        self.poll()
        if not self._rollback_request.is_set():
            return False
        self._rollback_request.clear()
        return True

    def pop_trajectory_visualization_request(self) -> bool:
        """Consume one request to display the next complete model prediction."""
        self.poll()
        if not self._trajectory_visualization_request.is_set():
            return False
        self._trajectory_visualization_request.clear()
        return True

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


def predicted_release_event_end(
    predicted_widths: np.ndarray,
    *,
    base_count: int,
    max_count: int,
    min_width_change: float,
) -> tuple[int, float]:
    """Return an exclusive row count that preserves a delayed release event.

    Receding-horizon execution can otherwise create a Zeno failure: every
    prediction contains a valid opening event in its unused suffix, but the
    fixed cutoff replans before reaching it and the next prediction moves the
    event into the future again.  This helper only recognizes opening motion;
    approach and closing remain closed-loop so grasp refinement is unaffected.
    """
    widths = np.asarray(predicted_widths, dtype=np.float64).reshape(-1)
    if widths.size == 0:
        return 0, 0.0
    base = min(widths.size, max(1, int(base_count)))
    limit = min(widths.size, max(base, int(max_count)))
    if limit <= base:
        return base, 0.0

    reference_width = float(widths[base - 1])
    suffix = widths[base:limit]
    peak_offset = int(np.argmax(suffix))
    opening_change = float(suffix[peak_offset] - reference_width)
    if not np.isfinite(opening_change) or opening_change < float(min_width_change):
        return base, max(0.0, opening_change)
    return base + peak_offset + 1, opening_change


def synchronize_gripper_controller_state(env: Any) -> list[list[float]]:
    """Match robosuite's integrated gripper target to the physical finger qpos.

    ``PandaGripper.format_action`` integrates direction commands into an
    internal normalized ``current_action``. Directly setting finger qpos during
    episode initialization does not update that state. Its default is zero,
    which maps to a half-open actuator target; consequently a later zero command
    closes a physically open gripper toward half width instead of holding it.

    This routine derives the normalized actuator target from the actual joint
    positions and updates both the gripper model state and MuJoCo controls. It
    uses only actuator/joint metadata and therefore has no task-specific logic.
    """
    sim = get_sim(env)
    model = sim.model
    synchronized: list[list[float]] = []

    for robot in getattr(env, "robots", []):
        raw_grippers = getattr(robot, "gripper", None)
        if raw_grippers is None:
            continue
        grippers = list(raw_grippers.values()) if isinstance(raw_grippers, dict) else [raw_grippers]
        for gripper in grippers:
            actuator_names = list(getattr(gripper, "actuators", []))
            if not actuator_names:
                continue
            actuator_ids = np.asarray(
                [model.actuator_name2id(name) for name in actuator_names],
                dtype=np.int64,
            )
            ctrl_range = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            if np.any(np.abs(weight) < 1e-12):
                raise RuntimeError(f"Gripper actuator has a degenerate control range: {ctrl_range!r}.")

            joint_ids = np.asarray(model.actuator_trnid[actuator_ids, 0], dtype=np.int64)
            qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
            physical_targets = np.asarray(sim.data.qpos[qpos_addresses], dtype=np.float64)
            normalized = np.clip((physical_targets - bias) / weight, -1.0, 1.0)
            gripper.current_action = normalized.copy()
            sim.data.ctrl[actuator_ids] = physical_targets
            synchronized.append(normalized.astype(np.float32).tolist())

    sim.forward()
    return synchronized


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
    gripper_delta_alignment: str = "current_minus_previous",
    gripper_previous_width: float | None = None,
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
    delta_alignment = str(gripper_delta_alignment)
    if delta_alignment not in {"current_minus_previous", "next_minus_current"}:
        raise ValueError(f"Unsupported gripper_delta_alignment={delta_alignment!r}.")
    previous_width = float(chunk[0, -1] if gripper_previous_width is None else gripper_previous_width)
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
            if delta_alignment == "current_minus_previous":
                # Row i is the endpoint of the previous->current trajectory
                # interval, so its gripper event belongs to this same env step.
                delta_width = float(row[-1] - previous_width)
                previous_width = float(row[-1])
            else:
                # Legacy behavior triggers the i->i+1 transition while the arm
                # is still moving to row i, i.e. one waypoint early.
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
        "env_seed": int(cfg.get("env_seed", 0)),
        "evaluation_protocol": evaluation_protocol_for_config(cfg),
        "environment_domain": {
            "dataset_domain_env": bool(cfg.get("dataset_domain_env", False)),
            "dataset_domain_oracle_actions": bool(
                cfg.get("dataset_domain_oracle_actions", False)
            ),
            "benchmark_comparable": not bool(cfg.get("dataset_domain_env", False)),
            "demo_root": (
                str(cfg.get("dataset_domain_demo_root"))
                if bool(cfg.get("dataset_domain_env", False))
                else None
            ),
            "state_observation_offset": (
                int(cfg.get("dataset_domain_state_observation_offset", 1))
                if bool(cfg.get("dataset_domain_env", False))
                else None
            ),
        },
        "use_suite_max_steps": bool(cfg.get("use_suite_max_steps", False)),
        "suite_max_steps": {
            suite_name: (
                int(LIBERO_STANDARD_MAX_STEPS[suite_name])
                if bool(cfg.get("use_suite_max_steps", False))
                and suite_name in LIBERO_STANDARD_MAX_STEPS
                else int(cfg["control"]["max_steps"])
            )
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
            "mode": (
                "source_demo_exact_observation"
                if bool(cfg.get("dataset_domain_env", False))
                else "standard_libero_settled"
            ),
            "settling_applied": not bool(cfg.get("dataset_domain_env", False)),
            "settle_steps": (
                0
                if bool(cfg.get("dataset_domain_env", False))
                else int(cfg["settle_steps"])
            ),
            "settle_min_seconds": (
                0.0
                if bool(cfg.get("dataset_domain_env", False))
                else float(cfg["settle_min_seconds"])
            ),
            "settle_stable_seconds": (
                0.0
                if bool(cfg.get("dataset_domain_env", False))
                else float(cfg["settle_stable_seconds"])
            ),
            "settle_max_seconds": (
                0.0
                if bool(cfg.get("dataset_domain_env", False))
                else float(cfg["settle_max_seconds"])
            ),
            "settle_require_stable": (
                False
                if bool(cfg.get("dataset_domain_env", False))
                else bool(cfg["settle_require_stable"])
            ),
            "settle_keep_robot_fixed": (
                False
                if bool(cfg.get("dataset_domain_env", False))
                else bool(cfg["settle_keep_robot_fixed"])
            ),
            "initial_gripper_open": (
                False
                if bool(cfg.get("dataset_domain_env", False))
                else bool(cfg["initial_gripper_open"])
            ),
            "warmup_steps": (
                0
                if bool(cfg.get("dataset_domain_env", False))
                else int(cfg["control"].get("warmup_steps", 0))
            ),
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
            "gripper_delta_alignment": str(cfg["control"]["gripper_delta_alignment"]),
            "synchronize_gripper_controller_state": bool(
                cfg["control"]["synchronize_gripper_controller_state"]
            ),
            "gripper_target_tolerance": float(cfg["control"]["gripper_target_tolerance"]),
            "exec_action_steps": int(cfg["control"]["exec_action_steps"]),
            "adaptive_exec_max_steps": int(cfg["control"]["adaptive_exec_max_steps"]),
            "adaptive_exec_position_error_threshold": float(
                cfg["control"]["adaptive_exec_position_error_threshold"]
            ),
            "adaptive_exec_rotation_error_threshold": float(
                cfg["control"]["adaptive_exec_rotation_error_threshold"]
            ),
            "adaptive_exec_position_error_max": float(
                cfg["control"]["adaptive_exec_position_error_max"]
            ),
            "adaptive_exec_rotation_error_max": float(
                cfg["control"]["adaptive_exec_rotation_error_max"]
            ),
            "grasp_exec_steps": int(cfg["control"]["grasp_exec_steps"]),
            "grasp_width_min": float(cfg["control"]["grasp_width_min"]),
            "grasp_width_max": float(cfg["control"]["grasp_width_max"]),
            "grasp_lift_threshold": float(cfg["control"]["grasp_lift_threshold"]),
            "release_event_exec_enable": bool(
                cfg["control"]["release_event_exec_enable"]
            ),
            "release_event_exec_max_steps": int(
                cfg["control"]["release_event_exec_max_steps"]
            ),
            "release_event_min_width_change": float(
                cfg["control"]["release_event_min_width_change"]
            ),
            "waypoint_max_hold_steps": int(cfg["control"]["waypoint_max_hold_steps"]),
            "waypoint_position_tolerance": float(
                cfg["control"]["waypoint_position_tolerance"]
            ),
            "waypoint_rotation_tolerance": float(
                cfg["control"]["waypoint_rotation_tolerance"]
            ),
            "waypoint_gripper_tolerance": float(
                cfg["control"]["waypoint_gripper_tolerance"]
            ),
            "rollback_chunks": int(cfg["control"]["rollback_chunks"]),
            "rollback_max_steps": int(cfg["control"]["rollback_max_steps"]),
            "rollback_position_tolerance": float(
                cfg["control"]["rollback_position_tolerance"]
            ),
            "rollback_rotation_tolerance": float(
                cfg["control"]["rollback_rotation_tolerance"]
            ),
            "action_index": int(cfg["control"]["action_index"]),
            "control_freq": float(cfg["control"].get("control_freq", cfg.get("control_freq", 5))),
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
        "oracle_source_raw_actions",
        "oracle_source_action_indices",
        "oracle_scaled_deltas",
        "model_action_rows",
        "predicted_action_chunks",
        "oracle_chunk_source_indices",
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
        "initial_object_positions",
        "initial_object_quaternions",
        "object_positions",
        "object_quaternions",
        "goal_predicate_values",
        "scene_joint_values",
        "contact_pair_strings",
        "robot_scene_contact_pair_strings",
        "contact_counts",
        "robot_scene_contact_counts",
        "waypoint_hold_counts",
        "release_event_predicted_width_changes",
        "chunk_executed_waypoint_counts",
        "chunk_release_event_end_counts",
        "chunk_grasp_upward_displacements",
        "chunk_transported_grasp_flags",
        "rollback_actions",
        "rollback_target_controller_pose9",
        "rollback_achieved_controller_pose9",
        "rollback_position_errors",
        "rollback_rotation_errors",
        "rollback_chunk_counts",
    }
    record = {k: v for k, v in result.items() if k not in drop_keys}
    record["episode_index"] = int(episode_idx)
    if action_npz is not None:
        record["action_npz"] = action_npz
    return json_safe(record)


def save_episode_actions(result: dict[str, Any], episode_dir: Path) -> str | None:
    arrays = {
        "libero_actions": np.asarray(result.get("libero_actions", []), dtype=np.float32),
        "oracle_source_raw_actions": np.asarray(
            result.get("oracle_source_raw_actions", []), dtype=np.float32
        ),
        "oracle_source_action_indices": np.asarray(
            result.get("oracle_source_action_indices", []), dtype=np.int64
        ),
        "oracle_scaled_deltas": np.asarray(
            result.get("oracle_scaled_deltas", []), dtype=np.float32
        ),
        "model_action_rows": np.asarray(result.get("model_action_rows", []), dtype=np.float32),
        "predicted_action_chunks": np.asarray(
            result.get("predicted_action_chunks", []), dtype=np.float32
        ),
        "oracle_chunk_source_indices": np.asarray(
            result.get("oracle_chunk_source_indices", []), dtype=np.int64
        ),
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
        "release_event_predicted_width_changes": np.asarray(
            result.get("release_event_predicted_width_changes", []), dtype=np.float32
        ),
        "chunk_executed_waypoint_counts": np.asarray(
            result.get("chunk_executed_waypoint_counts", []), dtype=np.int16
        ),
        "chunk_release_event_end_counts": np.asarray(
            result.get("chunk_release_event_end_counts", []), dtype=np.int16
        ),
        "chunk_grasp_upward_displacements": np.asarray(
            result.get("chunk_grasp_upward_displacements", []), dtype=np.float32
        ),
        "chunk_transported_grasp_flags": np.asarray(
            result.get("chunk_transported_grasp_flags", []), dtype=np.bool_
        ),
        "rollback_actions": np.asarray(result.get("rollback_actions", []), dtype=np.float32),
        "rollback_target_controller_pose9": np.asarray(
            result.get("rollback_target_controller_pose9", []), dtype=np.float32
        ),
        "rollback_achieved_controller_pose9": np.asarray(
            result.get("rollback_achieved_controller_pose9", []), dtype=np.float32
        ),
        "rollback_position_errors": np.asarray(
            result.get("rollback_position_errors", []), dtype=np.float32
        ),
        "rollback_rotation_errors": np.asarray(
            result.get("rollback_rotation_errors", []), dtype=np.float32
        ),
        "rollback_chunk_counts": np.asarray(
            result.get("rollback_chunk_counts", []), dtype=np.int16
        ),
        "object_pose_names": np.asarray(result.get("object_pose_names", []), dtype=np.str_),
        "initial_object_positions": np.asarray(
            result.get("initial_object_positions", []), dtype=np.float32
        ),
        "initial_object_quaternions": np.asarray(
            result.get("initial_object_quaternions", []), dtype=np.float32
        ),
        "object_positions": np.asarray(result.get("object_positions", []), dtype=np.float32),
        "object_quaternions": np.asarray(result.get("object_quaternions", []), dtype=np.float32),
        "goal_predicate_names": np.asarray(result.get("goal_predicate_names", []), dtype=np.str_),
        "initial_goal_predicate_values": np.asarray(
            result.get("initial_goal_predicate_values", []), dtype=np.bool_
        ),
        "goal_predicate_values": np.asarray(result.get("goal_predicate_values", []), dtype=np.bool_),
        "scene_joint_names": np.asarray(result.get("scene_joint_names", []), dtype=np.str_),
        "scene_joint_types": np.asarray(result.get("scene_joint_types", []), dtype=np.str_),
        "scene_joint_ranges": np.asarray(result.get("scene_joint_ranges", []), dtype=np.float32),
        "initial_scene_joint_values": np.asarray(
            result.get("initial_scene_joint_values", []), dtype=np.float32
        ),
        "scene_joint_values": np.asarray(result.get("scene_joint_values", []), dtype=np.float32),
        "contact_pair_strings": np.asarray(result.get("contact_pair_strings", []), dtype=np.str_),
        "robot_scene_contact_pair_strings": np.asarray(
            result.get("robot_scene_contact_pair_strings", []), dtype=np.str_
        ),
        "contact_counts": np.asarray(result.get("contact_counts", []), dtype=np.int32),
        "robot_scene_contact_counts": np.asarray(
            result.get("robot_scene_contact_counts", []), dtype=np.int32
        ),
        "waypoint_hold_counts": np.asarray(
            result.get("waypoint_hold_counts", []), dtype=np.int16
        ),
    }
    if arrays["libero_actions"].size == 0:
        return None
    path = episode_dir / "actions.npz"
    np.savez_compressed(path, **arrays)
    return str(path)


def observable_object_pose_names(raw_obs: dict[str, Any]) -> list[str]:
    """Return scene-object names with directly observable world poses.

    LIBERO exposes keys such as ``akita_black_bowl_1_pos`` and
    ``akita_black_bowl_1_quat``. Robot proprioception and relative
    ``*_to_robot0_eef_*`` fields are intentionally excluded. These values are
    recorded only for post-hoc evaluation diagnostics and never enter policy
    inference.
    """
    names: list[str] = []
    for key in raw_obs:
        if not key.endswith("_pos"):
            continue
        name = key[: -len("_pos")]
        if name.startswith("robot") or "_to_robot" in name:
            continue
        quat_key = f"{name}_quat"
        if quat_key not in raw_obs:
            continue
        position = np.asarray(raw_obs[key]).reshape(-1)
        quaternion = np.asarray(raw_obs[quat_key]).reshape(-1)
        if position.size == 3 and quaternion.size == 4:
            names.append(name)
    return sorted(names)


def capture_observable_object_poses(
    raw_obs: dict[str, Any],
    object_pose_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not object_pose_names:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    positions = np.stack(
        [np.asarray(raw_obs[f"{name}_pos"], dtype=np.float32).reshape(3) for name in object_pose_names],
        axis=0,
    )
    quaternions = np.stack(
        [np.asarray(raw_obs[f"{name}_quat"], dtype=np.float32).reshape(4) for name in object_pose_names],
        axis=0,
    )
    return positions, quaternions


def find_libero_task_env(env: Any) -> Any | None:
    """Find the inner LIBERO task object without relying on wrapper depth."""
    for candidate in reversed(_iter_env_chain(env)):
        if hasattr(candidate, "parsed_problem") and callable(getattr(candidate, "_eval_predicate", None)):
            return candidate
    return None


def goal_predicate_spec(env: Any) -> tuple[list[str], list[Any]]:
    """Return stable labels and raw LIBERO goal predicates for diagnostics only."""
    task_env = find_libero_task_env(env)
    if task_env is None:
        return [], []
    states = list(getattr(task_env, "parsed_problem", {}).get("goal_state", []) or [])
    labels = [" ".join(map(str, state)) for state in states]
    return labels, states


def capture_goal_predicates(env: Any, states: list[Any]) -> np.ndarray:
    """Evaluate each benchmark predicate without feeding it back into control."""
    if not states:
        return np.empty((0,), dtype=np.bool_)
    task_env = find_libero_task_env(env)
    if task_env is None:
        return np.zeros((len(states),), dtype=np.bool_)
    values: list[bool] = []
    for state in states:
        try:
            values.append(bool(task_env._eval_predicate(state)))  # noqa: SLF001
        except Exception:
            values.append(False)
    return np.asarray(values, dtype=np.bool_)


def scene_scalar_joint_spec(env: Any) -> list[dict[str, Any]]:
    """Describe non-robot slide/hinge joints such as drawers, doors and knobs."""
    sim = get_sim(env)
    model = sim.model
    excluded_qpos: set[int] = set()
    try:
        arm_qpos, _ = robot_arm_indices(env)
        excluded_qpos.update(map(int, arm_qpos))
    except Exception:
        pass
    try:
        excluded_qpos.update(int(record["qpos_adr"]) for record in gripper_joint_records(env))
    except Exception:
        pass

    names = model_names(model, "joint")
    records: list[dict[str, Any]] = []
    for joint_id in range(int(getattr(model, "njnt", len(names)))):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (2, 3):
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        name = names[joint_id] if joint_id < len(names) else f"joint_{joint_id}"
        if qpos_address in excluded_qpos or re.search(r"robot|panda|gripper|finger", name, re.I):
            continue
        joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float32).reshape(-1)
        records.append(
            {
                "name": str(name),
                "type": _joint_type_name(model, joint_id),
                "qpos_adr": qpos_address,
                "range": joint_range[:2] if joint_range.size >= 2 else np.asarray([np.nan, np.nan]),
            }
        )
    return records


def capture_scene_scalar_joints(env: Any, records: list[dict[str, Any]]) -> np.ndarray:
    if not records:
        return np.empty((0,), dtype=np.float32)
    qpos = np.asarray(get_sim(env).data.qpos)
    return np.asarray([qpos[int(record["qpos_adr"])] for record in records], dtype=np.float32)


def robot_contact_geom_names(env: Any) -> set[str]:
    """Collect end-effector contact geoms, excluding fixed robot-base contacts."""
    names: set[str] = set()
    for env_layer in _iter_env_chain(env):
        for robot in getattr(env_layer, "robots", []) or []:
            raw_grippers = getattr(robot, "gripper", None)
            grippers = list(raw_grippers.values()) if isinstance(raw_grippers, dict) else [raw_grippers]
            for gripper in grippers:
                if gripper is not None:
                    names.update(str(name) for name in (getattr(gripper, "contact_geoms", None) or []))
    if not names:
        # Defensive fallback for wrappers that do not expose gripper metadata.
        for geom_name in model_names(get_sim(env).model, "geom"):
            if re.search(r"gripper|finger|hand", geom_name, re.I) and not re.search(r"vis", geom_name, re.I):
                names.add(str(geom_name))
    return names


def capture_contact_pairs(env: Any, robot_geoms: set[str]) -> tuple[str, str, int, int]:
    """Serialize current contact pairs and the robot-scene subset per step."""
    sim = get_sim(env)
    model = sim.model
    geom_names = model_names(model, "geom")
    all_pairs: set[str] = set()
    robot_scene_pairs: set[str] = set()
    for contact_index in range(int(getattr(sim.data, "ncon", 0))):
        contact = sim.data.contact[contact_index]
        geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
        geom1 = geom_names[geom1_id] if geom1_id < len(geom_names) else f"geom_{geom1_id}"
        geom2 = geom_names[geom2_id] if geom2_id < len(geom_names) else f"geom_{geom2_id}"
        pair = " <-> ".join(sorted((str(geom1), str(geom2))))
        all_pairs.add(pair)
        if (geom1 in robot_geoms) != (geom2 in robot_geoms):
            robot_scene_pairs.add(pair)
    all_sorted = sorted(all_pairs)
    robot_sorted = sorted(robot_scene_pairs)
    return "; ".join(all_sorted), "; ".join(robot_sorted), len(all_sorted), len(robot_sorted)

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
    task_init_states: np.ndarray | None,
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
        ensure_libero_config(cfg.get("libero_config_path"), configured_demo_root(cfg))
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


def _axis_points(
    origin: np.ndarray,
    rot: np.ndarray,
    *,
    scale: float = 0.04,
    samples: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    axes = [
        (rot[:, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        (rot[:, 1], np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
        (rot[:, 2], np.asarray([0.0, 0.31, 1.0], dtype=np.float32)),
    ]
    points = []
    colors = []
    alpha = np.linspace(0.0, scale, samples, dtype=np.float32)
    for direction, color in axes:
        points.append(origin[None, :] + alpha[:, None] * direction[None, :])
        colors.append(np.tile(color[None, :], (samples, 1)))
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def _to_visualization_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _trajectory_time_colors(num_steps: int) -> np.ndarray:
    """Blue/cyan at the chunk start, green in the middle, red at the end."""
    if num_steps <= 1:
        return np.asarray([[0.10, 0.75, 0.25]], dtype=np.float32)
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)[:, None]
    start = np.asarray([0.05, 0.55, 1.0], dtype=np.float32)
    middle = np.asarray([0.10, 0.85, 0.25], dtype=np.float32)
    end = np.asarray([1.0, 0.18, 0.05], dtype=np.float32)
    first = (1.0 - 2.0 * t) * start + (2.0 * t) * middle
    second = (2.0 - 2.0 * t) * middle + (2.0 * t - 1.0) * end
    return np.where(t <= 0.5, first, second).clip(0.0, 1.0)


def _sample_colored_trajectory(
    positions: np.ndarray,
    waypoint_colors: np.ndarray,
    *,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty
    if len(positions) == 1:
        return positions.astype(np.float32, copy=True), waypoint_colors.astype(np.float32, copy=True)

    sampled_points: list[np.ndarray] = []
    sampled_colors: list[np.ndarray] = []
    spacing = max(float(spacing), 1e-5)
    for index in range(len(positions) - 1):
        start = positions[index]
        end = positions[index + 1]
        distance = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(distance / spacing)) + 1)
        alpha = np.linspace(
            0.0,
            1.0,
            sample_count,
            endpoint=index == len(positions) - 2,
            dtype=np.float32,
        )[:, None]
        sampled_points.append((1.0 - alpha) * start + alpha * end)
        sampled_colors.append(
            (1.0 - alpha) * waypoint_colors[index] + alpha * waypoint_colors[index + 1]
        )
    return np.concatenate(sampled_points, axis=0), np.concatenate(sampled_colors, axis=0)


def build_umi_visualization_cloud(
    action_chunk: Any,
    point_cloud: Any,
    *,
    max_points: int = 50000,
    max_frames: int = 12,
    frame_scale: float = 0.035,
    trajectory_spacing: float = 0.002,
) -> np.ndarray:
    """Build one RGB point cloud containing the observation and pose9 trajectory.

    This deliberately creates no OpenGL / Open3D objects.  The result can be
    streamed to the isolated viewer process or written directly to a PLY file.
    """
    actions = _to_visualization_numpy(action_chunk).astype(np.float32, copy=False)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[-1] < 9:
        raise ValueError(f"Expected UMI action chunk shape (T, >=9), got {actions.shape}.")

    cloud = _to_visualization_numpy(point_cloud).astype(np.float32, copy=False)
    while cloud.ndim > 2 and cloud.shape[0] == 1:
        cloud = cloud[0]
    if cloud.ndim != 2 or cloud.shape[-1] < 3:
        raise ValueError(f"Expected point-cloud shape (N, >=3), got {cloud.shape}.")

    cloud_valid = np.isfinite(cloud[:, :3]).all(axis=1)
    cloud = cloud[cloud_valid]
    scene_xyz = cloud[:, :3]
    if cloud.shape[-1] >= 6:
        scene_rgb = cloud[:, 3:6].copy()
        if scene_rgb.size and scene_rgb.max(initial=0.0) > 1.0:
            scene_rgb /= 255.0
        scene_rgb = np.clip(scene_rgb, 0.0, 1.0)
    else:
        scene_rgb = np.full((len(scene_xyz), 3), 0.55, dtype=np.float32)

    positions = actions[:, :3]
    finite_poses = np.isfinite(actions[:, :9]).all(axis=1)
    positions = positions[finite_poses]
    pose_actions = actions[finite_poses]
    waypoint_colors = _trajectory_time_colors(len(positions))
    path_xyz, path_rgb = _sample_colored_trajectory(
        positions,
        waypoint_colors,
        spacing=trajectory_spacing,
    )

    overlay_xyz: list[np.ndarray] = [path_xyz]
    overlay_rgb: list[np.ndarray] = [path_rgb]
    origin_xyz, origin_rgb = _axis_points(
        np.zeros(3, dtype=np.float32),
        np.eye(3, dtype=np.float32),
        scale=frame_scale * 1.2,
    )
    overlay_xyz.append(origin_xyz)
    overlay_rgb.append(origin_rgb)

    if len(pose_actions):
        frame_stride = max(1, int(np.ceil(len(pose_actions) / max(1, int(max_frames)))))
        frame_indices = list(range(0, len(pose_actions), frame_stride))
        if len(pose_actions) - 1 not in frame_indices:
            frame_indices.append(len(pose_actions) - 1)
        poses = pose9_to_homo_np(pose_actions[:, :9])
        for frame_index in frame_indices:
            scale = frame_scale * (1.35 if frame_index in (0, len(pose_actions) - 1) else 1.0)
            frame_xyz, frame_rgb = _axis_points(
                poses[frame_index, :3, 3],
                poses[frame_index, :3, :3],
                scale=scale,
            )
            overlay_xyz.append(frame_xyz)
            overlay_rgb.append(frame_rgb)

    trajectory_xyz = np.concatenate(overlay_xyz, axis=0).astype(np.float32, copy=False)
    trajectory_rgb = np.concatenate(overlay_rgb, axis=0).astype(np.float32, copy=False)
    max_points = max(1, int(max_points))
    if len(trajectory_xyz) >= max_points:
        indices = np.linspace(0, len(trajectory_xyz) - 1, max_points, dtype=np.int64)
        trajectory_xyz = trajectory_xyz[indices]
        trajectory_rgb = trajectory_rgb[indices]
        scene_xyz = scene_xyz[:0]
        scene_rgb = scene_rgb[:0]
    else:
        scene_budget = max_points - len(trajectory_xyz)
        if len(scene_xyz) > scene_budget:
            indices = np.linspace(0, len(scene_xyz) - 1, scene_budget, dtype=np.int64)
            scene_xyz = scene_xyz[indices]
            scene_rgb = scene_rgb[indices]

    xyz = np.concatenate([scene_xyz, trajectory_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, trajectory_rgb], axis=0)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32, copy=False)


def _rgb_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return colors.reshape(-1, 3).astype(np.uint8)
    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def _write_colored_point_cloud_ply(path: Path, xyzrgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    all_points = xyzrgb[:, :3]
    all_colors = _rgb_uint8(xyzrgb[:, 3:6])

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
        for point, color in zip(all_points, all_colors, strict=True):
            f.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


_STANDALONE_UMI_VISUALIZER: Any | None = None


def _close_standalone_umi_visualizer() -> None:
    global _STANDALONE_UMI_VISUALIZER
    if _STANDALONE_UMI_VISUALIZER is not None:
        _STANDALONE_UMI_VISUALIZER.close()
        _STANDALONE_UMI_VISUALIZER = None


def vis_umi_data(
    action_chunk: Any,
    point_cloud: Any,
    *,
    visualizer: Any | None = None,
    save_path: str | Path | None = None,
    max_points: int = 50000,
) -> np.ndarray:
    """Visualize a model-space UMI action chunk without touching Open3D here.

    `visualizer` is expected to expose `update_colored`; the evaluator supplies
    an isolated subprocess-backed viewer.  If neither `visualizer` nor
    `save_path` is supplied, a module-level isolated viewer is created so the
    function can be called directly from a debugger.  `save_path` remains
    useful on a fully headless server and writes the exact same visualization.
    """
    global _STANDALONE_UMI_VISUALIZER
    xyzrgb = build_umi_visualization_cloud(
        action_chunk,
        point_cloud,
        max_points=max_points,
    )
    if save_path is not None:
        _write_colored_point_cloud_ply(Path(save_path), xyzrgb)
    if visualizer is None and save_path is None:
        from lerobot.policies.smolvla.inference_diagnostics import ForegroundScoreVisualizer

        if _STANDALONE_UMI_VISUALIZER is None:
            _STANDALONE_UMI_VISUALIZER = ForegroundScoreVisualizer(
                enabled=True,
                max_points=max_points,
                window_name=(
                    "UMI model observation + predicted trajectory: "
                    "blue/cyan=start, green=middle, red=end"
                ),
                print_every=1000000,
            )
            atexit.register(_close_standalone_umi_visualizer)
        visualizer = _STANDALONE_UMI_VISUALIZER
    if visualizer is not None:
        visualizer.update_colored(xyzrgb)
    return xyzrgb


def write_umi_debug_ply(path: Path, action_chunk: np.ndarray, point_cloud: np.ndarray) -> None:
    vis_umi_data(action_chunk, point_cloud, save_path=path)


def _get_umi_trajectory_visualizer(infer: Any, max_points: int) -> Any:
    visualizer = getattr(infer, "_umi_trajectory_visualizer", None)
    if visualizer is None:
        from lerobot.policies.smolvla.inference_diagnostics import ForegroundScoreVisualizer

        visualizer = ForegroundScoreVisualizer(
            enabled=True,
            max_points=max_points,
            window_name=(
                "UMI model observation + predicted trajectory: "
                "blue/cyan=start, green=middle, red=end"
            ),
            print_every=1000000,
        )
        infer._umi_trajectory_visualizer = visualizer
    else:
        visualizer.enable()
    return visualizer


def _close_umi_trajectory_visualizer(infer: Any) -> None:
    visualizer = getattr(infer, "_umi_trajectory_visualizer", None)
    if visualizer is not None:
        visualizer.close()
        delattr(infer, "_umi_trajectory_visualizer")


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
    reset_env: bool = True,
    oracle_raw_action_trajectory: np.ndarray | None = None,
    oracle_absolute_action_trajectory: np.ndarray | None = None,
    oracle_scaled_delta_trajectory: np.ndarray | None = None,
    oracle_raw_action_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    control = cfg["control"]
    dataset_domain_env = bool(cfg.get("dataset_domain_env", False))
    oracle_actions_enabled = oracle_raw_action_trajectory is not None
    if bool(cfg.get("dataset_domain_oracle_actions", False)) != oracle_actions_enabled:
        raise ValueError(
            "dataset_domain_oracle_actions and oracle_raw_action_trajectory must be enabled together."
        )
    if oracle_actions_enabled:
        assert oracle_raw_action_trajectory is not None
        raw_oracle = np.asarray(oracle_raw_action_trajectory)
        absolute_oracle = np.asarray(oracle_absolute_action_trajectory)
        scaled_oracle = np.asarray(oracle_scaled_delta_trajectory)
        raw_indices = np.asarray(oracle_raw_action_indices)
        if raw_oracle.ndim != 2 or raw_oracle.shape[1] != 7:
            raise ValueError(
                f"Expected oracle raw actions with shape (T, 7), got {raw_oracle.shape}."
            )
        if absolute_oracle.shape != raw_oracle.shape:
            raise ValueError(
                "oracle_absolute_action_trajectory must match raw actions, "
                f"got raw={raw_oracle.shape}, absolute={absolute_oracle.shape}."
            )
        if scaled_oracle.shape != (len(raw_oracle), 6):
            raise ValueError(
                "oracle_scaled_delta_trajectory must have shape (T, 6), "
                f"got raw={raw_oracle.shape}, scaled={scaled_oracle.shape}."
            )
        if raw_indices.shape != (len(raw_oracle),):
            raise ValueError(
                "oracle_raw_action_indices must contain one source index per raw action, "
                f"got actions={raw_oracle.shape}, indices={raw_indices.shape}."
            )
    configured_max_steps = int(control.get("max_steps", getattr(env, "horizon", 1000)))
    max_steps = (
        int(LIBERO_STANDARD_MAX_STEPS[suite_name])
        if bool(cfg.get("use_suite_max_steps", False)) and suite_name in LIBERO_STANDARD_MAX_STEPS
        else configured_max_steps
    )
    environment_horizon = None
    environment_ignore_done = False
    for env_layer in reversed(_iter_env_chain(env)):
        if environment_horizon is None and hasattr(env_layer, "horizon"):
            environment_horizon = int(env_layer.horizon)
        if hasattr(env_layer, "ignore_done"):
            environment_ignore_done = bool(env_layer.ignore_done)
    if (
        environment_horizon is not None
        and not environment_ignore_done
        and environment_horizon < max_steps
    ):
        raise RuntimeError(
            "LIBERO environment horizon is shorter than the requested evaluation limit: "
            f"environment_horizon={environment_horizon}, max_steps={max_steps}."
        )
    action_index = max(0, int(control.get("action_index", 0)))
    exec_action_steps = int(control.get("exec_action_steps", 12))
    adaptive_exec_max_steps = max(
        exec_action_steps,
        int(control.get("adaptive_exec_max_steps", exec_action_steps)),
    )
    adaptive_exec_position_error_threshold = max(
        0.0, float(control.get("adaptive_exec_position_error_threshold", 0.012))
    )
    adaptive_exec_rotation_error_threshold = max(
        0.0, float(control.get("adaptive_exec_rotation_error_threshold", 0.10))
    )
    adaptive_exec_position_error_max = max(
        adaptive_exec_position_error_threshold,
        float(control.get("adaptive_exec_position_error_max", 0.03)),
    )
    adaptive_exec_rotation_error_max = max(
        adaptive_exec_rotation_error_threshold,
        float(control.get("adaptive_exec_rotation_error_max", 0.15)),
    )
    grasp_exec_steps = max(
        exec_action_steps,
        int(control.get("grasp_exec_steps", exec_action_steps)),
    )
    grasp_width_min = max(0.0, float(control.get("grasp_width_min", 0.003)))
    grasp_width_max = max(
        grasp_width_min,
        float(control.get("grasp_width_max", 0.070)),
    )
    grasp_lift_threshold = max(
        0.0,
        float(control.get("grasp_lift_threshold", 0.015)),
    )
    release_event_exec_enable = bool(control.get("release_event_exec_enable", False))
    release_event_exec_max_steps = max(
        grasp_exec_steps,
        int(control.get("release_event_exec_max_steps", 32)),
    )
    release_event_min_width_change = max(
        0.0,
        float(control.get("release_event_min_width_change", 0.02)),
    )
    waypoint_max_hold_steps = max(1, int(control.get("waypoint_max_hold_steps", 1)))
    waypoint_position_tolerance = max(
        0.0, float(control.get("waypoint_position_tolerance", 0.002))
    )
    waypoint_rotation_tolerance = max(
        0.0, float(control.get("waypoint_rotation_tolerance", 0.03))
    )
    waypoint_gripper_tolerance = max(
        0.0, float(control.get("waypoint_gripper_tolerance", 0.004))
    )
    rollback_chunks = max(1, int(control.get("rollback_chunks", 2)))
    rollback_max_steps = max(1, int(control.get("rollback_max_steps", 50)))
    rollback_position_tolerance = max(
        0.0,
        float(control.get("rollback_position_tolerance", waypoint_position_tolerance)),
    )
    rollback_rotation_tolerance = max(
        0.0,
        float(control.get("rollback_rotation_tolerance", waypoint_rotation_tolerance)),
    )
    configured_warmup_steps = max(0, int(control.get("warmup_steps", 0)))
    warmup_steps = 0 if dataset_domain_env else configured_warmup_steps
    gripper_threshold = float(control.get("gripper_threshold", 0.5))
    gripper_max_width = float(cfg.get("gripper_qpos_max_width", 0.08))
    gripper_control_mode = str(control.get("gripper_control_mode", "delta_width"))
    gripper_delta_threshold = float(control.get("gripper_delta_threshold", 0.002))
    gripper_delta_alignment = str(control.get("gripper_delta_alignment", "current_minus_previous"))
    gripper_target_tolerance = float(control.get("gripper_target_tolerance", 0.004))
    policy_noise_seed_base = int(cfg.get("policy_noise_seed", 0))


    if reset_env:
        raw_obs = env.reset()
    else:
        raw_obs = get_raw_obs(env, force_update=True)
    raw_obs = env.set_init_state(init_state)

    if dataset_domain_env:
        # This state is the exact simulator snapshot rendered into the first
        # converted training observation. Advancing physics or forcing the
        # fingers open here would immediately move evaluation out of that domain.
        raw_obs = get_raw_obs(env, force_update=True)
    else:
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
    synchronized_gripper_controller_actions: list[list[float]] = []
    effective_synchronize_gripper_controller_state = bool(
        control.get("synchronize_gripper_controller_state", True)
    ) and not oracle_actions_enabled
    if effective_synchronize_gripper_controller_state:
        synchronized_gripper_controller_actions = synchronize_gripper_controller_state(env)
        raw_obs = get_raw_obs(env, force_update=True)


    # Absolute pose execution only.
    for robot in env.robots:
        robot.controller.use_delta = False
    refresh_arm_controller_state(env)


    for _ in range(warmup_steps):
        raw_obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    if not bool(getattr(infer, "shared_parallel_inference", False)):
        infer.policy.reset()
        infer.policy_reset()

    policy_rgb_camera_map: dict[str, str] = {}
    if infer.policy.config.vla_adapter_enable:
        policy_rgb_camera_map = resolve_policy_rgb_cameras(infer, raw_obs, cfg)
        # print(f"[info] adapter RGB camera mapping: {policy_rgb_camera_map}", flush=True)

    object_pose_names = observable_object_pose_names(raw_obs)
    initial_object_positions, initial_object_quaternions = capture_observable_object_poses(
        raw_obs,
        object_pose_names,
    )
    object_positions: list[np.ndarray] = []
    object_quaternions: list[np.ndarray] = []
    goal_predicate_names, raw_goal_predicates = goal_predicate_spec(env)
    initial_goal_predicate_values = capture_goal_predicates(env, raw_goal_predicates)
    goal_predicate_values: list[np.ndarray] = []
    scene_joint_records = scene_scalar_joint_spec(env)
    scene_joint_names = [str(record["name"]) for record in scene_joint_records]
    scene_joint_types = [str(record["type"]) for record in scene_joint_records]
    scene_joint_ranges = np.asarray(
        [record["range"] for record in scene_joint_records], dtype=np.float32
    ).reshape(-1, 2)
    initial_scene_joint_values = capture_scene_scalar_joints(env, scene_joint_records)
    scene_joint_values: list[np.ndarray] = []
    robot_contact_geoms = robot_contact_geom_names(env)
    contact_pair_strings: list[str] = []
    robot_scene_contact_pair_strings: list[str] = []
    contact_counts: list[int] = []
    robot_scene_contact_counts: list[int] = []

    pc_camera_names = pointcloud_camera_names_from_config(cfg)
    save_video = bool(cfg.get("save_video", True))
    video_frames: dict[str, list[np.ndarray]] = {}
    if save_video:
        append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))

    rewards: list[float] = []
    libero_actions: list[np.ndarray] = []
    oracle_source_raw_actions: list[np.ndarray] = []
    oracle_source_action_indices: list[int] = []
    oracle_scaled_deltas: list[np.ndarray] = []
    model_action_rows: list[np.ndarray] = []
    # Keep every complete policy prediction for post-hoc diagnosis.  The
    # executed rows alone cannot reveal whether replanning truncated a
    # coherent sub-action in the unused suffix of an action chunk.
    predicted_action_chunks: list[np.ndarray] = []
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
    model_waypoints_executed = 0
    adaptive_waypoints_executed = 0
    grasp_extended_waypoints_executed = 0
    grasp_extended_model_calls = 0
    grasp_anchor_position: np.ndarray | None = None
    grasp_max_upward_displacement = 0.0
    release_event_extended_waypoints_executed = 0
    release_event_extended_model_calls = 0
    release_event_predicted_width_changes: list[float] = []
    chunk_executed_waypoint_counts: list[int] = []
    chunk_release_event_end_counts: list[int] = []
    chunk_grasp_upward_displacements: list[float] = []
    chunk_transported_grasp_flags: list[bool] = []
    waypoint_hold_counts: list[int] = []
    executed_chunk_start_controller_world_history: list[np.ndarray] = []
    rollback_actions: list[np.ndarray] = []
    rollback_target_controller_pose9: list[np.ndarray] = []
    rollback_achieved_controller_pose9: list[np.ndarray] = []
    rollback_position_errors: list[float] = []
    rollback_rotation_errors: list[float] = []
    rollback_chunk_counts: list[int] = []
    rollback_request_count = 0
    rollback_completed_count = 0
    rollback_step_count = 0

    def update_grasp_transport_state(pose9_gripper: np.ndarray) -> bool:
        """Track whether an intermediate-width closure has subsequently lifted."""
        nonlocal grasp_anchor_position, grasp_max_upward_displacement
        pose = np.asarray(pose9_gripper, dtype=np.float32).reshape(-1)
        width = float(pose[-1])
        if not (grasp_width_min < width < grasp_width_max):
            grasp_anchor_position = None
            grasp_max_upward_displacement = 0.0
            return False
        if grasp_anchor_position is None:
            grasp_anchor_position = pose[:3].copy()
            grasp_max_upward_displacement = 0.0
        grasp_max_upward_displacement = max(
            grasp_max_upward_displacement,
            float(pose[2] - grasp_anchor_position[2]),
        )
        return grasp_max_upward_displacement >= grasp_lift_threshold

    success_ever = False
    done = False
    model_call_count = 0
    policy_forward_call_count = 0
    oracle_cursor = -1
    oracle_chunk_source_indices: list[np.ndarray] = []
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
    if bool(cfg.get("keyboard_control_enabled", True)) or str(
        cfg.get("render_mode", "offscreen")
    ).lower() == "viewer3d":
        print(
            "[eval] controls: 'n' = fail/next episode, "
            "'r' = rollback recent policy chunks, "
            "'v' = visualize next predicted UMI trajectory",
            flush=True,
        )

    def execute_manual_rollback() -> bool:
        """Move the robot back to an earlier completed chunk start, then replan."""
        nonlocal raw_obs, steps, done, manual_failure
        nonlocal previous_issued_target_model_world
        nonlocal grasp_anchor_position, grasp_max_upward_displacement
        nonlocal rollback_request_count, rollback_completed_count, rollback_step_count

        rollback_request_count += 1
        available_chunks = len(executed_chunk_start_controller_world_history)
        if available_chunks == 0:
            rollback_chunk_counts.append(0)
            print(
                "[eval] rollback requested, but no executed chunk start pose is available; "
                "replan from the current pose",
                flush=True,
            )
            previous_issued_target_model_world = None
            grasp_anchor_position = None
            grasp_max_upward_displacement = 0.0
            return False

        chunk_count = min(rollback_chunks, available_chunks)
        target_controller_world = np.asarray(
            executed_chunk_start_controller_world_history[-chunk_count],
            dtype=np.float32,
        ).copy()
        target_pose9 = matrix_to_pose9(target_controller_world)
        rollback_action = np.concatenate(
            [
                world_pose_to_libero_absolute_action(target_controller_world),
                np.zeros(1, dtype=np.float32),
            ]
        ).astype(np.float32)
        rollback_chunk_counts.append(int(chunk_count))
        print(
            f"[eval] rollback: returning to the start pose before the previous "
            f"{chunk_count} completed policy chunk(s)",
            flush=True,
        )

        reached = False
        for _rollback_index in range(rollback_max_steps):
            if steps >= max_steps or done:
                break
            if keyboard.poll():
                manual_failure = True
                break
            try:
                raw_obs, reward, done, _ = env.step(rollback_action)
            except ValueError as exc:
                if "terminated episode" in str(exc):
                    done = True
                    break
                raise

            steps += 1
            rollback_step_count += 1
            reward = float(reward)
            rewards.append(reward)
            rollback_actions.append(rollback_action.copy())
            rollback_target_controller_pose9.append(target_pose9.copy())

            step_object_positions, step_object_quaternions = capture_observable_object_poses(
                raw_obs,
                object_pose_names,
            )
            object_positions.append(step_object_positions)
            object_quaternions.append(step_object_quaternions)
            goal_predicate_values.append(capture_goal_predicates(env, raw_goal_predicates))
            scene_joint_values.append(capture_scene_scalar_joints(env, scene_joint_records))
            all_contacts, robot_contacts, all_contact_count, robot_contact_count = capture_contact_pairs(
                env,
                robot_contact_geoms,
            )
            contact_pair_strings.append(all_contacts)
            robot_scene_contact_pair_strings.append(robot_contacts)
            contact_counts.append(int(all_contact_count))
            robot_scene_contact_counts.append(int(robot_contact_count))
            render_viewer3d(env, cfg, steps)
            if save_video:
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))

            achieved_controller_world = current_controller_eef_world(env)
            achieved_pose9 = matrix_to_pose9(achieved_controller_world)
            rollback_achieved_controller_pose9.append(achieved_pose9)
            position_error = float(
                np.linalg.norm(
                    achieved_controller_world[:3, 3] - target_controller_world[:3, 3]
                )
            )
            rotation_error = rotation_error_radians(
                achieved_controller_world[:3, :3],
                target_controller_world[:3, :3],
            )
            rollback_position_errors.append(position_error)
            rollback_rotation_errors.append(rotation_error)
            if (
                position_error <= rollback_position_tolerance
                and rotation_error <= rollback_rotation_tolerance
            ):
                reached = True
                break

        previous_issued_target_model_world = None
        grasp_anchor_position = None
        grasp_max_upward_displacement = 0.0
        if reached:
            del executed_chunk_start_controller_world_history[-chunk_count:]
            rollback_completed_count += 1
            print(
                f"[eval] rollback complete; "
                f"{len(executed_chunk_start_controller_world_history)} earlier chunk start pose(s) remain",
                flush=True,
            )
            return True

        print(
            "[eval] rollback stopped before reaching tolerance; replan from the current pose",
            flush=True,
        )
        return False

    def execute_raw_action_oracle() -> None:
        nonlocal raw_obs, steps, done, success_ever, manual_failure, oracle_cursor
        # A source demonstration action is already one policy-frequency control
        # interval. Execute it exactly once. Holding or skipping rows would
        # change the source controller integration and contact dynamics.
        assert oracle_raw_action_trajectory is not None
        assert oracle_absolute_action_trajectory is not None
        assert oracle_scaled_delta_trajectory is not None
        assert oracle_raw_action_indices is not None
        raw_oracle = np.asarray(oracle_raw_action_trajectory, dtype=np.float64)
        absolute_oracle = np.asarray(
            oracle_absolute_action_trajectory, dtype=np.float64
        )
        scaled_oracle = np.asarray(
            oracle_scaled_delta_trajectory, dtype=np.float64
        )
        raw_indices = np.asarray(oracle_raw_action_indices, dtype=np.int64)
        oracle_chunk_source_indices.append(raw_indices.copy())
        chunk_start_model_worlds.append(
            pose9_to_homo_np(eef_pose9_gripper_from_obs(raw_obs)[:9]).astype(np.float32)
        )
        oracle_executed_count = 0
        for (
            source_action,
            absolute_action,
            scaled_delta,
            source_action_index,
        ) in zip(
            raw_oracle,
            absolute_oracle,
            scaled_oracle,
            raw_indices,
            strict=True,
        ):
            if steps >= max_steps or done or success_ever:
                break
            if keyboard.poll():
                manual_failure = True
                break

            current_model_world = pose9_to_homo_np(
                eef_pose9_gripper_from_obs(raw_obs)[:9]
            )
            current_controller_world = current_controller_eef_world(env)
            controller_to_model = (
                fast_inverse_homogeneous(current_controller_world) @ current_model_world
            )
            target_controller_world = np.eye(4, dtype=np.float64)
            target_controller_world[:3, :3] = R.from_rotvec(
                absolute_action[3:6]
            ).as_matrix()
            target_controller_world[:3, 3] = absolute_action[:3]
            target_model_world = target_controller_world @ controller_to_model

            try:
                raw_obs, reward, done, _ = env.step(absolute_action)
            except ValueError as exc:
                if "terminated episode" in str(exc):
                    done = True
                    break
                raise

            steps += 1
            oracle_executed_count += 1
            oracle_cursor = int(source_action_index)
            reward = float(reward)
            rewards.append(reward)
            libero_actions.append(np.asarray(absolute_action, dtype=np.float32))
            oracle_source_raw_actions.append(np.asarray(source_action, dtype=np.float32))
            oracle_source_action_indices.append(int(source_action_index))
            oracle_scaled_deltas.append(np.asarray(scaled_delta, dtype=np.float32))
            target_controller_pose9.append(matrix_to_pose9(target_controller_world))
            target_model_worlds.append(np.asarray(target_model_world, dtype=np.float32))
            gripper_commands.append(float(source_action[-1]))
            gripper_actual_widths.append(float(gripper_scalar(raw_obs)))
            waypoint_hold_counts.append(1)

            achieved_pose = eef_pose9_gripper_from_obs(raw_obs)
            achieved_model_world = pose9_to_homo_np(achieved_pose[:9])
            achieved_model_worlds.append(
                np.asarray(achieved_model_world, dtype=np.float32)
            )
            tracking_position_errors.append(
                float(
                    np.linalg.norm(
                        achieved_model_world[:3, 3] - target_model_world[:3, 3]
                    )
                )
            )
            tracking_rotation_errors.append(
                rotation_error_radians(
                    achieved_model_world[:3, :3],
                    target_model_world[:3, :3],
                )
            )

            step_object_positions, step_object_quaternions = capture_observable_object_poses(
                raw_obs,
                object_pose_names,
            )
            object_positions.append(step_object_positions)
            object_quaternions.append(step_object_quaternions)
            goal_predicate_values.append(
                capture_goal_predicates(env, raw_goal_predicates)
            )
            scene_joint_values.append(
                capture_scene_scalar_joints(env, scene_joint_records)
            )
            (
                all_contacts,
                robot_contacts,
                all_contact_count,
                robot_contact_count,
            ) = capture_contact_pairs(env, robot_contact_geoms)
            contact_pair_strings.append(all_contacts)
            robot_scene_contact_pair_strings.append(robot_contacts)
            contact_counts.append(int(all_contact_count))
            robot_scene_contact_counts.append(int(robot_contact_count))
            render_viewer3d(env, cfg, steps)
            if save_video:
                append_video_frames(
                    video_frames,
                    raw_obs,
                    list(cfg["camera_names"]),
                )

            try:
                success_ever = success_ever or bool(env.check_success())
            except Exception:
                success_ever = success_ever or bool(reward > 0.0)

        chunk_executed_waypoint_counts.append(int(oracle_executed_count))

    try:
        if oracle_actions_enabled:
            execute_raw_action_oracle()
        while (
            not oracle_actions_enabled
            and steps < max_steps
            and not done
            and not success_ever
        ):
            if keyboard.poll():
                manual_failure = True
                break
            if keyboard.pop_rollback_request():
                execute_manual_rollback()
                if manual_failure or done:
                    break
                continue

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

            transported_grasp = update_grasp_transport_state(eef_pose)

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
            policy_forward_call_count += 1
            if hasattr(chunk_batch, "detach"):
                chunk = chunk_batch[0].detach().cpu().numpy()
            else:
                chunk = np.asarray(chunk_batch)[0]
            predicted_action_chunks.append(np.asarray(chunk, dtype=np.float32))
            model_call_count += 1

            one_shot_trajectory_vis = keyboard.pop_trajectory_visualization_request()
            continuous_trajectory_vis = bool(
                cfg.get("visualize_action_trajectory", False)
            ) and (
                model_call_count
                % int(cfg.get("trajectory_vis_every_n_model_calls", 1))
                == 0
            )
            if one_shot_trajectory_vis or continuous_trajectory_vis:
                try:
                    trajectory_visualizer = _get_umi_trajectory_visualizer(
                        infer,
                        int(cfg.get("trajectory_vis_max_points", 50000)),
                    )
                    vis_umi_data(
                        chunk,
                        model_observation["point_cloud"],
                        visualizer=trajectory_visualizer,
                        max_points=int(cfg.get("trajectory_vis_max_points", 50000)),
                    )
                except Exception as exc:
                    print(
                        f"[warn] UMI trajectory visualization update failed: {exc!r}",
                        flush=True,
                    )

            if keyboard.poll():
                manual_failure = True
                break
            if keyboard.pop_rollback_request():
                execute_manual_rollback()
                if manual_failure or done:
                    break
                continue

            start_idx = min(action_index, max(0, len(chunk) - 1))
            measured_gripper_width = float(eef_pose[-1])
            inferred_stable_grasp = (
                grasp_exec_steps > exec_action_steps
                and grasp_width_min < measured_gripper_width < grasp_width_max
                and transported_grasp
            )
            chunk_grasp_upward_displacements.append(
                float(grasp_max_upward_displacement)
            )
            chunk_transported_grasp_flags.append(bool(inferred_stable_grasp))
            current_exec_action_steps = (
                grasp_exec_steps if inferred_stable_grasp else exec_action_steps
            )
            if inferred_stable_grasp:
                grasp_extended_model_calls += 1
            normal_base_end_idx = (
                len(chunk)
                if exec_action_steps <= 0
                else min(len(chunk), start_idx + current_exec_action_steps)
            )
            release_event_end_count = max(0, normal_base_end_idx - start_idx)
            release_event_width_change = 0.0
            if (
                exec_action_steps > 0
                and release_event_exec_enable
                and inferred_stable_grasp
            ):
                release_event_end_count, release_event_width_change = (
                    predicted_release_event_end(
                        chunk[start_idx:, -1],
                        base_count=current_exec_action_steps,
                        max_count=release_event_exec_max_steps,
                        min_width_change=release_event_min_width_change,
                    )
                )
            committed_end_idx = min(len(chunk), start_idx + release_event_end_count)
            release_event_planned = committed_end_idx > normal_base_end_idx
            if release_event_planned:
                release_event_extended_model_calls += 1
            release_event_predicted_width_changes.append(
                float(release_event_width_change)
            )
            chunk_release_event_end_counts.append(
                int(committed_end_idx - start_idx)
            )
            end_idx = (
                len(chunk)
                if exec_action_steps <= 0
                else min(
                    len(chunk),
                    max(
                        start_idx + adaptive_exec_max_steps,
                        normal_base_end_idx,
                        committed_end_idx,
                    ),
                )
            )
            selected_chunk = np.asarray(chunk[start_idx:end_idx], dtype=np.float32)
            # previous_predicted_width = float(
            #     chunk[start_idx - 1, -1] if start_idx > 0 else chunk[start_idx, -1]
            # )
            gripper_previous_width = float(measured_gripper_width)
            actions, model_worlds, controller_pose9 = action_chunk_to_absolute_libero_actions(
                env=env,
                current_eef_pose9_gripper=eef_pose,
                action_chunk=selected_chunk,
                gripper_threshold=gripper_threshold,
                gripper_max_width=gripper_max_width,
                gripper_control_mode=gripper_control_mode,
                gripper_delta_threshold=gripper_delta_threshold,
                gripper_delta_alignment=gripper_delta_alignment,
                gripper_previous_width=previous_gripper_width,
            )

            latest_position_error = 0.0
            latest_rotation_error = 0.0
            normal_base_selected_count = max(0, normal_base_end_idx - start_idx)
            committed_selected_count = max(0, committed_end_idx - start_idx)
            executed_waypoints_this_chunk = 0
            chunk_execution_start_steps = steps
            chunk_start_controller_world = current_controller_eef_world(env).copy()
            rollback_requested = False
            for selected_row_index, (row, action, model_world, controller_pose) in enumerate(zip(
                selected_chunk,
                actions,
                model_worlds,
                controller_pose9,
                strict=True,
            )):
                if selected_row_index >= normal_base_selected_count:
                    stale_chunk = (
                        latest_position_error > adaptive_exec_position_error_max
                        or latest_rotation_error > adaptive_exec_rotation_error_max
                    )
                    if stale_chunk:
                        break
                    if selected_row_index < committed_selected_count:
                        release_event_extended_waypoints_executed += 1
                    else:
                        if (
                            latest_position_error <= adaptive_exec_position_error_threshold
                            and latest_rotation_error <= adaptive_exec_rotation_error_threshold
                        ):
                            break
                        adaptive_waypoints_executed += 1
                if steps >= max_steps or done or success_ever:
                    break
                if keyboard.poll():
                    manual_failure = True
                    break
                if keyboard.pop_rollback_request():
                    rollback_requested = True
                    break
                if inferred_stable_grasp and selected_row_index >= exec_action_steps:
                    grasp_extended_waypoints_executed += 1
                model_waypoints_executed += 1
                executed_waypoints_this_chunk += 1
                hold_count = 0
                waypoint_gripper_direction = float(action[-1])
                for _hold_index in range(waypoint_max_hold_steps):
                    if steps >= max_steps or done or success_ever:
                        break
                    if keyboard.poll():
                        manual_failure = True
                        break
                    if keyboard.pop_rollback_request():
                        rollback_requested = True
                        break
                    step_action = np.asarray(action, dtype=np.float32).copy()
                    if gripper_control_mode == "target_width":
                        step_action[-1] = gripper_target_width_command(
                            float(row[-1]),
                            gripper_scalar(raw_obs),
                            tolerance=gripper_target_tolerance,
                            max_physical_width=gripper_max_width,
                        )
                    elif _hold_index > 0:
                        # LIBERO's Panda gripper integrates a directional
                        # command into an internal target. Repeating the same
                        # event while holding an arm waypoint changes one
                        # predicted open/close event into several events.
                        # A zero command keeps tracking the target established
                        # on the first controller step.
                        step_action[-1] = 0.0
                    gripper_reach_direction = (
                        float(step_action[-1])
                        if gripper_control_mode == "target_width"
                        else waypoint_gripper_direction
                    )
                    try:
                        raw_obs, reward, done, _ = env.step(step_action)
                    except ValueError as exc:
                        if "terminated episode" in str(exc):
                            done = True
                            break
                        raise

                    steps += 1
                    hold_count += 1
                    step_object_positions, step_object_quaternions = capture_observable_object_poses(
                        raw_obs,
                        object_pose_names,
                    )
                    object_positions.append(step_object_positions)
                    object_quaternions.append(step_object_quaternions)
                    goal_predicate_values.append(capture_goal_predicates(env, raw_goal_predicates))
                    scene_joint_values.append(capture_scene_scalar_joints(env, scene_joint_records))
                    all_contacts, robot_contacts, all_contact_count, robot_contact_count = capture_contact_pairs(
                        env,
                        robot_contact_geoms,
                    )
                    contact_pair_strings.append(all_contacts)
                    robot_scene_contact_pair_strings.append(robot_contacts)
                    contact_counts.append(int(all_contact_count))
                    robot_scene_contact_counts.append(int(robot_contact_count))
                    render_viewer3d(env, cfg, steps)
                    reward = float(reward)
                    rewards.append(reward)
                    libero_actions.append(np.asarray(step_action, dtype=np.float32))
                    model_action_rows.append(np.asarray(row, dtype=np.float32))
                    target_model_worlds.append(np.asarray(model_world, dtype=np.float32))
                    target_controller_pose9.append(np.asarray(controller_pose, dtype=np.float32))
                    gripper_commands.append(float(step_action[-1]))
                    gripper_raw_widths.append(float(row[-1]))
                    gripper_width_pcts.append(
                        gripper_width_percent_from_scalar(
                            float(row[-1]), max_physical_width=gripper_max_width
                        )
                    )
                    achieved_pose = eef_pose9_gripper_from_obs(raw_obs)
                    update_grasp_transport_state(achieved_pose)
                    gripper_actual_widths.append(float(achieved_pose[-1]))
                    gripper_width_errors.append(float(achieved_pose[-1] - row[-1]))
                    achieved_model_world = pose9_to_homo_np(
                        np.asarray(achieved_pose[:9], dtype=np.float32)
                    )
                    achieved_model_worlds.append(np.asarray(achieved_model_world, dtype=np.float32))
                    position_error = float(
                        np.linalg.norm(achieved_model_world[:3, 3] - model_world[:3, 3])
                    )
                    rotation_error = rotation_error_radians(
                        achieved_model_world[:3, :3], model_world[:3, :3]
                    )
                    latest_position_error = position_error
                    latest_rotation_error = rotation_error
                    tracking_position_errors.append(position_error)
                    tracking_rotation_errors.append(rotation_error)
                    previous_issued_target_model_world = np.asarray(model_world, dtype=np.float32)

                    if save_video:
                        append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))

                    if keyboard.poll():
                        manual_failure = True
                        break
                    if keyboard.pop_rollback_request():
                        rollback_requested = True
                        break

                    try:
                        success_ever = success_ever or bool(env.check_success())
                    except Exception:
                        success_ever = success_ever or bool(reward > 0.0)
                    if success_ever or done:
                        break

                    gripper_reached = True
                    if gripper_control_mode == "target_width":
                        if gripper_reach_direction > 0.0:
                            gripper_reached = float(achieved_pose[-1]) <= (
                                float(row[-1]) + waypoint_gripper_tolerance
                            )
                        elif gripper_reach_direction < 0.0:
                            gripper_reached = float(achieved_pose[-1]) >= (
                                float(row[-1]) - waypoint_gripper_tolerance
                            )
                    if (
                        position_error <= waypoint_position_tolerance
                        and rotation_error <= waypoint_rotation_tolerance
                        and gripper_reached
                    ):
                        break
                waypoint_hold_counts.append(int(hold_count))
                if manual_failure or rollback_requested or done or success_ever:
                    break
            chunk_executed_waypoint_counts.append(int(executed_waypoints_this_chunk))
            if steps > chunk_execution_start_steps:
                executed_chunk_start_controller_world_history.append(
                    np.asarray(chunk_start_controller_world, dtype=np.float32).copy()
                )

            if rollback_requested:
                execute_manual_rollback()
                if manual_failure or done:
                    break
                continue
            if manual_failure:
                break
        manual_failure = manual_failure or keyboard.poll()
    finally:
        keyboard.close()

    if manual_failure:
        success_ever = False

    final_eef_pose = eef_pose9_gripper_from_obs(raw_obs)
    return {
        "evaluation_protocol": evaluation_protocol_for_config(cfg),
        "initialization_mode": (
            "source_demo_exact_observation"
            if dataset_domain_env
            else "standard_libero_settled"
        ),
        "settling_applied": not dataset_domain_env,
        "forced_initial_gripper_open_applied": (
            bool(cfg["initial_gripper_open"]) and not dataset_domain_env
        ),
        "warmup_steps_applied": int(warmup_steps),
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
            else "source_actions_exhausted"
            if oracle_actions_enabled
            else "max_steps"
        ),
        "steps": int(steps),
        "max_steps": int(max_steps),
        "environment_horizon": (
            int(environment_horizon) if environment_horizon is not None else None
        ),
        "environment_ignore_done": bool(environment_ignore_done),
        "action_rows_executed": int(len(libero_actions)),
        "model_waypoints_executed": int(model_waypoints_executed),
        "adaptive_waypoints_executed": int(adaptive_waypoints_executed),
        "grasp_extended_waypoints_executed": int(grasp_extended_waypoints_executed),
        "grasp_extended_model_calls": int(grasp_extended_model_calls),
        "release_event_extended_waypoints_executed": int(
            release_event_extended_waypoints_executed
        ),
        "release_event_extended_model_calls": int(release_event_extended_model_calls),
        "model_call_count": int(model_call_count),
        "policy_forward_call_count": int(policy_forward_call_count),
        "action_source": (
            "source_demo_raw_delta_to_source_anchored_absolute_osc"
            if oracle_actions_enabled
            else "policy_flow_matching_sample"
        ),
        "oracle_final_source_cursor": (
            int(oracle_cursor) if oracle_actions_enabled else None
        ),
        "oracle_trajectory_length": (
            int(len(oracle_raw_action_trajectory))
            if oracle_raw_action_trajectory is not None
            else None
        ),
        "sum_reward": float(np.sum(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "wall_s": float(time.perf_counter() - start_s),
        "gripper_threshold": float(gripper_threshold),
        "gripper_control_mode": gripper_control_mode,
        "gripper_delta_threshold": float(gripper_delta_threshold),
        "gripper_delta_alignment": gripper_delta_alignment,
        "synchronize_gripper_controller_state": bool(
            effective_synchronize_gripper_controller_state
        ),
        "initial_gripper_controller_actions": synchronized_gripper_controller_actions,
        "gripper_target_tolerance": float(gripper_target_tolerance),
        "waypoint_max_hold_steps": int(waypoint_max_hold_steps),
        "adaptive_exec_max_steps": int(adaptive_exec_max_steps),
        "adaptive_exec_position_error_threshold": float(
            adaptive_exec_position_error_threshold
        ),
        "adaptive_exec_rotation_error_threshold": float(
            adaptive_exec_rotation_error_threshold
        ),
        "adaptive_exec_position_error_max": float(adaptive_exec_position_error_max),
        "adaptive_exec_rotation_error_max": float(adaptive_exec_rotation_error_max),
        "grasp_exec_steps": int(grasp_exec_steps),
        "grasp_width_min": float(grasp_width_min),
        "grasp_width_max": float(grasp_width_max),
        "grasp_lift_threshold": float(grasp_lift_threshold),
        "release_event_exec_enable": bool(release_event_exec_enable),
        "release_event_exec_max_steps": int(release_event_exec_max_steps),
        "release_event_min_width_change": float(release_event_min_width_change),
        "waypoint_position_tolerance": float(waypoint_position_tolerance),
        "waypoint_rotation_tolerance": float(waypoint_rotation_tolerance),
        "waypoint_gripper_tolerance": float(waypoint_gripper_tolerance),
        "rollback_chunks": int(rollback_chunks),
        "rollback_max_steps": int(rollback_max_steps),
        "rollback_position_tolerance": float(rollback_position_tolerance),
        "rollback_rotation_tolerance": float(rollback_rotation_tolerance),
        "rollback_request_count": int(rollback_request_count),
        "rollback_completed_count": int(rollback_completed_count),
        "rollback_step_count": int(rollback_step_count),
        "rollback_history_remaining": int(len(executed_chunk_start_controller_world_history)),
        "waypoint_hold_mean": (
            float(np.mean(waypoint_hold_counts)) if waypoint_hold_counts else 0.0
        ),
        "waypoint_hold_max": int(np.max(waypoint_hold_counts)) if waypoint_hold_counts else 0,
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
        "goal_predicate_names": goal_predicate_names,
        "initial_goal_predicate_values": initial_goal_predicate_values,
        "final_goal_predicate_values": (
            goal_predicate_values[-1] if goal_predicate_values else initial_goal_predicate_values
        ),
        "scene_joint_names": scene_joint_names,
        "scene_joint_types": scene_joint_types,
        "scene_joint_ranges": scene_joint_ranges,
        "initial_scene_joint_values": initial_scene_joint_values,
        "final_scene_joint_values": (
            scene_joint_values[-1] if scene_joint_values else initial_scene_joint_values
        ),
        "robot_scene_contact_step_count": int(np.sum(np.asarray(robot_scene_contact_counts) > 0)),
        "robot_scene_contact_step_ratio": (
            float(np.mean(np.asarray(robot_scene_contact_counts) > 0))
            if robot_scene_contact_counts
            else 0.0
        ),
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
        "oracle_source_raw_actions": np.asarray(
            oracle_source_raw_actions, dtype=np.float32
        ),
        "oracle_source_action_indices": np.asarray(
            oracle_source_action_indices, dtype=np.int64
        ),
        "oracle_scaled_deltas": np.asarray(oracle_scaled_deltas, dtype=np.float32),
        "model_action_rows": np.asarray(model_action_rows, dtype=np.float32),
        "predicted_action_chunks": np.asarray(predicted_action_chunks, dtype=np.float32),
        "oracle_chunk_source_indices": np.asarray(
            oracle_chunk_source_indices, dtype=np.int64
        ),
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
        "release_event_predicted_width_changes": np.asarray(
            release_event_predicted_width_changes, dtype=np.float32
        ),
        "chunk_executed_waypoint_counts": np.asarray(
            chunk_executed_waypoint_counts, dtype=np.int16
        ),
        "chunk_release_event_end_counts": np.asarray(
            chunk_release_event_end_counts, dtype=np.int16
        ),
        "chunk_grasp_upward_displacements": np.asarray(
            chunk_grasp_upward_displacements, dtype=np.float32
        ),
        "chunk_transported_grasp_flags": np.asarray(
            chunk_transported_grasp_flags, dtype=np.bool_
        ),
        "rollback_actions": np.asarray(rollback_actions, dtype=np.float32),
        "rollback_target_controller_pose9": np.asarray(
            rollback_target_controller_pose9, dtype=np.float32
        ),
        "rollback_achieved_controller_pose9": np.asarray(
            rollback_achieved_controller_pose9, dtype=np.float32
        ),
        "rollback_position_errors": np.asarray(rollback_position_errors, dtype=np.float32),
        "rollback_rotation_errors": np.asarray(rollback_rotation_errors, dtype=np.float32),
        "rollback_chunk_counts": np.asarray(rollback_chunk_counts, dtype=np.int16),
        "object_pose_names": object_pose_names,
        "initial_object_positions": initial_object_positions,
        "initial_object_quaternions": initial_object_quaternions,
        "object_positions": np.asarray(object_positions, dtype=np.float32),
        "object_quaternions": np.asarray(object_quaternions, dtype=np.float32),
        "goal_predicate_values": np.asarray(goal_predicate_values, dtype=np.bool_),
        "scene_joint_values": np.asarray(scene_joint_values, dtype=np.float32),
        "contact_pair_strings": np.asarray(contact_pair_strings, dtype=np.str_),
        "robot_scene_contact_pair_strings": np.asarray(
            robot_scene_contact_pair_strings, dtype=np.str_
        ),
        "contact_counts": np.asarray(contact_counts, dtype=np.int32),
        "robot_scene_contact_counts": np.asarray(robot_scene_contact_counts, dtype=np.int32),
        "waypoint_hold_counts": np.asarray(waypoint_hold_counts, dtype=np.int16),
        "video_frames": video_frames,
    }


def resolve_dataset_domain_task_source(
    *,
    suite: Any,
    suite_name: str,
    task_id: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the exact source HDF5 demos used by dataset conversion for one task."""

    if __package__ and __package__.startswith("benchmarks."):
        from .libero_hdf5_to_dataset import (
            find_demo_file,
            iter_demo_group_names,
            normalized_name,
        )
    else:
        from libero_setting.libero_hdf5_to_dataset import (
            find_demo_file,
            iter_demo_group_names,
            normalized_name,
        )

    demo_root_value = cfg.get("dataset_domain_demo_root")
    if not demo_root_value:
        raise ValueError(
            "--dataset-domain-env requires --dataset-domain-demo-root or demo_root in the config."
        )
    demo_root = Path(demo_root_value).expanduser().resolve()
    if not demo_root.is_dir():
        raise FileNotFoundError(f"Dataset-domain demo root does not exist: {demo_root}")

    task = suite.get_task(int(task_id))
    source_cfg = dict(cfg)
    source_cfg["suite"] = str(suite_name)
    # Evaluation must never silently download a different source dataset.
    source_cfg["download_demos"] = False
    demo_file = find_demo_file(task, demo_root, None, source_cfg)
    candidate_key = normalized_name(demo_file.stem)
    expected_task_keys = {
        normalized_name(str(getattr(task, "name", ""))),
        normalized_name(Path(str(getattr(task, "bddl_file", ""))).stem),
    }
    expected_task_keys.discard("")
    if not any(task_key in candidate_key for task_key in expected_task_keys):
        expected = ", ".join(sorted(expected_task_keys))
        raise FileNotFoundError(
            "Dataset-domain evaluation refused an ambiguous demonstration match: "
            f"suite={suite_name!r}, task_id={task_id}, expected task key in "
            f"{{{expected}}}, but the best candidate was {demo_file}. "
            "Ensure --dataset-domain-demo-root contains the HDF5 file for this exact task."
        )
    demo_names = iter_demo_group_names(demo_file)
    if not demo_names:
        raise RuntimeError(f"No demonstrations with states were found in {demo_file}.")
    return {
        "demo_root": str(demo_root),
        "demo_file": str(demo_file),
        "demo_names": demo_names,
    }


def restore_dataset_domain_episode(
    *,
    env: Any,
    task_source: dict[str, Any],
    episode_index: int,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore one demo's MuJoCo model and return its converted first observation state."""

    if __package__ and __package__.startswith("benchmarks."):
        from .libero_hdf5_to_dataset import load_demo_group, restore_demo_model
    else:
        from libero_setting.libero_hdf5_to_dataset import load_demo_group, restore_demo_model

    demo_names = list(task_source["demo_names"])
    if episode_index < 0 or episode_index >= len(demo_names):
        raise IndexError(
            f"Dataset-domain episode {episode_index} is unavailable; "
            f"source file provides {len(demo_names)} demo(s)."
        )
    demo_file = Path(task_source["demo_file"])
    demo_name = str(demo_names[episode_index])
    states, _actions, model_xml, source_fps = load_demo_group(demo_file, demo_name)
    dataset_state_index = int(cfg.get("dataset_domain_state_observation_offset", 1))
    if dataset_state_index not in (0, 1):
        raise ValueError(
            "dataset_domain_state_observation_offset must be 0 or 1, "
            f"got {dataset_state_index}."
        )
    # Exact source-action replay starts before actions[0]. Policy diagnostics
    # still start from the first observation represented in the converted
    # dataset, which is normally states[1] for official LIBERO v1 files.
    state_index = (
        0
        if bool(cfg.get("dataset_domain_oracle_actions", False))
        else dataset_state_index
    )
    if states.ndim != 2 or state_index >= len(states):
        raise ValueError(
            f"Cannot initialize dataset-domain episode from {demo_file}:{demo_name}: "
            f"states shape={states.shape}, requested state index={state_index}."
        )

    eval_control_freq = float(cfg["control"]["control_freq"])
    if source_fps is not None and not np.isclose(float(source_fps), eval_control_freq):
        raise ValueError(
            "Dataset-domain evaluation requires source and evaluation control frequencies "
            f"to match, but {demo_file}:{demo_name} was collected at {source_fps:g} Hz "
            f"and evaluation is configured for {eval_control_freq:g} Hz."
        )

    model_sha256 = restore_demo_model(env, model_xml, required=True)
    # reset_from_xml_string replaces the MuJoCo model while the outer env
    # object keeps the same identity. Do not reuse joint-address metadata from
    # the previous demo model.
    _GRIPPER_JOINT_CACHE.pop(id(env), None)
    metadata = {
        "enabled": True,
        "benchmark_comparable": False,
        "demo_file": str(demo_file),
        "demo_name": demo_name,
        "demo_episode_index": int(episode_index),
        "model_sha256": model_sha256,
        "source_fps": None if source_fps is None else float(source_fps),
        "state_index": int(state_index),
        "state_mapping": f"initial_state = states[{state_index}]",
        "dataset_observation_state_index": int(dataset_state_index),
        "source_state_count": int(len(states)),
    }
    return np.asarray(states[state_index]).copy(), metadata


def load_dataset_domain_raw_action_trajectory(
    *,
    env: Any,
    task_source: dict[str, Any],
    episode_index: int,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load source controls aligned with the restored dataset observation.

    Exact source replay starts from ``states[0]`` and executes ``actions[0:]``.
    This is intentionally distinct from a policy dataset-domain diagnostic,
    whose first converted observation is normally rendered from ``states[1]``.
    """

    if __package__ and __package__.startswith("benchmarks."):
        from .libero_hdf5_to_dataset import load_demo_group
    else:
        from libero_setting.libero_hdf5_to_dataset import load_demo_group

    demo_names = list(task_source["demo_names"])
    demo_name = str(demo_names[int(episode_index)])
    demo_file = Path(task_source["demo_file"])
    states, actions, _model_xml, source_fps = load_demo_group(demo_file, demo_name)
    dataset_state_offset = int(cfg.get("dataset_domain_state_observation_offset", 1))
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"Expected source LIBERO actions with shape (T, 7), got "
            f"{actions.shape} in {demo_file}:{demo_name}."
        )
    if dataset_state_offset < 0 or dataset_state_offset >= len(actions):
        raise ValueError(
            f"Cannot align source actions for states shape={states.shape}, "
            f"actions shape={actions.shape}, state_offset={dataset_state_offset}."
        )
    source_action_indices = np.arange(0, len(actions), dtype=np.int64)
    raw_trajectory = np.ascontiguousarray(
        np.asarray(actions[source_action_indices], dtype=np.float64)
    )

    for robot in env.robots:
        robot.controller.use_delta = False
    absolute_actions: list[np.ndarray] = []
    scaled_deltas: list[np.ndarray] = []
    for source_action_index, source_action in zip(
        source_action_indices,
        raw_trajectory,
        strict=True,
    ):
        # Anchor the desired setpoint to the source pose at which this command
        # was issued. This is a coordinate conversion of the original command,
        # not an added offset or task-specific contact heuristic.
        env.set_init_state(states[int(source_action_index)])
        absolute_action, _target_world, scaled_delta = (
            libero_delta_action_to_absolute_action(
                env,
                source_action,
                force_controller_update=True,
            )
        )
        absolute_actions.append(absolute_action)
        scaled_deltas.append(scaled_delta)

    # Leave the simulator at the observation used to start evaluation. The
    # episode runner sets this state again, but restoring it here prevents this
    # loader from exposing the final source state to any caller.
    env.set_init_state(states[0])
    refresh_arm_controller_state(env)
    absolute_trajectory = np.ascontiguousarray(
        np.stack(absolute_actions), dtype=np.float64
    )
    scaled_delta_trajectory = np.ascontiguousarray(
        np.stack(scaled_deltas), dtype=np.float64
    )
    return (
        raw_trajectory,
        absolute_trajectory,
        scaled_delta_trajectory,
        source_action_indices,
        {
            "demo_file": str(demo_file),
            "demo_name": demo_name,
            "source_fps": None if source_fps is None else float(source_fps),
            "initial_state_index": 0,
            "dataset_observation_state_index": int(dataset_state_offset),
            "source_action_indices": source_action_indices.tolist(),
            "trajectory_length": int(len(raw_trajectory)),
            "trajectory_format": "libero_normalized_osc_delta_plus_gripper",
            "execution_format": (
                "source_state_anchored_absolute_osc_pose_plus_source_gripper"
            ),
            "controller_use_delta": False,
        },
    )


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
    dataset_domain_env = bool(cfg.get("dataset_domain_env", False))
    task_source = (
        resolve_dataset_domain_task_source(
            suite=suite,
            suite_name=suite_name,
            task_id=int(task_id),
            cfg=cfg,
        )
        if dataset_domain_env
        else None
    )
    init_states = None
    if not dataset_domain_env:
        init_states = (
            get_task_init_states(suite, int(task_id))
            if task_init_states is None
            else np.asarray(task_init_states)
        )
        available_episode_count = int(len(init_states))
    else:
        assert task_source is not None
        available_episode_count = int(len(task_source["demo_names"]))

    if episode_indices is None:
        configured_episode_ids = cfg.get("episode_ids")
        if configured_episode_ids is None:
            resolved_episode_indices = list(range(int(cfg["episodes"])))
        else:
            resolved_episode_indices = [int(episode_index) for episode_index in configured_episode_ids]
    else:
        resolved_episode_indices = [int(episode_index) for episode_index in episode_indices]
    if len(set(resolved_episode_indices)) != len(resolved_episode_indices):
        raise ValueError(f"Duplicate episode indices for task {task_id}: {resolved_episode_indices}")
    invalid_episode_indices = [
        episode_index
        for episode_index in resolved_episode_indices
        if episode_index < 0 or episode_index >= available_episode_count
    ]
    if invalid_episode_indices:
        source_name = "source HDF5 demos" if dataset_domain_env else "task init states"
        raise ValueError(
            f"Invalid episode indices for task {task_id}: {invalid_episode_indices}; "
            f"{source_name} provide {available_episode_count} episode(s)."
        )

    # Standard LIBERO evaluation advances the seeded hard-reset sequence because
    # flattened init states omit model-level fixture placement. Dataset-domain
    # mode instead restores the exact per-demo XML, so no reset-RNG warmup is used.
    resolved_episode_indices = sorted(resolved_episode_indices)
    full_episode_indices = list(range(available_episode_count))
    if resolved_episode_indices != full_episode_indices:
        if dataset_domain_env:
            print(
                f"[info] suite={suite_name} task={task_id} uses training-source demo "
                f"subset {resolved_episode_indices}; this is a dataset-domain diagnostic, "
                f"not a standard LIBERO benchmark score.",
                flush=True,
            )
        else:
            print(
                f"[info] suite={suite_name} task={task_id} uses official fixed init-state "
                f"subset {resolved_episode_indices}; a full OpenVLA-style LIBERO score uses "
                f"all {len(full_episode_indices)} indices.",
                flush=True,
            )
    task_results: list[dict[str, Any]] = []
    task_definition = suite.get_task(int(task_id))
    task_name = str(getattr(task_definition, "name", f"task_{int(task_id):03d}"))
    task_language = str(getattr(task_definition, "language", f"{suite_name}:{int(task_id)}"))

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
                control_freq=float(cfg["control"].get("control_freq", 5.0)),
                horizon=(
                    int(LIBERO_STANDARD_MAX_STEPS[suite_name])
                    if bool(cfg.get("use_suite_max_steps", False))
                    and suite_name in LIBERO_STANDARD_MAX_STEPS
                    else int(cfg["control"]["max_steps"])
                ),
                ignore_done=False,
                env_seed=int(cfg.get("env_seed", 0)),
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

        # make_libero_env() seeds the environment and performs exactly one
        # reset, so a freshly-created environment already owns canonical
        # fixture layout 0. Advance from that layout only as needed. This keeps
        # serial, targeted, and episode-sharded evaluation on the same reset
        # sequence without consuming one extra random fixture placement.
        shared_layout_index = 0
        for episode_idx in resolved_episode_indices:
            environment_alignment: dict[str, Any] | None = None
            oracle_raw_action_trajectory: np.ndarray | None = None
            oracle_absolute_action_trajectory: np.ndarray | None = None
            oracle_scaled_delta_trajectory: np.ndarray | None = None
            oracle_raw_action_indices: np.ndarray | None = None
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
                if dataset_domain_env:
                    assert task_source is not None
                    episode_init_state, environment_alignment = restore_dataset_domain_episode(
                        env=env,
                        task_source=task_source,
                        episode_index=int(episode_idx),
                        cfg=cfg,
                    )
                    if bool(cfg.get("dataset_domain_oracle_actions", False)):
                        (
                            oracle_raw_action_trajectory,
                            oracle_absolute_action_trajectory,
                            oracle_scaled_delta_trajectory,
                            oracle_raw_action_indices,
                            oracle_metadata,
                        ) = load_dataset_domain_raw_action_trajectory(
                            env=env,
                            task_source=task_source,
                            episode_index=int(episode_idx),
                            cfg=cfg,
                        )
                        environment_alignment["oracle_actions"] = oracle_metadata
                    reset_warmup_count = 0
                    print(
                        "[eval] dataset-domain environment restored: "
                        f"suite={suite_name} task={task_id} episode={episode_idx} "
                        f"demo={environment_alignment['demo_name']} "
                        f"state_index={environment_alignment['state_index']}",
                        flush=True,
                    )
                else:
                    assert init_states is not None
                    episode_init_state = init_states[episode_idx]
                    if bool(cfg.get("recreate_env_per_episode", False)):
                        reset_warmup_count = int(episode_idx)
                    else:
                        if int(episode_idx) < shared_layout_index:
                            raise RuntimeError(
                                "Episode reset sequence cannot move backwards: "
                                f"current={shared_layout_index}, requested={episode_idx}."
                            )
                        reset_warmup_count = int(episode_idx) - shared_layout_index
                    for _ in range(reset_warmup_count):
                        env.reset()
                    if reset_warmup_count:
                        print(
                            "[eval] advanced LIBERO hard-reset RNG sequence: "
                            f"suite={suite_name} task={task_id} episode={episode_idx} "
                            f"skipped_resets={reset_warmup_count}",
                            flush=True,
                        )

                    if not bool(cfg.get("recreate_env_per_episode", False)):
                        shared_layout_index = int(episode_idx)
                result = run_episode(
                    infer=infer,
                    env=env,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    episode_index=int(episode_idx),
                    task_language=task_language,
                    init_state=episode_init_state,
                    cfg=cfg,
                    reset_env=False,
                    oracle_raw_action_trajectory=oracle_raw_action_trajectory,
                    oracle_absolute_action_trajectory=oracle_absolute_action_trajectory,
                    oracle_scaled_delta_trajectory=oracle_scaled_delta_trajectory,
                    oracle_raw_action_indices=oracle_raw_action_indices,
                )
                result["evaluation_protocol"] = evaluation_protocol_for_config(cfg)
                result["environment_alignment"] = (
                    environment_alignment
                    if environment_alignment is not None
                    else {
                        "enabled": False,
                        "benchmark_comparable": True,
                        "initial_state_source": "task_suite.get_task_init_states",
                    }
                )
                result["hard_reset_sequence_index"] = (
                    None if dataset_domain_env else int(episode_idx)
                )
                result["hard_reset_warmup_count"] = int(reset_warmup_count)

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
                    "evaluation_protocol": evaluation_protocol_for_config(cfg),
                    "environment_alignment": environment_alignment,
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
    task_init_states_by_id: dict[int, np.ndarray | None],
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

        ensure_libero_config(cfg.get("libero_config_path"), configured_demo_root(cfg))
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
    cfg["dataset_domain_env"] = bool(
        cfg_get(cfg, args.dataset_domain_env, "dataset_domain_env", False)
    )
    cfg["dataset_domain_oracle_actions"] = bool(
        cfg_get(
            cfg,
            args.dataset_domain_oracle_actions,
            "dataset_domain_oracle_actions",
            False,
        )
    )
    dataset_domain_demo_root = cfg_get(
        cfg,
        args.dataset_domain_demo_root,
        "dataset_domain_demo_root",
        cfg.get("demo_root"),
    )
    cfg["dataset_domain_demo_root"] = (
        str(Path(dataset_domain_demo_root).expanduser().resolve())
        if dataset_domain_demo_root
        else None
    )
    cfg["dataset_domain_state_observation_offset"] = int(
        cfg.get(
            "dataset_domain_state_observation_offset",
            cfg.get("state_observation_offset", 1),
        )
    )
    if cfg["dataset_domain_state_observation_offset"] not in (0, 1):
        raise ValueError(
            "state_observation_offset must be 0 or 1 for dataset-domain evaluation, "
            f"got {cfg['dataset_domain_state_observation_offset']}."
        )
    if cfg["dataset_domain_env"] and not cfg["dataset_domain_demo_root"]:
        raise ValueError(
            "--dataset-domain-env requires --dataset-domain-demo-root or demo_root in the config."
        )
    if cfg["dataset_domain_oracle_actions"] and not cfg["dataset_domain_env"]:
        raise ValueError(
            "--dataset-domain-oracle-actions requires --dataset-domain-env."
        )
    cfg["evaluation_identity"]["environment_domain"] = {
        "dataset_domain_env": bool(cfg["dataset_domain_env"]),
        "dataset_domain_oracle_actions": bool(
            cfg["dataset_domain_oracle_actions"]
        ),
        "demo_root": cfg["dataset_domain_demo_root"] if cfg["dataset_domain_env"] else None,
        "state_observation_offset": (
            int(cfg["dataset_domain_state_observation_offset"])
            if cfg["dataset_domain_env"]
            else None
        ),
        "benchmark_comparable": not bool(cfg["dataset_domain_env"]),
    }
    cfg["env_seed"] = int(cfg_get(cfg, args.env_seed, "env_seed", 0))
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
    cfg["visualize_action_trajectory"] = bool(
        cfg_get(
            cfg,
            args.visualize_action_trajectory,
            "visualize_action_trajectory",
            False,
        )
    )
    cfg["trajectory_vis_max_points"] = max(
        1,
        int(
            cfg_get(
                cfg,
                args.trajectory_vis_max_points,
                "trajectory_vis_max_points",
                50000,
            )
        ),
    )
    cfg["trajectory_vis_every_n_model_calls"] = max(
        1,
        int(
            cfg_get(
                cfg,
                args.trajectory_vis_every_n_model_calls,
                "trajectory_vis_every_n_model_calls",
                1,
            )
        ),
    )
    cfg["add_gripper_cloud"] = bool(cfg_get(cfg, args.add_gripper_cloud, "add_gripper_cloud", True))
    if args.gripper_points is not None:
        cfg["gripper_points"] = int(args.gripper_points)
    cfg.setdefault("camera_names", ["agentview", "robot0_eye_in_hand"])

    cfg["control"]["control_freq"] = float(
        cfg_get(cfg["control"], args.control_freq, "control_freq", cfg.get("control_freq", 5.0))
    )
    cfg["control"]["action_index"] = int(cfg_get(cfg["control"], args.action_index, "action_index", 0))
    cfg["control"]["exec_action_steps"] = int(cfg_get(cfg["control"], args.exec_action_steps, "exec_action_steps", 12))
    cfg["control"]["adaptive_exec_max_steps"] = max(
        int(cfg["control"]["exec_action_steps"]),
        int(
            cfg_get(
                cfg["control"],
                args.adaptive_exec_max_steps,
                "adaptive_exec_max_steps",
                int(cfg["control"]["exec_action_steps"]),
            )
        ),
    )
    cfg["control"]["adaptive_exec_position_error_threshold"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.adaptive_exec_position_error_threshold,
                "adaptive_exec_position_error_threshold",
                0.012,
            )
        ),
    )
    cfg["control"]["adaptive_exec_rotation_error_threshold"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.adaptive_exec_rotation_error_threshold,
                "adaptive_exec_rotation_error_threshold",
                0.10,
            )
        ),
    )
    cfg["control"]["adaptive_exec_position_error_max"] = max(
        float(cfg["control"]["adaptive_exec_position_error_threshold"]),
        float(
            cfg_get(
                cfg["control"],
                args.adaptive_exec_position_error_max,
                "adaptive_exec_position_error_max",
                0.03,
            )
        ),
    )
    cfg["control"]["adaptive_exec_rotation_error_max"] = max(
        float(cfg["control"]["adaptive_exec_rotation_error_threshold"]),
        float(
            cfg_get(
                cfg["control"],
                args.adaptive_exec_rotation_error_max,
                "adaptive_exec_rotation_error_max",
                0.15,
            )
        ),
    )
    cfg["control"]["grasp_exec_steps"] = max(
        int(cfg["control"]["exec_action_steps"]),
        int(
            cfg_get(
                cfg["control"],
                args.grasp_exec_steps,
                "grasp_exec_steps",
                int(cfg["control"]["exec_action_steps"]),
            )
        ),
    )
    cfg["control"]["grasp_width_min"] = max(
        0.0,
        float(cfg_get(cfg["control"], args.grasp_width_min, "grasp_width_min", 0.003)),
    )
    cfg["control"]["grasp_width_max"] = max(
        float(cfg["control"]["grasp_width_min"]),
        float(cfg_get(cfg["control"], args.grasp_width_max, "grasp_width_max", 0.070)),
    )
    cfg["control"]["grasp_lift_threshold"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.grasp_lift_threshold,
                "grasp_lift_threshold",
                0.015,
            )
        ),
    )
    cfg["control"]["release_event_exec_enable"] = bool(
        cfg_get(
            cfg["control"],
            args.release_event_exec_enable,
            "release_event_exec_enable",
            False,
        )
    )
    cfg["control"]["release_event_exec_max_steps"] = max(
        int(cfg["control"]["grasp_exec_steps"]),
        int(
            cfg_get(
                cfg["control"],
                args.release_event_exec_max_steps,
                "release_event_exec_max_steps",
                32,
            )
        ),
    )
    cfg["control"]["release_event_min_width_change"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.release_event_min_width_change,
                "release_event_min_width_change",
                0.02,
            )
        ),
    )
    cfg["control"]["waypoint_max_hold_steps"] = max(
        1,
        int(
            cfg_get(
                cfg["control"],
                args.waypoint_max_hold_steps,
                "waypoint_max_hold_steps",
                1,
            )
        ),
    )
    cfg["control"]["waypoint_position_tolerance"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.waypoint_position_tolerance,
                "waypoint_position_tolerance",
                0.002,
            )
        ),
    )
    cfg["control"]["waypoint_rotation_tolerance"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.waypoint_rotation_tolerance,
                "waypoint_rotation_tolerance",
                0.03,
            )
        ),
    )
    cfg["control"]["waypoint_gripper_tolerance"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.waypoint_gripper_tolerance,
                "waypoint_gripper_tolerance",
                0.004,
            )
        ),
    )
    cfg["control"]["rollback_chunks"] = max(
        1,
        int(cfg_get(cfg["control"], args.rollback_chunks, "rollback_chunks", 2)),
    )
    cfg["control"]["rollback_max_steps"] = max(
        1,
        int(cfg_get(cfg["control"], args.rollback_max_steps, "rollback_max_steps", 50)),
    )
    cfg["control"]["rollback_position_tolerance"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.rollback_position_tolerance,
                "rollback_position_tolerance",
                cfg["control"]["waypoint_position_tolerance"],
            )
        ),
    )
    cfg["control"]["rollback_rotation_tolerance"] = max(
        0.0,
        float(
            cfg_get(
                cfg["control"],
                args.rollback_rotation_tolerance,
                "rollback_rotation_tolerance",
                cfg["control"]["waypoint_rotation_tolerance"],
            )
        ),
    )
    cfg["control"]["max_steps"] = int(cfg_get(cfg["control"], args.max_steps, "max_steps", 1000))
    cfg["use_suite_max_steps"] = bool(
        cfg_get(cfg, args.use_suite_max_steps, "use_suite_max_steps", False)
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
    cfg["control"]["gripper_delta_alignment"] = str(
        cfg_get(
            cfg["control"],
            args.gripper_delta_alignment,
            "gripper_delta_alignment",
            "current_minus_previous",
        )
    )
    cfg["control"]["synchronize_gripper_controller_state"] = bool(
        cfg_get(
            cfg["control"],
            args.synchronize_gripper_controller_state,
            "synchronize_gripper_controller_state",
            True,
        )
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
    cfg["episode_ids"] = (
        [int(episode_index) for episode_index in args.episode_id]
        if args.episode_id is not None
        else None
    )

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
    if cfg.get("episode_ids") is not None and configured_environment_workers > 1:
        raise ValueError(
            "--episode-id is a targeted debugging mode and currently requires "
            "--task-workers 1 --episode-workers-per-task 1."
        )
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
        if bool(cfg.get("visualize_action_trajectory", False)):
            raise ValueError(
                "Online action-trajectory visualization requires serial evaluation: "
                "--task-workers 1 --episode-workers-per-task 1."
            )
    if cfg["episode_workers_per_task"] > 1 and cfg["task_worker_backend"] != "process":
        raise ValueError("--episode-workers-per-task > 1 requires --task-worker-backend process.")
    if cfg["isolated_policy_workers"] > 1 and cfg["episode_workers_per_task"] > 1:
        raise ValueError(
            "--isolated-policy-workers > 1 already provides independent rollout processes; "
            "combine it with --episode-workers-per-task 1."
        )
    if cfg["isolated_policy_workers"] > 1 and bool(cfg.get("visualize_foreground", False)):
        raise ValueError("Isolated policy workers require --no-visualize-foreground.")
    if cfg["isolated_policy_workers"] > 1 and bool(
        cfg.get("visualize_action_trajectory", False)
    ):
        raise ValueError(
            "Online action-trajectory visualization requires --isolated-policy-workers 1."
        )

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
    acquire_evaluation_run_lock(output_dir)
    if args.suite_gpu_ids is not None:
        run_multi_gpu_suite_launcher(
            args=args,
            cfg=cfg,
            suite_names=suite_names,
            output_dir=output_dir,
        )
        return
    ensure_libero_config(cfg.get("libero_config_path"), configured_demo_root(cfg))

    from libero.libero import benchmark
    selected_episode_count = (
        len(cfg["episode_ids"])
        if cfg.get("episode_ids") is not None
        else int(cfg["episodes"])
    )
    episode_horizons = {
        suite_name: (
            int(LIBERO_STANDARD_MAX_STEPS[suite_name])
            if bool(cfg.get("use_suite_max_steps", False))
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
            f"episodes={selected_episode_count}, inference_batch_size=1, "
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
                episodes_per_task=int(selected_episode_count),
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
        f"suites={suite_names}, episodes={selected_episode_count}, "
        f"task_workers={cfg['task_workers']}, "
        f"isolated_policy_workers={cfg['isolated_policy_workers']}, "
        f"episode_workers_per_task={cfg['episode_workers_per_task']}, "
        f"task_worker_backend={cfg['task_worker_backend']}, "
        f"inference_batch_size={cfg['inference_batch_size']}, "
        f"recreate_env_per_episode={cfg['recreate_env_per_episode']}, "
        f"exec_action_steps={cfg['control']['exec_action_steps']}, "
        f"adaptive_exec_max_steps={cfg['control']['adaptive_exec_max_steps']}, "
        f"adaptive_exec_error_band=("
        f"{cfg['control']['adaptive_exec_position_error_threshold']},"
        f"{cfg['control']['adaptive_exec_position_error_max']})m/("
        f"{cfg['control']['adaptive_exec_rotation_error_threshold']},"
        f"{cfg['control']['adaptive_exec_rotation_error_max']})rad, "
        f"grasp_exec_steps={cfg['control']['grasp_exec_steps']}, "
        f"grasp_width_band=({cfg['control']['grasp_width_min']},"
        f"{cfg['control']['grasp_width_max']})m, "
        f"grasp_lift_threshold={cfg['control']['grasp_lift_threshold']}m, "
        f"release_event_exec={cfg['control']['release_event_exec_enable']}/"
        f"{cfg['control']['release_event_exec_max_steps']}rows/"
        f"{cfg['control']['release_event_min_width_change']}m, "
        f"waypoint_max_hold_steps={cfg['control']['waypoint_max_hold_steps']}, "
        f"gripper_threshold={cfg['control']['gripper_threshold']}, "
        f"gripper_control_mode={cfg['control']['gripper_control_mode']}, "
        f"gripper_delta_threshold={cfg['control']['gripper_delta_threshold']}, "
        f"gripper_target_tolerance={cfg['control']['gripper_target_tolerance']}, "
        f"policy_noise_seed={cfg['policy_noise_seed']}, "
        f"deterministic_torch={cfg['deterministic_torch']}, "
        f"env_seed={cfg['env_seed']}, "
        f"dataset_domain_env={cfg['dataset_domain_env']}, "
        f"dataset_domain_oracle_actions={cfg['dataset_domain_oracle_actions']}, "
        f"benchmark_comparable={not cfg['dataset_domain_env']}, "
        f"use_suite_max_steps={cfg['use_suite_max_steps']}, "
        f"episode_horizons={episode_horizons}, "
        f"save_video={cfg['save_video']}, "
        f"render_mode={cfg.get('render_mode')}, "
        f"render_every_n_steps={cfg.get('render_every_n_steps')}, "
        f"visualize_foreground={cfg.get('visualize_foreground')}, "
        f"visualize_action_trajectory={cfg.get('visualize_action_trajectory')}, "
        f"trajectory_vis_every_n_model_calls="
        f"{cfg.get('trajectory_vis_every_n_model_calls')}"
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
                episodes_per_task=int(selected_episode_count),
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
                task_init_states_by_id: dict[int, np.ndarray | None]
                if bool(cfg.get("dataset_domain_env", False)):
                    # Dataset-domain workers resolve source demos themselves.
                    # Avoid loading unrelated standard benchmark init states.
                    task_init_states_by_id = {
                        int(task_id): None for task_id in task_ids
                    }
                else:
                    task_init_states_by_id = {
                        int(task_id): init_states_as_numpy(
                            get_task_init_states(suite, int(task_id))
                        )
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
        _close_umi_trajectory_visualizer(infer)
        infer.close()

    write_eval_reports(output_dir, cfg, suite_names, all_task_summaries)
    print(json.dumps(aggregate_task_results(all_task_summaries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
