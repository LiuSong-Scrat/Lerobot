#!/usr/bin/env python
"""Fast, non-blocking LIBERO point-cloud inference/evaluation runner.

VERSION_MARKER: V14_GEOM_ATTACH_NO_CONTACT_TUNNEL

This script is intentionally *not* a LIBERO benchmark-compatible evaluator. It is
for rapid policy sanity checks with the Song point-cloud SmolVLA contract:

  observation.point_cloud: (N, 6) xyzrgb in the current end-effector frame,
  observation.state:      pose9 identity + measured gripper width,
  action:                 pose9 target + continuous gripper width.

The default executor uses a kinematic MuJoCo IK step, so the model UMI
trajectory is first converted to absolute world/controller targets, and the arm
target plus continuous gripper target from the same action row are applied in
the same simulation instant. A lightweight AttachManager keeps a grasped free-joint object
rigidly attached to the EEF while the gripper stays closed, which is useful for
fast task testing when you do not want to wait for robosuite controller dynamics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Keep headless mode usable by default. Override from shell when you want a GUI.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def configure_mujoco_render_backend(render_mode: str) -> None:
    """Switch MuJoCo/OpenGL backend for a visible desktop viewer.

    The fast runner defaults to EGL for headless RGB-D. For viewer3d/onscreen,
    EGL will not create a visible desktop window, so switch to GLX on Linux and
    remove PYOPENGL_PLATFORM if it was forced to egl/glfw.
    """
    mode = str(render_mode or "none").lower()
    if mode not in {"viewer3d", "mujoco", "onscreen", "headed", "human"}:
        return
    if not sys.platform.startswith("linux"):
        return
    mujoco_gl = os.environ.get("MUJOCO_GL", "").lower().strip()
    if mujoco_gl in {"", "egl", "glfw"}:
        os.environ["MUJOCO_GL"] = "glx"
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "").lower().strip()
    if pyopengl_platform in {"egl", "glfw"}:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    print(
        f"[info] render_backend mode={mode} "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}"
    )


# -----------------------------------------------------------------------------
# Small geometry helpers
# -----------------------------------------------------------------------------

IDENTITY_POSE9 = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def pose9_to_hmat(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float64).reshape(-1)
    if pose9.shape[0] < 9:
        raise ValueError(f"pose9 must have at least 9 values, got {pose9.shape}")
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = pose9[:3]
    x_axis = pose9[3:6]
    y_axis = pose9[6:9]
    Rm = rot6d_to_matrix(np.concatenate([x_axis, y_axis], axis=0))
    H[:3, :3] = Rm
    return H


def hmat_to_pose9(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    return np.concatenate([H[:3, 3], H[:3, 0], H[:3, 1]], axis=0).astype(np.float32)


def hmat_inv(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = H[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ H[:3, 3]
    return out


def quat_wxyz_from_mat(Rm: np.ndarray) -> np.ndarray:
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def mat_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def pose_error_axis_angle(target_R: np.ndarray, current_R: np.ndarray) -> np.ndarray:
    # Error that rotates current_R into target_R in world coordinates.
    return R.from_matrix(target_R @ current_R.T).as_rotvec()


def ensure_2d_action_chunk(action: Any) -> np.ndarray:
    if torch.is_tensor(action):
        action = action.detach().float().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 3:
        action = action[0]
    elif action.ndim == 2:
        pass
    elif action.ndim == 1:
        action = action[None, :]
    else:
        raise ValueError(f"Expected action chunk with ndim 1/2/3, got {action.shape}")
    if action.shape[-1] < 10:
        raise ValueError(f"Expected pose9 + gripper action, got {action.shape}")
    return np.ascontiguousarray(action[:, :10], dtype=np.float32)


def sample_or_pad_points(xyzrgb: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    if xyzrgb.ndim != 2 or xyzrgb.shape[-1] != 6:
        raise ValueError(f"Expected point cloud shape (N, 6), got {xyzrgb.shape}")
    if len(xyzrgb) == n:
        return xyzrgb
    if len(xyzrgb) == 0:
        return np.zeros((n, 6), dtype=np.float32)
    if len(xyzrgb) > n:
        return xyzrgb[rng.choice(len(xyzrgb), n, replace=False)]
    extra = rng.choice(len(xyzrgb), n - len(xyzrgb), replace=True)
    return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0).astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Configuration / task resolution
# -----------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg.get(key, default)


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int


def parse_task_spec(text: str) -> TaskSpec:
    # Accept libero_object:0, libero_object/0, libero_object,0
    m = re.match(r"^\s*([^:/,\s]+)\s*[:/,]\s*(\d+)\s*$", text)
    if not m:
        raise argparse.ArgumentTypeError(f"Task spec must look like 'libero_object:0', got {text!r}")
    return TaskSpec(m.group(1), int(m.group(2)))


def resolve_task_specs(args: argparse.Namespace, cfg: dict[str, Any]) -> list[TaskSpec]:
    """Resolve task specs.

    Modes:
      * --task-spec suite:id can be repeated for explicit task selection.
      * --suite ... --all-tasks expands every task id in each requested suite.
      * --suite ... --task-id ... forms a Cartesian product.
      * with no task args, keep the previous fast sanity-check default: task 0
        from libero_spatial/object/goal/10.
    """
    if args.task_spec:
        specs = list(args.task_spec)
    else:
        default_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        suites = args.suite if args.suite else cfg.get("suites", cfg.get("suite"))
        if isinstance(suites, str):
            suites = [suites]

        if bool(getattr(args, "all_tasks", False)):
            suites = list(suites) if suites is not None else default_suites
            from libero_pointcloud_utils import ensure_libero_config

            ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
            from libero.libero import benchmark

            benchmark_dict = benchmark.get_benchmark_dict()
            specs = []
            for suite_name in suites:
                suite_name = str(suite_name)
                suite_cls = benchmark_dict[suite_name]
                suite = suite_cls()
                task_count = len(getattr(suite, "tasks", []) or [])
                if task_count <= 0:
                    # Fallback for LIBERO versions that expose get_num_tasks().
                    get_num_tasks = getattr(suite, "get_num_tasks", None)
                    if callable(get_num_tasks):
                        task_count = int(get_num_tasks())
                if task_count <= 0:
                    raise RuntimeError(f"Could not determine task count for suite={suite_name!r}.")
                specs.extend(TaskSpec(suite_name, task_id) for task_id in range(int(task_count)))
        else:
            task_ids = args.task_id if args.task_id is not None else cfg.get("task_ids")
            if suites is None and task_ids is None:
                specs = [
                    TaskSpec("libero_spatial", 0),
                    TaskSpec("libero_object", 0),
                    TaskSpec("libero_goal", 0),
                    TaskSpec("libero_10", 0),
                ]
            else:
                suites = list(suites) if suites is not None else default_suites
                if task_ids is None:
                    task_ids = [0]
                specs = [TaskSpec(str(suite), int(task_id)) for suite in suites for task_id in task_ids]

    if args.first_n_tasks is not None:
        specs = specs[: int(args.first_n_tasks)]
    if bool(getattr(args, "require_four_tasks", False)) and not bool(getattr(args, "all_tasks", False)) and len(specs) != 4:
        raise ValueError(f"--require-four-tasks expected 4 task specs, got {len(specs)}: {specs}")
    return specs

# -----------------------------------------------------------------------------
# Policy loading / inference.
#
# IMPORTANT: for this project, online inference should use smolvla_model_inference.py.
# The generic LeRobot factory preprocessor can enter umi_processor, which expects
# a world-frame EEF pose sequence and will fail with obs_pose9_eff_to_world=None
# for the minimal online batch. Therefore the default path below is the same
# contract used by libero_pointcloud_eval.py:
#   infer.predict_action_chunk_obs({"point_cloud": pc, "state": identity_pose9_gripper(w)},
#                                  task=..., postprocess=True,
#                                  state_pose_mode="identity")
# The factory path is retained only as an explicit debug fallback.
# -----------------------------------------------------------------------------


class IdentityProcessor:
    def __call__(self, x):
        return x


@dataclass
class PolicyBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device
    infer: Any | None = None
    identity_pose9_gripper: Any | None = None
    loader_name: str = "unknown"


def _device_string(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return str(device)


def try_load_policy_with_inference_wrapper(args: argparse.Namespace, device: torch.device) -> PolicyBundle:
    """Load the local inference wrapper used by existing LIBERO point-cloud eval scripts."""
    try:
        from smolvla_model_inference import SmolVLA_ModelInference, identity_pose9_gripper
    except Exception as exc:
        raise RuntimeError(
            "Could not import smolvla_model_inference.py from the benchmark scripts directory. "
            "Run this script from benchmarks/song_real_libero/scripts or keep that directory on PYTHONPATH."
        ) from exc

    policy_path = str(args.policy_path) if args.policy_path is not None else None
    policy_repo_id = getattr(args, "policy_repo_id", None)
    infer = SmolVLA_ModelInference(
        policy_path=policy_path,
        policy_repo_id=policy_repo_id,
        device=_device_string(device),
    )
    policy = getattr(infer, "policy", None)
    if policy is not None:
        try:
            policy.to(device)
        except Exception:
            pass
        try:
            policy.eval()
        except Exception:
            pass
    print("[info] policy_loader=SmolVLA_ModelInference predict_action_chunk_obs state_pose_mode=identity")
    return PolicyBundle(
        policy=policy,
        preprocessor=IdentityProcessor(),
        postprocessor=IdentityProcessor(),
        device=device,
        infer=infer,
        identity_pose9_gripper=identity_pose9_gripper,
        loader_name="SmolVLA_ModelInference",
    )


def make_dataset_metadata(dataset_repo_id: str | None, dataset_root: str | Path | None):
    if not dataset_repo_id and not dataset_root:
        return None
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        repo_id = dataset_repo_id or root.name
        return LeRobotDatasetMetadata(repo_id, root=root)
    return LeRobotDatasetMetadata(str(dataset_repo_id))


def try_load_policy_with_factory(args: argparse.Namespace, device: torch.device) -> PolicyBundle | None:
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    except Exception:
        return None

    try:
        policy_cfg = PreTrainedConfig.from_pretrained(str(args.policy_path))
    except Exception:
        return None

    for name, value in (
        ("pretrained_path", str(args.policy_path)),
        ("path", str(args.policy_path)),
        ("device", _device_string(device)),
        ("push_to_hub", False),
    ):
        if hasattr(policy_cfg, name):
            try:
                setattr(policy_cfg, name, value)
            except Exception:
                pass

    ds_meta = make_dataset_metadata(args.dataset_repo_id, args.dataset_root)
    rename_map = getattr(args, "rename_map", None) or {}

    try:
        policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta, rename_map=rename_map)
    except TypeError:
        policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()

    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    if ds_meta is not None:
        try:
            processor_kwargs["preprocessor_overrides"] = {
                "device_processor": {"device": _device_string(device)},
                "normalizer_processor": {
                    "stats": ds_meta.stats,
                    "features": {**policy.config.input_features, **policy.config.output_features},
                    "norm_map": policy.config.normalization_mapping,
                },
                "rename_observations_processor": {"rename_map": rename_map},
            }
            postprocessor_kwargs["postprocessor_overrides"] = {
                "unnormalizer_processor": {
                    "stats": ds_meta.stats,
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                }
            }
        except Exception:
            pass
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(args.policy_path),
            **processor_kwargs,
            **postprocessor_kwargs,
        )
    except Exception:
        preprocessor, postprocessor = IdentityProcessor(), IdentityProcessor()
    print("[warn] policy_loader=factory; this path is only a fallback and may hit umi_processor requirements")
    return PolicyBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        loader_name="factory",
    )


def try_load_policy_direct(args: argparse.Namespace, device: torch.device) -> PolicyBundle:
    candidates = [
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        ("lerobot.policies.smolvla.policy_smolvla", "SmolVLAPolicy"),
        ("lerobot.policies.smolvla", "SmolVLAPolicy"),
    ]
    last_err: Exception | None = None
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, "from_pretrained"):
                policy = cls.from_pretrained(str(args.policy_path))
            else:
                policy = cls(str(args.policy_path))
            policy.to(device)
            policy.eval()
            print("[warn] policy_loader=direct; normalizers/preprocessors may be missing")
            return PolicyBundle(
                policy=policy,
                preprocessor=IdentityProcessor(),
                postprocessor=IdentityProcessor(),
                device=device,
                loader_name="direct",
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            last_err = exc
    raise RuntimeError(
        "Could not load policy directly. Prefer the SmolVLA_ModelInference loader, "
        "or pass --allow-factory-policy-loader to debug the generic loader."
    ) from last_err


def load_policy_bundle(args: argparse.Namespace) -> PolicyBundle:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not bool(getattr(args, "allow_factory_policy_loader", False)):
        # Default and intended path. Do not silently fall back to factory, because
        # that hides the real problem and reproduces the umi_processor None crash.
        return try_load_policy_with_inference_wrapper(args, device)

    try:
        return try_load_policy_with_inference_wrapper(args, device)
    except Exception as exc:
        print(f"[warn] SmolVLA_ModelInference loader failed, trying factory fallback: {exc!r}")
    bundle = try_load_policy_with_factory(args, device)
    if bundle is not None:
        return bundle
    return try_load_policy_direct(args, device)


def reset_policy_bundle(bundle: PolicyBundle) -> None:
    """Reset recurrent / action-queue state at episode boundaries when available."""
    infer = getattr(bundle, "infer", None)
    if infer is not None:
        for obj, names in ((getattr(infer, "policy", None), ("reset",)), (infer, ("policy_reset", "reset"))):
            if obj is None:
                continue
            for name in names:
                fn = getattr(obj, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        return
    policy = getattr(bundle, "policy", None)
    fn = getattr(policy, "reset", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = move_tensors_to_device(value, device)
        else:
            out[key] = value
    return out


@torch.no_grad()
def predict_action_chunk(
    bundle: PolicyBundle,
    point_cloud_eff: np.ndarray,
    gripper_width: float,
    task_language: str,
    current_world_pose9: np.ndarray | None = None,
) -> np.ndarray:
    point_cloud_eff = np.ascontiguousarray(point_cloud_eff, dtype=np.float32)

    if getattr(bundle, "infer", None) is not None:
        identity_fn = bundle.identity_pose9_gripper
        if identity_fn is None:
            state = np.concatenate([IDENTITY_POSE9, np.asarray([gripper_width], dtype=np.float32)], axis=0)
        else:
            # This mirrors libero_pointcloud_eval.py exactly, including identity pose mode.
            state = identity_fn(float(gripper_width))
        pred = bundle.infer.predict_action_chunk_obs(
            {"point_cloud": point_cloud_eff, "state": state},
            task=str(task_language),
            postprocess=True,
            state_pose_mode="identity",
        )
        return ensure_2d_action_chunk(pred)

    # Explicit fallback path only. Use LeRobot training-like shapes: point cloud
    # (B, T_obs, N, 6), state (B, T_obs, 10), and provide several world-pose aliases
    # so local UMI processors that look for obs_pose9_eff_to_world do not receive None.
    world_pose9 = np.asarray(current_world_pose9 if current_world_pose9 is not None else IDENTITY_POSE9, dtype=np.float32)
    state = np.concatenate([IDENTITY_POSE9, np.asarray([gripper_width], dtype=np.float32)], axis=0)
    state_world = np.concatenate([world_pose9, np.asarray([gripper_width], dtype=np.float32)], axis=0)
    batch: dict[str, Any] = {
        "task": [str(task_language)],
        "observation.point_cloud": torch.from_numpy(point_cloud_eff).unsqueeze(0).unsqueeze(0),
        "observation.point_cloud_is_pad": torch.zeros((1, 1, point_cloud_eff.shape[0]), dtype=torch.bool),
        "observation.state": torch.from_numpy(state).view(1, 1, 10),
        "observation.state_world": torch.from_numpy(state_world).view(1, 1, 10),
        "observation.world_ee_pose": torch.from_numpy(world_pose9).view(1, 1, 9),
        "observation.ee_pose_world": torch.from_numpy(world_pose9).view(1, 1, 9),
        "worldflow.current_ee_pose": torch.from_numpy(world_pose9).view(1, 9),
        "worldflow.ee_poses": torch.from_numpy(world_pose9).view(1, 1, 9),
        "worldflow.step_is_pad": torch.zeros((1, 1), dtype=torch.bool),
    }
    batch = move_tensors_to_device(batch, bundle.device)
    model_batch = bundle.preprocessor(batch)
    if hasattr(bundle.policy, "predict_action_chunk"):
        pred = bundle.policy.predict_action_chunk(model_batch)
    elif hasattr(bundle.policy, "select_action"):
        pred = bundle.policy.select_action(model_batch)
    else:
        raise AttributeError("Policy has neither predict_action_chunk() nor select_action().")
    pred = bundle.postprocessor(pred)
    return ensure_2d_action_chunk(pred)


# -----------------------------------------------------------------------------
# MuJoCo / robosuite low-level helpers
# -----------------------------------------------------------------------------


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


def settle_scene_after_reset(
    env: Any,
    *,
    steps: int,
    keep_robot_fixed: bool = True,
    render: bool = False,
    cfg: dict[str, Any] | None = None,
    debug: bool = False,
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
        except Exception as exc:
            if debug:
                print(f"[warn] settle could not infer arm indexes; robot not fixed: {exc!r}")
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
        if render and cfg is not None and settle_step % max(1, int(cfg.get("render_every_n_steps", 1))) == 0:
            render_mujoco_viewer(env, cfg, settle_step)
    sim.forward()
    raw_obs = get_raw_obs(env, force_update=True)
    if debug:
        try:
            eef_pos = np.asarray(raw_obs.get("robot0_eef_pos", []), dtype=np.float32).reshape(-1)
            print(f"[debug] settle_done steps={steps} keep_robot_fixed={bool(keep_robot_fixed)} robot0_eef_pos={eef_pos.round(5).tolist()}")
        except Exception:
            print(f"[debug] settle_done steps={steps} keep_robot_fixed={bool(keep_robot_fixed)}")
    return raw_obs


def safe_check_success(env: Any) -> bool:
    for name in ("check_success", "_check_success"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    inner = getattr(env, "env", None)
    if inner is not None:
        for name in ("check_success", "_check_success"):
            fn = getattr(inner, name, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    pass
    return False


def safe_reward(env: Any) -> float:
    fn = getattr(env, "reward", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            return 0.0
    return 0.0


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


def name_to_id(model: Any, kind: str, name: str) -> int:
    for method in (f"{kind}_name2id", f"name2id"):
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


def body_pose_hmat(sim: Any, body_id: int) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    H[:3, :3] = mat_from_quat_wxyz(np.asarray(sim.data.body_xquat[body_id], dtype=np.float64))
    return H


def site_pose_hmat(sim: Any, site_id: int) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    H[:3, :3] = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return H


def current_controller_eef_world(env: Any) -> np.ndarray:
    """Return the exact EEF world pose used by robosuite OSC/controller.

    This mirrors libero_pointcloud_eval.py: LIBERO observation's model EEF frame
    is not necessarily the same as the controller/site frame.  Even in this
    instant-IK runner we should use the controller frame as the execution frame,
    otherwise the wrist can look correct in UMI space but wrong in MuJoCo.
    """
    if not getattr(env, "robots", None):
        raise ValueError("Expected LIBERO env to expose at least one robot.")
    controller = getattr(env.robots[0], "controller", None)
    if controller is None:
        raise ValueError("Expected the first LIBERO robot to expose an OSC controller.")
    try:
        controller.update(force=True)
    except TypeError:
        controller.update()
    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = np.asarray(controller.ee_ori_mat, dtype=np.float32)
    H[:3, 3] = np.asarray(controller.ee_pos, dtype=np.float32)
    return H


def pose_frame_error(current_world: np.ndarray, target_world: np.ndarray) -> tuple[float, float]:
    current_world = np.asarray(current_world, dtype=np.float64)
    target_world = np.asarray(target_world, dtype=np.float64)
    pos_error = float(np.linalg.norm(target_world[:3, 3] - current_world[:3, 3]))
    rot_delta = target_world[:3, :3] @ current_world[:3, :3].T
    rot_error = float(np.linalg.norm(R.from_matrix(rot_delta).as_rotvec()))
    return pos_error, rot_error


def _site_score_to_controller(sim: Any, site_id: int, controller_world: np.ndarray) -> float:
    site_H = site_pose_hmat(sim, site_id)
    pos_err, rot_err = pose_frame_error(site_H, controller_world)
    return float(pos_err + 0.05 * rot_err)


def find_eef_site(env: Any) -> tuple[str, int]:
    """Find the MuJoCo site whose pose matches the robosuite controller EEF.

    The previous fast runner returned the first plausible "grip/eef" site.  In
    LIBERO there can be several gripper/body/site frames, and choosing the wrong
    one makes instant IK chase a target in the wrong frame.  The ref evaluator
    uses controller.ee_pos / controller.ee_ori_mat as the execution frame, so we
    select the site closest to that frame.
    """
    sim = get_sim(env)
    model = sim.model
    site_names = model_names(model, "site")
    if not site_names:
        raise RuntimeError("Could not infer EEF site: model exposes no sites.")

    controller_world: np.ndarray | None = None
    try:
        controller_world = current_controller_eef_world(env)
    except Exception:
        controller_world = None

    robot = getattr(env, "robots", [None])[0]
    candidates: list[Any] = []
    if robot is not None:
        # Prefer controller attributes first because they define the frame the
        # ref absolute-pose evaluator actually commands.
        controller = getattr(robot, "controller", None)
        if controller is not None:
            for attr in ("eef_name", "eef_site", "ref_name"):
                if hasattr(controller, attr):
                    candidates.append(getattr(controller, attr))
        for attr in ("eef_site_id", "eef_site_name", "eef_site", "grip_site", "grip_site_id"):
            if hasattr(robot, attr):
                candidates.append(getattr(robot, attr))

    candidate_ids: list[int] = []
    for cand in candidates:
        if isinstance(cand, dict):
            cand = next(iter(cand.values()))
        if isinstance(cand, (int, np.integer)) and 0 <= int(cand) < len(site_names):
            candidate_ids.append(int(cand))
        elif isinstance(cand, str) and cand in site_names:
            candidate_ids.append(name_to_id(model, "site", cand))

    regexes = [r"grip_site", r"eef", r"right.*grip", r"robot0.*grip", r"hand"]
    for rgx in regexes:
        for i, name in enumerate(site_names):
            if re.search(rgx, name, flags=re.I):
                candidate_ids.append(i)

    # Deduplicate while preserving order.
    seen: set[int] = set()
    candidate_ids = [i for i in candidate_ids if not (i in seen or seen.add(i))]
    if not candidate_ids:
        candidate_ids = list(range(len(site_names)))

    if controller_world is not None:
        best_id = min(candidate_ids, key=lambda sid: _site_score_to_controller(sim, sid, controller_world))
    else:
        best_id = candidate_ids[0]

    name = site_names[int(best_id)]
    if controller_world is not None:
        score = _site_score_to_controller(sim, int(best_id), controller_world)
        print(f"[info] ik_eef_site={name} site_id={int(best_id)} score_to_controller={score:.6f}")
    else:
        print(f"[info] ik_eef_site={name} site_id={int(best_id)}")
    return name, int(best_id)

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


def get_site_jacobian(sim: Any, site_name: str, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    nv = int(sim.model.nv)
    data = sim.data
    if hasattr(data, "get_site_jacp") and hasattr(data, "get_site_jacr"):
        jacp = np.asarray(data.get_site_jacp(site_name), dtype=np.float64).reshape(3, nv)
        jacr = np.asarray(data.get_site_jacr(site_name), dtype=np.float64).reshape(3, nv)
        return jacp, jacr
    try:
        import mujoco

        jacp = np.zeros((3, nv), dtype=np.float64)
        jacr = np.zeros((3, nv), dtype=np.float64)
        mj_model = getattr(sim.model, "_model", sim.model)
        mj_data = getattr(sim.data, "_data", sim.data)
        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
        return jacp, jacr
    except Exception as exc:
        raise RuntimeError("Could not compute MuJoCo site Jacobian.") from exc


@dataclass
class IKConfig:
    iters: int = 80
    damping: float = 1e-3
    pos_tol: float = 1e-4
    rot_tol: float = 2e-3
    step_clip: float = 0.05
    rot_weight: float = 0.5


class InstantIKExecutor:
    def __init__(self, env: Any, cfg: IKConfig):
        self.env = env
        self.sim = get_sim(env)
        self.site_name, self.site_id = find_eef_site(env)
        self.qpos_idx, self.qvel_idx = robot_arm_indices(env)
        self.cfg = cfg

    def current_pose(self) -> np.ndarray:
        return site_pose_hmat(self.sim, self.site_id)

    def goto(self, target_H_world: np.ndarray) -> dict[str, float]:
        target_H_world = np.asarray(target_H_world, dtype=np.float64)
        c = self.cfg
        last_pos_err = math.inf
        last_rot_err = math.inf
        converged = False
        for _ in range(int(c.iters)):
            self.sim.forward()
            current_H = self.current_pose()
            pos_err = target_H_world[:3, 3] - current_H[:3, 3]
            rot_err = pose_error_axis_angle(target_H_world[:3, :3], current_H[:3, :3])
            last_pos_err = float(np.linalg.norm(pos_err))
            last_rot_err = float(np.linalg.norm(rot_err))
            if last_pos_err <= c.pos_tol and last_rot_err <= c.rot_tol:
                converged = True
                break
            jacp, jacr = get_site_jacobian(self.sim, self.site_name, self.site_id)
            J = np.concatenate([jacp[:, self.qvel_idx], c.rot_weight * jacr[:, self.qvel_idx]], axis=0)
            err = np.concatenate([pos_err, c.rot_weight * rot_err], axis=0)
            A = J @ J.T + float(c.damping) * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -float(c.step_clip), float(c.step_clip))
            q = np.asarray(self.sim.data.qpos[self.qpos_idx], dtype=np.float64).copy()
            self.sim.data.qpos[self.qpos_idx] = q + dq[: len(self.qpos_idx)]
            if hasattr(self.sim.data, "qvel"):
                self.sim.data.qvel[self.qvel_idx] = 0.0
        self.sim.forward()
        return {"ik_pos_err": last_pos_err, "ik_rot_err": last_rot_err, "ik_converged": float(converged)}


# -----------------------------------------------------------------------------
# Gripper and object attach logic
# -----------------------------------------------------------------------------


_GRIPPER_JOINT_CACHE: dict[int, list[dict[str, Any]]] = {}


def _joint_type_name(model: Any, jid: int) -> str:
    try:
        jtype = int(model.jnt_type[jid])
    except Exception:
        return "unknown"
    return {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, str(jtype))


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


def gripper_joint_indices(env: Any) -> list[int]:
    return [int(rec["qpos_adr"]) for rec in gripper_joint_records(env)]


def describe_gripper_joints(env: Any) -> list[dict[str, Any]]:
    records = gripper_joint_records(env)
    sim = get_sim(env)
    out = []
    for rec in records:
        item = dict(rec)
        qpos_adr = int(rec["qpos_adr"])
        qvel_adr = int(rec["qvel_adr"])
        item["qpos"] = float(sim.data.qpos[qpos_adr]) if qpos_adr < len(sim.data.qpos) else None
        item["qvel"] = float(sim.data.qvel[qvel_adr]) if hasattr(sim.data, "qvel") and qvel_adr < len(sim.data.qvel) else None
        out.append(item)
    return out


def resolve_gripper_width_for_execution(
    predicted_width: float,
    *,
    mode: str,
    threshold: float,
    open_width: float,
    close_width: float,
    qpos_max_width: float,
) -> tuple[float, str]:
    """Map the model's continuous gripper-width prediction to an executed width.

    The model output is a physical width in the same units used by the dataset.
    For fast binary testing, threshold mode treats small predicted widths as
    "close" and large predicted widths as "open".  Continuous mode writes the
    model width directly after clipping.
    """
    max_width = max(float(qpos_max_width), 0.0)
    pred = float(np.clip(float(predicted_width), 0.0, max_width))
    mode = str(mode).lower().strip()
    if mode == "continuous":
        return pred, "continuous"
    if mode != "threshold":
        raise ValueError(f"Unsupported gripper_control_mode={mode!r}; expected 'threshold' or 'continuous'.")

    threshold = float(np.clip(float(threshold), 0.0, max_width))
    close_width = float(np.clip(float(close_width), 0.0, max_width))
    open_width = float(np.clip(float(open_width), 0.0, max_width))
    if pred < threshold:
        return close_width, "close"
    return open_width, "open"


def _gripper_joint_sign(rec: dict[str, Any], sim: Any) -> float:
    """Return the signed qpos direction for a Panda-style two-finger gripper.

    In the user's LIBERO model the debug output shows:
      gripper0_finger_joint1 qpos ~= +0.0208
      gripper0_finger_joint2 qpos ~= -0.0208
    Therefore a physical opening width W should be written as [+W/2, -W/2].
    We infer the sign first from the current qpos, then from the joint name.
    """
    qpos_adr = int(rec["qpos_adr"])
    try:
        current = float(sim.data.qpos[qpos_adr])
        if abs(current) > 1e-6:
            return 1.0 if current > 0 else -1.0
    except Exception:
        pass
    name = str(rec.get("name", "")).lower()
    if re.search(r"finger_joint2|right", name):
        return -1.0
    return 1.0


def set_gripper_width(env: Any, width: float, qpos_max_width: float) -> float:
    """Set a symmetric parallel-jaw physical opening width by direct qpos write.

    For this LIBERO / Panda model the two slide joints are mirrored in qpos:
    joint1 is positive when open and joint2 is negative when open.  A requested
    physical width W is therefore written as [+W/2, -W/2].  This makes both
    fingers move toward / away from the center instead of one finger chasing the
    other.
    """
    sim = get_sim(env)
    width = float(np.clip(width, 0.0, float(qpos_max_width)))
    records = gripper_joint_records(env)
    if records:
        if len(records) >= 2:
            per_finger = width / 2.0
            for rec in records[:2]:
                sign = _gripper_joint_sign(rec, sim)
                sim.data.qpos[int(rec["qpos_adr"])] = sign * per_finger
                if hasattr(sim.data, "qvel"):
                    qvel_adr = int(rec["qvel_adr"])
                    if qvel_adr < len(sim.data.qvel):
                        sim.data.qvel[qvel_adr] = 0.0
            # Any extra fallback joints should be held still; writing them can
            # distort passive knuckles or visual helper joints.
            for rec in records[2:]:
                if hasattr(sim.data, "qvel"):
                    qvel_adr = int(rec["qvel_adr"])
                    if qvel_adr < len(sim.data.qvel):
                        sim.data.qvel[qvel_adr] = 0.0
        else:
            # Single-joint fallback: keep old behavior.
            rec = records[0]
            sim.data.qpos[int(rec["qpos_adr"])] = width
            if hasattr(sim.data, "qvel"):
                qvel_adr = int(rec["qvel_adr"])
                if qvel_adr < len(sim.data.qvel):
                    sim.data.qvel[qvel_adr] = 0.0
        sim.forward()
    return width


def measured_gripper_width(env: Any, qpos_max_width: float) -> float:
    sim = get_sim(env)
    records = gripper_joint_records(env)
    if not records:
        return 0.0
    values = np.asarray([float(sim.data.qpos[int(rec["qpos_adr"])]) for rec in records[:2]], dtype=np.float32)
    if len(values) >= 2:
        return float(np.clip(np.sum(np.abs(values)), 0.0, qpos_max_width))
    return float(np.clip(abs(values[0]), 0.0, qpos_max_width))

def free_joint_body_map(sim: Any) -> dict[int, int]:
    out: dict[int, int] = {}
    model = sim.model
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) == 0:  # free joint
            out[int(model.jnt_bodyid[jid])] = jid
    return out


def candidate_object_bodies(env: Any, regex: str | None = None) -> list[int]:
    sim = get_sim(env)
    model = sim.model
    body_names = model_names(model, "body")
    free_bodies = free_joint_body_map(sim)
    blocked = re.compile(r"world|robot|panda|gripper|finger|mount|base|table|floor|arena|camera|vis", re.I)
    user_re = re.compile(regex, re.I) if regex else None
    out: list[int] = []
    for bid in sorted(free_bodies):
        name = body_names[bid] if bid < len(body_names) else str(bid)
        if blocked.search(name):
            continue
        if user_re is not None and not user_re.search(name):
            continue
        out.append(bid)
    return out


def set_free_body_pose(sim: Any, body_id: int, H_world: np.ndarray) -> bool:
    free_map = free_joint_body_map(sim)
    if body_id not in free_map:
        return False
    jid = free_map[body_id]
    adr = int(sim.model.jnt_qposadr[jid])
    sim.data.qpos[adr : adr + 3] = H_world[:3, 3]
    sim.data.qpos[adr + 3 : adr + 7] = quat_wxyz_from_mat(H_world[:3, :3])
    # Directly teleporting an attached object must also kill its free-joint
    # velocity; otherwise the next MuJoCo step can immediately let it drift or
    # tunnel away before the following AttachManager update.
    try:
        dof_adr = int(sim.model.jnt_dofadr[jid])
        sim.data.qvel[dof_adr : dof_adr + 6] = 0.0
    except Exception:
        pass
    sim.forward()
    return True


def _body_to_free_root_map(sim: Any) -> dict[int, int]:
    """Map every descendant body id to its free-joint object root body."""
    model = sim.model
    free_roots = set(free_joint_body_map(sim).keys())
    out: dict[int, int] = {}
    try:
        parents = np.asarray(model.body_parentid, dtype=np.int64).reshape(-1)
        nbody = int(model.nbody)
    except Exception:
        return out
    for bid in range(nbody):
        cur = int(bid)
        for _ in range(nbody + 1):
            if cur in free_roots:
                out[int(bid)] = int(cur)
                break
            if cur < 0 or cur >= len(parents):
                break
            parent = int(parents[cur])
            if parent == cur:
                break
            cur = parent
    return out


def nearest_object_geom_to_eef(env: Any, eef_pos: np.ndarray, object_regex: str | None = None) -> dict[str, Any] | None:
    """Find the nearest free-joint object geometry to the EEF position.

    The old attach test used the free body origin only.  Many LIBERO objects
    have their root/body origin at the object center or at a helper frame that is
    not close to the grasp contact point.  In instant IK + direct qpos mode the
    fingers can visually pass through a geom while the root-origin distance is
    still larger than --attach-distance.  This helper attaches based on actual
    MuJoCo geom centers / approximate surfaces instead.
    """
    sim = get_sim(env)
    model = sim.model
    free_roots = set(candidate_object_bodies(env, object_regex))
    if not free_roots:
        return None

    body_to_root = _body_to_free_root_map(sim)
    body_names = model_names(model, "body")
    geom_names = model_names(model, "geom")
    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    best: dict[str, Any] | None = None
    try:
        ngeom = int(model.ngeom)
    except Exception:
        ngeom = len(getattr(model, "geom_bodyid", []))

    for gid in range(ngeom):
        try:
            geom_body = int(model.geom_bodyid[gid])
        except Exception:
            continue
        root_body = body_to_root.get(geom_body)
        if root_body is None or int(root_body) not in free_roots:
            continue
        try:
            center = np.asarray(sim.data.geom_xpos[gid], dtype=np.float64).reshape(3)
        except Exception:
            continue
        try:
            size = np.asarray(model.geom_size[gid], dtype=np.float64).reshape(-1)
            approx_radius = float(np.nanmax(np.abs(size))) if size.size else 0.0
        except Exception:
            approx_radius = 0.0
        center_dist = float(np.linalg.norm(center - eef_pos))
        surface_dist = float(center_dist - max(0.0, approx_radius))
        item = {
            "body_id": int(root_body),
            "body_name": body_names[int(root_body)] if int(root_body) < len(body_names) else str(root_body),
            "geom_id": int(gid),
            "geom_name": geom_names[int(gid)] if int(gid) < len(geom_names) else str(gid),
            "geom_body_id": int(geom_body),
            "center_distance": center_dist,
            "surface_distance": surface_dist,
            "approx_radius": approx_radius,
        }
        metric = surface_dist
        if best is None or metric < float(best["surface_distance"]):
            best = item

    if best is not None:
        return best

    # Fallback to body origin if the model did not expose useful geoms.
    distances = []
    for bid in free_roots:
        obj_H = body_pose_hmat(sim, int(bid))
        distances.append((float(np.linalg.norm(obj_H[:3, 3] - eef_pos)), int(bid), obj_H))
    if not distances:
        return None
    distances.sort(key=lambda x: x[0])
    dist, bid, _ = distances[0]
    return {
        "body_id": int(bid),
        "body_name": body_names[int(bid)] if int(bid) < len(body_names) else str(bid),
        "geom_id": None,
        "geom_name": None,
        "geom_body_id": int(bid),
        "center_distance": float(dist),
        "surface_distance": float(dist),
        "approx_radius": 0.0,
    }


class AttachManager:
    def __init__(
        self,
        env: Any,
        *,
        attach_distance: float = 0.12,
        close_width: float = 0.018,
        release_width: float = 0.035,
        object_regex: str | None = None,
        geom_distance: float = 0.08,
        debug: bool = False,
    ):
        self.env = env
        self.sim = get_sim(env)
        self.attach_distance = float(attach_distance)
        self.close_width = float(close_width)
        self.release_width = float(release_width)
        self.object_regex = object_regex
        self.geom_distance = float(geom_distance)
        self.debug = bool(debug)
        self.attached_body_id: int | None = None
        self.T_obj_in_eef: np.ndarray | None = None
        self.events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.attached_body_id = None
        self.T_obj_in_eef = None
        self.events.clear()

    def update(self, eef_H_world: np.ndarray, gripper_width: float, step: int) -> None:
        if self.attached_body_id is not None:
            if gripper_width >= self.release_width:
                self.events.append({"step": int(step), "event": "detach", "body_id": int(self.attached_body_id)})
                self.attached_body_id = None
                self.T_obj_in_eef = None
                return
            assert self.T_obj_in_eef is not None
            set_free_body_pose(self.sim, self.attached_body_id, eef_H_world @ self.T_obj_in_eef)
            return

        if gripper_width > self.close_width:
            return
        eef_pos = np.asarray(eef_H_world[:3, 3], dtype=np.float64)
        nearest = nearest_object_geom_to_eef(self.env, eef_pos, self.object_regex)
        if nearest is None:
            return
        surface_dist = float(nearest.get("surface_distance", float("inf")))
        center_dist = float(nearest.get("center_distance", float("inf")))
        should_attach = surface_dist <= self.geom_distance or center_dist <= self.attach_distance
        if self.debug:
            print(
                "[debug] attach_candidate "
                f"step={int(step)} body={nearest.get('body_name')} geom={nearest.get('geom_name')} "
                f"surface_dist={surface_dist:.4f} center_dist={center_dist:.4f} "
                f"limits=({self.geom_distance:.4f},{self.attach_distance:.4f}) attach={should_attach}"
            )
        if should_attach:
            bid = int(nearest["body_id"])
            obj_H = body_pose_hmat(self.sim, bid)
            self.attached_body_id = int(bid)
            self.T_obj_in_eef = hmat_inv(eef_H_world) @ obj_H
            # Snap immediately and zero free-joint velocity so the object cannot
            # continue tunneling through the fingers on the next physics tick.
            set_free_body_pose(self.sim, self.attached_body_id, eef_H_world @ self.T_obj_in_eef)
            self.events.append({
                "step": int(step),
                "event": "attach",
                "body_id": int(bid),
                "body_name": nearest.get("body_name"),
                "geom_id": nearest.get("geom_id"),
                "geom_name": nearest.get("geom_name"),
                "surface_distance": surface_dist,
                "center_distance": center_dist,
                "approx_radius": float(nearest.get("approx_radius", 0.0)),
            })


# -----------------------------------------------------------------------------
# Point-cloud observation and visualization
# -----------------------------------------------------------------------------


def image_from_obs(raw_obs: dict[str, Any], camera_names: Iterable[str]) -> np.ndarray | None:
    for camera in camera_names:
        key = str(camera)
        if not key.endswith("_image"):
            key = f"{key}_image"
        if key in raw_obs:
            img = np.asarray(raw_obs[key])
            if img.ndim == 3 and img.shape[-1] == 3:
                return np.clip(img, 0, 255).astype(np.uint8)
    return None


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
    from libero_pointcloud_utils import (
        add_world_gripper_cloud_to_point_cloud,
        gripper_width_percent_from_scalar,
        observation_to_point_clouds,
        pointcloud_camera_names_from_config,
    )

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


def write_ply(path: Path, xyzrgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    xyz = xyzrgb[:, :3]
    rgb = np.clip(xyzrgb[:, 3:6], 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _rgb_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    if colors.size == 0:
        return colors.reshape(-1, 3).astype(np.uint8)
    if float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors[:, :3], 0.0, 255.0).astype(np.uint8)


def _time_gradient(num_steps: int) -> np.ndarray:
    if num_steps <= 1:
        return np.asarray([[25, 190, 80]], dtype=np.uint8)
    t = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float32)[:, None]
    start = np.asarray([20, 120, 255], dtype=np.float32)
    mid = np.asarray([30, 220, 80], dtype=np.float32)
    end = np.asarray([255, 45, 30], dtype=np.float32)
    first = (1.0 - 2.0 * t) * start + (2.0 * t) * mid
    second = (2.0 - 2.0 * t) * mid + (2.0 * t - 1.0) * end
    return np.where(t <= 0.5, first, second).clip(0, 255).astype(np.uint8)


def transform_point_cloud_hmat(point_cloud: np.ndarray, H: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    out = pc.copy()
    xyz = pc[:, :3]
    H = np.asarray(H, dtype=np.float32)
    out[:, :3] = xyz @ H[:3, :3].T + H[:3, 3]
    return out


def matrices_to_pose9(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    if mats.ndim == 2:
        mats = mats[None, ...]
    return np.concatenate([mats[..., :3, 3], mats[..., :3, 0], mats[..., :3, 1]], axis=-1).astype(np.float32)


def _append_action_frame_geometry(
    vertices: list[list[float]],
    colors: list[list[int]],
    edges: list[list[int]],
    edge_colors: list[list[int]],
    pose9: np.ndarray,
    *,
    frame_color: np.ndarray,
    frame_scale: float,
    bead_radius: float,
) -> int:
    """Append origin bead + xyz axes for one pose9 frame and return origin vertex index."""
    H = pose9_to_hmat(np.asarray(pose9, dtype=np.float32)[:9])
    origin = H[:3, 3].astype(float)
    origin_idx = len(vertices)
    vertices.append(origin.tolist())
    colors.append([int(frame_color[0]), int(frame_color[1]), int(frame_color[2])])

    # Small 6-point cross bead around the target.  This makes target positions
    # visible even in PLY viewers that ignore edge elements.
    bead_color = [int(frame_color[0]), int(frame_color[1]), int(frame_color[2])]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            p = origin.copy()
            p[axis] += sign * float(bead_radius)
            vertices.append(p.tolist())
            colors.append(bead_color)
            edges.append([origin_idx, len(vertices) - 1])
            edge_colors.append(bead_color)

    axis_colors = ([255, 0, 0], [0, 255, 0], [0, 80, 255])
    for axis_idx, axis_color in enumerate(axis_colors):
        end = origin + float(frame_scale) * H[:3, axis_idx]
        vertices.append(end.tolist())
        colors.append(axis_color)
        edges.append([origin_idx, len(vertices) - 1])
        edge_colors.append(axis_color)
    return origin_idx


def write_pointcloud_action_ply(
    path: Path,
    point_cloud: np.ndarray,
    action_chunk: np.ndarray,
    *,
    max_action_frames: int = 16,
    frame_scale: float = 0.04,
    bead_radius: float = 0.006,
    point_stride: int = 1,
) -> None:
    """Write the exact model input point cloud plus predicted pose9 action chunk.

    This mirrors the ood_case_inference debugging idea: the PLY lives in the
    same coordinate frame as the action being visualized.  For UMI action debug,
    the origin frame is the current EEF frame, the point cloud is pc_eff, and the
    action poses are the raw model UMI chunk.  For world/controller debug, pass a
    world-frame point cloud and a world-frame pose9 trajectory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pc = np.asarray(point_cloud, dtype=np.float32)
    action = ensure_2d_action_chunk(action_chunk)
    if pc.ndim != 2 or pc.shape[-1] < 3:
        raise ValueError(f"Expected point cloud (N, >=3), got {pc.shape}")
    if int(point_stride) > 1:
        pc = pc[:: int(point_stride)]

    vertices: list[list[float]] = pc[:, :3].astype(float).tolist()
    if pc.shape[-1] >= 6:
        colors = _rgb_uint8(pc[:, 3:6]).astype(int).tolist()
    else:
        colors = np.full((len(pc), 3), 128, dtype=np.uint8).astype(int).tolist()
    edges: list[list[int]] = []
    edge_colors: list[list[int]] = []

    # Draw the current UMI/reference frame at origin.  In world/controller PLYs
    # this is still useful as a visual reference; it is not used for execution.
    origin_pose9 = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    _append_action_frame_geometry(
        vertices,
        colors,
        edges,
        edge_colors,
        origin_pose9,
        frame_color=np.asarray([255, 255, 255], dtype=np.uint8),
        frame_scale=float(frame_scale) * 1.2,
        bead_radius=float(bead_radius) * 0.8,
    )

    n = int(action.shape[0])
    max_frames = max(1, int(max_action_frames))
    frame_stride = max(1, int(math.ceil(n / max_frames)))
    frame_indices = list(range(0, n, frame_stride))
    if n - 1 not in frame_indices:
        frame_indices.append(n - 1)
    time_colors = _time_gradient(n)

    origin_indices: list[int] = []
    prev_origin_idx: int | None = None
    for idx in range(n):
        pose = action[idx]
        color = time_colors[idx]
        # Always draw trajectory beads/line; draw axes sparsely.
        H = pose9_to_hmat(pose[:9])
        pos = H[:3, 3].astype(float)
        origin_idx = len(vertices)
        vertices.append(pos.tolist())
        colors.append([int(color[0]), int(color[1]), int(color[2])])
        origin_indices.append(origin_idx)
        if prev_origin_idx is not None:
            edges.append([prev_origin_idx, origin_idx])
            edge_colors.append([int(color[0]), int(color[1]), int(color[2])])
        prev_origin_idx = origin_idx
        if idx in frame_indices:
            # Reuse the origin vertex just added for axis edges.
            axis_colors = ([255, 0, 0], [0, 255, 0], [0, 80, 255])
            for axis_idx, axis_color in enumerate(axis_colors):
                end = pos + float(frame_scale) * H[:3, axis_idx]
                vertices.append(end.tolist())
                colors.append(axis_color)
                edges.append([origin_idx, len(vertices) - 1])
                edge_colors.append(axis_color)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(vertices, colors):
            f.write(f"{float(p[0]):.7f} {float(p[1]):.7f} {float(p[2]):.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for e, c in zip(edges, edge_colors):
            f.write(f"{int(e[0])} {int(e[1])} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_model_inference_debug_artifacts(
    *,
    output_dir: Path,
    step: int,
    model_call_index: int,
    point_cloud_umi: np.ndarray,
    action_chunk_umi: np.ndarray,
    task_language: str,
    current_world_pose9: np.ndarray,
    obs_gripper: float,
    planned_model_world: np.ndarray | None = None,
    planned_controller_world: np.ndarray | None = None,
    point_stride: int = 1,
    max_action_frames: int = 16,
) -> dict[str, str]:
    """Save raw model input/output and optional converted targets for debugging."""
    debug_dir = output_dir / "model_inference_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"call_{int(model_call_index):04d}_step_{int(step):04d}"
    action_chunk_umi = ensure_2d_action_chunk(action_chunk_umi)
    current_world = pose9_to_hmat(np.asarray(current_world_pose9, dtype=np.float32)[:9])
    point_cloud_world = transform_point_cloud_hmat(point_cloud_umi, current_world)

    paths: dict[str, str] = {}
    umi_ply = debug_dir / f"{stem}_umi_input_pc_pred_action.ply"
    write_pointcloud_action_ply(
        umi_ply,
        point_cloud_umi,
        action_chunk_umi,
        point_stride=int(point_stride),
        max_action_frames=int(max_action_frames),
    )
    paths["umi_ply"] = str(umi_ply)

    if planned_model_world is not None and len(planned_model_world) > 0:
        model_world_action = np.concatenate(
            [matrices_to_pose9(planned_model_world), action_chunk_umi[: len(planned_model_world), 9:10]],
            axis=-1,
        )
        model_world_ply = debug_dir / f"{stem}_model_world_pc_action.ply"
        write_pointcloud_action_ply(
            model_world_ply,
            point_cloud_world,
            model_world_action,
            point_stride=int(point_stride),
            max_action_frames=int(max_action_frames),
        )
        paths["model_world_ply"] = str(model_world_ply)

    if planned_controller_world is not None and len(planned_controller_world) > 0:
        controller_action = np.concatenate(
            [matrices_to_pose9(planned_controller_world), action_chunk_umi[: len(planned_controller_world), 9:10]],
            axis=-1,
        )
        controller_ply = debug_dir / f"{stem}_controller_targets_on_world_pc.ply"
        write_pointcloud_action_ply(
            controller_ply,
            point_cloud_world,
            controller_action,
            point_stride=int(point_stride),
            max_action_frames=int(max_action_frames),
        )
        paths["controller_world_ply"] = str(controller_ply)

    npz_path = debug_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        point_cloud_umi=np.asarray(point_cloud_umi, dtype=np.float32),
        point_cloud_world=np.asarray(point_cloud_world, dtype=np.float32),
        action_chunk_umi=np.asarray(action_chunk_umi, dtype=np.float32),
        current_world_pose9=np.asarray(current_world_pose9, dtype=np.float32),
        current_world_hmat=np.asarray(current_world, dtype=np.float32),
        obs_gripper=np.asarray(obs_gripper, dtype=np.float32),
        planned_model_world=np.asarray(planned_model_world, dtype=np.float32) if planned_model_world is not None else np.empty((0, 4, 4), dtype=np.float32),
        planned_controller_world=np.asarray(planned_controller_world, dtype=np.float32) if planned_controller_world is not None else np.empty((0, 4, 4), dtype=np.float32),
    )
    paths["npz"] = str(npz_path)

    meta = {
        "step": int(step),
        "model_call_index": int(model_call_index),
        "task_language": str(task_language),
        "point_cloud_umi_shape": list(np.asarray(point_cloud_umi).shape),
        "action_chunk_shape": list(np.asarray(action_chunk_umi).shape),
        "action_start_xyz": np.asarray(action_chunk_umi[0, :3], dtype=float).round(6).tolist() if len(action_chunk_umi) else [],
        "action_end_xyz": np.asarray(action_chunk_umi[-1, :3], dtype=float).round(6).tolist() if len(action_chunk_umi) else [],
        "gripper_start": float(action_chunk_umi[0, 9]) if len(action_chunk_umi) else None,
        "gripper_end": float(action_chunk_umi[-1, 9]) if len(action_chunk_umi) else None,
        "current_world_pose9": np.asarray(current_world_pose9, dtype=float).round(6).tolist(),
        "obs_gripper": float(obs_gripper),
        "files": paths,
        "legend": {
            "point_cloud": "model input point cloud; UMI PLY is in current EEF frame, world PLYs use current EEF pose",
            "trajectory_color": "blue/green=start, red=end",
            "axes": "red=x, green=y, blue=z; white frame is the reference frame",
        },
    }
    json_path = debug_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    paths["json"] = str(json_path)
    return paths


class Open3DNonBlockingViewer:
    def __init__(self, every: int = 1):
        self.every = max(1, int(every))
        self.vis = None
        self.pcd = None
        self.created = False

    def update(self, step: int, point_cloud: np.ndarray, action_chunk: np.ndarray | None = None) -> None:
        if step % self.every != 0:
            return
        try:
            import open3d as o3d
        except Exception:
            return
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="LIBERO fast eval", width=960, height=720, visible=True)
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            self.created = True
        assert self.pcd is not None and self.vis is not None
        pc = np.asarray(point_cloud, dtype=np.float32)
        self.pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        self.pcd.colors = o3d.utility.Vector3dVector(np.clip(pc[:, 3:6] / 255.0, 0, 1))
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()
        self.vis = None
        self.pcd = None


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
            from libero_pointcloud_utils import attach_mujoco_3d_viewer

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


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------


def planned_world_poses_from_umi_chunk(current_model_world_pose9: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Convert a UMI action chunk into model-world absolute EEF targets.

    This intentionally matches libero_pointcloud_eval.py:
      current_world = pose9_to_homo_np(eef_pose[:9])
      planned_relative = pose9_to_homo_np(chunk[:, :9])
      target_world = current_world @ planned_relative
    """
    from libero_pointcloud_utils import pose9_to_homo_np

    current_world = pose9_to_homo_np(np.asarray(current_model_world_pose9, dtype=np.float32)[..., :9])
    planned_relative = pose9_to_homo_np(np.asarray(chunk, dtype=np.float32)[:, :9])
    return np.asarray(current_world @ planned_relative, dtype=np.float32)

def model_world_pose_to_controller_world(
    target_model_world: np.ndarray,
    current_model_world: np.ndarray,
    current_controller_world: np.ndarray,
) -> np.ndarray:
    """Map model EEF target frame to the actual controller/IK site frame.

    Same transform as the reference evaluator:
        model_to_controller = inv(current_model_world) @ current_controller_world
        controller_target = target_model_world @ model_to_controller
    """
    from libero_pointcloud_utils import fast_inverse_homogeneous

    model_to_controller = fast_inverse_homogeneous(np.asarray(current_model_world, dtype=np.float32)) @ np.asarray(
        current_controller_world, dtype=np.float32
    )
    return np.asarray(target_model_world @ model_to_controller, dtype=np.float32)

def action_row_to_world_target(row: np.ndarray, current_world_pose9: np.ndarray, action_pose_frame: str) -> np.ndarray:
    # Kept for explicit fallback / debugging. Normal fast eval now converts the
    # whole UMI chunk once with planned_world_poses_from_umi_chunk(), then maps it
    # to the actual controller/IK site frame.
    if action_pose_frame == "world":
        return pose9_to_hmat(row[:9])
    if action_pose_frame == "current_eff":
        return pose9_to_hmat(current_world_pose9) @ pose9_to_hmat(row[:9])
    raise ValueError(f"Unknown action_pose_frame={action_pose_frame!r}")


@dataclass
class EpisodeResult:
    suite: str
    task_id: int
    episode: int
    task_language: str
    success: bool
    reward: float
    steps: int
    elapsed_s: float
    output_dir: str


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _json_atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _task_metric_key(suite: str, task_id: int) -> str:
    return f"{suite}_task_{int(task_id):03d}"


def build_live_metrics(
    all_results: list[EpisodeResult],
    specs: list[TaskSpec],
    *,
    episodes_per_task: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Aggregate live overall / suite / task metrics after every episode."""
    expected_per_task = max(0, int(episodes_per_task))
    total_expected = len(specs) * expected_per_task
    result_dicts = [asdict(r) for r in all_results]

    task_metrics: dict[str, dict[str, Any]] = {}
    for spec in specs:
        key = _task_metric_key(spec.suite, spec.task_id)
        task_metrics[key] = {
            "suite": str(spec.suite),
            "task_id": int(spec.task_id),
            "task_language": "",
            "episodes_completed": 0,
            "episodes_expected": expected_per_task,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_steps": 0.0,
            "mean_elapsed_s": 0.0,
            "last_episode": None,
            "last_success": None,
            "last_reward": None,
            "last_steps": None,
        }

    for r in all_results:
        key = _task_metric_key(r.suite, r.task_id)
        if key not in task_metrics:
            task_metrics[key] = {
                "suite": str(r.suite),
                "task_id": int(r.task_id),
                "task_language": str(r.task_language),
                "episodes_completed": 0,
                "episodes_expected": expected_per_task,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_steps": 0.0,
                "mean_elapsed_s": 0.0,
                "last_episode": None,
                "last_success": None,
                "last_reward": None,
                "last_steps": None,
            }

    grouped: dict[str, list[EpisodeResult]] = {key: [] for key in task_metrics}
    for r in all_results:
        grouped.setdefault(_task_metric_key(r.suite, r.task_id), []).append(r)

    for key, rows in grouped.items():
        metric = task_metrics[key]
        rows_sorted = sorted(rows, key=lambda x: int(x.episode))
        successes = [bool(r.success) for r in rows_sorted]
        metric["episodes_completed"] = int(len(rows_sorted))
        metric["success_count"] = int(sum(successes))
        metric["failure_count"] = int(len(rows_sorted) - sum(successes))
        metric["success_rate"] = float(np.mean(successes)) if successes else 0.0
        metric["mean_reward"] = _mean_or_zero([float(r.reward) for r in rows_sorted])
        metric["mean_steps"] = _mean_or_zero([float(r.steps) for r in rows_sorted])
        metric["mean_elapsed_s"] = _mean_or_zero([float(r.elapsed_s) for r in rows_sorted])
        if rows_sorted:
            last = rows_sorted[-1]
            metric["task_language"] = str(last.task_language)
            metric["last_episode"] = int(last.episode)
            metric["last_success"] = bool(last.success)
            metric["last_reward"] = float(last.reward)
            metric["last_steps"] = int(last.steps)

    suite_metrics: dict[str, dict[str, Any]] = {}
    for spec in specs:
        suite_metrics.setdefault(
            str(spec.suite),
            {
                "suite": str(spec.suite),
                "tasks_total": 0,
                "tasks_completed": 0,
                "episodes_completed": 0,
                "episodes_expected": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_steps": 0.0,
                "mean_elapsed_s": 0.0,
            },
        )
        suite_metrics[str(spec.suite)]["tasks_total"] += 1
        suite_metrics[str(spec.suite)]["episodes_expected"] += expected_per_task

    for suite_name, metric in suite_metrics.items():
        rows = [r for r in all_results if str(r.suite) == suite_name]
        successes = [bool(r.success) for r in rows]
        metric["episodes_completed"] = int(len(rows))
        metric["success_count"] = int(sum(successes))
        metric["failure_count"] = int(len(rows) - sum(successes))
        metric["success_rate"] = float(np.mean(successes)) if successes else 0.0
        metric["mean_reward"] = _mean_or_zero([float(r.reward) for r in rows])
        metric["mean_steps"] = _mean_or_zero([float(r.steps) for r in rows])
        metric["mean_elapsed_s"] = _mean_or_zero([float(r.elapsed_s) for r in rows])
        metric["tasks_completed"] = int(sum(
            1
            for tm in task_metrics.values()
            if tm["suite"] == suite_name and int(tm["episodes_completed"]) >= int(tm["episodes_expected"])
        ))

    successes = [bool(r.success) for r in all_results]
    overall = {
        "num_tasks": int(len(specs)),
        "episodes_completed": int(len(all_results)),
        "episodes_expected": int(total_expected),
        "success_count": int(sum(successes)),
        "failure_count": int(len(all_results) - sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": _mean_or_zero([float(r.reward) for r in all_results]),
        "mean_steps": _mean_or_zero([float(r.steps) for r in all_results]),
        "mean_elapsed_s": _mean_or_zero([float(r.elapsed_s) for r in all_results]),
    }
    progress = {
        "episodes_completed": overall["episodes_completed"],
        "episodes_expected": overall["episodes_expected"],
        "episode_progress": (float(overall["episodes_completed"]) / float(overall["episodes_expected"])) if overall["episodes_expected"] else 0.0,
        "success_rate": overall["success_rate"],
        "updated_at_unix_s": float(time.time()),
    }
    return {
        "overall": overall,
        "suite_metrics": suite_metrics,
        "task_metrics": task_metrics,
        "progress": progress,
        "results": result_dicts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
    }


def write_live_metrics(
    output_root: Path,
    all_results: list[EpisodeResult],
    specs: list[TaskSpec],
    args: argparse.Namespace,
    *,
    last_result: EpisodeResult | None = None,
    print_line: bool = True,
) -> dict[str, Any]:
    """Rewrite live metrics files after every completed / failed episode."""
    metrics = build_live_metrics(all_results, specs, episodes_per_task=int(args.episodes), args=args)
    overall = metrics["overall"]
    task_metrics = metrics["task_metrics"]
    suite_metrics = metrics["suite_metrics"]

    summary_payload = {
        "num_tasks": overall["num_tasks"],
        "num_episodes": overall["episodes_completed"],
        "episodes_expected": overall["episodes_expected"],
        "success_rate": overall["success_rate"],
        "success_count": overall["success_count"],
        "failure_count": overall["failure_count"],
        "mean_reward": overall["mean_reward"],
        "mean_steps": overall["mean_steps"],
        "mean_elapsed_s": overall["mean_elapsed_s"],
        "results": metrics["results"],
        "task_metrics": task_metrics,
        "suite_metrics": suite_metrics,
        "progress": metrics["progress"],
        "args": metrics["args"],
    }
    _json_atomic_write(output_root / "summary.json", summary_payload)
    _json_atomic_write(output_root / "progress.json", metrics["progress"])
    _json_atomic_write(output_root / "task_metrics.json", task_metrics)
    _json_atomic_write(output_root / "suite_metrics.json", suite_metrics)

    jsonl_tmp = output_root / "episode_results.jsonl.tmp"
    with open(jsonl_tmp, "w", encoding="utf-8") as f:
        for item in metrics["results"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    jsonl_tmp.replace(output_root / "episode_results.jsonl")

    episode_csv_tmp = output_root / "episode_results.csv.tmp"
    with open(episode_csv_tmp, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["suite", "task_id", "episode", "task_language", "success", "reward", "steps", "elapsed_s", "output_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics["results"]:
            writer.writerow({k: item.get(k, "") for k in fieldnames})
    episode_csv_tmp.replace(output_root / "episode_results.csv")

    task_csv_tmp = output_root / "task_metrics.csv.tmp"
    with open(task_csv_tmp, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "suite", "task_id", "episodes_completed", "episodes_expected", "success_count", "failure_count",
            "success_rate", "mean_reward", "mean_steps", "mean_elapsed_s", "last_episode", "last_success", "task_language",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(task_metrics):
            writer.writerow({k: task_metrics[key].get(k, "") for k in fieldnames})
    task_csv_tmp.replace(output_root / "task_metrics.csv")

    suite_csv_tmp = output_root / "suite_metrics.csv.tmp"
    with open(suite_csv_tmp, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "suite", "tasks_completed", "tasks_total", "episodes_completed", "episodes_expected", "success_count",
            "failure_count", "success_rate", "mean_reward", "mean_steps", "mean_elapsed_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(suite_metrics):
            writer.writerow({k: suite_metrics[key].get(k, "") for k in fieldnames})
    suite_csv_tmp.replace(output_root / "suite_metrics.csv")

    if print_line and bool(getattr(args, "print_metrics", True)) and last_result is not None:
        task_key = _task_metric_key(last_result.suite, last_result.task_id)
        task_metric = task_metrics.get(task_key, {})
        suite_metric = suite_metrics.get(str(last_result.suite), {})
        print(
            f"[metrics] overall={overall['success_count']}/{overall['episodes_completed']} "
            f"sr={overall['success_rate']:.3f} | "
            f"task={last_result.suite}:{int(last_result.task_id)} "
            f"{task_metric.get('success_count', 0)}/{task_metric.get('episodes_completed', 0)} "
            f"sr={float(task_metric.get('success_rate', 0.0)):.3f} | "
            f"suite={last_result.suite} "
            f"{suite_metric.get('success_count', 0)}/{suite_metric.get('episodes_completed', 0)} "
            f"sr={float(suite_metric.get('success_rate', 0.0)):.3f}"
        )
    return summary_payload


def run_one_episode(
    *,
    env: Any,
    task_language: str,
    bundle: PolicyBundle,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
) -> EpisodeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    raw_obs = env.reset()
    if raw_obs is None:
        raw_obs = get_raw_obs(env, force_update=True)
    # Let object free joints settle before the first model call.  This is not
    # policy control and does not use env.step(); it only advances MuJoCo physics
    # from the freshly reset scene, with the robot optionally held fixed.
    raw_obs = settle_scene_after_reset(
        env,
        steps=int(args.settle_steps),
        keep_robot_fixed=bool(args.settle_keep_robot_fixed),
        render=bool(args.settle_render),
        cfg=cfg,
        debug=bool(args.debug_settle),
    )
    reset_policy_bundle(bundle)

    executor = InstantIKExecutor(
        env,
        IKConfig(
            iters=int(args.ik_iters),
            damping=float(args.ik_damping),
            pos_tol=float(args.ik_pos_tol),
            rot_tol=float(args.ik_rot_tol),
            step_clip=float(args.ik_step_clip),
            rot_weight=float(args.ik_rot_weight),
        ),
    )
    if bool(getattr(args, "debug_gripper_joints", False)):
        print("[debug] selected_gripper_joints=", json.dumps(describe_gripper_joints(env), ensure_ascii=False))
    attach = AttachManager(
        env,
        attach_distance=float(args.attach_distance),
        close_width=float(args.attach_close_width),
        release_width=float(args.attach_release_width),
        object_regex=args.attach_object_regex,
        geom_distance=float(args.attach_geom_distance),
        debug=bool(args.debug_attach),
    ) if args.enable_attach else None

    viewer = Open3DNonBlockingViewer(args.vis3d_every) if args.render_mode == "open3d" else None
    frames: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    executed_rows: list[np.ndarray] = []
    gripper_predicted_widths: list[float] = []
    gripper_targets: list[float] = []
    gripper_actuals: list[float] = []
    gripper_modes: list[str] = []
    ik_logs: list[dict[str, float]] = []
    model_world_targets: list[np.ndarray] = []
    controller_world_targets: list[np.ndarray] = []
    plan_model_world_frames: list[np.ndarray] = []
    plan_controller_world_frames: list[np.ndarray] = []
    model_call_index = 0
    saved_inference_debug_count = 0

    start_s = time.perf_counter()
    step = 0
    success = False
    last_reward = 0.0
    pbar = tqdm(total=int(args.max_steps), desc="episode", leave=False, disable=bool(args.no_progress))
    try:
        while step < int(args.max_steps):
            pc_eff, current_world_pose9, obs_gripper = build_point_cloud_observation(env, raw_obs, cfg, seed + step)
            action_chunk = predict_action_chunk(bundle, pc_eff, obs_gripper, task_language, current_world_pose9)
            pred_chunks.append(action_chunk)
            if viewer is not None:
                viewer.update(step, pc_eff, action_chunk)
            if args.save_ply and step % max(1, int(args.vis3d_every)) == 0:
                write_ply(output_dir / "ply" / f"step_{step:04d}_cloud_eff.ply", pc_eff)

            start_index = int(np.clip(args.action_index, 0, len(action_chunk) - 1))
            if args.replan_every_step:
                rows = action_chunk[start_index : start_index + 1]
            else:
                rows = action_chunk[start_index : start_index + int(args.exec_action_steps)]
            if len(rows) == 0:
                rows = action_chunk[-1:]

            # Correct UMI execution path:
            #   1) observed world EEF pose defines the replanning frame EEF0;
            #   2) model predicts an absolute UMI trajectory in EEF0;
            #   3) convert the full planned chunk once to model-world targets;
            #   4) map model-world target to the actual IK/controller site frame.
            if args.action_pose_frame == "world":
                planned_model_world = np.asarray([pose9_to_hmat(row[:9]) for row in rows], dtype=np.float32)
            else:
                planned_model_world = planned_world_poses_from_umi_chunk(current_world_pose9, rows)
            current_model_world = pose9_to_hmat(current_world_pose9)
            # Use the exact execution frame from the robosuite controller, as in the ref evaluator.
            # The IK site was selected to match this frame, so executor.goto() still performs direct IK.
            current_controller_world = current_controller_eef_world(env)
            planned_controller_world = np.asarray(
                [
                    model_world_pose_to_controller_world(target_world, current_model_world, current_controller_world)
                    for target_world in planned_model_world
                ],
                dtype=np.float32,
            )

            should_save_debug = bool(args.save_inference_ply)
            should_save_debug = should_save_debug and (model_call_index % max(1, int(args.inference_ply_every_model_call)) == 0)
            should_save_debug = should_save_debug and saved_inference_debug_count < max(0, int(args.inference_ply_max_per_episode))
            if should_save_debug:
                try:
                    paths = save_model_inference_debug_artifacts(
                        output_dir=output_dir,
                        step=step,
                        model_call_index=model_call_index,
                        point_cloud_umi=pc_eff,
                        action_chunk_umi=action_chunk,
                        task_language=task_language,
                        current_world_pose9=current_world_pose9,
                        obs_gripper=obs_gripper,
                        planned_model_world=planned_model_world,
                        planned_controller_world=planned_controller_world,
                        point_stride=max(1, int(args.inference_ply_point_stride)),
                        max_action_frames=max(1, int(args.inference_ply_max_action_frames)),
                    )
                    saved_inference_debug_count += 1
                    if bool(args.print_inference_ply_paths):
                        print(f"[debug] saved model inference PLY/NPZ: {paths}")
                except Exception as exc:
                    print(f"[warn] failed to save model inference debug artifacts at step {step}: {exc!r}")
            model_call_index += 1

            model_world_targets.extend([np.asarray(x, dtype=np.float32).copy() for x in planned_model_world])
            controller_world_targets.extend([np.asarray(x, dtype=np.float32).copy() for x in planned_controller_world])
            plan_model_world_frames.append(np.asarray(current_model_world, dtype=np.float32).copy())
            plan_controller_world_frames.append(np.asarray(current_controller_world, dtype=np.float32).copy())

            for row, target_H_world in zip(rows, planned_controller_world):
                if step >= int(args.max_steps):
                    break
                ik_info = executor.goto(target_H_world)
                predicted_gripper = float(row[9])
                target_gripper, gripper_mode = resolve_gripper_width_for_execution(
                    predicted_gripper,
                    mode=str(args.gripper_control_mode),
                    threshold=float(args.gripper_threshold),
                    open_width=float(args.gripper_open_width),
                    close_width=float(args.gripper_close_width),
                    qpos_max_width=float(cfg.get("gripper_qpos_max_width", 0.08)),
                )
                target_gripper = set_gripper_width(env, target_gripper, float(cfg.get("gripper_qpos_max_width", 0.08)))
                eef_H = executor.current_pose()
                if attach is not None:
                    attach.update(eef_H, target_gripper, step)
                # Advance MuJoCo a small configurable number of ticks without using
                # robosuite's controller action queue. This keeps the test fast and
                # avoids an extra blocking controller loop.
                sim = get_sim(env)
                for _ in range(int(args.physics_steps_per_action)):
                    try:
                        if attach is not None:
                            attach.update(executor.current_pose(), target_gripper, step)
                        sim.step()
                        if attach is not None:
                            attach.update(executor.current_pose(), target_gripper, step)
                    except Exception:
                        break
                sim.forward()
                if attach is not None:
                    attach.update(executor.current_pose(), target_gripper, step)
                raw_obs = get_raw_obs(env, force_update=True)
                if bool(getattr(args, "debug_obs_refresh", False)):
                    try:
                        refreshed_pose9 = np.asarray(raw_obs.get("robot0_eef_pos", []), dtype=np.float32).reshape(-1)
                        print(f"[debug] obs_refresh step={step} robot0_eef_pos={refreshed_pose9.round(5).tolist()}")
                    except Exception:
                        pass

                executed_rows.append(np.asarray(row, dtype=np.float32).copy())
                gripper_predicted_widths.append(float(predicted_gripper))
                gripper_targets.append(float(target_gripper))
                gripper_actuals.append(float(measured_gripper_width(env, float(cfg.get("gripper_qpos_max_width", 0.08)))))
                gripper_modes.append(str(gripper_mode))
                try:
                    site_pos_err, site_rot_err = pose_frame_error(executor.current_pose(), target_H_world)
                    ctrl_pos_err, ctrl_rot_err = pose_frame_error(current_controller_eef_world(env), target_H_world)
                    ik_info.update({
                        "ik_site_name": executor.site_name,
                        "post_site_pos_err": float(site_pos_err),
                        "post_site_rot_err": float(site_rot_err),
                        "post_controller_pos_err": float(ctrl_pos_err),
                        "post_controller_rot_err": float(ctrl_rot_err),
                    })
                except Exception:
                    pass
                ik_logs.append(ik_info)
                img = image_from_obs(raw_obs, cfg.get("camera_names", ["agentview"]))
                if args.save_video and img is not None:
                    frames.append(img)
                render_mujoco_viewer(env, cfg, step)

                step += 1
                pbar.update(1)
                success = safe_check_success(env)
                last_reward = safe_reward(env)
                if success:
                    break
            if success:
                break
    finally:
        pbar.close()
        if viewer is not None:
            viewer.close()

    elapsed = time.perf_counter() - start_s
    np.save(output_dir / "pred_action_chunks.npy", np.asarray(pred_chunks, dtype=np.float32))
    np.save(output_dir / "executed_action_rows.npy", np.asarray(executed_rows, dtype=np.float32))
    np.save(output_dir / "gripper_predicted_widths.npy", np.asarray(gripper_predicted_widths, dtype=np.float32))
    np.save(output_dir / "gripper_targets.npy", np.asarray(gripper_targets, dtype=np.float32))
    np.save(output_dir / "gripper_actuals.npy", np.asarray(gripper_actuals, dtype=np.float32))
    with open(output_dir / "gripper_modes.json", "w", encoding="utf-8") as f:
        json.dump(gripper_modes, f, indent=2)
    np.save(output_dir / "model_world_targets.npy", np.asarray(model_world_targets, dtype=np.float32))
    np.save(output_dir / "controller_world_targets.npy", np.asarray(controller_world_targets, dtype=np.float32))
    np.save(output_dir / "plan_model_world_frames.npy", np.asarray(plan_model_world_frames, dtype=np.float32))
    np.save(output_dir / "plan_controller_world_frames.npy", np.asarray(plan_controller_world_frames, dtype=np.float32))
    with open(output_dir / "ik_logs.json", "w", encoding="utf-8") as f:
        json.dump(ik_logs, f, indent=2)
    if attach is not None:
        with open(output_dir / "attach_events.json", "w", encoding="utf-8") as f:
            json.dump(attach.events, f, indent=2)
    if args.save_video and frames:
        try:
            import imageio.v3 as iio

            iio.imwrite(output_dir / "rollout.mp4", np.asarray(frames, dtype=np.uint8), fps=int(cfg.get("fps", 30)))
        except Exception as exc:
            print(f"[warn] failed to save video: {exc}")

    return EpisodeResult(
        suite=str(args._current_suite),
        task_id=int(args._current_task_id),
        episode=int(args._current_episode),
        task_language=str(task_language),
        success=bool(success),
        reward=float(last_reward),
        steps=int(step),
        elapsed_s=float(elapsed),
        output_dir=str(output_dir),
    )


def close_env_safely(env: Any) -> None:
    """Close viewer/env without blocking the next episode/task."""
    if env is None:
        return
    try:
        base_env = unwrap_base_env(env)
        viewer = getattr(base_env, "viewer", None)
        if viewer is not None:
            for name in ("close", "finish", "destroy", "destroy_window"):
                fn = getattr(viewer, name, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception:
                        pass
            try:
                setattr(base_env, "viewer", None)
            except Exception:
                pass
    except Exception:
        pass
    try:
        env.close()
    except Exception as exc:
        print(f"[warn] env.close failed: {exc!r}")


def make_env_for_task(suite_name: str, task_id: int, cfg: dict[str, Any]):
    from libero_pointcloud_utils import ensure_libero_config, make_libero_env, render_camera_names_from_config

    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    requested_mode = str(cfg.get("render_mode", "none")).lower()
    libero_render_mode = requested_mode if requested_mode in {"viewer3d", "mujoco", "onscreen", "headed", "human"} else "offscreen"
    print(f"[info] make_env render_mode={libero_render_mode} render_camera={cfg.get('render_camera', 'agentview')}")
    env, task = make_libero_env(
        suite,
        int(task_id),
        int(cfg["observation_height"]),
        int(cfg["observation_width"]),
        render_camera_names_from_config(cfg),
        render_mode=libero_render_mode,
        render_camera=str(cfg.get("render_camera", "agentview")),
        render_gpu_device_id=int(cfg.get("render_gpu_device_id", -1)),
    )
    return env, task


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_json(args.config) if args.config else {}
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points", 50000))
    cfg["observation_height"] = int(cfg_get(cfg, args.observation_height, "observation_height", 128))
    cfg["observation_width"] = int(cfg_get(cfg, args.observation_width, "observation_width", 128))
    cfg["fps"] = int(cfg_get(cfg, args.fps, "fps", 30))
    cfg["camera_names"] = args.camera_name if args.camera_name else cfg.get("camera_names", ["agentview"])
    cfg["pointcloud_camera_names"] = args.pointcloud_camera_name if args.pointcloud_camera_name else cfg.get("pointcloud_camera_names", ["agentview"])
    cfg["add_gripper_cloud"] = bool(cfg_get(cfg, args.add_gripper_cloud, "add_gripper_cloud", True))
    cfg["gripper_points"] = int(cfg_get(cfg, args.gripper_points, "gripper_points", 500))
    cfg["gripper_len"] = float(cfg_get(cfg, args.gripper_len, "gripper_len", 0.06))
    cfg["gripper_template"] = str(cfg_get(cfg, args.gripper_template, "gripper_template", "reap"))
    cfg["gripper_drop_strategy"] = str(cfg_get(cfg, args.gripper_drop_strategy, "gripper_drop_strategy", "tail"))
    cfg["gripper_shuffle_points"] = bool(cfg_get(cfg, args.gripper_shuffle_points, "gripper_shuffle_points", False))
    cfg["gripper_qpos_max_width"] = float(cfg_get(cfg, args.gripper_qpos_max_width, "gripper_qpos_max_width", 0.08))
    if args.gripper_threshold is None:
        args.gripper_threshold = float(cfg.get("gripper_threshold", 0.5 * float(cfg["gripper_qpos_max_width"])))
    if args.gripper_open_width is None:
        args.gripper_open_width = float(cfg.get("gripper_open_width", float(cfg["gripper_qpos_max_width"])))
    if args.gripper_close_width is None:
        args.gripper_close_width = float(cfg.get("gripper_close_width", 0.0))
    cfg["render_mode"] = "viewer3d" if bool(args.headed) else str(args.render_mode)
    cfg["render_camera"] = str(args.render_camera)
    cfg["render_every_n_steps"] = int(args.render_every_n_steps)
    cfg["render_gpu_device_id"] = int(args.render_gpu_device_id)
    configure_mujoco_render_backend(cfg["render_mode"])

    output_root = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = resolve_task_specs(args, cfg)
    bundle = load_policy_bundle(args)
    print("[info] action_execution=fresh_obs_geom_attach_ref_umi_chunk_to_controller_frame_instant_ik")
    print(
        f"[info] gripper_control mode={args.gripper_control_mode} "
        f"threshold={float(args.gripper_threshold):.4f} "
        f"close_width={float(args.gripper_close_width):.4f} "
        f"open_width={float(args.gripper_open_width):.4f}"
    )
    print(
        f"[info] eval_plan tasks={len(specs)} episodes_per_task={int(args.episodes)} "
        f"total_episodes={len(specs) * int(args.episodes)} recreate_env_per_episode={bool(args.recreate_env_per_episode)}"
    )
    print(
        f"[info] initial_settle steps={int(args.settle_steps)} "
        f"keep_robot_fixed={bool(args.settle_keep_robot_fixed)} render={bool(args.settle_render)}"
    )

    all_results: list[EpisodeResult] = []

    def _write_metrics(last_result: EpisodeResult | None = None, *, print_line: bool = True) -> dict[str, Any]:
        return write_live_metrics(output_root, all_results, specs, args, last_result=last_result, print_line=print_line)

    def _record_failure(spec: TaskSpec, ep: int, task_language: str, ep_dir: Path, exc: Exception) -> None:
        ep_dir.mkdir(parents=True, exist_ok=True)
        failure_payload = {
            "suite": str(spec.suite),
            "task_id": int(spec.task_id),
            "episode": int(ep),
            "task_language": str(task_language),
            "success": False,
            "reward": 0.0,
            "steps": 0,
            "elapsed_s": 0.0,
            "output_dir": str(ep_dir),
            "error": repr(exc),
        }
        with open(ep_dir / "error.json", "w", encoding="utf-8") as f:
            json.dump(failure_payload, f, indent=2, ensure_ascii=False)
        with open(ep_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(failure_payload, f, indent=2, ensure_ascii=False)
        all_results.append(
            EpisodeResult(
                suite=str(spec.suite),
                task_id=int(spec.task_id),
                episode=int(ep),
                task_language=str(task_language),
                success=False,
                reward=0.0,
                steps=0,
                elapsed_s=0.0,
                output_dir=str(ep_dir),
            )
        )
        failed_result = all_results[-1]
        print(f"[WARN] suite={spec.suite} task={spec.task_id} ep={ep} failed: {exc!r}. Continuing.")
        _write_metrics(failed_result)

    for spec_index, spec in enumerate(specs):
        task_dir = output_root / f"{spec.suite}_task_{spec.task_id:03d}"
        if bool(args.recreate_env_per_episode):
            for ep in range(int(args.episodes)):
                env = None
                task = None
                task_language = f"{spec.suite}:{spec.task_id}"
                ep_dir = task_dir / f"episode_{ep:03d}"
                print(f"[episode-start] task_index={spec_index + 1}/{len(specs)} suite={spec.suite} task={spec.task_id} ep={ep}/{int(args.episodes)-1}")
                try:
                    env, task = make_env_for_task(spec.suite, spec.task_id, cfg)
                    task_language = str(getattr(task, "language", getattr(task, "name", f"{spec.suite}:{spec.task_id}")))
                    args._current_suite = spec.suite
                    args._current_task_id = int(spec.task_id)
                    args._current_episode = int(ep)
                    result = run_one_episode(
                        env=env,
                        task_language=task_language,
                        bundle=bundle,
                        cfg=cfg,
                        args=args,
                        output_dir=ep_dir,
                        seed=int(args.seed) + 100000 * int(spec.task_id) + 1000 * ep,
                    )
                    all_results.append(result)
                    with open(ep_dir / "result.json", "w", encoding="utf-8") as f:
                        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
                    print(
                        f"[eval] suite={result.suite} task={result.task_id} ep={result.episode} "
                        f"success={int(result.success)} reward={result.reward:.3f} steps={result.steps} "
                        f"time={result.elapsed_s:.2f}s"
                    )
                    _write_metrics(result)
                except KeyboardInterrupt:
                    close_env_safely(env)
                    raise
                except Exception as exc:
                    _record_failure(spec, ep, task_language, ep_dir, exc)
                    if bool(args.stop_on_error):
                        close_env_safely(env)
                        raise
                finally:
                    close_env_safely(env)
        else:
            env = None
            task = None
            task_language = f"{spec.suite}:{spec.task_id}"
            try:
                env, task = make_env_for_task(spec.suite, spec.task_id, cfg)
                task_language = str(getattr(task, "language", getattr(task, "name", f"{spec.suite}:{spec.task_id}")))
                for ep in range(int(args.episodes)):
                    ep_dir = task_dir / f"episode_{ep:03d}"
                    print(f"[episode-start] task_index={spec_index + 1}/{len(specs)} suite={spec.suite} task={spec.task_id} ep={ep}/{int(args.episodes)-1}")
                    try:
                        args._current_suite = spec.suite
                        args._current_task_id = int(spec.task_id)
                        args._current_episode = int(ep)
                        result = run_one_episode(
                            env=env,
                            task_language=task_language,
                            bundle=bundle,
                            cfg=cfg,
                            args=args,
                            output_dir=ep_dir,
                            seed=int(args.seed) + 100000 * int(spec.task_id) + 1000 * ep,
                        )
                        all_results.append(result)
                        with open(ep_dir / "result.json", "w", encoding="utf-8") as f:
                            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
                        print(
                            f"[eval] suite={result.suite} task={result.task_id} ep={result.episode} "
                            f"success={int(result.success)} reward={result.reward:.3f} steps={result.steps} "
                            f"time={result.elapsed_s:.2f}s"
                        )
                        _write_metrics(result)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        _record_failure(spec, ep, task_language, ep_dir, exc)
                        if bool(args.stop_on_error):
                            raise
            finally:
                close_env_safely(env)

    summary = _write_metrics(None, print_line=False)
    print(
        f"[done] success_rate={summary['success_rate']:.3f} "
        f"episodes={summary['num_episodes']}/{summary['episodes_expected']} output={output_root}"
    )
    return summary

def parse_args() -> argparse.Namespace:


    p = argparse.ArgumentParser(description="Fast non-blocking LIBERO point-cloud instant-IK evaluator.")
    p.add_argument("--config", type=Path, default=Path("benchmarks/song_real_libero/configs/libero.json"))
    p.add_argument("--policy.path", "--policy_path", dest="policy_path", type=Path, required=False, default="benchmarks/song_real_libero/outputs/train_libero_fresh_post/checkpoints/last/pretrained_model")
    p.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=None)
    p.add_argument("--allow-factory-policy-loader", action="store_true", help="Debug only: allow generic LeRobot factory loader fallback instead of SmolVLA_ModelInference.")
    p.add_argument("--dataset.repo_id", dest="dataset_repo_id", default=None)
    p.add_argument("--dataset.root", dest="dataset_root", default="benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/libero_fast_instant_eval"))
    p.add_argument("--overwrite", action="store_true")

    # Task selection. With no suite/task args, the default is 4 tasks: task 0 from
    # libero_spatial/object/goal/10.
    p.add_argument("--task-spec", type=parse_task_spec, action="append", default=None, help="suite:task_id, repeatable")
    p.add_argument("--suite", action="append", default=None)
    p.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=False, help="Evaluate all task ids in the selected suites.")
    p.add_argument("--task-id", type=int, action="append", default=None)
    p.add_argument("--first-n-tasks", type=int, default=None)
    p.add_argument("--require-four-tasks", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--recreate-env-per-episode", action=argparse.BooleanOptionalAction, default=True, help="Create/close a fresh LIBERO env for every episode so viewer/cache/direct-qpos state cannot block the next episode.")
    p.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=False, help="Stop immediately on an episode error instead of recording error.json and continuing.")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--settle-steps", type=int, default=80, help="MuJoCo physics ticks after env.reset() before the first policy inference, so free objects can settle.")
    p.add_argument("--settle-keep-robot-fixed", action=argparse.BooleanOptionalAction, default=True, help="During initial settling, restore arm/gripper qpos each sim tick so only free objects settle.")
    p.add_argument("--settle-render", action=argparse.BooleanOptionalAction, default=False, help="Render viewer during initial settling; useful only for visual debugging.")
    p.add_argument("--debug-settle", action="store_true", help="Print initial settle diagnostics.")

    # Observation / point-cloud contract.
    p.add_argument("--num-points", type=int, default=None)
    p.add_argument("--observation-height", type=int, default=None)
    p.add_argument("--observation-width", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--camera-name", action="append", default=None)
    p.add_argument("--pointcloud-camera-name", action="append", default=None)
    p.add_argument("--add-gripper-cloud", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gripper-points", type=int, default=None)
    p.add_argument("--gripper-len", type=float, default=None)
    p.add_argument("--gripper-template", default=None)
    p.add_argument("--gripper-drop-strategy", default=None)
    p.add_argument("--gripper-shuffle-points", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gripper-qpos-max-width", type=float, default=None)
    p.add_argument(
        "--gripper-control-mode",
        choices=["threshold", "continuous"],
        default="threshold",
        help="threshold: predicted width < threshold closes, otherwise opens; continuous: write model width directly.",
    )
    p.add_argument("--gripper-threshold", type=float, default=None, help="Physical width threshold. Default is 0.5 * gripper-qpos-max-width.")
    p.add_argument("--gripper-open-width", type=float, default=None, help="Width written when predicted width >= threshold. Default is gripper-qpos-max-width.")
    p.add_argument("--gripper-close-width", type=float, default=None, help="Width written when predicted width < threshold. Default is 0.0.")
    p.add_argument("--debug-gripper-joints", action="store_true", help="Print selected MuJoCo gripper joints and their qpos/qvel addresses.")
    p.add_argument("--debug-obs-refresh", action="store_true", help="Print refreshed raw_obs EEF position after each direct IK/sim update.")

    # Action execution. Default is chunk execution without waiting: same row's
    # arm target and gripper width are applied together by IK + direct qpos.
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--action-index", type=int, default=0)
    p.add_argument("--exec-action-steps", type=int, default=16)
    p.add_argument("--replan-every-step", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--action-pose-frame", choices=["current_eff", "world"], default="current_eff")
    p.add_argument("--physics-steps-per-action", type=int, default=1)

    # IK knobs.
    p.add_argument("--ik-iters", type=int, default=80)
    p.add_argument("--ik-damping", type=float, default=1e-3)
    p.add_argument("--ik-pos-tol", type=float, default=1e-4)
    p.add_argument("--ik-rot-tol", type=float, default=2e-3)
    p.add_argument("--ik-step-clip", type=float, default=0.05)
    p.add_argument("--ik-rot-weight", type=float, default=0.5)

    # Attach logic. Widths are in the same continuous gripper-width units saved in
    # the dataset / predicted by the model.
    p.add_argument("--enable-attach", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attach-distance", type=float, default=0.12, help="Fallback distance from EEF to object root/body center for fast attach.")
    p.add_argument("--attach-geom-distance", type=float, default=0.08, help="Distance from EEF to nearest object geom surface that triggers fast attach when gripper is closed.")
    p.add_argument("--attach-close-width", type=float, default=0.018)
    p.add_argument("--attach-release-width", type=float, default=0.035)
    p.add_argument("--attach-object-regex", default=None)
    p.add_argument("--debug-attach", action="store_true", help="Print nearest object geom distances used by AttachManager.")

    # Visualization: all modes are non-blocking; open3d uses poll_events.
    p.add_argument("--render-mode", choices=["none", "viewer3d", "onscreen", "open3d"], default="none")
    p.add_argument("--headed", action=argparse.BooleanOptionalAction, default=False, help="Alias for --render-mode viewer3d")
    p.add_argument("--render-camera", default="agentview")
    p.add_argument("--render-every-n-steps", type=int, default=1)
    p.add_argument("--render-gpu-device-id", type=int, default=-1)
    p.add_argument("--vis3d-every", type=int, default=5)
    p.add_argument("--save-ply", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--save-inference-ply",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save model input point cloud + predicted UMI action chunk PLY/NPZ for debugging.",
    )
    p.add_argument("--inference-ply-every-model-call", type=int, default=1)
    p.add_argument("--inference-ply-max-per-episode", type=int, default=32)
    p.add_argument("--inference-ply-point-stride", type=int, default=1, help="Use >1 to downsample saved point vertices only.")
    p.add_argument("--inference-ply-max-action-frames", type=int, default=16)
    p.add_argument("--print-inference-ply-paths", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--print-metrics", action=argparse.BooleanOptionalAction, default=True, help="Print live overall / suite / task success-rate metrics after every episode.")
    args = p.parse_args()
    if args.headed:
        args.render_mode = "viewer3d"
    return args


if __name__ == "__main__":
    run(parse_args())