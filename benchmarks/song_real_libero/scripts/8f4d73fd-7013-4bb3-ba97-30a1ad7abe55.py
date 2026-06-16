#!/usr/bin/env python
# Fixed synchronized LIBERO benchmark evaluator: anti-twitch pose/gripper gating.
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from smolvla_model_inference import (
    SmolVLA_ModelInference,
    identity_pose9_gripper,
    vis_umi_data,
    write_trajectory_ply,
)
from libero_collect_dataset import (
    append_video_frames,
    export_episode_videos,
    resolve_suite_names,
    resolve_task_ids_for_suite,
)
from libero_pointcloud_utils import (
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
    observation_to_world_point_cloud,
    pointcloud_camera_names_from_config,
    pose9_to_homo_np,
    render_camera_names_from_config,
)


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
VSCODE_DEBUG_DEFAULT_ENV = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
}
VSCODE_DEBUG_DEFAULT_ARGS = [
    "--config",
    str(BENCHMARK_ROOT / "configs" / "libero.json"),
    "--policy.path",
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh_post/checkpoints/last/pretrained_model",
    "--suite",
    "libero_spatial",
    "--suite",
    "libero_object",
    "--all-tasks",
    "--episodes",
    "10",
    "--action-index",
    "0",
    "--exec-action-steps",
    "16",
    "--control-mode",
    "absolute_pose",
    "--pose-wait-until-reached",
    "--pose-wait-pos-tolerance",
    "0.015",
    "--pose-wait-rot-tolerance",
    "0.25",
    "--pose-wait-max-steps",
    "8",
    "--no-replan-every-step",
    "--gripper-wait-until-reached",
    "--gripper-wait-tolerance",
    "0.01",
    "--gripper-wait-max-steps",
    "12",
    "--save-video",
    "--render-mode",
    "viewer3d",
    "--output-dir",
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero_post/outputs/eval_libero_4suite",
    "--control-freq",
    "20",
]

import math
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack
def create_frame(position, rot_matrix, scale=0.03):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scale,
        origin=[0, 0, 0]
    )
    frame.rotate(rot_matrix, center=np.zeros(3))
    frame.translate(position)
    return frame
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def _skew(vec: Tensor) -> Tensor:
    x, y, z = vec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _eye4_like(shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.expand(*shape, 4, 4).clone()


def so3_exp(omega: Tensor, eps: float = 1e-6) -> Tensor:
    omega = omega.to(dtype=torch.float32)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(eps),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2


def so3_log(rot: Tensor, eps: float = 1e-6) -> Tensor:
    rot = rot.to(dtype=torch.float32)
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(vee, dim=-1)
    theta = torch.atan2(sine, cosine)
    theta2 = theta * theta
    factor = torch.where(
        sine > eps,
        theta / (2.0 * sine.clamp_min(eps)),
        0.5 + theta2 / 12.0 + theta2 * theta2 / 720.0,
    )
    return factor.unsqueeze(-1) * vee


def se3_exp(xi: Tensor, eps: float = 1e-6) -> Tensor:
    xi = xi.to(dtype=torch.float32)
    v = xi[..., :3]
    omega = xi[..., 3:6]
    rot = so3_exp(omega, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    a = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )
    b = torch.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(eps),
    )
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
    v_matrix = eye3 + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2
    trans = (v_matrix @ v.unsqueeze(-1)).squeeze(-1)
    out = _eye4_like(xi.shape[:-1], device=xi.device, dtype=xi.dtype)
    out[..., :3, :3] = rot
    out[..., :3, 3] = trans
    return out


def se3_log(transform: Tensor, eps: float = 1e-6) -> Tensor:
    transform = transform.to(dtype=torch.float32)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    omega = so3_log(rot, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    half_theta = 0.5 * theta
    c = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0,
        (1.0 / theta2.clamp_min(eps))
        - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(eps)),
    )
    eye3 = torch.eye(3, device=transform.device, dtype=transform.dtype).expand(*transform.shape[:-2], 3, 3)
    v_inv = eye3 - 0.5 * k + c.unsqueeze(-1) * k2
    v = (v_inv @ trans.unsqueeze(-1)).squeeze(-1)
    # `half_theta` is kept to make the small-angle branch explicit and silence over-eager simplifiers.
    _ = half_theta
    return torch.cat([v, omega], dim=-1)


def se3_left_apply(delta_xi: Tensor, transform: Tensor) -> Tensor:
    return se3_exp(delta_xi) @ transform


def se3_geodesic_loss(pred: Tensor, target: Tensor, trans_weight: float = 1.0, rot_weight: float = 1.0) -> Tensor:
    trans = F.smooth_l1_loss(pred[..., :3, 3], target[..., :3, 3], reduction="none").sum(dim=-1)
    rot = _rotation_geodesic(pred[..., :3, :3], target[..., :3, :3])
    return trans_weight * trans + rot_weight * rot


def _transform_point_cloud_xyzrgb(point_cloud: Tensor, transform: Tensor) -> Tensor:
    xyz = point_cloud[..., :3].to(dtype=torch.float32)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    xyz_out = torch.matmul(xyz, rot.transpose(-1, -2)) + trans.unsqueeze(-2)
    return torch.cat([xyz_out, point_cloud[..., 3:6].to(dtype=torch.float32)], dim=-1)


def _sample_random_se3(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    trans_scale: float = 0.20,
    rot_scale: float = 0.75,
) -> Tensor:
    xi = torch.randn(batch_size, 6, device=device, dtype=dtype)
    xi[..., :3] = xi[..., :3] * float(trans_scale)
    xi[..., 3:6] = xi[..., 3:6] * float(rot_scale)
    return se3_exp(xi)


def _to_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _time_gradient(num_steps: int) -> np.ndarray:
    if num_steps <= 1:
        return np.array([[0.1, 0.75, 0.25]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)[:, None]
    start = np.array([0.05, 0.55, 1.0], dtype=np.float64)
    middle = np.array([0.10, 0.85, 0.25], dtype=np.float64)
    end = np.array([1.0, 0.18, 0.05], dtype=np.float64)
    first_half = (1.0 - 2.0 * t) * start + (2.0 * t) * middle
    second_half = (2.0 - 2.0 * t) * middle + (2.0 * t - 1.0) * end
    return np.where(t <= 0.5, first_half, second_half).clip(0.0, 1.0)


def _make_sphere(center: np.ndarray, radius: float, color: np.ndarray) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    sphere.translate(center)
    sphere.paint_uniform_color(color.tolist())
    return sphere


def _make_trajectory_lines(positions: np.ndarray, colors: np.ndarray) -> o3d.geometry.LineSet | None:
    if positions.shape[0] < 2:
        return None
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector([[idx, idx + 1] for idx in range(positions.shape[0] - 1)])
    line_colors = 0.5 * (colors[:-1] + colors[1:])
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def vis_umi_data(
    action,
    pointcloud,
    *,
    frame_stride: int | None = None,
    max_frames: int = 12,
    frame_scale: float = 0.035,
    point_radius: float = 0.008,
):
    """Visualize a UMI pose9 trajectory with explicit temporal order.

    The trajectory is colored from blue/green at the beginning to red at the end.
    Coordinate frames are drawn sparsely so dense chunks remain readable.
    """
    actions = _to_numpy_array(action).astype(np.float32, copy=False)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[-1] < 9:
        raise ValueError(f"Expected action shape (T, >=9), got {actions.shape}.")

    cloud = _to_numpy_array(pointcloud).astype(np.float32, copy=False)
    while cloud.ndim > 2 and cloud.shape[0] == 1:
        cloud = cloud[0]
    if cloud.ndim != 2 or cloud.shape[-1] < 3:
        raise ValueError(f"Expected pointcloud shape (N, >=3), got {cloud.shape}.")

    positions = actions[:, :3]
    colors = _time_gradient(positions.shape[0])
    if frame_stride is None:
        frame_stride = max(1, int(math.ceil(positions.shape[0] / max(1, max_frames))))
    frame_indices = list(range(0, positions.shape[0], max(1, int(frame_stride))))
    if positions.shape[0] - 1 not in frame_indices:
        frame_indices.append(positions.shape[0] - 1)

    geometries = [create_frame(np.array([0.0, 0.0, 0.0]), np.eye(3), scale=frame_scale * 1.2)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    if cloud.shape[-1] >= 6:
        rgb = np.clip(cloud[:, 3:6] / 255.0, 0.0, 1.0)
    else:
        rgb = np.full((cloud.shape[0], 3), 0.55, dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geometries.append(pcd)

    trajectory_lines = _make_trajectory_lines(positions, colors)
    if trajectory_lines is not None:
        geometries.append(trajectory_lines)

    # Small colored beads make the time direction visible even when frames overlap.
    for idx, (position, color) in enumerate(zip(positions, colors, strict=True)):
        radius = point_radius * (1.6 if idx in (0, positions.shape[0] - 1) else 1.0)
        geometries.append(_make_sphere(position, radius, color))

    for idx in frame_indices:
        rot6d = torch.as_tensor(actions[idx, 3:9], dtype=torch.float32)
        rotmat = rot6d_to_matrix(rot6d).cpu().numpy()
        scale = frame_scale * (1.35 if idx in (0, positions.shape[0] - 1) else 1.0)
        geometries.append(create_frame(positions[idx], rotmat, scale=scale))

    print(
        f"Visualizing {positions.shape[0]} poses: blue/green=start, red=end, "
        f"frames={frame_indices}, start={positions[0].round(4)}, end={positions[-1].round(4)}"
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="UMI trajectory: blue/green=start, red=end",
    )





def load_config(path: Path) -> dict[str, Any]:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str) -> Any:
    return cli_value if cli_value is not None else cfg[key]


def configure_mujoco_render_backend(cfg: dict[str, Any]) -> None:
    render_mode = str(cfg.get("render_mode", "offscreen")).lower()
    if render_mode not in {"onscreen", "headed", "human", "viewer3d", "mujoco"}:
        return
    if not sys.platform.startswith("linux"):
        return

    mujoco_gl = os.environ.get("MUJOCO_GL", "").lower().strip()
    if mujoco_gl in {"", "glfw", "egl"}:
        # robosuite can override glfw to egl when GPU rendering is enabled in
        # macros; glx keeps it on the desktop path and avoids EGL imports.
        os.environ["MUJOCO_GL"] = "glx"

    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "").lower().strip()
    if pyopengl_platform in {"glfw", "egl"}:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        for key, value in VSCODE_DEBUG_DEFAULT_ENV.items():
            os.environ.setdefault(key, value)
        argv = list(VSCODE_DEBUG_DEFAULT_ARGS)
        print("[debug] No CLI args detected; using built-in VSCode debug defaults.")
    parser = argparse.ArgumentParser(description="Evaluate the point-cloud SmolVLA policy on LIBERO.")
    parser.add_argument("--control-freq", type=float, default=None)
    parser.add_argument("--config", type=Path, default=BENCHMARK_ROOT / "configs" / "libero.json")
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", default="benchmarks/song_real_libero/outputs/train_libero_fresh_post/checkpoints/last/pretrained_model")
    parser.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/eval_libero_4suite")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode", choices=("offscreen", "onscreen", "viewer3d"), default="viewer3d")
    parser.add_argument("--recreate-env-per-episode", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-episode-lifecycle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--headed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-camera", default=None)
    parser.add_argument("--render-every-n-steps", type=int, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--action-index", type=int, default=None)
    parser.add_argument("--exec-action-steps", type=int, default=None)
    parser.add_argument("--control-mode", choices=("absolute_pose", "delta_pose"), default=None)
    parser.add_argument("--replan-every-step", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--pose-wait-until-reached", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-wait-pos-tolerance", type=float, default=None)
    parser.add_argument("--pose-wait-rot-tolerance", type=float, default=None)
    parser.add_argument("--pose-wait-max-steps", type=int, default=None)
    parser.add_argument("--gripper-wait-until-reached", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gripper-wait-tolerance", type=float, default=None)
    parser.add_argument("--gripper-wait-max-steps", type=int, default=None)
    parser.add_argument("--gripper-close-threshold", type=float, default=None)
    parser.add_argument("--gripper-contact-stall-tolerance", type=float, default=None)
    parser.add_argument("--gripper-contact-stall-steps", type=int, default=None)
    parser.add_argument("--gripper-close-pose-wait-max-steps", type=int, default=None)
    parser.add_argument("--gripper-close-hold-steps", type=int, default=None)
    parser.add_argument("--gripper-grasp-required-to-advance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gripper-grasp-wait-max-steps", type=int, default=None)
    parser.add_argument("--execution-mode", choices=("sync_wait", "fast_physics"), default=None)
    parser.add_argument("--fast-physics-min-steps", type=int, default=None)
    parser.add_argument("--fast-physics-max-steps", type=int, default=None)
    parser.add_argument("--fast-physics-max-duration-s", type=float, default=None)
    parser.add_argument("--fast-physics-pos-step-m", type=float, default=None)
    parser.add_argument("--fast-physics-rot-step-rad", type=float, default=None)
    parser.add_argument("--fast-physics-close-steps", type=int, default=None)
    parser.add_argument("--fast-physics-open-steps", type=int, default=None)
    parser.add_argument("--fast-physics-close-after-pose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fast-physics-success-check-every", type=int, default=None)
    parser.add_argument("--fast-physics-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keyboard-vis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keyboard-vis-mode", choices=("ply", "window"), default=None)
    return parser.parse_args(argv)


def matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    rot = matrix[..., :3, :3]
    return np.concatenate([matrix[..., :3, 3], rot[..., :, 0], rot[..., :, 1]], axis=-1).astype(np.float32)


def planned_world_poses_from_umi_chunk(current_eef_pose9: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    current_world = pose9_to_homo_np(np.asarray(current_eef_pose9, dtype=np.float32)[..., :9])
    planned_relative = pose9_to_homo_np(np.asarray(chunk, dtype=np.float32)[:, :9])
    return current_world @ planned_relative


def pose9_to_libero_motion_action(
    action_pose9: np.ndarray,
    trans_scale: float,
    rot_scale: float,
    *,
    current_eef_pose9: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action_pose9, dtype=np.float32).reshape(-1)
    relative = pose9_to_homo_np(action[:9])
    if current_eef_pose9 is None:
        delta_pos = relative[:3, 3]
        delta_rot = relative[:3, :3]
    else:
        current_world = pose9_to_homo_np(np.asarray(current_eef_pose9, dtype=np.float32)[..., :9])
        target_world = current_world @ relative
        delta_pos = target_world[:3, 3] - current_world[:3, 3]
        delta_rot = target_world[:3, :3] @ current_world[:3, :3].T

    trans = np.clip(delta_pos / max(float(trans_scale), 1e-6), -1.0, 1.0)
    rotvec = R.from_matrix(delta_rot).as_rotvec().astype(np.float32)
    rotvec = np.clip(rotvec / max(float(rot_scale), 1e-6), -1.0, 1.0)
    return np.concatenate([trans, rotvec]).astype(np.float32)


def world_pose_to_libero_absolute_action(target_world: np.ndarray) -> np.ndarray:
    target_world = np.asarray(target_world, dtype=np.float32)
    rotvec = R.from_matrix(target_world[:3, :3]).as_rotvec().astype(np.float32)
    return np.concatenate([target_world[:3, 3], rotvec]).astype(np.float32)


def current_controller_eef_world(env: Any) -> np.ndarray:
    """Return the exact world pose controlled by robosuite OSC.

    LIBERO observations expose eef position from the gripper site but eef quaternion
    from the hand body. OSC controls the gripper site pose, so absolute commands
    need this site orientation instead of the observation body orientation.
    """
    if not getattr(env, "robots", None):
        raise ValueError("Expected LIBERO env to expose at least one robot.")
    controller = getattr(env.robots[0], "controller", None)
    if controller is None:
        raise ValueError("Expected the first LIBERO robot to expose an OSC controller.")
    controller.update(force=True)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(controller.ee_ori_mat, dtype=np.float32)
    transform[:3, 3] = np.asarray(controller.ee_pos, dtype=np.float32)
    return transform


def model_world_pose_to_controller_world(
    target_model_world: np.ndarray,
    current_model_world: np.ndarray,
    current_controller_world: np.ndarray,
) -> np.ndarray:
    model_to_controller = fast_inverse_homogeneous(current_model_world) @ current_controller_world
    return (target_model_world @ model_to_controller).astype(np.float32)


def pose_error_to_target(current_world: np.ndarray, target_world: np.ndarray) -> tuple[float, float]:
    current_world = np.asarray(current_world, dtype=np.float32)
    target_world = np.asarray(target_world, dtype=np.float32)
    pos_error = float(np.linalg.norm(target_world[:3, 3] - current_world[:3, 3]))
    rot_delta = target_world[:3, :3] @ current_world[:3, :3].T
    rot_error = float(np.linalg.norm(R.from_matrix(rot_delta).as_rotvec()))
    return pos_error, rot_error


def gripper_width_delta_action(
    target_width: float,
    current_width: float,
    *,
    max_physical_width: float,
) -> float:
    """Map an absolute predicted gripper width target to LIBERO's continuous command.

    The model predicts physical gripper width. LIBERO / robosuite expects a
    velocity-like command where negative opens and positive closes, so the command
    is only the signed normalized width error. No thresholding, smoothing, dwell,
    or open/close class is applied here.
    """
    target_pct = gripper_width_percent_from_scalar(
        float(target_width),
        max_physical_width=float(max_physical_width),
    )
    current_pct = gripper_width_percent_from_scalar(
        float(current_width),
        max_physical_width=float(max_physical_width),
    )
    return float(np.clip(current_pct - target_pct, -1.0, 1.0))


def canonical_gripper_width(width: float, *, max_physical_width: float) -> float:
    return float(
        gripper_width_percent_from_scalar(
            float(width),
            max_physical_width=float(max_physical_width),
        )
        * float(max_physical_width)
    )


def resolve_gripper_width_for_execution(
    predicted_width: float,
    *,
    mode: str | None = None,
    threshold: float | None = None,
    open_width: float | None = None,
    close_width: float | None = None,
    qpos_max_width: float,
) -> tuple[float, str]:
    """Use the model action's last dimension directly as physical gripper width.

    Mirrors libero_fast_instant_eval V23+: no thresholding, no open/close class,
    and no task-specific close threshold.  Legacy arguments are accepted only so
    old configs / command lines keep running; they intentionally do not affect
    execution.  The returned width is the postprocessed inference action value
    clipped to the valid physical gripper-width range.
    """
    _ = (mode, threshold, open_width, close_width)
    max_width = max(float(qpos_max_width), 0.0)
    executed_width = float(np.clip(float(predicted_width), 0.0, max_width))
    return executed_width, "model_action_last_dim"


def _control_list(control: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = control.get(key, default)
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    return [str(part).strip().lower() for part in value if str(part).strip()]


def task_matches_keywords(task_language: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    task_lower = str(task_language).lower()
    return any(keyword in task_lower for keyword in keywords)


def task_matches_all_keywords(task_language: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    task_lower = str(task_language).lower()
    return all(str(keyword).lower() in task_lower for keyword in keywords)


def gripper_close_threshold_for_task(task_language: str, control: dict[str, Any]) -> float:
    default_threshold = float(control.get("gripper_close_threshold", 0.07))
    rules = control.get("gripper_close_threshold_rules", []) or []
    if not isinstance(rules, list):
        return default_threshold
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = [str(item).strip().lower() for item in rule.get("keywords", []) or [] if str(item).strip()]
        if not task_matches_all_keywords(task_language, keywords):
            continue
        try:
            return float(rule["threshold"])
        except (KeyError, TypeError, ValueError):
            continue
    return default_threshold


def iter_libero_env_chain(env: Any, max_depth: int = 32) -> list[Any]:
    """Return env plus wrapped inner envs, outermost first, with cycle protection."""
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


def unwrap_libero_env(env: Any) -> Any:
    chain = iter_libero_env_chain(env)
    return chain[-1] if chain else env


def _viewer_attrs_to_clear() -> tuple[str, ...]:
    return (
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
    )


def _find_mujoco_viewer(env: Any) -> Any | None:
    """Find a viewer handle attached anywhere in the wrapper chain."""
    for obj in reversed(iter_libero_env_chain(env)):
        for attr in _viewer_attrs_to_clear():
            viewer = getattr(obj, attr, None)
            if viewer is not None:
                return viewer
    return None


def _viewer_running(viewer: Any) -> bool:
    if viewer is None:
        return False
    for name in ("is_running", "is_alive"):
        value = getattr(viewer, name, None)
        try:
            if callable(value):
                return bool(value())
            if value is not None:
                return bool(value)
        except Exception:
            pass
    for name in ("_closed", "closed"):
        value = getattr(viewer, name, None)
        if value is not None:
            try:
                return not bool(value)
            except Exception:
                pass
    # Robosuite viewer wrappers often do not expose a running flag.  Treat a
    # present handle as usable and let render/sync raise if it is stale.
    return True


def _clear_mujoco_viewer_refs(env: Any) -> None:
    """Clear viewer handles / init flags on every wrapper layer.

    attach_mujoco_3d_viewer() can attach the passive viewer to different objects
    depending on the robosuite / LIBERO wrapper stack.  Clearing only the deepest
    base_env.viewer leaves stale handles or "already attached" flags behind, so
    the next episode may execute without opening/syncing a new 3D viewer.
    """
    for obj in iter_libero_env_chain(env):
        for attr in _viewer_attrs_to_clear():
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, None)
                except Exception:
                    pass
        for attr in (
            "viewer3d_attached",
            "_viewer3d_attached",
            "mujoco_3d_viewer_attached",
            "_mujoco_3d_viewer_attached",
            "viewer_initialized",
            "_viewer_initialized",
            "mujoco_viewer_initialized",
            "_mujoco_viewer_initialized",
        ):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, False)
                except Exception:
                    pass


def close_mujoco_viewer_safely(env: Any) -> None:
    """Close and fully detach MuJoCo's passive 3D viewer without closing env.

    The important part is clearing viewer references on *all* wrapper layers.  If
    a closed passive viewer object survives on any layer, the next episode can run
    normally while attach_mujoco_3d_viewer() silently reuses that dead handle and
    no new 3D window appears.
    """
    if env is None:
        return
    viewers: list[Any] = []
    try:
        for obj in iter_libero_env_chain(env):
            for attr in _viewer_attrs_to_clear():
                viewer = getattr(obj, attr, None)
                if viewer is not None and not any(viewer is old for old in viewers):
                    viewers.append(viewer)
        for viewer in viewers:
            for name in ("close", "finish", "destroy", "destroy_window"):
                fn = getattr(viewer, name, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception:
                        pass
        _clear_mujoco_viewer_refs(env)
    except Exception as exc:
        print(f"[WARN] failed to close MuJoCo viewer cleanly: {exc!r}")
        try:
            _clear_mujoco_viewer_refs(env)
        except Exception:
            pass

def goal_predicate_status(env: Any) -> dict[str, Any]:
    base_env = unwrap_libero_env(env)
    goal_state = list(getattr(base_env, "parsed_problem", {}).get("goal_state", []) or [])
    eval_predicate = getattr(base_env, "_eval_predicate", None)
    predicates: list[dict[str, Any]] = []
    for state in goal_state:
        state_items = [str(item) for item in state]
        satisfied = False
        error = None
        if callable(eval_predicate):
            try:
                satisfied = bool(eval_predicate(state))
            except Exception as exc:
                error = repr(exc)
        predicates.append(
            {
                "predicate": state_items,
                "satisfied": bool(satisfied),
                **({"error": error} if error is not None else {}),
            }
        )
    return {
        "all_satisfied": bool(predicates) and all(item["satisfied"] for item in predicates),
        "predicates": predicates,
    }


def gripper_grasp_status(env: Any) -> dict[str, Any]:
    """Return per-object Panda fingerpad contact / grasp diagnostics.

    LIBERO does not create a weld when an object is picked. Robosuite's grasp
    test is contact-based: both left and right fingerpads must touch object
    collision geoms. This helper records that signal without changing physics.
    """
    base_env = unwrap_libero_env(env)
    objects_dict = getattr(base_env, "objects_dict", {}) or {}
    obj_names = [
        str(name)
        for name in getattr(base_env, "obj_of_interest", []) or []
        if str(name) in objects_dict
    ]
    if not obj_names:
        obj_names = list(objects_dict.keys())

    robots = getattr(base_env, "robots", None) or []
    gripper = getattr(robots[0], "gripper", None) if robots else None
    check_contact = getattr(base_env, "check_contact", None)
    check_grasp = getattr(base_env, "_check_grasp", None)
    statuses: list[dict[str, Any]] = []
    for name in obj_names:
        obj = objects_dict.get(name)
        if obj is None or gripper is None or not callable(check_contact):
            continue
        contact_geoms = list(getattr(obj, "contact_geoms", []) or [])
        if not contact_geoms:
            continue
        important_geoms = getattr(gripper, "important_geoms", {}) or {}
        left_geoms = important_geoms.get("left_fingerpad", [])
        right_geoms = important_geoms.get("right_fingerpad", [])
        left_contact = False
        right_contact = False
        grasped = False
        try:
            left_contact = bool(check_contact(left_geoms, contact_geoms))
            right_contact = bool(check_contact(right_geoms, contact_geoms))
            grasped = bool(left_contact and right_contact)
            if callable(check_grasp):
                grasped = bool(check_grasp(gripper, obj))
        except Exception:
            left_contact = False
            right_contact = False
            grasped = False
        statuses.append(
            {
                "object": name,
                "category": str(getattr(obj, "category_name", "")),
                "left_contact": bool(left_contact),
                "right_contact": bool(right_contact),
                "any_contact": bool(left_contact or right_contact),
                "grasped": bool(grasped),
            }
        )

    return {
        "objects": statuses,
        "any_contact": any(item["any_contact"] for item in statuses),
        "any_grasped": any(item["grasped"] for item in statuses),
        "grasped_objects": [item["object"] for item in statuses if item["grasped"]],
        "contact_objects": [item["object"] for item in statuses if item["any_contact"]],
    }


def add_timing(
    totals: dict[str, float],
    counts: dict[str, int],
    name: str,
    start_s: float,
) -> None:
    totals[name] += time.perf_counter() - start_s
    counts[name] += 1


def summarize_timings(
    totals: dict[str, float],
    counts: dict[str, int],
    *,
    wall_s: float,
) -> dict[str, Any]:
    sections = {}
    for name in sorted(totals):
        total_s = float(totals[name])
        count = int(counts.get(name, 0))
        sections[name] = {
            "total_s": total_s,
            "count": count,
            "mean_s": float(total_s / count) if count > 0 else 0.0,
            "wall_percent": float(100.0 * total_s / wall_s) if wall_s > 0 else 0.0,
        }
    return {"wall_s": float(wall_s), "sections": sections}


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


def episode_result_record(result: dict[str, Any], episode_idx: int) -> dict[str, Any]:
    record = {
        "episode_index": int(episode_idx),
        **{
            k: v
            for k, v in result.items()
            if not isinstance(v, np.ndarray) and k != "video_frames"
        },
    }
    return json_safe(record)


def task_success_rate(task_results: list[dict[str, Any]]) -> float:
    return float(np.mean([bool(item.get("success", False)) for item in task_results])) if task_results else 0.0


def make_task_summary(
    *,
    suite_name: str,
    task_id: int,
    task_name: str,
    task_language: str,
    task_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "task_id": int(task_id),
        "task_name": task_name,
        "task_language": task_language,
        "episodes": json_safe(task_results),
        "success_rate": task_success_rate(task_results),
    }


def aggregate_task_results(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = int(sum(len(task.get("episodes", [])) for task in task_results))
    success_count = int(
        sum(
            1
            for task in task_results
            for episode in task.get("episodes", [])
            if bool(episode.get("success", False))
        )
    )
    task_success_rates = [float(task.get("success_rate", 0.0)) for task in task_results]
    return {
        "task_count": int(len(task_results)),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": float(success_count / episode_count) if episode_count else 0.0,
        "task_success_rate_mean": float(np.mean(task_success_rates)) if task_success_rates else 0.0,
    }


def build_eval_summary(
    *,
    cfg: dict[str, Any],
    suite_names: list[str],
    all_results: list[dict[str, Any]],
) -> dict[str, Any]:
    suite_reports = []
    for suite_name in suite_names:
        suite_results = [item for item in all_results if item.get("suite") == suite_name]
        suite_reports.append(
            {
                "suite": suite_name,
                **aggregate_task_results(suite_results),
                "tasks": suite_results,
            }
        )
    overall = aggregate_task_results(all_results)
    now_s = time.time()
    return {
        "created_unix_s": now_s,
        "updated_unix_s": now_s,
        "policy_path": cfg["policy_path"],
        "suites": suite_names,
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": pointcloud_camera_names_from_config(cfg),
        "render_camera_names": render_camera_names_from_config(cfg),
        "render_mode": str(cfg.get("render_mode", "offscreen")),
        "render_camera": str(cfg.get("render_camera", "agentview")),
        "render_every_n_steps": int(cfg.get("render_every_n_steps", 1)),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "gripper_control": dict(cfg.get("control", {})),
        "suite_reports": suite_reports,
        "overall": overall,
        "results": all_results,
        "overall_success_rate": float(overall["success_rate"]),
        "overall_task_success_rate_mean": float(overall["task_success_rate_mean"]),
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_eval_reports(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    suite_names: list[str],
    all_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = build_eval_summary(cfg=cfg, suite_names=suite_names, all_results=all_results)
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "overall_report.json", {"overall": summary["overall"], "suites": summary["suite_reports"]})
    for suite_report in summary["suite_reports"]:
        write_json_atomic(output_dir / str(suite_report["suite"]) / "suite_report.json", suite_report)
    return summary


def correct_rim_grasp_target_world(
    target_world: np.ndarray,
    point_cloud_world: np.ndarray,
    *,
    enabled: bool,
    min_points: int,
    axis_limit: float,
    depth_limit: float,
    z_min: float,
    z_max: float,
    top_quantile: float,
    center_alpha: float,
    max_axis_shift: float,
    max_depth_shift: float,
    lift: float,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    metrics: dict[str, float | bool | int] = {
        "applied": False,
        "point_count": 0,
        "axis_shift": 0.0,
        "depth_shift": 0.0,
        "lift": 0.0,
    }
    if not enabled:
        return target_world, metrics

    points = np.asarray(point_cloud_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return target_world, metrics
    xyz = points[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    if not bool(finite.any()):
        return target_world, metrics

    target_world = np.asarray(target_world, dtype=np.float32)
    world_to_target = fast_inverse_homogeneous(target_world)
    xyz_h = np.concatenate([xyz[finite], np.ones((int(finite.sum()), 1), dtype=np.float32)], axis=-1)
    local = (world_to_target @ xyz_h.T).T[:, :3]
    mask = (
        (np.abs(local[:, 0]) <= float(axis_limit))
        & (np.abs(local[:, 1]) <= float(depth_limit))
        & (local[:, 2] >= float(z_min))
        & (local[:, 2] <= float(z_max))
    )
    candidates = local[mask]
    metrics["point_count"] = int(candidates.shape[0])
    if candidates.shape[0] < int(min_points):
        return target_world, metrics

    z_cut = float(np.quantile(candidates[:, 2], np.clip(float(top_quantile), 0.0, 1.0)))
    rim_points = candidates[candidates[:, 2] >= z_cut]
    if rim_points.shape[0] < int(min_points):
        rim_points = candidates

    axis_shift = float(np.median(rim_points[:, 0])) * float(center_alpha)
    axis_shift = float(np.clip(axis_shift, -float(max_axis_shift), float(max_axis_shift)))
    depth_shift = float(np.median(rim_points[:, 1])) * float(center_alpha)
    depth_shift = float(np.clip(depth_shift, -float(max_depth_shift), float(max_depth_shift)))
    lift = float(lift)
    if abs(axis_shift) <= 1e-6 and abs(depth_shift) <= 1e-6 and abs(lift) <= 1e-6:
        return target_world, metrics

    corrected = target_world.copy()
    local_shift = np.asarray([axis_shift, depth_shift, 0.0], dtype=np.float32)
    corrected[:3, 3] = corrected[:3, 3] + corrected[:3, :3] @ local_shift
    corrected[2, 3] = corrected[2, 3] + lift
    metrics.update({"applied": True, "axis_shift": axis_shift, "depth_shift": depth_shift, "lift": lift})
    return corrected.astype(np.float32), metrics


def _sync_or_render_viewer(viewer: Any) -> bool:
    """Advance a MuJoCo passive viewer handle or robosuite viewer wrapper."""
    if viewer is None:
        return False
    # mujoco.viewer.launch_passive returns a handle whose public method is sync().
    for name in ("sync", "render"):
        fn = getattr(viewer, name, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception as exc:
                print(f"[WARN] viewer.{name} failed: {exc!r}")
                return False
    return False


def render_onscreen_env(env: Any, cfg: dict[str, Any], step: int, *, force: bool = False) -> None:
    render_mode = str(cfg.get("render_mode", "offscreen")).lower()
    if render_mode not in {"onscreen", "headed", "human", "viewer3d", "mujoco"}:
        return

    every_n = int(cfg.get("render_every_n_steps", 1) or 1)
    if every_n <= 0:
        return
    if not force and int(step) % every_n != 0:
        return

    render_camera = normalize_render_camera_name(str(cfg.get("render_camera", "agentview")))

    if render_mode in {"viewer3d", "mujoco"}:
        viewer = _find_mujoco_viewer(env)
        if viewer is not None and not _viewer_running(viewer):
            close_mujoco_viewer_safely(env)
            viewer = None

        if viewer is None:
            try:
                attach_mujoco_3d_viewer(env, render_camera=render_camera)
            except Exception as exc:
                print(f"[WARN] attach_mujoco_3d_viewer failed: {exc!r}")
            viewer = _find_mujoco_viewer(env)

        if _sync_or_render_viewer(viewer):
            time.sleep(1.0 / 60.0)
            return

        # If the handle existed but could not sync/render, clear it once and try a
        # fresh attach.  This fixes the episode-1 case where a closed passive
        # viewer object was still cached on one wrapper layer.
        if viewer is not None:
            close_mujoco_viewer_safely(env)
            try:
                attach_mujoco_3d_viewer(env, render_camera=render_camera)
            except Exception as exc:
                print(f"[WARN] reattach_mujoco_3d_viewer failed: {exc!r}")
            viewer = _find_mujoco_viewer(env)
            if _sync_or_render_viewer(viewer):
                time.sleep(1.0 / 60.0)
                return

    render_fn = getattr(env, "render", None)
    if callable(render_fn):
        try:
            render_fn()
            if render_mode in {"onscreen", "headed", "human", "viewer3d", "mujoco"}:
                time.sleep(1.0 / 60.0)
            return
        except Exception as exc:
            print(f"[WARN] env.render failed: {exc!r}")
            return

    inner_env = getattr(env, "env", None)
    inner_render = getattr(inner_env, "render", None)
    if callable(inner_render):
        try:
            inner_render()
            return
        except Exception as exc:
            print(f"[WARN] inner env.render failed: {exc!r}")
            return

    raise AttributeError(f"The LIBERO environment does not expose a render() method for {render_mode} mode.")

def _rgb_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return colors.reshape(-1, 3).astype(np.uint8)
    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


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


def save_keyboard_debug_visualization(
    *,
    output_dir: Path,
    step: int,
    action_chunk: np.ndarray,
    point_cloud: np.ndarray,
) -> Path:
    debug_dir = output_dir / "keyboard_vis"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step_{int(step):04d}_{int(time.time() * 1000)}"
    ply_path = debug_dir / f"{stem}.ply"
    write_umi_debug_ply(ply_path, action_chunk, point_cloud)
    np.savez_compressed(
        debug_dir / f"{stem}.npz",
        action_chunk=np.asarray(action_chunk, dtype=np.float32),
        point_cloud=np.asarray(point_cloud, dtype=np.float32),
        step=np.asarray(step, dtype=np.int64),
    )
    return ply_path


class TerminalKeyWatcher:
    """Non-blocking single-key reader for interactive rollout debugging."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyWatcher":
        if not self.enabled or not sys.stdin.isatty():
            self.enabled = False
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            print("[debug] Press 'v' in this terminal to visualize the latest predicted UMI action chunk.")
        except Exception as exc:
            self.enabled = False
            print(f"[debug] Keyboard visualization disabled: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def consume_visualize_request(self) -> bool:
        if not self.enabled:
            return False
        requested = False
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                break
            char = sys.stdin.read(1)
            if char == "":
                break
            if char == "\x03":
                raise KeyboardInterrupt
            if char.lower() == "v":
                requested = True
        return requested



def _control_freq_hz(cfg: dict[str, Any], control: dict[str, Any]) -> float:
    """Return the robosuite control frequency used by make_libero_env()."""
    try:
        return float(cfg.get("control_freq", control.get("control_freq", 20.0)))
    except Exception:
        return 20.0


def _fast_physics_action_budget(
    *,
    cfg: dict[str, Any],
    control: dict[str, Any],
    pre_pose_pos_error: float,
    pre_pose_rot_error: float,
    gripper_should_close: bool,
) -> int:
    """Compute a small physical-servo budget for one model action row.

    This is not a separate pose/gripper wait loop.  It is a bounded number of
    real robosuite/MuJoCo env.step() calls used to let OSC and gripper actuators
    physically track one target before the next model action row is consumed.
    The budget is adaptive, so tiny motions use only a few physics ticks while
    large pose deltas still get enough simulated time to avoid action/EEF drift.
    """
    min_steps = max(1, int(control.get("fast_physics_min_steps", 2)))
    max_steps_cfg = max(min_steps, int(control.get("fast_physics_max_steps", 12)))
    max_duration_s = float(control.get("fast_physics_max_duration_s", 0.0) or 0.0)
    if max_duration_s > 0.0:
        hz = max(_control_freq_hz(cfg, control), 1.0)
        # Duration is a cap expressed in physical control time.  Keep the explicit
        # step cap as the user's upper bound when it is larger.
        max_steps_cfg = max(max_steps_cfg, int(math.ceil(max_duration_s * hz)))

    pos_per_step = max(1e-6, float(control.get("fast_physics_pos_step_m", 0.012)))
    rot_per_step = max(1e-6, float(control.get("fast_physics_rot_step_rad", 0.18)))
    estimated_pose_steps = int(
        math.ceil(
            max(
                float(pre_pose_pos_error) / pos_per_step,
                float(pre_pose_rot_error) / rot_per_step,
            )
        )
    )
    gripper_steps = int(
        control.get(
            "fast_physics_close_steps" if bool(gripper_should_close) else "fast_physics_open_steps",
            6 if bool(gripper_should_close) else 2,
        )
    )
    gripper_steps = max(0, gripper_steps)
    return int(np.clip(max(min_steps, estimated_pose_steps + gripper_steps), min_steps, max_steps_cfg))


def execute_fast_physics_action(
    *,
    env: Any,
    cfg: dict[str, Any],
    raw_obs: dict[str, Any],
    motion_action: np.ndarray,
    controller_target_world: np.ndarray,
    gripper_target_width: float,
    gripper_should_close: bool,
    gripper_max_width: float,
    pose_wait_pos_tolerance: float,
    pose_wait_rot_tolerance: float,
    gripper_wait_tolerance: float,
    timing_totals: dict[str, float],
    timing_counts: dict[str, int],
    outer_step: int,
) -> dict[str, Any]:
    """Bounded physical execution of one model action row.

    The original synchronized evaluator had two serial blockers: pose wait and
    gripper wait.  This executor removes those separate wait states, but it still
    uses real robosuite OSC + gripper actuator env.step() calls.  It repeatedly
    commands the same absolute target for a short adaptive budget, re-measures the
    actual EEF/gripper state after each physics tick, and only then advances the
    model action queue.  Closing can be delayed until the EEF is near the target,
    so the gripper does not clamp before the fingers are at the predicted grasp
    pose.  No direct qpos teleporting or kinematic attach is used here.
    """
    control = cfg["control"]
    motion_action = np.asarray(motion_action, dtype=np.float32).reshape(6)
    controller_target_world = np.asarray(controller_target_world, dtype=np.float32)
    close_after_pose = bool(control.get("fast_physics_close_after_pose", True))
    success_check_every = max(1, int(control.get("fast_physics_success_check_every", 1)))
    debug = bool(control.get("fast_physics_debug", False))

    timing_start = time.perf_counter()
    pre_controller_world = current_controller_eef_world(env)
    add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)
    pre_pose_pos_error, pre_pose_rot_error = pose_error_to_target(pre_controller_world, controller_target_world)
    budget_steps = _fast_physics_action_budget(
        cfg=cfg,
        control=control,
        pre_pose_pos_error=pre_pose_pos_error,
        pre_pose_rot_error=pre_pose_rot_error,
        gripper_should_close=bool(gripper_should_close),
    )
    required_active_gripper_steps = max(
        0,
        int(
            control.get(
                "fast_physics_close_steps" if bool(gripper_should_close) else "fast_physics_open_steps",
                6 if bool(gripper_should_close) else 2,
            )
        ),
    )
    min_steps = max(1, int(control.get("fast_physics_min_steps", 2)))
    active_gripper_steps = 0
    physical_steps = 0
    reward_sum = 0.0
    max_reward = 0.0
    done = False
    success_now = False
    terminated_value_error = False
    last_libero_action = np.concatenate([motion_action, np.asarray([0.0], dtype=np.float32)]).astype(np.float32)
    pose_pos_error = float(pre_pose_pos_error)
    pose_rot_error = float(pre_pose_rot_error)
    gripper_actual_width = canonical_gripper_width(float(gripper_scalar(raw_obs)), max_physical_width=gripper_max_width)
    gripper_width_error = abs(gripper_actual_width - float(gripper_target_width))

    # Guarantee that closing still gets a short physical actuation window even
    # if the arm never reaches the pose tolerance within this bounded budget.
    force_gripper_start = max(0, budget_steps - required_active_gripper_steps)

    for local_idx in range(budget_steps):
        timing_start = time.perf_counter()
        current_controller_world = current_controller_eef_world(env)
        add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)
        pose_pos_error, pose_rot_error = pose_error_to_target(current_controller_world, controller_target_world)
        pose_reached = pose_pos_error <= pose_wait_pos_tolerance and pose_rot_error <= pose_wait_rot_tolerance

        current_width = canonical_gripper_width(float(gripper_scalar(raw_obs)), max_physical_width=gripper_max_width)
        gripper_cmd = gripper_width_delta_action(
            float(gripper_target_width),
            current_width,
            max_physical_width=gripper_max_width,
        )
        if bool(gripper_should_close):
            gripper_cmd = max(float(gripper_cmd), 0.0)
        else:
            gripper_cmd = min(float(gripper_cmd), 0.0)

        gripper_allowed = True
        if bool(gripper_should_close) and close_after_pose and not pose_reached and local_idx < force_gripper_start:
            gripper_allowed = False
            gripper_cmd = 0.0
        if abs(gripper_cmd) > 1e-6:
            active_gripper_steps += 1

        last_libero_action = np.concatenate(
            [motion_action.astype(np.float32), np.asarray([gripper_cmd], dtype=np.float32)]
        ).astype(np.float32)

        try:
            timing_start = time.perf_counter()
            raw_obs, reward, done, _ = env.step(last_libero_action)
            add_timing(timing_totals, timing_counts, "env_step", timing_start)
            physical_steps += 1
        except ValueError as exc:
            if "terminated episode" in str(exc):
                terminated_value_error = True
                done = True
                try:
                    success_now = success_now or bool(env.check_success())
                except Exception:
                    pass
                break
            raise

        reward = float(reward)
        reward_sum += reward
        max_reward = max(max_reward, reward)

        if physical_steps % success_check_every == 0:
            try:
                timing_start = time.perf_counter()
                success_now = success_now or bool(env.check_success())
                add_timing(timing_totals, timing_counts, "success_check", timing_start)
            except Exception:
                pass
        if done or success_now:
            break

        gripper_actual_width = canonical_gripper_width(
            float(gripper_scalar(raw_obs)),
            max_physical_width=gripper_max_width,
        )
        gripper_width_error = abs(gripper_actual_width - float(gripper_target_width))
        # Stop early only after a minimum number of physical ticks and after the
        # gripper has received the requested short actuator window.  For closing
        # targets, gripper_width_error may stay nonzero because an object blocks
        # the fingers; that is good, because the positive actuator command is what
        # creates the physical grasp force.
        enough_gripper = active_gripper_steps >= required_active_gripper_steps
        gripper_reached_or_closing = gripper_width_error <= gripper_wait_tolerance or bool(gripper_should_close)
        if local_idx + 1 >= min_steps and pose_reached and enough_gripper and gripper_reached_or_closing:
            break

    timing_start = time.perf_counter()
    post_controller_world = current_controller_eef_world(env)
    add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)
    pose_pos_error, pose_rot_error = pose_error_to_target(post_controller_world, controller_target_world)
    gripper_actual_width = canonical_gripper_width(
        float(gripper_scalar(raw_obs)),
        max_physical_width=gripper_max_width,
    )
    gripper_width_error = abs(gripper_actual_width - float(gripper_target_width))

    grasp_status = {"any_contact": False, "any_grasped": False, "grasped_objects": [], "contact_objects": []}
    if bool(gripper_should_close):
        try:
            timing_start = time.perf_counter()
            grasp_status = gripper_grasp_status(env)
            add_timing(timing_totals, timing_counts, "gripper_grasp_check", timing_start)
        except Exception:
            pass

    if debug:
        print(
            "[debug] fast_physics "
            f"step={outer_step} budget={budget_steps} used={physical_steps} "
            f"pre_pos={pre_pose_pos_error:.4f} post_pos={pose_pos_error:.4f} "
            f"pre_rot={pre_pose_rot_error:.3f} post_rot={pose_rot_error:.3f} "
            f"grip_target={float(gripper_target_width):.4f} grip_actual={gripper_actual_width:.4f} "
            f"active_grip_steps={active_gripper_steps} done={done} success={success_now}",
            flush=True,
        )

    return {
        "raw_obs": raw_obs,
        "reward_sum": float(reward_sum),
        "max_reward": float(max_reward),
        "done": bool(done),
        "success_now": bool(success_now),
        "terminated_value_error": bool(terminated_value_error),
        "physical_steps": int(physical_steps),
        "budget_steps": int(budget_steps),
        "active_gripper_steps": int(active_gripper_steps),
        "last_libero_action": last_libero_action,
        "pose_pos_error": float(pose_pos_error),
        "pose_rot_error": float(pose_rot_error),
        "gripper_actual_width": float(gripper_actual_width),
        "gripper_width_error": float(gripper_width_error),
        "grasp_status": grasp_status,
    }

def run_episode(
    *,
    infer: SmolVLA_ModelInference,
    env: Any,
    task_language: str,
    init_state: np.ndarray,
    cfg: dict[str, Any],
    key_watcher: TerminalKeyWatcher | None = None,
) -> dict[str, Any]:
    control = cfg["control"]
    max_steps = control.get("max_steps")
    lifecycle_debug = bool(cfg.get("debug_episode_lifecycle", False))
    if lifecycle_debug:
        print("[debug] episode_lifecycle: env.reset begin", flush=True)
    env.reset()
    if lifecycle_debug:
        print("[debug] episode_lifecycle: env.reset done; set_init_state begin", flush=True)
    raw_obs = env.set_init_state(init_state)
    if lifecycle_debug:
        print("[debug] episode_lifecycle: set_init_state done", flush=True)
    warmup_steps = int(control.get("warmup_steps", 0))
    for _ in range(max(0, warmup_steps)):
        raw_obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, 0])
        try:
            base_env = unwrap_libero_env(env)
            if hasattr(base_env, "viewer") and hasattr(base_env.viewer, "render"):
                base_env.viewer.render()
        except Exception as e:
            print("[WARN] viewer sync failed:", repr(e))
    infer.policy.reset()
    infer.policy_reset()
    action_queue: list[np.ndarray] = []
    libero_actions: list[np.ndarray] = []
    pose_actions: list[np.ndarray] = []
    controller_pose_actions: list[np.ndarray] = []
    model_pose_actions: list[np.ndarray] = []
    pose_pos_errors: list[float] = []
    pose_rot_errors: list[float] = []
    pose_wait_flags: list[bool] = []
    gripper_targets: list[float] = []
    gripper_actuals: list[float] = []
    gripper_modes: list[str] = []
    gripper_width_errors: list[float] = []
    gripper_should_close_flags: list[bool] = []
    gripper_close_command_flags: list[bool] = []
    gripper_wait_flags: list[bool] = []
    gripper_contact_stall_flags: list[bool] = []
    gripper_rim_correction_flags: list[bool] = []
    gripper_rim_axis_shifts: list[float] = []
    gripper_rim_depth_shifts: list[float] = []
    gripper_rim_lifts: list[float] = []
    gripper_rim_point_counts: list[int] = []
    gripper_progress_indices: list[int] = []
    gripper_close_hold_flags: list[bool] = []
    gripper_grasp_flags: list[bool] = []
    gripper_contact_flags: list[bool] = []
    rewards: list[float] = []
    success_ever = False
    max_reward = 0.0
    timing_totals: dict[str, float] = defaultdict(float)
    timing_counts: dict[str, int] = defaultdict(int)
    episode_start_s = time.perf_counter()
    model_call_count = 0
    total_physical_env_steps = 0
    fast_physics_budget_steps: list[int] = []
    fast_physics_used_steps: list[int] = []
    fast_physics_active_gripper_steps: list[int] = []
    video_frames: dict[str, list[np.ndarray]] = {}
    max_steps = int(max_steps or getattr(env, "horizon", 500))
    action_index = max(0, int(control.get("action_index", 0)))
    control_mode = str(control.get("control_mode", "absolute_pose")).lower()
    if control_mode not in {"absolute_pose", "delta_pose"}:
        raise ValueError(f"Unsupported control_mode={control_mode!r}; expected 'absolute_pose' or 'delta_pose'.")
    use_absolute_pose_control = control_mode == "absolute_pose"
    execution_mode = str(control.get("execution_mode", "fast_physics")).lower()
    if execution_mode not in {"sync_wait", "fast_physics"}:
        raise ValueError(f"Unsupported execution_mode={execution_mode!r}; expected 'sync_wait' or 'fast_physics'.")
    use_fast_physics_execution = execution_mode == "fast_physics"
    for robot in env.robots:
        robot.controller.use_delta = not use_absolute_pose_control
    replan_every_step = bool(control.get("replan_every_step", False))
    pose_wait_until_reached = bool(control.get("pose_wait_until_reached", True))
    pose_wait_pos_tolerance = max(0.0, float(control.get("pose_wait_pos_tolerance", 0.015)))
    pose_wait_rot_tolerance = max(0.0, float(control.get("pose_wait_rot_tolerance", 0.25)))
    pose_wait_max_steps = max(0, int(control.get("pose_wait_max_steps", 8)))
    gripper_close_pose_wait_max_steps = max(
        0,
        int(control.get("gripper_close_pose_wait_max_steps", max(20, pose_wait_max_steps))),
    )
    gripper_wait_until_reached = bool(control.get("gripper_wait_until_reached", True))
    gripper_wait_tolerance = max(0.0, float(control.get("gripper_wait_tolerance", 0.004)))
    gripper_wait_max_steps = max(0, int(control.get("gripper_wait_max_steps", 12)))
    # Gripper execution follows libero_fast_instant_eval: the model action's
    # last dimension is a continuous physical target width, not an open/close
    # class.  Legacy threshold keys may still exist in old configs but are ignored.
    gripper_close_threshold: float | None = None
    gripper_contact_stall_tolerance = max(0.0, float(control.get("gripper_contact_stall_tolerance", 0.0007)))
    gripper_contact_stall_steps = max(1, int(control.get("gripper_contact_stall_steps", 4)))
    gripper_close_hold_steps = max(0, int(control.get("gripper_close_hold_steps", 0)))
    gripper_grasp_required_to_advance = bool(control.get("gripper_grasp_required_to_advance", False))
    gripper_grasp_wait_max_steps = max(0, int(control.get("gripper_grasp_wait_max_steps", 8)))
    rim_correction_keywords = _control_list(control, "gripper_rim_correction_task_keywords", ["bowl"])
    rim_correction_enable = bool(control.get("gripper_rim_correction_enable", True)) and task_matches_keywords(
        task_language,
        rim_correction_keywords,
    )
    rim_correction_min_points = max(1, int(control.get("gripper_rim_correction_min_points", 20)))
    rim_correction_axis_limit = max(0.0, float(control.get("gripper_rim_correction_axis_limit", 0.075)))
    rim_correction_depth_limit = max(0.0, float(control.get("gripper_rim_correction_depth_limit", 0.06)))
    rim_correction_z_min = float(control.get("gripper_rim_correction_z_min", -0.11))
    rim_correction_z_max = float(control.get("gripper_rim_correction_z_max", 0.04))
    rim_correction_top_quantile = float(control.get("gripper_rim_correction_top_quantile", 0.7))
    rim_correction_center_alpha = float(control.get("gripper_rim_correction_center_alpha", 0.8))
    rim_correction_max_axis_shift = max(0.0, float(control.get("gripper_rim_correction_max_axis_shift", 0.018)))
    rim_correction_max_depth_shift = max(0.0, float(control.get("gripper_rim_correction_max_depth_shift", 0.018)))
    rim_correction_lift = float(control.get("gripper_rim_correction_lift", 0.012))
    side_grasp_keywords = _control_list(control, "gripper_side_grasp_task_keywords", [])
    side_grasp_enable = task_matches_keywords(task_language, side_grasp_keywords)
    side_grasp_top_quantile = float(control.get("gripper_side_grasp_top_quantile", 0.5))
    gripper_max_width = float(cfg.get("gripper_qpos_max_width", 0.08))
    pc_camera_names = pointcloud_camera_names_from_config(cfg)
    save_video = bool(cfg.get("save_video", True))
    if save_video:
        append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
    render_onscreen_env(env, cfg, 0, force=True)
    pending_visualization = False
    latest_vis_chunk: np.ndarray | None = None
    latest_vis_point_cloud: np.ndarray | None = None
    keyboard_vis_mode = str(cfg.get("keyboard_vis_mode", "ply")).lower()
    keyboard_vis_output_dir = Path(cfg.get("output_dir", ".")).expanduser().resolve()
    planned_world_poses: np.ndarray | None = None
    planned_step_index = 0
    action_target_indices: list[int] = []
    pose_waiting = False
    pose_wait_steps = 0
    gripper_waiting = False
    gripper_wait_steps = 0
    gripper_close_stall_steps = 0
    gripper_close_hold_counter = 0
    gripper_grasp_wait_steps = 0
    last_waited_gripper_target_width = canonical_gripper_width(
        gripper_scalar(raw_obs),
        max_physical_width=gripper_max_width,
    )

    for step in range(max_steps):
        if total_physical_env_steps >= max_steps:
            break
        if key_watcher is not None and key_watcher.consume_visualize_request():
            pending_visualization = True
        need_model_point_cloud = (replan_every_step and not gripper_waiting and not pose_waiting) or not action_queue
        point_cloud: np.ndarray | None = None
        point_cloud_world: np.ndarray | None = None
        if need_model_point_cloud:
            timing_start = time.perf_counter()
            point_cloud, point_cloud_world, eef_pose = observation_to_point_clouds(
                env,
                raw_obs,
                pc_camera_names,
                int(cfg["observation_height"]),
                int(cfg["observation_width"]),
                int(cfg["num_points"]),
                seed=step,
            )
            add_timing(timing_totals, timing_counts, "pointcloud_from_obs", timing_start)
            if cfg.get("add_gripper_cloud", True):
                timing_start = time.perf_counter()
                point_cloud = add_world_gripper_cloud_to_point_cloud(
                    point_cloud_world,
                    eef_pose,
                    gripper_width_percent_from_scalar(
                        float(eef_pose[-1]),
                        max_physical_width=gripper_max_width,
                    ),
                    total_points=int(cfg["num_points"]),
                    gripper_points=int(cfg.get("gripper_points", 500)),
                    gripper_len=float(cfg.get("gripper_len", 0.06)),
                    gripper_template=str(cfg.get("gripper_template", "reap")),
                    seed=step,
                    drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
                    shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
                )
                add_timing(timing_totals, timing_counts, "pointcloud_add_gripper", timing_start)
        else:
            eef_pose = eef_pose9_gripper_from_obs(raw_obs)
        if need_model_point_cloud:
            if point_cloud is None:
                raise RuntimeError("Internal error: model inference requested without a point cloud.")
            timing_start = time.perf_counter()
            chunk = infer.predict_action_chunk_obs(
                {"point_cloud": point_cloud, "state": identity_pose9_gripper(float(eef_pose[-1]))},
                task=task_language,
                postprocess=True,
                state_pose_mode="identity",
            )[0].detach().cpu().numpy()
            add_timing(timing_totals, timing_counts, "model_predict_action_chunk", timing_start)
            model_call_count += 1
            latest_vis_chunk = chunk.copy()
            latest_vis_point_cloud = point_cloud.copy()
            selected_index = min(action_index, len(chunk) - 1)
            if replan_every_step:
                action_queue.clear()
                action_target_indices.clear()
                planned_chunk = chunk[selected_index : selected_index + 1].copy()
                action_queue.append(planned_chunk[0])
                action_target_indices.append(0)
            else:
                exec_action_steps = int(control.get("exec_action_steps", 16))
                if exec_action_steps <= 0:
                    exec_action_steps = len(chunk)
                end_index = min(len(chunk), selected_index + exec_action_steps)
                planned_chunk = chunk[selected_index:end_index].copy()

                # Do NOT convert the chunk to adjacent open-loop deltas.
                # The dataset / training action is a UMI trajectory; during eval we
                # closed-loop track each predicted target pose from the current EEF.
                action_queue.extend(list(planned_chunk))
                action_target_indices.extend(list(range(len(planned_chunk))))
            planned_world_poses = planned_world_poses_from_umi_chunk(eef_pose, planned_chunk)
            planned_step_index = 0
            pose_waiting = False
            pose_wait_steps = 0
            gripper_waiting = False
            gripper_wait_steps = 0
            gripper_close_stall_steps = 0
            gripper_close_hold_counter = 0
            gripper_grasp_wait_steps = 0

        if pending_visualization:
            if latest_vis_chunk is not None and latest_vis_point_cloud is not None:
                ply_path = save_keyboard_debug_visualization(
                    output_dir=keyboard_vis_output_dir,
                    step=step,
                    action_chunk=latest_vis_chunk,
                    point_cloud=latest_vis_point_cloud,
                )
                print(f"[debug] Saved latest UMI action chunk visualization to {ply_path}.")
                if keyboard_vis_mode == "window":
                    print("[debug] Opening Open3D window; use --keyboard-vis-mode ply in EGL/headless sessions.")
                    vis_umi_data(latest_vis_chunk, latest_vis_point_cloud)
                pending_visualization = False
            else:
                print("[debug] Visualization requested; waiting for the first predicted action chunk.")

        model_pose_action = np.asarray(action_queue[0], dtype=np.float32)
        model_pose_actions.append(model_pose_action.copy())

        target_idx = action_target_indices[0] if action_target_indices else planned_step_index
        pose_action = model_pose_action.copy()
        controller_pose_action = model_pose_action.copy()
        gripper_progress_idx = int(target_idx)

        current_world = pose9_to_homo_np(np.asarray(eef_pose, dtype=np.float32)[..., :9])
        timing_start = time.perf_counter()
        current_controller_world = current_controller_eef_world(env)
        add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)
        if planned_world_poses is not None and len(planned_world_poses) > 0:
            target_idx = min(int(target_idx), len(planned_world_poses) - 1)
            target_world = planned_world_poses[target_idx]
        else:
            target_world = current_world @ pose9_to_homo_np(pose_action[:9])

        gripper_predicted_width = float(pose_action[-1])
        gripper_target_width, gripper_mode = resolve_gripper_width_for_execution(
            gripper_predicted_width,
            mode=str(control.get("gripper_control_mode", "direct")),
            threshold=control.get("gripper_close_threshold"),
            open_width=control.get("gripper_open_width", gripper_max_width),
            close_width=control.get("gripper_close_width", 0.0),
            qpos_max_width=gripper_max_width,
        )
        # Keep pose_action's gripper slot equal to the value actually executed / logged.
        pose_action[-1] = gripper_target_width
        controller_pose_action[-1] = gripper_target_width
        gripper_target = gripper_target_width
        gripper_width_before = canonical_gripper_width(
            float(gripper_scalar(raw_obs)),
            max_physical_width=gripper_max_width,
        )
        gripper_command_raw = gripper_width_delta_action(
            gripper_target_width,
            gripper_width_before,
            max_physical_width=gripper_max_width,
        )
        # This flag now means "the continuous target asks for a narrower width";
        # it no longer comes from a task-specific threshold.
        gripper_should_close = gripper_command_raw > 0.0
        rim_correction_metrics = {
            "applied": False,
            "axis_shift": 0.0,
            "depth_shift": 0.0,
            "lift": 0.0,
            "point_count": 0,
        }
        if gripper_should_close:
            if point_cloud_world is None and rim_correction_enable:
                timing_start = time.perf_counter()
                point_cloud_world, _ = observation_to_world_point_cloud(
                    env,
                    raw_obs,
                    pc_camera_names,
                    int(cfg["observation_height"]),
                    int(cfg["observation_width"]),
                    int(cfg["num_points"]),
                    seed=step,
                )
                add_timing(timing_totals, timing_counts, "pointcloud_world_for_correction", timing_start)
            if point_cloud_world is not None:
                active_top_quantile = side_grasp_top_quantile if side_grasp_enable else rim_correction_top_quantile
                active_lift = 0.0 if side_grasp_enable else rim_correction_lift
                corrected_world, rim_correction_metrics = correct_rim_grasp_target_world(
                    target_world,
                    point_cloud_world,
                    enabled=rim_correction_enable,
                    min_points=rim_correction_min_points,
                    axis_limit=rim_correction_axis_limit,
                    depth_limit=rim_correction_depth_limit,
                    z_min=rim_correction_z_min,
                    z_max=rim_correction_z_max,
                    top_quantile=active_top_quantile,
                    center_alpha=rim_correction_center_alpha,
                    max_axis_shift=rim_correction_max_axis_shift,
                    max_depth_shift=rim_correction_max_depth_shift,
                    lift=active_lift,
                )
                if bool(rim_correction_metrics["applied"]):
                    target_world = corrected_world

        if use_absolute_pose_control:
            pose_action[:9] = matrix_to_pose9(target_world)
            controller_target_world = model_world_pose_to_controller_world(
                target_world,
                current_world,
                current_controller_world,
            )
            controller_pose_action[:9] = matrix_to_pose9(controller_target_world)
            motion_action = world_pose_to_libero_absolute_action(controller_target_world)
            hold_motion_action = world_pose_to_libero_absolute_action(current_controller_world)
        else:
            relative_to_current = fast_inverse_homogeneous(current_world) @ target_world
            pose_action[:9] = matrix_to_pose9(relative_to_current)
            controller_pose_action[:9] = pose_action[:9]
            controller_target_world = current_controller_world @ pose9_to_homo_np(controller_pose_action[:9])
            motion_action = pose9_to_libero_motion_action(
                pose_action,
                float(control["trans_scale"]),
                float(control["rot_scale"]),
                current_eef_pose9=eef_pose,
            )
            hold_motion_action = np.zeros(6, dtype=np.float32)

        # Continuous gripper execution: convert the postprocessed model target
        # width into robosuite's signed gripper command.  No threshold / binary
        # open-close mapping is applied.
        # Keep an un-gated target motion for fast_physics; the legacy sync_wait
        # branch below may replace motion_action with a hold command while it is
        # waiting for pose/gripper completion, but fast_physics performs its own
        # bounded physical servo and must track the actual model target.
        target_motion_action = motion_action.copy()
        gripper_command = gripper_command_raw

        # Pre-step completion check. This prevents the controller from repeatedly
        # sending the same absolute pose target after the arm has already arrived,
        # which is a common source of twitching in absolute-pose OSC control.
        pre_pose_pos_error, pre_pose_rot_error = pose_error_to_target(
            current_controller_world,
            controller_target_world,
        )
        pre_pose_reached = (
            pre_pose_pos_error <= pose_wait_pos_tolerance
            and pre_pose_rot_error <= pose_wait_rot_tolerance
        )
        pre_pose_timed_out = pose_wait_max_steps == 0 or pose_wait_steps >= pose_wait_max_steps
        pre_pose_done = (not pose_wait_until_reached) or pre_pose_reached or pre_pose_timed_out
        pre_close_pose_timed_out = (
            gripper_close_pose_wait_max_steps == 0 or pose_wait_steps >= gripper_close_pose_wait_max_steps
        )
        pre_pose_done_for_close = (
            (not pose_wait_until_reached) or pre_pose_reached or pre_close_pose_timed_out
        )
        pre_pose_ready = pre_pose_done_for_close if gripper_should_close else pre_pose_done

        pre_gripper_actual = float(gripper_scalar(raw_obs))
        pre_gripper_actual_width = canonical_gripper_width(
            pre_gripper_actual,
            max_physical_width=gripper_max_width,
        )
        pre_gripper_width_error = abs(pre_gripper_actual_width - gripper_target_width)
        pre_gripper_reached = pre_gripper_width_error <= gripper_wait_tolerance
        pre_gripper_contact_stalled = gripper_should_close and gripper_close_stall_steps >= gripper_contact_stall_steps
        pre_gripper_timed_out = gripper_wait_max_steps == 0 or gripper_wait_steps >= gripper_wait_max_steps
        pre_gripper_done = (
            (not gripper_wait_until_reached)
            or pre_gripper_reached
            or pre_gripper_contact_stalled
            or pre_gripper_timed_out
        )
        pre_grasp_status = {"any_grasped": False}
        if gripper_should_close and gripper_grasp_required_to_advance and pre_pose_ready and pre_gripper_done:
            timing_start = time.perf_counter()
            pre_grasp_status = gripper_grasp_status(env)
            add_timing(timing_totals, timing_counts, "gripper_grasp_check", timing_start)
        pre_close_hold_active = (
            gripper_should_close
            and pre_pose_ready
            and pre_gripper_done
            and gripper_close_hold_counter < gripper_close_hold_steps
        )
        pre_grasp_wait_active = (
            gripper_should_close
            and pre_pose_ready
            and pre_gripper_done
            and gripper_grasp_required_to_advance
            and not bool(pre_grasp_status.get("any_grasped", False))
            and gripper_grasp_wait_steps < gripper_grasp_wait_max_steps
        )

        # Target-level synchronization:
        # 1) For closing targets, do not close while still moving toward the pose.
        # 2) Once the pose is reached but the gripper is not, hold the arm pose and
        #    only actuate the gripper.
        # 3) In absolute-pose mode, holding the arm means commanding the current
        #    controller pose, NOT zeros. A zero absolute pose would pull the robot
        #    toward the world origin and create twitching.
        active_gripper_control = True
        if gripper_should_close and not pre_pose_ready:
            gripper_command = 0.0
            active_gripper_control = False
        if pre_pose_ready and not pre_gripper_done:
            motion_action = hold_motion_action.copy()
        if pre_close_hold_active or pre_grasp_wait_active:
            motion_action = hold_motion_action.copy()
            gripper_command = max(float(gripper_command_raw), 0.0)
            active_gripper_control = abs(gripper_command) > 1e-6
        elif pre_pose_ready and pre_gripper_done:
            motion_action = hold_motion_action.copy()
            gripper_command = 0.0
            active_gripper_control = False

        libero_action = np.concatenate([motion_action.astype(np.float32), np.asarray([gripper_command], dtype=np.float32)])

        if use_fast_physics_execution:
            physics_result = execute_fast_physics_action(
                env=env,
                cfg=cfg,
                raw_obs=raw_obs,
                motion_action=target_motion_action,
                controller_target_world=controller_target_world,
                gripper_target_width=gripper_target_width,
                gripper_should_close=bool(gripper_should_close),
                gripper_max_width=gripper_max_width,
                pose_wait_pos_tolerance=pose_wait_pos_tolerance,
                pose_wait_rot_tolerance=pose_wait_rot_tolerance,
                gripper_wait_tolerance=gripper_wait_tolerance,
                timing_totals=timing_totals,
                timing_counts=timing_counts,
                outer_step=step,
            )
            raw_obs = physics_result["raw_obs"]
            reward = float(physics_result["reward_sum"])
            done = bool(physics_result["done"])
            success_now = bool(physics_result["success_now"])
            total_physical_env_steps += int(physics_result["physical_steps"])
            fast_physics_budget_steps.append(int(physics_result["budget_steps"]))
            fast_physics_used_steps.append(int(physics_result["physical_steps"]))
            fast_physics_active_gripper_steps.append(int(physics_result["active_gripper_steps"]))
            libero_action = np.asarray(physics_result["last_libero_action"], dtype=np.float32)
            max_reward = max(max_reward, float(physics_result["max_reward"]))
            success_ever = success_ever or success_now
            success = success_now
            pose_pos_error = float(physics_result["pose_pos_error"])
            pose_rot_error = float(physics_result["pose_rot_error"])
            gripper_actual_width = float(physics_result["gripper_actual_width"])
            gripper_actual = float(gripper_scalar(raw_obs))
            gripper_width_error = float(physics_result["gripper_width_error"])
            grasp_status = physics_result.get("grasp_status", {}) or {}
            gripper_grasped = bool(grasp_status.get("any_grasped", False))
            gripper_contact_any = bool(grasp_status.get("any_contact", False))
            gripper_contact_stalled = False
            should_wait_pose = False
            should_wait_gripper = False
            should_hold_close = False
            should_wait_grasp = False
            pose_waiting = False
            gripper_waiting = False

            if action_queue:
                action_queue.pop(0)
            if action_target_indices:
                action_target_indices.pop(0)
            planned_step_index += 1
            gripper_wait_steps = 0
            pose_wait_steps = 0
            gripper_close_stall_steps = 0
            gripper_close_hold_counter = 0
            gripper_grasp_wait_steps = 0
            last_waited_gripper_target_width = gripper_target_width

            timing_start = time.perf_counter()
            render_onscreen_env(env, cfg, total_physical_env_steps)
            add_timing(timing_totals, timing_counts, "render", timing_start)
            if save_video:
                timing_start = time.perf_counter()
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
                add_timing(timing_totals, timing_counts, "video_frame_append", timing_start)
            libero_actions.append(libero_action)
            pose_actions.append(pose_action)
            controller_pose_actions.append(controller_pose_action)
            pose_pos_errors.append(pose_pos_error)
            pose_rot_errors.append(pose_rot_error)
            pose_wait_flags.append(False)
            gripper_targets.append(gripper_target)
            gripper_actuals.append(gripper_actual)
            gripper_modes.append(str(gripper_mode))
            gripper_width_errors.append(gripper_width_error)
            gripper_should_close_flags.append(bool(gripper_should_close))
            gripper_close_command_flags.append(bool(libero_action[-1] > 0.0))
            gripper_wait_flags.append(False)
            gripper_contact_stall_flags.append(bool(gripper_contact_stalled))
            gripper_rim_correction_flags.append(bool(rim_correction_metrics["applied"]))
            gripper_rim_axis_shifts.append(float(rim_correction_metrics["axis_shift"]))
            gripper_rim_depth_shifts.append(float(rim_correction_metrics["depth_shift"]))
            gripper_rim_lifts.append(float(rim_correction_metrics["lift"]))
            gripper_rim_point_counts.append(int(rim_correction_metrics["point_count"]))
            gripper_progress_indices.append(gripper_progress_idx)
            gripper_close_hold_flags.append(False)
            gripper_grasp_flags.append(bool(gripper_grasped))
            gripper_contact_flags.append(bool(gripper_contact_any))
            rewards.append(float(reward))
            if done or success or total_physical_env_steps >= max_steps:
                break
            continue

        try:
            timing_start = time.perf_counter()
            raw_obs, reward, done, _ = env.step(libero_action)
            add_timing(timing_totals, timing_counts, "env_step", timing_start)
            total_physical_env_steps += 1
        except ValueError as exc:
            if "terminated episode" in str(exc):
                # The env is already done. Do not count this as a policy failure.
                try:
                    success_ever = success_ever or bool(env.check_success())
                except Exception:
                    pass
                break
            raise

        reward = float(reward)
        max_reward = max(max_reward, reward)

        try:
            timing_start = time.perf_counter()
            success_now = bool(env.check_success())
            add_timing(timing_totals, timing_counts, "success_check", timing_start)
        except Exception:
            success_now = False

        success_ever = success_ever or success_now

        timing_start = time.perf_counter()
        post_controller_world = current_controller_eef_world(env)
        add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)
        pose_pos_error, pose_rot_error = pose_error_to_target(post_controller_world, controller_target_world)
        success = success_now

        pose_reached = pose_pos_error <= pose_wait_pos_tolerance and pose_rot_error <= pose_wait_rot_tolerance
        pose_timed_out = pose_wait_max_steps == 0 or pose_wait_steps >= pose_wait_max_steps
        pose_done = (not pose_wait_until_reached) or pose_reached or pose_timed_out
        close_pose_timed_out = (
            gripper_close_pose_wait_max_steps == 0 or pose_wait_steps >= gripper_close_pose_wait_max_steps
        )
        pose_done_for_close = (not pose_wait_until_reached) or pose_reached or close_pose_timed_out
        pose_ready = pose_done_for_close if gripper_should_close else pose_done

        gripper_actual = float(gripper_scalar(raw_obs))
        gripper_actual_width = canonical_gripper_width(
            gripper_actual,
            max_physical_width=gripper_max_width,
        )
        gripper_width_error = abs(gripper_actual_width - gripper_target_width)
        gripper_reached = gripper_width_error <= gripper_wait_tolerance
        gripper_close_progress = gripper_width_before - gripper_actual_width
        if active_gripper_control and gripper_should_close:
            if gripper_close_progress <= gripper_contact_stall_tolerance:
                gripper_close_stall_steps += 1
            else:
                gripper_close_stall_steps = 0
        elif not gripper_should_close:
            gripper_close_stall_steps = 0
        gripper_contact_stalled = gripper_should_close and gripper_close_stall_steps >= gripper_contact_stall_steps
        gripper_timed_out = gripper_wait_max_steps == 0 or gripper_wait_steps >= gripper_wait_max_steps
        gripper_done = (
            (not gripper_wait_until_reached)
            or gripper_reached
            or gripper_contact_stalled
            or gripper_timed_out
        )
        grasp_status = {"any_contact": False, "any_grasped": False, "grasped_objects": [], "contact_objects": []}
        if gripper_should_close:
            timing_start = time.perf_counter()
            grasp_status = gripper_grasp_status(env)
            add_timing(timing_totals, timing_counts, "gripper_grasp_check", timing_start)
        gripper_grasped = bool(grasp_status.get("any_grasped", False))
        gripper_contact_any = bool(grasp_status.get("any_contact", False))

        # Only count gripper waiting after the arm pose is done, or for open targets
        # that are safe to execute during motion. Closing is intentionally delayed
        # until pose_done to match the demonstrations' "move -> close -> move" rhythm.
        should_wait_pose = (
            pose_wait_until_reached
            and not done
            and not success
            and not pose_ready
        )
        should_wait_gripper = (
            gripper_wait_until_reached
            and not done
            and not success
            and not gripper_done
            and (pose_ready or not gripper_should_close)
        )
        should_hold_close = (
            gripper_should_close
            and pose_ready
            and gripper_done
            and not done
            and not success
            and gripper_close_hold_counter < gripper_close_hold_steps
        )
        should_wait_grasp = (
            gripper_should_close
            and pose_ready
            and gripper_done
            and gripper_grasp_required_to_advance
            and not gripper_grasped
            and not done
            and not success
            and gripper_grasp_wait_steps < gripper_grasp_wait_max_steps
        )
        if should_wait_pose:
            pose_waiting = True
            pose_wait_steps += 1
        else:
            pose_waiting = False

        if should_wait_gripper:
            gripper_waiting = True
            if active_gripper_control:
                gripper_wait_steps += 1
        elif should_hold_close or should_wait_grasp:
            gripper_waiting = True
        else:
            gripper_waiting = False
        if should_hold_close:
            gripper_close_hold_counter += 1
        if should_wait_grasp:
            gripper_grasp_wait_steps += 1

        if should_wait_gripper or should_wait_pose or should_hold_close or should_wait_grasp:
            pass
        else:
            action_queue.pop(0)
            if action_target_indices:
                action_target_indices.pop(0)
            planned_step_index += 1
            gripper_wait_steps = 0
            pose_wait_steps = 0
            gripper_close_stall_steps = 0
            gripper_close_hold_counter = 0
            gripper_grasp_wait_steps = 0
            last_waited_gripper_target_width = gripper_target_width
        timing_start = time.perf_counter()
        render_onscreen_env(env, cfg, step + 1)
        add_timing(timing_totals, timing_counts, "render", timing_start)
        if save_video:
            timing_start = time.perf_counter()
            append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
            add_timing(timing_totals, timing_counts, "video_frame_append", timing_start)
        libero_actions.append(libero_action)
        pose_actions.append(pose_action)
        controller_pose_actions.append(controller_pose_action)
        pose_pos_errors.append(pose_pos_error)
        pose_rot_errors.append(pose_rot_error)
        pose_wait_flags.append(bool(should_wait_pose))
        gripper_targets.append(gripper_target)
        gripper_actuals.append(gripper_actual)
        gripper_modes.append(str(gripper_mode))
        gripper_width_errors.append(gripper_width_error)
        gripper_should_close_flags.append(bool(gripper_should_close))
        gripper_close_command_flags.append(bool(libero_action[-1] > 0.0))
        gripper_wait_flags.append(bool(should_wait_gripper))
        gripper_contact_stall_flags.append(bool(gripper_contact_stalled))
        gripper_rim_correction_flags.append(bool(rim_correction_metrics["applied"]))
        gripper_rim_axis_shifts.append(float(rim_correction_metrics["axis_shift"]))
        gripper_rim_depth_shifts.append(float(rim_correction_metrics["depth_shift"]))
        gripper_rim_lifts.append(float(rim_correction_metrics["lift"]))
        gripper_rim_point_counts.append(int(rim_correction_metrics["point_count"]))
        gripper_progress_indices.append(gripper_progress_idx)
        gripper_close_hold_flags.append(bool(should_hold_close))
        gripper_grasp_flags.append(bool(gripper_grasped))
        gripper_contact_flags.append(bool(gripper_contact_any))
        rewards.append(float(reward))
        if done or success:
            break

    final_goal_status = goal_predicate_status(env)
    if str(cfg.get("render_mode", "offscreen")).lower() in {"viewer3d", "mujoco"}:
        close_mujoco_viewer_safely(env)
    wall_s = time.perf_counter() - episode_start_s
    return {
        "success": bool(success_ever),
        "steps": int(total_physical_env_steps if total_physical_env_steps > 0 else step + 1),
        "action_rows_executed": int(len(libero_actions)),
        "execution_mode": str(execution_mode),
        "fast_physics_budget_steps": np.asarray(fast_physics_budget_steps, dtype=np.int64),
        "fast_physics_used_steps": np.asarray(fast_physics_used_steps, dtype=np.int64),
        "fast_physics_active_gripper_steps": np.asarray(fast_physics_active_gripper_steps, dtype=np.int64),
        "model_call_count": int(model_call_count),
        "sum_reward": float(np.sum(rewards)),
        "max_reward": float(max_reward),
        "timings": summarize_timings(timing_totals, timing_counts, wall_s=wall_s),
        "goal_predicates_final": final_goal_status,
        "gripper_execution_mode": "model_action_last_dim",
        "gripper_thresholding_disabled": True,
        "gripper_close_threshold": None,
        "gripper_side_grasp_enable": bool(side_grasp_enable),
        "gripper_close_pose_wait_max_steps": int(gripper_close_pose_wait_max_steps),
        "gripper_close_target_steps": int(np.sum(gripper_should_close_flags)),
        "gripper_close_command_steps": int(np.sum(gripper_close_command_flags)),
        "gripper_close_hold_steps": int(np.sum(gripper_close_hold_flags)),
        "gripper_grasp_detected_any": bool(any(gripper_grasp_flags)),
        "gripper_grasp_detected_steps": int(np.sum(gripper_grasp_flags)),
        "gripper_contact_detected_any": bool(any(gripper_contact_flags)),
        "gripper_contact_detected_steps": int(np.sum(gripper_contact_flags)),
        "libero_actions": np.asarray(libero_actions, dtype=np.float32),
        "pose_actions": np.asarray(pose_actions, dtype=np.float32),
        "controller_pose_actions": np.asarray(controller_pose_actions, dtype=np.float32),
        "model_pose_actions": np.asarray(model_pose_actions, dtype=np.float32),
        "pose_pos_errors": np.asarray(pose_pos_errors, dtype=np.float32),
        "pose_rot_errors": np.asarray(pose_rot_errors, dtype=np.float32),
        "pose_wait_flags": np.asarray(pose_wait_flags, dtype=bool),
        "gripper_targets": np.asarray(gripper_targets, dtype=np.float32),
        "gripper_actuals": np.asarray(gripper_actuals, dtype=np.float32),
        "gripper_modes": np.asarray(gripper_modes, dtype=object),
        "gripper_width_errors": np.asarray(gripper_width_errors, dtype=np.float32),
        "gripper_should_close_flags": np.asarray(gripper_should_close_flags, dtype=bool),
        "gripper_close_command_flags": np.asarray(gripper_close_command_flags, dtype=bool),
        "gripper_wait_flags": np.asarray(gripper_wait_flags, dtype=bool),
        "gripper_contact_stall_flags": np.asarray(gripper_contact_stall_flags, dtype=bool),
        "gripper_rim_correction_flags": np.asarray(gripper_rim_correction_flags, dtype=bool),
        "gripper_rim_axis_shifts": np.asarray(gripper_rim_axis_shifts, dtype=np.float32),
        "gripper_rim_depth_shifts": np.asarray(gripper_rim_depth_shifts, dtype=np.float32),
        "gripper_rim_lifts": np.asarray(gripper_rim_lifts, dtype=np.float32),
        "gripper_rim_point_counts": np.asarray(gripper_rim_point_counts, dtype=np.int64),
        "gripper_progress_indices": np.asarray(gripper_progress_indices, dtype=np.int64),
        "gripper_close_hold_flags": np.asarray(gripper_close_hold_flags, dtype=bool),
        "gripper_grasp_flags": np.asarray(gripper_grasp_flags, dtype=bool),
        "gripper_contact_flags": np.asarray(gripper_contact_flags, dtype=bool),
        "video_frames": video_frames,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["policy_path"] = cfg_get(cfg, args.policy_path, "policy_path")
    cfg["policy_repo_id"] = args.policy_repo_id if args.policy_repo_id is not None else cfg.get("policy_repo_id")
    suite_names = resolve_suite_names(args.suite, cfg)
    cfg["suites"] = suite_names
    cfg["all_tasks"] = bool(args.all_tasks if args.all_tasks is not None else cfg.get("all_tasks", False))
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes"))
    cfg["device"] = cfg_get(cfg, args.device, "device")
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points"))
    cfg["save_video"] = bool(args.save_video)
    if args.render_mode is not None:
        cfg["render_mode"] = str(args.render_mode)
    if args.headed is not None:
        cfg["render_mode"] = "viewer3d" if bool(args.headed) else "offscreen"
    if args.render_camera is not None:
        cfg["render_camera"] = str(args.render_camera)
    if args.render_every_n_steps is not None:
        cfg["render_every_n_steps"] = int(args.render_every_n_steps)
    if args.render_gpu_device_id is not None:
        cfg["render_gpu_device_id"] = int(args.render_gpu_device_id)
    render_mode_for_lifecycle = str(cfg.get("render_mode", "offscreen")).lower()
    if args.recreate_env_per_episode is not None:
        cfg["recreate_env_per_episode"] = bool(args.recreate_env_per_episode)
    else:
        # MuJoCo passive viewer / GL contexts are safest when a LIBERO env is not
        # reset and reused across episodes.  This mirrors libero_fast_instant_eval's
        # default and avoids episode-1 hangs in viewer3d mode.
        cfg["recreate_env_per_episode"] = bool(
            cfg.get("recreate_env_per_episode", render_mode_for_lifecycle in {"viewer3d", "mujoco"})
        )
    if args.debug_episode_lifecycle is not None:
        cfg["debug_episode_lifecycle"] = bool(args.debug_episode_lifecycle)
    else:
        cfg["debug_episode_lifecycle"] = bool(
            cfg.get("debug_episode_lifecycle", render_mode_for_lifecycle in {"viewer3d", "mujoco"})
        )
    configure_mujoco_render_backend(cfg)
    cfg.setdefault("control", {})
    for deprecated_gripper_key in (
        "gripper_threshold",
        "gripper_control_mode",
        "gripper_width_deadband",
        "gripper_width_gain",
        "gripper_smooth_window",
        "gripper_progress_lookahead",
        "gripper_close_threshold",
        "gripper_close_threshold_rules",
    ):
        cfg["control"].pop(deprecated_gripper_key, None)
    cfg["control"].setdefault("gripper_wait_until_reached", True)
    cfg["control"].setdefault("gripper_wait_tolerance", 0.004)
    cfg["control"].setdefault("gripper_wait_max_steps", 12)
    cfg["control"].setdefault("control_mode", "absolute_pose")
    cfg["control"].setdefault("pose_wait_until_reached", True)
    cfg["control"].setdefault("pose_wait_pos_tolerance", 0.015)
    cfg["control"].setdefault("pose_wait_rot_tolerance", 0.25)
    cfg["control"].setdefault("pose_wait_max_steps", 8)
    cfg["control"].setdefault("gripper_close_pose_wait_max_steps", 20)
    # Gripper threshold/rule defaults intentionally removed: execution uses
    # the model action's last dimension directly as a continuous width target.
    cfg["control"].setdefault("gripper_contact_stall_tolerance", 0.0007)
    cfg["control"].setdefault("gripper_contact_stall_steps", 4)
    cfg["control"].setdefault("gripper_close_hold_steps", 8)
    cfg["control"].setdefault("gripper_grasp_required_to_advance", False)
    cfg["control"].setdefault("gripper_grasp_wait_max_steps", 8)
    cfg["control"].setdefault("gripper_rim_correction_enable", True)
    cfg["control"].setdefault("gripper_rim_correction_task_keywords", ["bowl"])
    cfg["control"].setdefault("gripper_rim_correction_min_points", 20)
    cfg["control"].setdefault("gripper_rim_correction_axis_limit", 0.075)
    cfg["control"].setdefault("gripper_rim_correction_depth_limit", 0.06)
    cfg["control"].setdefault("gripper_rim_correction_z_min", -0.11)
    cfg["control"].setdefault("gripper_rim_correction_z_max", 0.04)
    cfg["control"].setdefault("gripper_rim_correction_top_quantile", 0.7)
    cfg["control"].setdefault("gripper_rim_correction_center_alpha", 0.8)
    cfg["control"].setdefault("gripper_rim_correction_max_axis_shift", 0.018)
    cfg["control"].setdefault("gripper_rim_correction_max_depth_shift", 0.018)
    cfg["control"].setdefault("gripper_rim_correction_lift", 0.012)
    cfg["control"].setdefault(
        "gripper_side_grasp_task_keywords",
        ["bottle", "sauce", "milk", "juice", "dressing", "ketchup", "soup", "pudding"],
    )
    cfg["control"].setdefault("gripper_side_grasp_top_quantile", 0.5)
    # Fast physical execution is the default for this evaluator: each model
    # action row is tracked by a short adaptive robosuite/MuJoCo OSC rollout,
    # instead of two serial pose/gripper wait loops.  It still uses env.step(),
    # so object contacts, grasping, and release remain physical.
    cfg["control"].setdefault("execution_mode", "fast_physics")
    cfg["control"].setdefault("fast_physics_min_steps", 2)
    cfg["control"].setdefault("fast_physics_max_steps", 12)
    cfg["control"].setdefault("fast_physics_max_duration_s", 0.0)
    cfg["control"].setdefault("fast_physics_pos_step_m", 0.012)
    cfg["control"].setdefault("fast_physics_rot_step_rad", 0.18)
    cfg["control"].setdefault("fast_physics_close_steps", 6)
    cfg["control"].setdefault("fast_physics_open_steps", 2)
    cfg["control"].setdefault("fast_physics_close_after_pose", True)
    cfg["control"].setdefault("fast_physics_success_check_every", 1)
    cfg["control"].setdefault("fast_physics_debug", False)
    if args.control_freq is not None:
        cfg["control"]["control_freq"] = float(args.control_freq)
    if args.action_index is not None:
        cfg["control"]["action_index"] = int(args.action_index)
    if args.exec_action_steps is not None:
        cfg["control"]["exec_action_steps"] = int(args.exec_action_steps)
    if args.control_mode is not None:
        cfg["control"]["control_mode"] = str(args.control_mode)
    if args.replan_every_step is not None:
        cfg["control"]["replan_every_step"] = bool(args.replan_every_step)
    if args.warmup_steps is not None:
        cfg["control"]["warmup_steps"] = int(args.warmup_steps)
    if args.pose_wait_until_reached is not None:
        cfg["control"]["pose_wait_until_reached"] = bool(args.pose_wait_until_reached)
    if args.pose_wait_pos_tolerance is not None:
        cfg["control"]["pose_wait_pos_tolerance"] = float(args.pose_wait_pos_tolerance)
    if args.pose_wait_rot_tolerance is not None:
        cfg["control"]["pose_wait_rot_tolerance"] = float(args.pose_wait_rot_tolerance)
    if args.pose_wait_max_steps is not None:
        cfg["control"]["pose_wait_max_steps"] = int(args.pose_wait_max_steps)
    if args.gripper_close_pose_wait_max_steps is not None:
        cfg["control"]["gripper_close_pose_wait_max_steps"] = int(args.gripper_close_pose_wait_max_steps)
    if args.gripper_wait_until_reached is not None:
        cfg["control"]["gripper_wait_until_reached"] = bool(args.gripper_wait_until_reached)
    if args.gripper_wait_tolerance is not None:
        cfg["control"]["gripper_wait_tolerance"] = float(args.gripper_wait_tolerance)
    if args.gripper_wait_max_steps is not None:
        cfg["control"]["gripper_wait_max_steps"] = int(args.gripper_wait_max_steps)
    if args.gripper_close_threshold is not None:
        print("[info] --gripper-close-threshold is accepted for compatibility but ignored; gripper uses model action last dim directly.")
    if args.gripper_contact_stall_tolerance is not None:
        cfg["control"]["gripper_contact_stall_tolerance"] = float(args.gripper_contact_stall_tolerance)
    if args.gripper_contact_stall_steps is not None:
        cfg["control"]["gripper_contact_stall_steps"] = int(args.gripper_contact_stall_steps)
    if args.gripper_close_hold_steps is not None:
        cfg["control"]["gripper_close_hold_steps"] = int(args.gripper_close_hold_steps)
    if args.gripper_grasp_required_to_advance is not None:
        cfg["control"]["gripper_grasp_required_to_advance"] = bool(args.gripper_grasp_required_to_advance)
    if args.gripper_grasp_wait_max_steps is not None:
        cfg["control"]["gripper_grasp_wait_max_steps"] = int(args.gripper_grasp_wait_max_steps)
    if args.execution_mode is not None:
        cfg["control"]["execution_mode"] = str(args.execution_mode)
    if args.fast_physics_min_steps is not None:
        cfg["control"]["fast_physics_min_steps"] = int(args.fast_physics_min_steps)
    if args.fast_physics_max_steps is not None:
        cfg["control"]["fast_physics_max_steps"] = int(args.fast_physics_max_steps)
    if args.fast_physics_max_duration_s is not None:
        cfg["control"]["fast_physics_max_duration_s"] = float(args.fast_physics_max_duration_s)
    if args.fast_physics_pos_step_m is not None:
        cfg["control"]["fast_physics_pos_step_m"] = float(args.fast_physics_pos_step_m)
    if args.fast_physics_rot_step_rad is not None:
        cfg["control"]["fast_physics_rot_step_rad"] = float(args.fast_physics_rot_step_rad)
    if args.fast_physics_close_steps is not None:
        cfg["control"]["fast_physics_close_steps"] = int(args.fast_physics_close_steps)
    if args.fast_physics_open_steps is not None:
        cfg["control"]["fast_physics_open_steps"] = int(args.fast_physics_open_steps)
    if args.fast_physics_close_after_pose is not None:
        cfg["control"]["fast_physics_close_after_pose"] = bool(args.fast_physics_close_after_pose)
    if args.fast_physics_success_check_every is not None:
        cfg["control"]["fast_physics_success_check_every"] = int(args.fast_physics_success_check_every)
    if args.fast_physics_debug is not None:
        cfg["control"]["fast_physics_debug"] = bool(args.fast_physics_debug)
    if args.keyboard_vis is not None:
        cfg["keyboard_vis"] = bool(args.keyboard_vis)
    if args.keyboard_vis_mode is not None:
        cfg["keyboard_vis_mode"] = str(args.keyboard_vis_mode)
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
    output_dir = Path(args.output_dir or cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    from libero.libero import benchmark

    infer = SmolVLA_ModelInference(
        policy_path=cfg["policy_path"],
        policy_repo_id=cfg.get("policy_repo_id"),
        device=cfg["device"],
    )
    print(
        f"[info] execution_mode={cfg['control'].get('execution_mode')} "
        f"fast_physics_min/max={cfg['control'].get('fast_physics_min_steps')}/{cfg['control'].get('fast_physics_max_steps')} "
        f"close_steps={cfg['control'].get('fast_physics_close_steps')} "
        f"open_steps={cfg['control'].get('fast_physics_open_steps')}"
    )
    print(
        f"[info] gripper_execution=model_action_last_dim_direct "
        f"qpos_max_width={float(cfg.get('gripper_qpos_max_width', 0.08)):.4f}; "
        "legacy threshold/open/close args ignored"
    )
    print(
        f"[info] episode_lifecycle recreate_env_per_episode={bool(cfg.get('recreate_env_per_episode', False))} "
        f"debug_episode_lifecycle={bool(cfg.get('debug_episode_lifecycle', False))}"
    )

    all_results = []
    benchmark_dict = benchmark.get_benchmark_dict()
    with TerminalKeyWatcher(enabled=bool(cfg.get("keyboard_vis", True))) as key_watcher:
        for suite_name in suite_names:
            suite_cls = benchmark_dict[suite_name]
            suite = suite_cls()
            task_ids = resolve_task_ids_for_suite(
                suite_name=suite_name,
                task_count=len(suite.tasks),
                cli_task_ids=args.task_id,
                cfg=cfg,
            )
            for task_id in task_ids:
                init_states = get_task_init_states(suite, int(task_id))
                task_results = []
                task_name = f"task_{int(task_id):03d}"
                task_language = f"{suite_name}:{int(task_id)}"
                recreate_env_per_episode = bool(cfg.get("recreate_env_per_episode", False))

                def _make_env_for_task_episode() -> tuple[Any, Any]:
                    if bool(cfg.get("debug_episode_lifecycle", False)):
                        print(
                            f"[debug] episode_lifecycle: make_env begin suite={suite_name} task={int(task_id)}",
                            flush=True,
                        )
                    env_obj, task_obj = make_libero_env(
                        suite,
                        int(task_id),
                        int(cfg["observation_height"]),
                        int(cfg["observation_width"]),
                        render_camera_names_from_config(cfg),
                        render_mode=str(cfg.get("render_mode", "offscreen")),
                        render_camera=str(cfg.get("render_camera", "agentview")),
                        render_gpu_device_id=int(cfg.get("render_gpu_device_id", -1)),
                        control_delta=str(cfg.get("control", {}).get("control_mode", "absolute_pose")).lower()
                        == "delta_pose",
                        control_freq=float(cfg.get("control_freq", cfg.get("control", {}).get("control_freq", 5))),
                    )
                    if bool(cfg.get("debug_episode_lifecycle", False)):
                        print(
                            f"[debug] episode_lifecycle: make_env done suite={suite_name} task={int(task_id)}",
                            flush=True,
                        )
                    return env_obj, task_obj

                def _write_success_and_reports(
                    *,
                    result: dict[str, Any],
                    episode_idx: int,
                    episode_dir: Path,
                    current_task_name: str,
                    current_task_language: str,
                ) -> None:
                    record = {
                        "episode_index": episode_idx,
                        "demo_name": "rollout",
                        "video_dir_name": episode_dir.name,
                    }
                    video_paths = export_episode_videos(result, episode_dir.parent, record, cfg)
                    episode_record = episode_result_record(result, episode_idx)
                    if video_paths:
                        episode_record["videos"] = video_paths
                    write_json_atomic(episode_dir / "result.json", episode_record)
                    task_results.append(episode_record)
                    current_results = [
                        *all_results,
                        make_task_summary(
                            suite_name=suite_name,
                            task_id=int(task_id),
                            task_name=current_task_name,
                            task_language=current_task_language,
                            task_results=task_results,
                        ),
                    ]
                    write_eval_reports(
                        output_dir=output_dir,
                        cfg=cfg,
                        suite_names=suite_names,
                        all_results=current_results,
                    )
                    print(
                        f"{suite_name} task={task_id} episode={episode_idx} "
                        f"success={result['success']} "
                        f"steps={result['steps']} "
                        f"model_calls={result['model_call_count']} "
                        f"sum_reward={result['sum_reward']:.3f} "
                        f"max_reward={result['max_reward']:.3f}",
                        flush=True,
                    )

                def _write_failure_and_reports(
                    *,
                    exc: Exception,
                    episode_idx: int,
                    episode_dir: Path,
                    current_task_name: str,
                    current_task_language: str,
                ) -> None:
                    failure = {
                        "episode_index": int(episode_idx),
                        "success": False,
                        "steps": 0,
                        "sum_reward": 0.0,
                        "error": repr(exc),
                    }
                    task_results.append(failure)
                    with open(episode_dir / "error.json", "w", encoding="utf-8") as f:
                        json.dump(failure, f, indent=2, ensure_ascii=False)
                    write_json_atomic(episode_dir / "result.json", failure)
                    current_results = [
                        *all_results,
                        make_task_summary(
                            suite_name=suite_name,
                            task_id=int(task_id),
                            task_name=current_task_name,
                            task_language=current_task_language,
                            task_results=task_results,
                        ),
                    ]
                    write_eval_reports(
                        output_dir=output_dir,
                        cfg=cfg,
                        suite_names=suite_names,
                        all_results=current_results,
                    )
                    print(
                        f"[WARN] {suite_name} task={task_id} episode={episode_idx} failed: {exc!r}. "
                        "Continuing with the next episode/task.",
                        flush=True,
                    )

                if recreate_env_per_episode:
                    if bool(cfg.get("debug_episode_lifecycle", False)):
                        print(
                            f"[debug] episode_lifecycle: recreating env per episode for suite={suite_name} task={int(task_id)}",
                            flush=True,
                        )
                    for episode_idx in range(cfg["episodes"]):
                        episode_dir = output_dir / suite_name / f"task_{int(task_id):03d}" / f"episode_{episode_idx:03d}"
                        episode_dir.mkdir(parents=True, exist_ok=True)
                        env = None
                        task = None
                        print(f"[eval] start {suite_name} task={task_id} episode={episode_idx}", flush=True)
                        try:
                            env, task = _make_env_for_task_episode()
                            task_name = str(getattr(task, "name", task_name))
                            task_language = str(getattr(task, "language", task_language))
                            result = run_episode(
                                infer=infer,
                                env=env,
                                task_language=task_language,
                                init_state=init_states[episode_idx % len(init_states)],
                                cfg=cfg,
                                key_watcher=key_watcher,
                            )
                            _write_success_and_reports(
                                result=result,
                                episode_idx=episode_idx,
                                episode_dir=episode_dir,
                                current_task_name=task_name,
                                current_task_language=task_language,
                            )
                        except Exception as exc:
                            _write_failure_and_reports(
                                exc=exc,
                                episode_idx=episode_idx,
                                episode_dir=episode_dir,
                                current_task_name=task_name,
                                current_task_language=task_language,
                            )
                        finally:
                            if bool(cfg.get("debug_episode_lifecycle", False)):
                                print(
                                    f"[debug] episode_lifecycle: close_env begin suite={suite_name} task={int(task_id)} episode={episode_idx}",
                                    flush=True,
                                )
                            try:
                                close_mujoco_viewer_safely(env)
                            except Exception:
                                pass
                            try:
                                if env is not None:
                                    env.close()
                            except Exception as exc:
                                print(f"[WARN] failed to close env for {suite_name} task={task_id}: {exc!r}", flush=True)
                            if bool(cfg.get("debug_episode_lifecycle", False)):
                                print(
                                    f"[debug] episode_lifecycle: close_env done suite={suite_name} task={int(task_id)} episode={episode_idx}",
                                    flush=True,
                                )
                    all_results.append(
                        make_task_summary(
                            suite_name=suite_name,
                            task_id=int(task_id),
                            task_name=task_name,
                            task_language=task_language,
                            task_results=task_results,
                        )
                    )
                    continue

                env = None
                task = None
                try:
                    env, task = _make_env_for_task_episode()
                    task_name = str(getattr(task, "name", task_name))
                    task_language = str(getattr(task, "language", task_language))
                    for episode_idx in range(cfg["episodes"]):
                        episode_dir = output_dir / suite_name / f"task_{int(task_id):03d}" / f"episode_{episode_idx:03d}"
                        episode_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[eval] start {suite_name} task={task_id} episode={episode_idx}", flush=True)
                        try:
                            result = run_episode(
                                infer=infer,
                                env=env,
                                task_language=task_language,
                                init_state=init_states[episode_idx % len(init_states)],
                                cfg=cfg,
                                key_watcher=key_watcher,
                            )
                            _write_success_and_reports(
                                result=result,
                                episode_idx=episode_idx,
                                episode_dir=episode_dir,
                                current_task_name=task_name,
                                current_task_language=task_language,
                            )
                        except Exception as exc:
                            _write_failure_and_reports(
                                exc=exc,
                                episode_idx=episode_idx,
                                episode_dir=episode_dir,
                                current_task_name=task_name,
                                current_task_language=task_language,
                            )
                finally:
                    try:
                        close_mujoco_viewer_safely(env)
                    except Exception:
                        pass
                    try:
                        if env is not None:
                            env.close()
                    except Exception as exc:
                        print(f"[WARN] failed to close env for {suite_name} task={task_id}: {exc!r}", flush=True)
                all_results.append(
                    make_task_summary(
                        suite_name=suite_name,
                        task_id=int(task_id),
                        task_name=task_name,
                        task_language=task_language,
                        task_results=task_results,
                    )
                )

    summary = write_eval_reports(
        output_dir=output_dir,
        cfg=cfg,
        suite_names=suite_names,
        all_results=all_results,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()