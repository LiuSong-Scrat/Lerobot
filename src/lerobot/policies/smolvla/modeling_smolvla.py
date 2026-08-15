#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import (
    SmolVLMWithExpertModel,
    load_pretrained_action_expert_weights,
)
from lerobot.policies.smolvla.song_pointseg import (
    MOTION_PRIOR_DIM,
    ROLE_FOREGROUND,
    PseudoLabelConfig,
    SongPointSegLoss,
    SongPointSegLossConfig,
    SongPointSegNet,
    generate_pseudo_labels_from_priors,
    infer_litept_output_channels,
    invert_transform,
    matrix_to_pose9,
    pose9_to_matrix,
)
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.utils import get_safe_dtype

from .litept.model import LitePT


SYMMETRIC_MULTIVIEW_WORLD_POINT_PREFIXES = (
    "model.worldflow_branch.scene_encoder.",
    "model.worldflow_branch.scene_context_proj.",
    "model.worldflow_branch.point_action_adapter.",
    "model.ego_scene_to_expert.",
    "model.world_scene_to_expert.",
)


@contextmanager
def _batchnorm_eval_on_single_value(module: nn.Module):
    batchnorm_modules = [
        child for child in module.modules() if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    training_states = [child.training for child in batchnorm_modules]
    handles = []

    def pre_hook(child: nn.modules.batchnorm._BatchNorm, inputs: tuple[Tensor, ...]) -> None:
        if not inputs:
            return
        input_tensor = inputs[0]
        if not torch.is_tensor(input_tensor) or input_tensor.ndim < 2 or input_tensor.shape[1] <= 0:
            return
        values_per_channel = input_tensor.numel() // int(input_tensor.shape[1])
        use_running_stats = (
            child.training
            and values_per_channel <= 1
            and child.running_mean is not None
            and child.running_var is not None
        )
        child._song_restore_training = child.training
        child._song_force_eval = use_running_stats
        if use_running_stats:
            child.eval()

    def post_hook(child: nn.modules.batchnorm._BatchNorm, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        if bool(getattr(child, "_song_force_eval", False)):
            child.train(bool(getattr(child, "_song_restore_training", True)))
        for attr in ("_song_force_eval", "_song_restore_training"):
            if hasattr(child, attr):
                delattr(child, attr)

    try:
        for child in batchnorm_modules:
            handles.append(child.register_forward_pre_hook(pre_hook))
            handles.append(child.register_forward_hook(post_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for child, was_training in zip(batchnorm_modules, training_states, strict=True):
            child.train(was_training)
            for attr in ("_song_force_eval", "_song_restore_training"):
                if hasattr(child, attr):
                    delattr(child, attr)


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


def transform_se3_twist(twist: Tensor, transform: Tensor) -> Tensor:
    """Apply the SE(3) adjoint for twists ordered as ``[v, omega]``."""

    if twist.shape[-1] != 6 or transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected twist (...,6) and transform (...,4,4), got {twist.shape}, {transform.shape}.")
    rotation = transform[..., :3, :3].to(device=twist.device, dtype=twist.dtype)
    translation = transform[..., :3, 3].to(device=twist.device, dtype=twist.dtype)
    omega = torch.matmul(rotation, twist[..., 3:6].unsqueeze(-1)).squeeze(-1)
    linear = torch.matmul(rotation, twist[..., :3].unsqueeze(-1)).squeeze(-1)
    linear = linear + torch.cross(translation, omega, dim=-1)
    return torch.cat([linear, omega], dim=-1)


def symmetric_world_ego_twist_fusion(
    ego_velocity: Tensor,
    world_twist: Tensor,
    world_to_ego_transform: Tensor,
) -> Tensor:
    """Equally fuse conjugate Ego/World twists while preserving Ego gripper."""

    if ego_velocity.shape[:-1] != world_twist.shape[:-1] or ego_velocity.shape[-1] < 7:
        raise ValueError(
            f"Expected matching Ego (...,>=7) and World (...,6), got {ego_velocity.shape}, {world_twist.shape}."
        )
    world_in_ego = transform_se3_twist(world_twist, world_to_ego_transform)
    fused_twist = 0.5 * (ego_velocity[..., :6] + world_in_ego)
    return torch.cat([fused_twist, ego_velocity[..., 6:]], dim=-1)


def se3_geodesic_flow_state(
    noise_transform: Tensor,
    target_transform: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the SE(3) geodesic state and its left-trivialized velocity.

    The path is ``H(t) = Exp(t Log(H1 H0^-1)) H0``.  Both returned tensors
    stay in physical units: ``H(t)`` is a valid rigid transform and the
    velocity is a 6D twist in metres/radians per unit flow time.
    """

    if noise_transform.shape != target_transform.shape or noise_transform.shape[-2:] != (4, 4):
        raise ValueError(
            "SE(3) flow endpoints must have matching (...,4,4) shapes; "
            f"got {noise_transform.shape} and {target_transform.shape}."
        )
    if time.ndim != 1 or time.shape[0] != noise_transform.shape[0]:
        raise ValueError(
            f"Expected time shape ({noise_transform.shape[0]},), got {time.shape}."
        )
    relative = target_transform @ invert_transform(noise_transform)
    total_twist = se3_log(relative)
    time_expanded = time.reshape(time.shape[0], *([1] * (total_twist.ndim - 1)))
    state = se3_exp(time_expanded * total_twist) @ noise_transform
    remaining = (1.0 - time).clamp_min(1e-4)
    remaining_expanded = remaining.reshape(remaining.shape[0], *([1] * (total_twist.ndim - 1)))
    velocity = se3_log(target_transform @ invert_transform(state)) / remaining_expanded
    return state, velocity


def pose9_velocity_to_spatial_twist(current_pose9: Tensor, pose9_velocity: Tensor) -> Tensor:
    """Project an instantaneous pose9 derivative onto a spatial SE(3) twist.

    ``pose9_velocity`` stores ``[p_dot, r1_dot, r2_dot]`` where ``r1`` and
    ``r2`` are the first two columns of the current rotation.  The closest
    tangent rotation derivative is obtained from the skew part of
    ``R_dot R^T``.  For the left/spatial convention used by
    :func:`se3_left_apply`, ``p_dot = omega x p + v``; therefore the returned
    translational twist is ``v = p_dot - omega x p``.

    This is an output parameterization adapter only.  It does not project the
    state or the target: SE(3) training states and integration remain on the
    group throughout the complete denoising path.
    """

    if current_pose9.shape[:-1] != pose9_velocity.shape[:-1]:
        raise ValueError(
            "pose9 state and velocity must have matching leading dimensions; "
            f"got {current_pose9.shape} and {pose9_velocity.shape}."
        )
    if current_pose9.shape[-1] < 9 or pose9_velocity.shape[-1] < 9:
        raise ValueError(
            "pose9 state and velocity require at least 9 channels; "
            f"got {current_pose9.shape[-1]} and {pose9_velocity.shape[-1]}."
        )

    transform = pose9_to_matrix(current_pose9[..., :9].to(dtype=torch.float32))
    rotation = transform[..., :3, :3]
    position = transform[..., :3, 3]
    r1 = rotation[..., :, 0]
    r2 = rotation[..., :, 1]
    r1_dot = pose9_velocity[..., 3:6].to(dtype=torch.float32)
    r2_dot = pose9_velocity[..., 6:9].to(dtype=torch.float32)
    r3_dot = torch.cross(r1_dot, r2, dim=-1) + torch.cross(r1, r2_dot, dim=-1)
    rotation_dot = torch.stack([r1_dot, r2_dot, r3_dot], dim=-1)

    angular_matrix = rotation_dot @ rotation.transpose(-1, -2)
    angular_matrix = 0.5 * (angular_matrix - angular_matrix.transpose(-1, -2))
    omega = torch.stack(
        [
            angular_matrix[..., 2, 1],
            angular_matrix[..., 0, 2],
            angular_matrix[..., 1, 0],
        ],
        dim=-1,
    )
    position_dot = pose9_velocity[..., :3].to(dtype=torch.float32)
    linear = position_dot - torch.cross(omega, position, dim=-1)
    return torch.cat([linear, omega], dim=-1)


def body_twist_to_pose9_velocity(current_pose9: Tensor, body_twist: Tensor) -> Tensor:
    """Lift a right-trivialized SE(3) tangent into the legacy pose9 gauge.

    For a body twist ``xi = [v, omega]``, the physical tangent is
    ``T_dot = T hat(xi)``.  Hence ``p_dot = R v`` and
    ``R_dot = R hat(omega)``.  The pretrained action chart does not store an
    orthonormal rotation directly: its two raw 3-vectors may contain scale and
    shear that are removed by the rot6d projection.  Preserve those gauge
    coefficients while lifting ``R_dot`` so adding a zero twist is exactly the
    original Euclidean vector field and a nonzero twist changes only the
    represented physical tangent.
    """

    if current_pose9.shape[:-1] != body_twist.shape[:-1]:
        raise ValueError(
            "pose9 state and body twist must have matching leading dimensions; "
            f"got {current_pose9.shape} and {body_twist.shape}."
        )
    if current_pose9.shape[-1] < 9 or body_twist.shape[-1] != 6:
        raise ValueError(
            "body-twist lifting requires pose9 state dim >=9 and twist dim 6; "
            f"got {current_pose9.shape[-1]} and {body_twist.shape[-1]}."
        )

    current = current_pose9[..., :9].to(dtype=torch.float32)
    twist = body_twist.to(device=current.device, dtype=torch.float32)
    transform = pose9_to_matrix(current)
    rotation = transform[..., :3, :3]

    linear_body = twist[..., :3]
    angular_body = twist[..., 3:6]
    position_dot = (rotation @ linear_body.unsqueeze(-1)).squeeze(-1)
    rotation_dot = rotation @ _skew(angular_body)
    basis_1_dot = rotation_dot[..., :, 0]
    basis_2_dot = rotation_dot[..., :, 1]

    raw_1 = current[..., 3:6]
    raw_2 = current[..., 6:9]
    basis_1 = rotation[..., :, 0]
    basis_2 = rotation[..., :, 1]
    scale_1 = (raw_1 * basis_1).sum(dim=-1, keepdim=True)
    shear_12 = (raw_2 * basis_1).sum(dim=-1, keepdim=True)
    scale_2 = (raw_2 * basis_2).sum(dim=-1, keepdim=True)
    raw_1_dot = scale_1 * basis_1_dot
    raw_2_dot = shear_12 * basis_1_dot + scale_2 * basis_2_dot
    return torch.cat([position_dot, raw_1_dot, raw_2_dot], dim=-1)


def pose9_endpoint_velocity_to_spatial_twist(
    current_pose9: Tensor,
    pose9_velocity: Tensor,
    time: Tensor,
    endpoint_base_pose9: Tensor | None = None,
) -> Tensor:
    """Convert a legacy pose9 flow velocity to an exact SE(3) velocity.

    The original Euclidean flow head represents a clean endpoint as
    ``x_hat_1 = x_t + (1 - t) * v_pose9``.  Reusing that endpoint semantics is
    substantially better conditioned when warm-starting from a trained pose9
    policy than interpreting ``v_pose9`` as an instantaneous rotation-matrix
    derivative.  The candidate rotation is first projected to SO(3) by the
    standard rot6d conversion, then the exact spatial/left-trivialized twist
    from the current transform to that endpoint is returned.

    This adapter does not make the flow Euclidean: the random prior and every
    state remain on SE(3), and inference still integrates with group products.
    """

    if current_pose9.shape[:-1] != pose9_velocity.shape[:-1]:
        raise ValueError(
            "pose9 state and velocity must have matching leading dimensions; "
            f"got {current_pose9.shape} and {pose9_velocity.shape}."
        )
    if current_pose9.shape[-1] < 9 or pose9_velocity.shape[-1] < 9:
        raise ValueError(
            "pose9 state and velocity require at least 9 channels; "
            f"got {current_pose9.shape[-1]} and {pose9_velocity.shape[-1]}."
        )
    if time.ndim != 1 or time.shape[0] != current_pose9.shape[0]:
        raise ValueError(f"Expected time shape ({current_pose9.shape[0]},), got {time.shape}.")

    remaining = (1.0 - time.to(device=current_pose9.device, dtype=torch.float32)).clamp_min(1e-3)
    remaining_expanded = remaining.reshape(
        remaining.shape[0], *([1] * (current_pose9.ndim - 1))
    )
    if endpoint_base_pose9 is None:
        endpoint_base_pose9 = current_pose9
    if endpoint_base_pose9.shape[:-1] != current_pose9.shape[:-1] or endpoint_base_pose9.shape[-1] < 9:
        raise ValueError(
            "endpoint pose9 chart and current group state must have matching leading dimensions; "
            f"got {endpoint_base_pose9.shape} and {current_pose9.shape}."
        )
    endpoint_pose9 = (
        endpoint_base_pose9[..., :9].to(dtype=torch.float32)
        + remaining_expanded * pose9_velocity[..., :9].to(dtype=torch.float32)
    )
    current_transform = pose9_to_matrix(current_pose9[..., :9].to(dtype=torch.float32))
    endpoint_transform = pose9_to_matrix(endpoint_pose9)
    return se3_log(endpoint_transform @ invert_transform(current_transform)) / remaining_expanded


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


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def train(self, mode: bool = True):
        """Set policy mode and optionally preserve pretrained point-path BN statistics.

        These are fine-tuning stability controls, not camera-fusion or World
        fusion modules. All affine parameters retain their existing
        ``requires_grad`` state; only running mean/variance updates are
        disabled so training and inference use the same population statistics.
        """

        super().train(mode)
        if mode and bool(getattr(self.config, "pointseg_freeze_batchnorm_stats", False)):
            conditioner = getattr(getattr(self, "model", None), "pointseg_conditioner", None)
            if conditioner is not None:
                for module in conditioner.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        if mode and bool(getattr(self.config, "worldflow_freeze_batchnorm_stats", False)):
            worldflow = getattr(getattr(self, "model", None), "worldflow_branch", None)
            scene_encoder = getattr(worldflow, "scene_encoder", None)
            if scene_encoder is not None:
                for module in scene_encoder.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

    def initialize_action_expert_from_pretrained(
        self,
        source: str | None = None,
    ) -> dict[str, int | str]:
        """Initialize only a freshly constructed policy's Action Expert."""

        resolved_source = source or self.config.action_expert_weights_path or self.config.vlm_weights_path
        if resolved_source is None:
            raise ValueError(
                "No Action Expert checkpoint source was supplied. Set "
                "action_expert_weights_path or vlm_weights_path."
            )
        report = load_pretrained_action_expert_weights(
            self.model,
            str(resolved_source),
            load_action_projections=self.config.load_action_expert_projection_weights,
        )
        self.action_expert_initialization_report = report
        return report

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_worldflow_ego_tangent_world_only_parameter_ids(self) -> set[int]:
        """Return World point-path parameters that must retain World gradients.

        Symmetric multi-view adaptation intentionally moves the direct World
        point consumers into the high-LR point-input optimizer group.  The
        historical Ego-tangent gradient protection used optimizer group names
        as a proxy for physical roles, so it would otherwise misclassify these
        World-only parameters as Ego/shared and erase their World-sample
        gradients.  Keeping this role query on the policy decouples physical
        gradient semantics from learning-rate grouping without changing the
        forward graph or objective.
        """

        if not bool(
            getattr(
                self.config,
                "multiview_input_symmetric_point_path_adaptation",
                False,
            )
        ):
            return set()
        if not bool(getattr(self.config, "worldflow_enable", False)):
            raise RuntimeError("Symmetric World point-path roles require WorldFlow.")
        matches = {
            prefix: [
                parameter
                for name, parameter in self.named_parameters()
                if name.startswith(prefix) and parameter.requires_grad
            ]
            for prefix in SYMMETRIC_MULTIVIEW_WORLD_POINT_PREFIXES
        }
        missing = [prefix for prefix, parameters in matches.items() if not parameters]
        if missing:
            raise RuntimeError(
                "Symmetric World point-path role resolution found no trainable parameters for "
                f"{missing}."
            )
        return {
            id(parameter)
            for parameters in matches.values()
            for parameter in parameters
        }

    def get_optim_params(self):
        if self.config.vla_adapter_enable:
            trainable = [
                (name, parameter)
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ]
            if (
                getattr(self.config, "camera_view_fusion", "legacy_budget") == "primary_residual"
                and not getattr(self.config, "worldflow_enable", False)
                and (
                    getattr(self.config, "multiview_pretrained_lr_multiplier", 1.0) != 1.0
                    or getattr(self.config, "multiview_residual_lr_multiplier", 1.0) != 1.0
                )
            ):
                residual = [
                    parameter
                    for name, parameter in trainable
                    if name.startswith("model.multiview_residual_proj.")
                ]
                pretrained = [
                    parameter
                    for name, parameter in trainable
                    if not name.startswith("model.multiview_residual_proj.")
                ]
                if not residual or not pretrained:
                    raise RuntimeError(
                        "Two-timescale multi-view optimization requires non-empty pretrained and residual groups."
                    )
                base_lr = float(self.config.optimizer_lr)
                return [
                    {
                        "params": pretrained,
                        "lr": base_lr
                        * float(getattr(self.config, "multiview_pretrained_lr_multiplier", 1.0)),
                        "group_name": "pretrained_primary_path",
                    },
                    {
                        "params": residual,
                        "lr": base_lr
                        * float(getattr(self.config, "multiview_residual_lr_multiplier", 1.0)),
                        "group_name": "new_secondary_matrix_residual",
                    },
                ]
            if (
                getattr(self.config, "camera_view_fusion", "legacy_budget")
                in {
                    "fps",
                    "voxel_fps",
                    "voxel_cover_fps",
                    "novelty_union",
                    "multiscale_novelty_union",
                    "consensus_multiscale_novelty_union",
                    "transport_novelty_union",
                    "uniform_union",
                    "full_union",
                }
                and (
                    getattr(self.config, "multiview_input_pretrained_lr_multiplier", 1.0) != 1.0
                    or getattr(self.config, "multiview_input_point_lr_multiplier", 1.0) != 1.0
                )
            ):
                ego_point_prefixes = (
                    "model.pointseg_conditioner.",
                    "model.pointseg_object_proj.",
                    "model.pointseg_background_proj.",
                    "model.point_action_fusion.",
                )
                symmetric_point_adaptation = bool(
                    getattr(
                        self.config,
                        "multiview_input_symmetric_point_path_adaptation",
                        False,
                    )
                )
                point_prefixes = ego_point_prefixes + (
                    SYMMETRIC_MULTIVIEW_WORLD_POINT_PREFIXES
                    if symmetric_point_adaptation
                    else ()
                )
                point_input = [
                    parameter
                    for name, parameter in trainable
                    if name.startswith(point_prefixes)
                ]
                if getattr(self.config, "worldflow_enable", False):
                    world_pretrained_multiplier = float(
                        self.config.worldflow_pretrained_lr_multiplier
                    )
                    multiview_pretrained_multiplier = float(
                        getattr(self.config, "multiview_input_pretrained_lr_multiplier", 1.0)
                    )
                    if not math.isclose(
                        world_pretrained_multiplier,
                        multiview_pretrained_multiplier,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ):
                        raise ValueError(
                            "Joint input-layer multi-view + WorldFlow optimization requires "
                            "multiview_input_pretrained_lr_multiplier to equal "
                            "worldflow_pretrained_lr_multiplier so the shared Ego path has one "
                            "unambiguous learning rate."
                        )
                    new_world_prefixes = (
                        "model.worldflow_branch.",
                        "model.ego_scene_to_expert.",
                        "model.world_scene_to_expert.",
                        "model.world_action_out_proj.",
                        "model.world_se3_action_out_proj.",
                        "model.world_twist_residual_out_proj.",
                        "model.ego_to_world_cross_norm.",
                        "model.world_to_ego_cross_norm.",
                        "model.ego_to_world_cross_attn.",
                        "model.world_to_ego_cross_attn.",
                    )
                    new_world_exact = {
                        "model.world_ego_scene_type_embedding",
                        "model.world_ego_action_type_embedding",
                    }
                    residual_prefix = "model.world_twist_residual_out_proj."
                    residual = [
                        parameter
                        for name, parameter in trainable
                        if name.startswith(residual_prefix)
                    ]
                    new_world = [
                        parameter
                        for name, parameter in trainable
                        if (name in new_world_exact or name.startswith(new_world_prefixes))
                        and not name.startswith(point_prefixes)
                        and not name.startswith(residual_prefix)
                    ]
                    pretrained = [
                        parameter
                        for name, parameter in trainable
                        if not name.startswith(point_prefixes)
                        and name not in new_world_exact
                        and not name.startswith(new_world_prefixes)
                    ]
                    residual_multiplier = getattr(
                        self.config, "worldflow_residual_lr_multiplier", None
                    )
                    if residual_multiplier is None:
                        new_world.extend(residual)
                        residual = []
                    if (
                        not pretrained
                        or not point_input
                        or not new_world
                        or (residual_multiplier is not None and not residual)
                    ):
                        raise RuntimeError(
                            "Joint input-layer multi-view + WorldFlow optimization requires "
                            "non-empty shared-Ego, point-input, World, and explicitly requested "
                            "residual parameter groups."
                        )
                    grouped_ids = {
                        id(parameter)
                        for parameter in (*pretrained, *point_input, *new_world, *residual)
                    }
                    if len(grouped_ids) != len(trainable):
                        raise RuntimeError(
                            "Joint input-layer multi-view + WorldFlow optimizer groups overlap "
                            "or omit trainable parameters."
                        )
                    base_lr = float(self.config.optimizer_lr)
                    groups = [
                        {
                            "params": pretrained,
                            "lr": base_lr * multiview_pretrained_multiplier,
                            "group_name": "pretrained_ego_shared_nonpoint",
                        },
                        {
                            "params": point_input,
                            "lr": base_lr
                            * float(getattr(self.config, "multiview_input_point_lr_multiplier", 1.0)),
                            "group_name": "point_input_adaptation_path",
                        },
                        {
                            "params": new_world,
                            "lr": base_lr * float(self.config.worldflow_new_lr_multiplier),
                            "group_name": "new_world_bidirectional",
                        },
                    ]
                    if residual:
                        groups.append(
                            {
                                "params": residual,
                                "lr": base_lr * float(residual_multiplier),
                                "group_name": "world_physical_residual_head",
                            }
                        )
                    return groups
                pretrained = [
                    parameter
                    for name, parameter in trainable
                    if not name.startswith(point_prefixes)
                ]
                if not point_input or not pretrained:
                    raise RuntimeError(
                        "Input-layer multi-view discriminative optimization requires non-empty "
                        "point-input and pretrained action parameter groups."
                    )
                grouped_ids = {id(parameter) for parameter in (*pretrained, *point_input)}
                if len(grouped_ids) != len(trainable):
                    raise RuntimeError(
                        "Input-layer multi-view optimizer parameter groups overlap or omit trainable parameters."
                    )
                base_lr = float(self.config.optimizer_lr)
                return [
                    {
                        "params": pretrained,
                        "lr": base_lr
                        * float(getattr(self.config, "multiview_input_pretrained_lr_multiplier", 1.0)),
                        "group_name": "pretrained_action_path_jointly_trainable",
                    },
                    {
                        "params": point_input,
                        "lr": base_lr
                        * float(getattr(self.config, "multiview_input_point_lr_multiplier", 1.0)),
                        "group_name": "point_input_adaptation_path",
                    },
                ]
            if self.config.worldflow_enable and (
                self.config.worldflow_pretrained_lr_multiplier != 1.0
                or self.config.worldflow_new_lr_multiplier != 1.0
                or getattr(self.config, "worldflow_residual_lr_multiplier", None) is not None
            ):
                new_world_prefixes = (
                    "model.worldflow_branch.",
                    "model.ego_scene_to_expert.",
                    "model.world_scene_to_expert.",
                    "model.world_action_out_proj.",
                    "model.world_se3_action_out_proj.",
                    "model.world_twist_residual_out_proj.",
                    "model.ego_to_world_cross_norm.",
                    "model.world_to_ego_cross_norm.",
                    "model.ego_to_world_cross_attn.",
                    "model.world_to_ego_cross_attn.",
                )
                new_world_exact = {
                    "model.world_ego_scene_type_embedding",
                    "model.world_ego_action_type_embedding",
                }
                residual_prefix = "model.world_twist_residual_out_proj."
                residual = [
                    parameter
                    for name, parameter in trainable
                    if name.startswith(residual_prefix)
                ]
                new_world = [
                    parameter
                    for name, parameter in trainable
                    if (name in new_world_exact or name.startswith(new_world_prefixes))
                    and not name.startswith(residual_prefix)
                ]
                pretrained = [
                    parameter
                    for name, parameter in trainable
                    if name not in new_world_exact and not name.startswith(new_world_prefixes)
                ]
                residual_multiplier = getattr(
                    self.config, "worldflow_residual_lr_multiplier", None
                )
                if not new_world or not pretrained or (residual_multiplier is not None and not residual):
                    raise RuntimeError(
                        "Discriminative WorldFlow optimization requires non-empty pretrained, World, "
                        "and (when explicitly split) residual parameter groups."
                    )
                # Historical checkpoints/configs omit the residual multiplier.
                # Preserve their exact two-group optimizer by returning the
                # residual tensors to the World group.
                if residual_multiplier is None:
                    new_world.extend(residual)
                    residual = []
                grouped_ids = {id(parameter) for parameter in (*pretrained, *new_world, *residual)}
                if len(grouped_ids) != len(trainable):
                    raise RuntimeError("WorldFlow optimizer parameter groups overlap or omit trainable parameters.")
                base_lr = float(self.config.optimizer_lr)
                groups = [
                    {
                        "params": pretrained,
                        "lr": base_lr * float(self.config.worldflow_pretrained_lr_multiplier),
                        "group_name": "pretrained_ego_shared",
                    },
                    {
                        "params": new_world,
                        "lr": base_lr * float(self.config.worldflow_new_lr_multiplier),
                        "group_name": "new_world_bidirectional",
                    },
                ]
                if residual:
                    groups.append(
                        {
                            "params": residual,
                            "lr": base_lr * float(residual_multiplier),
                            "group_name": "world_physical_residual_head",
                        }
                    )
                return groups
            return [parameter for _, parameter in trainable]
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        pc_feats, pc_masks = self.prepare_point_clouds(batch)
        images, image_masks = self.prepare_images(batch) if self.config.vla_adapter_enable else (None, None)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        current_ee_pose = batch.get("worldflow.current_ee_pose")
        if self.config.worldflow_enable:
            if not torch.is_tensor(current_ee_pose):
                raise ValueError(
                    "WorldFlow inference requires 'worldflow.current_ee_pose' with the current "
                    "EEF pose9 in the same world frame used by the point-cloud conversion."
                )
            current_ee_pose = current_ee_pose.to(device=state.device, dtype=torch.float32)

        actions = self.model.sample_actions(
            pc_feats,
            pc_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            images=images,
            image_masks=image_masks,
            current_ee_pose=current_ee_pose,
            **kwargs,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        pc_feats, pc_masks = self.prepare_point_clouds(batch)
        images, image_masks = self.prepare_images(batch) if self.config.vla_adapter_enable else (None, None)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get(f"{ACTION}_is_pad")
        if torch.is_tensor(actions_is_pad):
            actions_is_pad = actions_is_pad.to(device=actions.device, dtype=torch.bool)
        worldflow_context = None
        if self.config.worldflow_enable:
            independent_world_trajectory = (
                getattr(self.config, "worldflow_target_type", "legacy_eef")
                == "world_eef_trajectory"
            )
            target_batch_key = (
                "worldflow.eef_trajectory"
                if independent_world_trajectory
                else "worldflow.ee_poses"
            )
            required_worldflow_keys = (
                "worldflow.current_ee_pose",
                target_batch_key,
                "worldflow.step_is_pad",
            )
            missing = [key for key in required_worldflow_keys if key not in batch]
            if missing:
                raise ValueError(f"WorldFlow training batch is missing required keys: {missing}")
            worldflow_context = {
                "current_ee_pose": batch["worldflow.current_ee_pose"],
                (
                    "eef_trajectory" if independent_world_trajectory else "ee_poses"
                ): batch[target_batch_key],
                "step_is_pad": batch["worldflow.step_is_pad"],
            }
        loss_dict = {}
        losses = self.model.forward(
            pc_feats,
            pc_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            actions_is_pad=actions_is_pad,
            images=images,
            image_masks=image_masks,
            worldflow_context=worldflow_context,
        )
        pointseg_aux_loss = self.model.last_pointseg_aux_loss
        pointseg_aux_weight = self.config.pointseg_aux_loss_weight if pointseg_aux_loss is not None else 0.0
        loss_dict["losses_after_forward"] = losses.clone().mean().item()
        loss_dict["loss_pointseg_aux"] = (
            pointseg_aux_loss.detach().item() if torch.is_tensor(pointseg_aux_loss) else 0.0
        )
        for key, value in self.model.last_pointseg_metrics.items():
            if torch.is_tensor(value):
                loss_dict[key] = value.detach().item()
        for key, value in self.model.last_se3_metrics.items():
            if torch.is_tensor(value):
                loss_dict[key] = value.detach().item()
        for key, value in self.model.last_action_metrics.items():
            if torch.is_tensor(value):
                loss_dict[key] = value.detach().item()
        worldflow_aux = self.model.compute_worldflow_aux_loss()
        for key, value in self.model.last_worldflow_metrics.items():
            if torch.is_tensor(value):
                loss_dict[key] = value.detach().item()

        # Remove padding
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        losses = losses * self._action_loss_dimension_weights(
            original_action_dim,
            device=losses.device,
            dtype=losses.dtype,
        )
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        valid_action_counts = torch.full(
            (losses.shape[0],),
            losses.shape[1],
            dtype=losses.dtype,
            device=losses.device,
        )


        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            valid_action_counts = in_episode_bound.sum(dim=1).clamp_min(1).to(dtype=losses.dtype)
            valid_denominator = valid_action_counts.sum() * original_action_dim
            loss_dict["losses_after_in_ep_bound"] = (losses.sum() / valid_denominator).item()


        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_action_loss = losses.sum(dim=(1, 2)) / (valid_action_counts * original_action_dim)
            loss_dict["loss_action"] = per_sample_action_loss.mean().item()
            per_sample_loss = per_sample_action_loss
            if torch.is_tensor(pointseg_aux_loss) and pointseg_aux_weight > 0:
                per_sample_loss = per_sample_loss + pointseg_aux_weight * pointseg_aux_loss
            if worldflow_aux is not None:
                per_sample_loss = per_sample_loss + worldflow_aux["per_sample_loss"]
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            action_loss = losses.sum() / (valid_action_counts.sum() * original_action_dim)
            loss_dict["loss_action"] = action_loss.item()
            loss = action_loss
            if torch.is_tensor(pointseg_aux_loss) and pointseg_aux_weight > 0:
                loss = loss + pointseg_aux_weight * pointseg_aux_loss
            if worldflow_aux is not None:
                loss = loss + worldflow_aux["per_sample_loss"].mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _action_loss_dimension_weights(
        self,
        action_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return mean-one weights for pose9 + gripper flow-matching dimensions."""

        weights = torch.ones(action_dim, device=device, dtype=dtype)
        if action_dim < 10:
            return weights
        weights[:3] = float(self.config.action_loss_translation_weight)
        weights[3:9] = float(self.config.action_loss_rotation_weight)
        weights[9] = float(self.config.action_loss_gripper_weight)
        return weights * (float(action_dim) / weights.sum().clamp_min(torch.finfo(dtype).eps))

    def prepare_point_clouds(self, batch):
        """Extract point cloud features from the batch.

        This variant replaces the original SmolVLA image preprocessing with a point cloud
        feature extractor. The batch is expected to contain `observation.point_cloud`.
        """
        if "observation.point_cloud" not in batch:
            raise ValueError(
                f"Point cloud feature 'observation.point_cloud' is missing from the batch. "
                f"Batch keys: {batch.keys()}"
            )

        pc = batch["observation.point_cloud"]
        point_is_pad = batch.get("observation.point_cloud_is_pad")
        if pc.ndim == 4 and pc.shape[1] == 1:
            # Some preprocessors pad a singleton time/channel axis: (B, 1, N, C)
            pc = pc.squeeze(1)
            if torch.is_tensor(point_is_pad) and point_is_pad.ndim == 3 and point_is_pad.shape[1] == 1:
                point_is_pad = point_is_pad.squeeze(1)
        elif pc.ndim != 3:
            raise ValueError(f"Expected observation.point_cloud shape (B, N, C), got {pc.shape}")
        if pc.shape[-1] != 6:
            raise ValueError(
                f"LitePTEncoder expects 6-channel point clouds, got last dim={pc.shape[-1]}. "
                "Use observation.point_cloud with shape (B, N, 6)."
            )

        bsize = pc.shape[0]
        device = pc.device
        pc = pc.to(dtype=torch.float32)
        target_points = int(getattr(self.config, "camera_view_fps_target_points", 10_000))
        fusion = getattr(self.config, "camera_view_fusion", "legacy_budget")
        if fusion == "full_union":
            # Paired primary/full-union batches are padded to the longest item.
            # The primary copy stays exact because its padded tail is masked by
            # point_is_pad throughout PointSeg and point encoding.
            if pc.shape[1] < target_points:
                raise ValueError(
                    f"full_union expects at least the {target_points}-point primary cloud, "
                    f"got {pc.shape[1]}."
                )
        elif pc.shape[1] != target_points:
            raise ValueError(
                "SmolVLA expects an input-adapted fixed-size point cloud. "
                f"Expected {target_points} points, got {pc.shape[1]}. Apply single/multi-view "
                "fusion and FPS/voxel sampling in the data or inference input layer."
            )
        point_cloud_payload: Tensor | dict[str, Tensor] = pc
        pointseg_keys = (
            "pointseg.priors",
            "pointseg.labels",
            "pointseg.weights",
            "pointseg.class_scores",
            "pointseg.role_scores",
            "pointseg.foreground_score",
        )
        if self.model.pointseg_conditioner is not None or torch.is_tensor(point_is_pad) or any(key in batch for key in pointseg_keys):
            payload = {"point_cloud": pc}
            if torch.is_tensor(point_is_pad):
                payload["point_is_pad"] = point_is_pad.to(device=device, dtype=torch.bool)
            for key in pointseg_keys:
                if key not in batch:
                    continue
                value = batch[key]
                if torch.is_tensor(value) and value.ndim >= 3 and value.shape[1] == 1:
                    value = value.squeeze(1)
                payload[key] = value
            point_cloud_payload = payload
        mask = torch.ones(bsize, dtype=torch.bool, device=device)
        # if "observation.point_cloud_is_pad" in batch:
        #     mask = batch["observation.point_cloud_is_pad"].bool()
        # else:
        #     mask = torch.ones(bsize, dtype=torch.bool, device=device)

        return [point_cloud_payload], [mask]

    def prepare_images(self, batch):
        """Prepare one static RGB frame per configured camera for the frozen vision encoder."""
        images = []
        image_masks = []
        present_keys = [key for key in self.config.image_features if key in batch]
        missing_keys = [key for key in self.config.image_features if key not in batch]
        optional_empty_keys = {
            f"{OBS_IMAGES}.empty_camera_{index}" for index in range(self.config.empty_cameras)
        }
        required_missing_keys = [key for key in missing_keys if key not in optional_empty_keys]

        if not present_keys:
            raise ValueError(
                "vla_adapter_enable=True requires an RGB image feature in every batch. "
                f"Expected one of {list(self.config.image_features)}, got keys {list(batch)}. "
                f"Use a dataset containing a feature such as '{OBS_IMAGES}.overhead'."
            )
        if required_missing_keys:
            raise ValueError(
                "Frozen-VLM adapter batch is missing configured RGB camera(s): "
                f"{required_missing_keys}. Available keys: {list(batch)}."
            )

        for key in present_keys:
            image = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if image.ndim != 4:
                raise ValueError(f"Expected image {key} shape (B,C,H,W), got {tuple(image.shape)}.")
            if image.shape[1] not in (1, 3, 4):
                raise ValueError(f"Expected channel-first image {key}, got {tuple(image.shape)}.")
            image = image[:, :3]
            if image.shape[1] == 1:
                image = image.expand(-1, 3, -1, -1)
            image = image.to(dtype=torch.float32)
            if self.config.resize_imgs_with_padding is not None:
                image = resize_with_pad(image, *self.config.resize_imgs_with_padding, pad_value=0)
            # LeRobot visual features and the inference wrappers provide [0, 1].
            # Keep the official SmolVLA conversion and avoid a GPU min/max sync.
            image = image * 2.0 - 1.0

            batch_size = image.shape[0]
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].to(device=image.device, dtype=torch.bool)
                if mask.ndim > 1:
                    mask = mask[:, -1]
            else:
                mask = torch.ones(batch_size, dtype=torch.bool, device=image.device)
            images.append(image)
            image_masks.append(mask)

        # Preserve the official empty-camera behavior without silently treating
        # an actually required RGB camera as optional.
        for _key in missing_keys:
            if _key not in optional_empty_keys:
                continue
            images.append(torch.full_like(images[0], -1.0))
            image_masks.append(torch.zeros_like(image_masks[0]))
        return images, image_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|se3_action_out_proj|world_action_out_proj|"
            "world_se3_action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging
 
            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class SelfAttention(nn.Module):
    """ Self‑Attention module for point features of shape (B, N, C) """

    def __init__(self, C):
        super().__init__()
        self.C = C

        # 把每个点的特征映射到 Q, K, V
        self.query = nn.Linear(C, C, bias=False)
        self.key   = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)

        # Optional：可加 LayerNorm，对点维做 normalize
        self.norm = nn.LayerNorm(C)

    def forward(self, x, mask=None):
        # x: (B, N, C)
        B, N, C = x.shape

        # 1) 投影到 Q, K, V: (B, N, C)
        Q = self.query(x)  # (B, N, C)
        K = self.key(x)    # (B, N, C)
        V = self.value(x)  # (B, N, C)

        # 2) 计算 Attention Score: (B, N, N)
        # Q K^T / sqrt(C)
        scores = torch.bmm(Q, K.transpose(1, 2))  # (B, N, N)
        scores = scores / (C ** 0.5)
        if mask is not None:
            key_mask = mask[:, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        # 3) softmax 沿 N 维 (即每行是 N 个点的权重)
        attn_weights = F.softmax(scores, dim=2)  # (B, N, N)

        # 4) 加权得到输出: (B, N, C)
        x_out = torch.bmm(attn_weights, V)  # (B, N, C)

        # 5) 可选：加 LayerNorm + 残差
        x_out = self.norm(x_out + x)  # (B, N, C)
        if mask is not None:
            x_out = x_out * mask.unsqueeze(-1).to(dtype=x_out.dtype)

        return x_out

class LitePTTokenizer(nn.Module):
    """
    input:  pc (B,N,C) XYZ m RGB 0-255
    output: xyz_tok (B,Pmax,3), tok (B,Pmax,dim), g (B,dim), tok_mask (B,Pmax)
    """

    def __init__(self, in_dim=6, dim=128, n_tokens=512, grid_size=0.005, enc_mode=False):
        super().__init__()
        self.dim = dim
        self.n_tokens = n_tokens
        self.grid_size = grid_size

        self.backbone = LitePT(in_channels=in_dim, enc_mode=enc_mode)
        self.out_proj = nn.Linear(infer_litept_output_channels(self.backbone), dim)

    def train(self, mode: bool = True):
        """Keep serialization shuffling as training-only augmentation.

        LitePT stores ``shuffle_orders`` as a plain boolean, so ``model.eval()``
        does not disable it automatically.  Leaving it enabled entangles the
        point-cloud encoding order with the Flow Matching RNG seed and makes
        identical singleton observations produce different policy features.
        """
        super().train(mode)
        if hasattr(self.backbone, "shuffle_orders"):
            self.backbone.shuffle_orders = bool(mode)
        return self

    def _is_degenerate(self, xyz, eps=1e-6):
        rng = (xyz.max(dim=0).values - xyz.min(dim=0).values).abs().sum()
        return (not torch.isfinite(xyz).all()) or (rng < eps)

    def _grid_sample_batch(self, coord, feat, batch):
        """
        batch-aware grid sampling
        coord: (P,3)
        feat:  (P,C)
        batch: (P,)
        """

        device = coord.device

        # -------- voxel --------
        grid = torch.floor(coord / self.grid_size)

        # 把 batch 拼进去，避免不同样本混合
        grid = torch.cat([batch.unsqueeze(-1), grid], dim=1)

        unique, inverse = torch.unique(grid, dim=0, return_inverse=True)

        N = inverse.shape[0]
        perm = torch.arange(N, device=device)

        idx = torch.zeros(unique.shape[0], device=device, dtype=torch.long)
        idx.scatter_(0, inverse.flip(0), perm.flip(0))

        coord = coord[idx]
        feat = feat[idx]
        batch = batch[idx]

        return coord, feat, batch

    def forward(self, pc, point_is_pad=None):
        B, N, C = pc.shape
        device = pc.device
        if point_is_pad is None:
            point_is_pad = torch.zeros(B, N, dtype=torch.bool, device=device)
        else:
            point_is_pad = point_is_pad.to(device=device, dtype=torch.bool)
            if point_is_pad.shape != pc.shape[:2]:
                raise ValueError(f"Expected point_is_pad shape {pc.shape[:2]}, got {point_is_pad.shape}.")

        # ========= 输出 =========
        empty_len = 1
        global_xyz_tok = torch.zeros(B, empty_len, 3, device=device, dtype=pc.dtype)
        tok = torch.zeros(B, empty_len, self.dim, device=device, dtype=pc.dtype)
        g = torch.zeros(B, self.dim, device=device, dtype=pc.dtype)
        tok_mask = torch.zeros(B, empty_len, device=device, dtype=torch.bool)

        # ========= mask valid =========
        valid_mask = []
        for b in range(B):
            point_valid = ~point_is_pad[b]
            valid_mask.append(bool(point_valid.any().item()) and not self._is_degenerate(pc[b, point_valid, :3]))
        valid_mask = torch.tensor(valid_mask, device=device)

        if valid_mask.sum() == 0:
            return global_xyz_tok, tok, g, tok_mask

        pc_v = pc[valid_mask]  # (Bv,N,C)
        point_is_pad_v = point_is_pad[valid_mask]
        Bv = pc_v.shape[0]

        # ========= flatten =========
        flat_valid = (~point_is_pad_v).reshape(-1)
        coord = pc_v[:, :, :3].reshape(-1, 3)[flat_valid].contiguous()

        feat = pc_v.reshape(-1, C)[flat_valid].contiguous()
        if C > 3:
            feat = torch.cat([feat[:, :3], feat[:, 3:] / 255.0], dim=1)
        else:
            feat = feat[:, :3]

        # batch index
        batch = torch.arange(Bv, device=device).repeat_interleave((~point_is_pad_v).sum(dim=1))

        # ========= grid sample =========
        coord, feat, batch = self._grid_sample_batch(coord, feat, batch)

        if coord.shape[0] == 0:
            return global_xyz_tok, tok, g, tok_mask

        # ========= offset =========
        counts = torch.bincount(batch, minlength=Bv)
        offset = torch.cumsum(counts, dim=0)

        data_dict = {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": self.grid_size,
        }

        # ========= LitePT forward =========
        point = self.backbone(data_dict)

        feat_p = self.out_proj(point.feat)
        xyz_p = point.coord

        # ========= recover batch =========
        if hasattr(point, "batch"):
            b_p = point.batch.to(torch.long)
        else:
            # fallback（关键🔥）
            b_p = torch.bucketize(
                torch.arange(xyz_p.shape[0], device=device),
                offset,
                right=True,
            )

        # ========= 按batch拆分 =========
        valid_idx = torch.nonzero(valid_mask).squeeze(1)
        token_counts = torch.bincount(b_p, minlength=Bv)
        max_tokens = max(int(token_counts.max().item()), 1)

        global_xyz_tok = torch.zeros(B, max_tokens, 3, device=device, dtype=pc.dtype)
        tok = torch.zeros(B, max_tokens, self.dim, device=device, dtype=pc.dtype)
        tok_mask = torch.zeros(B, max_tokens, device=device, dtype=torch.bool)

        for i in range(Bv):
            global_b = valid_idx[i].item()

            P = int(token_counts[i].item())

            if P == 0:
                continue

            idx_all = torch.nonzero(b_p == i, as_tuple=False).squeeze(1)

            global_xyz_tok[global_b, :P] = xyz_p[idx_all]
            tok[global_b, :P] = feat_p[idx_all]
            tok_mask[global_b, :P] = True
            g[global_b] = feat_p[idx_all].max(dim=0).values

        return global_xyz_tok, tok, g, tok_mask


class LitePTEncoder(nn.Module):
    """
    input:  pc (B,N,C)
    output: global feature (B,dim)
    """

    def __init__(self, in_dim=6, dim=128, n_tokens=512, grid_size=0.05):
        super().__init__()
        self.dim = dim
        self.n_tokens = n_tokens
        self.grid_size = grid_size
        self.pc_backbone = LitePTTokenizer(in_dim=in_dim, dim=dim, n_tokens=n_tokens,grid_size=grid_size)
        self.pc_backbone1 = LitePTTokenizer(in_dim=in_dim, dim=dim, n_tokens=n_tokens,grid_size=grid_size)


        # self.dense_xattn_full = DenseGeoCrossAttnFull(dim=dim, heads=4, chunk_size=128)
        # self.proj_pc = nn.Linear(dim, dim)

        self.attention = SelfAttention(dim)
        self.att = nn.Sequential(
            nn.Linear(dim, 1),    # 将每个点的特征映射到一个标量
            nn.Softmax(dim=1)   # 对 N 个点做 softmax，得到权重
        ) #神来之笔，把无用点过滤，然后放心信息聚合，实现点级特征BNC到场景级特征BC的降维
        self.attention1 = SelfAttention(dim)
        self.att1 = nn.Sequential(
            nn.Linear(dim, 1),    # 将每个点的特征映射到一个标量
            nn.Softmax(dim=1)   # 对 N 个点做 softmax，得到权重
        ) #神来之笔，把无用点过滤，然后放心信息聚合，实现点级特征BNC到场景级特征BC的降维
        # self.tok_self_attn = TokSelfAttention(dim, num_heads=4, depth=2)
        # self.slot_attn = SlotAttention(dim, num_slots=3)
        # # self.token_learner = TokenLearner(dim, num_tokens=3)
        # self.relation = RelationTransformer(dim)
        # self.token_fusion_mlp = TokenFusionMLP(dim)

    def _is_degenerate(self, xyz, eps=1e-6):
        rng = (xyz.max(dim=0).values - xyz.min(dim=0).values).abs().sum()
        return (not torch.isfinite(xyz).all()) or (rng < eps)



    def _masked_softmax(self, scores, mask):
        mask = mask.unsqueeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        weights = weights * mask.to(dtype=weights.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return weights / denom

    def forward(self, scene_pc, point_is_pad=None, *, return_tokens: bool = False):
        scene_xyz, scene_tok, _scene_g, scene_mask = self.pc_backbone(scene_pc, point_is_pad)
        scene_tok = self.attention(scene_tok, scene_mask)
        alpha = self._masked_softmax(self.att[0](scene_tok), scene_mask)
        global_feat = (scene_tok * alpha).sum(dim=1)
        # center = (scene_xyz * alpha).sum(dim=1)

        # centroid_xyz = scene_pc[..., :3] - center.unsqueeze(-2)
        # scene_pc1 = torch.cat([centroid_xyz, scene_pc[..., 3:]], dim=-1)
        # _scene_xyz1, scene_tok1, _scene_g1, scene_mask1 = self.pc_backbone1(scene_pc1, point_is_pad)
        # scene_tok1 = self.attention1(scene_tok1, scene_mask1)
        # alpha1 = self._masked_softmax(self.att1[0](scene_tok1), scene_mask1)

        # global_feat = (scene_tok1 * alpha1).sum(dim=1)
        if return_tokens:
            return {
                "global_feat": global_feat,
                "scene_tok1": scene_tok,
                "scene_mask1": scene_mask,
            }
        return global_feat




        # # has_cond = torch.ones(global_xyz_tok.shape[0])
        # # t_xyz, t_tok = global_xyz_tok, tok
        # # c_xyz, c_tok = global_xyz_tok, tok
        # # t_tok_rel = self.dense_xattn_full(t_xyz, t_tok, c_xyz, c_tok, has_cond)  # (B,T,C)
        # # # Only Cross Attention
        # # t_mem = self.proj_pc(t_tok_rel)  # (B,T,D)q
        # # global_feat = t_mem.max(dim=1).values

        # tok = self.attention(tok)
        # # 3. 权重加权求和得到全局特征
        # alpha = self.att(tok)        # [B, N, C]
        # global_feat = (tok * alpha).sum(dim=1)  # [B, C]

        # # # slot3 = self.slot_attn(tok) #3cluster cond target else
        # # # slot_relation_feature = self.relation(slot3) #cluster cross attention
        # # # fused_feature = self.token_fusion_mlp(slot_relation_feature.reshape(B, -1))


        # return global_feat


class PointActionSelfAttention(nn.Module):
    """PointACT-style joint self-attention over final LitePT point tokens and action tokens.

    The original point-cloud prefix path stays unchanged: this module only uses
    scene_tok1 as geometry memory and returns updated action tokens.  It does
    not feed action-conditioned point tokens back into LitePT pooling.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        point_dim: int,
        max_action_steps: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if point_dim % int(num_heads) != 0:
            raise ValueError(f"point_dim={point_dim} must be divisible by num_heads={num_heads}.")
        if int(max_action_steps) <= 0:
            raise ValueError(f"max_action_steps must be positive, got {max_action_steps}.")
        self.action_dim = int(action_dim)
        self.point_dim = int(point_dim)
        self.max_action_steps = int(max_action_steps)
        self.num_heads = int(num_heads)

        # PointAction fusion runs before the Action Expert applies RoPE.  Give
        # each action token an explicit step identity here so that equal noisy
        # action values at different chunk positions are still distinguishable.
        action_step_pos = torch.zeros(self.max_action_steps, self.point_dim, dtype=torch.float32)
        positions = torch.arange(self.max_action_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.point_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / self.point_dim)
        )
        action_step_pos[:, 0::2] = torch.sin(positions * div_term)
        action_step_pos[:, 1::2] = torch.cos(positions * div_term[: action_step_pos[:, 1::2].shape[1]])
        # This encoding is deterministic and should not become an additional
        # checkpoint compatibility requirement.
        self.register_buffer("action_step_pos_embedding", action_step_pos, persistent=False)

        self.point_norm = nn.LayerNorm(self.point_dim)
        self.action_norm = nn.LayerNorm(self.action_dim)
        self.action_to_point = nn.Linear(self.action_dim, self.point_dim)
        self.joint_norm = nn.LayerNorm(self.point_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.point_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.action_ffn_norm = nn.LayerNorm(self.point_dim)
        self.action_ffn = nn.Sequential(
            nn.Linear(self.point_dim, self.point_dim * 4),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.point_dim * 4, self.point_dim),
        )
        self.point_to_action = nn.Linear(self.point_dim, self.action_dim)

    @staticmethod
    def _normalize_point_mask(point_tokens: Tensor, point_mask: Tensor | None) -> Tensor:
        bsize, num_points = point_tokens.shape[:2]
        if point_mask is None:
            return torch.ones(bsize, num_points, dtype=torch.bool, device=point_tokens.device)
        point_mask = point_mask.to(device=point_tokens.device, dtype=torch.bool)
        if point_mask.shape != point_tokens.shape[:2]:
            raise ValueError(f"Expected point_mask shape {point_tokens.shape[:2]}, got {point_mask.shape}.")
        return point_mask

    @staticmethod
    def _normalize_action_pad_mask(action_tokens: Tensor, actions_is_pad: Tensor | None) -> Tensor:
        bsize, num_actions = action_tokens.shape[:2]
        if actions_is_pad is None:
            return torch.zeros(bsize, num_actions, dtype=torch.bool, device=action_tokens.device)
        actions_is_pad = actions_is_pad.to(device=action_tokens.device, dtype=torch.bool)
        if actions_is_pad.shape != action_tokens.shape[:2]:
            raise ValueError(
                f"Expected actions_is_pad shape {action_tokens.shape[:2]}, got {actions_is_pad.shape}."
            )
        return actions_is_pad

    def forward(
        self,
        action_tokens: Tensor,
        point_tokens: Tensor,
        point_mask: Tensor | None = None,
        actions_is_pad: Tensor | None = None,
    ) -> Tensor:
        if action_tokens.ndim != 3:
            raise ValueError(f"Expected action_tokens shape (B,T,C), got {action_tokens.shape}.")
        if point_tokens.ndim != 3:
            raise ValueError(f"Expected point_tokens shape (B,N,C), got {point_tokens.shape}.")
        if action_tokens.shape[0] != point_tokens.shape[0]:
            raise ValueError(
                f"Batch mismatch between action_tokens {action_tokens.shape} and point_tokens {point_tokens.shape}."
            )
        if point_tokens.shape[-1] != self.point_dim:
            raise ValueError(f"Expected point token dim {self.point_dim}, got {point_tokens.shape[-1]}.")
        if action_tokens.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action token dim {self.action_dim}, got {action_tokens.shape[-1]}.")
        if action_tokens.shape[1] > self.max_action_steps:
            raise ValueError(
                f"Action sequence length {action_tokens.shape[1]} exceeds max_action_steps={self.max_action_steps}."
            )

        point_mask = self._normalize_point_mask(point_tokens, point_mask)
        actions_is_pad = self._normalize_action_pad_mask(action_tokens, actions_is_pad)
        action_mask = ~actions_is_pad
        point_latent = self.point_norm(point_tokens)
        action_latent = self.action_to_point(self.action_norm(action_tokens))
        action_step_pos = self.action_step_pos_embedding[: action_tokens.shape[1]].to(dtype=action_latent.dtype)
        action_latent = action_latent + action_step_pos.unsqueeze(0)

        joint = torch.cat([point_latent, action_latent], dim=1)
        joint_valid = torch.cat([point_mask, action_mask], dim=1)

        # MultiheadAttention cannot softmax a row when every key is masked.
        # Such a sample carries no usable point/action token, so expose one
        # harmless key for numerical safety and suppress all padded updates
        # below.
        safe_joint_valid = joint_valid.clone()
        no_valid_keys = ~safe_joint_valid.any(dim=1)
        if no_valid_keys.any():
            safe_joint_valid[no_valid_keys, 0] = True

        attn_in = self.joint_norm(joint)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=~safe_joint_valid,
            need_weights=False,
        )
        joint = joint + attn_out

        action_joint = joint[:, point_tokens.shape[1] :]
        action_delta = action_joint - action_latent
        action_delta = action_delta + self.action_ffn(self.action_ffn_norm(action_delta))
        action_update = self.point_to_action(action_delta)
        action_update = action_update * action_mask.unsqueeze(-1).to(dtype=action_update.dtype)
        return action_tokens + action_update


class WorldFlowActionBranch(nn.Module):
    """World-frame point/action token front-end for the shared Action Expert.

    The branch is independent from Ego up to its output tokens: it owns a
    dedicated LitePT encoder, PointAction adapter, language embedding and
    action/time projections.  It deliberately does *not* own another Action
    Expert. Its single global scene token and point-fused action tokens are
    joined with the corresponding Ego tokens and processed by the policy's
    shared Action Expert.

    Only predicted foreground XYZRGB points enter this module.  PointSeg
    probabilities, pseudo labels and role/evidence channels are not inputs.
    """

    pose_dim = 9

    def __init__(
        self,
        config: SmolVLAConfig,
        *,
        action_hidden_dim: int,
        language_vocab_size: int,
    ) -> None:
        super().__init__()
        self.chunk_size = int(config.chunk_size)
        self.feature_dim = int(config.worldflow_feature_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.min_period = float(config.min_period)
        self.max_period = float(config.max_period)
        if self.feature_dim % int(config.point_action_fusion_heads) != 0:
            raise ValueError(
                f"worldflow_feature_dim={self.feature_dim} must be divisible by "
                f"point_action_fusion_heads={config.point_action_fusion_heads}."
            )

        self.scene_encoder = LitePTEncoder(
            in_dim=6,
            dim=self.feature_dim,
            n_tokens=256,
            grid_size=float(config.worldflow_grid_size),
        )
        # LitePTEncoder retains historical second-stage modules for Ego
        # checkpoint compatibility, although its active forward uses only the
        # first stage. They are not part of this branch's computation.
        for inactive_module in (
            self.scene_encoder.pc_backbone1,
            self.scene_encoder.attention1,
            self.scene_encoder.att1,
        ):
            inactive_module.requires_grad_(False)

        self.action_in_proj = nn.Linear(self.pose_dim, self.action_hidden_dim)
        self.action_time_mlp_in = nn.Linear(self.action_hidden_dim * 2, self.action_hidden_dim)
        self.action_time_mlp_out = nn.Linear(self.action_hidden_dim, self.action_hidden_dim)
        self.scene_context_proj = nn.Linear(self.feature_dim, self.action_hidden_dim)

        # A separate embedding avoids sharing even the language lookup table
        # with the Ego/VLM branch. Masked mean language context is sufficient
        # here because WorldFlow is a geometric auxiliary objective.
        self.language_embedding = nn.Embedding(int(language_vocab_size), self.action_hidden_dim)
        self.language_norm = nn.LayerNorm(self.action_hidden_dim)
        self.point_action_adapter = PointActionSelfAttention(
            action_dim=self.action_hidden_dim,
            point_dim=self.feature_dim,
            max_action_steps=self.chunk_size,
            num_heads=int(config.point_action_fusion_heads),
            dropout=float(config.point_action_fusion_dropout),
        )

    def encode_scene(
        self,
        point_cloud_world: Tensor,
        *,
        point_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if point_cloud_world.ndim != 3 or point_cloud_world.shape[-1] != 6:
            raise ValueError(f"Expected WorldFlow point cloud shape (B,N,6), got {point_cloud_world.shape}.")

        with _batchnorm_eval_on_single_value(self.scene_encoder):
            scene = self.scene_encoder(
                point_cloud_world.to(dtype=torch.float32),
                point_is_pad,
                return_tokens=True,
            )
        return {
            "scene_tokens": scene["scene_tok1"],
            "scene_mask": scene["scene_mask1"].to(dtype=torch.bool),
            "global_feat": scene["global_feat"],
        }

    def embed_action_tokens(
        self,
        scene: dict[str, Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        noisy_spatial_pose9: Tensor,
        time: Tensor,
        *,
        actions_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if noisy_spatial_pose9.ndim != 3 or noisy_spatial_pose9.shape[-1] != self.pose_dim:
            raise ValueError(
                f"Expected noisy WorldFlow actions shape (B,T,{self.pose_dim}), "
                f"got {noisy_spatial_pose9.shape}."
            )
        if noisy_spatial_pose9.shape[1] != self.chunk_size:
            raise ValueError(
                f"Expected WorldFlow action chunk {self.chunk_size}, got {noisy_spatial_pose9.shape[1]}."
            )
        if time.shape != (noisy_spatial_pose9.shape[0],):
            raise ValueError(f"Expected WorldFlow time shape {(noisy_spatial_pose9.shape[0],)}, got {time.shape}.")

        action_tokens = self.action_in_proj(noisy_spatial_pose9.to(dtype=torch.float32))
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.action_hidden_dim,
            self.min_period,
            self.max_period,
            device=action_tokens.device,
        ).to(dtype=action_tokens.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_tokens)
        action_tokens = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(torch.cat([action_tokens, time_emb], dim=-1)))
        )

        lang_emb = self.language_embedding(lang_tokens.to(device=action_tokens.device, dtype=torch.long))
        lang_context = _masked_language_mean(
            lang_emb,
            lang_masks.to(device=action_tokens.device, dtype=torch.bool),
        )
        action_tokens = (
            action_tokens
            + self.scene_context_proj(scene["global_feat"]).unsqueeze(1)
            + self.language_norm(lang_context).unsqueeze(1)
        )
        action_tokens = self.point_action_adapter(
            action_tokens,
            scene["scene_tokens"],
            point_mask=scene["scene_mask"],
            actions_is_pad=actions_is_pad,
        )

        if actions_is_pad is None:
            action_valid = torch.ones(
                action_tokens.shape[:2], dtype=torch.bool, device=action_tokens.device
            )
        else:
            actions_is_pad = actions_is_pad.to(device=action_tokens.device, dtype=torch.bool)
            if actions_is_pad.shape != action_tokens.shape[:2]:
                raise ValueError(
                    f"Expected WorldFlow actions_is_pad shape {action_tokens.shape[:2]}, "
                    f"got {actions_is_pad.shape}."
                )
            action_valid = ~actions_is_pad
        action_tokens = action_tokens * action_valid.unsqueeze(-1).to(dtype=action_tokens.dtype)
        return action_tokens, action_valid

    def forward(
        self,
        point_cloud_world: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        noisy_spatial_pose9: Tensor,
        time: Tensor,
        *,
        point_is_pad: Tensor | None = None,
        actions_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        scene = self.encode_scene(point_cloud_world, point_is_pad=point_is_pad)
        action_tokens, action_mask = self.embed_action_tokens(
            scene,
            lang_tokens,
            lang_masks,
            noisy_spatial_pose9,
            time,
            actions_is_pad=actions_is_pad,
        )
        return {
            **scene,
            "action_tokens": action_tokens,
            "action_mask": action_mask,
        }


class _MaskedPointMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, point_cloud: Tensor, point_is_pad: Tensor | None = None) -> Tensor:
        feat = point_cloud.to(dtype=torch.float32).clone()
        if feat.shape[-1] > 3:
            feat[..., 3:] = feat[..., 3:] / 255.0
        feat = self.net(feat)
        if point_is_pad is None:
            return feat.mean(dim=1)
        valid = ~point_is_pad.to(device=feat.device, dtype=torch.bool)
        weights = valid.unsqueeze(-1).to(dtype=feat.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (feat * weights).sum(dim=1) / denom


def _rotation_geodesic(pred_rot: Tensor, target_rot: Tensor) -> Tensor:
    rel = pred_rot.transpose(-1, -2) @ target_rot
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    sine_vec = torch.stack(
        [
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(sine_vec, dim=-1)
    return torch.atan2(sine, cosine)


def _masked_step_mean(values: Tensor, valid: Tensor) -> Tensor:
    valid_f = valid.to(device=values.device, dtype=values.dtype)
    return (values * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)


def _worldflow_carrier_matrix(current_pose_world: Tensor, frame_origin: str) -> Tensor:
    """Build the invertible Ego-to-WorldFlow coordinate carrier."""

    if current_pose_world.ndim != 2 or current_pose_world.shape[-1] != 9:
        raise ValueError(f"Expected current world pose shape (B,9), got {current_pose_world.shape}.")
    carrier = pose9_to_matrix(current_pose_world.to(dtype=torch.float32))
    if frame_origin == "global":
        return carrier
    if frame_origin == "current_ee":
        carrier = carrier.clone()
        carrier[..., :3, 3] = 0.0
        return carrier
    raise ValueError(f"Unknown WorldFlow frame_origin={frame_origin!r}.")


def _ego_point_cloud_to_world(
    point_cloud_ego: Tensor,
    current_pose_world: Tensor,
    *,
    frame_origin: str = "global",
) -> Tensor:
    if point_cloud_ego.ndim != 3 or point_cloud_ego.shape[-1] != 6:
        raise ValueError(f"Expected ego point cloud shape (B,N,6), got {point_cloud_ego.shape}.")
    transform = _worldflow_carrier_matrix(
        current_pose_world.to(device=point_cloud_ego.device, dtype=torch.float32),
        frame_origin,
    )
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    xyz_world = torch.matmul(point_cloud_ego[..., :3].to(dtype=torch.float32), rot.transpose(-1, -2))
    xyz_world = xyz_world + trans.unsqueeze(1)
    return torch.cat([xyz_world, point_cloud_ego[..., 3:6].to(dtype=torch.float32)], dim=-1)


def _masked_language_mean(lang_emb: Tensor, lang_masks: Tensor) -> Tensor:
    mask = lang_masks.to(device=lang_emb.device, dtype=torch.bool)
    weights = mask.unsqueeze(-1).to(dtype=lang_emb.dtype)
    denom = weights.sum(dim=1).clamp_min(torch.finfo(lang_emb.dtype).tiny)
    return (lang_emb * weights).sum(dim=1) / denom


class SongPointCloudConditioner(nn.Module):
    """Trainable SongPointSeg front-end that emits object/background point-cloud features."""

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        self.feature_dim = int(config.pointseg_feature_dim)
        self.foreground_ratio = float(config.pointseg_foreground_ratio)
        self.background_ratio = float(config.pointseg_background_ratio)
        self.min_foreground_points = int(config.pointseg_min_foreground_points)
        self.min_background_points = int(config.pointseg_min_background_points)
        self.aux_loss_weight = float(config.pointseg_aux_loss_weight)

        self.segmenter = SongPointSegNet(
            backbone_type=config.pointseg_backbone_type,
            grid_size=config.pointseg_grid_size,
        )
        self._load_pointseg_checkpoint(config.pointseg_checkpoint_path)
        self.foreground_encoder = LitePTEncoder(
            in_dim=6,
            dim=self.feature_dim,
            n_tokens=256,
            grid_size=config.pointseg_grid_size,
        )
        self.background_encoder = LitePTTokenizer(in_dim=6, dim=self.feature_dim, n_tokens=256, grid_size=config.pointseg_grid_size,enc_mode=True)
        self.null_background_feat = nn.Parameter(torch.zeros(self.feature_dim))
        self.pseudo_config = PseudoLabelConfig()
        self.pointseg_loss = SongPointSegLoss(SongPointSegLossConfig())

    def _load_pointseg_checkpoint(self, checkpoint_path: str | None) -> None:
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Song pointseg checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        try:
            missing, unexpected = self.segmenter.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load SongPointSegNet checkpoint. The segmentation network now uses only "
                "6-channel XYZRGB point features; checkpoints trained with motion-prior input channels "
                "must be retrained."
            ) from exc
        if missing:
            print(f"SongPointSegNet checkpoint missing keys: {missing}")
        if unexpected:
            print(f"SongPointSegNet checkpoint unexpected keys: {unexpected}")
        for param in self.segmenter.parameters():
            param.requires_grad_(True)

    @staticmethod
    def _squeeze_optional_temporal_axis(value: Tensor) -> Tensor:
        if value.ndim >= 3 and value.shape[1] == 1:
            return value.squeeze(1)
        return value

    @staticmethod
    def _select_points(
        point_cloud: Tensor,
        scores: Tensor,
        count: int,
        *,
        largest: bool,
        point_is_pad: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        return_has_candidates: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        if point_cloud.ndim != 3:
            raise ValueError(f"Expected point_cloud shape (B,N,C), got {point_cloud.shape}.")
        if scores.shape != point_cloud.shape[:2]:
            raise ValueError(f"Expected scores shape {point_cloud.shape[:2]}, got {scores.shape}.")
        if candidate_mask is not None and candidate_mask.shape != point_cloud.shape[:2]:
            raise ValueError(f"Expected candidate_mask shape {point_cloud.shape[:2]}, got {candidate_mask.shape}.")

        bsize, n_points, channels = point_cloud.shape
        if n_points <= 0:
            raise ValueError("Cannot select points from an empty point cloud.")
        count = max(1, int(count))
        scores = scores.to(device=point_cloud.device)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(device=point_cloud.device, dtype=torch.bool)

        if point_is_pad is None:
            valid_mask = torch.ones(bsize, n_points, dtype=torch.bool, device=point_cloud.device)
        else:
            point_is_pad = point_is_pad.to(device=point_cloud.device, dtype=torch.bool)
            if point_is_pad.shape != point_cloud.shape[:2]:
                raise ValueError(f"Expected point_is_pad shape {point_cloud.shape[:2]}, got {point_is_pad.shape}.")
            valid_mask = ~point_is_pad

        arange = torch.arange(n_points, device=point_cloud.device)
        first_valid = torch.where(valid_mask, arange.unsqueeze(0), n_points).argmin(dim=1)
        fallback_mask = torch.zeros_like(valid_mask)
        fallback_mask.scatter_(1, first_valid[:, None], True)

        if candidate_mask is None:
            source_mask = torch.where(valid_mask.any(dim=1, keepdim=True), valid_mask, fallback_mask)
            has_candidates = torch.ones(bsize, dtype=torch.bool, device=point_cloud.device)
        else:
            candidate_source_mask = torch.where(valid_mask.any(dim=1, keepdim=True), valid_mask, fallback_mask)
            candidate_mask = candidate_mask & candidate_source_mask
            has_candidates = candidate_mask.any(dim=1)
            source_mask = torch.where(has_candidates[:, None], candidate_mask, fallback_mask)

        masked_scores = scores.masked_fill(~source_mask, -torch.inf if largest else torch.inf)
        topk_count = min(count, n_points)
        top_indices = torch.topk(masked_scores, k=topk_count, dim=1, largest=largest).indices
        source_count = source_mask.sum(dim=1).clamp_min(1)
        gather_ranks = torch.arange(count, device=point_cloud.device).unsqueeze(0) % source_count.unsqueeze(1)
        indices = top_indices.gather(1, gather_ranks)

        selected = point_cloud.gather(1, indices[..., None].expand(bsize, count, channels))
        selected_scores = scores.gather(1, indices)
        if return_has_candidates:
            return selected, selected_scores, has_candidates
        return selected, selected_scores

    def _target_count(self, n_points: int, ratio: float, minimum: int) -> int:
        return max(int(minimum), math.ceil(n_points * float(ratio)))

    def _make_pseudo(self, payload: dict[str, Tensor], priors: Tensor | None) -> dict[str, Tensor] | None:
        labels = payload.get("pointseg.labels")
        weights = payload.get("pointseg.weights")
        class_scores = payload.get("pointseg.class_scores")
        if labels is not None and weights is not None and class_scores is not None:
            labels = self._squeeze_optional_temporal_axis(labels).to(dtype=torch.long)
            pseudo_priors = (
                priors
                if priors is not None
                else torch.zeros(*labels.shape, MOTION_PRIOR_DIM, device=labels.device, dtype=torch.float32)
            )
            pseudo = {
                "priors": pseudo_priors,
                "labels": labels,
                "weights": self._squeeze_optional_temporal_axis(weights).to(dtype=torch.float32),
                "class_scores": self._squeeze_optional_temporal_axis(class_scores).to(dtype=torch.float32),
            }
            if "pointseg.foreground_score" in payload:
                pseudo["foreground_score"] = self._squeeze_optional_temporal_axis(
                    payload["pointseg.foreground_score"]
                ).to(dtype=torch.float32)
            else:
                pseudo["foreground_score"] = pseudo["class_scores"][..., ROLE_FOREGROUND]
            if "pointseg.role_scores" in payload:
                pseudo["role_scores"] = self._squeeze_optional_temporal_axis(payload["pointseg.role_scores"]).to(
                    dtype=torch.float32
                )
            return pseudo
        if priors is None:
            return None
        return generate_pseudo_labels_from_priors(priors, config=self.pseudo_config)

    def _get_temporal_priors(self, payload: dict[str, Tensor], point_cloud: Tensor) -> Tensor | None:
        priors = payload.get("pointseg.priors")
        if priors is None:
            return None
        priors = self._squeeze_optional_temporal_axis(priors).to(device=point_cloud.device, dtype=torch.float32)
        if priors.shape[:2] != point_cloud.shape[:2] or priors.shape[-1] != MOTION_PRIOR_DIM:
            raise ValueError(
                f"Expected pointseg.priors shape (B,N,{MOTION_PRIOR_DIM}) matching point cloud, "
                f"got {priors.shape} for point cloud {point_cloud.shape}."
            )
        return priors

    def _selection_scores(self, operation_prob: Tensor) -> Tensor:
        return operation_prob

    def _background_candidate_mask(
        self,
        selection_scores: Tensor,
        point_is_pad: Tensor | None = None,
    ) -> Tensor:
        candidate = selection_scores <= 0.5
        if point_is_pad is not None:
            candidate = candidate & ~point_is_pad.to(device=selection_scores.device, dtype=torch.bool)
        return candidate

    def forward(self, payload: dict[str, Tensor]) -> dict[str, Tensor]:
        point_cloud = payload["point_cloud"].to(dtype=torch.float32)
        point_is_pad = payload.get("point_is_pad")
        if point_is_pad is not None:
            point_is_pad = point_is_pad.to(device=point_cloud.device, dtype=torch.bool)
            if point_is_pad.ndim == 3 and point_is_pad.shape[1] == 1:
                point_is_pad = point_is_pad.squeeze(1)
        temporal_priors = self._get_temporal_priors(payload, point_cloud)
        pseudo = self._make_pseudo(payload, temporal_priors)

        with _batchnorm_eval_on_single_value(self.segmenter):
            seg_outputs = self.segmenter(point_cloud, priors=temporal_priors, point_is_pad=point_is_pad)
        operation_prob = seg_outputs["operation_prob"]
        selection_scores = self._selection_scores(operation_prob)
        valid_points = (
            (~point_is_pad).sum(dim=1).max().item()
            if point_is_pad is not None
            else point_cloud.shape[1]
        )
        foreground_count = self._target_count(int(valid_points), self.foreground_ratio, self.min_foreground_points)
        background_count = self._target_count(int(valid_points), self.background_ratio, self.min_background_points)
        foreground_pc, foreground_prob = self._select_points(
            point_cloud, selection_scores, foreground_count, largest=True, point_is_pad=point_is_pad
        )
        background_candidate_mask = self._background_candidate_mask(selection_scores, point_is_pad)
        background_pc, background_prob, background_has_candidates = self._select_points(
            point_cloud,
            selection_scores,
            background_count,
            largest=False,
            point_is_pad=point_is_pad,
            candidate_mask=background_candidate_mask,
            return_has_candidates=True,
        )

        fg_weight = foreground_prob
        fg_weight = torch.where(fg_weight >= 0.1, fg_weight, torch.zeros_like(fg_weight))
        bg_weight = 1.0 - background_prob
        bg_weight = torch.where(bg_weight >= 0.1, bg_weight, torch.zeros_like(bg_weight))

        with _batchnorm_eval_on_single_value(self.foreground_encoder):
            foreground_encoded = self.foreground_encoder(
                foreground_pc,
                return_tokens=bool(self.config.point_action_fusion_enable),
            )
        if isinstance(foreground_encoded, dict):
            object_feat = foreground_encoded["global_feat"]
        else:
            object_feat = foreground_encoded
        background_feat = self.null_background_feat.to(
            device=object_feat.device, dtype=object_feat.dtype
        ).unsqueeze(0).expand(point_cloud.shape[0], -1).clone()
        if bool(background_has_candidates.any().item()):
            background_pc_to_encode = background_pc[background_has_candidates]
            with _batchnorm_eval_on_single_value(self.background_encoder):
                _scene_xyz, _scene_tok, encoded_background_feat, _scene_mask = self.background_encoder(
                    background_pc_to_encode
                )
            background_feat[background_has_candidates] = encoded_background_feat.to(dtype=background_feat.dtype)

        result = {
            "object_feat": object_feat,
            "background_feat": background_feat,
            "foreground_pc": foreground_pc,
            "background_pc": background_pc,
            "foreground_prob": foreground_prob,
            "background_prob": background_prob,
            "operation_prob": operation_prob,
            "pointseg_outputs": seg_outputs,
            "pointseg_priors": temporal_priors,
            "pointseg_selection_scores": selection_scores,
            "pointseg_background_has_candidates": background_has_candidates,
        }
        if isinstance(foreground_encoded, dict):
            result["foreground_scene_tok1"] = foreground_encoded["scene_tok1"]
            result["foreground_scene_mask1"] = foreground_encoded["scene_mask1"]
        if pseudo is not None and "role_scores" in pseudo:
            result["role_scores"] = pseudo["role_scores"].to(device=point_cloud.device, dtype=torch.float32)
        if self.training and self.aux_loss_weight > 0:
            if pseudo is not None:
                if point_is_pad is not None:
                    pseudo["point_is_pad"] = point_is_pad
                aux_loss, aux_metrics = self.pointseg_loss(seg_outputs, pseudo, point_cloud)
                result["pointseg_aux_loss"] = aux_loss
                result["pointseg_aux_metrics"] = aux_metrics
        return result


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            vlm_weights_path=self.config.vlm_weights_path,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
        self.se3_action_out_proj = (
            nn.Linear(self.vlm_with_expert.expert_hidden_size, 7)
            if self.config.se3_enable and self.config.se3_twist_head_mode == "direct_twist"
            else None
        )
        use_pointseg = self.config.pointseg_enable or self.config.pointseg_checkpoint_path is not None
        use_primary_residual = self.config.camera_view_fusion == "primary_residual"
        if use_primary_residual and not use_pointseg:
            raise ValueError(
                "camera_view_fusion='primary_residual' requires PointSeg so both views use the "
                "same geometry encoder before residual fusion."
            )
        if self.config.worldflow_enable and not use_pointseg:
            raise ValueError(
                "worldflow_enable requires pointseg_enable=True or pointseg_checkpoint_path because "
                "WorldFlow uses SongPointCloudConditioner predicted foreground points as its scene input."
            )
        if self.config.worldflow_enable and self.config.max_action_dim < 9:
            raise ValueError("worldflow_enable=True requires pose9 Ego actions (max_action_dim >= 9).")
        if self.config.se3_enable and self.config.max_action_dim < 10:
            raise ValueError("se3_enable=True requires max_action_dim >= 10 for pose9 + gripper actions.")
        if self.config.se3_enable and self._rtc_enabled():
            raise ValueError("se3_enable=True is not supported with RTC enabled in v1.")
        self.extractor = None if use_pointseg else LitePTEncoder(in_dim=6, dim=64, n_tokens=256, grid_size=0.005)
        self.pointcloud_proj = (
            None if use_pointseg else nn.Linear(64, self.vlm_with_expert.config.text_config.hidden_size)
        )
        self.pointseg_conditioner = SongPointCloudConditioner(config) if use_pointseg else None
        self.pointseg_object_proj = (
            nn.Linear(self.config.pointseg_feature_dim, self.vlm_with_expert.config.text_config.hidden_size)
            if use_pointseg
            else None
        )
        self.pointseg_background_proj = (
            nn.Linear(self.config.pointseg_feature_dim, self.vlm_with_expert.config.text_config.hidden_size)
            if use_pointseg
            else None
        )
        self.multiview_residual_proj = (
            nn.Linear(
                2 * self.config.pointseg_feature_dim,
                2 * self.config.pointseg_feature_dim,
                bias=False,
            )
            if use_primary_residual
            else None
        )
        if self.multiview_residual_proj is not None:
            # Full-matrix residual adapter, not a scalar gate. At initialization
            # the primary path is exactly the loaded single-view checkpoint;
            # every matrix element is subsequently trainable.
            nn.init.zeros_(self.multiview_residual_proj.weight)
        self.point_action_fusion = (
            PointActionSelfAttention(
                action_dim=self.vlm_with_expert.expert_hidden_size,
                point_dim=self.config.pointseg_feature_dim,
                max_action_steps=self.config.chunk_size,
                num_heads=self.config.point_action_fusion_heads,
                dropout=self.config.point_action_fusion_dropout,
            )
            if use_pointseg and self.config.point_action_fusion_enable
            else None
        )
        self.last_pointseg_aux_loss: Tensor | None = None
        self.last_pointseg_metrics: dict[str, Tensor] = {}
        self.last_worldflow_payload: dict[str, Tensor] | None = None
        self.last_point_action_tokens: Tensor | None = None
        self.last_point_action_mask: Tensor | None = None
        self.last_ego_scene_global_feat: Tensor | None = None
        self.last_ego_scene_global_mask: Tensor | None = None
        self.last_body_pose9_prediction: Tensor | None = None
        self.last_worldflow_aux: dict[str, Tensor] | None = None
        # Runtime-only diagnostics. These are plain Python attributes so they
        # never alter checkpoints or normal training / inference behavior.
        self.inference_ablation_modalities: frozenset[str] = frozenset()
        self.capture_pointseg_visualization = False
        self.last_pointseg_visualization: dict[str, Tensor] | None = None
        self.last_worldflow_metrics: dict[str, Tensor] = {}
        # Per-sample stochastic-depth assignment for training-only optimizer
        # routing. This is runtime state, not a checkpoint tensor.
        self.last_worldflow_world_to_ego_keep_mask: Tensor | None = None
        self.last_se3_metrics: dict[str, Tensor] = {}
        self.last_action_metrics: dict[str, Tensor] = {}

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # Instantiate the auxiliary branch only after every Ego-path module so
        # toggling worldflow_enable cannot perturb Ego initialization through
        # additional RNG consumption.
        self.worldflow_branch = (
            WorldFlowActionBranch(
                config,
                action_hidden_dim=self.vlm_with_expert.expert_hidden_size,
                language_vocab_size=self.vlm_with_expert.config.text_config.vocab_size,
            )
            if self.config.worldflow_enable
            else None
        )
        if self.worldflow_branch is not None:
            expert_dim = self.vlm_with_expert.expert_hidden_size
            direct_world_trajectory = (
                getattr(self.config, "worldflow_target_type", "legacy_eef")
                == "world_eef_trajectory"
            )
            self.ego_scene_to_expert = nn.Linear(self.config.pointseg_feature_dim, expert_dim)
            self.world_scene_to_expert = nn.Linear(self.config.worldflow_feature_dim, expert_dim)
            self.world_action_out_proj = (
                None
                if direct_world_trajectory
                else nn.Linear(expert_dim, WorldFlowActionBranch.pose_dim)
            )
            self.world_se3_action_out_proj = (
                nn.Linear(expert_dim, 6)
                if direct_world_trajectory
                or self.config.se3_enable
                and (
                    self.config.se3_twist_head_mode == "direct_twist"
                    or self.config.worldflow_se3_head_enable
                )
                else None
            )
            if direct_world_trajectory:
                # A true independent World-trajectory checkpoint has no physical
                # residual head at all. World can affect Ego only through the
                # explicit token cross-attention path below.
                self.world_twist_residual_out_proj = None
            else:
                self.world_twist_residual_out_proj = nn.Linear(expert_dim, 6)
                nn.init.zeros_(self.world_twist_residual_out_proj.weight)
                nn.init.zeros_(self.world_twist_residual_out_proj.bias)
            # Explicit identities distinguish the appended scene and World
            # action blocks. Ego actions remain byte-for-byte unchanged for
            # checkpoint compatibility.
            self.world_ego_scene_type_embedding = nn.Parameter(torch.empty(2, expert_dim))
            self.world_ego_action_type_embedding = nn.Parameter(torch.empty(2, expert_dim))
            nn.init.normal_(self.world_ego_scene_type_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.world_ego_action_type_embedding, mean=0.0, std=0.02)
            cross_heads = int(self.config.point_action_fusion_heads)
            self.ego_to_world_cross_norm = nn.LayerNorm(expert_dim)
            self.world_to_ego_cross_norm = nn.LayerNorm(expert_dim)
            self.ego_to_world_cross_attn = nn.MultiheadAttention(
                expert_dim,
                cross_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.world_to_ego_cross_attn = nn.MultiheadAttention(
                expert_dim,
                cross_heads,
                dropout=0.0,
                batch_first=True,
            )
            # Function-preserving expansion: this full matrix starts at zero
            # and learns immediately. It is not a scalar gate, and neither
            # coordinate stream is frozen.
            nn.init.zeros_(self.world_to_ego_cross_attn.out_proj.weight)
            nn.init.zeros_(self.world_to_ego_cross_attn.out_proj.bias)
        else:
            self.ego_scene_to_expert = None
            self.world_scene_to_expert = None
            self.world_action_out_proj = None
            self.world_se3_action_out_proj = None
            self.world_twist_residual_out_proj = None
            self.ego_to_world_cross_norm = None
            self.world_to_ego_cross_norm = None
            self.ego_to_world_cross_attn = None
            self.world_to_ego_cross_attn = None
            self.register_parameter("world_ego_scene_type_embedding", None)
            self.register_parameter("world_ego_action_type_embedding", None)

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    @torch.no_grad()
    def bootstrap_worldflow_from_ego(self) -> dict[str, object]:
        """Initialize the World stream from compatible trained Ego modules.

        This is a one-shot transfer used before joint fine-tuning. It does not
        share or freeze parameters: every copied World tensor remains an
        independent trainable parameter. Both explicit cross-attention output
        projections start at zero so the transfer preserves the loaded Ego
        function while the two directions learn from their own objectives.
        """

        world = self.worldflow_branch
        if world is None:
            raise RuntimeError("WorldFlow must be enabled before bootstrapping from Ego.")
        if self.pointseg_conditioner is None or self.point_action_fusion is None:
            raise RuntimeError("WorldFlow bootstrap requires the trained Ego point modules.")
        required = (
            self.ego_scene_to_expert,
            self.world_scene_to_expert,
            self.world_action_out_proj,
            self.ego_to_world_cross_attn,
            self.world_to_ego_cross_attn,
        )
        if any(module is None for module in required):
            raise RuntimeError("WorldFlow bootstrap modules are not initialized.")

        world.scene_encoder.load_state_dict(
            self.pointseg_conditioner.foreground_encoder.state_dict(),
            strict=True,
        )
        world.point_action_adapter.load_state_dict(
            self.point_action_fusion.state_dict(),
            strict=True,
        )
        world.action_in_proj.weight.copy_(self.action_in_proj.weight[:, : WorldFlowActionBranch.pose_dim])
        world.action_in_proj.bias.copy_(self.action_in_proj.bias)
        world.action_time_mlp_in.load_state_dict(self.action_time_mlp_in.state_dict(), strict=True)
        world.action_time_mlp_out.load_state_dict(self.action_time_mlp_out.state_dict(), strict=True)
        world.scene_context_proj.load_state_dict(self.ego_scene_to_expert.state_dict(), strict=True)
        self.world_scene_to_expert.load_state_dict(self.ego_scene_to_expert.state_dict(), strict=True)
        self.world_action_out_proj.weight.copy_(
            self.action_out_proj.weight[: WorldFlowActionBranch.pose_dim]
        )
        self.world_action_out_proj.bias.copy_(
            self.action_out_proj.bias[: WorldFlowActionBranch.pose_dim]
        )

        # The VLM language table has a different hidden width. A zero language
        # residual is the identity-compatible initialization; it learns jointly.
        world.language_embedding.weight.zero_()
        world.language_norm.weight.fill_(1.0)
        world.language_norm.bias.zero_()
        self.world_ego_scene_type_embedding.zero_()
        self.world_ego_action_type_embedding.zero_()
        for attention in (self.ego_to_world_cross_attn, self.world_to_ego_cross_attn):
            attention.out_proj.weight.zero_()
            if attention.out_proj.bias is not None:
                attention.out_proj.bias.zero_()
        if self.world_twist_residual_out_proj is not None:
            self.world_twist_residual_out_proj.weight.zero_()
            self.world_twist_residual_out_proj.bias.zero_()
        if self.world_se3_action_out_proj is not None and self.config.worldflow_se3_head_enable:
            self.world_se3_action_out_proj.weight.zero_()
            self.world_se3_action_out_proj.bias.zero_()

        return {
            "status": "bootstrapped",
            "source": "trained_ego_action_point_modules",
            "world_parameters_shared": False,
            "ego_frozen": False,
            "bidirectional_cross_attention_zero_output_init": True,
        }

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = bool(self.config.encode_robot_state and self.config.train_state_proj)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )/10
        return noise

    def sample_worldflow_noise(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample valid SE(3) pose9 noise for one World action chunk."""
        twist = torch.randn(
            int(batch_size),
            self.config.chunk_size,
            6,
            device=device,
            dtype=torch.float32,
        )
        twist[..., :3] = twist[..., :3] * float(self.config.worldflow_noise_trans_scale)
        twist[..., 3:6] = twist[..., 3:6] * float(self.config.worldflow_noise_rot_scale)
        return matrix_to_pose9(se3_exp(twist))

    def sample_pose9_action_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample valid SE(3) pose9 + gripper noise for the standard Ego flow."""

        if len(shape) != 3 or shape[-1] < 10:
            raise ValueError(f"Expected Ego action noise shape [B,T,D>=10], got {shape}.")
        twist = torch.randn(*shape[:2], 6, device=device, dtype=torch.float32)
        twist[..., :3] *= float(self.config.pose9_action_noise_trans_scale)
        twist[..., 3:6] *= float(self.config.pose9_action_noise_rot_scale)
        gripper = torch.randn(*shape[:2], 1, device=device, dtype=torch.float32)
        gripper *= float(self.config.pose9_action_noise_gripper_scale)
        noise = torch.cat([matrix_to_pose9(se3_exp(twist)), gripper], dim=-1)
        if shape[-1] > 10:
            noise = pad_vector(noise, shape[-1])
        return noise

    def conjugate_ego_noise_to_world(self, ego_noise: Tensor, current_pose: Tensor) -> Tensor:
        """Derive the World spatial prior from the Ego body prior.

        For current EEF-to-World carrier ``C`` and Ego/body transform ``B``,
        the corresponding World/spatial transform is ``G = C B C^{-1}``.
        Keeping this relation at the random-flow origin prevents the two
        streams from describing different physical noisy trajectories.
        """

        if ego_noise.ndim != 3 or ego_noise.shape[-1] < 9:
            raise ValueError(f"Expected Ego noise shape (B,T,D>=9), got {ego_noise.shape}.")
        if current_pose.ndim != 2 or current_pose.shape != (ego_noise.shape[0], 9):
            raise ValueError(
                f"Expected current pose shape {(ego_noise.shape[0], 9)}, got {current_pose.shape}."
            )
        ego_transform = pose9_to_matrix(ego_noise[..., :9].to(dtype=torch.float32))
        current = _worldflow_carrier_matrix(
            current_pose.to(device=ego_noise.device, dtype=torch.float32),
            self.config.worldflow_frame_origin,
        )
        current_inv = invert_transform(current)
        world_transform = (
            current.unsqueeze(1)
            @ ego_transform
            @ current_inv.unsqueeze(1)
        )
        return matrix_to_pose9(world_transform)

    def project_ego_chart_path_to_world(
        self,
        ego_x_t: Tensor,
        current_pose: Tensor,
        spatial_gt_pose9: Tensor,
        time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build an exactly co-located World state and endpoint velocity.

        The pretrained Ego expert keeps its original Euclidean chart path. At
        every time, that chart is projected to SE(3) and conjugated into World.
        The World velocity is defined by the same endpoint convention used by
        the pose9 flow head, so ``x_t + (1-t) u_t`` reaches the World target.
        """

        world_x_t = self.conjugate_ego_noise_to_world(ego_x_t, current_pose)
        remaining = (1.0 - time).clamp_min(1e-4)[:, None, None]
        world_u_t = (spatial_gt_pose9 - world_x_t) / remaining
        return world_x_t, world_u_t

    def sample_time(self, bsize, device):
        mode = self.config.flow_time_sampling
        if mode == "integration_grid":
            num_steps = int(self.config.num_steps)
            zero_probability = float(self.config.flow_time_zero_probability)
            if num_steps == 1:
                return torch.zeros(int(bsize), device=device, dtype=torch.float32)
            if zero_probability > 0:
                choose_zero = torch.rand(int(bsize), device=device) < zero_probability
                step = torch.randint(
                    low=1,
                    high=num_steps,
                    size=(int(bsize),),
                    device=device,
                )
                step = torch.where(choose_zero, torch.zeros_like(step), step)
                return step.to(dtype=torch.float32) / float(num_steps)
            step = torch.randint(
                low=0,
                high=num_steps,
                size=(int(bsize),),
                device=device,
            )
            return step.to(dtype=torch.float32) / float(num_steps)
        if mode == "uniform":
            return torch.rand(int(bsize), device=device, dtype=torch.float32) * 0.999
        if mode != "beta":
            raise ValueError(f"Unsupported flow_time_sampling={mode!r}.")
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def sample_se3_action_noise(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if actions.shape[-1] < 10:
            raise ValueError(f"se3_enable=True expects action dim >= 10, got {actions.shape[-1]}.")
        if self.config.se3_twist_head_mode == "pose9_chart_endpoint":
            # Preserve the exact v0.4.2 Action Expert input convention: the
            # pose9 chart starts near zero rather than at the unit columns of
            # a rotation matrix. The physical state is the SE(3) projection of
            # that chart and is carried separately throughout the flow.
            noise_action = torch.randn(
                *actions.shape,
                device=actions.device,
                dtype=torch.float32,
            )
            noise_action[..., :3] *= float(self.config.se3_noise_trans_scale)
            noise_action[..., 3:9] *= float(self.config.se3_noise_rot_scale)
            noise_action[..., 9:10] *= float(self.config.se3_noise_gripper_scale)
            if actions.shape[-1] > 10:
                noise_action[..., 10:] = 0.0
            pose_noise = pose9_to_matrix(noise_action[..., :9])
            return pose_noise, noise_action[..., 9:10], noise_action
        xi_noise = torch.randn(*actions.shape[:2], 6, device=actions.device, dtype=torch.float32)
        xi_noise[..., :3] = xi_noise[..., :3] * float(self.config.se3_noise_trans_scale)
        xi_noise[..., 3:6] = xi_noise[..., 3:6] * float(self.config.se3_noise_rot_scale)
        pose_noise = se3_exp(xi_noise)
        gripper_noise = torch.randn(
            *actions.shape[:2],
            1,
            device=actions.device,
            dtype=torch.float32,
        ) * float(self.config.se3_noise_gripper_scale)
        noise_action = torch.cat([matrix_to_pose9(pose_noise), gripper_noise], dim=-1)
        if actions.shape[-1] > 10:
            noise_action = pad_vector(noise_action, actions.shape[-1])
        return pose_noise, gripper_noise, noise_action

    @staticmethod
    def _masked_scalar_mean(values: Tensor, valid: Tensor | None = None) -> Tensor:
        if valid is None:
            return values.mean()
        valid_f = valid.to(device=values.device, dtype=values.dtype)
        return (values * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def _inject_point_action_features(
        self,
        suffix_embs: Tensor,
        actions_is_pad: Tensor | None = None,
    ) -> Tensor:
        if self.point_action_fusion is None:
            return suffix_embs
        if self.last_point_action_tokens is None:
            return suffix_embs
        return self.point_action_fusion(
            action_tokens=suffix_embs,
            point_tokens=self.last_point_action_tokens,
            point_mask=self.last_point_action_mask,
            actions_is_pad=actions_is_pad,
        )

    def _record_standard_action_metrics(
        self,
        *,
        actions: Tensor,
        x_t: Tensor,
        u_t: Tensor,
        pred_velocity: Tensor,
        time: Tensor,
        actions_is_pad: Tensor | None,
    ) -> None:
        """Record physically interpretable diagnostics for pose9 flow matching."""

        self.last_action_metrics = {}
        if min(actions.shape[-1], x_t.shape[-1], pred_velocity.shape[-1]) < 10:
            return
        valid = None
        if torch.is_tensor(actions_is_pad):
            valid = ~actions_is_pad.to(device=actions.device, dtype=torch.bool)

        flow_error = (pred_velocity - u_t).square()
        endpoint = x_t + (1.0 - time[:, None, None]) * pred_velocity
        endpoint_pose = pose9_to_matrix(endpoint[..., :9])
        target_pose = pose9_to_matrix(actions[..., :9])

        self.last_action_metrics = {
            "loss_action_translation": self._masked_scalar_mean(flow_error[..., :3].mean(-1), valid)
            .detach(),
            "loss_action_rotation6d": self._masked_scalar_mean(flow_error[..., 3:9].mean(-1), valid)
            .detach(),
            "loss_action_gripper": self._masked_scalar_mean(flow_error[..., 9], valid).detach(),
            "action_endpoint_trans_err": self._masked_scalar_mean(
                torch.linalg.norm(endpoint_pose[..., :3, 3] - target_pose[..., :3, 3], dim=-1),
                valid,
            ).detach(),
            "action_endpoint_rot_err_deg": torch.rad2deg(
                self._masked_scalar_mean(
                    _rotation_geodesic(endpoint_pose[..., :3, :3], target_pose[..., :3, :3]),
                    valid,
                )
            ).detach(),
            "action_endpoint_gripper_err": self._masked_scalar_mean(
                (endpoint[..., 9] - actions[..., 9]).abs(),
                valid,
            ).detach(),
        }

    def _se3_predict_from_suffix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        x_t: Tensor,
        time: Tensor,
        actions_is_pad: Tensor | None = None,
        ego_group_x_t: Tensor | None = None,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t,
            time,
            actions_is_pad=actions_is_pad,
        )
        suffix_embs = self._inject_point_action_features(suffix_embs, actions_is_pad=actions_is_pad)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self._predict_ego_se3_velocity(
            suffix_out,
            x_t,
            time,
            ego_group_x_t=ego_group_x_t,
        )

    def _run_ego_suffix_expert(
        self,
        prefix_embs: Tensor | None,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor | None,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        *,
        past_key_values=None,
    ) -> Tensor:
        """Run the checkpoint-compatible Ego-only Action Expert path."""

        if prefix_embs is not None:
            if prefix_att_masks is None:
                raise ValueError("prefix_att_masks are required when prefix_embs are provided.")
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            attention_mask = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
        else:
            if past_key_values is None:
                raise ValueError("Cached Ego expert inference requires past_key_values.")
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_attention = prefix_pad_masks[:, None, :].expand(
                prefix_pad_masks.shape[0], suffix_len, prefix_len
            )
            suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            attention_mask = torch.cat([prefix_attention, suffix_attention], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1]
        return suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)

    def _predict_ego_se3_velocity(
        self,
        expert_out: Tensor,
        ego_x_t: Tensor,
        time: Tensor,
        *,
        ego_group_x_t: Tensor | None = None,
        return_pose9_velocity: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Decode Ego expert features as spatial twist plus gripper velocity."""

        if self.config.se3_twist_head_mode == "direct_twist":
            if self.se3_action_out_proj is None:
                raise RuntimeError("Direct Ego SE(3) twist head is not initialized.")
            return self.se3_action_out_proj(expert_out)
        if self.config.se3_twist_head_mode not in {
            "projected_pose9",
            "pose9_endpoint",
            "pose9_chart_endpoint",
        }:
            raise RuntimeError(f"Unknown se3_twist_head_mode={self.config.se3_twist_head_mode!r}.")
        pose9_velocity = self.action_out_proj(expert_out)
        if pose9_velocity.shape[-1] < 10:
            raise RuntimeError("Projected Ego SE(3) mode requires a pose9 + gripper output head.")
        if self.config.se3_twist_head_mode in {"pose9_endpoint", "pose9_chart_endpoint"}:
            current_group = ego_x_t if ego_group_x_t is None else ego_group_x_t
            endpoint_base = ego_x_t if self.config.se3_twist_head_mode == "pose9_chart_endpoint" else None
            twist = pose9_endpoint_velocity_to_spatial_twist(
                current_group,
                pose9_velocity,
                time,
                endpoint_base_pose9=endpoint_base,
            )
        else:
            current_group = ego_x_t if ego_group_x_t is None else ego_group_x_t
            twist = pose9_velocity_to_spatial_twist(current_group, pose9_velocity)
        velocity = torch.cat([twist, pose9_velocity[..., 9:10]], dim=-1)
        if return_pose9_velocity:
            return velocity, pose9_velocity
        return velocity

    def _predict_world_se3_velocity(
        self,
        expert_out: Tensor,
        world_x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Decode World expert features as a spatial twist."""

        if (
            getattr(self.config, "worldflow_target_type", "legacy_eef")
            == "world_eef_trajectory"
        ):
            if self.world_se3_action_out_proj is None:
                raise RuntimeError("World-EEF trajectory SE(3) twist head is not initialized.")
            return self.world_se3_action_out_proj(expert_out)
        if self.config.worldflow_se3_head_enable:
            if self.world_se3_action_out_proj is None:
                raise RuntimeError("Dedicated World SE(3) twist head is not initialized.")
            return self.world_se3_action_out_proj(expert_out)
        if self.config.se3_twist_head_mode == "direct_twist":
            if self.world_se3_action_out_proj is None:
                raise RuntimeError("Direct World SE(3) twist head is not initialized.")
            return self.world_se3_action_out_proj(expert_out)
        if self.config.se3_twist_head_mode not in {
            "projected_pose9",
            "pose9_endpoint",
            "pose9_chart_endpoint",
        }:
            raise RuntimeError(f"Unknown se3_twist_head_mode={self.config.se3_twist_head_mode!r}.")
        if self.world_action_out_proj is None:
            raise RuntimeError("Projected World SE(3) mode requires the pose9 World output head.")
        pose9_velocity = self.world_action_out_proj(expert_out)
        if self.config.se3_twist_head_mode in {"pose9_endpoint", "pose9_chart_endpoint"}:
            return pose9_endpoint_velocity_to_spatial_twist(world_x_t, pose9_velocity, time)
        return pose9_velocity_to_spatial_twist(world_x_t, pose9_velocity)

    def _compose_conjugate_residual_world_velocity(
        self,
        ego_velocity: Tensor,
        world_expert_out: Tensor,
        ego_to_world_transform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compose the World score from the conjugated Ego score and World residual.

        This is the single decoder used by ordinary training, augmented-frame
        equivariance training, and online denoising. Keeping those paths on the
        same decoder prevents an auxiliary loss from silently optimizing the
        legacy independent World pose-chart head.
        """

        if self.world_twist_residual_out_proj is None:
            raise RuntimeError("World residual twist head is not initialized.")
        if ego_velocity.shape[-1] < 6:
            raise ValueError(f"Expected Ego velocity dim >= 6, got {ego_velocity.shape}.")
        residual = self.world_twist_residual_out_proj(world_expert_out)
        world_velocity = transform_se3_twist(
            ego_velocity[..., :6],
            ego_to_world_transform,
        ) + residual
        return world_velocity, residual

    def _compose_conjugate_residual_consensus(
        self,
        ego_velocity: Tensor,
        world_expert_out: Tensor,
        ego_to_world_transform: Tensor,
        world_to_ego_transform: Tensor,
        *,
        detach_ego_for_world_supervision: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build an exact, fixed 1:1 World/Ego residual consensus.

        Let ``p`` be the Ego twist and ``r`` the World-head residual. The old
        residual path returned ``Ad_C(p) + r`` for World while correcting Ego
        by only ``0.5 Ad_C^-1(r)``. Those outputs were not conjugate, so the
        bridge objective could reduce its loss by suppressing ``r``. Here the
        two descriptions meet at the same physical midpoint::

            q_ego   = p + 0.5 Ad_C^-1(r)
            q_world = Ad_C(q_ego) = Ad_C(p) + 0.5 r

        The residual head and all checkpoint keys are unchanged. Both paths
        remain jointly trainable; there is no learned gate or frozen branch.
        """

        _world_candidate, residual = self._compose_conjugate_residual_world_velocity(
            ego_velocity,
            world_expert_out,
            ego_to_world_transform,
        )
        corrected_ego_velocity = torch.cat(
            [
                ego_velocity[..., :6]
                + 0.5 * transform_se3_twist(residual, world_to_ego_transform),
                ego_velocity[..., 6:],
            ],
            dim=-1,
        )
        world_ego_twist = corrected_ego_velocity[..., :6]
        if detach_ego_for_world_supervision:
            # Preserve the exact forward value while preventing the World
            # auxiliary objective from improving by moving the direct Ego
            # score in the opposite direction. Gradients through the residual
            # and its bidirectional expert context remain intact. The main Ego
            # action objective still differentiates corrected_ego_velocity, so
            # neither coordinate branch is frozen.
            world_ego_twist = (
                world_ego_twist
                - ego_velocity[..., :6]
                + ego_velocity[..., :6].detach()
            )
        consensus_world_velocity = transform_se3_twist(
            world_ego_twist,
            ego_to_world_transform,
        )
        return corrected_ego_velocity, consensus_world_velocity, residual

    def _compose_endpoint_geodesic_consensus(
        self,
        ego_x_t: Tensor,
        ego_velocity: Tensor,
        world_x_t: Tensor,
        world_velocity: Tensor,
        time: Tensor,
        world_to_ego_transform: Tensor,
        ego_to_world_transform: Tensor,
    ) -> Tensor:
        """Fuse pose9 flow endpoints as a coordinate-invariant SE(3) midpoint.

        The legacy Ego chart and its velocity are left intact up to the final
        physical consensus.  At the current flow time, both streams predict an
        endpoint.  The World endpoint is conjugated back into the Ego frame and
        the fixed 1:1 geodesic midpoint is used as the common endpoint.  The
        midpoint is then represented as a legacy pose9 velocity so integration
        remains byte-compatible with the pretrained Ego sampler.

        This operation has no parameters, confidence heuristic, or task rule.
        Gradients from the Ego action objective reach both endpoint predictors.
        """

        if ego_x_t.shape[:-1] != ego_velocity.shape[:-1] or ego_x_t.shape[-1] < 9:
            raise ValueError(
                f"Expected matching Ego (...,>=9) state/velocity, got {ego_x_t.shape} "
                f"and {ego_velocity.shape}."
            )
        if world_x_t.shape != world_velocity.shape or world_x_t.shape[-1] != 9:
            raise ValueError(
                f"Expected matching World (...,9) state/velocity, got {world_x_t.shape} "
                f"and {world_velocity.shape}."
            )
        if ego_x_t.shape[:-1] != world_x_t.shape[:-1]:
            raise ValueError(
                f"Ego and World flow layouts must match, got {ego_x_t.shape} and {world_x_t.shape}."
            )
        if time.ndim != 1 or time.shape[0] != ego_x_t.shape[0]:
            raise ValueError(f"Expected time shape ({ego_x_t.shape[0]},), got {time.shape}.")

        remaining = (1.0 - time).clamp_min(1e-4)[:, None, None]
        ego_endpoint = pose9_to_matrix(
            ego_x_t[..., :9].to(dtype=torch.float32)
            + remaining * ego_velocity[..., :9].to(dtype=torch.float32)
        )
        world_endpoint = pose9_to_matrix(
            world_x_t.to(dtype=torch.float32)
            + remaining * world_velocity.to(dtype=torch.float32)
        )
        world_endpoint_in_ego = (
            world_to_ego_transform @ world_endpoint @ ego_to_world_transform
        )
        ego_to_world_endpoint = world_endpoint_in_ego @ invert_transform(ego_endpoint)
        midpoint = se3_exp(0.5 * se3_log(ego_to_world_endpoint)) @ ego_endpoint
        midpoint_pose9 = matrix_to_pose9(midpoint)
        fused_pose9_velocity = (
            midpoint_pose9 - ego_x_t[..., :9].to(dtype=torch.float32)
        ) / remaining
        return torch.cat(
            [fused_pose9_velocity.to(dtype=ego_velocity.dtype), ego_velocity[..., 9:]],
            dim=-1,
        )

    def _compose_endpoint_residual_boosting(
        self,
        ego_x_t: Tensor,
        ego_velocity: Tensor,
        world_x_t: Tensor,
        world_features: Tensor,
        time: Tensor,
        ego_to_world_transform: Tensor,
        world_to_ego_transform: Tensor,
        *,
        apply_world_to_ego: bool = True,
        detach_ego_for_world_supervision: bool = False,
        world_to_ego_keep_mask: Tensor | None = None,
        detach_retained_ego_anchor: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict only the physical endpoint error left by the Ego policy.

        The zero residual is exactly the legacy Ego function. By default a
        learned spatial World residual left-multiplies the conjugated Ego
        endpoint. The optional carrier-frame parameterization applies the same
        six output values before conjugation. The optional body-frame
        parameterization right-multiplies the predicted Ego endpoint, so the
        correction follows the endpoint's own axes. The optional body-velocity
        parameterization instead lifts a right-trivialized tangent at the
        current Ego Flow state directly into the pretrained pose9 vector
        field. All alternatives are invariant to an arbitrary
        reparameterization of World coordinates. The same corrected physical
        motion supervises both descriptions.
        """

        if self.world_twist_residual_out_proj is None:
            raise RuntimeError("Endpoint residual boosting requires the World residual twist head.")
        if ego_x_t.shape[:-1] != ego_velocity.shape[:-1] or ego_x_t.shape[-1] < 9:
            raise ValueError(
                f"Expected matching Ego (...,>=9) state/velocity, got {ego_x_t.shape} "
                f"and {ego_velocity.shape}."
            )
        if world_x_t.shape[-1] != 9 or world_x_t.shape[:-1] != ego_x_t.shape[:-1]:
            raise ValueError(
                f"Expected World state matching Ego with dim 9, got {world_x_t.shape}."
            )
        if time.ndim != 1 or time.shape[0] != ego_x_t.shape[0]:
            raise ValueError(f"Expected time shape ({ego_x_t.shape[0]},), got {time.shape}.")

        keep = None
        if world_to_ego_keep_mask is not None:
            keep = world_to_ego_keep_mask.to(device=ego_velocity.device, dtype=torch.bool)
            if keep.shape != (ego_velocity.shape[0],):
                raise ValueError(
                    "Expected one World-to-Ego keep decision per batch item, "
                    f"got {tuple(keep.shape)} for batch {ego_velocity.shape[0]}."
                )
        if detach_retained_ego_anchor:
            if keep is None:
                raise ValueError(
                    "detach_retained_ego_anchor requires a per-sample World-to-Ego keep mask."
                )
            # Preserve the exact forward value while routing the corrected
            # samples' action gradient into the residual booster. The dropped
            # samples still optimize the ordinary Ego prediction below.
            ego_velocity_anchor = torch.where(
                keep[:, None, None],
                ego_velocity.detach(),
                ego_velocity,
            )
        else:
            ego_velocity_anchor = ego_velocity

        remaining = (1.0 - time).clamp_min(1e-4)[:, None, None]
        ego_endpoint = pose9_to_matrix(
            ego_x_t[..., :9].to(dtype=torch.float32)
            + remaining * ego_velocity_anchor[..., :9].to(dtype=torch.float32)
        )
        baseline_world_endpoint = (
            ego_to_world_transform @ ego_endpoint @ world_to_ego_transform
        )
        residual = self.world_twist_residual_out_proj(world_features)
        residual_is_body_velocity = bool(
            getattr(
                getattr(self, "config", None),
                "worldflow_body_velocity_residual_parameterization",
                False,
            )
        )
        residual_has_endpoint_rate_boundary = bool(
            getattr(
                getattr(self, "config", None),
                "worldflow_endpoint_residual_rate_parameterization",
                False,
            )
        )
        if residual_has_endpoint_rate_boundary:
            # A bounded flow-velocity error induces an endpoint error that
            # contracts with the remaining integration horizon. Encode that
            # terminal boundary analytically instead of asking the World
            # network to relearn it from a finite time grid. The same
            # effective physical twist supervises World and returns to Ego.
            residual = residual * remaining.to(dtype=residual.dtype)
        residual_in_ego_frame = bool(
            getattr(
                getattr(self, "config", None),
                "worldflow_endpoint_residual_ego_frame_parameterization",
                False,
            )
        )
        residual_in_body_frame = bool(
            getattr(
                getattr(self, "config", None),
                "worldflow_endpoint_residual_body_frame_parameterization",
                False,
            )
        )
        if residual_is_body_velocity and (
            residual_has_endpoint_rate_boundary
            or residual_in_ego_frame
            or residual_in_body_frame
        ):
            raise ValueError(
                "Body-velocity residuals are mutually exclusive with finite endpoint-residual "
                "rate and frame parameterizations."
            )
        if residual_in_ego_frame and residual_in_body_frame:
            raise ValueError(
                "Carrier-frame and body-frame endpoint residual parameterizations are mutually exclusive."
            )
        if residual_is_body_velocity:
            # The World head predicts a right-trivialized velocity at the
            # *current* Ego Flow state.  Lift that physical tangent into the
            # same raw pose9 gauge as the pretrained Euclidean vector field;
            # no finite endpoint or 1/(1-t) reconstruction is involved.
            pose9_delta = body_twist_to_pose9_velocity(
                ego_x_t[..., :9],
                residual,
            ).to(dtype=ego_velocity_anchor.dtype)
            corrected_ego_velocity = torch.cat(
                [
                    ego_velocity_anchor[..., :9] + pose9_delta,
                    ego_velocity_anchor[..., 9:],
                ],
                dim=-1,
            )
            corrected_ego_endpoint = pose9_to_matrix(
                ego_x_t[..., :9].to(dtype=torch.float32)
                + remaining * corrected_ego_velocity[..., :9].to(dtype=torch.float32)
            )
            corrected_world_endpoint = (
                ego_to_world_transform @ corrected_ego_endpoint @ world_to_ego_transform
            )
            if detach_ego_for_world_supervision:
                supervised_ego_velocity = torch.cat(
                    [
                        ego_velocity[..., :9].detach().to(dtype=pose9_delta.dtype)
                        + pose9_delta,
                        ego_velocity[..., 9:].detach().to(dtype=pose9_delta.dtype),
                    ],
                    dim=-1,
                )
                supervised_ego_endpoint = pose9_to_matrix(
                    ego_x_t[..., :9].to(dtype=torch.float32)
                    + remaining * supervised_ego_velocity[..., :9].to(dtype=torch.float32)
                )
                corrected_world_endpoint = (
                    ego_to_world_transform
                    @ supervised_ego_endpoint
                    @ world_to_ego_transform
                )
            corrected_world_pose9 = matrix_to_pose9(corrected_world_endpoint)
            world_velocity = (
                corrected_world_pose9 - world_x_t.to(dtype=torch.float32)
            ) / remaining
            if not apply_world_to_ego:
                return ego_velocity, world_velocity.to(dtype=ego_velocity.dtype), residual
            if keep is not None:
                corrected_ego_velocity = torch.where(
                    keep[:, None, None],
                    corrected_ego_velocity,
                    ego_velocity,
                )
            return (
                corrected_ego_velocity,
                world_velocity.to(dtype=ego_velocity.dtype),
                residual,
            )
        if residual_in_body_frame:
            corrected_ego_endpoint = ego_endpoint @ se3_exp(residual.to(dtype=torch.float32))
            neutral_ego_endpoint = (
                ego_endpoint @ se3_exp(torch.zeros_like(residual, dtype=torch.float32))
            )
            corrected_world_endpoint = (
                ego_to_world_transform @ corrected_ego_endpoint @ world_to_ego_transform
            )
            neutral_world_endpoint = (
                ego_to_world_transform @ neutral_ego_endpoint @ world_to_ego_transform
            )
        elif residual_in_ego_frame:
            corrected_ego_endpoint = se3_exp(residual.to(dtype=torch.float32)) @ ego_endpoint
            neutral_ego_endpoint = (
                se3_exp(torch.zeros_like(residual, dtype=torch.float32)) @ ego_endpoint
            )
            corrected_world_endpoint = (
                ego_to_world_transform @ corrected_ego_endpoint @ world_to_ego_transform
            )
            neutral_world_endpoint = (
                ego_to_world_transform @ neutral_ego_endpoint @ world_to_ego_transform
            )
        else:
            corrected_world_endpoint = (
                se3_exp(residual.to(dtype=torch.float32)) @ baseline_world_endpoint
            )
            # Use the exact same matrix path for the neutral reference. Taking
            # their pose9 difference, rather than re-encoding the baseline
            # chart itself, makes a zero residual bit-exactly preserve the
            # legacy Ego velocity while gradients still flow through
            # ``residual``.
            neutral_world_endpoint = (
                se3_exp(torch.zeros_like(residual, dtype=torch.float32))
                @ baseline_world_endpoint
            )
            corrected_ego_endpoint = (
                world_to_ego_transform @ corrected_world_endpoint @ ego_to_world_transform
            )
            neutral_ego_endpoint = (
                world_to_ego_transform @ neutral_world_endpoint @ ego_to_world_transform
            )
        corrected_world_pose9 = matrix_to_pose9(corrected_world_endpoint)
        if detach_ego_for_world_supervision:
            supervised_ego_endpoint = pose9_to_matrix(
                ego_x_t[..., :9].to(dtype=torch.float32)
                + remaining * ego_velocity[..., :9].detach().to(dtype=torch.float32)
            )
            if residual_in_body_frame:
                supervised_world_endpoint = (
                    ego_to_world_transform
                    @ supervised_ego_endpoint
                    @ se3_exp(residual.to(dtype=torch.float32))
                    @ world_to_ego_transform
                )
            elif residual_in_ego_frame:
                supervised_world_endpoint = (
                    ego_to_world_transform
                    @ se3_exp(residual.to(dtype=torch.float32))
                    @ supervised_ego_endpoint
                    @ world_to_ego_transform
                )
            else:
                supervised_world_endpoint = (
                    se3_exp(residual.to(dtype=torch.float32))
                    @ ego_to_world_transform
                    @ supervised_ego_endpoint
                    @ world_to_ego_transform
                )
            corrected_world_pose9 = matrix_to_pose9(supervised_world_endpoint)
        world_velocity = (
            corrected_world_pose9 - world_x_t.to(dtype=torch.float32)
        ) / remaining

        if not apply_world_to_ego:
            return ego_velocity, world_velocity.to(dtype=ego_velocity.dtype), residual

        # Lift the corrected SO(3) basis back into the *same* rotation-6D
        # gauge as the legacy raw endpoint.  The legacy endpoint generally has
        # arbitrary vector lengths and shear; replacing it by canonical unit
        # columns would change the pretrained Euclidean chart even when the
        # physical rotation is unchanged.
        raw_ego_endpoint_pose9 = (
            ego_x_t[..., :9].to(dtype=torch.float32)
            + remaining * ego_velocity_anchor[..., :9].to(dtype=torch.float32)
        )
        raw_a1 = raw_ego_endpoint_pose9[..., 3:6]
        raw_a2 = raw_ego_endpoint_pose9[..., 6:9]
        neutral_b1 = neutral_ego_endpoint[..., :3, 0]
        neutral_b2 = neutral_ego_endpoint[..., :3, 1]
        corrected_b1 = corrected_ego_endpoint[..., :3, 0]
        corrected_b2 = corrected_ego_endpoint[..., :3, 1]
        scale_1 = (raw_a1 * neutral_b1).sum(dim=-1, keepdim=True)
        shear_12 = (raw_a2 * neutral_b1).sum(dim=-1, keepdim=True)
        scale_2 = (raw_a2 * neutral_b2).sum(dim=-1, keepdim=True)
        neutral_rot6 = torch.cat(
            [
                scale_1 * neutral_b1,
                shear_12 * neutral_b1 + scale_2 * neutral_b2,
            ],
            dim=-1,
        )
        corrected_rot6 = torch.cat(
            [
                scale_1 * corrected_b1,
                shear_12 * corrected_b1 + scale_2 * corrected_b2,
            ],
            dim=-1,
        )
        corrected_ego_pose9 = torch.cat(
            [
                raw_ego_endpoint_pose9[..., :3]
                + (
                    corrected_ego_endpoint[..., :3, 3]
                    - neutral_ego_endpoint[..., :3, 3]
                ),
                raw_ego_endpoint_pose9[..., 3:9]
                + (corrected_rot6 - neutral_rot6),
            ],
            dim=-1,
        )
        corrected_ego_velocity = torch.cat(
            [
                ego_velocity_anchor[..., :9]
                + (
                    corrected_ego_pose9 - raw_ego_endpoint_pose9
                ).to(dtype=ego_velocity_anchor.dtype)
                / remaining.to(dtype=ego_velocity_anchor.dtype),
                ego_velocity_anchor[..., 9:],
            ],
            dim=-1,
        )
        if keep is not None:
            corrected_ego_velocity = torch.where(
                keep[:, None, None],
                corrected_ego_velocity,
                ego_velocity,
            )
        return corrected_ego_velocity, world_velocity.to(dtype=ego_velocity.dtype), residual

    def embed_prefix(
        self,
        point_clouds,
        point_cloud_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the official image/language prefix with additional trainable point tokens."""
        self.last_pointseg_aux_loss = None
        self.last_pointseg_metrics = {}
        self.last_worldflow_payload = None
        self.last_point_action_tokens = None
        self.last_point_action_mask = None
        self.last_ego_scene_global_feat = None
        self.last_ego_scene_global_mask = None
        self.last_pointseg_visualization = None
        # Some lightweight unit-test / export shells construct this module
        # without running the full initializer.
        diagnostic_ablations = getattr(self, "inference_ablation_modalities", frozenset())
        ablate_rgb = "rgb" in diagnostic_ablations
        ablate_point = "point" in diagnostic_ablations
        ablate_language = "language" in diagnostic_ablations
        embs = []
        pad_masks = []
        att_masks = []
        point_action_token_chunks = []
        point_action_mask_chunks = []
        ego_scene_global_feat_chunks = []
        ego_scene_global_mask_chunks = []

        if self.config.vla_adapter_enable and images is None:
            raise ValueError("Frozen-VLM adapter mode requires static RGB images for the prefix.")
        if images is not None:
            if image_masks is None or len(images) != len(image_masks):
                raise ValueError("images and image_masks must be provided with matching lengths.")
            for image, image_mask in zip(images, image_masks, strict=True):
                if self.add_image_special_tokens:
                    image_start = self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=image.device)
                    ).unsqueeze(0).expand(image.shape[0], -1, -1)
                    image_start_mask = torch.ones(
                        image_start.shape[:2], dtype=torch.bool, device=image_start.device
                    )
                    if ablate_rgb:
                        image_start_mask.zero_()
                    embs.append(image_start)
                    pad_masks.append(image_start_mask)
                    att_masks += [0] * image_start.shape[1]

                image_emb = self.vlm_with_expert.embed_image(image)
                image_emb = image_emb * math.sqrt(image_emb.shape[-1])
                batch_size, num_image_tokens = image_emb.shape[:2]
                image_mask = image_mask.to(device=image_emb.device, dtype=torch.bool)
                if image_mask.ndim != 1 or image_mask.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected one image-valid flag per sample, got {tuple(image_mask.shape)}."
                    )
                image_mask = image_mask[:, None].expand(batch_size, num_image_tokens)
                if ablate_rgb:
                    image_mask = torch.zeros_like(image_mask)
                embs.append(image_emb)
                pad_masks.append(image_mask)
                att_masks += [0] * num_image_tokens

                if self.add_image_special_tokens:
                    image_end = self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=image.device)
                    ).unsqueeze(0).expand(image.shape[0], -1, -1)
                    image_end_mask = torch.ones(
                        image_end.shape[:2], dtype=torch.bool, device=image_end.device
                    )
                    if ablate_rgb:
                        image_end_mask.zero_()
                    embs.append(image_end)
                    pad_masks.append(image_end_mask)
                    att_masks += [0] * image_end.shape[1]

        for _pc_idx, (
            pc,
            pc_mask,
        ) in enumerate(zip(point_clouds, point_cloud_masks, strict=False)):
            # Point tokens are additional prefix tokens; they do not replace
            # image tokens in frozen-VLM adapter mode.
            payload = pc if isinstance(pc, dict) else {"point_cloud": pc}
            pc = payload["point_cloud"]
            if pc.ndim != 3:
                raise ValueError(f"Expected point cloud input shape (B, N, C), got {pc.shape}")

            if self.pointseg_conditioner is not None:
                conditioned = self.pointseg_conditioner(payload)
                if self.pointseg_object_proj is None or self.pointseg_background_proj is None:
                    raise RuntimeError("Pointseg projections are not initialized.")
                object_feat = conditioned["object_feat"]
                background_feat = conditioned["background_feat"]
                secondary_conditioned = None
                secondary_pc = payload.get("secondary_point_cloud")
                ablate_secondary_view = "secondary_view" in getattr(
                    self, "inference_ablation_modalities", frozenset()
                )
                residual_is_exact_identity = (
                    not self.training
                    and self.multiview_residual_proj is not None
                    and (
                        ablate_secondary_view
                        or int(torch.count_nonzero(self.multiview_residual_proj.weight).item()) == 0
                    )
                )
                if torch.is_tensor(secondary_pc) and not residual_is_exact_identity:
                    if self.multiview_residual_proj is None:
                        raise RuntimeError(
                            "A secondary point cloud requires the primary_residual adapter."
                        )
                    secondary_payload = {"point_cloud": secondary_pc}
                    secondary_point_is_pad = payload.get("secondary_point_is_pad")
                    if torch.is_tensor(secondary_point_is_pad):
                        secondary_payload["point_is_pad"] = secondary_point_is_pad
                    secondary_conditioned = self.pointseg_conditioner(secondary_payload)
                    secondary_features = torch.cat(
                        [
                            secondary_conditioned["object_feat"],
                            secondary_conditioned["background_feat"],
                        ],
                        dim=-1,
                    )
                    multiview_residual = self.multiview_residual_proj(secondary_features)
                    object_residual, background_residual = multiview_residual.chunk(2, dim=-1)
                    object_feat = object_feat + object_residual
                    background_feat = background_feat + background_residual
                elif torch.is_tensor(secondary_pc) and self.multiview_residual_proj is None:
                    raise RuntimeError(
                        "A secondary point cloud requires the primary_residual adapter."
                    )
                object_emb = self.pointseg_object_proj(object_feat)
                background_emb = self.pointseg_background_proj(background_feat)
                pc_emb = torch.stack([object_emb, background_emb], dim=1)
                if self.point_action_fusion is not None and not ablate_point:
                    foreground_tokens = conditioned.get("foreground_scene_tok1")
                    foreground_mask = conditioned.get("foreground_scene_mask1")
                    if torch.is_tensor(foreground_tokens):
                        point_action_token_chunks.append(foreground_tokens)
                        if torch.is_tensor(foreground_mask):
                            point_action_mask_chunks.append(
                                foreground_mask.to(device=foreground_tokens.device, dtype=torch.bool)
                            )
                        else:
                            point_action_mask_chunks.append(
                                torch.ones(
                                    foreground_tokens.shape[:2],
                                    dtype=torch.bool,
                                    device=foreground_tokens.device,
                                )
                            )
                        ego_scene_global_feat_chunks.append(object_feat)
                        scene_global_mask = pc_mask.to(
                            device=foreground_tokens.device,
                            dtype=torch.bool,
                        )
                        if scene_global_mask.ndim > 1:
                            scene_global_mask = scene_global_mask.reshape(scene_global_mask.shape[0], -1).any(dim=1)
                        ego_scene_global_mask_chunks.append(scene_global_mask)
                if getattr(self, "worldflow_branch", None) is not None:
                    # The foreground has already been selected by predicted
                    # PointSeg scores. WorldFlow receives XYZRGB only: no
                    # probabilities, pseudo labels or role/evidence channels.
                    foreground_clouds = [conditioned["foreground_pc"].to(dtype=torch.float32)]
                    if secondary_conditioned is not None:
                        foreground_clouds.append(
                            secondary_conditioned["foreground_pc"].to(dtype=torch.float32)
                        )
                    self.last_worldflow_payload = {
                        "foreground_pc_ego": torch.cat(foreground_clouds, dim=1),
                    }
                self.last_pointseg_aux_loss = conditioned.get("pointseg_aux_loss")
                operation_prob = conditioned["operation_prob"].detach()
                selection_scores = conditioned["pointseg_selection_scores"].detach()
                point_is_pad = payload.get("point_is_pad")
                if self.capture_pointseg_visualization:
                    snapshot_point_is_pad = (
                        point_is_pad.detach()
                        if torch.is_tensor(point_is_pad)
                        else torch.zeros_like(operation_prob, dtype=torch.bool)
                    )
                    if snapshot_point_is_pad.ndim == 3 and snapshot_point_is_pad.shape[1] == 1:
                        snapshot_point_is_pad = snapshot_point_is_pad.squeeze(1)
                    self.last_pointseg_visualization = {
                        "point_cloud": pc.detach(),
                        "point_is_pad": snapshot_point_is_pad.to(device=pc.device, dtype=torch.bool),
                        "operation_prob": operation_prob,
                        "selection_scores": selection_scores,
                    }
                if torch.is_tensor(point_is_pad):
                    valid_points = (~point_is_pad.to(device=operation_prob.device, dtype=torch.bool)).to(
                        dtype=operation_prob.dtype
                    )
                else:
                    valid_points = torch.ones_like(operation_prob)
                valid_denom = valid_points.sum().clamp_min(1.0)
                self.last_pointseg_metrics = {
                    "pointseg_foreground_ratio": (
                        ((operation_prob >= 0.5).to(dtype=operation_prob.dtype) * valid_points).sum()
                        / valid_denom
                    ),
                    "pointseg_operation_prob_mean": (operation_prob * valid_points).sum() / valid_denom,
                    "pointseg_selection_score_mean": (selection_scores * valid_points).sum() / valid_denom,
                }
                if secondary_conditioned is not None:
                    self.last_pointseg_metrics["multiview_residual_norm"] = (
                        multiview_residual.detach().norm(dim=-1).mean()
                    )
                background_has_candidates = conditioned.get("pointseg_background_has_candidates")
                if torch.is_tensor(background_has_candidates):
                    self.last_pointseg_metrics["pointseg_background_candidate_ratio"] = (
                        background_has_candidates.to(device=operation_prob.device, dtype=operation_prob.dtype).mean()
                    )
                for key, value in conditioned.get("pointseg_aux_metrics", {}).items():
                    if torch.is_tensor(value):
                        self.last_pointseg_metrics[key] = value.detach()
            else:
                if self.extractor is None or self.pointcloud_proj is None:
                    raise RuntimeError("Point cloud extractor is not initialized.")
                global_feat = self.extractor(pc, payload.get("point_is_pad"))  # (B, C)
                pc_emb = self.pointcloud_proj(global_feat).unsqueeze(1)
            pc_emb_dim = pc_emb.shape[-1]
            pc_emb = pc_emb * math.sqrt(pc_emb_dim)
            num_pc_tokens = pc_emb.shape[1]

            bsize = pc_emb.shape[0]
            if pc_mask.ndim == 1:
                pc_mask = pc_mask[:, None].expand(-1, num_pc_tokens)
            elif pc_mask.ndim == 2 and pc_mask.shape[1] == 1:
                pc_mask = pc_mask.expand(-1, num_pc_tokens)
            if ablate_point:
                pc_mask = torch.zeros_like(pc_mask, dtype=torch.bool)

            embs.append(pc_emb)
            pad_masks.append(pc_mask)
            att_masks += [0] * num_pc_tokens

        if point_action_token_chunks:
            self.last_point_action_tokens = torch.cat(point_action_token_chunks, dim=1)
            self.last_point_action_mask = torch.cat(point_action_mask_chunks, dim=1)
            global_feats = torch.stack(ego_scene_global_feat_chunks, dim=1)
            global_masks = torch.stack(ego_scene_global_mask_chunks, dim=1)
            global_weights = global_masks.unsqueeze(-1).to(dtype=global_feats.dtype)
            self.last_ego_scene_global_feat = (
                (global_feats * global_weights).sum(dim=1)
                / global_weights.sum(dim=1).clamp_min(1.0)
            )
            self.last_ego_scene_global_mask = global_masks.any(dim=1)

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        effective_lang_masks = lang_masks
        if ablate_language:
            effective_lang_masks = torch.zeros_like(lang_masks, dtype=torch.bool)
        pad_masks.append(effective_lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs



        if self.config.encode_robot_state:
            if state is None:
                raise ValueError("encode_robot_state=True requires a state tensor.")
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device
            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1] * states_seq_len




        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep, actions_is_pad: Tensor | None = None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        if actions_is_pad is None:
            action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        else:
            actions_is_pad = actions_is_pad.to(device=device, dtype=torch.bool)
            if actions_is_pad.shape != (bsize, action_time_dim):
                raise ValueError(
                    f"Expected actions_is_pad shape {(bsize, action_time_dim)}, got {actions_is_pad.shape}."
                )
            action_time_mask = ~actions_is_pad
        pad_masks.append(action_time_mask)

        # Start one new block for the whole action chunk. All valid action
        # tokens attend bidirectionally within this block and to the prefix,
        # while prefix tokens cannot attend to action tokens.
        att_masks += [1] + ([0] * (action_time_dim - 1))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def _prepare_worldflow_foreground(
        self,
        current_pose: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Map the predicted Ego foreground into its configured scene frame.

        The scene frame is deliberately independent of the action-flow
        carrier. A local, world-aligned action carrier avoids the global-origin
        lever arm in ``C B C^-1``; the scene can still retain absolute World
        position as sensory evidence. Historical checkpoints use ``carrier``
        and therefore preserve the previous exact behavior.
        """
        if self.worldflow_branch is None:
            raise RuntimeError("WorldFlow is disabled.")
        payload = self.last_worldflow_payload
        if payload is None or not torch.is_tensor(payload.get("foreground_pc_ego")):
            raise ValueError(
                "WorldFlow is enabled but the current prefix pass did not cache predicted foreground points."
            )
        point_cloud_ego = payload["foreground_pc_ego"].to(dtype=torch.float32)
        if point_cloud_ego.ndim != 3 or point_cloud_ego.shape[-1] != 6:
            raise ValueError(
                f"Expected cached predicted foreground shape (B,N,6), got {point_cloud_ego.shape}."
            )
        max_points = int(self.config.worldflow_max_points)
        if max_points > 0 and point_cloud_ego.shape[1] > max_points:
            # The conditioner already orders its selected foreground.  This is
            # a pure memory cap, not an additional role/probability feature.
            point_cloud_ego = point_cloud_ego[:, :max_points]

        point_is_pad = ~torch.isfinite(point_cloud_ego).all(dim=-1)
        if point_is_pad.any():
            point_cloud_ego = torch.where(
                point_is_pad.unsqueeze(-1),
                torch.zeros_like(point_cloud_ego),
                point_cloud_ego,
            )
        current_pose = current_pose.to(device=point_cloud_ego.device, dtype=torch.float32)
        if current_pose.ndim != 2 or current_pose.shape != (point_cloud_ego.shape[0], 9):
            raise ValueError(
                f"Expected current World EEF pose shape {(point_cloud_ego.shape[0], 9)}, "
                f"got {current_pose.shape}."
            )
        scene_frame_origin = self.config.worldflow_frame_origin
        if getattr(self.config, "worldflow_scene_frame_origin", "carrier") == "global":
            scene_frame_origin = "global"
        point_cloud_world = _ego_point_cloud_to_world(
            point_cloud_ego,
            current_pose,
            frame_origin=scene_frame_origin,
        )
        return point_cloud_world, point_is_pad, current_pose

    def _build_world_ego_joint_suffix(
        self,
        ego_action_tokens: Tensor,
        ego_action_mask: Tensor,
        world_tokens: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, slice]]:
        """Build a checkpoint-compatible asymmetric World–Ego suffix.

        Ego action tokens occupy the exact first suffix block used by the
        pretrained policy, with unchanged values and position IDs. World
        scene/action tokens are appended in later causal blocks, so they can
        read Ego while Ego cannot be perturbed by randomly initialized World
        tokens. Both losses still update the same Action Expert parameters;
        this is joint multi-task training without a gate or a frozen branch.
        """
        required_modules = (
            self.ego_scene_to_expert,
            self.world_scene_to_expert,
            self.world_ego_scene_type_embedding,
            self.world_ego_action_type_embedding,
        )
        if any(module is None for module in required_modules):
            raise RuntimeError("World–Ego joint token modules are not initialized.")
        if self.last_ego_scene_global_feat is None or self.last_ego_scene_global_mask is None:
            raise ValueError("World–Ego joint expert requires the Ego LitePT global scene feature.")

        world_global_feat = world_tokens.get("global_feat")
        world_point_mask = world_tokens.get("scene_mask")
        if not torch.is_tensor(world_global_feat) or not torch.is_tensor(world_point_mask):
            raise ValueError("World–Ego joint expert requires the World LitePT global scene feature and mask.")
        if world_global_feat.ndim != 2:
            raise ValueError(f"Expected World global scene feature shape (B,C), got {world_global_feat.shape}.")
        if world_point_mask.ndim != 2 or world_point_mask.shape[0] != world_global_feat.shape[0]:
            raise ValueError(
                "Expected World scene mask shape (B,N) matching the global scene feature, "
                f"got {world_point_mask.shape} and {world_global_feat.shape}."
            )

        ego_scene = self.ego_scene_to_expert(self.last_ego_scene_global_feat).unsqueeze(1)
        world_scene = self.world_scene_to_expert(world_global_feat).unsqueeze(1)
        ego_scene_mask = self.last_ego_scene_global_mask.to(
            device=ego_scene.device,
            dtype=torch.bool,
        ).unsqueeze(1)
        world_scene_mask = world_point_mask.to(
            device=world_scene.device,
            dtype=torch.bool,
        ).any(dim=1, keepdim=True)
        world_action_tokens = world_tokens["action_tokens"]
        world_action_mask = world_tokens["action_mask"].to(
            device=world_action_tokens.device,
            dtype=torch.bool,
        )
        ego_action_mask = ego_action_mask.to(device=ego_action_tokens.device, dtype=torch.bool)

        diagnostic_ablations = getattr(self, "inference_ablation_modalities", frozenset())
        if "world" in diagnostic_ablations:
            # Keep the exact suffix layout and positional IDs while removing
            # every World-stream key/value. This makes fixed-noise comparisons
            # isolate World information instead of changing model topology.
            world_scene_mask = torch.zeros_like(world_scene_mask)
            world_action_mask = torch.zeros_like(world_action_mask)

        ego_scene = ego_scene + self.world_ego_scene_type_embedding[0]
        world_scene = world_scene + self.world_ego_scene_type_embedding[1]
        world_action_tokens = world_action_tokens + self.world_ego_action_type_embedding[1]

        ego_scene_len = ego_scene.shape[1]
        world_scene_len = world_scene.shape[1]
        ego_action_len = ego_action_tokens.shape[1]
        world_action_len = world_action_tokens.shape[1]
        if ego_action_len != self.config.chunk_size or world_action_len != self.config.chunk_size:
            raise ValueError(
                "World–Ego joint expert requires both action streams to use "
                f"chunk_size={self.config.chunk_size}, got {ego_action_len} and {world_action_len}."
            )

        suffix_embs = torch.cat(
            [ego_action_tokens, ego_scene, world_scene, world_action_tokens],
            dim=1,
        )
        suffix_pad_masks = torch.cat(
            [ego_action_mask, ego_scene_mask, world_scene_mask, world_action_mask],
            dim=1,
        )

        scene_len = ego_scene_len + world_scene_len
        if scene_len <= 0 or ego_action_len <= 0 or world_action_len <= 0:
            raise ValueError("World–Ego joint suffix requires non-empty scene and action blocks.")
        suffix_att_masks = torch.tensor(
            [1]
            + [0] * (ego_action_len - 1)
            + [1]
            + [0] * (scene_len - 1)
            + [1]
            + [0] * (world_action_len - 1),
            dtype=suffix_embs.dtype,
            device=suffix_embs.device,
        )[None, :].expand(suffix_embs.shape[0], -1)

        ego_action_start = 0
        ego_scene_start = ego_action_len
        world_scene_start = ego_scene_start + ego_scene_len
        world_action_start = world_scene_start + world_scene_len
        layout = {
            "ego_scene": slice(ego_scene_start, world_scene_start),
            "world_scene": slice(world_scene_start, world_action_start),
            "ego_action": slice(ego_action_start, ego_action_len),
            "world_action": slice(world_action_start, world_action_start + world_action_len),
        }
        return suffix_embs, suffix_pad_masks, suffix_att_masks, layout

    def _bidirectional_world_ego_cross_attention(
        self,
        ego_out: Tensor,
        world_out: Tensor,
        ego_mask: Tensor,
        world_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Exchange Ego/World information with an identity Ego initialization."""

        modules = (
            self.ego_to_world_cross_norm,
            self.world_to_ego_cross_norm,
            self.ego_to_world_cross_attn,
            self.world_to_ego_cross_attn,
        )
        if any(module is None for module in modules):
            raise RuntimeError("Bidirectional World–Ego cross-attention is not initialized.")
        ego_valid = ego_mask.to(device=ego_out.device, dtype=torch.bool)
        world_valid = world_mask.to(device=world_out.device, dtype=torch.bool)
        ego_norm = self.ego_to_world_cross_norm(ego_out)
        world_norm = self.world_to_ego_cross_norm(world_out)
        world_delta, _ = self.ego_to_world_cross_attn(
            query=world_norm,
            key=ego_norm,
            value=ego_norm,
            key_padding_mask=~ego_valid,
            need_weights=False,
        )
        world_out = world_out + world_delta
        # Endpoint consensus supplies the World->Ego path in physical SE(3),
        # so an additional unconstrained latent correction would count the
        # World stream twice and destroy coordinate interpretability.  Ego
        # still conditions World above; World returns through the endpoint
        # midpoint after both output heads.
        if (
            getattr(getattr(self, "config", None), "worldflow_action_fusion", None)
            in {"endpoint_geodesic_consensus", "endpoint_residual_boosting"}
        ):
            return ego_out, world_out
        if "world_to_ego" in getattr(self, "inference_ablation_modalities", frozenset()):
            return ego_out, world_out
        world_updated_norm = self.world_to_ego_cross_norm(world_out)
        ego_delta, _ = self.world_to_ego_cross_attn(
            query=ego_norm,
            key=world_updated_norm,
            value=world_updated_norm,
            key_padding_mask=~world_valid,
            need_weights=False,
        )
        return ego_out + ego_delta, world_out

    def _run_world_ego_joint_expert(
        self,
        prefix_embs: Tensor | None,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor | None,
        ego_action_tokens: Tensor,
        ego_action_mask: Tensor,
        world_tokens: dict[str, Tensor],
        *,
        past_key_values=None,
    ) -> tuple[Tensor, Tensor]:
        suffix_embs, suffix_pad_masks, suffix_att_masks, layout = self._build_world_ego_joint_suffix(
            ego_action_tokens,
            ego_action_mask,
            world_tokens,
        )
        if prefix_embs is not None:
            if prefix_att_masks is None:
                raise ValueError("prefix_att_masks are required when prefix_embs are provided.")
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            attention_mask = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
        else:
            if past_key_values is None:
                raise ValueError("Cached World–Ego expert inference requires past_key_values.")
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_attention = prefix_pad_masks[:, None, :].expand(
                prefix_pad_masks.shape[0],
                suffix_len,
                prefix_len,
            )
            suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            attention_mask = torch.cat([prefix_attention, suffix_attention], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1]

        suffix_out = suffix_out.to(dtype=torch.float32)
        ego_out = suffix_out[:, layout["ego_action"]]
        world_out = suffix_out[:, layout["world_action"]]
        return self._bidirectional_world_ego_cross_attention(
            ego_out,
            world_out,
            suffix_pad_masks[:, layout["ego_action"]],
            suffix_pad_masks[:, layout["world_action"]],
        )

    def _prepare_worldflow_training_state(
        self,
        worldflow_context: dict[str, Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        time: Tensor,
        actions_is_pad: Tensor | None,
        ego_noise: Tensor | None = None,
        ego_x_t: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if self.worldflow_branch is None:
            raise RuntimeError("WorldFlow is disabled.")
        independent_world_trajectory = (
            getattr(self.config, "worldflow_target_type", "legacy_eef")
            == "world_eef_trajectory"
        )
        target_context_key = "eef_trajectory" if independent_world_trajectory else "ee_poses"
        required = ("current_ee_pose", target_context_key, "step_is_pad")
        missing = [key for key in required if key not in worldflow_context]
        if missing:
            raise ValueError(f"WorldFlow training context is missing required keys: {missing}")

        point_cloud_world, point_is_pad, current_pose = self._prepare_worldflow_foreground(
            worldflow_context["current_ee_pose"]
        )
        target_poses = worldflow_context[target_context_key].to(
            device=point_cloud_world.device,
            dtype=torch.float32,
        )
        step_is_pad = worldflow_context["step_is_pad"].to(
            device=point_cloud_world.device,
            dtype=torch.bool,
        )
        expected_shape = (point_cloud_world.shape[0], self.config.chunk_size, 9)
        if target_poses.shape != expected_shape:
            raise ValueError(f"Expected WorldFlow target poses shape {expected_shape}, got {target_poses.shape}.")
        if step_is_pad.shape != expected_shape[:2]:
            raise ValueError(f"Expected WorldFlow step_is_pad shape {expected_shape[:2]}, got {step_is_pad.shape}.")

        valid = ~step_is_pad
        if torch.is_tensor(actions_is_pad):
            action_pad = actions_is_pad.to(device=valid.device, dtype=torch.bool)
            if action_pad.shape != valid.shape:
                raise ValueError(f"Expected action_is_pad shape {valid.shape}, got {action_pad.shape}.")
            valid = valid & ~action_pad

        absolute_current = pose9_to_matrix(current_pose)
        absolute_current_inv = invert_transform(absolute_current)
        current = _worldflow_carrier_matrix(current_pose, self.config.worldflow_frame_origin)
        coordinate_frame_transform = None
        if self.training and bool(
            getattr(self.config, "worldflow_training_coordinate_frame_augmentation", False)
        ):
            coordinate_frame_transform = _sample_random_se3(
                point_cloud_world.shape[0],
                point_cloud_world.device,
                point_cloud_world.dtype,
                trans_scale=float(self.config.worldflow_augmentation_trans_scale),
                rot_scale=float(self.config.worldflow_augmentation_rot_scale),
            )
            point_cloud_world = _transform_point_cloud_xyzrgb(
                point_cloud_world,
                coordinate_frame_transform,
            )
            current = coordinate_frame_transform @ current
        current_inv = invert_transform(current)
        target = pose9_to_matrix(target_poses)
        if independent_world_trajectory:
            # This target is already the commanded future EEF trajectory in
            # the fixed robot-base frame. It must never be conjugated through
            # the current EEF carrier or represented as an Ego residual.
            spatial_gt = target
            identity = torch.eye(
                4,
                device=current.device,
                dtype=current.dtype,
            ).expand(current.shape[0], -1, -1)
            current = identity
            current_inv = identity
        else:
            body_gt = absolute_current_inv.unsqueeze(1) @ target
            spatial_gt = current.unsqueeze(1) @ body_gt @ current_inv.unsqueeze(1)
        spatial_gt_pose9 = matrix_to_pose9(spatial_gt)

        ego_coupled_noise = not independent_world_trajectory and self.config.worldflow_noise_coupling in {
            "conjugate_ego",
            "projected_ego_chart",
            "projected_ego_path",
        }
        if ego_noise is None:
            if ego_coupled_noise:
                raise ValueError("Conjugate WorldFlow noise requires the sampled Ego noise tensor.")
            ego_noise = torch.zeros(
                *spatial_gt_pose9.shape[:2],
                10,
                device=point_cloud_world.device,
                dtype=torch.float32,
            )
            ego_noise[..., 3] = 1.0
            ego_noise[..., 7] = 1.0

        if ego_coupled_noise:
            if coordinate_frame_transform is None:
                noise_pose9 = self.conjugate_ego_noise_to_world(ego_noise, current_pose)
                noise_spatial = pose9_to_matrix(noise_pose9)
            else:
                ego_noise_transform = pose9_to_matrix(ego_noise[..., :9].to(dtype=torch.float32))
                noise_spatial = (
                    current.unsqueeze(1)
                    @ ego_noise_transform
                    @ current_inv.unsqueeze(1)
                )
                noise_pose9 = matrix_to_pose9(noise_spatial)
        else:
            noise_twist = torch.randn(
                *spatial_gt_pose9.shape[:2],
                6,
                device=point_cloud_world.device,
                dtype=torch.float32,
            )
            noise_twist[..., :3] = (
                noise_twist[..., :3] * float(self.config.worldflow_noise_trans_scale)
            )
            noise_twist[..., 3:6] = (
                noise_twist[..., 3:6] * float(self.config.worldflow_noise_rot_scale)
            )
            noise_spatial = se3_exp(noise_twist)
            noise_pose9 = matrix_to_pose9(noise_spatial)

        if independent_world_trajectory:
            expected_conjugate_noise = noise_spatial
        elif coordinate_frame_transform is None:
            expected_conjugate_noise = pose9_to_matrix(
                self.conjugate_ego_noise_to_world(ego_noise, current_pose)
            )
        else:
            expected_conjugate_noise = (
                current.unsqueeze(1)
                @ pose9_to_matrix(ego_noise[..., :9].to(dtype=torch.float32))
                @ current_inv.unsqueeze(1)
            )
        noise_conjugacy_error = se3_geodesic_loss(
            noise_spatial,
            expected_conjugate_noise,
            trans_weight=float(self.config.worldflow_trans_weight),
            rot_weight=float(self.config.worldflow_rot_weight),
        )
        if self.config.worldflow_noise_coupling == "projected_ego_path":
            if ego_x_t is None:
                raise ValueError("projected_ego_path requires the current Ego chart state.")
            if coordinate_frame_transform is None:
                x_t, u_t = self.project_ego_chart_path_to_world(
                    ego_x_t,
                    current_pose,
                    spatial_gt_pose9,
                    time,
                )
            else:
                ego_h_t = pose9_to_matrix(ego_x_t[..., :9].to(dtype=torch.float32))
                world_h_t = current.unsqueeze(1) @ ego_h_t @ current_inv.unsqueeze(1)
                x_t = matrix_to_pose9(world_h_t)
                remaining = (1.0 - time).clamp_min(1e-4)[:, None, None]
                u_t = (spatial_gt_pose9 - x_t) / remaining
        elif self.config.se3_enable or independent_world_trajectory:
            world_h_t, u_t = se3_geodesic_flow_state(noise_spatial, spatial_gt, time)
            x_t = matrix_to_pose9(world_h_t)
        else:
            time_expanded = time[:, None, None]
            x_t = (1.0 - time_expanded) * noise_pose9 + time_expanded * spatial_gt_pose9
            u_t = spatial_gt_pose9 - noise_pose9

        if independent_world_trajectory or ego_x_t is None:
            path_conjugacy_error = torch.zeros_like(noise_conjugacy_error)
        else:
            ego_h_t = pose9_to_matrix(ego_x_t[..., :9].to(dtype=torch.float32))
            expected_world_h_t = current.unsqueeze(1) @ ego_h_t @ current_inv.unsqueeze(1)
            path_conjugacy_error = se3_geodesic_loss(
                pose9_to_matrix(x_t),
                expected_world_h_t,
                trans_weight=float(self.config.worldflow_trans_weight),
                rot_weight=float(self.config.worldflow_rot_weight),
            )

        world_tokens = self.worldflow_branch(
            point_cloud_world,
            lang_tokens,
            lang_masks,
            x_t,
            time,
            point_is_pad=point_is_pad,
            actions_is_pad=~valid,
        )
        state = {
            "point_cloud_world": point_cloud_world,
            "point_is_pad": point_is_pad,
            "current": current,
            "current_inv": current_inv,
            "spatial_gt": spatial_gt,
            "noise_spatial": noise_spatial,
            "noise_conjugacy_error": noise_conjugacy_error,
            "path_conjugacy_error": path_conjugacy_error,
            "x_t": x_t,
            "u_t": u_t,
            "valid": valid,
            "world_tokens": world_tokens,
            "independent_world_trajectory": torch.tensor(
                independent_world_trajectory,
                device=point_cloud_world.device,
                dtype=torch.bool,
            ),
        }
        if coordinate_frame_transform is not None:
            state["coordinate_frame_transform"] = coordinate_frame_transform
        return state

    def _augment_worldflow_training_state(
        self,
        state: dict[str, Tensor | dict[str, Tensor]],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        time: Tensor,
    ) -> tuple[dict[str, Tensor | dict[str, Tensor]], Tensor]:
        if self.worldflow_branch is None:
            raise RuntimeError("WorldFlow is disabled.")
        point_cloud_world = state["point_cloud_world"]
        point_is_pad = state["point_is_pad"]
        spatial_gt = state["spatial_gt"]
        noise_spatial = state["noise_spatial"]
        current_x_t = state["x_t"]
        valid = state["valid"]
        if not all(
            torch.is_tensor(value)
            for value in (
                point_cloud_world,
                point_is_pad,
                spatial_gt,
                noise_spatial,
                current_x_t,
                valid,
            )
        ):
            raise TypeError("WorldFlow augmentation state contains a non-tensor value.")

        transform = _sample_random_se3(
            point_cloud_world.shape[0],
            point_cloud_world.device,
            point_cloud_world.dtype,
            trans_scale=float(self.config.worldflow_augmentation_trans_scale),
            rot_scale=float(self.config.worldflow_augmentation_rot_scale),
        )
        transform_inv = invert_transform(transform)
        point_cloud_aug = _transform_point_cloud_xyzrgb(point_cloud_world, transform)
        spatial_gt_aug = transform.unsqueeze(1) @ spatial_gt @ transform_inv.unsqueeze(1)
        noise_spatial_aug = transform.unsqueeze(1) @ noise_spatial @ transform_inv.unsqueeze(1)
        if self.config.worldflow_noise_coupling == "projected_ego_path":
            current_spatial = pose9_to_matrix(current_x_t)
            current_spatial_aug = (
                transform.unsqueeze(1) @ current_spatial @ transform_inv.unsqueeze(1)
            )
            x_t_aug = matrix_to_pose9(current_spatial_aug)
            spatial_gt_aug_pose9 = matrix_to_pose9(spatial_gt_aug)
            remaining = (1.0 - time).clamp_min(1e-4)[:, None, None]
            u_t_aug = (spatial_gt_aug_pose9 - x_t_aug) / remaining
        elif self.config.se3_enable:
            h_t_aug, u_t_aug = se3_geodesic_flow_state(
                noise_spatial_aug,
                spatial_gt_aug,
                time,
            )
            x_t_aug = matrix_to_pose9(h_t_aug)
        else:
            spatial_gt_aug_pose9 = matrix_to_pose9(spatial_gt_aug)
            noise_aug_pose9 = matrix_to_pose9(noise_spatial_aug)
            time_expanded = time[:, None, None]
            x_t_aug = (1.0 - time_expanded) * noise_aug_pose9 + time_expanded * spatial_gt_aug_pose9
            u_t_aug = spatial_gt_aug_pose9 - noise_aug_pose9
        world_tokens_aug = self.worldflow_branch(
            point_cloud_aug,
            lang_tokens,
            lang_masks,
            x_t_aug,
            time,
            point_is_pad=point_is_pad,
            actions_is_pad=~valid,
        )
        return {
            "spatial_gt": spatial_gt_aug,
            "x_t": x_t_aug,
            "u_t": u_t_aug,
            "world_tokens": world_tokens_aug,
        }, transform

    def _finalize_worldflow_training_loss(
        self,
        state: dict[str, Tensor | dict[str, Tensor]],
        pred_velocity: Tensor,
        pred_body_pose9: Tensor,
        time: Tensor,
        *,
        augmented_state: dict[str, Tensor | dict[str, Tensor]] | None = None,
        pred_aug_velocity: Tensor | None = None,
        augmentation_transform: Tensor | None = None,
    ) -> dict[str, Tensor]:
        tensor_names = (
            "x_t",
            "u_t",
            "spatial_gt",
            "current",
            "current_inv",
            "valid",
            "point_cloud_world",
        )
        tensors = {name: state[name] for name in tensor_names}
        if not all(torch.is_tensor(value) for value in tensors.values()):
            raise TypeError("WorldFlow loss state contains a non-tensor value.")
        x_t = tensors["x_t"]
        u_t = tensors["u_t"]
        spatial_gt = tensors["spatial_gt"]
        current = tensors["current"]
        current_inv = tensors["current_inv"]
        valid = tensors["valid"]
        point_cloud_world = tensors["point_cloud_world"]
        noise_conjugacy_error = state.get("noise_conjugacy_error")
        if not torch.is_tensor(noise_conjugacy_error):
            raise TypeError("WorldFlow state is missing tensor noise_conjugacy_error.")
        path_conjugacy_error = state.get("path_conjugacy_error")
        if not torch.is_tensor(path_conjugacy_error):
            raise TypeError("WorldFlow state is missing tensor path_conjugacy_error.")

        independent_world_trajectory = bool(
            torch.is_tensor(state.get("independent_world_trajectory"))
            and bool(state["independent_world_trajectory"].item())
        )
        time_expanded = time[:, None, None]
        if self.config.se3_enable or independent_world_trajectory:
            remaining = (1.0 - time).clamp_min(1e-4)[:, None, None]
            pred_spatial = se3_left_apply(remaining * pred_velocity, pose9_to_matrix(x_t))
            pred_spatial_pose9 = matrix_to_pose9(pred_spatial)
            flow_step = F.smooth_l1_loss(pred_velocity, u_t, reduction="none").mean(dim=-1)
        else:
            pred_spatial_pose9 = x_t + (1.0 - time_expanded) * pred_velocity
            pred_spatial = pose9_to_matrix(pred_spatial_pose9)
            flow_step = F.mse_loss(pred_velocity, u_t, reduction="none").mean(dim=-1)
        geo_step = se3_geodesic_loss(
            pred_spatial,
            spatial_gt,
            trans_weight=float(self.config.worldflow_trans_weight),
            rot_weight=float(self.config.worldflow_rot_weight),
        )
        if independent_world_trajectory:
            # The World target and Ego action use different coordinate
            # parameterizations. Cross-token interaction is the only bridge;
            # no analytic residual/rate equality is imposed.
            pred_body_from_world = pred_spatial
            bridge_step = torch.zeros_like(geo_step)
        else:
            pred_body_from_world = current_inv.unsqueeze(1) @ pred_spatial @ current.unsqueeze(1)
            pred_body = pose9_to_matrix(pred_body_pose9[..., :9])
            bridge_step = se3_geodesic_loss(
                pred_body_from_world,
                pred_body,
                trans_weight=float(self.config.worldflow_trans_weight),
                rot_weight=float(self.config.worldflow_rot_weight),
            )
        equiv_step = torch.zeros_like(geo_step)

        if augmented_state is not None:
            if pred_aug_velocity is None or augmentation_transform is None:
                raise ValueError("Augmented WorldFlow loss requires velocity and coordinate transform.")
            x_t_aug = augmented_state["x_t"]
            u_t_aug = augmented_state["u_t"]
            spatial_gt_aug = augmented_state["spatial_gt"]
            if not all(torch.is_tensor(value) for value in (x_t_aug, u_t_aug, spatial_gt_aug)):
                raise TypeError("Augmented WorldFlow state contains a non-tensor value.")
            if self.config.se3_enable or independent_world_trajectory:
                pred_aug_spatial = se3_left_apply(
                    remaining * pred_aug_velocity,
                    pose9_to_matrix(x_t_aug),
                )
                flow_step_aug = F.smooth_l1_loss(
                    pred_aug_velocity,
                    u_t_aug,
                    reduction="none",
                ).mean(dim=-1)
            else:
                pred_aug_pose9 = x_t_aug + (1.0 - time_expanded) * pred_aug_velocity
                pred_aug_spatial = pose9_to_matrix(pred_aug_pose9)
                flow_step_aug = F.mse_loss(
                    pred_aug_velocity,
                    u_t_aug,
                    reduction="none",
                ).mean(dim=-1)
            geo_step_aug = se3_geodesic_loss(
                pred_aug_spatial,
                spatial_gt_aug,
                trans_weight=float(self.config.worldflow_trans_weight),
                rot_weight=float(self.config.worldflow_rot_weight),
            )
            flow_step = 0.5 * (flow_step + flow_step_aug)
            geo_step = 0.5 * (geo_step + geo_step_aug)
            transform_inv = invert_transform(augmentation_transform)
            expected_aug_spatial = (
                augmentation_transform.unsqueeze(1)
                @ pred_spatial
                @ transform_inv.unsqueeze(1)
            )
            equiv_step = se3_geodesic_loss(
                pred_aug_spatial,
                expected_aug_spatial,
                trans_weight=float(self.config.worldflow_trans_weight),
                rot_weight=float(self.config.worldflow_rot_weight),
            )

        per_sample_flow = _masked_step_mean(flow_step, valid)
        per_sample_geo = _masked_step_mean(geo_step, valid)
        per_sample_bridge = _masked_step_mean(bridge_step, valid)
        per_sample_equiv = _masked_step_mean(equiv_step, valid)
        per_sample_total = (
            self.config.worldflow_loss_weight * per_sample_flow
            + self.config.worldflow_geo_loss_weight * per_sample_geo
            + self.config.worldflow_bridge_loss_weight * per_sample_bridge
            + self.config.worldflow_equiv_loss_weight * per_sample_equiv
        )

        valid_f = valid.to(dtype=pred_velocity.dtype)
        valid_denom = valid_f.sum().clamp_min(1.0)
        trans_err = (
            torch.linalg.norm(pred_spatial[..., :3, 3] - spatial_gt[..., :3, 3], dim=-1) * valid_f
        ).sum() / valid_denom
        rot_err = (
            _rotation_geodesic(pred_spatial[..., :3, :3], spatial_gt[..., :3, :3]) * valid_f
        ).sum() / valid_denom
        self.last_worldflow_metrics = {
            "loss_worldflow_flow": per_sample_flow.mean().detach(),
            "loss_worldflow_geo": per_sample_geo.mean().detach(),
            "loss_worldflow_bridge": per_sample_bridge.mean().detach(),
            "loss_worldflow_equiv": per_sample_equiv.mean().detach(),
            "worldflow_trans_err": trans_err.detach(),
            "worldflow_rot_err_deg": torch.rad2deg(rot_err.detach()),
            "worldflow_valid_ratio": valid_f.mean().detach(),
            "worldflow_foreground_points": torch.tensor(
                point_cloud_world.shape[1],
                device=point_cloud_world.device,
                dtype=torch.float32,
            ),
            "worldflow_noise_conjugacy_error": _masked_step_mean(
                noise_conjugacy_error,
                valid,
            ).mean().detach(),
            "worldflow_path_conjugacy_error": _masked_step_mean(
                path_conjugacy_error,
                valid,
            ).mean().detach(),
        }
        coordinate_frame_transform = state.get("coordinate_frame_transform")
        if torch.is_tensor(coordinate_frame_transform):
            identity_rotation = torch.eye(
                3,
                device=coordinate_frame_transform.device,
                dtype=coordinate_frame_transform.dtype,
            ).expand(coordinate_frame_transform.shape[0], -1, -1)
            self.last_worldflow_metrics.update(
                {
                    "worldflow_coordinate_frame_augmentation_active": torch.ones(
                        (), device=point_cloud_world.device, dtype=torch.float32
                    ),
                    "worldflow_coordinate_frame_translation": torch.linalg.vector_norm(
                        coordinate_frame_transform[..., :3, 3], dim=-1
                    ).mean().detach(),
                    "worldflow_coordinate_frame_rotation_deg": torch.rad2deg(
                        _rotation_geodesic(
                            coordinate_frame_transform[..., :3, :3],
                            identity_rotation,
                        ).mean().detach()
                    ),
                }
            )
        else:
            self.last_worldflow_metrics["worldflow_coordinate_frame_augmentation_active"] = torch.zeros(
                (), device=point_cloud_world.device, dtype=torch.float32
            )
        result = {
            "loss_flow": per_sample_flow.mean(),
            "loss_geo": per_sample_geo.mean(),
            "loss_bridge": per_sample_bridge.mean(),
            "loss_equiv": per_sample_equiv.mean(),
            "per_sample_loss": per_sample_total,
            "valid_counts": valid.sum(dim=1).clamp_min(1),
            "pred_spatial": pred_spatial,
            "pred_spatial_pose9": pred_spatial_pose9,
            "pred_body_pose9": matrix_to_pose9(pred_body_from_world),
            "pred_velocity": pred_velocity,
        }
        if independent_world_trajectory:
            result["pred_world_eef_trajectory_pose9"] = pred_spatial_pose9
        return result

    def compute_worldflow_aux_loss(
        self,
        _batch: dict[str, Tensor] | None = None,
        _lang_tokens: Tensor | None = None,
        _lang_masks: Tensor | None = None,
        _actions_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor] | None:
        """Return the loss produced by the latest joint training forward."""
        return self.last_worldflow_aux

    def forward_se3(
        self,
        pc_feats,
        pc_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        actions_is_pad: Tensor | None = None,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
        worldflow_context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        if actions.shape[-1] < 10:
            raise ValueError(f"se3_enable=True expects pose9 + gripper actions, got {actions.shape}.")
        self.last_se3_metrics = {}
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        h_gt = pose9_to_matrix(actions[..., :9])
        gripper_gt = actions[..., 9:10].to(dtype=torch.float32)
        if noise is None:
            h_noise, gripper_noise, _noise_action = self.sample_se3_action_noise(actions)
        else:
            if noise.shape[-1] < 10:
                raise ValueError(f"SE(3) noise must have dim >= 10, got {noise.shape}.")
            _noise_action = noise.to(device=actions.device, dtype=torch.float32)
            h_noise = pose9_to_matrix(_noise_action[..., :9])
            gripper_noise = _noise_action[..., 9:10]

        time_pose = time[:, None, None]
        h_t, xi_target = se3_geodesic_flow_state(h_noise, h_gt, time)
        remaining = (1.0 - time).clamp_min(1e-3)
        gripper_t = (1.0 - time_pose) * gripper_noise + time_pose * gripper_gt
        gripper_target = gripper_gt - gripper_noise
        group_x_t = torch.cat([matrix_to_pose9(h_t), gripper_t], dim=-1)
        if self.config.se3_twist_head_mode == "pose9_chart_endpoint":
            # The expert sees exactly the original Euclidean interpolation,
            # while group_x_t remains the physical geodesic state used for
            # SE(3) losses, integration, and World conjugacy.
            x_t = (1.0 - time_pose) * _noise_action + time_pose * actions
        else:
            x_t = group_x_t
        if actions.shape[-1] > 10:
            x_t = pad_vector(x_t, actions.shape[-1])
            group_x_t = pad_vector(group_x_t, actions.shape[-1])

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats,
            pc_masks,
            lang_tokens,
            lang_masks,
            state=state,
            images=images,
            image_masks=image_masks,
        )
        if self.worldflow_branch is None:
            pred = self._se3_predict_from_suffix(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                time,
                actions_is_pad=actions_is_pad,
                ego_group_x_t=(
                    group_x_t
                    if self.config.se3_twist_head_mode == "pose9_chart_endpoint"
                    else None
                ),
            )
            world_state = None
            world_velocity = None
            augmented_state = None
            pred_aug_velocity = None
            augmentation_transform = None
        else:
            if worldflow_context is None:
                raise ValueError(
                    "worldflow_enable=True requires current and future world EEF poses during "
                    "SE(3) training."
                )
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                x_t,
                time,
                actions_is_pad=actions_is_pad,
            )
            suffix_embs = self._inject_point_action_features(
                suffix_embs,
                actions_is_pad=actions_is_pad,
            )
            world_state = self._prepare_worldflow_training_state(
                worldflow_context,
                lang_tokens,
                lang_masks,
                time,
                actions_is_pad,
                _noise_action,
                group_x_t,
            )
            world_tokens = world_state["world_tokens"]
            if not isinstance(world_tokens, dict):
                raise TypeError("WorldFlow token state must be a dictionary.")
            ego_expert_out, world_expert_out = self._run_world_ego_joint_expert(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                suffix_embs,
                suffix_pad_masks,
                world_tokens,
            )
            if self.config.se3_twist_head_mode == "pose9_chart_endpoint":
                joint_pred = self._predict_ego_se3_velocity(
                    ego_expert_out,
                    x_t,
                    time,
                    ego_group_x_t=group_x_t,
                )
            else:
                joint_pred = self._predict_ego_se3_velocity(ego_expert_out, x_t, time)
            pred = joint_pred
            world_x_t = world_state["x_t"]
            if not torch.is_tensor(world_x_t):
                raise TypeError("WorldFlow SE(3) state is missing tensor x_t.")
            current = world_state.get("current")
            current_inv = world_state.get("current_inv")
            if not torch.is_tensor(current) or not torch.is_tensor(current_inv):
                raise TypeError("World/Ego fusion requires carrier transforms.")
            if self.config.worldflow_action_fusion in {
                "conjugate_residual_consensus",
                "conjugate_residual_boosting",
            }:
                pred, world_velocity, _ = self._compose_conjugate_residual_consensus(
                    pred,
                    world_expert_out,
                    current.unsqueeze(1),
                    current_inv.unsqueeze(1),
                    detach_ego_for_world_supervision=(
                        self.config.worldflow_action_fusion == "conjugate_residual_boosting"
                    ),
                )
            elif self.config.worldflow_action_fusion == "conjugate_residual":
                world_velocity, world_residual = self._compose_conjugate_residual_world_velocity(
                    pred,
                    world_expert_out,
                    current.unsqueeze(1),
                )
                pred = torch.cat(
                    [
                        pred[..., :6]
                        + 0.5 * transform_se3_twist(world_residual, current_inv.unsqueeze(1)),
                        pred[..., 6:],
                    ],
                    dim=-1,
                )
            else:
                world_velocity = self._predict_world_se3_velocity(world_expert_out, world_x_t, time)
            if self.config.worldflow_action_fusion == "symmetric_twist":
                pred = symmetric_world_ego_twist_fusion(
                    pred,
                    world_velocity,
                    current_inv.unsqueeze(1),
                )

            augmented_state = None
            pred_aug_velocity = None
            augmentation_transform = None
            if self.config.worldflow_equiv_loss_weight > 0:
                augmented_state, augmentation_transform = self._augment_worldflow_training_state(
                    world_state,
                    lang_tokens,
                    lang_masks,
                    time,
                )
                augmented_tokens = augmented_state["world_tokens"]
                if not isinstance(augmented_tokens, dict):
                    raise TypeError("Augmented WorldFlow token state must be a dictionary.")
                ego_aug_expert_out, world_aug_expert_out = self._run_world_ego_joint_expert(
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    suffix_embs,
                    suffix_pad_masks,
                    augmented_tokens,
                )
                world_x_t_aug = augmented_state["x_t"]
                if not torch.is_tensor(world_x_t_aug):
                    raise TypeError("Augmented WorldFlow SE(3) state is missing tensor x_t.")
                if self.config.worldflow_action_fusion in {
                    "conjugate_residual",
                    "conjugate_residual_consensus",
                    "conjugate_residual_boosting",
                }:
                    if self.config.se3_twist_head_mode == "pose9_chart_endpoint":
                        ego_aug_velocity = self._predict_ego_se3_velocity(
                            ego_aug_expert_out,
                            x_t,
                            time,
                            ego_group_x_t=group_x_t,
                        )
                    else:
                        ego_aug_velocity = self._predict_ego_se3_velocity(
                            ego_aug_expert_out,
                            x_t,
                            time,
                        )
                    augmented_carrier = augmentation_transform @ current
                    if self.config.worldflow_action_fusion in {
                        "conjugate_residual_consensus",
                        "conjugate_residual_boosting",
                    }:
                        _, pred_aug_velocity, _ = self._compose_conjugate_residual_consensus(
                            ego_aug_velocity,
                            world_aug_expert_out,
                            augmented_carrier.unsqueeze(1),
                            torch.linalg.inv(augmented_carrier).unsqueeze(1),
                            detach_ego_for_world_supervision=(
                                self.config.worldflow_action_fusion
                                == "conjugate_residual_boosting"
                            ),
                        )
                    else:
                        pred_aug_velocity, _ = self._compose_conjugate_residual_world_velocity(
                            ego_aug_velocity,
                            world_aug_expert_out,
                            augmented_carrier.unsqueeze(1),
                        )
                else:
                    pred_aug_velocity = self._predict_world_se3_velocity(
                        world_aug_expert_out,
                        world_x_t_aug,
                        time,
                    )
        twist_pred = pred[..., :6]
        gripper_vel_pred = pred[..., 6:7]

        twist_step = F.smooth_l1_loss(twist_pred, xi_target, reduction="none").mean(dim=-1)
        h_endpoint = se3_left_apply((remaining[:, None, None] * twist_pred), h_t)
        self.last_body_pose9_prediction = matrix_to_pose9(h_endpoint)
        endpoint_step = se3_geodesic_loss(h_endpoint, h_gt)
        gripper_step = F.mse_loss(gripper_vel_pred, gripper_target, reduction="none").mean(dim=-1)

        equiv_step = torch.zeros_like(twist_step)
        h_final = h_endpoint

        step_total = (
            self.config.se3_pose_loss_weight * twist_step
            + self.config.se3_endpoint_loss_weight * endpoint_step
            + self.config.se3_gripper_loss_weight * gripper_step
            + self.config.se3_equivariance_loss_weight * equiv_step
        )

        valid = None
        if torch.is_tensor(actions_is_pad):
            valid = ~actions_is_pad.to(device=step_total.device, dtype=torch.bool)
        trans_err = torch.linalg.norm(h_final[..., :3, 3] - h_gt[..., :3, 3], dim=-1)
        rot_err = _rotation_geodesic(h_final[..., :3, :3], h_gt[..., :3, :3])
        self.last_se3_metrics = {
            "loss_se3_twist": self._masked_scalar_mean(twist_step, valid).detach(),
            "loss_se3_endpoint": self._masked_scalar_mean(endpoint_step, valid).detach(),
            "loss_se3_gripper": self._masked_scalar_mean(gripper_step, valid).detach(),
            "loss_se3_equivariance": self._masked_scalar_mean(equiv_step, valid).detach(),
            "se3_action_trans_err": self._masked_scalar_mean(trans_err, valid).detach(),
            "se3_action_rot_err_deg": torch.rad2deg(self._masked_scalar_mean(rot_err, valid).detach()),
        }

        if world_state is not None:
            if world_velocity is None:
                raise RuntimeError("World SE(3) velocity was not produced.")
            self.last_worldflow_aux = self._finalize_worldflow_training_loss(
                world_state,
                world_velocity,
                self.last_body_pose9_prediction,
                time,
                augmented_state=augmented_state,
                pred_aug_velocity=pred_aug_velocity,
                augmentation_transform=augmentation_transform,
            )

        # The policy wrapper expects one loss value per original action channel
        # before applying padding masks and taking the channel mean.  The SE(3)
        # objective is already a scalar per action step, so repeat it uniformly;
        # placing it only in channel 0 would accidentally rescale the complete
        # objective whenever physical per-channel action weights are configured.
        return step_total.unsqueeze(-1).expand_as(actions)

    def forward(
        self,
        pc_feats,
        pc_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        actions_is_pad=None,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
        worldflow_context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        self.last_worldflow_aux = None
        self.last_worldflow_metrics = {}
        self.last_worldflow_world_to_ego_keep_mask = None
        self.last_action_metrics = {}
        if self.config.se3_enable:
            return self.forward_se3(
                pc_feats,
                pc_masks,
                lang_tokens,
                lang_masks,
                state,
                actions,
                noise=noise,
                time=time,
                actions_is_pad=actions_is_pad,
                images=images,
                image_masks=image_masks,
                worldflow_context=worldflow_context,
            )
        if noise is None:
            if self.config.pose9_action_noise_enable:
                noise = self.sample_pose9_action_noise(tuple(actions.shape), actions.device)
            else:
                noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = (1 - time_expanded) * noise + time_expanded * actions
        u_t = actions - noise
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats,
            pc_masks,
            lang_tokens,
            lang_masks,
            state=state,
            images=images,
            image_masks=image_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t,
            time,
            actions_is_pad=actions_is_pad,
        )
        suffix_embs = self._inject_point_action_features(suffix_embs, actions_is_pad=actions_is_pad)

        if self.worldflow_branch is not None:
            if worldflow_context is None:
                raise ValueError(
                    "worldflow_enable=True requires current and future world EEF poses during training."
                )
            world_state = self._prepare_worldflow_training_state(
                worldflow_context,
                lang_tokens,
                lang_masks,
                time,
                actions_is_pad,
                noise,
                x_t,
            )
            world_tokens = world_state["world_tokens"]
            if not isinstance(world_tokens, dict):
                raise TypeError("WorldFlow token state must be a dictionary.")
            ego_expert_out, world_expert_out = self._run_world_ego_joint_expert(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                suffix_embs,
                suffix_pad_masks,
                world_tokens,
            )
            joint_v_t = self.action_out_proj(ego_expert_out)
            v_t = joint_v_t
            independent_world_trajectory = (
                getattr(self.config, "worldflow_target_type", "legacy_eef")
                == "world_eef_trajectory"
            )
            if independent_world_trajectory:
                world_x_t = world_state.get("x_t")
                if not torch.is_tensor(world_x_t):
                    raise TypeError("World-EEF trajectory state is missing tensor x_t.")
                world_velocity = self._predict_world_se3_velocity(
                    world_expert_out,
                    world_x_t,
                    time,
                )
            else:
                if self.world_action_out_proj is None:
                    raise RuntimeError("WorldFlow pose9 output projection is not initialized.")
                world_velocity = self.world_action_out_proj(world_expert_out)
            world_to_ego_keep_mask = None
            if x_t.shape[-1] < 9 or v_t.shape[-1] < 9:
                raise ValueError("World–Ego bridge requires Ego pose9 actions.")
            if self.config.worldflow_action_fusion in {
                "endpoint_geodesic_consensus",
                "endpoint_residual_boosting",
            }:
                current = world_state.get("current")
                current_inv = world_state.get("current_inv")
                world_x_t = world_state.get("x_t")
                if not all(torch.is_tensor(value) for value in (current, current_inv, world_x_t)):
                    raise TypeError("Endpoint consensus requires World/Ego states and carrier transforms.")
                if self.config.worldflow_action_fusion == "endpoint_geodesic_consensus":
                    v_t = self._compose_endpoint_geodesic_consensus(
                        x_t,
                        v_t,
                        world_x_t,
                        world_velocity,
                        time,
                        current_inv.unsqueeze(1),
                        current.unsqueeze(1),
                    )
                else:
                    dropout_probability = float(
                        getattr(
                            self.config,
                            "worldflow_training_world_to_ego_dropout_probability",
                            0.0,
                        )
                    )
                    if dropout_probability > 0:
                        world_to_ego_keep_mask = (
                            torch.rand(x_t.shape[0], device=x_t.device) >= dropout_probability
                        )
                    v_t, world_velocity, _ = self._compose_endpoint_residual_boosting(
                        x_t,
                        v_t,
                        world_x_t,
                        world_expert_out,
                        time,
                        current.unsqueeze(1),
                        current_inv.unsqueeze(1),
                        detach_ego_for_world_supervision=True,
                        world_to_ego_keep_mask=world_to_ego_keep_mask,
                        detach_retained_ego_anchor=bool(
                            getattr(
                                self.config,
                                "worldflow_training_residual_anchor_stop_gradient",
                                False,
                            )
                        ),
                    )
            endpoint = x_t + (1.0 - time_expanded) * v_t
            self.last_body_pose9_prediction = endpoint[..., :9]

            augmented_state = None
            pred_aug_velocity = None
            augmentation_transform = None
            if self.config.worldflow_equiv_loss_weight > 0:
                augmented_state, augmentation_transform = self._augment_worldflow_training_state(
                    world_state,
                    lang_tokens,
                    lang_masks,
                    time,
                )
                augmented_tokens = augmented_state["world_tokens"]
                if not isinstance(augmented_tokens, dict):
                    raise TypeError("Augmented WorldFlow token state must be a dictionary.")
                ego_aug_expert_out, world_aug_expert_out = self._run_world_ego_joint_expert(
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    suffix_embs,
                    suffix_pad_masks,
                    augmented_tokens,
                )
                if self.config.worldflow_action_fusion == "endpoint_residual_boosting":
                    augmented_world_x_t = augmented_state.get("x_t")
                    augmentation_transform = augmentation_transform.to(
                        device=current.device,
                        dtype=current.dtype,
                    )
                    augmented_carrier = augmentation_transform @ current
                    if not torch.is_tensor(augmented_world_x_t):
                        raise TypeError("Endpoint residual augmentation requires a World state.")
                    ego_aug_velocity = self.action_out_proj(ego_aug_expert_out)
                    _, pred_aug_velocity, _ = self._compose_endpoint_residual_boosting(
                        x_t,
                        ego_aug_velocity,
                        augmented_world_x_t,
                        world_aug_expert_out,
                        time,
                        augmented_carrier.unsqueeze(1),
                        invert_transform(augmented_carrier).unsqueeze(1),
                        detach_ego_for_world_supervision=True,
                    )
                else:
                    if independent_world_trajectory:
                        augmented_world_x_t = augmented_state.get("x_t")
                        if not torch.is_tensor(augmented_world_x_t):
                            raise TypeError("Augmented World-EEF trajectory state is missing tensor x_t.")
                        pred_aug_velocity = self._predict_world_se3_velocity(
                            world_aug_expert_out,
                            augmented_world_x_t,
                            time,
                        )
                    else:
                        if self.world_action_out_proj is None:
                            raise RuntimeError("WorldFlow pose9 output projection is not initialized.")
                        pred_aug_velocity = self.world_action_out_proj(world_aug_expert_out)

            self.last_worldflow_aux = self._finalize_worldflow_training_loss(
                world_state,
                world_velocity,
                self.last_body_pose9_prediction,
                time,
                augmented_state=augmented_state,
                pred_aug_velocity=pred_aug_velocity,
                augmentation_transform=augmentation_transform,
            )
            if world_to_ego_keep_mask is not None:
                self.last_worldflow_world_to_ego_keep_mask = world_to_ego_keep_mask.detach()
                self.last_worldflow_metrics["worldflow_world_to_ego_keep_ratio"] = (
                    world_to_ego_keep_mask.to(dtype=torch.float32).mean().detach()
                )
            self._record_standard_action_metrics(
                actions=actions,
                x_t=x_t,
                u_t=u_t,
                pred_velocity=v_t,
                time=time,
                actions_is_pad=actions_is_pad,
            )
            return F.mse_loss(u_t, v_t, reduction="none")

        suffix_out = self._run_ego_suffix_expert(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
        )
        v_t = self.action_out_proj(suffix_out)
        if x_t.shape[-1] >= 9 and v_t.shape[-1] >= 9:
            endpoint = x_t + (1.0 - time_expanded) * v_t
            self.last_body_pose9_prediction = endpoint[..., :9]
        else:
            self.last_body_pose9_prediction = None
        self._record_standard_action_metrics(
            actions=actions,
            x_t=x_t,
            u_t=u_t,
            pred_velocity=v_t,
            time=time,
            actions_is_pad=actions_is_pad,
        )
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self,
        pc_feats,
        pc_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
        current_ee_pose: Tensor | None = None,
        worldflow_noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        if self.config.se3_enable:
            return self.sample_actions_se3(
                pc_feats,
                pc_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                images=images,
                image_masks=image_masks,
                current_ee_pose=current_ee_pose,
                worldflow_noise=worldflow_noise,
                **kwargs,
            )
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            if self.config.pose9_action_noise_enable:
                noise = self.sample_pose9_action_noise(actions_shape, device)
            else:
                noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats,
            pc_masks,
            lang_tokens,
            lang_masks,
            state=state,
            images=images,
            image_masks=image_masks,
        )
        world_scene = None
        world_x_t = None
        ego_to_world_transform = None
        world_to_ego_transform = None
        if self.worldflow_branch is not None:
            if self._rtc_enabled():
                raise ValueError("Joint World–Ego inference is not compatible with RTC.")
            if current_ee_pose is None:
                raise ValueError("Joint World–Ego inference requires current_ee_pose.")
            current_ee_pose = current_ee_pose.to(device=device, dtype=torch.float32)
            ego_to_world_transform = _worldflow_carrier_matrix(
                current_ee_pose,
                self.config.worldflow_frame_origin,
            ).unsqueeze(1)
            world_to_ego_transform = invert_transform(ego_to_world_transform)
            point_cloud_world, point_is_pad, _ = self._prepare_worldflow_foreground(current_ee_pose)
            world_scene = self.worldflow_branch.encode_scene(
                point_cloud_world,
                point_is_pad=point_is_pad,
            )
            if worldflow_noise is None:
                if self.config.worldflow_noise_coupling in {
                    "conjugate_ego",
                    "projected_ego_chart",
                    "projected_ego_path",
                }:
                    world_x_t = self.conjugate_ego_noise_to_world(
                        noise,
                        current_ee_pose.to(device=device, dtype=torch.float32),
                    )
                else:
                    world_x_t = self.sample_worldflow_noise(bsize, device)
            else:
                expected_world_noise_shape = (bsize, self.config.chunk_size, 9)
                if worldflow_noise.shape != expected_world_noise_shape:
                    raise ValueError(
                        f"Expected worldflow_noise shape {expected_world_noise_shape}, "
                        f"got {worldflow_noise.shape}."
                    )
                world_x_t = worldflow_noise.to(device=device, dtype=torch.float32)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = 1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                if self.worldflow_branch is None:
                    v_t = denoise_step_partial_call(x_t)
                else:
                    if world_scene is None or world_x_t is None:
                        raise RuntimeError("WorldFlow inference state was not initialized.")
                    if self.config.worldflow_noise_coupling == "projected_ego_path":
                        world_x_t = self.conjugate_ego_noise_to_world(
                            x_t,
                            current_ee_pose.to(device=device, dtype=torch.float32),
                        )
                    v_t, world_v_t = self.denoise_step_world_ego(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        ego_x_t=x_t,
                        world_x_t=world_x_t,
                        timestep=time_tensor,
                        world_scene=world_scene,
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        ego_to_world_transform=ego_to_world_transform,
                        world_to_ego_transform=world_to_ego_transform,
                    )
                    if self.config.worldflow_noise_coupling != "projected_ego_path":
                        if (
                            getattr(self.config, "worldflow_target_type", "legacy_eef")
                            == "world_eef_trajectory"
                        ):
                            world_x_t = matrix_to_pose9(
                                se3_left_apply(
                                    dt * world_v_t,
                                    pose9_to_matrix(world_x_t),
                                )
                            )
                        else:
                            world_x_t = world_x_t + dt * world_v_t

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step_world_ego(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        ego_x_t: Tensor,
        world_x_t: Tensor,
        timestep: Tensor,
        world_scene: dict[str, Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        ego_group_x_t: Tensor | None = None,
        return_pose9_velocity: bool = False,
        ego_to_world_transform: Tensor | None = None,
        world_to_ego_transform: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Jointly denoise Ego body actions and World spatial transforms."""
        if self.worldflow_branch is None:
            raise RuntimeError("Joint World–Ego inference modules are not initialized.")
        ego_tokens, ego_mask, ego_att_mask = self.embed_suffix(ego_x_t, timestep)
        ego_tokens = self._inject_point_action_features(ego_tokens)
        world_action_tokens, world_action_mask = self.worldflow_branch.embed_action_tokens(
            world_scene,
            lang_tokens,
            lang_masks,
            world_x_t,
            timestep,
        )
        world_tokens = {
            **world_scene,
            "action_tokens": world_action_tokens,
            "action_mask": world_action_mask,
        }
        ego_out, world_out = self._run_world_ego_joint_expert(
            None,
            prefix_pad_masks,
            None,
            ego_tokens,
            ego_mask,
            world_tokens,
            past_key_values=past_key_values,
        )
        if self.config.se3_enable:
            joint_prediction = self._predict_ego_se3_velocity(
                ego_out,
                ego_x_t,
                timestep,
                ego_group_x_t=ego_group_x_t,
                return_pose9_velocity=return_pose9_velocity,
            )
            if return_pose9_velocity:
                joint_ego_velocity, joint_pose9_velocity = joint_prediction
            else:
                joint_ego_velocity = joint_prediction
                joint_pose9_velocity = None
            if self.config.worldflow_action_fusion in {
                "conjugate_residual",
                "conjugate_residual_consensus",
                "conjugate_residual_boosting",
            }:
                if (
                    self.world_twist_residual_out_proj is None
                    or ego_to_world_transform is None
                    or world_to_ego_transform is None
                ):
                    raise RuntimeError("Conjugate-residual fusion modules/carriers are not initialized.")
                if "world_to_ego" in getattr(self, "inference_ablation_modalities", frozenset()):
                    world_velocity = transform_se3_twist(
                        joint_ego_velocity[..., :6],
                        ego_to_world_transform,
                    )
                elif self.config.worldflow_action_fusion in {
                    "conjugate_residual_consensus",
                    "conjugate_residual_boosting",
                }:
                    joint_ego_velocity, world_velocity, _ = (
                        self._compose_conjugate_residual_consensus(
                            joint_ego_velocity,
                            world_out,
                            ego_to_world_transform,
                            world_to_ego_transform,
                        )
                    )
                else:
                    world_velocity, residual = self._compose_conjugate_residual_world_velocity(
                        joint_ego_velocity,
                        world_out,
                        ego_to_world_transform,
                    )
                    joint_ego_velocity = torch.cat(
                        [
                            joint_ego_velocity[..., :6]
                            + 0.5 * transform_se3_twist(residual, world_to_ego_transform),
                            joint_ego_velocity[..., 6:],
                        ],
                        dim=-1,
                    )
            else:
                world_velocity = self._predict_world_se3_velocity(world_out, world_x_t, timestep)
            if return_pose9_velocity:
                return joint_ego_velocity, world_velocity, joint_pose9_velocity
            return joint_ego_velocity, world_velocity
        joint_ego_velocity = self.action_out_proj(ego_out)
        if (
            getattr(self.config, "worldflow_target_type", "legacy_eef")
            == "world_eef_trajectory"
        ):
            world_velocity = self._predict_world_se3_velocity(world_out, world_x_t, timestep)
        else:
            if self.world_action_out_proj is None:
                raise RuntimeError("Joint World–Ego pose9 output projection is not initialized.")
            world_velocity = self.world_action_out_proj(world_out)
        if getattr(self.config, "worldflow_action_fusion", "cross_attention") == "endpoint_residual_boosting":
            if ego_to_world_transform is None or world_to_ego_transform is None:
                raise RuntimeError("Endpoint residual carrier transforms are not initialized.")
            joint_ego_velocity, world_velocity, _ = self._compose_endpoint_residual_boosting(
                ego_x_t,
                joint_ego_velocity,
                world_x_t,
                world_out,
                timestep,
                ego_to_world_transform,
                world_to_ego_transform,
                apply_world_to_ego=(
                    "world_to_ego"
                    not in getattr(self, "inference_ablation_modalities", frozenset())
                ),
            )
        if (
            getattr(self.config, "worldflow_action_fusion", "cross_attention")
            == "endpoint_geodesic_consensus"
            and "world_to_ego" not in getattr(self, "inference_ablation_modalities", frozenset())
        ):
            if ego_to_world_transform is None or world_to_ego_transform is None:
                raise RuntimeError("Endpoint consensus carrier transforms are not initialized.")
            joint_ego_velocity = self._compose_endpoint_geodesic_consensus(
                ego_x_t,
                joint_ego_velocity,
                world_x_t,
                world_velocity,
                timestep,
                world_to_ego_transform,
                ego_to_world_transform,
            )
        return joint_ego_velocity, world_velocity

    def sample_actions_se3(
        self,
        pc_feats,
        pc_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
        current_ee_pose: Tensor | None = None,
        worldflow_noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if self._rtc_enabled():
            raise ValueError("se3_enable=True is not supported with RTC enabled in v1.")
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            dummy_actions = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, device=device)
            _h_noise, _gripper_noise, x_t = self.sample_se3_action_noise(dummy_actions)
        else:
            x_t = noise.to(device=device, dtype=torch.float32)
            if x_t.shape[-1] < 10:
                raise ValueError(f"SE(3) inference noise must have dim >= 10, got {x_t.shape}.")
            if x_t.shape[-1] < self.config.max_action_dim:
                x_t = pad_vector(x_t, self.config.max_action_dim)
        chart_endpoint_mode = self.config.se3_twist_head_mode == "pose9_chart_endpoint"
        ego_group_x_t = torch.cat(
            [matrix_to_pose9(pose9_to_matrix(x_t[..., :9])), x_t[..., 9:10]],
            dim=-1,
        )
        if ego_group_x_t.shape[-1] < self.config.max_action_dim:
            ego_group_x_t = pad_vector(ego_group_x_t, self.config.max_action_dim)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats,
            pc_masks,
            lang_tokens,
            lang_masks,
            state=state,
            images=images,
            image_masks=image_masks,
        )
        world_scene = None
        world_x_t = None
        if self.worldflow_branch is not None:
            if current_ee_pose is None:
                raise ValueError("Joint World–Ego SE(3) inference requires current_ee_pose.")
            current_ee_pose = current_ee_pose.to(device=device, dtype=torch.float32)
            ego_to_world_transform = _worldflow_carrier_matrix(
                current_ee_pose,
                self.config.worldflow_frame_origin,
            ).unsqueeze(1)
            world_to_ego_transform = invert_transform(ego_to_world_transform)
            point_cloud_world, point_is_pad, _ = self._prepare_worldflow_foreground(current_ee_pose)
            world_scene = self.worldflow_branch.encode_scene(
                point_cloud_world,
                point_is_pad=point_is_pad,
            )
            if worldflow_noise is None:
                if self.config.worldflow_noise_coupling in {
                    "conjugate_ego",
                    "projected_ego_chart",
                    "projected_ego_path",
                }:
                    world_x_t = self.conjugate_ego_noise_to_world(ego_group_x_t, current_ee_pose)
                else:
                    world_x_t = self.sample_worldflow_noise(bsize, device)
            else:
                expected_shape = (bsize, self.config.chunk_size, 9)
                if worldflow_noise.shape != expected_shape:
                    raise ValueError(
                        f"Expected worldflow_noise shape {expected_shape}, got {worldflow_noise.shape}."
                    )
                world_x_t = worldflow_noise.to(device=device, dtype=torch.float32)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = 1.0 / num_steps
        for step in range(num_steps):
            time = step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            if self.worldflow_branch is None:
                prediction = self.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                    ego_group_x_t=ego_group_x_t if chart_endpoint_mode else None,
                    return_pose9_velocity=chart_endpoint_mode,
                )
                if chart_endpoint_mode:
                    pred, pose9_velocity = prediction
                else:
                    pred = prediction
                    pose9_velocity = None
                world_twist = None
            else:
                if world_scene is None or world_x_t is None:
                    raise RuntimeError("Joint World–Ego SE(3) inference state was not initialized.")
                prediction = self.denoise_step_world_ego(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    ego_x_t=x_t,
                    world_x_t=world_x_t,
                    timestep=time_tensor,
                    world_scene=world_scene,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    ego_group_x_t=ego_group_x_t if chart_endpoint_mode else None,
                    return_pose9_velocity=chart_endpoint_mode,
                    ego_to_world_transform=ego_to_world_transform,
                    world_to_ego_transform=world_to_ego_transform,
                )
                if chart_endpoint_mode:
                    pred, world_twist, pose9_velocity = prediction
                else:
                    pred, world_twist = prediction
                    pose9_velocity = None
                if self.config.worldflow_action_fusion == "symmetric_twist":
                    if world_to_ego_transform is None:
                        raise RuntimeError("Symmetric World/Ego fusion carrier was not initialized.")
                    pred = symmetric_world_ego_twist_fusion(
                        pred,
                        world_twist,
                        world_to_ego_transform,
                    )
            twist = pred[..., :6]
            gripper_vel = pred[..., 6:7]
            h_t = pose9_to_matrix(ego_group_x_t[..., :9])
            h_next = se3_left_apply(dt * twist, h_t)
            if chart_endpoint_mode:
                x_t = x_t + dt * pose9_velocity
                gripper_next = x_t[..., 9:10]
            else:
                gripper_next = ego_group_x_t[..., 9:10] + dt * gripper_vel
                x_t = torch.cat([matrix_to_pose9(h_next), gripper_next], dim=-1)
                if x_t.shape[-1] < self.config.max_action_dim:
                    x_t = pad_vector(x_t, self.config.max_action_dim)
            ego_group_x_t = torch.cat([matrix_to_pose9(h_next), gripper_next], dim=-1)
            if ego_group_x_t.shape[-1] < self.config.max_action_dim:
                ego_group_x_t = pad_vector(ego_group_x_t, self.config.max_action_dim)
            if world_twist is not None:
                world_h_next = se3_left_apply(dt * world_twist, pose9_to_matrix(world_x_t))
                world_x_t = matrix_to_pose9(world_h_next)

        return ego_group_x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        ego_group_x_t=None,
        return_pose9_velocity: bool = False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        suffix_embs = self._inject_point_action_features(suffix_embs)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        if "action_context" in self.inference_ablation_modalities:
            # Preserve each action token and prefix cross-attention while
            # removing communication between different action steps.
            action_identity = torch.eye(
                suffix_len,
                dtype=torch.bool,
                device=suffix_att_2d_masks.device,
            ).unsqueeze(0)
            suffix_att_2d_masks = suffix_att_2d_masks & action_identity

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        if self.config.se3_enable:
            v_t = self._predict_ego_se3_velocity(
                suffix_out,
                x_t,
                timestep,
                ego_group_x_t=ego_group_x_t,
                return_pose9_velocity=return_pose9_velocity,
            )
        else:
            v_t = self.action_out_proj(suffix_out)
        return v_t
