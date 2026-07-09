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
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.smolvla.song_pointseg import (
    MOTION_PRIOR_DIM,
    PseudoLabelConfig,
    ROLE_FOREGROUND,
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
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype

from .litept.model import LitePT


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

    def get_optim_params(self) -> dict:
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
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            pc_feats, pc_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
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
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(
            pc_feats, pc_masks, lang_tokens, lang_masks, state, actions, noise, time, actions_is_pad=actions_is_pad
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
        worldflow_aux = self.model.compute_worldflow_aux_loss(
            batch,
            lang_tokens,
            lang_masks,
            actions_is_pad,
            cached_lang_emb=self.model.last_language_emb,
        )
        for key, value in self.model.last_worldflow_metrics.items():
            if torch.is_tensor(value):
                loss_dict[key] = value.detach().item()

        # Remove padding
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
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
            "state_proj|action_in_proj|action_out_proj|se3_action_out_proj|action_time_mlp_in|action_time_mlp_out"
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
        self.feature_in_dim = 3 if int(in_dim) >= 6 else max(int(in_dim) - 3, 1)

        self.backbone = LitePT(in_channels=self.feature_in_dim, enc_mode=enc_mode)
        self.out_proj = nn.Linear(infer_litept_output_channels(self.backbone), dim)

    def _build_point_features(self, raw_feat):
        """Use XYZ only as coordinates; feed RGB/non-coordinate channels as point features."""
        if raw_feat.shape[-1] >= 6:
            feat = raw_feat[:, 3:6] / 255.0
        elif raw_feat.shape[-1] > 3:
            feat = raw_feat[:, 3:]
        else:
            feat = raw_feat.new_zeros(raw_feat.shape[0], 0)

        if feat.shape[-1] < self.feature_in_dim:
            pad = feat.new_zeros(feat.shape[0], self.feature_in_dim - feat.shape[-1])
            feat = torch.cat([feat, pad], dim=-1)
        elif feat.shape[-1] > self.feature_in_dim:
            feat = feat[:, : self.feature_in_dim]
        return feat.contiguous()

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
        feat = self._build_point_features(feat)

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

    def forward(self, scene_pc, point_is_pad=None):
        scene_xyz, scene_tok, _scene_g, scene_mask = self.pc_backbone(scene_pc, point_is_pad)
        scene_tok = self.attention(scene_tok, scene_mask)
        alpha = self._masked_softmax(self.att[0](scene_tok), scene_mask)
        center = (scene_xyz * alpha).sum(dim=1)

        centroid_xyz = scene_pc[..., :3] - center.unsqueeze(-2)
        scene_pc1 = torch.cat([centroid_xyz, scene_pc[..., 3:]], dim=-1)
        _scene_xyz1, scene_tok1, _scene_g1, scene_mask1 = self.pc_backbone1(scene_pc1, point_is_pad)
        scene_tok1 = self.attention1(scene_tok1, scene_mask1)
        alpha1 = self._masked_softmax(self.att1[0](scene_tok1), scene_mask1)

        global_feat = (scene_tok1 * alpha1).sum(dim=1)
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


def _ego_point_cloud_to_world(point_cloud_ego: Tensor, current_pose_world: Tensor) -> Tensor:
    if point_cloud_ego.ndim != 3 or point_cloud_ego.shape[-1] != 6:
        raise ValueError(f"Expected ego point cloud shape (B,N,6), got {point_cloud_ego.shape}.")
    if current_pose_world.ndim != 2 or current_pose_world.shape[-1] != 9:
        raise ValueError(f"Expected current world pose shape (B,9), got {current_pose_world.shape}.")
    transform = pose9_to_matrix(current_pose_world.to(device=point_cloud_ego.device, dtype=torch.float32))
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    xyz_world = torch.matmul(point_cloud_ego[..., :3].to(dtype=torch.float32), rot.transpose(-1, -2))
    xyz_world = xyz_world + trans.unsqueeze(1)
    return torch.cat([xyz_world, point_cloud_ego[..., 3:6].to(dtype=torch.float32)], dim=-1)


def _transform_xyz(points: Tensor, transform: Tensor) -> Tensor:
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    return torch.matmul(points, rot.transpose(-1, -2)) + trans.unsqueeze(-2)


def _masked_language_mean(lang_emb: Tensor, lang_masks: Tensor) -> Tensor:
    mask = lang_masks.to(device=lang_emb.device, dtype=torch.bool)
    weights = mask.unsqueeze(-1).to(dtype=lang_emb.dtype)
    denom = weights.sum(dim=1).clamp_min(torch.finfo(lang_emb.dtype).tiny)
    return (lang_emb * weights).sum(dim=1) / denom


def _ensure_min_transport_weights(
    scores: Tensor,
    valid: Tensor,
    *,
    min_points: int,
    threshold: float,
) -> Tensor:
    scores = scores.to(dtype=torch.float32).clamp_min(0.0)
    valid = valid.to(device=scores.device, dtype=torch.bool)
    weights = torch.where(valid & (scores >= float(threshold)), scores, torch.zeros_like(scores))
    if min_points <= 0:
        return weights
    need_fallback = (weights > 0).sum(dim=1) < int(min_points)
    if not bool(need_fallback.any().item()):
        return weights

    out = weights.clone()
    fallback_scores = torch.where(valid, scores, torch.full_like(scores, -torch.inf))
    for bidx in torch.nonzero(need_fallback, as_tuple=False).flatten().tolist():
        valid_count = int(valid[bidx].sum().item())
        if valid_count <= 0:
            continue
        k = min(int(min_points), valid_count)
        idx = torch.topk(fallback_scores[bidx], k=k, largest=True).indices
        fallback_weight = scores[bidx, idx].clamp_min(1e-3)
        out[bidx, idx] = torch.maximum(out[bidx, idx], fallback_weight)
    return out


def select_objectflow_points(
    point_cloud: Tensor,
    role_scores: Tensor,
    point_is_pad: Tensor | None = None,
    *,
    max_points: int = 2048,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select a balanced automatic condition/target subset for dense ObjectFlow.

    role_scores is generated by motion priors and uses channels:
    0=gripper, 1=condition/manipulated object, 2=target. Gripper points are
    suppressed so the dense branch models object/target geometry rather than
    memorizing the robot hand cloud.
    """

    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
        raise ValueError(f"Expected point_cloud shape (B,N,6), got {point_cloud.shape}.")
    if role_scores.ndim != 3 or role_scores.shape[:2] != point_cloud.shape[:2] or role_scores.shape[-1] < 3:
        raise ValueError(
            f"Expected role_scores shape (B,N,>=3) matching point cloud, got {role_scores.shape}."
        )

    bsize, n_points, channels = point_cloud.shape
    if n_points <= 0:
        raise ValueError("Cannot select ObjectFlow points from an empty point cloud.")
    if point_is_pad is None:
        point_is_pad = torch.zeros(bsize, n_points, dtype=torch.bool, device=point_cloud.device)
    else:
        point_is_pad = point_is_pad.to(device=point_cloud.device, dtype=torch.bool)
        if point_is_pad.shape != point_cloud.shape[:2]:
            raise ValueError(f"Expected point_is_pad shape {point_cloud.shape[:2]}, got {point_is_pad.shape}.")
    point_is_pad = point_is_pad | ~torch.isfinite(point_cloud[..., :3]).all(dim=-1)

    roles = role_scores[..., :3].to(device=point_cloud.device, dtype=torch.float32).clamp(0.0, 1.0)
    max_points = int(max_points)
    if max_points <= 0 or n_points <= max_points:
        return point_cloud.to(dtype=torch.float32), roles, point_is_pad
    max_points = min(max_points, n_points)

    nongripper = (1.0 - roles[..., 0]).clamp(0.0, 1.0)
    condition_scores = (roles[..., 1] * nongripper).masked_fill(point_is_pad, -torch.inf)
    target_scores = (roles[..., 2] * nongripper).masked_fill(point_is_pad, -torch.inf)

    condition_count = max(1, max_points // 2)
    target_count = max_points - condition_count
    condition_indices = torch.topk(condition_scores, k=condition_count, dim=1).indices
    if target_count > 0:
        target_scores = target_scores.scatter(1, condition_indices, -torch.inf)
        target_indices = torch.topk(target_scores, k=target_count, dim=1).indices
        indices = torch.cat([condition_indices, target_indices], dim=1)
    else:
        indices = condition_indices

    point_cloud = point_cloud.to(dtype=torch.float32).gather(
        1, indices[..., None].expand(bsize, indices.shape[1], channels)
    )
    roles = roles.gather(1, indices[..., None].expand(bsize, indices.shape[1], roles.shape[-1]))
    point_is_pad = point_is_pad.gather(1, indices)
    return point_cloud, roles, point_is_pad


def weighted_kabsch_transform(
    source_xyz: Tensor,
    target_xyz: Tensor,
    weights: Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Fit source->target rigid transforms with one batched SVD.

    Shapes are source_xyz=(B,N,3), target_xyz=(B,T,N,3), weights=(B,N).
    Returns transform=(B,T,4,4) and a valid mask (B,T).
    """

    if source_xyz.ndim != 3 or source_xyz.shape[-1] != 3:
        raise ValueError(f"Expected source_xyz shape (B,N,3), got {source_xyz.shape}.")
    if target_xyz.ndim != 4 or target_xyz.shape[-2:] != source_xyz.shape[-2:]:
        raise ValueError(
            f"Expected target_xyz shape (B,T,N,3) matching source {source_xyz.shape}, got {target_xyz.shape}."
        )
    if weights.shape != source_xyz.shape[:2]:
        raise ValueError(f"Expected weights shape {source_xyz.shape[:2]}, got {weights.shape}.")

    source_xyz = source_xyz.to(dtype=torch.float32)
    target_xyz = target_xyz.to(device=source_xyz.device, dtype=torch.float32)
    weights = weights.to(device=source_xyz.device, dtype=torch.float32).clamp_min(0.0)
    finite = torch.isfinite(source_xyz).all(dim=-1) & torch.isfinite(weights)
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    weight_sum = weights.sum(dim=-1, keepdim=True)
    normalized = weights / weight_sum.clamp_min(eps)
    source_center = (source_xyz * normalized.unsqueeze(-1)).sum(dim=1)
    target_center = (target_xyz * normalized[:, None, :, None]).sum(dim=2)
    source_centered = source_xyz - source_center[:, None, :]
    target_centered = target_xyz - target_center[:, :, None, :]
    covariance = torch.einsum("bn,bni,btnj->btij", normalized, source_centered, target_centered)

    u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    reflection = torch.linalg.det(rotation) < 0
    correction = torch.ones(*rotation.shape[:-2], 3, device=rotation.device, dtype=rotation.dtype)
    correction[..., -1] = torch.where(reflection, -torch.ones_like(correction[..., -1]), correction[..., -1])
    rotation = vh.transpose(-1, -2) @ torch.diag_embed(correction) @ u.transpose(-1, -2)
    translation = target_center - (rotation @ source_center[:, None, :, None]).squeeze(-1)

    transform = _eye4_like(rotation.shape[:-2], device=rotation.device, dtype=rotation.dtype)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    active_points = (weights > eps).sum(dim=-1)
    finite_transform = torch.isfinite(transform).flatten(2).all(dim=-1) & torch.isfinite(singular_values).all(dim=-1)
    valid = (weight_sum.squeeze(-1) > eps)[:, None] & (active_points >= 3)[:, None] & finite_transform
    identity = _eye4_like(transform.shape[:-2], device=transform.device, dtype=transform.dtype)
    transform = torch.where(valid[..., None, None], transform, identity)
    return transform, valid


class DenseRigidObjectFlowHead(nn.Module):
    """Dense automatic ObjectFlow head over the fixed Overview/world point cloud."""

    ROLE_DIM = 3

    def __init__(self, config: SmolVLAConfig, language_dim: int):
        super().__init__()
        self.chunk_size = int(config.chunk_size)
        self.feature_dim = int(config.worldflow_feature_dim)
        self.point_encoder = nn.Sequential(
            nn.Linear(6 + self.ROLE_DIM, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
        )
        self.lang_proj = nn.Linear(language_dim, self.feature_dim)
        self.context_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 4, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
        )
        self.flow_head = nn.Linear(self.feature_dim, self.chunk_size * 3)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        point_cloud_world: Tensor,
        role_scores: Tensor,
        lang_emb: Tensor,
        lang_masks: Tensor,
        point_is_pad: Tensor | None = None,
    ) -> Tensor:
        if point_cloud_world.ndim != 3 or point_cloud_world.shape[-1] != 6:
            raise ValueError(f"Expected point_cloud_world shape (B,N,6), got {point_cloud_world.shape}.")
        if role_scores.shape != (*point_cloud_world.shape[:2], self.ROLE_DIM):
            raise ValueError(
                f"Expected automatic role_scores shape {(*point_cloud_world.shape[:2], self.ROLE_DIM)}, "
                f"got {role_scores.shape}."
            )
        point_cloud_world = point_cloud_world.to(dtype=torch.float32)
        role_scores = role_scores.to(device=point_cloud_world.device, dtype=torch.float32).clamp(0.0, 1.0)
        rgb = point_cloud_world[..., 3:6] / 255.0
        point_inputs = torch.cat([point_cloud_world[..., :3], rgb, role_scores], dim=-1)
        point_feat = self.point_encoder(point_inputs)
        if point_is_pad is not None:
            point_is_pad = point_is_pad.to(device=point_cloud_world.device, dtype=torch.bool)

        nongripper = (1.0 - role_scores[..., 0]).clamp(0.0, 1.0)
        if point_is_pad is not None:
            nongripper = nongripper * (~point_is_pad).to(dtype=nongripper.dtype)
        condition_weights = nongripper * role_scores[..., 1]
        target_weights = nongripper * role_scores[..., 2]
        condition_feat = (point_feat * condition_weights.unsqueeze(-1)).sum(dim=1) / condition_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        target_feat = (point_feat * target_weights.unsqueeze(-1)).sum(dim=1) / target_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        lang_feat = self.lang_proj(_masked_language_mean(lang_emb, lang_masks).to(dtype=point_feat.dtype))
        global_context = torch.cat([condition_feat, target_feat, lang_feat], dim=-1).unsqueeze(1).expand(
            -1, point_feat.shape[1], -1
        )
        fused = self.context_fusion(torch.cat([point_feat, global_context], dim=-1))
        flow = self.flow_head(fused).reshape(point_feat.shape[0], point_feat.shape[1], self.chunk_size, 3)
        return flow.permute(0, 2, 1, 3).contiguous()


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
            object_feat = self.foreground_encoder(foreground_pc)
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
            nn.Linear(self.vlm_with_expert.expert_hidden_size, 7) if self.config.se3_enable else None
        )
        use_pointseg = self.config.pointseg_enable or self.config.pointseg_checkpoint_path is not None
        if self.config.worldflow_enable and not use_pointseg:
            raise ValueError(
                "worldflow_enable requires pointseg_enable=True or pointseg_checkpoint_path because "
                "worldflow uses SongPointCloudConditioner foreground/background selections as its point cloud input."
            )
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
        self.last_pointseg_aux_loss: Tensor | None = None
        self.last_pointseg_metrics: dict[str, Tensor] = {}
        self.last_objectflow_payload: dict[str, Tensor] | None = None
        self.last_language_emb: Tensor | None = None
        self.last_body_pose9_prediction: Tensor | None = None
        self.worldflow_head = (
            DenseRigidObjectFlowHead(config, self.vlm_with_expert.config.text_config.hidden_size)
            if self.config.worldflow_enable
            else None
        )
        self.last_worldflow_metrics: dict[str, Tensor] = {}
        self.last_se3_metrics: dict[str, Tensor] = {}

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

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

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def sample_se3_action_noise(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if actions.shape[-1] < 10:
            raise ValueError(f"se3_enable=True expects action dim >= 10, got {actions.shape[-1]}.")
        xi_noise = torch.randn(*actions.shape[:2], 6, device=actions.device, dtype=torch.float32)
        xi_noise[..., :3] = xi_noise[..., :3] * float(self.config.se3_noise_trans_scale)
        xi_noise[..., 3:6] = xi_noise[..., 3:6] * float(self.config.se3_noise_rot_scale)
        pose_noise = se3_exp(xi_noise)
        gripper_noise = self.sample_noise((*actions.shape[:2], 1), actions.device)
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

    def _se3_predict_from_suffix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
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
        if self.se3_action_out_proj is None:
            raise RuntimeError("se3_action_out_proj is not initialized.")
        return self.se3_action_out_proj(suffix_out)

    def embed_prefix(
        self, point_clouds, point_cloud_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed point cloud features and language tokens to prepare for SmolVLM transformer processing.
        """
        self.last_pointseg_aux_loss = None
        self.last_pointseg_metrics = {}
        self.last_objectflow_payload = None
        self.last_language_emb = None
        embs = []
        pad_masks = []
        att_masks = []

        for _pc_idx, (
            pc,
            pc_mask,
        ) in enumerate(zip(point_clouds, point_cloud_masks, strict=False)):
            # Point cloud features replace the original image features.
            # The original SmolVLA image processing is intentionally disabled.
            payload = pc if isinstance(pc, dict) else {"point_cloud": pc}
            pc = payload["point_cloud"]
            if pc.ndim != 3:
                raise ValueError(f"Expected point cloud input shape (B, N, C), got {pc.shape}")

            if self.pointseg_conditioner is not None:
                conditioned = self.pointseg_conditioner(payload)
                if self.pointseg_object_proj is None or self.pointseg_background_proj is None:
                    raise RuntimeError("Pointseg projections are not initialized.")
                object_emb = self.pointseg_object_proj(conditioned["object_feat"])
                background_emb = self.pointseg_background_proj(conditioned["background_feat"])
                pc_emb = torch.stack([object_emb, background_emb], dim=1)
                if self.worldflow_head is not None:
                    role_scores = payload.get("pointseg.role_scores")
                    if torch.is_tensor(role_scores) and role_scores.ndim >= 3 and role_scores.shape[1] == 1:
                        role_scores = role_scores.squeeze(1)
                    if not torch.is_tensor(role_scores):
                        role_scores = conditioned.get("role_scores")
                    point_is_pad = payload.get("point_is_pad")
                    if torch.is_tensor(point_is_pad) and point_is_pad.ndim == 3 and point_is_pad.shape[1] == 1:
                        point_is_pad = point_is_pad.squeeze(1)
                    self.last_objectflow_payload = {
                        "point_cloud_ego": pc.to(dtype=torch.float32),
                    }
                    if torch.is_tensor(role_scores):
                        self.last_objectflow_payload["role_scores"] = role_scores.to(
                            device=pc.device, dtype=torch.float32
                        )
                    if torch.is_tensor(point_is_pad):
                        self.last_objectflow_payload["point_is_pad"] = point_is_pad.to(
                            device=pc.device, dtype=torch.bool
                        )
                self.last_pointseg_aux_loss = conditioned.get("pointseg_aux_loss")
                operation_prob = conditioned["operation_prob"].detach()
                selection_scores = conditioned["pointseg_selection_scores"].detach()
                point_is_pad = payload.get("point_is_pad")
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

            embs.append(pc_emb)
            pad_masks.append(pc_mask)
            att_masks += [0] * num_pc_tokens

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        self.last_language_emb = lang_emb
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
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

    def embed_suffix(self, noisy_actions, timestep):
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
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def compute_worldflow_aux_loss(
        self,
        batch: dict[str, Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions_is_pad: Tensor | None = None,
        cached_lang_emb: Tensor | None = None,
    ) -> dict[str, Tensor] | None:
        self.last_worldflow_metrics = {}
        if self.worldflow_head is None:
            return None

        required = (
            "worldflow.current_ee_pose",
            "worldflow.ee_poses",
            "worldflow.step_is_pad",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Worldflow is enabled but batch is missing required keys: {missing}")

        payload = self.last_objectflow_payload
        if payload is None or not torch.is_tensor(payload.get("point_cloud_ego")):
            raise ValueError(
                "Worldflow is enabled but the current forward pass did not cache an ObjectFlow point cloud."
            )
        if not torch.is_tensor(payload.get("role_scores")):
            raise ValueError(
                "Dense ObjectFlow requires automatic pointseg.role_scores "
                "(channels: gripper, condition/manipulated object, target). "
                "Rebuild the PointSeg cache with the current script or enable online pseudo labels."
            )
        point_cloud_ego = payload["point_cloud_ego"].to(dtype=torch.float32)
        role_scores = payload["role_scores"].to(device=point_cloud_ego.device, dtype=torch.float32)
        if role_scores.shape[:2] != point_cloud_ego.shape[:2] or role_scores.shape[-1] < DenseRigidObjectFlowHead.ROLE_DIM:
            raise ValueError(
                f"Expected pointseg.role_scores shape (B,N,>={DenseRigidObjectFlowHead.ROLE_DIM}) matching "
                f"{point_cloud_ego.shape[:2]}, got {role_scores.shape}."
            )
        role_scores = role_scores[..., : DenseRigidObjectFlowHead.ROLE_DIM]
        point_is_pad = payload.get("point_is_pad")
        if torch.is_tensor(point_is_pad):
            point_is_pad = point_is_pad.to(device=point_cloud_ego.device, dtype=torch.bool)
        else:
            point_is_pad = torch.zeros(point_cloud_ego.shape[:2], device=point_cloud_ego.device, dtype=torch.bool)
        point_cloud_ego, role_scores, point_is_pad = select_objectflow_points(
            point_cloud_ego,
            role_scores,
            point_is_pad,
            max_points=int(self.config.worldflow_max_points),
        )
        current_pose = batch["worldflow.current_ee_pose"].to(device=point_cloud_ego.device, dtype=torch.float32)
        point_cloud_world = _ego_point_cloud_to_world(point_cloud_ego, current_pose)

        target_poses = batch["worldflow.ee_poses"].to(device=point_cloud_world.device, dtype=torch.float32)
        step_is_pad = batch["worldflow.step_is_pad"].to(device=point_cloud_world.device, dtype=torch.bool)
        if target_poses.ndim != 3 or target_poses.shape[-1] != 9:
            raise ValueError(f"Expected worldflow.ee_poses shape (B,T,9), got {target_poses.shape}.")
        if current_pose.ndim != 2 or current_pose.shape[-1] != 9:
            raise ValueError(f"Expected worldflow.current_ee_pose shape (B,9), got {current_pose.shape}.")
        if target_poses.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"Expected worldflow.ee_poses time dim={self.config.chunk_size}, got {target_poses.shape[1]}."
            )
        if step_is_pad.shape != target_poses.shape[:2]:
            raise ValueError(f"Expected worldflow.step_is_pad shape {target_poses.shape[:2]}, got {step_is_pad.shape}.")

        if torch.is_tensor(cached_lang_emb) and cached_lang_emb.shape[:2] == lang_masks.shape:
            lang_emb = cached_lang_emb.to(device=point_cloud_world.device)
        else:
            lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        current = pose9_to_matrix(current_pose)
        target = pose9_to_matrix(target_poses)
        current_inv = invert_transform(current)
        spatial_gt = target @ current_inv.unsqueeze(1)

        valid = ~step_is_pad
        if torch.is_tensor(actions_is_pad):
            action_pad = actions_is_pad.to(device=valid.device, dtype=torch.bool)
            if action_pad.shape[1] != valid.shape[1]:
                raise ValueError(f"Expected actions_id_pad shape {valid.shape}, got {action_pad.shape}.")
            valid = valid & ~action_pad

        pred_flow = self.worldflow_head(point_cloud_world, role_scores, lang_emb, lang_masks, point_is_pad)
        world_xyz = point_cloud_world[..., :3].to(dtype=torch.float32)
        gt_target_xyz = torch.einsum("btij,bnj->btni", spatial_gt[..., :3, :3], world_xyz)
        gt_target_xyz = gt_target_xyz + spatial_gt[..., :3, 3].unsqueeze(2)
        gt_flow = gt_target_xyz - world_xyz.unsqueeze(1)

        # role_scores[..., 1] is generated from temporal held/static residuals and
        # represents the manipulated/transported object. It explicitly suppresses
        # role_scores[..., 0], the automatically detected virtual-gripper region.
        transport_weights = role_scores[..., 1].clamp(0.0, 1.0)
        transport_weights = transport_weights * (~point_is_pad).to(dtype=transport_weights.dtype)
        active_transport = (
            transport_weights >= float(self.config.worldflow_transport_score_threshold)
        ).sum(dim=1)
        has_transport = active_transport >= int(self.config.worldflow_min_transport_points)

        # Kabsch supplies the analytic rigid carrier, but SVD gradients are
        # intentionally avoided: repeated singular values are common for
        # symmetric objects. Gradients flow through the dense flow losses below.
        with torch.no_grad():
            pred_spatial, rigid_valid = weighted_kabsch_transform(
                world_xyz,
                world_xyz.unsqueeze(1) + pred_flow.detach(),
                transport_weights,
            )
        pred_body_from_object = current_inv.unsqueeze(1) @ pred_spatial @ current.unsqueeze(1)

        valid = valid & has_transport.unsqueeze(1) & rigid_valid
        valid_count = valid.sum(dim=1).clamp_min(1)

        point_flow_error = F.smooth_l1_loss(pred_flow, gt_flow, reduction="none").mean(dim=-1)
        weight_denom = transport_weights.sum(dim=1).clamp_min(1e-6)
        flow_step = (
            point_flow_error * transport_weights.unsqueeze(1)
        ).sum(dim=-1) / weight_denom.unsqueeze(1)

        fitted_target_xyz = torch.einsum("btij,bnj->btni", pred_spatial[..., :3, :3], world_xyz)
        fitted_target_xyz = fitted_target_xyz + pred_spatial[..., :3, 3].unsqueeze(2)
        rigid_point_error = F.smooth_l1_loss(
            world_xyz.unsqueeze(1) + pred_flow,
            fitted_target_xyz.detach(),
            reduction="none",
        ).mean(dim=-1)
        rigid_step = (
            rigid_point_error * transport_weights.unsqueeze(1)
        ).sum(dim=-1) / weight_denom.unsqueeze(1)
        rigid_rot = _rotation_geodesic(pred_spatial[..., :3, :3], spatial_gt[..., :3, :3])

        body_pose9 = self.last_body_pose9_prediction
        bridge_step = torch.zeros_like(rigid_step)
        if torch.is_tensor(body_pose9):
            body_pose9 = body_pose9.to(device=point_cloud_world.device, dtype=torch.float32)
            if body_pose9.shape[:2] != target_poses.shape[:2] or body_pose9.shape[-1] < 9:
                raise ValueError(
                    f"Expected Ego body prediction shape (B,T,>=9) matching {target_poses.shape[:2]}, "
                    f"got {body_pose9.shape}."
                )
            pred_body = pose9_to_matrix(body_pose9[..., :9])
            pred_body_spatial = current.unsqueeze(1) @ pred_body @ current_inv.unsqueeze(1)
            body_target_xyz = torch.einsum("btij,bnj->btni", pred_body_spatial[..., :3, :3], world_xyz)
            body_target_xyz = body_target_xyz + pred_body_spatial[..., :3, 3].unsqueeze(2)
            body_flow = body_target_xyz - world_xyz.unsqueeze(1)
            bridge_error = F.smooth_l1_loss(pred_flow, body_flow, reduction="none").mean(dim=-1)
            bridge_step = (
                bridge_error * transport_weights.unsqueeze(1)
            ).sum(dim=-1) / weight_denom.unsqueeze(1)

        equiv_step = torch.zeros_like(rigid_step)
        if self.config.worldflow_equiv_loss_weight > 0:
            transform = _sample_random_se3(
                point_cloud_world.shape[0],
                point_cloud_world.device,
                point_cloud_world.dtype,
                trans_scale=0.20,
                rot_scale=0.75,
            )
            point_cloud_aug = _transform_point_cloud_xyzrgb(point_cloud_world, transform)
            pred_aug_flow = self.worldflow_head(point_cloud_aug, role_scores, lang_emb, lang_masks, point_is_pad)
            expected_aug_flow = torch.einsum("bij,btnj->btni", transform[..., :3, :3], pred_flow.detach())
            equiv_error = F.smooth_l1_loss(pred_aug_flow, expected_aug_flow, reduction="none").mean(dim=-1)
            equiv_step = (
                equiv_error * transport_weights.unsqueeze(1)
            ).sum(dim=-1) / weight_denom.unsqueeze(1)

        per_sample_flow = _masked_step_mean(flow_step, valid)
        per_sample_rigid = _masked_step_mean(rigid_step, valid)
        per_sample_bridge = _masked_step_mean(bridge_step, valid)
        per_sample_equiv = _masked_step_mean(equiv_step, valid)
        loss_flow = per_sample_flow.mean()
        loss_rigid = per_sample_rigid.mean()
        loss_bridge = per_sample_bridge.mean()
        loss_equiv = per_sample_equiv.mean()
        per_sample_total = (
            self.config.worldflow_loss_weight * per_sample_flow
            + self.config.worldflow_geo_loss_weight * per_sample_rigid
            + self.config.worldflow_bridge_loss_weight * per_sample_bridge
            + self.config.worldflow_equiv_loss_weight * per_sample_equiv
        )

        valid_f = valid.to(dtype=pred_flow.dtype)
        valid_denom = valid_f.sum().clamp_min(1.0)
        trans_err = (
            torch.linalg.norm(pred_spatial[..., :3, 3] - spatial_gt[..., :3, 3], dim=-1) * valid_f
        ).sum() / valid_denom
        rot_err = (rigid_rot * valid_f).sum() / valid_denom
        transport_point_ratio = (
            (transport_weights >= float(self.config.worldflow_transport_score_threshold))
            .to(dtype=pred_flow.dtype)
            .mean()
        )
        self.last_worldflow_metrics = {
            "loss_worldflow_flow": loss_flow.detach(),
            "loss_worldflow_rigid": loss_rigid.detach(),
            "loss_worldflow_bridge": loss_bridge.detach(),
            "loss_worldflow_equiv": loss_equiv.detach(),
            "worldflow_trans_err": trans_err.detach(),
            "worldflow_rot_err_deg": torch.rad2deg(rot_err.detach()),
            "worldflow_valid_ratio": valid_f.mean().detach(),
            "worldflow_transport_point_ratio": transport_point_ratio.detach(),
        }
        return {
            "loss_flow": loss_flow,
            "loss_rigid": loss_rigid,
            "loss_bridge": loss_bridge,
            "loss_equiv": loss_equiv,
            "per_sample_loss": per_sample_total,
            "valid_counts": valid_count,
            "pred_spatial": pred_spatial,
            "pred_spatial_pose9": matrix_to_pose9(pred_spatial),
            "pred_body_pose9": matrix_to_pose9(pred_body_from_object),
            "pred_flow": pred_flow,
        }

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
            h_noise = pose9_to_matrix(noise[..., :9])
            gripper_noise = noise[..., 9:10].to(dtype=torch.float32)

        time_pose = time[:, None, None]
        h_delta = h_gt @ invert_transform(h_noise)
        xi_total = se3_log(h_delta)
        h_t = se3_exp(time_pose * xi_total) @ h_noise
        remaining = (1.0 - time).clamp_min(1e-3)
        xi_target = se3_log(h_gt @ invert_transform(h_t)) / remaining[:, None, None]
        gripper_t = (1.0 - time_pose) * gripper_noise + time_pose * gripper_gt
        gripper_target = gripper_gt - gripper_noise
        x_t = torch.cat([matrix_to_pose9(h_t), gripper_t], dim=-1)
        if actions.shape[-1] > 10:
            x_t = pad_vector(x_t, actions.shape[-1])

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats, pc_masks, lang_tokens, lang_masks, state=state
        )
        pred = self._se3_predict_from_suffix(prefix_embs, prefix_pad_masks, prefix_att_masks, x_t, time)
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

        losses = actions.new_zeros(actions.shape)
        losses[..., 0] = step_total * actions.shape[-1]
        return losses

    def forward(
        self, pc_feats, pc_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, actions_is_pad=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
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
            )
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = (1 - time_expanded) * noise + time_expanded * actions
        u_t = actions - noise
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats, pc_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

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
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if x_t.shape[-1] >= 9 and v_t.shape[-1] >= 9:
            endpoint = x_t + (1.0 - time_expanded) * v_t
            self.last_body_pose9_prediction = endpoint[..., :9]
        else:
            self.last_body_pose9_prediction = None
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
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        if self.config.se3_enable:
            return self.sample_actions_se3(
                pc_feats, pc_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
            )
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats, pc_masks, lang_tokens, lang_masks, state=state
        )
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
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def sample_actions_se3(
        self,
        pc_feats,
        pc_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pc_feats, pc_masks, lang_tokens, lang_masks, state=state
        )
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
            pred = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            twist = pred[..., :6]
            gripper_vel = pred[..., 6:7]
            h_t = pose9_to_matrix(x_t[..., :9])
            h_next = se3_left_apply(dt * twist, h_t)
            gripper_next = x_t[..., 9:10] + dt * gripper_vel
            x_t = torch.cat([matrix_to_pose9(h_next), gripper_next], dim=-1)
            if x_t.shape[-1] < self.config.max_action_dim:
                x_t = pad_vector(x_t, self.config.max_action_dim)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

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
            if self.se3_action_out_proj is None:
                raise RuntimeError("se3_action_out_proj is not initialized.")
            v_t = self.se3_action_out_proj(suffix_out)
        else:
            v_t = self.action_out_proj(suffix_out)
        return v_t
