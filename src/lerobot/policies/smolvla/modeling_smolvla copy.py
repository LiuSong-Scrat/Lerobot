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
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype

from .litept.model import LitePT



import numpy as np
import torch.nn.functional as F
import open3d as o3d
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
def vis_umi_data(action,pointcloud):
    ##########UMI
    # ================= Pred =================

    geometries =[]
    origin_frame = create_frame(np.array([0,0,0]), np.eye(3), scale=0.03)
    geometries.append(origin_frame)
    for per_pred_action in action: ####GT
        per_pred_action = per_pred_action
        pred_xyz = per_pred_action[:3]
        pred_rot6d = per_pred_action[3:9]
        pred_rotmat = rot6d_to_matrix(torch.tensor(pred_rot6d)).cpu().numpy()
        frame = create_frame(pred_xyz, pred_rotmat, scale=0.03)
        geometries.append(frame)

    # ================= Scene Point Cloud =================
    cloud = pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] / 255)
    geometries.append(pcd)

    o3d.visualization.draw_geometries(geometries)
    ##########UMI


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
        actions_is_pad = batch.get(f"{ACTION}_is_pad")
        loss_dict = {}
        losses = self.model.forward(pc_feats, pc_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

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
            per_sample_loss = losses.sum(dim=(1, 2)) / (valid_action_counts * original_action_dim)
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.sum() / (valid_action_counts.sum() * original_action_dim)
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
        if pc.ndim == 4 and pc.shape[1] == 1:
            # Some preprocessors pad a singleton time/channel axis: (B, 1, N, C)
            pc = pc.squeeze(1)
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
        mask = torch.ones(bsize, dtype=torch.bool, device=device)
        # if "observation.point_cloud_is_pad" in batch:
        #     mask = batch["observation.point_cloud_is_pad"].bool()
        # else:
        #     mask = torch.ones(bsize, dtype=torch.bool, device=device)

        return [pc], [mask]

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
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
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

    def forward(self, x):
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

        # 3) softmax 沿 N 维 (即每行是 N 个点的权重)
        attn_weights = F.softmax(scores, dim=2)  # (B, N, N)

        # 4) 加权得到输出: (B, N, C)
        x_out = torch.bmm(attn_weights, V)  # (B, N, C)

        # 5) 可选：加 LayerNorm + 残差
        x_out = self.norm(x_out + x)  # (B, N, C)

        return x_out   

class LitePTTokenizer(nn.Module):
    """
    input:  pc (B,N,C) XYZ m RGB 0-255
    output: xyz_tok (B,T,3), tok (B,T,dim), g (B,dim)
    """

    def __init__(self, in_dim=6, dim=128, n_tokens=512, grid_size=0.005):
        super().__init__()
        self.dim = dim
        self.n_tokens = n_tokens
        self.grid_size = grid_size

        self.backbone = LitePT(in_channels=in_dim)
        self.out_proj = nn.LazyLinear(dim)

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

    def forward(self, pc):
        B, N, C = pc.shape
        T = self.n_tokens
        device = pc.device

        # ========= 输出 =========
        global_xyz_tok = torch.zeros(B, T, 3, device=device, dtype=pc.dtype)
        tok = torch.zeros(B, T, self.dim, device=device, dtype=pc.dtype)
        g = torch.zeros(B, self.dim, device=device, dtype=pc.dtype)

        # ========= mask valid =========
        valid_mask = []
        for b in range(B):
            valid_mask.append(not self._is_degenerate(pc[b, :, :3]))
        valid_mask = torch.tensor(valid_mask, device=device)

        if valid_mask.sum() == 0:
            return global_xyz_tok, tok, g

        pc_v = pc[valid_mask]  # (Bv,N,C)
        Bv = pc_v.shape[0]

        # ========= flatten =========
        coord = pc_v[:, :, :3].reshape(-1, 3).contiguous()

        feat = pc_v.reshape(-1, C).contiguous()
        if C > 3:
            feat = torch.cat([feat[:, :3], feat[:, 3:] / 255.0], dim=1)
        else:
            feat = feat[:, :3]

        # batch index
        batch = torch.arange(Bv, device=device).repeat_interleave(N)

        # ========= grid sample =========
        coord, feat, batch = self._grid_sample_batch(coord, feat, batch)

        if coord.shape[0] == 0:
            return global_xyz_tok, tok, g

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
            b_p = point.batch
        else:
            # fallback（关键🔥）
            b_p = torch.bucketize(
                torch.arange(xyz_p.shape[0], device=device),
                offset
            )

        # ========= 按batch拆分 =========
        valid_idx = torch.nonzero(valid_mask).squeeze(1)

        for i in range(Bv):
            global_b = valid_idx[i]

            idx_all = torch.nonzero(b_p == i, as_tuple=False).squeeze(1)
            P = idx_all.numel()

            if P == 0:
                continue

            if P >= T:
                sel = idx_all[torch.randperm(P, device=device)[:T]]
            else:
                sel = idx_all[torch.randint(0, P, (T,), device=device)]

            global_xyz_tok[global_b] = xyz_p[sel]
            tok[global_b] = feat_p[sel]
            g[global_b] = feat_p[idx_all].max(dim=0).values

        return global_xyz_tok, tok, g
    

class LitePTEncoder(nn.Module):
    """
    input:  pc (B,N,C)
    output: xyz_tok (B,T,3), tok (B,T,dim), g (B,dim)
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



    def forward(self, scene_pc):
        scene_xyz, scene_tok, scenen_g = self.pc_backbone(scene_pc)   # (B,T,3),(B,T,C),(B,C)
        scene_tok = self.attention(scene_tok)
        # 3. 权重加权求和得到全局特征
        alpha = self.att(scene_tok)        # [B, N, C]


        center = (scene_xyz * alpha).sum(dim=1)
        centroid_xyz = scene_pc[...,:3] - center.unsqueeze(-2)
        scene_pc1 = torch.cat([centroid_xyz, scene_pc[...,3:]], dim=-1)
        scene_xyz1, scene_tok1, scenen_g1 = self.pc_backbone1(scene_pc1)   # (B,T,3),(B,T,C),(B,C)
        scene_tok1 = self.attention1(scene_tok1)
        # 3. 权重加权求和得到全局特征
        alpha1 = self.att1(scene_tok1)        # [B, N, C]


        global_feat = (scene_tok1 * alpha1).sum(dim=1)  # [B, C]
        return global_feat



        
        # # has_cond = torch.ones(global_xyz_tok.shape[0])
        # # t_xyz, t_tok = global_xyz_tok, tok
        # # c_xyz, c_tok = global_xyz_tok, tok
        # # t_tok_rel = self.dense_xattn_full(t_xyz, t_tok, c_xyz, c_tok, has_cond)  # (B,T,C)  
        # # # Only Cross Attention
        # # t_mem = self.proj_pc(t_tok_rel)  # (B,T,D)
        # # global_feat = t_mem.max(dim=1).values 

        # tok = self.attention(tok)
        # # 3. 权重加权求和得到全局特征
        # alpha = self.att(tok)        # [B, N, C]
        # global_feat = (tok * alpha).sum(dim=1)  # [B, C]

        # # # slot3 = self.slot_attn(tok) #3cluster cond target else
        # # # slot_relation_feature = self.relation(slot3) #cluster cross attention
        # # # fused_feature = self.token_fusion_mlp(slot_relation_feature.reshape(B, -1))


        # return global_feat 
    



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
        self.extractor = LitePTEncoder(in_dim=6, dim=64, n_tokens=256, grid_size=0.005)
        self.pointcloud_proj = nn.Linear(64, self.vlm_with_expert.config.text_config.hidden_size)

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
            params.requires_grad = self.config.train_state_proj

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

    def embed_prefix(
        self, point_clouds, point_cloud_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed point cloud features and language tokens to prepare for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        for _pc_idx, (
            pc,
            pc_mask,
        ) in enumerate(zip(point_clouds, point_cloud_masks, strict=False)):
            # Point cloud features replace the original image features.
            # The original SmolVLA image processing is intentionally disabled.
            if pc.ndim != 3:
                raise ValueError(f"Expected point cloud input shape (B, N, C), got {pc.shape}")

            global_feat = self.extractor(pc)  # (B, C)
            pc_emb = self.pointcloud_proj(global_feat).unsqueeze(1)
            pc_emb_dim = pc_emb.shape[-1]
            pc_emb = pc_emb * math.sqrt(pc_emb_dim)
            num_pc_tokens = 1
            pc_emb = pc_emb.expand(-1, num_pc_tokens, -1)  # replicate the global feature token

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
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs



        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # Set attention masks so that point cloud and language inputs do not attend to state or actions
        att_masks += [1] * states_seq_len




        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
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

    def forward(
        self, pc_feats, pc_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
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
        v_t = self.action_out_proj(suffix_out)
        return v_t
