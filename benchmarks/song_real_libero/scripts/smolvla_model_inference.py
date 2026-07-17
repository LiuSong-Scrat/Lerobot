#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.song_pointseg import open_episode_point_clouds
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_IMAGES,
    OBS_STATE,
)

if __package__:
    from .libero_setting.libero_pointcloud_utils import (
        add_local_gripper_cloud_to_point_cloud,
        add_world_gripper_cloud_to_point_cloud,
        gripper_width_percent_from_scalar,
    )
else:
    from libero_setting.libero_pointcloud_utils import (
        add_local_gripper_cloud_to_point_cloud,
        add_world_gripper_cloud_to_point_cloud,
        gripper_width_percent_from_scalar,
    )


DEFAULT_POLICY_PATH = (
    "/home/liusong/ProgramFiles/Huggingface/lerobot/"
    "outputs/train/my_smolvla_song1/checkpoints/last/pretrained_model"
)
DEFAULT_POLICY_REPO_ID = "/home/liusong/scp_receive/smolvla"


class PointCloudMemmapDataset(torch.utils.data.Dataset):
    """Inject point clouds from per-episode zarr/npy arrays into a LeRobotDataset item."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        point_cloud_dir: str | Path,
        key: str = "observation.point_cloud",
        mmap_mode: str = "r",
    ) -> None:
        self.dataset = dataset
        self.point_cloud_dir = Path(point_cloud_dir)
        self.key = key
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[int, np.ndarray] = {}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, episode_index: int) -> np.ndarray:
        point_clouds = self._point_cloud_cache.get(episode_index)
        if point_clouds is None:
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dir,
                episode_index,
                mmap_mode=self.mmap_mode,
            )
            self._point_cloud_cache[episode_index] = point_clouds
        return point_clouds

    def __getitem__(self, idx):
        item = self.dataset[idx]
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        point_cloud = self._episode_point_clouds(episode_index)[frame_index]
        item[self.key] = torch.from_numpy(point_cloud).unsqueeze(0)
        return item


def maybe_wrap_point_cloud_memmap_dataset(dataset):
    root_value = getattr(dataset, "root", None)
    if root_value is None:
        root_value = dataset.meta.root
    root = Path(root_value)
    point_cloud_dir = root / "point_clouds"
    if not point_cloud_dir.is_dir():
        return dataset
    mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
    return PointCloudMemmapDataset(dataset, point_cloud_dir=point_cloud_dir, mmap_mode=mmap_mode)

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

def pose9_to_homo(pose9: torch.Tensor) -> torch.Tensor:
    t = pose9[..., 0:3]
    rot = rot6d_to_matrix(pose9[..., 3:9])
    H = torch.zeros(*pose9.shape[:-1], 4, 4, device=pose9.device, dtype=pose9.dtype)
    H[..., 3, 3] = 1.0
    H[..., :3, :3] = rot
    H[..., :3, 3] = t
    return H


def pose9_to_traj6(pose9: np.ndarray) -> np.ndarray:
    t = pose9[:3].astype(np.float32)
    r1 = pose9[3:6].astype(np.float32)
    r2 = pose9[6:9].astype(np.float32)
    u1 = r1 / np.clip(np.linalg.norm(r1), 1e-8, None)
    r2 = r2 - np.dot(u1, r2) * u1
    u2 = r2 / np.clip(np.linalg.norm(r2), 1e-8, None)
    u3 = np.cross(u1, u2)
    rot = np.stack([u1, u2, u3], axis=1)
    euler_zyx = R.from_matrix(rot).as_euler("zyx", degrees=False).astype(np.float32)
    return np.concatenate([t, euler_zyx], axis=0)


def from_H_to_trajectory(H: np.ndarray) -> np.ndarray:
    position = H[...,:3, 3]
    rotation_matrix = H[..., :3, :3]
    euler_zyx = R.from_matrix(rotation_matrix).as_euler("zyx", degrees=False)
    return np.hstack((position, euler_zyx)).astype(np.float32)


def traj6_to_pose9(traj6: np.ndarray) -> np.ndarray:
    traj6 = np.asarray(traj6, dtype=np.float32)
    if traj6.ndim < 2:
        t = traj6[:3].astype(np.float32)
        euler_zyx = traj6[3:].astype(np.float32)
        rot = R.from_euler("zyx", euler_zyx, degrees=False).as_matrix().astype(np.float32)
        rot6d = np.hstack((rot[:, 0], rot[:, 1]))
        return np.concatenate([t, rot6d], axis=0)

    t = traj6[:, :3].astype(np.float32)
    euler_zyx = traj6[:, 3:].astype(np.float32)
    rot = R.from_euler("zyx", euler_zyx, degrees=False).as_matrix().astype(np.float32)
    rot6d = np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=1)
    return np.concatenate([t, rot6d], axis=1)


def fast_inverse_homogeneous(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    T_inv = np.zeros_like(T)
    rot = T[..., :3, :3]
    trans = T[..., :3, 3:4]
    rot_inv = np.swapaxes(rot, -1, -2)
    T_inv[..., :3, :3] = rot_inv
    T_inv[..., :3, 3:4] = -(rot_inv @ trans)
    T_inv[..., 3, 3] = 1
    return T_inv


class SmolVLA_ModelInference:
    """Small inference wrapper for the local point-cloud SmolVLA policy."""

    def __init__(
        self,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        policy_repo_id: str | Path | None = DEFAULT_POLICY_REPO_ID,
        *,
        device: str = "cuda",
        processor_path: str | Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.policy_path = str(policy_path)
        self.policy_repo_id = str(policy_repo_id) if policy_repo_id is not None else None
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        config = PreTrainedConfig.from_pretrained(
            self.policy_path,
            cli_overrides=[f"--device={self.device}"],
            local_files_only=local_files_only,
        )
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}.")
        self.policy = SmolVLAPolicy.from_pretrained(
            self.policy_path,
            config=config,
            local_files_only=local_files_only,
        )
        self.policy.eval()
        self.policy.reset()

        self.processor_path = self._resolve_processor_path(processor_path)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            self.processor_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        self.dataset: torch.utils.data.Dataset | None = None
        self.predict_action_queue = deque()
        self.last_model_point_cloud: np.ndarray | None = None
        self.policy.n_action_steps = 16
        self.policy.horizon = 32
    def _resolve_processor_path(self, processor_path: str | Path | None) -> str | None:
        candidates = [processor_path, self.policy_path, self.policy_repo_id]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_path = Path(candidate)
            if (candidate_path / "policy_preprocessor.json").is_file():
                return str(candidate_path)
        return None

    @staticmethod
    def _resolve_dataset_location(dataset_repo_id: str | Path, root: str | Path | None = None) -> tuple[str, Path | None]:
        if root is not None:
            return str(dataset_repo_id), Path(root).expanduser().resolve()

        candidate = Path(dataset_repo_id).expanduser()
        if candidate.exists():
            resolved = candidate.resolve()
            if not (resolved / "meta" / "info.json").exists():
                raise FileNotFoundError(
                    f"Local dataset path exists but is missing meta/info.json: {resolved}"
                )
            return resolved.name, resolved

        return str(dataset_repo_id), None

    def load_dataset(
        self,
        dataset_repo_id: str | Path,
        *,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
    ) -> torch.utils.data.Dataset:
        repo_id, root_path = self._resolve_dataset_location(dataset_repo_id, root)
        meta = LeRobotDatasetMetadata(repo_id, root=root_path)
        delta_timestamps = resolve_delta_timestamps(self.policy.config, meta)
        dataset = LeRobotDataset(
            repo_id,
            root=root_path,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
        self.dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
        return self.dataset

    @torch.inference_mode()
    def predict_from_dataset(
        self,
        index: int = 0,
        *,
        dataset_repo_id: str | Path | None = None,
        dataset_root: str | Path | None = None,
        task: str | None = None,
        postprocess: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        if self.dataset is None:
            if dataset_repo_id is None:
                raise ValueError("dataset_repo_id is required before the dataset has been loaded.")
            self.load_dataset(dataset_repo_id, root=dataset_root)

        assert self.dataset is not None
        batch = default_collate([dict(self.dataset[index])])
        if task is not None:
            batch["task"] = [task]

        model_batch = self.preprocessor(batch)
        action_chunk = self.policy.predict_action_chunk(model_batch)
        first_action = action_chunk[:, 0]

        if postprocess:
            action_chunk = self.postprocessor(action_chunk)
            first_action = self.postprocessor(first_action)

        gt_action_chunk = model_batch.get(ACTION)
        if gt_action_chunk is not None:
            gt_action_chunk = gt_action_chunk.detach().cpu()

        return {
            "action": first_action.detach().cpu(),
            "action_chunk": action_chunk.detach().cpu(),
            "gt_action_chunk": gt_action_chunk,
        }

    @torch.inference_mode()
    def predict_action_chunk_obs(
        self,
        observation: dict[str, Any],
        *,
        task: str | list[str] = "",
        postprocess: bool = True,
        state_pose_mode: str = "identity",
    ) -> torch.Tensor:
        model_batch = self.build_model_batch(
            observation,
            task=task,
            state_pose_mode=state_pose_mode,
        )
        action_chunk = self.policy.predict_action_chunk(model_batch)
        if postprocess:
            action_chunk = self.postprocessor(action_chunk)
        return action_chunk.detach().cpu()

    @torch.inference_mode()
    def select_action(
        self,
        observation: dict[str, Any],
        *,
        task: str | list[str] = "",
        postprocess: bool = True,
        state_pose_mode: str = "identity",
    ) -> torch.Tensor:
        model_batch = self.build_model_batch(
            observation,
            task=task,
            state_pose_mode=state_pose_mode,
        )
        action = self.policy.select_action(model_batch)
        if postprocess:
            action = self.postprocessor(action)
        return action.detach().cpu()

    @torch.inference_mode()
    def single_inference(
        self,
        cur_model_observation: dict[str, Any],
        visualize: bool = False,
        *,
        task: str = "",
        num_points: int = 10000,
        add_gripper_cloud: bool = True,
        gripper_points: int = 500,
        gripper_len: float = 0.06,
        gripper_template: str = "reap",
        gripper_drop_strategy: str = "tail",
        gripper_shuffle_points: bool = False,
        gripper_qpos_max_width: float = 0.08,
    ) -> np.ndarray:
        """Run one real-robot style inference step.

        The input format matches the pickle produced by the DP3 realtime path:
        the first seven non-point-cloud entries are interpreted as
        xyz + euler_zyx + gripper, and `point_cloud` is expected to be (N, 6).
        Returns xyz + euler_zyx + gripper in the world frame.
        """
        one_step_agent_pos = self._real_observation_to_pose9_gripper(cur_model_observation)
        point_cloud_world = self._to_numpy(cur_model_observation["point_cloud"]).astype(np.float32)
        gripper_width_percent = gripper_width_percent_from_scalar(
            float(one_step_agent_pos[-1]),
            max_physical_width=gripper_qpos_max_width,
        )
        if add_gripper_cloud:
            one_step_point_cloud = add_world_gripper_cloud_to_point_cloud(
                point_cloud_world,
                one_step_agent_pos,
                gripper_width_percent,
                total_points=num_points,
                gripper_points=gripper_points,
                gripper_len=gripper_len,
                gripper_template=gripper_template,
                drop_strategy=gripper_drop_strategy,
                shuffle_points=gripper_shuffle_points,
            )
        else:
            one_step_point_cloud = self._world_point_cloud_to_current_eff(
                point_cloud_world,
                one_step_agent_pos,
            )
            one_step_point_cloud = prepare_inference_point_cloud(
                one_step_point_cloud,
                num_points=num_points,
                add_gripper_cloud=False,
                gripper_width_percent=gripper_width_percent,
                gripper_points=gripper_points,
                gripper_len=gripper_len,
                gripper_template=gripper_template,
                gripper_drop_strategy=gripper_drop_strategy,
                gripper_shuffle_points=gripper_shuffle_points,
            )
        self.last_model_point_cloud = one_step_point_cloud
        if len(self.predict_action_queue)==0:
            model_observation = {
                "point_cloud": one_step_point_cloud,
                "state": one_step_agent_pos,
            }
            for image_alias in ("overhead", "hand"):
                if image_alias in cur_model_observation:
                    model_observation[image_alias] = cur_model_observation[image_alias]
            action_chunk = self.predict_action_chunk_obs(
                model_observation,
                task=task,
                postprocess=True,
                state_pose_mode="identity",
            )
            if action_chunk.ndim != 3:
                raise ValueError(f"Expected action chunk shape (B, T, D), got {tuple(action_chunk.shape)}")
            pred_eff_to_eff0 = action_chunk[0].to(dtype=torch.float32)
            current_eff_to_world = pose9_to_homo(torch.as_tensor(one_step_agent_pos)).cpu().numpy()
            pred_world_pose9_gripper = self._umi_action_to_world_pose9_gripper(
                pred_eff_to_eff0,
                current_eff_to_world,
            )
            self.predict_action_queue.extend(pred_world_pose9_gripper)
            if visualize:
                vis_umi_data(pred_world_pose9_gripper, one_step_point_cloud)

        pred_action = self.predict_action_queue.popleft()
        pred_pose9_gripper_np = pred_action
        traj6_np = pose9_to_traj6(pred_pose9_gripper_np)
        traj6_np_with_gripper = np.concatenate([traj6_np, np.array([pred_pose9_gripper_np[-1]])], axis=0)

        return traj6_np_with_gripper


    def build_model_batch(
        self,
        observation: dict[str, Any],
        *,
        task: str | list[str] = "",
        state_pose_mode: str = "identity",
    ) -> dict[str, torch.Tensor]:
        point_cloud = self._get_observation_value(observation, "observation.point_cloud", "point_cloud")
        state = self._get_observation_value(observation, OBS_STATE, "state", default=None)

        pc = self._to_tensor(point_cloud, dtype=torch.float32)
        if pc.ndim == 2:
            pc = pc.unsqueeze(0)
        elif pc.ndim == 4 and pc.shape[1] == 1:
            pc = pc.squeeze(1)
        if pc.ndim != 3 or pc.shape[-1] != 6:
            raise ValueError(f"Expected point cloud shape (N, 6) or (B, N, 6), got {tuple(pc.shape)}")
        batch_size = pc.shape[0]

        state_tensor = self._prepare_state_tensor(state, state_pose_mode=state_pose_mode)
        state_tensor = self._match_batch_size(state_tensor, batch_size, "observation.state")
        language = self._tokenize_task(task, batch_size)

        batch = {
            "observation.point_cloud": pc.to(self.device),
            OBS_STATE: state_tensor.to(self.device),
            OBS_LANGUAGE_TOKENS: language["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: language["attention_mask"].to(self.device, dtype=torch.bool),
        }
        if self.policy.config.vla_adapter_enable:
            batch.update(self._prepare_rgb_observation_batch(observation, batch_size))
        return batch

    def _prepare_rgb_observation_batch(
        self,
        observation: dict[str, Any],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        image_features = dict(self.policy.config.image_features)
        if not image_features:
            raise ValueError(
                "This frozen-VLM adapter checkpoint has no image feature in its config. "
                f"Expected a feature such as '{OBS_IMAGES}.overhead'."
            )
        image_batch = {}
        for image_key in image_features:
            image_value = self._get_rgb_observation_value(observation, image_key)
            if image_value is not None:
                image_batch[image_key] = self._prepare_image_tensor(
                    image_value, batch_size, image_key
                ).to(self.device)
        if not image_batch:
            raise ValueError(
                "Frozen-VLM adapter inference requires RGB input. "
                f"Expected one of {list(image_features)} or aliases 'overhead'/'hand'; "
                f"available keys are {sorted(str(key) for key in observation)}."
            )
        return image_batch

    def _get_rgb_observation_value(self, observation: dict[str, Any], image_key: str) -> Any | None:
        if image_key in observation:
            return observation[image_key]
        short_key = image_key[len(f"{OBS_IMAGES}.") :] if image_key.startswith(f"{OBS_IMAGES}.") else image_key
        candidates = [short_key, short_key.replace(".", "_"), short_key.replace("_rgb", "")]
        lowered = image_key.lower()
        if any(name in lowered for name in ("overhead", "overview", "top")):
            candidates += ["overhead", "overview", "top"]
        if any(name in lowered for name in ("hand", "wrist")):
            candidates += ["hand", "wrist"]
        return next((observation[key] for key in candidates if key in observation), None)

    def _prepare_image_tensor(self, image: Any, batch_size: int, image_key: str) -> torch.Tensor:
        tensor = self._to_tensor(image, dtype=torch.float32)
        if tensor.numel():
            if tensor.detach().amax() > 2.0:
                tensor = tensor / 255.0
            elif tensor.detach().amin() < 0.0:
                tensor = (tensor + 1.0) * 0.5
        if tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
                tensor = tensor[..., :3].permute(2, 0, 1)
            elif tensor.shape[0] in (1, 3, 4):
                tensor = tensor[:3]
            else:
                raise ValueError(f"Expected {image_key} as HWC/CHW RGB, got {tuple(tensor.shape)}.")
            if tensor.shape[0] == 1:
                tensor = tensor.expand(3, -1, -1)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4):
                tensor = tensor[..., :3].permute(0, 3, 1, 2)
            elif tensor.shape[1] in (1, 3, 4):
                tensor = tensor[:, :3]
            else:
                raise ValueError(f"Expected batched {image_key} as BHWC/BCHW, got {tuple(tensor.shape)}.")
            if tensor.shape[1] == 1:
                tensor = tensor.expand(-1, 3, -1, -1)
        elif tensor.ndim == 5:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[2] not in (1, 3, 4):
                tensor = tensor[..., :3].permute(0, 1, 4, 2, 3)
            elif tensor.shape[2] in (1, 3, 4):
                tensor = tensor[:, :, :3]
            else:
                raise ValueError(f"Expected temporal {image_key} as BTHWC/BTCHW, got {tuple(tensor.shape)}.")
            if tensor.shape[2] == 1:
                tensor = tensor.expand(-1, -1, 3, -1, -1)
        else:
            raise ValueError(f"Expected {image_key} ndim 3/4/5, got {tuple(tensor.shape)}.")

        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, *tensor.shape[1:]).clone()
        elif tensor.shape[0] != batch_size:
            raise ValueError(
                f"Image {image_key} batch size {tensor.shape[0]} does not match point cloud batch {batch_size}."
            )
        return tensor.contiguous()

    def _prepare_state_tensor(self, state: Any | None, *, state_pose_mode: str) -> torch.Tensor:
        state_feature = self.policy.config.robot_state_feature
        state_dim = state_feature.shape[0] if state_feature is not None else self.policy.config.max_state_dim
        if state is None:
            state_tensor = torch.zeros(1, state_dim, dtype=torch.float32)
        else:
            state_tensor = self._to_tensor(state, dtype=torch.float32)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            elif state_tensor.ndim == 2 and state_tensor.shape[-1] == state_dim and state_tensor.shape[0] != 1:
                state_tensor = state_tensor.unsqueeze(0)
            if state_tensor.ndim > 2:
                state_tensor = state_tensor[:, -1]

        if state_tensor.shape[-1] < state_dim:
            pad = torch.zeros(*state_tensor.shape[:-1], state_dim - state_tensor.shape[-1])
            state_tensor = torch.cat([state_tensor, pad], dim=-1)
        state_tensor = state_tensor[..., :state_dim]

        if state_pose_mode == "identity":
            state_tensor = state_tensor.clone()
            state_tensor[..., :9] = 0.0
            state_tensor[..., 3] = 1.0
            state_tensor[..., 7] = 1.0
        elif state_pose_mode != "raw":
            raise ValueError("state_pose_mode must be 'identity' or 'raw'.")

        return state_tensor

    def _tokenize_task(self, task: str | list[str], batch_size: int) -> dict[str, torch.Tensor]:
        if isinstance(task, str):
            tasks = [task] * batch_size
        else:
            tasks = list(task)
            if len(tasks) == 1 and batch_size > 1:
                tasks = tasks * batch_size
            elif len(tasks) != batch_size:
                raise ValueError(f"Expected {batch_size} task strings, got {len(tasks)}.")
        tasks = [t if t.endswith("\n") else f"{t}\n" for t in tasks]
        return self.tokenizer(
            tasks,
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
            padding=self.policy.config.pad_language_to,
            padding_side="right",
            return_tensors="pt",
        )

    @staticmethod
    def _match_batch_size(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:]).clone()
        raise ValueError(f"{name} batch size {tensor.shape[0]} does not match point cloud batch size {batch_size}.")

    def _real_observation_to_pose9_gripper(self, observation: dict[str, Any]) -> np.ndarray:
        if "pose_eular" in observation:
            traj6 = self._to_numpy(observation["pose_eular"]).astype(np.float32).reshape(-1)
            if traj6.shape[0] < 6:
                raise ValueError(f"Expected pose_eular to contain 6 values, got shape {traj6.shape}.")
            if "gripper_width" in observation:
                gripper_width = self._to_float_scalar(observation["gripper_width"])
            elif "joint_7" in observation:
                gripper_width = self._to_float_scalar(observation["joint_7"])
            else:
                raise ValueError("Expected 'gripper_width' or 'joint_7' when using 'pose_eular'.")
            pose9 = traj6_to_pose9(traj6[:6])
            return np.concatenate([pose9, np.asarray([gripper_width], dtype=np.float32)], axis=0).astype(
                np.float32
            )

        pose_keys = [
            key
            for key in observation
            if key not in {"point_cloud", "overhead", "hand"}
            and not str(key).startswith(f"{OBS_IMAGES}.")
        ]
        if len(pose_keys) < 7:
            raise ValueError(
                "Expected at least 7 pose/gripper entries plus 'point_cloud' in cur_model_observation."
            )

        eff_xyz_euler_gripper = [
            self._to_float_scalar(observation[key])
            for key in pose_keys[:7]
        ]
        gripper_width = eff_xyz_euler_gripper[-1] * 0.5
        pose9 = traj6_to_pose9(np.asarray(eff_xyz_euler_gripper[:6], dtype=np.float32))
        return np.concatenate([pose9, np.asarray([gripper_width], dtype=np.float32)], axis=0).astype(
            np.float32
        )

    @staticmethod
    def _world_point_cloud_to_current_eff(
        point_cloud_world: np.ndarray,
        current_pose9_gripper: np.ndarray,
    ) -> np.ndarray:
        pc = np.asarray(point_cloud_world, dtype=np.float32)
        squeeze = pc.ndim == 2
        if squeeze:
            pc = pc[None]
        if pc.ndim != 3 or pc.shape[-1] != 6:
            raise ValueError(f"Expected point_cloud shape (N, 6) or (B, N, 6), got {point_cloud_world.shape}")

        pose = np.asarray(current_pose9_gripper, dtype=np.float32)
        if pose.ndim == 1:
            pose = pose[None]
        if pose.shape[0] == 1 and pc.shape[0] > 1:
            pose = np.repeat(pose, pc.shape[0], axis=0)
        if pose.shape[0] != pc.shape[0]:
            raise ValueError(f"Pose batch {pose.shape[0]} does not match point cloud batch {pc.shape[0]}.")

        eff_to_world = pose9_to_homo(torch.from_numpy(pose)).cpu().numpy()
        world_to_eff = fast_inverse_homogeneous(eff_to_world)
        xyz_h = np.concatenate([pc[..., :3], np.ones((*pc.shape[:2], 1), dtype=np.float32)], axis=-1)
        eff_xyz_h = np.einsum("bij,bnj->bni", world_to_eff, xyz_h)
        pc_eff = np.concatenate([eff_xyz_h[..., :3], pc[..., 3:]], axis=-1).astype(np.float32)
        return pc_eff[0] if squeeze else pc_eff

    @staticmethod
    def _umi_action_to_world_pose9_gripper(
        effi_to_eff0_pose9_gripper: torch.Tensor,
        current_eff_to_world: np.ndarray,
    ) -> np.ndarray:
        effi_to_eff0 = pose9_to_homo(effi_to_eff0_pose9_gripper).detach().cpu().numpy()
        effi_to_world = current_eff_to_world @ effi_to_eff0
        traj6_world = from_H_to_trajectory(effi_to_world)
        pose9_world = traj6_to_pose9(traj6_world)
        gripper = effi_to_eff0_pose9_gripper.detach().cpu().numpy()[...,-1:]
        return np.concatenate([pose9_world, gripper], axis=-1).astype(np.float32)

   
    @staticmethod
    def _to_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _to_float_scalar(cls, value: Any) -> float:
        arr = cls._to_numpy(value)
        return float(np.asarray(arr).reshape(-1)[0])

    @staticmethod
    def _get_observation_value(
        observation: dict[str, Any],
        canonical_key: str,
        short_key: str,
        *,
        default: Any = ...,
    ) -> Any:
        if canonical_key in observation:
            return observation[canonical_key]
        if short_key in observation:
            return observation[short_key]
        if default is not ...:
            return default
        raise KeyError(f"Observation must contain '{canonical_key}' or '{short_key}'.")
    


    def policy_reset(self) -> None:
        self.predict_action_queue = deque()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local SmolVLA point-cloud inference.")
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=DEFAULT_POLICY_REPO_ID)
    parser.add_argument("--processor.path", "--processor_path", dest="processor_path", default=None)
    parser.add_argument("--dataset.repo_id", "--dataset_repo_id", dest="dataset_repo_id", default=None)
    parser.add_argument("--dataset.root", "--dataset_root", dest="dataset_root", default=None)
    parser.add_argument("--obs.path", "--obs_path", dest="obs_path", default=None)
    parser.add_argument("--ply.path", "--ply_path", dest="ply_path", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--task", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--add-gripper-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--gripper-len", type=float, default=0.06)
    parser.add_argument("--gripper-template", choices=("reap", "panda"), default="reap")
    parser.add_argument("--gripper-drop-strategy", choices=("tail", "random", "near_gripper"), default="tail")
    parser.add_argument("--gripper-shuffle-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-width-percent", type=float, default=1.0)
    parser.add_argument("--gripper-qpos-max-width", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-trajectory-ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-postprocess", action="store_true")
    return parser.parse_args()


def sample_or_repeat_points(xyzrgb: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    if xyzrgb.ndim != 2 or xyzrgb.shape[1] != 6:
        raise ValueError(f"Expected point cloud shape (N, 6), got {xyzrgb.shape}")
    if num_points <= 0 or xyzrgb.shape[0] == num_points:
        return xyzrgb
    if xyzrgb.shape[0] == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    rng = np.random.default_rng(seed)
    replace = xyzrgb.shape[0] < num_points
    indices = rng.choice(xyzrgb.shape[0], num_points, replace=replace)
    return xyzrgb[indices].astype(np.float32, copy=False)


def prepare_inference_point_cloud(
    xyzrgb_eff: np.ndarray,
    *,
    num_points: int,
    add_gripper_cloud: bool,
    gripper_width_percent: float,
    gripper_points: int,
    gripper_len: float,
    gripper_template: str,
    gripper_drop_strategy: str,
    gripper_shuffle_points: bool,
    seed: int = 0,
) -> np.ndarray:
    if add_gripper_cloud:
        return add_local_gripper_cloud_to_point_cloud(
            xyzrgb_eff,
            gripper_width_percent,
            total_points=int(num_points),
            gripper_points=int(gripper_points),
            gripper_len=float(gripper_len),
            gripper_template=str(gripper_template),
            seed=int(seed),
            drop_strategy=str(gripper_drop_strategy),
            shuffle_points=bool(gripper_shuffle_points),
        )
    return sample_or_repeat_points(xyzrgb_eff, int(num_points), seed=int(seed))


def read_ply_xyzrgb(path: str | Path, num_points: int) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.size == 0:
        raise ValueError(f"PLY contains no points: {path}")
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape != xyz.shape:
        colors = np.full_like(xyz, 0.5, dtype=np.float32)
    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0
    return sample_or_repeat_points(np.concatenate([xyz, colors], axis=1), num_points)


def identity_pose9_gripper(gripper: float = 0.0) -> np.ndarray:
    state = np.zeros(10, dtype=np.float32)
    state[3] = 1.0
    state[7] = 1.0
    state[-1] = float(gripper)
    return state


def write_trajectory_ply(path: Path, actions: np.ndarray) -> None:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.shape[-1] < 3:
        raise ValueError(f"Expected actions with xyz in first 3 dims, got {actions.shape}")
    points = actions[:, :3]
    lines = np.asarray([[idx, idx + 1] for idx in range(max(0, len(points) - 1))], dtype=np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if len(lines) > 0:
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([[0.0, 0.7, 1.0]]), (len(lines), 1)))
    o3d.io.write_line_set(str(path), line_set)


def save_inference_outputs(
    output_dir: Path | None,
    *,
    action: np.ndarray,
    task: str,
    source: str,
    save_trajectory_ply: bool,
    model_point_cloud: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    action = np.asarray(action, dtype=np.float32)
    np.save(output_dir / "action.npy", action)
    if model_point_cloud is not None:
        model_point_cloud = np.asarray(model_point_cloud, dtype=np.float32)
        np.save(output_dir / "model_point_cloud.npy", model_point_cloud)

    result = {
        "source": source,
        "task": task,
        "action_shape": list(action.shape),
        "created_unix_s": time.time(),
    }
    if model_point_cloud is not None:
        result["model_point_cloud_shape"] = list(model_point_cloud.shape)
        result["model_point_cloud_path"] = "model_point_cloud.npy"
    if metadata:
        result.update(metadata)
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )
    if save_trajectory_ply:
        write_trajectory_ply(output_dir / "trajectory.ply", action)


def main() -> None:
    args = _parse_args()
    infer = SmolVLA_ModelInference(
        policy_path=args.policy_path,
        policy_repo_id=args.policy_repo_id,
        device=args.device,
        processor_path=args.processor_path,
    )

    if args.obs_path is not None:
        with open(args.obs_path, "rb") as f:
            cur_model_observation = pickle.load(f)
        action = infer.single_inference(
            cur_model_observation,
            visualize=args.visualize,
            task=args.task,
            num_points=args.num_points,
            add_gripper_cloud=args.add_gripper_cloud,
            gripper_points=args.gripper_points,
            gripper_len=args.gripper_len,
            gripper_template=args.gripper_template,
            gripper_drop_strategy=args.gripper_drop_strategy,
            gripper_shuffle_points=args.gripper_shuffle_points,
            gripper_qpos_max_width=args.gripper_qpos_max_width,
        )
        print(f"Predicted single action shape: {tuple(action.shape)}")
        print(action)
        save_inference_outputs(
            args.output_dir,
            action=action,
            task=args.task,
            source=str(args.obs_path),
            save_trajectory_ply=args.save_trajectory_ply,
            model_point_cloud=infer.last_model_point_cloud,
            metadata={
                "point_cloud_contract": {
                    "num_points": args.num_points,
                    "add_gripper_cloud": args.add_gripper_cloud,
                    "gripper_points": args.gripper_points,
                    "gripper_template": args.gripper_template,
                    "coordinate_frame": "current_end_effector",
                }
            },
        )
        return

    if args.ply_path is not None:
        point_cloud = read_ply_xyzrgb(args.ply_path, args.num_points)
        point_cloud = prepare_inference_point_cloud(
            point_cloud,
            num_points=args.num_points,
            add_gripper_cloud=args.add_gripper_cloud,
            gripper_width_percent=float(args.gripper_width_percent),
            gripper_points=args.gripper_points,
            gripper_len=args.gripper_len,
            gripper_template=args.gripper_template,
            gripper_drop_strategy=args.gripper_drop_strategy,
            gripper_shuffle_points=args.gripper_shuffle_points,
        )
        action_chunk = infer.predict_action_chunk_obs(
            {"point_cloud": point_cloud, "state": identity_pose9_gripper(float(args.gripper_width_percent))},
            task=args.task,
            postprocess=not args.no_postprocess,
            state_pose_mode="identity",
        )
        action_np = action_chunk[0].detach().cpu().numpy().astype(np.float32)
        print(f"Predicted action chunk shape: {tuple(action_np.shape)}")
        print(action_np)
        save_inference_outputs(
            args.output_dir,
            action=action_np,
            task=args.task,
            source=str(args.ply_path),
            save_trajectory_ply=args.save_trajectory_ply,
            model_point_cloud=point_cloud,
            metadata={
                "point_cloud_contract": {
                    "num_points": args.num_points,
                    "add_gripper_cloud": args.add_gripper_cloud,
                    "gripper_points": args.gripper_points,
                    "gripper_template": args.gripper_template,
                    "coordinate_frame": "current_end_effector",
                    "gripper_width_percent": float(args.gripper_width_percent),
                }
            },
        )
        if args.visualize:
            vis_umi_data(action_np, point_cloud)
        return

    if args.dataset_repo_id is None:
        print("Model loaded. Pass --obs.path or --dataset.repo_id to run inference.")
        return

    result = infer.predict_from_dataset(
        args.index,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        task=args.task or None,
        postprocess=not args.no_postprocess,
    )
    print(f"Predicted first action shape: {tuple(result['action'].shape)}")
    print(result["action"])
    print(f"Predicted action chunk shape: {tuple(result['action_chunk'].shape)}")
    if result["gt_action_chunk"] is not None:
        print(f"Ground-truth action chunk shape: {tuple(result['gt_action_chunk'].shape)}")
    save_inference_outputs(
        args.output_dir,
        action=result["action_chunk"].detach().cpu().numpy(),
        task=args.task,
        source=str(args.dataset_repo_id),
        save_trajectory_ply=args.save_trajectory_ply,
    )


if __name__ == "__main__":
    main()
