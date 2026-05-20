#!/usr/bin/env python

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import default_collate
from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


DEFAULT_POLICY_PATH = (
    "/home/liusong/ProgramFiles/Huggingface/lerobot/"
    "outputs/train/my_smolvla_song1/checkpoints/last/pretrained_model"
)
DEFAULT_POLICY_REPO_ID = "/home/liusong/scp_receive/smolvla"

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

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy.config.vlm_model_name,
            local_files_only=local_files_only,
        )
        self.dataset: LeRobotDataset | None = None
        self.predict_action_queue = deque()
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

    def load_dataset(
        self,
        dataset_repo_id: str | Path,
        *,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
    ) -> LeRobotDataset:
        meta = LeRobotDatasetMetadata(str(dataset_repo_id), root=root)
        delta_timestamps = resolve_delta_timestamps(self.policy.config, meta)
        self.dataset = LeRobotDataset(
            str(dataset_repo_id),
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
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
    ) -> np.ndarray:
        """Run one real-robot style inference step.

        The input format matches the pickle produced by the DP3 realtime path:
        the first seven non-point-cloud entries are interpreted as
        xyz + euler_zyx + gripper, and `point_cloud` is expected to be (N, 6).
        Returns xyz + euler_zyx + gripper in the world frame.
        """
        one_step_agent_pos = self._real_observation_to_pose9_gripper(cur_model_observation)
        point_cloud_world = self._to_numpy(cur_model_observation["point_cloud"]).astype(np.float32)
        one_step_point_cloud = self._world_point_cloud_to_current_eff(
            point_cloud_world,
            one_step_agent_pos,
        )
        if len(self.predict_action_queue)==0:
            action_chunk = self.predict_action_chunk_obs(
                {
                    "point_cloud": one_step_point_cloud,
                    "state": one_step_agent_pos,
                },
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
                vis_umi_data(pred_world_pose9_gripper,point_cloud_world)

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

        return {
            "observation.point_cloud": pc.to(self.device),
            OBS_STATE: state_tensor.to(self.device),
            OBS_LANGUAGE_TOKENS: language["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: language["attention_mask"].to(self.device, dtype=torch.bool),
        }

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

        pose_keys = [key for key in observation.keys() if key != "point_cloud"]
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
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--task", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    return parser.parse_args()


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
        # task="place, red_cube, eff_open, None"
        task="move_towards, broom, eff_open, None"
        visualize=True
        action = infer.single_inference(
            cur_model_observation,
            visualize=visualize,
            task=task,
        )
        print(f"Predicted single action shape: {tuple(action.shape)}")
        print(action)
        # return

    if args.dataset_repo_id is None:
        print("Model loaded. Pass --obs.path or --dataset.repo_id to run inference.")
        # return

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


import sys
sys.argv = [
    "train.py",  # dummy script name
    "--policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song1/checkpoints/last/pretrained_model",
    # "--policy.type=smolvla",
    "--policy.repo_id=/home/liusong/scp_receive/smolvla",
    "--obs.path=/home/liusong/temp/obs_dict_umi_trash.pkl",
    "--dataset.repo_id=/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/temp",
    "--device=cuda",
]


if __name__ == "__main__":
    main()



# model_va = SmolVLA_ModelInference(
#     policy_path="/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song1/checkpoints/last/pretrained_model",
#     policy_repo_id="/home/liusong/scp_receive/smolvla",
#     device="cuda",
# )
# with open("/home/liusong/temp/obs_umi1.pkl", "rb") as f:
#     cur_model_observation = pickle.load(f)
# action = model_va.single_inference(
#     cur_model_observation,
#     visualize=True,
#     task="place, yellow_mug, eff_open, None",
# )
# print(f"Predicted single action shape: {tuple(action.shape)}")
# print(action)




# def random_repeat_sample_points(xyzrgb: np.ndarray, M: int):
#     N = xyzrgb.shape[0]
#     if N == 0:
#         return np.zeros((M, 6))
#     if N >= M:
#         idx = np.random.choice(N, M, replace=False)
#         return xyzrgb[idx]
#     else:
#         extra = np.random.choice(N, M - N, replace=True)
#         return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)   
# batch['task'][0] = 'place, red_cube, eff_open, None\n'
# scene_pcd = o3d.io.read_point_cloud(f"/home/liusong/temp/ood_test_new1.ply",)
# scene_point_cloud = np.concatenate((np.asarray(scene_pcd.points[:]),np.asarray(scene_pcd.colors[:])*255), axis=1)
# scene_point_cloud = random_repeat_sample_points(scene_point_cloud, 1024)
# batch['observation.point_cloud'][0][0] = torch.tensor(scene_point_cloud).to("cuda")
# model_batch = self.preprocessor(batch)
# action_pred = self.policy.predict_action_chunk(model_batch)
# vis_umi_data(action_pred.cpu().numpy()[0],model_batch['observation.point_cloud'].cpu().numpy()[0][0])
# print(action_pred[0][:,-1])