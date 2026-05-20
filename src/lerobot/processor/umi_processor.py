#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ProcessorStep, ProcessorStepRegistry
    

@ProcessorStepRegistry.register("umi")
class UMIProcessor(ProcessorStep):
    """
    Processor to convert world-coordinate actions, states, and point clouds to egocentric coordinates
    relative to the end-effector pose at the start of each chunk.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, transition):
        """
        Apply UMI transformation to the transition.

        Args:
            transition: EnvTransition containing observation data.

        Returns:
            Transformed transition with egocentric coordinates.
        """
        from .core import TransitionKey
        
        # Create a copy of the transition
        new_transition = transition.copy()
        
        # Get observation from transition
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            return new_transition
            
        # Extract data - observation keys have 'observation.' prefix stripped
        if 'observation.state' in observation:
            state_np = observation['observation.state'].cpu().numpy()  # (batch_size, seq_len, 9)
        else:
            # No state data to process
            return new_transition
        
        # Only process if point cloud exists
        if 'observation.point_cloud' in observation:
            point_cloud = observation['observation.point_cloud'].cpu().numpy()  # (batch_size, seq_len, N, 6)
        else:
            point_cloud = None
            
        # Get action from transition
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            action_np = action.cpu().numpy()  # (batch_size, seq_len, D_action)
        else:
            action_np = None

        # Convert to egocentric coordinates
        agent_pos = action_np
        obs_pose9_eff_to_world = agent_pos
        obs_pose9_data_eff_2_eff0 = self.from_world_to_umi_tra_pose9(obs_pose9_eff_to_world)
        
        # Update observation state
        observation['observation.state'][..., :9] = torch.from_numpy(obs_pose9_data_eff_2_eff0[...,0,:]).unsqueeze(-2).to(observation['observation.state'].device)

        # Update action if it exists (assuming action[..., :9] is the pose part)
        if action_np is not None:
            action_np[..., :9] = obs_pose9_data_eff_2_eff0
            new_transition[TransitionKey.ACTION] = torch.from_numpy(action_np).to(action.device)

        # self.vis_umi_data(obs_pose9_data_eff_2_eff0[0],point_cloud.squeeze(1)[0])

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """UMI processor doesn't change feature shapes or types, just coordinate systems."""
        return features

    def from_world_to_umi_tra_pose9(self, obs_pose9_eff_to_world):
        # obs_pose9_eff_to_world shape: (batch_size, seq_len, 9)
        batch_size, seq_len = obs_pose9_eff_to_world.shape[:2]

        # Convert each pose9 to homogeneous matrix
        T_world_list = []
        for i in range(batch_size):
            for j in range(seq_len):
                pose9 = obs_pose9_eff_to_world[i, j]
                T_world = self.pose9_to_homo(torch.tensor(pose9).unsqueeze(0)).squeeze(0).cpu().numpy()
                T_world_list.append(T_world)

        T_world = np.array(T_world_list).reshape(batch_size, seq_len, 4, 4)  # (batch_size, seq_len, 4, 4)

        # Get the transformation from eff0 to world
        T_eff0_to_world = T_world[:, 0]  # (batch_size, 4, 4) - first timestep

        # Compute inverse transformations
        T_world_inv = self.fast_inverse_homogeneous(T_world)  # (batch_size, seq_len, 4, 4)

        # Compute transformations from eff0 to each eff
        # T_eff0_to_eff = T_world_inv @ T_eff0_to_world for each batch
        T_eff0_to_eff = np.zeros_like(T_world)
        for i in range(batch_size):
            T_eff0_to_world_i = T_eff0_to_world[i]  # (4, 4)
            T_world_inv_i = T_world_inv[i]  # (seq_len, 4, 4)
            T_eff0_to_eff[i] = T_world_inv_i @ T_eff0_to_world_i  # (seq_len, 4, 4)

        # Compute transformations from each eff to eff0
        T_eff_to_eff0 = self.fast_inverse_homogeneous(T_eff0_to_eff)  # (batch_size, seq_len, 4, 4)

        # Convert back to trajectory format
        obs_pose_eular_effi = []
        for i in range(batch_size):
            for j in range(seq_len):
                T_effi_to_eff0 = T_eff_to_eff0[i, j]  # (4, 4)
                trajectory = self.from_H_to_trajectory(T_effi_to_eff0)
                obs_pose_eular_effi.append(trajectory)

        obs_pose_eular_data = np.array(obs_pose_eular_effi).reshape(batch_size, seq_len, 6)
        obs_pose9_data = self.traj6_to_pose9(obs_pose_eular_data)
        return obs_pose9_data

    def create_frame(self, position, rot_matrix, scale=0.03):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale,
            origin=[0, 0, 0]
        )
        frame.rotate(rot_matrix, center=np.zeros(3))
        frame.translate(position)
        return frame
    def vis_umi_data(self,action,pointcloud):
        ##########UMI
        # ================= Pred =================

        geometries =[]
        origin_frame = self.create_frame(np.array([0,0,0]), np.eye(3), scale=0.03)
        geometries.append(origin_frame)
        for per_pred_action in action: ####GT
            per_pred_action = per_pred_action
            pred_xyz = per_pred_action[:3]
            pred_rot6d = per_pred_action[3:9]
            pred_rotmat = self.rot6d_to_matrix(torch.tensor(pred_rot6d)).cpu().numpy()
            frame = self.create_frame(pred_xyz, pred_rotmat, scale=0.03)
            geometries.append(frame)

        # ================= Scene Point Cloud =================
        cloud = pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] / 255)
        geometries.append(pcd)

        o3d.visualization.draw_geometries(geometries)
        ##########UMI

    def from_world_to_umi_pointcloud(self, obs_pose9_eff_to_world, pointcloud_world):
        # pointcloud_world shape: (batch_size, seq_len, N, 6)
        batch_size, seq_len, N = pointcloud_world.shape[:3]

        # Convert pose9 to homogeneous matrices - same as in trajectory function
        T_world_list = []
        for i in range(batch_size):
            for j in range(seq_len):
                pose9 = obs_pose9_eff_to_world[i, j]
                T_world = self.pose9_to_homo(torch.tensor(pose9).unsqueeze(0)).squeeze(0).cpu().numpy()
                T_world_list.append(T_world)

        T_world = np.array(T_world_list).reshape(batch_size, seq_len, 4, 4)  # (batch_size, seq_len, 4, 4)

        # Compute inverse transformations
        T_inv_fast = self.fast_inverse_homogeneous(T_world)  # (batch_size, seq_len, 4, 4)

        # Expand for point cloud: (batch_size, seq_len, 1, 4, 4) -> (batch_size, seq_len, N, 4, 4)
        T_inv_fast_expand = np.repeat(T_inv_fast[:, :, np.newaxis, :, :], N, axis=2)  # (batch_size, seq_len, N, 4, 4)

        # Convert point cloud to homogeneous coordinates
        # pointcloud_world[..., :3] shape: (batch_size, seq_len, N, 3)
        # P_world_H shape: (batch_size, seq_len, N, 4)
        P_world_H = np.concatenate([
            pointcloud_world[..., :3],  # xyz
            np.ones((batch_size, seq_len, N, 1))  # homogeneous coordinate
        ], axis=-1)

        # Apply transformation
        # Reshape for matrix multiplication
        P_world_H_reshaped = P_world_H.reshape(-1, 4, 1)  # (batch_size*seq_len*N, 4, 1)
        T_inv_reshaped = T_inv_fast_expand.reshape(-1, 4, 4)  # (batch_size*seq_len*N, 4, 4)

        P_eff_H = (T_inv_reshaped @ P_world_H_reshaped).squeeze(-1)  # (batch_size*seq_len*N, 4)

        # Extract xyz coordinates and concatenate with rgb
        P_eff_xyz = P_eff_H[..., :3].reshape(batch_size, seq_len, N, 3)  # (batch_size, seq_len, N, 3)
        P_eff_rgb = pointcloud_world[..., 3:]  # (batch_size, seq_len, N, 3)

        P_eff = np.concatenate([P_eff_xyz, P_eff_rgb], axis=-1)  # (batch_size, seq_len, N, 6)

        return P_eff.astype(np.float32)

    def pose9_to_homo(self, pose9: torch.Tensor) -> torch.Tensor:
        t = pose9[..., 0:3]
        R = self.rot6d_to_matrix(pose9[..., 3:9])
        H = torch.zeros(*pose9.shape[:-1], 4, 4, device=pose9.device, dtype=pose9.dtype)
        H[..., 3, 3] = 1.0
        H[..., :3, :3] = R
        H[..., :3, 3] = t
        return H

    def rot6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
        a1 = d6[..., 0:3]
        a2 = d6[..., 3:6]
        b1 = F.normalize(a1, dim=-1, eps=1e-6)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1, eps=1e-6)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # columns

    def fast_inverse_homogeneous(self, T):
        """
        输入: T (..., 4, 4) - 可以是任意前缀维度
        输出: T_inv (..., 4, 4)
        """
        # 获取原始形状
        original_shape = T.shape
        batch_shape = original_shape[:-2]  # 除了最后的4x4之外的所有维度

        # 重塑为 (*batch_shape, 4, 4)
        T_reshaped = T.reshape(-1, 4, 4)

        # 1. 提取旋转部分 R 和 平移部分 t
        R = T_reshaped[:, :3, :3]  # (N, 3, 3)
        t = T_reshaped[:, :3, 3:]  # (N, 3, 1)

        # 2. 计算 R 的转置 (即 R 的逆)
        R_inv = np.transpose(R, (0, 2, 1))  # (N, 3, 3)

        # 3. 计算新的平移部分: -R^T * t
        t_inv = - (R_inv @ t)  # (N, 3, 1)

        # 4. 组装逆矩阵
        T_inv_reshaped = np.tile(np.eye(4), (T_reshaped.shape[0], 1, 1))  # (N, 4, 4)
        T_inv_reshaped[:, :3, :3] = R_inv
        T_inv_reshaped[:, :3, 3:] = t_inv

        # 重塑回原始形状
        T_inv = T_inv_reshaped.reshape(original_shape)

        return T_inv

    def from_H_to_trajectory(self, H):
        """从齐次矩阵转换为轨迹数据"""
        position = H[:3, 3]
        rotation_matrix = H[:3, :3]
        euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
        trajectory = np.hstack((position, euler_zyx))
        return trajectory

    def traj6_to_pose9(self, traj6: np.ndarray) -> np.ndarray:
        if len(traj6.shape) == 1:
            t = traj6[:3].astype(np.float32)
            euler_zyx = traj6[3:].astype(np.float32)
            Rm = R.from_euler('zyx', euler_zyx, degrees=False).as_matrix().astype(np.float32)
            rot6d = np.hstack((Rm[:, 0], Rm[:, 1]))
            return np.concatenate([t, rot6d], axis=0)

        # Handle multi-dimensional case
        original_shape = traj6.shape
        if len(original_shape) == 3:  # (batch_size, seq_len, 6)
            batch_size, seq_len = original_shape[:2]
            t = traj6[..., :3].reshape(-1, 3).astype(np.float32)  # (batch_size*seq_len, 3)
            euler_zyx = traj6[..., 3:].reshape(-1, 3).astype(np.float32)  # (batch_size*seq_len, 3)

            # 批量计算旋转矩阵
            rotations = R.from_euler('zyx', euler_zyx, degrees=False)
            Rm = rotations.as_matrix().astype(np.float32)  # (batch_size*seq_len, 3, 3)
            # 提取每行旋转矩阵的第0列和第1列，并拼接成 (batch_size*seq_len, 6)
            rot6d = np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)  # (batch_size*seq_len, 6)
            # 拼接平移和 rot6d
            pose9_flat = np.concatenate([t, rot6d], axis=1)  # (batch_size*seq_len, 9)
            # 重塑回原始形状
            pose9 = pose9_flat.reshape(batch_size, seq_len, 9)
            return pose9

        # Fallback for other shapes
        t = traj6[:, :3].astype(np.float32)
        euler_zyx = traj6[:, 3:].astype(np.float32)
        # 批量计算旋转矩阵
        rotations = R.from_euler('zyx', euler_zyx, degrees=False)
        Rm = rotations.as_matrix().astype(np.float32)  # (N, 3, 3)
        # 提取每行旋转矩阵的第0列和第1列，并拼接成 (N, 6)
        rot6d = np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)  # (N, 6)
        # 拼接平移和 rot6d
        pose9 = np.concatenate([t, rot6d], axis=1)  # (N, 9)

        return pose9
