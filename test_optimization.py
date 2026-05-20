import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

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
    R = rot6d_to_matrix(pose9[..., 3:9])
    H = torch.zeros(*pose9.shape[:-1], 4, 4, device=pose9.device, dtype=pose9.dtype)
    H[..., 3, 3] = 1.0
    H[..., :3, :3] = R
    H[..., :3, 3] = t
    return H

def traj6_to_pose9(traj6: np.ndarray) -> np.ndarray:
    if len(traj6.shape) < 2:
        t = traj6[:3].astype(np.float32)
        euler_zyx = traj6[3:].astype(np.float32)
        Rm = R.from_euler('zyx', euler_zyx, degrees=False).as_matrix().astype(np.float32)
        rot6d = np.hstack((Rm[:, 0],Rm[:, 1]))
        return np.concatenate([t, rot6d], axis=0)

    t = traj6[:,:3].astype(np.float32)
    euler_zyx = traj6[:,3:].astype(np.float32)
    rotations = R.from_euler('zyx', euler_zyx, degrees=False)
    Rm = rotations.as_matrix().astype(np.float32)
    rot6d = np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)
    pose9 = np.concatenate([t, rot6d], axis=1)
    return pose9

def fast_inverse_homogeneous(T):
    R = T[:, :3, :3]
    t = T[:, :3, 3:]
    R_inv = R.transpose(0, 2, 1)
    t_inv = - (R_inv @ t)
    T_inv = np.eye(4)[None, :, :].repeat(T.shape[0], axis=0)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv

# 优化后的函数
def from_world_to_umi_tra_pose9_optimized(obs_pose9_eff_to_world):
    T_world_list = pose9_to_homo(torch.tensor(obs_pose9_eff_to_world)).cpu().numpy()
    T_world = np.array(T_world_list)
    T_eff0_to_world = T_world[0]
    T_inv_fast = fast_inverse_homogeneous(T_world)
    T_eff0_to_eff = T_inv_fast @ T_eff0_to_world
    T_eff_to_eff0 = fast_inverse_homogeneous(T_eff0_to_eff)

    positions = T_eff_to_eff0[:, :3, 3]
    rotation_matrices = T_eff_to_eff0[:, :3, :3]
    rotations = R.from_matrix(rotation_matrices)
    euler_zyx = rotations.as_euler('zyx', degrees=False)
    obs_pose_eular_data = np.concatenate([positions, euler_zyx], axis=1)
    return traj6_to_pose9(obs_pose_eular_data)

def from_world_to_umi_pointcloud_optimized(obs_pose9_eff_to_world, pointcloud_world):
    T_world_list = pose9_to_homo(torch.tensor(obs_pose9_eff_to_world)).cpu().numpy()
    T_world = np.array(T_world_list)
    T_inv_fast = fast_inverse_homogeneous(T_world)

    B, N, _ = pointcloud_world.shape
    P_world_H = np.zeros((B, N, 4, 1), dtype=np.float32)
    P_world_H[:, :, :3, 0] = pointcloud_world[:, :, :3]
    P_world_H[:, :, 3, 0] = 1.0

    T_inv_fast_expanded = T_inv_fast[:, np.newaxis, :, :]
    P_eff_H = np.matmul(T_inv_fast_expanded, P_world_H)
    P_eff_xyz = P_eff_H[:, :, :3, 0]
    P_eff_rgb = pointcloud_world[:, :, 3:]
    P_eff = np.concatenate([P_eff_xyz, P_eff_rgb], axis=-1)
    return P_eff.astype(np.float32)

def fast_fps(points, num_points=1024):
    B, N, C = points.shape
    if N <= num_points:
        return points

    sampled_points = np.zeros((B, num_points, C), dtype=points.dtype)

    for b in range(B):
        cloud = points[b]
        xyz = cloud[:, :3]
        selected_indices = [0]
        distances = np.full(N, np.inf)
        distances[0] = 0

        for _ in range(num_points - 1):
            last_selected = selected_indices[-1]
            current_distances = np.sum((xyz - xyz[last_selected]) ** 2, axis=1)
            distances = np.minimum(distances, current_distances)
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)
            distances[farthest_idx] = 0

        sampled_points[b] = cloud[selected_indices]

    return sampled_points

# 性能测试
if __name__ == "__main__":
    print("测试优化效果...")

    # 创建测试数据
    batch_size = 50  # 减小batch size以避免内存问题
    obs_pose9_eff_to_world = np.random.randn(batch_size, 9).astype(np.float32)
    pointcloud_world = np.random.randn(batch_size, 2048, 6).astype(np.float32)

    # 测试轨迹转换
    print("测试轨迹转换...")
    start_time = time.time()
    result_traj = from_world_to_umi_tra_pose9_optimized(obs_pose9_eff_to_world)
    traj_time = time.time() - start_time
    print(f'轨迹转换: {batch_size}帧, 耗时: {traj_time:.4f}s, 平均: {traj_time/batch_size*1000:.2f}ms/帧')

    # 测试点云转换
    print("测试点云转换...")
    start_time = time.time()
    result_cloud = from_world_to_umi_pointcloud_optimized(obs_pose9_eff_to_world, pointcloud_world)
    cloud_time = time.time() - start_time
    print(f'点云转换: {batch_size}帧×{pointcloud_world.shape[1]}点, 耗时: {cloud_time:.4f}s, 平均: {cloud_time/batch_size*1000:.2f}ms/帧')

    # 测试FPS采样
    print("测试FPS采样...")
    start_time = time.time()
    result_fps = fast_fps(result_cloud, num_points=1024)
    fps_time = time.time() - start_time
    print(f'FPS采样: {batch_size}帧, 耗时: {fps_time:.4f}s, 平均: {fps_time/batch_size*1000:.2f}ms/帧')

    total_time = traj_time + cloud_time + fps_time
    print(f'总处理时间: {total_time:.4f}s, 平均: {total_time/batch_size*1000:.2f}ms/帧')
    print(f'预期处理速度: {batch_size/total_time:.1f} FPS')