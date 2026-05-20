from pathlib import Path
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch.nn.functional as F
import open3d as o3d

from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d.ops as torch3d_ops

from tqdm import tqdm
import os

def from_H_to_trajectory(H):
    """从齐次矩阵转换为轨迹数据"""
    position = H[:3, 3]
    rotation_matrix = H[:3, :3]
    euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
    trajectory = np.hstack((position, euler_zyx))
    return trajectory

def fast_inverse_homogeneous(T):
    """
    输入: T (B, 4, 4)
    输出: T_inv (B, 4, 4)
    """
    # 1. 提取旋转部分 R (B, 3, 3) 和 平移部分 t (B, 3, 1)
    R = T[:, :3, :3]
    t = T[:, :3, 3:]  # 保持维度为 (B, 3, 1) 方便计算
    
    # 2. 计算 R 的转置 (即 R 的逆)
    # 使用 transpose 交换最后两个维度
    R_inv = R.transpose(0, 2, 1) 
    
    # 3. 计算新的平移部分: -R^T * t
    # 使用 @ 进行批量矩阵乘法
    t_inv = - (R_inv @ t)
    
    # 4. 组装逆矩阵
    # 创建一个单位矩阵模板 (B, 4, 4)
    T_inv = np.eye(4)[None, :, :].repeat(T.shape[0], axis=0)
    
    # 填入计算好的部分
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    
    return T_inv



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
    # 批量计算旋转矩阵
    rotations = R.from_euler('zyx', euler_zyx, degrees=False)
    Rm = rotations.as_matrix().astype(np.float32)  # (N, 3, 3)
    # 提取每行旋转矩阵的第0列和第1列，并拼接成 (N, 6)
    rot6d = np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)  # (N, 6)
    # 拼接平移和 rot6d
    pose9 = np.concatenate([t, rot6d], axis=1)  # (N, 9)

    return pose9

def from_world_to_umi_tra_pose9(obs_pose9_eff_to_world):
    T_world_list = pose9_to_homo(torch.tensor(obs_pose9_eff_to_world)).cpu().numpy()
    T_world = np.array(T_world_list) #B 4 4
    T_eff0_to_world = T_world[0]
    T_inv_fast = fast_inverse_homogeneous(T_world)
    T_eff0_to_eff = T_inv_fast@T_eff0_to_world
    T_eff_to_eff0 = fast_inverse_homogeneous(T_eff0_to_eff)
    obs_pose_eular_effi = np.array([ from_H_to_trajectory(T_effi_to__eff0) for T_effi_to__eff0 in T_eff_to_eff0])
    obs_pose_eular_data = obs_pose_eular_effi
    obs_pose9_data = traj6_to_pose9(obs_pose_eular_data)
    return obs_pose9_data

def from_world_to_umi_pointcloud(obs_pose9_eff_to_world,pointcloud_world):
    T_world_list = pose9_to_homo(torch.tensor(obs_pose9_eff_to_world)).cpu().numpy()
    T_world = np.array(T_world_list) #B 4 4
    T_inv_fast = fast_inverse_homogeneous(T_world)
    T_inv_fast_expand = T_inv_fast[:, np.newaxis, :, :] # B 1 4 4
    P_world_H = np.tile(np.array([[0],[0],[0],[1]]).astype(np.float32),list(pointcloud_world.shape[:2])+[1,1])
    P_world_H[...,:3,0] = pointcloud_world[...,:3]
    P_eff_H = T_inv_fast_expand@P_world_H
    P_eff = np.concatenate((P_eff_H[...,:3,0],pointcloud_world[...,3:]),-1)
    return P_eff.astype(np.float32)

def batched_fps(cloud_rgb_overhead_array, num_points=1024, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    # ========= 1. 转成 tensor 并 stack =========
    points = torch.as_tensor(cloud_rgb_overhead_array, device=device)
    xyz = points[..., :3]   # [B, N, 3]
    rgb = points[..., 3:]   # [B, N, 3]

    B = xyz.shape[0]
    K = torch.full((B,), num_points, device=device)

    # ========= 2. 批量 FPS =========
    sampled_xyz, indices = torch3d_ops.sample_farthest_points(
        points=xyz,
        K=K
    )  # [B, K, 3]

    # ========= 3. gather RGB =========
    indices_expand = indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_rgb = torch.gather(rgb, 1, indices_expand)

    # ========= 4. 拼接 =========
    sampled_points = torch.cat([sampled_xyz, sampled_rgb], dim=-1)

    return sampled_points.cpu().numpy()

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

root = "/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/temp"

# 你自己的 HDF5 文件夹
hdf5_folder = Path("/home/liusong/temp/temp")

# 需要根据你的数据调整 shape
dataset_features = {
    "action": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]},
    "observation.point_cloud": {"dtype": "float32", "shape": (1024, 6), "names": ["x", "y", "z", "r", "g", "b"]},
    "observation.state": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]},
}

dataset = LeRobotDataset.create(
    repo_id="local_hdf5_converted",  # 只要本地目录名即可
    fps=30,
    features=dataset_features,
    robot_type="my_robot",
    root=root,
    use_videos=False,
)

for episode_index, h5_path in tqdm(enumerate(sorted(hdf5_folder.glob("*.hdf5"))), desc="Processing HDF5 files"):
    with h5py.File(h5_path, "r") as f:
        task_name = f['task_name'][()].decode('utf-8')
        # 假设你的 HDF5 是按时间序列存储
        obs_pose_eular_eff_to_world = f['observations/pose_eular'][:]
        pointcloud_world = f["observations/cloud_rgb/overhead"][:]


        ##########Overhead 2 Umi##########
        obs_pose9_eff_to_world = traj6_to_pose9(obs_pose_eular_eff_to_world)
        obs_pose9_data_eff_2_eff0 = from_world_to_umi_tra_pose9(obs_pose9_eff_to_world)
        P_eff = from_world_to_umi_pointcloud(obs_pose9_eff_to_world,pointcloud_world)

        # # UMI_VIS
        # vis_umi_data(obs_pose9_data_eff_2_eff0,P_eff[0])
        ##########Overhead 2 Umi##########
        

        # dp3 data process
        obs_pose9_data = obs_pose9_data_eff_2_eff0
        normalized_action_data= np.array(obs_pose9_data)
        gripper_width = f['observations/eff_angular'][:].astype(np.float32)*0.5  # 2finger /mm——>middle position /m
        normalized_pose9_with_gripper = np.concatenate((normalized_action_data, gripper_width), axis=1)
        delta_action_list = normalized_pose9_with_gripper
        
        action=np.array(delta_action_list)  #joint_with_gripper radin  
        
        
        # Data Package
        action_arr = action
        agent_pos_arr = action_arr
        cloud_rgb_overhead_list = list(P_eff)
        cloud_rgb_overhead_array = batched_fps(np.array(cloud_rgb_overhead_list))


        
        actions= agent_pos_arr
        point_clouds = cloud_rgb_overhead_array

        agent_pos_arr = action_arr

        
        task = task_name
        timestamps = f.get("timestamp")
        if timestamps is not None:
            timestamps = timestamps[()]
        else:
            timestamps = np.arange(len(actions), dtype=np.float32) / dataset.fps

        assert len(actions) == len(point_clouds)

        for t in range(len(actions)):
            frame = {
                "task": task,
                "action": np.asarray(actions[t], dtype=np.float32),
                "observation.state": np.asarray(agent_pos_arr[t], dtype=np.float32),
                "observation.point_cloud": np.asarray(point_clouds[t], dtype=np.float32),
            }
            dataset.add_frame(frame)

    dataset.save_episode()

dataset.finalize()