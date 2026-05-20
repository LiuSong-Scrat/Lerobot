from pathlib import Path
import concurrent.futures
import os
import shutil
import threading

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch.nn.functional as F
import open3d as o3d

from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d.ops as torch3d_ops

from tqdm import tqdm


ROOT = Path("/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/temp")
HDF5_FOLDER = Path("/home/liusong/temp/temp")
FPS_BATCH_SIZE = int(os.environ.get("SONG_FPS_BATCH_SIZE", "128"))
USE_CUDA_FPS = os.environ.get("SONG_USE_CUDA_FPS", "1") != "0"
CONVERT_WORKERS = int(os.environ.get("SONG_CONVERT_WORKERS", str(min(20, os.cpu_count() or 1))))
_FPS_CUDA_LOCK = threading.Lock()

def from_H_to_trajectory(H):
    """从齐次矩阵转换为轨迹数据"""
    position = H[:3, 3]
    rotation_matrix = H[:3, :3]
    euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
    trajectory = np.hstack((position, euler_zyx))
    return trajectory

def fast_inverse_homogeneous(T):
    """
    输入: T (..., 4, 4)
    输出: T_inv (..., 4, 4)
    """
    T = np.asarray(T, dtype=np.float32)
    rot = T[..., :3, :3]
    trans = T[..., :3, 3]

    rot_inv = np.swapaxes(rot, -1, -2)
    trans_inv = -(rot_inv @ trans[..., None])[..., 0]

    T_inv = np.zeros_like(T, dtype=np.float32)
    T_inv[..., :3, :3] = rot_inv
    T_inv[..., :3, 3] = trans_inv
    T_inv[..., 3, 3] = 1.0
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


def pose9_to_homo_np(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    t = pose9[..., :3]
    a1 = pose9[..., 3:6]
    a2 = pose9[..., 6:9]

    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-6, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-6, None)
    b3 = np.cross(b1, b2)

    H = np.zeros((*pose9.shape[:-1], 4, 4), dtype=np.float32)
    H[..., 3, 3] = 1.0
    H[..., :3, 0] = b1
    H[..., :3, 1] = b2
    H[..., :3, 2] = b3
    H[..., :3, 3] = t
    return H


def homo_to_pose9(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float32)
    t = H[..., :3, 3]
    rot6d = np.concatenate([H[..., :3, 0], H[..., :3, 1]], axis=-1)
    return np.concatenate([t, rot6d], axis=-1).astype(np.float32, copy=False)


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
    T_world = pose9_to_homo_np(obs_pose9_eff_to_world) # B 4 4
    T_eff0_to_world = T_world[0]
    T_inv_fast = fast_inverse_homogeneous(T_world)
    T_eff0_to_eff = T_inv_fast @ T_eff0_to_world
    T_eff_to_eff0 = fast_inverse_homogeneous(T_eff0_to_eff)
    return homo_to_pose9(T_eff_to_eff0)

def from_world_to_umi_pointcloud(obs_pose9_eff_to_world,pointcloud_world):
    pointcloud_world = np.asarray(pointcloud_world, dtype=np.float32)
    T_world = pose9_to_homo_np(obs_pose9_eff_to_world) # B 4 4
    T_inv_fast = fast_inverse_homogeneous(T_world)
    P_eff_xyz = np.einsum("bij,bnj->bni", T_inv_fast[:, :3, :3], pointcloud_world[..., :3])
    P_eff_xyz += T_inv_fast[:, None, :3, 3]
    return np.concatenate((P_eff_xyz, pointcloud_world[..., 3:]), axis=-1).astype(np.float32, copy=False)

def batched_fps(cloud_rgb_overhead_array, num_points=1024, use_cuda=True, batch_size=FPS_BATCH_SIZE):
    cloud_rgb_overhead_array = np.asarray(cloud_rgb_overhead_array, dtype=np.float32)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    num_frames = cloud_rgb_overhead_array.shape[0]
    sampled_array = np.empty((num_frames, num_points, 6), dtype=np.float32)

    def sample_batches():
        with torch.no_grad():
            for start in range(0, num_frames, batch_size):
                end = min(start + batch_size, num_frames)
                points = torch.as_tensor(cloud_rgb_overhead_array[start:end], device=device, dtype=torch.float32)
                xyz = points[..., :3]   # [B, N, 3]
                rgb = points[..., 3:]   # [B, N, 3]

                K = torch.full((xyz.shape[0],), num_points, dtype=torch.long, device=device)
                sampled_xyz, indices = torch3d_ops.sample_farthest_points(points=xyz, K=K)

                indices_expand = indices.unsqueeze(-1).expand(-1, -1, 3)
                sampled_rgb = torch.gather(rgb, 1, indices_expand)
                sampled_points = torch.cat([sampled_xyz, sampled_rgb], dim=-1)
                sampled_array[start:end] = sampled_points.cpu().numpy()

    # PyTorch3D FPS is usually the GPU bottleneck. Serializing CUDA calls avoids
    # multi-threaded episodes fighting for the same GPU memory while CPU/HDF5 work overlaps.
    if device.type == "cuda":
        with _FPS_CUDA_LOCK:
            sample_batches()
    else:
        sample_batches()

    return sampled_array

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

# 需要根据你的数据调整 shape
dataset_features = {
    "action": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]},
    "observation.point_cloud": {"dtype": "float32", "shape": (1024, 6), "names": ["x", "y", "z", "r", "g", "b"]},
    "observation.state": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]},
}

def recreate_empty_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_episode_buffer(dataset: LeRobotDataset, task: str, actions: np.ndarray, point_clouds: np.ndarray, timestamps: np.ndarray) -> dict:
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)

    if not (len(actions) == len(point_clouds) == len(timestamps)):
        raise ValueError(
            f"Episode length mismatch: actions={len(actions)}, point_clouds={len(point_clouds)}, "
            f"timestamps={len(timestamps)}"
        )

    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["size"] = len(actions)
    episode_buffer["task"] = [task] * len(actions)
    episode_buffer["frame_index"] = np.arange(len(actions), dtype=np.int64)
    episode_buffer["timestamp"] = timestamps
    episode_buffer["action"] = actions
    episode_buffer["observation.state"] = actions
    episode_buffer["observation.point_cloud"] = point_clouds
    return episode_buffer


def convert_hdf5_file(h5_path: Path, fps: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        task_name = f["task_name"][()].decode("utf-8")
        obs_pose_eular_eff_to_world = f["observations/pose_eular"][:].astype(np.float32)
        pointcloud_world = f["observations/cloud_rgb/overhead"][:].astype(np.float32)

        obs_pose9_eff_to_world = traj6_to_pose9(obs_pose_eular_eff_to_world)
        obs_pose9_data_eff_2_eff0 = from_world_to_umi_tra_pose9(obs_pose9_eff_to_world)
        P_eff = from_world_to_umi_pointcloud(obs_pose9_eff_to_world, pointcloud_world)

        # UMI_VIS
        # vis_umi_data(obs_pose9_data_eff_2_eff0, P_eff[0])

        gripper_width = f["observations/eff_angular"][:].astype(np.float32).reshape(-1, 1) * 0.5
        actions = np.concatenate((obs_pose9_data_eff_2_eff0, gripper_width), axis=1).astype(np.float32, copy=False)
        point_clouds = batched_fps(P_eff, use_cuda=USE_CUDA_FPS)

        timestamps_dataset = f.get("timestamp")
        if timestamps_dataset is not None:
            timestamps = timestamps_dataset[()]
        else:
            timestamps = np.arange(len(actions), dtype=np.float32) / fps

    return {
        "task": task_name,
        "actions": actions,
        "point_clouds": point_clouds,
        "timestamps": timestamps,
    }


def save_converted_episode(dataset: LeRobotDataset, episode: dict) -> None:
    episode_buffer = make_episode_buffer(
        dataset,
        episode["task"],
        episode["actions"],
        episode["point_clouds"],
        episode["timestamps"],
    )
    dataset.save_episode(episode_data=episode_buffer)


def convert_and_save_sequential(dataset: LeRobotDataset, h5_paths: list[Path]) -> None:
    for h5_path in tqdm(h5_paths, desc="Processing HDF5 files"):
        episode = convert_hdf5_file(h5_path, dataset.fps)
        save_converted_episode(dataset, episode)


def convert_and_save_parallel(dataset: LeRobotDataset, h5_paths: list[Path], max_workers: int) -> None:
    # LeRobotDataset's parquet/meta writers are stateful, so conversion is parallelized
    # but episode saving remains ordered and single-threaded.
    next_to_save = 0
    ready: dict[int, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_hdf5_file, h5_path, dataset.fps): idx
            for idx, h5_path in enumerate(h5_paths)
        }

        with tqdm(total=len(h5_paths), desc=f"Converting HDF5 files ({max_workers} threads)") as progress:
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                ready[idx] = future.result()
                progress.update(1)

                while next_to_save in ready:
                    save_converted_episode(dataset, ready.pop(next_to_save))
                    next_to_save += 1
                    progress.set_postfix(saved=next_to_save)


def main():

    dataset = LeRobotDataset.create(
        repo_id="local_hdf5_converted",  # 只要本地目录名即可
        fps=30,
        features=dataset_features,
        robot_type="my_robot",
        root=ROOT,
        use_videos=False,
    )

    h5_paths = sorted(HDF5_FOLDER.glob("*.hdf5"))
    if len(h5_paths) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in {HDF5_FOLDER}")

    if CONVERT_WORKERS <= 1 or len(h5_paths) == 1:
        convert_and_save_sequential(dataset, h5_paths)
    else:
        convert_and_save_parallel(dataset, h5_paths, CONVERT_WORKERS)

    dataset.finalize()


if __name__ == "__main__":
    main()
