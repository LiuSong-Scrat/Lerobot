import h5py
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
# 确保matplotlib使用Tkinter后端以支持交互

# 四元数转欧拉角 (x, y, z, w) -> (roll, pitch, yaw)
def quaternion_to_euler(q):
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw]) * (180 / np.pi)  # 转换为角度


def visualize_cloud_rgb(cloud_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_rgb[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb[:,3:]/255)
    geometries = []
    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)

# no_precise_episode_22.hdf5
# precise2_episode_12.hdf5
# precise2_episode_14.hdf5
# precise2_episode_17.hdf5
# precise2_episode_18.hdf5
# precise2_episode_19.hdf5
# precise2_episode_20.hdf5
# precise_episode_11.hdf5
# precise_episode_38.hdf5

# 1. 打开HDF5文件
hdf5_path = "benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo/noprecise_episode_0.hdf5"
with h5py.File(hdf5_path, "r") as f:
    # 查看结构（可选）
    print("HDF5文件内容：")
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)
    
    # 读取数据
    images_front = f["observations/images/hand"][...]
    images = f["observations/images/overhead"][...]
    
    poses = f["observations/pose_eular"][...]  # 假设格式为 [x, y, z, qx, qy, qz, qw]
    
    num_frames = len(images)
    if len(poses) != num_frames:
        print(f"警告：图像数量({num_frames})与位姿数量({len(poses)})不匹配")
        num_frames = min(num_frames, len(poses))


    # task_name = f['task_name'][()].decode('utf-8')
    # print(f"任务描述为 {task_name}")
    print(f"共读取 {num_frames} 帧数据。按右箭头键查看下一张，左箭头键查看上一张，按 'q' 退出。")

    # 提取位置数据用于轨迹显示
    positions = np.array(poses[:, :3])  # x, y, z坐标
    # quaternions = poses[:, 3:7]  # 四元数
    
    positions_color = np.zeros(positions.shape)
    positions_color[...,:1] = np.ones(positions.shape)[...,:1]*255
    trajectory_clouds_rgb = np.hstack((positions,positions_color))
    clouds_rgb = f["observations/cloud_rgb/overhead"][:][0]
    clouds_rgb_with_trajectory = np.vstack((clouds_rgb,trajectory_clouds_rgb))
    
    visualize_cloud_rgb(clouds_rgb_with_trajectory)

    # import sys
    # sys.path.append("/home/liusong/ProgramFiles/REAP/")
    # from StageGen.stagegen.visualization_utils import VisualizationUtils
    # VisualizationUtils.vis_4D_cloud_rgb(f["observations/cloud_rgb/overhead"][:])

    print( np.array(f["observations/eff_angular"][:]).max(),np.array(f["observations/eff_angular"][:]).min())

    gripper_width = np.array(f["observations/eff_angular"][:])  # 假设夹爪宽度是位姿中的最后一列
    gripper_width = gripper_width.reshape(gripper_width.shape[0])
    # 转换四元数为欧拉角
    euler_angles =  np.array(f["observations/pose_eular"][...][:,3:])
    
    # 计算位置数据的范围，用于设置坐标轴
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
    
    # 添加一些边距，使轨迹不紧贴边界
    margin_x = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    margin_y = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
    margin_z = (z_max - z_min) * 0.1 if z_max != z_min else 1.0
    
    x_lim = [x_min - margin_x, x_max + margin_x]
    y_lim = [y_min - margin_y, y_max + margin_y]
    z_lim = [z_min - margin_z, z_max + margin_z]
    
    # 创建图形和子图
    fig = plt.figure(figsize=(15, 8))
    ax_img_top = fig.add_subplot(221)  # 左上显示 top 图像
    ax_img_front = fig.add_subplot(222)  # 右上显示 front 图像
    ax_pose = fig.add_subplot(223, projection='3d')  # 下方显示 3D轨迹
    
    # 初始化图像显示
    img_display_top = ax_img_top.imshow(images[0])
    ax_img_top.axis('off')
    title_top = ax_img_top.set_title(f"Top Frame 1/{num_frames}")
    
    img_display_front = ax_img_front.imshow(images_front[0])
    ax_img_front.axis('off')
    title_front = ax_img_front.set_title(f"Front Frame 1/{num_frames}")
    
    # 初始化3D位姿显示
    ax_pose.set_xlabel('X')
    ax_pose.set_ylabel('Y')
    ax_pose.set_zlabel('Z')
    
    # 设置坐标轴范围，确保所有点都能显示
    ax_pose.set_xlim(x_lim)
    ax_pose.set_ylim(y_lim)
    ax_pose.set_zlim(z_lim)
    
    # 绘制完整轨迹
    ax_pose.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'gray', alpha=0.5)
    
    # 当前位置标记
    current_pos, = ax_pose.plot([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 'ro', markersize=8)
    
    # 显示姿态信息的文本
    pose_text = ax_pose.text2D(0.05, 0.95, "", transform=ax_pose.transAxes, 
                              bbox=dict(facecolor='white', alpha=0.8))
    
    # 设置视角
    ax_pose.view_init(elev=30, azim=45)
    
    # 当前帧索引
    current_idx = 0
    
    # 更新函数
    def update_frame(index):
        global current_idx
        current_idx = index
        
        # 更新 top 图像
        img_display_top.set_data(images[index])
        title_top.set_text(f"Top Frame {index+1}/{num_frames}")
        
        # 更新 front 图像
        img_display_front.set_data(images_front[index])
        title_front.set_text(f"Front Frame {index+1}/{num_frames}")
        
        # 更新当前位置
        current_pos.set_data([positions[index, 0]], [positions[index, 1]])
        current_pos.set_3d_properties([positions[index, 2]])
        
        # 更新姿态文本信息
        roll, pitch, yaw = euler_angles[index]
        gripper = gripper_width[index]  # 获取夹爪宽度
        pose_info = (f"Position:\nX: {positions[index, 0]:.2f}\nY: {positions[index, 1]:.2f}\nZ: {positions[index, 2]:.2f}\n\n"
                    f"Orientation (degrees):\nRoll: {roll:.1f}\nPitch: {pitch:.1f}\nYaw: {yaw:.1f}\n\n"
                    f"Gripper Width: {gripper:.2f}")
        pose_text.set_text(pose_info)
        
        return img_display_top, title_top, img_display_front, title_front, current_pos, pose_text
    
    # 键盘事件处理
    def on_key_press(event):
        global current_idx
        
        if event.key == 'right':  # 右箭头键：下一张
            new_idx = current_idx + 1
            if new_idx < num_frames:
                update_frame(new_idx)
                fig.canvas.draw_idle()
        elif event.key == 'left':  # 左箭头键：上一张
            new_idx = current_idx - 1
            if new_idx >= 0:
                update_frame(new_idx)
                fig.canvas.draw_idle()
        elif event.key == 'q':  # 按q退出
            plt.close(fig)
        elif event.key == 'r':  # 按r重置视角
            ax_pose.view_init(elev=30, azim=45)
            fig.canvas.draw_idle()
    
    # 初始显示
    update_frame(0)
    
    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    plt.tight_layout()
    plt.show()
