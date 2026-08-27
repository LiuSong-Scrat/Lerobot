# RH20T 真机数据处理全流程报告

已经还原出当前工作区中实际使用过的处理链。正式管线目前只完整支持 **RH20T cfg7**，主流程为：

```text
RH20T extracted RGB-D/robot logs
        ↓
轨迹筛选、时间同步、Action/State 构造
        ↓
多静止相机逐帧反投影和单视角去噪
        ↓
主视角—多视角最近邻共识，剔除漂浮/错误深度点
        ↓
转换到当前 EEF 坐标系
        ↓
追加 500 点虚拟 gripper
        ↓
LeRobot v3 + point-cloud/world-pose sidecar
        ↓
PointSeg 时序伪标签 cache
        ↓
SmolVLA + PointSeg 训练
```

当前正式数据已经生成，且数据/cache 数量完整：

- 数据集：`88 episodes / 17,763 frames / 5 tasks / 25 FPS`
- 任务范围：task 1–5
- 质量筛选：rating ≥ 8
- 主相机：`037522061512`
- 点云：每帧 `50,000 scene + 500 virtual gripper = 50,500`
- PointSeg cache：17,763 个样本，与数据帧一一对应
- cache 使用 6 个 rank，全部存在 done 标记，没有 `.failed` 或临时残留文件

关键入口：

- [主转换入口 s1_run_rh20t_cfg7.sh](/opt/data/private/liusong/RH20T/rh20t/scripts/s1_run_rh20t_cfg7.sh:4)
- [正式转换器 convert_rh20t_cfg7_to_lerobot.py](/opt/data/private/liusong/RH20T/rh20t/scripts/convert_rh20t_cfg7_to_lerobot.py:1194)
- [多视角审查和去漂移实现](/opt/data/private/liusong/RH20T/rh20t/scripts/export_review_ply_two_cameras.py:35)
- [虚拟夹爪审查实现](/opt/data/private/liusong/RH20T/rh20t/scripts/export_review_ply_reap_every10.py:69)
- [PointSeg cache 入口](/opt/data/private/liusong/RH20T/rh20t/scripts/s2_run_rh20t_pointseg_cache.sh:4)
- [正式 PointSeg cache 实现](/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py:886)
- [训练入口](/opt/data/private/liusong/RH20T/rh20t/scripts/s3_run_rh20t_train.sh:4)

## 1. 当前管线范围

`/opt/data/private/liusong/RH20T/extracted` 下存在 cfg1–cfg7，但当前正式转换器只实现了 cfg7。

原因是不同 cfg 的机器人位姿编码不完全相同：

- cfg1/cfg2：quaternion
- cfg3/cfg4：axis-angle rotation vector
- cfg5：需要 RH20T API 的额外标定修正
- cfg6/cfg7：Euler，但实际角度单位和文档有差异

因此，本报告中的可直接执行流程针对：

```text
/opt/data/private/liusong/RH20T/extracted/RH20T_cfg7
```

现有正式产物：

```text
/opt/data/private/liusong/RH20T/lerobot_dataset/
  rh20t_cfg7_tasks1to5_rating8_top_primary_intersection_20260820

/opt/data/private/liusong/RH20T/lerobot_dataset/
  rh20t_cfg7_tasks1to5_rating8_top_primary_intersection_20260820_cache
```

cfg1/cfg2 不能直接套用 cfg7 的 Action 和标定转换逻辑。

## 2. 原始数据要求

每条轨迹至少需要：

```text
task_xxxx_user_xxxx_scene_xxxx_cfg_0007/
  metadata.json
  robot_command/tcpcommand_timestamp.npy
  transformed/tcp_base.npy
  transformed/tcp.npy
  transformed/gripper.npy

  cam_<serial>/
    color.mp4
    depth.mp4
    timestamps.npy
```

共享标定：

```text
calib/<calib_id>/
  intrinsics.npy
  extrinsics.npy
  tcp.npy
  devices.npy              # 非硬性，但转换器会在存在时复制
```

多视角重建时，每台静止相机都必须同时具有：

```text
color.mp4
depth.mp4
timestamps.npy
```

下列两台是 wrist/in-hand 相机，不参与静止场景融合：

```text
135122075425
135122070361
```

## 3. 轨迹筛选

现有正式数据使用的筛选参数：

```text
task_min               = 1
task_max               = 5
min_rating             = 8
limit                  = 100
rank_by_rating         = true
human scenes           = 排除
primary camera         = 037522061512
```

最终只有 88 条符合条件：

| Task | Episodes | Frames |
|---|---:|---:|
| 1 | 18 | 2,300 |
| 2 | 20 | 4,278 |
| 3 | 17 | 4,454 |
| 4 | 20 | 4,609 |
| 5 | 13 | 2,122 |
| 合计 | 88 | 17,763 |

转换器的轨迹选择逻辑见[这里](/opt/data/private/liusong/RH20T/rh20t/scripts/convert_rh20t_cfg7_to_lerobot.py:752)。

## 4. 时间同步

RGB 相机时间戳是主时间轴。对每个 RGB 帧，分别寻找最近的：

- robot command
- 实际 `tcp_base`
- gripper record

必须同时满足：

```text
|command_ts - rgb_ts| <= 20 ms
|tcp_ts     - rgb_ts| <= 20 ms
|gripper_ts - rgb_ts| <= 20 ms
```

任一流不满足，当前 RGB 帧直接删除。实现见[时间戳对齐函数](/opt/data/private/liusong/RH20T/rh20t/scripts/convert_rh20t_cfg7_to_lerobot.py:271)。

这里有两个不同的同步阈值：

- RGB 与 robot command/TCP/gripper：20 ms
- 多静止相机之间：100 ms

后者允许多相机采集存在更大的硬件时间偏差。某个副相机超过 100 ms 会被跳过；主相机缺失则整帧失败。

实际 episode 0 第一帧使用了 7 台相机，另有 1 台因为时间差超过 100 ms 被跳过。

## 5. Action 和 State 构造

### 5.1 原始位姿

`tcp_base.npy` 中实际 EEF 位姿：

```text
[x, y, z, qw, qx, qy, qz]
```

cfg7 command：

```text
[x_mm, y_mm, z_mm, A, B, C, timestamp_ms]
```

虽然 RH20T 文档把角度写成 degree，但代码通过 command 与实际 TCP 的旋转误差比较，确认当前 cfg7 使用的是 radian，并选择：

```text
R = Rz(C) @ Ry(B) @ Rx(A)
```

即转换器中的 `observed` 模式。

### 5.2 Episode 坐标基准

令首个同步后的实际 TCP 为：

```text
T_base_eef0
```

则：

```text
T_eef0_obs(t)
  = inverse(T_base_eef0) @ T_base_tcp(t)

T_eef0_action(t)
  = inverse(T_base_eef0) @ T_base_command(t)
```

因此：

- `observation.state` 是实际 achieved EEF 位姿，相对于 episode 首帧
- `action` 是 command target，相对于 episode 首帧
- Action 不是 delta action，而是 episode 基准下的绝对 target
- `world_ee_poses` 单独保存机器人 base 坐标系中的绝对 achieved pose

### 5.3 10 维格式

Action 和 State 均为 10 维：

```text
[x, y, z,
 R00, R10, R20,
 R01, R11, R21,
 gripper_width_m]
```

旋转采用旋转矩阵前两列，即 rotation-6D。

### 5.4 Gripper 数值

每条 episode 独立归一化：

```text
opening_fraction
  = clip(raw_gripper_command / episode_max_raw_command, 0, 1)

gripper_width_m
  = opening_fraction * 0.08
```

同一个 `gripper_width_m` 写入：

- `action[-1]`
- `observation.state[-1]`

虚拟夹爪几何也使用相同的 `opening_fraction`。

## 6. RGB-D 解码和坐标变换

RGB 分辨率：

```text
640 × 360
```

Depth MP4 不是普通灰度深度视频，而是：

```text
640 × 720
```

其中：

```text
低 8 bit = depth_frame[0:360]
高 8 bit = depth_frame[360:720]

depth_mm = low | (high << 8)
```

有效深度范围：

```text
100–5000 mm
```

内参原始对应 1280×720，需要缩放到 640×360。

坐标转换链：

```text
p_base =
  T_base_from_calibration
  @ inverse(T_camera_from_calibration)
  @ p_camera

p_current_eef =
  inverse(T_base_eef_t)
  @ p_base
```

因此最终点云不是固定 world/base 坐标系，而是每帧的当前 EEF 坐标系：

```text
当前夹爪 TCP = (0, 0, 0)
```

实现见[坐标变换](/opt/data/private/liusong/RH20T/rh20t/scripts/convert_rh20t_cfg7_to_lerobot.py:505)。

## 7. 多视角重建和漂移点剔除

这里的“多视角重建”不是 TSDF、ICP 或全局 BA，而是逐帧利用标定外参，把多个静止 RGB-D 相机转换到同一个当前 EEF 坐标系，然后做主视角最近邻共识。

### 7.1 每台相机独立去噪

每个静止相机执行：

1. 解码 packed depth。
2. 保留 100–5000 mm。
3. 深度邻域一致性：
   - 8 邻域中至少 3 个近似深度点
   - 容差 `max(15 mm, depth × 1.5%)`
4. 局部深度伪影处理：
   - median kernel = 5
   - 局部容差 `max(20 mm, depth × 2%)`
   - 连通域至少 64 像素
5. Open3D 统计滤波：
   - neighbors = 20
   - std_ratio = 1.0
6. Open3D 半径滤波：
   - radius = 0.012 m
   - 至少 4 个邻居
7. 转换到当前 EEF。
8. 裁剪：
   - `EEF-Z <= 0.40 m`
9. 每台相机采样或重复到 50,000 点。

对应实现位于[read_camera_frame](/opt/data/private/liusong/RH20T/rh20t/scripts/export_review_ply_two_cameras.py:35)。

### 7.2 主视角最近邻共识

正式模式是：

```text
primary_camera_intersection
```

过程：

1. 主相机提供最终点的位置和 RGB。
2. 对每个主相机点，在其他静止相机点云中建立 KD-tree 最近邻。
3. 只保留至少被 1 台其他相机在 2 cm 内观测到的点。
4. 若匹配点少于 50,000，只重复匹配点。
5. 不会为了补足点数重新加入未匹配的漂移点。

实现见[nearest_correspondence_primary_intersection](/opt/data/private/liusong/RH20T/rh20t/scripts/export_review_ply_two_cameras.py:107)。

实际 episode 0 第一帧：

```text
主视角候选点：50,000
多视角匹配点：47,610
最终输出：50,000
最大最近邻距离：约 0.019999 m
```

### 7.3 “融合后再次滤波”的实际状态

审查脚本在多视角共识后还会执行：

```text
statistical neighbors = 30
std_ratio              = 0.8
radius                  = 0.015 m
min_neighbors           = 6
```

如果二次滤波把点全部删除，则回退到未二次滤波的共识点云。见[审查脚本融合后二次滤波](/opt/data/private/liusong/RH20T/rh20t/scripts/export_review_ply_two_cameras.py:381)。

但是，当前正式转换器的 `render_primary_camera_intersection_points()`：

- 执行了每台相机的深度/空间去噪
- 执行了多视角 2 cm 共识
- **没有调用这一步融合后二次统计/半径滤波**

因此，现有正式数据的漂移剔除来源是：

```text
单相机深度/空间滤波
+ 多视角最近邻共识
```

而不是“共识完成后再做一次 Open3D 滤波”。

如果要让正式数据与已审查流程完全一致，应先把二次滤波同步进正式转换器，再重建数据和 cache。

## 8. 虚拟 gripper

### 8.1 几何

使用 LIBERO REAP 四盒模板：

- 两根 finger
- 中间横梁
- handle
- 最大开口 0.06 m
- finger 长度 0.08 m
- 每帧 500 点
- RGB 固定为 `[255, 0, 0]`

输出点云布局严格为：

```text
[0:50000]       场景点
[50000:50500]   红色虚拟 gripper
```

PointSeg 和训练代码依赖“gripper 一定位于点云末尾”这一约定。

### 8.2 已确认的位姿不一致

人工审查接受的 2F85 TCP 相对位姿是：

```text
T_eef_gripper =
[0, 0, 1, -0.005]
[1, 0, 0,  0.000]
[0, 1, 0, -0.090]
[0, 0, 0,  1.000]
```

定义见[LIBERO_REAP_T_EEF_2F85](/opt/data/private/liusong/RH20T/rh20t/scripts/export_review_ply_reap_every10.py:37)。

但正式转换器当前仍使用：

```text
rotation    = 同一轴方向
translation = [0, 0, -0.060]
```

见[正式 reap_gripper_template](/opt/data/private/liusong/RH20T/rh20t/scripts/convert_rh20t_cfg7_to_lerobot.py:454)。

差异：

```text
X：5 mm
Z：30 mm
```

这意味着现有正式数据中的 500 点虚拟夹爪仍是旧位置。文档也明确记录了“审查位姿尚未写回正式转换器”。

若要建立新的 canonical 数据集，正确顺序应是：

1. 用审查 PLY 确认 `[-0.005, 0, -0.090]`。
2. 将该常量写回正式转换器。
3. 重建整个 LeRobot 数据集。
4. 删除或重新生成与旧点云对应的 PointSeg cache。
5. 不可只改 cache，因为旧 Zarr 中的夹爪几何已经不同。

## 9. 推荐的 PLY 审查命令

下面命令会导出主视角共识场景，并使用已接受的 2F85 gripper 位姿：

```bash
cd /opt/data/private/liusong/RH20T/rh20t

env -u LD_LIBRARY_PATH \
/home/liusong/anaconda3/envs/reap/bin/python \
scripts/export_review_ply_two_cameras.py \
  --input-root /opt/data/private/liusong/RH20T/extracted/RH20T_cfg7 \
  --output-root /opt/data/private/liusong/RH20T/output/review_primary_intersection \
  --task 1 \
  --scene-name task_0001_user_0014_scene_0001_cfg_0007 \
  --frame 0 \
  --primary-camera 037522061512 \
  --scene-construction primary_nearest_intersection \
  --scene-points 50000 \
  --points-per-camera 50000 \
  --gripper-points 500 \
  --camera-tolerance-ms 100 \
  --min-camera-support 1 \
  --correspondence-distance-m 0.02 \
  --eef-z-max 0.40
```

场景和虚拟夹爪均已处于当前 EEF 坐标系，可以直接同时加载到 CloudCompare，不需要再做人工配准。

## 10. 正式 LeRobot 转换命令

### 10.1 先 dry-run

该命令只输出选择结果，不创建数据集：

```bash
cd /opt/data/private/liusong/RH20T/rh20t

env -u LD_LIBRARY_PATH \
/home/liusong/anaconda3/envs/reap/bin/python \
scripts/convert_rh20t_cfg7_to_lerobot.py \
  --input-root /opt/data/private/liusong/RH20T/extracted/RH20T_cfg7 \
  --output-root /opt/data/private/liusong/RH20T/lerobot_dataset/_rh20t_dryrun_unused \
  --camera-serial 037522061512 \
  --limit 100 \
  --min-rating 8 \
  --task-min 1 \
  --task-max 5 \
  --rank-by-rating \
  --num-points 50000 \
  --gripper-points 500 \
  --point-seed 1000 \
  --point-cloud-mode primary_camera_intersection \
  --points-per-camera 50000 \
  --intersection-min-support 1 \
  --correspondence-distance-m 0.02 \
  --intersection-sync-ms 100 \
  --workers 8 \
  --dry-run
```

预期选择 88 条轨迹，command rotation mode 为 `observed`。

### 10.2 正式转换

建议写入新目录，不覆盖当前正式数据：

```bash
cd /opt/data/private/liusong/RH20T/rh20t

INPUT_ROOT=/opt/data/private/liusong/RH20T/extracted/RH20T_cfg7 \
OUTPUT_ROOT=/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild \
CAMERA_SERIAL=037522061512 \
LIMIT=100 \
MIN_RATING=8 \
TASK_MIN=1 \
TASK_MAX=5 \
RANK_BY_RATING=1 \
SCENE_POINTS=50000 \
GRIPPER_POINTS=500 \
POINT_SEED=1000 \
POINT_CLOUD_MODE=primary_camera_intersection \
POINTS_PER_CAMERA=50000 \
INTERSECTION_MIN_SUPPORT=1 \
INTERSECTION_SYNC_MS=100 \
CORRESPONDENCE_DISTANCE_M=0.02 \
VIRTUAL_GRIPPER=1 \
WORKERS=8 \
WORKER_THREADS=8 \
OVERWRITE=0 \
RESUME=0 \
RUN_CACHE=0 \
bash scripts/s1_run_rh20t_cfg7.sh
```

脚本随后自动运行：

```text
add_lerobot_v3_metadata.py
absolutize_lerobot_image_paths.py
write_lerobot_stats.py
```

注意：

- `OVERWRITE=1` 会通过 `shutil.rmtree()` 删除精确的输出目录。
- 已有数据断点续跑应使用 `RESUME=1`。
- 在修正虚拟 gripper/融合后二次滤波前，上述命令复现的是当前正式转换器行为。

## 11. 最终 LeRobot 数据格式

```text
<dataset_root>/
  data/chunk-000/file-xxx.parquet
  images/episode_xxxxxx/frame_xxxxxx.png

  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/episode_xxxxxx.json
    rh20t_conversion.json
    command_tcp_comparison.json

  point_clouds/
    episode_xxxxxx.zarr/
      xyz    [T, 50500, 3], float16
      rgb    [T, 50500, 3], uint8

  world_ee_poses/
    episode_xxxxxx.npy

  others/
    raw_tcp/
    raw_actions/
    action_target_ee_poses/
    observation_ee_poses/
    raw_gripper/
    raw_gripper_command/
    calibration_refs/
```

当前实际 Zarr 不是旧文档写的单个 `[T,N,6] float32 data` 数组，而是：

```text
xyz: float16
rgb: uint8
```

episode 0 已抽检：

```text
xyz shape = (87, 50500, 3)
rgb shape = (87, 50500, 3)
最后 500 点 RGB 全部为 [255, 0, 0]
```

## 12. PointSeg cache 生成

现有正式 cache 参数：

```text
version                    = 12
num_samples                = 17,763
cache_mode                 = indices
current_points             = 50,000
future_points              = 16,384
gripper_points             = 500
storage_dtype              = float16
distributed world_size     = 6
trajectory_mode            = sparse_full_episode
trajectory_samples         = 32
trajectory_pose_source     = observation.state
future_offsets             = [1,2,4,8,16,31]
temporal_offsets           = [0,-31,-16,-8,-4,-2,-1,1,2,4,8,16,31]
```

cache 不重复保存完整点云，而是保存：

```text
point_indices
labels
weights
class_scores
role_scores
foreground_score
episode_index
frame_index
dataset_index
```

训练时根据 `point_indices` 从 episode Zarr 中重新取点。

伪标签证据包括：

- `tool_comotion`
- `trajectory_approach`
- `near_contact`

关键尺度：

```text
held_sigma       = 0.025 m
static_sigma     = 0.025 m
motion_gap_eps   = 0.005 m
gripper_sigma    = 0.045 m
trajectory_sigma = 0.13 m
contact_sigma    = 0.10 m
contact_radius   = 0.12 m
```

### 12.1 必须覆盖失效的默认脚本路径

当前 `s2_run_rh20t_pointseg_cache.sh` 的默认路径仍指向：

```text
/home/liusong/ProgramFiles/rlbench/song_real_libero/scripts/...
```

该文件目前不存在。实际代码位于：

```text
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py
```

所以必须显式设置 `LEROBOT_CACHE_SCRIPT`。

### 12.2 推荐 cache 命令

```bash
cd /opt/data/private/liusong/RH20T/rh20t

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
DATASET_ROOT=/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild \
OUTPUT_DIR=/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild_cache \
LEROBOT_CACHE_SCRIPT=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
NPROC_PER_NODE=6 \
DEVICE=cuda \
BATCH_SIZE=24 \
NUM_WORKERS=4 \
CURRENT_POINTS=50000 \
FUTURE_POINTS=16384 \
GRIPPER_POINTS=500 \
FUTURE_OFFSETS=1,2,4,8,16,31 \
VISUALIZE=1 \
OVERWRITE=0 \
bash scripts/s2_run_rh20t_pointseg_cache.sh
```

如果目标 cache 已经存在且确定需要删除重建，才设置：

```bash
OVERWRITE=1
```

cache 生成后还会按 task 输出 PointSeg `operation_prob` 审查 PLY。

## 13. 训练命令

```bash
cd /opt/data/private/liusong/RH20T/rh20t

DATASET_ROOT=/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild \
CACHE_ROOT=/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild_cache \
GPU_IDS=3,4,5 \
BATCH_SIZE=50 \
STEPS=80000 \
SAVE_FREQ=500 \
EVAL_FREQ=2000 \
POINTSEG_ENABLE=true \
bash scripts/s3_run_rh20t_train.sh
```

现有成功训练配置中：

```text
policy.camera_view_fps_target_points = null
```

这是正确的现役行为：cache wrapper 已经用 `point_indices` 构造 50,000 点训练输入，不需要再做一次 policy-side FPS。旧文档中“该参数必须等于 cache current_points”的说明对当前单视角 cache 注入路径已经过时。

## 14. 完整性检查

```bash
DATASET=/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild
CACHE=${DATASET}_cache

jq '{total_episodes,total_frames,total_tasks,fps}' \
  "$DATASET/meta/info.json"

jq '{version,num_samples,current_points,future_points,gripper_points,distributed}' \
  "$CACHE/manifest.json"

find "$CACHE" -type f \( -name '*.failed' -o -name '*.tmp.*' \) -print
find "$CACHE/_dist_sync" -name 'rank_*.done' | sort
```

Zarr 抽检：

```bash
env -u LD_LIBRARY_PATH \
/home/liusong/anaconda3/envs/reap/bin/python - <<'PY'
import zarr
import numpy as np

root = "/opt/data/private/liusong/RH20T/lerobot_dataset/rh20t_cfg7_tasks1to5_rating8_primary_intersection_rebuild"
g = zarr.open(f"{root}/point_clouds/episode_000000.zarr", mode="r")

print("xyz:", g["xyz"].shape, g["xyz"].dtype)
print("rgb:", g["rgb"].shape, g["rgb"].dtype)

rgb = np.asarray(g["rgb"][0])
assert rgb.shape[0] == 50500
assert np.all(rgb[-500:] == np.array([255, 0, 0], dtype=np.uint8))
print("virtual gripper tail: OK")
PY
```

最终必须满足：

1. episode 数与 `meta/episodes`、Parquet、Zarr、`world_ee_poses` 一致。
2. 所有 frame 的 robot stream 对齐误差 ≤20 ms。
3. 每帧点云为 50,500 点。
4. 前 50,000 点是场景，后 500 点是红色虚拟夹爪。
5. 点云在当前 EEF 坐标系。
6. `world_ee_poses` 是 base 坐标系绝对 achieved pose7。
7. Action/State 是 episode 首帧 EEF0 下的 10D pose9。
8. cache `num_samples == total_frames`。
9. cache 无失败、无临时文件，每个 rank 均有 done。
10. 修改点云去噪、虚拟夹爪位姿或点序后，必须整体重建 cache。

## 15. 当前最需要修正的三个问题

1. **虚拟夹爪位姿尚未同步**

   正式数据仍是 `[0,0,-0.06]`，审查接受的是 `[-0.005,0,-0.09]`。新的 canonical 数据集应先修正再转换。

2. **正式转换缺少融合后二次滤波**

   审查脚本有 `30/0.8 + 0.015m/6` 二次滤波，正式转换没有。需要决定是否把审查流程正式化。

3. **cache wrapper 默认脚本路径失效**

   必须显式传入 Huggingface/lerobot 下的正式 cache 脚本路径。

另外，现有数据集的 `meta/rh20t_conversion.json` 中 `output_root` 仍保留了重命名前的 `_fast` 路径；这是 provenance 路径过期，不影响数据内容，但新数据集应避免转换完成后直接改目录名。

本次只完成了代码、manifest、实际 Zarr/cache 产物的只读核查，没有修改或覆盖任何文件。