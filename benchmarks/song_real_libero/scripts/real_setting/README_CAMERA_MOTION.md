# Ego数据采集--离线数据录制
PIPE=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/real_setting/song_rgbd_pipeline.sh

#动态相机---L515
  DYNAMIC_OUTPUT_DIR=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking5
  #数据采集
  bash "$PIPE" record \
    --camera L515 \
    --output "$DYNAMIC_OUTPUT_DIR" 
  #动态相机一键处理所有 segments
  bash "$PIPE" process-all \
    --mode dynamic \
    --camera L515 \
    --output "$DYNAMIC_OUTPUT_DIR" \
    --fast \
    --progress-every 10 \
    --segment-workers 20
  #单独处理并查看某个动态 segment
  bash "$PIPE" process-segment \
  --mode dynamic \
  --camera L515 \
  --output ${DYNAMIC_OUTPUT_DIR} \
  --segment 0 \
  --fast \
  --progress-every 10 \
  --view
  #查看已经处理好的轨迹
  bash "$PIPE" view \
  --mode dynamic \
  --output ${DYNAMIC_OUTPUT_DIR} \
  --segment 0


#固定相机---L515
#数据采集
STATIC_OUTPUT_DIR=/home/liusong/temp/temp_record
bash "$PIPE" process-all \
  --mode static \
  --camera L515 \
  --output ${STATIC_OUTPUT_DIR} 

#单独处理并查看静态 segment
bash "$PIPE" process-segment \
  --mode static \
  --camera L515 \
  --output ${STATIC_OUTPUT_DIR} \
  --segment 0 \
  --view
#查看已经处理好的轨迹
bash "$PIPE" view \
  --mode static \
  --output ${STATIC_OUTPUT_DIR}\
  --segment 0

# Ego数据处理
  ## 构造真机 HDF5 HDF5FromRawData  Raw--->HDF5
  cd ~/ProgramFiles/Huggingface/lerobot
  DYNAMIC_OUTPUT_DIR=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking5
  HDF5_OUTPUT_DIR=benchmarks/song_real_libero/data/real_setting/hdf5_raw

  ###############################
  #inference可视化  --run-inference or --show-inference
  #切片可视化  --show 
  #真机数据保持在 overhead 相机坐标系，不使用 camera_to_world 外参。
  #如果要一键离线推理再交互切片：
  python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
    --input "$DYNAMIC_OUTPUT_DIR" \
    --output-dir "$HDF5_OUTPUT_DIR" \
    --camera-pose-jsonl "$DYNAMIC_OUTPUT_DIR/orbslam3" \
    --require-camera-pose \
    --camera-pose-max-sync-error-ms 20 \
    --camera-reference-mode episode_first \
    --pose-frame camera \
    --align-to-episode-first \
    --ego-trajectory-filter se3_lowpass \
    --ego-trajectory-filter-cutoff-hz 4.0 \
    --ego-trajectory-filter-order 3 \
    --ego-trajectory-max-angular-speed-deg-s 900 \
    --reproject-rgb-to-episode-first \
    --rgb-reproject-workers 4 \
    --run-inference \
    --show-inference \
    --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
    --fast \
    --force-handedness right \
    --fusion-mode model-depth \
    --hand-depth-knn-backend cuda \
    --hand-depth-rigid-refinement \
    --hand-rigid-temporal-filter \
    --camera-names overhead,hand
    
  #直接用推理后的结果交互切片
  python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
    --input "$DYNAMIC_OUTPUT_DIR" \
    --jsonl "$DYNAMIC_OUTPUT_DIR/handpose_wilor.jsonl" \
    --output-dir "$HDF5_OUTPUT_DIR" \
    --camera-pose-jsonl "$DYNAMIC_OUTPUT_DIR/orbslam3" \
    --require-camera-pose \
    --camera-pose-max-sync-error-ms 20 \
    --camera-reference-mode episode_first \
    --pose-frame camera \
    --align-to-episode-first \
    --ego-trajectory-filter se3_lowpass \
    --ego-trajectory-filter-cutoff-hz 4.0 \
    --ego-trajectory-filter-order 3 \
    --ego-trajectory-max-angular-speed-deg-s 900 \
    --reproject-rgb-to-episode-first \
    --force-handedness right \
    --fusion-mode model-depth \
    --hand-depth-knn-backend auto \
    --hand-depth-rigid-refinement \
    --hand-rigid-temporal-filter \
    --camera-names overhead,hand

  #离线推理后切分点切片
  python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
    --input "$DYNAMIC_OUTPUT_DIR" \
    --jsonl "$DYNAMIC_OUTPUT_DIR/handpose_wilor.jsonl" \
    --output-dir "$HDF5_OUTPUT_DIR" \
    --no-interactive \
    --camera-pose-jsonl "$DYNAMIC_OUTPUT_DIR/orbslam3" \
    --require-camera-pose \
    --camera-pose-max-sync-error-ms 20 \
    --camera-reference-mode episode_first \
    --pose-frame camera \
    --align-to-episode-first \
    --ego-trajectory-filter se3_lowpass \
    --ego-trajectory-filter-cutoff-hz 4.0 \
    --ego-trajectory-filter-order 3 \
    --ego-trajectory-max-angular-speed-deg-s 900 \
    --reproject-rgb-to-episode-first \
    --force-handedness right \
    --fusion-mode model-depth \
    --hand-depth-knn-backend auto \
    --hand-depth-rigid-refinement \
    --hand-rigid-temporal-filter \
    --camera-names overhead,hand \
    --segments "$(cat "$DYNAMIC_OUTPUT_DIR/segments.txt")" \
    --max-points 50000 \
    --rgb-reproject-workers 4 \
    --segment-workers 4
    
只查看已经生成的直接 RGB-D JSONL（不会重新推理，也不会修复旧 JSONL）：
python /home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py\
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --input "$DYNAMIC_OUTPUT_DIR" \
  --jsonl "$DYNAMIC_OUTPUT_DIR/handpose_wilor.jsonl" \
  --reuse-jsonl \
  --show3d

当前 HDF5 构建默认要求 JSONL 版本为
`wilor_mano_mesh_rgbd_rigid_icp_temporal_v7`。如果上面的查看命令提示版本不兼容，必须去掉
`--reuse-jsonl` 重新推理。该版本保持完整 MANO 手的关节和 mesh 几何：RGB-D 先做全局深度
平移，再以可见表面 trimmed projective ICP 校正同一个刚体 SE(3)，最后仅对这一刚体
correction 做短缺口插值和 Savitzky–Golay 平滑；不会把各关节或顶点独立吸附到可见深度面，
原有虚拟二指夹爪拟合逻辑也没有改变。
若需要在 Open3D 中同时查看对齐后的完整 MANO mesh，应在重新推理时添加 `--include-mesh`；
没有该字段时看到的手表面是原始 RGB-D 点云，不是 MANO mesh。

### Ego 末端轨迹的离线 SE(3) 滤波

`--ego-trajectory-filter se3_lowpass` 在相机运动补偿完成后、写入每个 episode 的 HDF5
之前处理末端位姿。平移和符号连续的四元数使用离线零相位 Butterworth 低通，因此不会像
因果滑动平均那样引入动作时延；滤波后重新归一化四元数，并默认精确保留 episode 的首尾
位姿。夹爪开合、原始 `handpose_wilor.jsonl`、RGB-D 点云和 3D 手部关键点均不修改。

WiLoR 偶尔会为同一个平行夹爪选择相差 180 度的手部坐标基。默认开启的
`--ego-trajectory-parallel-jaw-symmetry` 会先在平行夹爪的等价姿态中选择时间连续的表示；
超过 `--ego-trajectory-max-angular-speed-deg-s` 的非物理旋转则视为坐标基重置，并把一个
恒定 SO(3) 修正延续到后续帧。它们处理的是姿态表示跳变，不是把正常快速运动逐帧截断。

对于约 30 Hz 的轨迹，建议从 4 Hz、三阶开始。3 Hz 可进一步压制噪声但可能削弱短促动作，
5 Hz 保留更多细节但也保留更多高频抖动。若要完全恢复旧行为，使用
`--ego-trajectory-filter none`；每段的滤波残差、轨迹长度、时间戳修复数和姿态基修复数都会
写入 HDF5 根属性 `ego_trajectory_filter_metrics_json`。


  ## 预处理---Continuous---HalfReduce---AddGripper---MixedStageGen
    ## Continuous----Check HDF5 Quality
      python benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py  --hdf5_dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo
    ## Continuous----Check HDF5 Quality

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/hdf5_edit_reduce.py

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/add_gripper_cloud_to_hdf5.py

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py



  ## LeRobotDatasetFromRealHDF5  HDF5-->(Add Gripper-->)Current EEF/UMI-->Zarr-->Dataset
    # 输入 HDF5 已经包含 observations/cloud_rgb/<camera>，不会下载 LIBERO 数据、
    # 重播轨迹、渲染深度图或从深度图反投影点云。
    # 如果 HDF5 已经由 add_gripper_cloud_to_hdf5.py 加过末端点云（StageGen Mixed），追加： --input-has-gripper-cloud 
    # 如果无虚拟末端点云，使用 --gripper-points 500 --gripper-max-width 0.08 
    # 当使用变长点云（适配StageGen Mixed）训练，使用 --num-points 0
    export HDF5_USE_FILE_LOCKING=FALSE
    python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
      --input-dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo \
      --output-root benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset \
      --repo-id song_real_pointcloud \
      --fps 15 \
      --num-points 0 \
      --input-has-gripper-cloud \
      --point-cloud-storage zarr \
      --workers 6 \
      --vis-count 2 \
      --overwrite \
      --task  "Place the Red Cube on the Blue Cube"


















# L515 / D435I 采集、位姿检查与移动相机补偿

本文说明真机 RGB-D 数据的坐标约定、L515/D435I 采集调试、静止相机漂移检查，以及移动
overview 相机轨迹写入训练数据的流程。该功能只修改真机采集和数据处理，不修改模型或
LIBERO 仿真。

## 1. 三种名称不能混用

- `overhead` / `hand`：相机角色和数据存储键。例如
  `observations/images/overhead`、`observations/cloud_rgb/overhead`。它们不是坐标系。
- `tracking`：外部 VIO/SLAM 的任意全局坐标。其原点和方向不作为模型 world。
- `world`：World/Ego 模型使用的稳定坐标。它可以定义为每个 episode 的第一帧相机
  （`episode_first`），也可以定义为共享 tracking/base 坐标中的同一个固定参考相机
  （`canonical`）。

外部定位提供：

```text
T_tracking<-camera(t)
```

转换器计算：

```text
T_world<-camera(t)
    = inverse(T_tracking<-camera(0)) @ T_tracking<-camera(t)

p_world(t)     = T_world<-camera(t) @ p_camera(t)
T_world<-ee(t) = T_world<-camera(t) @ T_camera<-ee(t)
```

这是 `episode_first` 模式，因此第一帧 `T_world<-camera(0)` 必须是单位矩阵。tracking 系的
任意全局平移和旋转会被首帧归一化消掉。

如果要求动态头部相机严格复现旧固定相机的 world 坐标，则使用 `canonical`：

```text
T_world<-camera(t)
    = inverse(T_tracking<-canonical_camera) @ T_tracking<-camera(t)
```

这里的两项必须来自同一个持久化 tracking/base 坐标。每次启动都重置原点的 SLAM
odometry 不能单独实现跨 episode 的 canonical 对齐；需要 RTAB-Map 持久地图定位、机器人
base/head FK 加相机外参，或者每次把相机物理复位到 canonical 初始姿态。

## 2. 采集模式与性能

### 2.1 固定相机：兼容旧数据的推荐命令

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera D435I \
  --output /home/liusong/temp/temp_record \
  --storage compressed \
  --space-toggle-recording \
  --record-imu \
  --camera-trajectory-mode static
```

`static` 是默认值。脚本为每一条已保存 RGB-D 帧写入同一个硬编码单位位姿，等价于
“固定 overview 相机本身就是 world”。它不改变旧版点云和末端位姿数值，只把原先隐含的
固定相机假设显式保存下来。

### 2.2 移动相机：不阻塞采集的 RGB-D 轨迹和动态 3D 回放

下面的命令先按相机原始速率保存 RGB-D，关闭相机后再估计整条轨迹，最后弹出动态 Open3D
窗口。所有点云被变换到第一条已保存 RGB-D 帧，窗口中有明显的固定 XYZ 原点坐标轴：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera L515 \
  --output /home/liusong/temp/temp_record \
  --storage compressed \
  --space-toggle-recording \
  --record-imu \
  --camera-trajectory-mode rgbd_odometry \
  --visualize-aligned-point-cloud \
  --aligned-point-cloud-fps 15 \
  --aligned-point-cloud-point-stride 2 \
  --aligned-point-cloud-axis-size-m 0.20
```

`rgbd_odometry` 是场景无人工分割的鲁棒 RGB-D 配准实现，适合检查坐标方向和生成候选轨迹。
它会拒绝无法可靠配准的帧，不会用上一帧位姿静默填充。但是它仍属于视觉诊断/回退方案，
不能自动等同于正式 SLAM/VIO ground truth。人手和物体大幅移动、静态背景很少或深度纹理
不足时，训练数据优先使用 RTAB-Map 等输出的 metric 全 SE(3) 轨迹。




正式外部轨迹模式：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera D435I \
  --output /home/liusong/temp/temp_record/ \
  --storage compressed \
  --num-frames 300 \
  --record-imu \
  --camera-trajectory-mode external \
  --external-camera-pose-jsonl /home/liusong/temp/temp_record/camera_pose.jsonl \
  --camera-trajectory-max-sync-error-ms 20 \
  --visualize-aligned-point-cloud
```

`external` 要求每个已保存 `record_index` 都有一个有效的全 SE(3) 位姿，缺帧、重复索引、
invalid 或时间同步超限都会直接报错。

### 2.3 为什么开启调试后只有约 8 FPS

下面这条是“逐帧重诊断”命令，不是只额外记录一条头部轨迹：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera D435I \
  --output /home/liusong/temp/record_temp \
  --storage compressed \
  --space-toggle-recording \
  --record-imu \
  --debug-visualization \
  --debug-rgbd-odometry \
  --debug-save-every 30
```

D435I 真机实测结果如下；数值会随场景和磁盘略有变化：

| 模式 | 实测采集速度 |
|---|---:|
| 原始 compressed RGB-D | 约 30.0 FPS |
| compressed RGB-D + 原始 IMU | 约 30.0 FPS |
| 同步 2D dashboard | 约 18.8 FPS |
| dashboard + 同步 RGB-D odometry（无窗口） | 约 12.7 FPS |
| 旧版 GUI + 30 ms `waitKey` | 约 8.3 FPS |
| 优化后 GUI、逐帧更新 | 约 14.7 FPS |
| 优化后 `--debug-update-every 2` | 约 20.9 FPS |
| `rgbd_odometry` 采集后处理模式 | 采集约 30.0 FPS |

主要开销是深度着色、点云重建与法线、局部/固定锚点 ICP、2D dashboard 合成和 GUI 等待；
原始 IMU 记录本身不是瓶颈。正式采集应去掉 `--debug-visualization` 和
`--debug-rgbd-odometry`，或使用上一节的采集后处理模式。边采边观察时可添加
`--debug-update-every 2` 或 `3`；这只降低 dashboard/ICP 更新频率，不丢失 RGB-D 数据帧。

`--camera overhead` 会优先尝试 L515、失败后尝试 D435I；`--camera hand` 顺序相反。实际打开的
设备型号、序列号、固件、内参和 IMU profile 都写入 `metadata.json`。

调试面板显示：

- 原始 RGB 和对齐到 RGB 后的深度；
- 有效深度比例以及 p05/p50/p95 深度；
- RGB 边缘与深度不连续边缘的叠加；
- 当前点云的 XZ/XY 投影；
- 主机采集 FPS、相机时间戳 FPS；
- 可选 gyro/accel 大小及其与 RGB 时间戳的差值；
- 动态鲁棒 RGB-D 配准的 world-anchor 状态、fitness、RMSE、累计平移和旋转。

快捷键：

- `Space`：开始/暂停片段；
- `S`：立即保存 dashboard；
- `R`：重置诊断 ICP 的 world 参考帧；
- `Q`/`Esc`：结束。

### 2.4 首帧对齐后的动态 3D 点云内容

只读查看已经采集完成的离线 RGB-D 和 `camera_pose.jsonl`：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --output /path/to/existing_sequence \
  --playback-only \
  --aligned-point-cloud-fps 15 \
  --aligned-point-cloud-point-stride 2 \
  --aligned-point-cloud-axis-size-m 0.20
```

该模式不打开相机，不重新采集，也不改写 `frames.jsonl`、`metadata.json` 或
`camera_pose.jsonl`。默认读取 `<output>/camera_pose.jsonl`。轨迹在其他位置时添加：

```bash
  --aligned-point-cloud-pose-jsonl /path/to/camera_pose.jsonl
```

无论 tracking 原点在哪里，`episode_first` 回放都会计算
`inverse(T_tracking<-camera(first_saved)) @ T_tracking<-camera(t)`，因此固定的大型 XYZ
坐标轴和黄色球始终表示第一条已保存 RGB-D 帧的相机原点。

Open3D 窗口同时显示：

- 当前帧对齐后的彩色点云；
- 灰色第一帧参考点云；
- 固定 world 原点 XYZ 坐标轴和黄色原点球；
- 当前相机的小型 XYZ 坐标轴；
- 橙色相机运动轨迹。

按 `Space` 暂停/继续，`R` 从头播放，`Q` 关闭；增加
`--aligned-point-cloud-loop` 可循环播放。自动测试时可用
`--no-aligned-point-cloud-hold-final` 在最后一帧后关闭窗口。用于审计的逐帧变换会写入
`debug/aligned_point_cloud_poses.jsonl`，配置摘要写入
`debug/aligned_point_cloud_playback.json`。

## 3. 静止相机位姿漂移检查

先固定相机；场景可以包含局部人手或物体运动，但必须保留足够的静态背景。执行：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera L515 \
  --output /home/liusong/temp/record_temp \
  --storage compressed \
  --num-frames 120 \
  --warmup-frames 30 \
  --stationary-pose-check \
  --record-imu \
  --require-imu \
  --stationary-max-translation-drift-m 0.01 \
  --stationary-max-rotation-drift-deg 1.0
```

D435I：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera D435I \
  --output /path/to/d435i_stationary_check \
  --storage compressed \
  --num-frames 120 \
  --warmup-frames 30 \
  --stationary-pose-check \
  --record-imu
```

无图形界面的服务器加 `--debug-headless`。若没有指定 `--num-frames` 或 `--duration-s`，
静止检查默认采集 `--stationary-min-frames` 帧。

输出：

```text
debug/
├── rgbd_odometry.jsonl
└── stationary_pose_report.json
```

`rgbd_odometry.jsonl` 逐帧保存诊断用 `T_model_world<-current_camera` 和配准指标。
`stationary_pose_report.json` 给出最大/p95/最终平移漂移、旋转漂移、固定 world 锚定接受率及
PASS/FAIL。锚定接受率过低也会失败，避免“所有帧都拒绝、位姿一直是单位矩阵”造成假通过。

诊断估计器不是普通逐帧累计 ICP。它使用：

1. 局部鲁棒 ICP 提供短时运动预测；
2. episode 第一帧作为固定 model-world 锚；
3. 多帧空间一致性自动保留静态背景，不需要人手或物体分割；
4. 对稳定背景使用 RGB + 几何配准约束平面内运动；
5. Tukey 核抑制移动人手、被操作物体和深度离群点；
6. 前 5 帧保持 world 原点并建立背景置信度，因此开始采集时相机需短暂静止。

最新版 L515 和 D435I 各 40 帧真机静止测试均通过：L515 最大漂移约 2.53 mm / 0.141°，
D435I 约 3.58 mm / 0.212°。在 L515 实测背景中额外注入约 28.6% 独立移动前景后，最大漂移约
2.87 mm / 0.225°。

该结果用于检查 RGB-D、时间同步和 SLAM 输出，不会自动冒充正式训练相机真值。

## 4. IMU 的边界

L515/D435I 配置若提供 gyro/accel，脚本会保存异步原始流到 `imu.jsonl`，并把 RGB 时间戳
最近的样本附到 `frames.jsonl`。脚本还保存 motion intrinsics、motion-to-color 外参和实际
采样率。

原始 gyro/accel 不是 6-DoF 相机位姿。简单二次积分会产生严重平移漂移，因此：

- IMU 只用于同步、设备状态检查和外部 VIO/SLAM 输入；
- dashboard 中的 gyro 积分方向只用于调试；
- 训练数据的相机轨迹必须来自同步的 metric VIO/SLAM 或外部 6-DoF tracking。

若当前设备必须有 IMU，使用 `--record-imu --require-imu`；设备不提供完整 gyro/accel profile
时脚本会立即报错。

L515 与 D435I 都已实测能同时打开 RGB、深度、gyro 和 accel。二者没有 T265 式硬件 pose
stream，因此正式相机位姿应由 RTAB-Map RGB-D-Inertial、OpenVINS、VINS-Fusion 等外部系统
融合得到。本机已经安装 ROS2 Humble、`realsense2_camera` 和 `rtabmap_ros`。

SLAM/VIO 的输出必须转换成下一节的 `camera_to_tracking` JSONL，再由现有严格同步检查接入。

## 5. 外部相机轨迹格式

推荐 JSONL 每行：

```json
{
  "record_index": 17,
  "timestamp_ms": 123456.789,
  "camera_to_tracking": [
    [1.0, 0.0, 0.0, 0.012],
    [0.0, 1.0, 0.0, -0.004],
    [0.0, 0.0, 1.0, 0.031],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "valid": true,
  "tracking_source": "your_vio_name"
}
```

也支持 `T_tracking_camera`、`transform_matrix`、`translation_m + quaternion_xyzw`。旧外部文件
中的 `camera_to_world`/`T_world_camera` 仅作为输入兼容别名读取，读取后仍解释为 tracking
坐标，不能把它误认为模型 world。

轨迹必须显式提供 `record_index`。若同时有时间戳，默认要求 RGB-D 与 pose 的绝对同步误差
不超过 20 ms。




## 6. 构造真机 HDF5

```bash
python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
  --input /path/to/raw_sequence \
  --jsonl /path/to/raw_sequence/handpose_wilor.jsonl \
  --require-camera-pose \
  --camera-pose-max-sync-error-ms 20 \
  --camera-reference-mode episode_first \
  --pose-frame camera \
  --segments 0:120,150:280 \
  --no-interactive \
  --output-dir /path/to/hdf5_raw
```

采集脚本已经把 `camera_pose.jsonl` 中的位姿嵌入 `frames.jsonl`，因此通常不必再传
`--camera-pose-jsonl`。如果位姿由采集结束后的独立流程生成，才显式传该参数。

原始 HDF5 保持点云和末端位姿位于各自的当前相机系，并额外保存：

```text
observations/camera_tracking_pose/overhead   (T, 4, 4)
  transform_direction = camera_to_tracking
  notation            = T_tracking<-camera
  translation_unit    = meter
  pose_format         = matrix
```

这里的 `overhead` 只是相机数据键。片段中任意一帧相机位姿缺失、无效或同步超限都会报错，
不会把部分移动相机轨迹静默当成固定相机。

跨 episode 对齐到旧固定相机时，改用：

```bash
  --camera-reference-mode canonical \
  --canonical-camera-to-tracking-matrix \
    r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz 0 0 0 1
```

矩阵是米制 `T_tracking<-canonical_camera`。HDF5 会保存该矩阵，后续转换默认自动读取。

## 7. 转成 LeRobot dataset

```bash
python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  --input-dir /path/to/hdf5_raw \
  --output-root /path/to/real_lerobot_dataset \
  --repo-id real_lerobot_dataset \
  --camera overhead \
  --camera-motion-compensation required \
  --camera-reference-mode auto \
  --camera-motion-debug \
  --camera-motion-debug-frames 5 \
  --camera-motion-debug-episodes 2 \
  --vis-count 2 \
  --point-cloud-storage zarr \
  --workers 4
```

`--camera-motion-compensation`：

- `required`：移动相机数据推荐；没有完整相机轨迹立即报错；
- `auto`：有轨迹则补偿，没有轨迹则按固定 overview 相机处理；
- `off`：显式忽略相机轨迹。

`--camera-reference-mode`：

- `auto`：读取 HDF5；没有声明时使用 `episode_first`；
- `episode_first`：每个 episode 的首帧各自为 world；
- `canonical`：所有 episode 对齐到同一个持久化参考相机。若 HDF5 未存 canonical 矩阵，
  转换命令必须额外提供 `--canonical-camera-to-tracking-matrix`。

主要输出：

- `world_ee_poses/episode_xxxxxx.npy`：`T_world<-ee(t)` 的 pose9；
- `camera_motion/episode_xxxxxx.npy`：`T_world<-camera(t)`；
- `point_clouds`：延续现有流程，最终位于 `current_eff`；
- RGB：保留 overview 相机原始图像，不做几何重投影。

可视化位于：

```text
<dataset>/visualizations/episode_xxxxxx/camera_motion/
```

关键检查：

1. `episode_first` 的第一帧 `T_world<-camera(0)` 为单位矩阵；`canonical` 的第一帧则应等于
   canonical 相机到本 episode 首帧相机的真实相对位姿；
2. 静态背景在 `overlay_aligned_to_world.ply` 中比 raw overlay 更锐利；
3. 头跟手运动时，world 中的 EEF 轨迹仍恢复为连续运动；
4. 相机轨迹没有不符合采样率的瞬时跳变；
5. aligned overlay 更模糊时，检查矩阵方向、时间同步、单位和 VIO 漂移。

## 8. 部署时如何定义 world

训练数据使用 `episode_first` 时，在线部署也在每次 rollout 开始时保存
`T_tracking<-camera(0)`，并持续计算：

```text
T_world<-camera(t) = inverse(T_tracking<-camera(0)) @ T_tracking<-camera(t)
```

这能消除同一 tracking 轨迹中的头部运动，但不同 episode 的第一帧位置和朝向仍会形成
SE(3) 数据增强。当前点云编码器和 action expert 并没有数学上保证全局 SE(3) 不变/等变，
尤其 XYZ 既参与几何邻域又参与特征时，不能假设模型会自动完全忽略首帧坐标差异。
world-flow 的 SE(3) 共轭变换能保证相对变换变量的坐标换算正确，但不能单独保证整个编码器
和动作生成器跨坐标系不变。

因此：

- 头部初始姿态变化较小、训练覆盖足够时，可以使用 `episode_first`；
- 要严格复用旧固定相机模型或减少跨 episode 坐标分布变化，使用 `canonical`；
- 训练、验证、真机部署必须采用同一种 reference mode；
- 点云、末端 pose9、动作及 world-flow 变量必须由同一个
  `T_world<-camera(t)` 同步转换，不能只移动点云。

## 9. 输出文件

```text
raw_sequence/
├── metadata.json
├── frames.jsonl
├── camera_pose.jsonl
├── imu.jsonl                         # 使用 --record-imu 时
├── segments.json / segments.txt     # 使用 Space 分段时
├── color_jpg/ 或 color/ 或 color.mp4
├── depth_png/ 或 depth_m/
└── debug/
    ├── aligned_point_cloud_poses.jsonl
    ├── aligned_point_cloud_playback.json
    ├── rgbd_odometry.jsonl
    └── stationary_pose_report.json
```

`metadata.json/capture_stats` 同时记录主机保存 FPS 和相机时间戳 FPS，便于区分相机掉帧与
同步调试造成的循环变慢。

## 10. 与模型和仿真的关系

- 不修改 SmolVLA、PointSeg、World/Ego 分支或 action expert；
- 不修改 LIBERO 数据生成和评测；
- `overhead` 和 `hand` 继续用于区分相机通道；
- 模型/数据坐标只称 `world`、`ego/current_eff`；
- 外部定位全局坐标只称 `tracking`。
