# WEP-VLA：基于软运动先验、点云与 SmolVLA 的真实机器人 / LIBERO 策略

> 项目主文档与快速回顾手册
>
> 文档基线：`wep_vla_v0.5.0_double_flow`，2026-08-02
> 如果后续代码行为与本文冲突，以当前代码和 checkpoint 内的 `config.json` 为准，并同步更新本节基线。

> [!IMPORTANT]
> **动作时序与随机流是 checkpoint 语义，不能只在评测命令中临时补偿。**
>
> - 修复后的 LIBERO post-action 数据必须训练为 `action_chunk_start_offset=1`；这表示预测
>   token 0 已对应 dataset `action[i+1]`，正式在线推理仍执行 `action_index=0`。禁止再跳一次到
>   `action_index=1`。
> - `pose9_action_noise_enable=true` 但三个 noise scale 全为 0，只是“单位位姿起点的确定性
>   flow”，不是随机扩散；多步 Euler 积分仍存在。
> - 随机双流必须设置 `worldflow_noise_coupling=conjugate_ego`，使初始噪声满足
>   `G0 = C B0 C^-1`。`independent` 仅用于复现旧实验。
> - 数据、cache、checkpoint、offset、noise prior 和评测 action index 必须作为一套不可拆分的
>   实验契约记录。

## 摘要

WEP-VLA 是建立在 LeRobot SmolVLA 上的视觉—语言—点云动作策略。模型使用冻结或可训练的
SmolVLM 视觉语言前缀、由运动轨迹自动构造的 PointSeg 软监督、LitePT 几何编码器，以及
PointACT 风格的点—动作 token 交互，最终由 Action Expert 通过 flow matching 生成一段
EEF 相对动作轨迹。

项目同时支持真实机器人 HDF5 与 LIBERO 官方 demonstration。两个数据域共享以下核心契约：

- 固定 overview 相机直接作为观测参考系；头戴相机使用同步 6-DoF 轨迹，可归一化到 episode
  第一帧相机系，也可对齐到共享 tracking/base 中的 canonical 固定相机。
- 点云在加入虚拟夹爪后转换到当前 EEF 坐标系，再作为模型输入。
- 动作采用 `xyz + rotation-6D + gripper` 的 10 维表示。
- PointSeg 的监督来自机械臂轨迹先验，不使用人工分割、人工光流、手工目标点或 PCA。
- 面向完整四套件的默认训练仍关闭 WorldFlow/SE(3)；但完整 SE(3)+WorldFlow 已有
  LIBERO Spatial task 5 官方 10/10 的受控单任务 checkpoint，见 8.8。

本文既是实现说明，也是后续开发时快速恢复上下文的唯一主入口。历史实验审计可继续参考
[LIBERO_EVALUATION_AUDIT.md](LIBERO_EVALUATION_AUDIT.md)，VLM adapter 的旧版专项说明见
[README_VLM_ADAPTER.md](README_VLM_ADAPTER.md)，但涉及当前行为时以本文为准。

---

## 1. 项目快速回顾

### 1.1 当前稳定配置

| 模块 | 当前建议 | 说明 |
|---|---:|---|
| SmolVLM RGB 前缀 | 开启 | adapter 模式加载官方预训练 VLM |
| 冻结 VLM | 开启 | 保留官方图像—语言表征，训练点云与动作相关模块 |
| PointSeg | 开启 | 使用 cache v7 软运动先验监督 |
| LitePT | 开启 | XYZ 作为 coord，同时 XYZRGB 作为 feat |
| Point–Action fusion | 开启 | 最终 LitePT token 与 action token 联合双向 self-attention |
| WorldFlow | 实验性开启 | 双流版为独立点云前端 + 共享 Action Expert，训练与推理均参与；已有单任务闭环证据，完整四套件仍需消融 |
| 完整 SE(3) flow | 默认关闭 | `pose9_endpoint` warm-start 已通过 task5 10/10；直接随机 twist 头仍不建议小数据使用 |
| PCA / canonical frame | 不使用 | 已从当前方案中排除 |
| 人工分割 / 人工点流 | 不使用 | 所有前景监督由轨迹自动产生 |

对应核心参数：

```bash
--policy.vla_adapter_enable=true
--policy.vla_adapter_freeze_vlm=true
--policy.pointseg_enable=true
--policy.point_action_fusion_enable=true
--policy.worldflow_enable=false
--policy.worldflow_se3_head_enable=false
--policy.se3_enable=false
--policy.se3_final_correction_enable=false
```

其中 `worldflow_se3_head_enable` 与 `se3_final_correction_enable` 仅为旧命令兼容参数，
当前配置类会提示它们被忽略。

当前模型的关键结构默认值：

| 配置 | 值 |
|---|---:|
| observation history | 1 |
| action chunk / 默认执行长度 | 32 / 16 |
| state / action 最大维度 | 10 / 10 |
| flow denoising steps | 10 |
| VLM 保留层数 | 16 |
| Action Expert 宽度倍率 | 0.75 |
| VLM/Expert attention mode | `cross_attn` |
| self-attention 插入间隔 | 每 2 层 |
| PointSeg / LitePT feature dim | 64 |
| LitePT `n_tokens` 构造值 | 256（当前不强制截断） |
| World/Ego shared-expert scene tokens | 每分支 1 个 `global_feat` |
| Point–Action attention heads | 4 |
| robot state prefix | 默认关闭 |

### 1.2 三条不能破坏的语义契约

1. **坐标契约**：存入 LeRobot dataset 的 `observation.point_cloud` 已经位于当前 EEF
   坐标系；训练和推理必须保持一致。
2. **LIBERO 标签契约**：`observation.state` 是执行后的实际 EEF 位姿，`action` 是原始
   delta OSC action 经控制器重建出的绝对命令目标。两者不应被强制设为相同。
3. **cache 绑定契约**：PointSeg cache 通过点索引引用 dataset 中的点。只要点云内容、
   点顺序、坐标系、episode/frame 映射或采样逻辑变化，就必须重新生成 cache。

### 1.3 最常用入口

| 任务 | 权威脚本 |
|---|---|
| LIBERO HDF5 → LeRobot | `scripts/libero_setting/libero_hdf5_to_dataset.py` |
| 真实 HDF5 → LeRobot | `scripts/real_setting/real_hdf5_to_dataset.py` |
| 离线生成 PointSeg cache | `scripts/song_cache_pointseg_samples.py` |
| 项目标准训练 | `scripts/train_song_benchmark.py` |
| 只读离线 loss 评测 | `../../src/lerobot/scripts/eval_song.py` |
| LIBERO rollout 评测 | `scripts/libero_setting/libero_pointcloud_eval.py` |
| dataset / pickle / PLY 推理 | `scripts/smolvla_model_inference.py` |
| 核心模型 | `../../src/lerobot/policies/smolvla/modeling_smolvla.py` |
| PointSeg 先验 | `../../src/lerobot/policies/smolvla/song_pointseg.py` |
| UMI 坐标处理 | `../../src/lerobot/processor/umi_processor.py` |

`modeling_smolvla copy.py`、`modeling_smolvla_source.py` 和
`libero_pointcloud_eval_src.py` 不是当前运行入口，不能据此判断现有模型行为。

---

## 2. 研究目标与设计边界

### 2.1 目标

模型需要从 overview RGB、场景点云、语言任务和当前机器人状态中生成可执行动作，同时尽量：

- 聚焦被操作物体、工具、目标接触区域等任务相关几何；
- 降低机械臂外观、背景变化和异构末端对策略的干扰；
- 使用物体间相对几何，而不是只记住固定轨迹；
- 在真实数据和 LIBERO 中使用同一套模型接口；
- 不依赖真实相机外参和人工对象标签。

### 2.2 当前没有声称解决的问题

- 当前 LitePT 把绝对 XYZ 同时输入 coord 和 feat，仍可能对坐标平移、相机重定位和离群点敏感。
- 固定 overview 相机仍可直接使用原路径；头戴相机必须为每个 RGB-D 帧提供同步的 full-SE(3)
  `T_tracking_camera` 轨迹，并在真实数据转换时对齐到 episode 第一帧或共享 canonical
  相机定义的模型 `world`。`overhead` 只作为相机/数据键名称。L515/D435I 原始 IMU 只有加速度和
  角速度，不能直接替代 6-DoF VIO/SLAM 位姿。
- 运动先验是软前景证据，不等价于实例分割或精确场景流。
- 低 flow-matching loss 不保证接触任务成功；闭环误差、控制器语义、数据对齐和动作执行频率
  都可能成为主要瓶颈。

---

## 3. 总体系统

```text
fixed overview RGB ───────────────┐
language task ────────────────────┼─> SmolVLM prefix
                                 │       │
EEF-frame XYZRGB point cloud ─> PointSeg│
                │                 │       │
                ├─ foreground ─> LitePT ─┼─> point prefix tokens
                └─ background ─> pooling │
                                         │
noisy action chunk + time ─> action tokens
                    │                    │
                    └─ final LitePT tokens
                         │
                  Point–Action self-attention
                         │
                  Action Expert + VLM
                         │
                  predicted flow velocity
                         │
                 Euler denoising, 10 steps
                         │
              EEF-relative pose9 + gripper
```

模型存在两条互补的点云信息路径：

- **全局前缀路径**：前景和背景点云各压缩为一个 token，与 RGB、语言共同构成 VLM prefix。
- **细粒度动作路径**：最终 LitePT point tokens 与整个 action chunk token 联合 self-attention，
  让动作在进入 Action Expert 前直接读取局部几何。

第二条路径受 PointACT 思想启发，但它是单层、末端融合模块，不是 PointACT 全部多尺度架构的复现。

---

## 4. 数据与坐标表示

### 4.1 模型输入

#### 点云

```text
observation.point_cloud: (B, N, 6)
channels: [x, y, z, r, g, b]
XYZ: metre
RGB: 0 ... 255
frame: current EEF
```

`observation.point_cloud_is_pad` 为 `True` 的位置是 padding。原始 episode 可以具有不同点数，
DataLoader 会补齐；PointSeg 的候选点选择会进一步构造固定数量的前景和背景点。

#### 图像

dataset 中通常保存 256×256 的 `agentview` 或 `overhead` RGB。策略的官方 SmolVLM processor
会将其 resize/pad 到 512×512，并转换到模型所需数值范围。因此“采集 256×256，VLM 输入
512×512”是有意设计，不是训练—推理尺寸错误。

#### 语言

每帧关联一个 task string。即使真实训练只有单任务，语言 token 仍进入 prefix；单任务条件下
语言分支信息量有限，但不会使点云分支失效。

#### 状态和动作

```text
[x, y, z, r6_0, r6_1, r6_2, r6_3, r6_4, r6_5, gripper]
```

默认 `max_state_dim=max_action_dim=10`，action chunk 长度为 32，部署时通常执行前 16 个或由
评测参数指定的若干动作。

### 4.2 rotation-6D

6D 旋转不是四元数，也不要求每个数位于 `[-1, 1]`。模型输出两条三维向量
\(\mathbf{a}_1,\mathbf{a}_2\)，解码时执行：

\[
\mathbf{b}_1 = \operatorname{normalize}(\mathbf{a}_1)
\]

\[
\mathbf{b}_2 = \operatorname{normalize}
\left(\mathbf{a}_2-(\mathbf{b}_1^\top\mathbf{a}_2)\mathbf{b}_1\right)
\]

\[
\mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2,\qquad
R=[\mathbf{b}_1,\mathbf{b}_2,\mathbf{b}_3].
\]

所以类似 `[0.008, -1.002, -0.015, 1.000, 0.011, 0.023]` 的值可以解码为合法旋转。
单位旋转的 6D 表示为 `[1, 0, 0, 0, 1, 0]`。

### 4.3 UMI 当前末端坐标系

训练处理器以当前真实观测 `observation.state` 为原点，将动作转换到当前 EEF 坐标系。不能再用
`action[0]` 充当观测原点，否则会人为抹掉第一条命令目标，使首个动作错误地接近单位位姿。

当前 UMI 语义是：

- 点云：转换脚本或在线推理包装器已变换到当前 EEF；
- 状态：当前 EEF 在自身坐标系中接近单位位姿；
- 动作：相对于当前实际 EEF 的未来命令目标；
- 推理输出：同样的 EEF-relative 目标，再由执行器还原为控制器目标。

### 4.4 L515 / D435I 移动 overview 相机：对齐到 model world

完整独立操作手册见
[`scripts/real_setting/README_CAMERA_MOTION.md`](scripts/real_setting/README_CAMERA_MOTION.md)。

头跟随手运动时，相机系中的末端位姿可能近似不变。若直接把它当作固定相机 state，真实运动会被
相机运动抵消，动作标签退化为接近零。`overhead` 只标识 overview 相机通道；稳定参考坐标叫
`world`。默认 `episode_first` 将它定义为 episode 第一帧 overview 相机：

```text
T_T_Ct  : 第 t 帧相机到 VIO/SLAM tracking frame 的变换
T_W_Ct = inverse(T_T_C0) @ T_T_Ct
p_W    = T_W_Ct @ p_Ct
T_W_Et = T_W_Ct @ T_Ct_Et
```

点云和 EEF 位姿必须一起变换。只移动点云而不移动 EEF 会制造坐标系错误。二者一起变换后，
最终送入模型的当前 EEF 系点云保持物理等价；真正被修复的是 `world_ee_poses`、state/action
轨迹及依赖该轨迹生成的运动先验。VIO/SLAM 的 tracking frame 只是外部定位坐标，不能称为
模型 world。

#### 采集与现场调试

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera L515 \
  --output /path/to/raw_sequence \
  --storage compressed \
  --record-imu \
  --debug-visualization \
  --debug-rgbd-odometry \
  --debug-save-every 30
```

调试窗口同时显示 RGB、深度有效率与尺度分位数、深度边缘在 RGB 上的重合情况、点云 XZ/XY
投影、采集 FPS、gyro/accel 与 RGB 的时间差，以及动态鲁棒 RGB-D world-anchor 的 fitness、
RMSE、累计平移与旋转。按 `S` 保存 dashboard，按 `R` 把下一帧设为新的 world 参考帧。自动截图位于
`<raw_sequence>/debug/`，原始多速率 IMU 位于 `imu.jsonl`，每条 `frames.jsonl` 记录还带有
最近邻 IMU 样本。`metadata.json` 同时保存实际选择的 gyro/accel rate、motion intrinsics、
motion sensor 到 color camera 的 librealsense 外参和设备/固件信息，便于后续 VIO 标定审计。

`--debug-rgbd-odometry` 使用固定首帧 world 锚、多帧静态背景一致性、RGB+几何配准和 Tukey
鲁棒核，能够抑制局部移动的人手与物体；它仍只用于质量检查，不会写成训练用 camera pose。
L515/D435I 的 gyro/accel 不能通过简单双积分可靠恢复平移。正式数据必须由同步
RGB-D-Inertial SLAM/VIO 或外部 6-DoF tracking 提供每帧
`T_tracking_camera`。tracking frame 的任意全局原点会在首帧归一化中消去，不会成为模型输入。

正式采集无需同步运行上述重调试。`--camera-trajectory-mode rgbd_odometry` 会先保持原速采集，
关闭相机后再估计轨迹；`--camera-trajectory-mode external` 则严格接入完整外部 SLAM/VIO
轨迹。添加 `--visualize-aligned-point-cloud` 后会动态播放所有首帧对齐点云，并显示固定大型
XYZ 原点轴、当前相机轴和相机轨迹。固定相机默认使用 `--camera-trajectory-mode static`，
为每帧写入单位位姿，数值行为与旧数据一致。完整参数与实测 FPS 见独立操作手册。

固定相机漂移检查可直接执行：

```bash
python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
  --camera L515 \
  --output /path/to/stationary_check \
  --num-frames 120 \
  --warmup-frames 30 \
  --stationary-pose-check \
  --record-imu
```

D435I 只需改为 `--camera D435I`；无显示服务器增加 `--debug-headless`。逐帧诊断位姿和
PASS/FAIL 报告分别保存在 `debug/rgbd_odometry.jsonl` 与
`debug/stationary_pose_report.json`。

#### 将 VIO/SLAM 轨迹写入 HDF5

相机轨迹 JSONL 必须按 `record_index` 与 RGB-D 帧显式匹配，不能猜测时间偏移。每行示例：

```json
{"record_index": 17, "timestamp_ms": 1234.5, "camera_to_tracking": [[1,0,0,0.02],[0,1,0,0],[0,0,1,0.01],[0,0,0,1]], "tracking_source": "vio", "valid": true}
```

构造 HDF5 时保持原始点云与手部位姿均在当前相机系：

```bash
python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
  --input /path/to/raw_sequence \
  --jsonl /path/to/handpose_wilor.jsonl \
  --camera-pose-jsonl /path/to/camera_pose.jsonl \
  --require-camera-pose \
  --camera-pose-max-sync-error-ms 20 \
  --camera-reference-mode episode_first \
  --pose-frame camera \
  --segments 0:200 \
  --no-interactive
```

输出字段为 `observations/camera_tracking_pose/<camera>`，矩阵语义固定为
`T_tracking<-camera`、平移单位
meter。一个 segment 中只要缺失任意一帧位姿就会报错，避免部分轨迹被静默当作固定相机。
当 RGB-D 与 pose 同时带时间戳时，默认还会检查两者绝对时间差不超过 20 ms，并把逐帧误差写入
`observations/camera_tracking_pose_sync_error_ms`。

#### 转换为 LeRobot dataset 并验收

```bash
python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  --input-dir /path/to/hdf5_raw \
  --output-root /path/to/lerobot_dataset \
  --repo-id real_head_camera_dataset \
  --camera overhead \
  --camera-motion-compensation required \
  --camera-reference-mode auto \
  --camera-motion-debug \
  --camera-motion-debug-episodes 2 \
  --vis-count 2
```

`auto` 模式在存在相机轨迹时补偿，缺失时保持原固定相机行为；正式头戴数据应使用 `required`。
审计文件包括：

- `world_ee_poses/episode_xxxxxx.npy`：`T_world<-ee`；
- `camera_motion/episode_xxxxxx.npy`：`T_world<-current_camera`；
- `visualizations/episode_xxxxxx/camera_motion/overlay_raw_as_if_camera_fixed.ply`；
- `.../overlay_aligned_to_world.ply`；
- `.../trajectories_camera_blue_raw_ee_red_aligned_ee_green.ply`；
- `.../trajectory.csv` 和 `README.txt`。

正确结果应满足：静态背景在 aligned overlay 中明显比 raw overlay 更锐利；当头跟随手时，红色
原始 camera-relative EEF 轨迹可能接近静止，但绿色首帧参考系 EEF 轨迹应恢复真实运动。

若要让所有 episode 和旧固定 overview 相机严格共用一个 world，使用
`--camera-reference-mode canonical` 并提供同一持久 tracking/base 坐标中的
`T_tracking<-canonical_camera`。每次重置原点的 odometry 无法单独提供这种跨 episode 对齐。

---

## 5. PointSeg：基于轨迹的软前景学习

### 5.1 为什么不用硬标签

机器人任务中的“前景”随阶段变化：抓取前，杯子和杯把重要；搬运时，杯子与夹爪形成共同刚体；
挂杯末端，杯架又必须持续重要。固定实例 mask 或一次性二分类无法完整表达这种关系。

当前先验只输出软概率与置信度：

- 跟随末端共同运动的点获得工具/被操作物体证据；
- 末端沿轨迹逐渐接近的点获得目标证据；
- 即使局部阶段速度很小，只要长期处于接触邻域，仍保留前景概率；
- 远离整段交互轨迹且缺乏其他证据的点获得背景置信度。

### 5.2 双向轨迹窗口

默认时间偏移为：

```text
0, -31, -16, -8, -4, -2, -1, +1, +2, +4, +8, +16, +31
```

它既观察过去也观察未来，适用于离线 cache 和训练时在线计算。由此避免只看单向未来时，在任务
末端失去已接触物体或目标结构。

轨迹优先来自实际 `observation.state`。只有旧数据集无法区分状态与动作时，才回退到 action。

### 5.3 八类几何先验

当前 cache 为每个采样点计算：

1. 当前点到 EEF 的距离；
2. 点到整段 EEF 轨迹的最小距离；
3. 沿时间窗口的接近程度；
4. 与末端共同运动的刚体残差；
5. 静态场景假设下的残差；
6. 共同运动与静态残差之间的差距；
7. 在轨迹邻域中的停留 / 接触持续性；
8. 该点在时间窗口中的可观测上下文。

这些量组合为三个软证据通道：

```text
tool_comotion
trajectory_approach
near_contact
```

最终前景概率使用 soft probabilistic-OR 聚合，而不是阈值硬拼接。背景置信度由“远离轨迹且
前景证据低”产生。训练标签本身保持 ignore，真正监督来自 soft class score 与 sample weight。

### 5.4 PointSeg 网络与损失

`SongPointSegNet` 只把当前帧原始 XYZRGB 作为网络输入。时间窗口先验默认仅用于构建监督，
不会在推理时成为额外输入，所以在线部署只需一帧点云。

PointSeg loss 主要由：

- soft binary cross entropy；
- 体素邻域平滑项；
- 有效点与先验置信度加权；

构成。当前只区分 foreground / background，不要求模型预测“杯子、杯架、夹爪”等人工角色。

`pointseg_use_temporal_priors_as_input` 和 `pointseg_use_pseudo_selection` 目前仅保留在配置与
历史命令中，现行主路径没有读取它们来改变网络输入或选点。它们设为 `false` 可表达当前实验
意图，但不要把效果差异归因于这两个参数。

### 5.5 前景和背景选点

PointSeg 输出每点前景概率。模型随后：

- 前景：按预测分数从高到低选择；
- 背景：优先选择 `score <= 0.5` 且分数最低的点；
- 数量：`max(min_points, ceil(max_valid_points_in_batch × ratio))`；
- 候选不足：循环重复已有候选，形成固定长度张量；
- 无效点：由 pad mask 排除。

因此，若一帧只有 3,072 点而请求 4,000 个前景，模型会保留候选并重复补齐，而不是凭空生成新点。
cache 的 `current-points` / `future-points` 不同：若源点数小于上限，cache 保留全部点，不重复。

---

## 6. LitePT 与点—动作交互

### 6.1 LitePT 当前特征输入

当前 `LitePTTokenizer` 将：

```text
coord = XYZ
feat  = [XYZ, RGB / 255]
```

送入 LitePT。也就是说 XYZ 被使用两次：

- 作为稀疏体素、邻域和序列化所需的空间坐标；
- 作为可学习 feature 的一部分，显式保留几何位置和尺度。

实验证明仅用 RGB 作为 feat 会显著削弱几何目标关注和真实机械臂零样本迁移，因此当前保留
XYZRGB feat。代价是模型不具备严格平移 / SE(3) 不变性；相机或坐标原点变化会改变 feature
分布。这是当前明确的鲁棒性限制。

### 6.2 已移除的 center 二次编码

`LitePTEncoder` 仍实例化了两个 backbone 以兼容历史 checkpoint，但现行 forward 只使用第一个：

```text
LitePT -> self-attention -> scalar alpha -> weighted global pooling
```

旧实现曾根据第一阶段 alpha 预测 center，把点云减去 center 后再次送入第二个 backbone。该路径
对很小的 center 偏差过度敏感：即使 alpha 关注对象几乎不变，XYZ 作为 feat 的整体偏移仍会被
后续 attention 放大，导致 global feature 和动作显著改变。因此该二次 center 路径已经注释停用。

不要简单使用点云几何均值替代预测 center。几何均值会受到背景范围、遮挡和离群点支配，也不能
代表操作对象中心。

### 6.3 Alpha 的含义

LitePT 内部 alpha 是特征聚合注意力，不是 PointSeg 标签。它可以自动聚焦任务细节，但当前不会
反向修改离线先验，也不应被直接当成硬前景监督。两者职责不同：

- PointSeg：从运动先验学习“哪些点与操作相关”；
- LitePT alpha：在选出的前景内部学习“哪些 token 对动作编码更重要”。

### 6.4 PointACT 风格融合

最终前景 LitePT token 与 flow-matching action token 拼接：

```text
[point tokens, action tokens + sinusoidal step embedding]
                         │
              bidirectional self-attention
                         │
                  keep action part
                         │
              FFN + residual action update
```

关键性质：

- fusion 内没有 causal mask，point 与 action token 双向交互；
- action chunk 的每一步具有固定正余弦位置编码；
- `action_is_pad=True` 被正确转换为无效 token，并同时参与 key padding 与输出屏蔽；
- 没有 self / VLM / point / FFN gate；
- action-conditioned point token 不会反馈回 LitePT 的全局 pooling；
- Action Expert 之后仍会通过 RoPE 获得其内部序列位置。

这里使用正余弦位置编码而不是 RoPE，是因为该模块是独立的 PyTorch
`MultiheadAttention`，发生在 Action Expert 的 RoPE 注意力之前。两者分别解决 fusion 层和
VLM/Expert 层中的位置身份问题。

---

## 7. SmolVLM 前缀与 Action Expert

### 7.1 Prefix token

adapter 模式下，prefix 由以下 token 构成：

1. overview RGB image tokens；
2. 前景点云全局 token；
3. 背景点云全局 token；
4. task language tokens；
5. 可选 robot state token，默认配置通常关闭。

图像与语言部分保持官方 SmolVLM 架构和 processor，以便正确加载
SmolVLA 0.45B 的预训练 VLM 权重。点云 token 通过投影进入相同 hidden space。

### 7.2 Adapter 训练范围

推荐设置：

```bash
--policy.vla_adapter_enable=true
--policy.vla_adapter_freeze_vlm=true
--policy.load_vlm_weights=true
```

此时冻结视觉编码器和 VLM Transformer；PointSeg、LitePT、点云投影、Point–Action fusion
与 Action Expert 仍可训练。冻结模块不等于切断梯度：点云 token 经过冻结 VLM 运算时，梯度
仍可回传到点云投影和编码器。

当前版本没有 adapter gate。早期实验中的 `self_gate`、`vlm_gate`、`point_gate`、
`ffn_gate` 不属于现行模型。

### 7.3 Prefix 与 suffix 的注意力

Action Expert 的 suffix 是 noisy action chunk 与 diffusion time 的 embedding。注意力块关系为：

```text
prefix: [RGB, point, language, optional state]
suffix: [action_0, action_1, ..., action_31]
```

- prefix 内部双向可见；
- 每个有效 action token 可以读取全部 prefix；
- action chunk 内部相互双向可见，不是 causal attention；
- prefix 不读取 suffix；
- VLM 与 Expert 的 Q/K 通过 position id 应用 RoPE；
- 默认 `self_attn_every_n_layers=2`，在联合 / 交叉注意力结构中交替传递信息。

该设计适合一次生成完整动作块。它不是自回归 next-token action generation，因此不需要把
action chunk 强制改成 causal attention。

### 7.4 Flow matching 动作生成

训练时的通用 rectified-flow 形式为：

\[
x_t=(1-t)x_0+t a,\qquad
u_t=a-x_0.
\]

其中 `x0` 和 `t` 由 checkpoint 配置决定：v0.4.2 兼容模式使用所有 channel 独立
`Normal(0, 0.1)` 与 `Beta(1.5,1)`；当前随机 pose9 模式先在 se(3) 按米/弧度采样并映射成
合法 SE(3) 起点，也可用 `integration_grid` 让训练时刻与 10 步 Euler 推理网格一致。具体区别见
8.7。

模型以 \(x_t,t\) 和 prefix 为条件预测速度 \(v_\theta(x_t,t,c)\)，优化：

\[
\mathcal{L}_{action}
=
\frac{\sum m\lVert v_\theta-u_t\rVert_2^2}
{\sum m},
\]

其中 \(m\) 排除 episode 尾部补齐的 action token。

推理从同分布噪声开始，默认使用 10 步 Euler 积分：

\[
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k,c),\qquad \Delta t=0.1.
\]

因此动作生成本身具有随机性。要严格比较两个入口、两次推理或 modality ablation，必须固定
相同 noise seed，并使用相同 batch 构成和数值执行路径。

启用 World–Ego v2 时，`independent` 兼容模式从同一 sample seed 派生独立 World SE(3)
noise；推荐的 `conjugate_ego` 模式不再独立采样，而是从已经固定 seed 的 Ego prior 和当前 EEF
pose 解析得到 World prior。两种方式都不依赖动态 batch 请求顺序，但只有后者描述同一个物理随机
变换。

---

## 8. WorldFlow 与 SE(3) 实验分支

### 8.1 World–Ego v2 当前实现

`worldflow_enable=true` 不再创建第二套 Action Expert，而是创建一套独立 World 点云前端，并让
World 与 Ego 在官方 SmolVLA Action Expert 中彼此学习。数据路径为：

1. PointSeg 按模型预测选出 Ego/body-frame XYZRGB 前景点；
2. World 分支只接收这些 XYZRGB 点，不接收前景概率、pseudo label、role score 或真实点流；
3. 用当前末端 world 位姿 \(C=T_{W\leftarrow E_0}\) 将同一批前景点解析地变换到 World；
4. Ego 与 World 各自使用独立 LitePT、动作/时间投影和 PointAction adapter；
5. 两个前端分别输出一个 `global_feat` scene token 与已经融合完整点特征的 `action tokens`；
6. 两个 scene token 和两组 action token 进入同一个官方 Action Expert，分别预测 Ego body action velocity
   与 World spatial-transform velocity；
7. 训练和 rollout 推理均同时维护、去噪两组 action chunk，最终只向控制器返回 Ego chunk。

World 分支的独立边界止于 PointAction 输出。共享专家之前没有通过加权求和或 gate 混合两个
分支，也没有另建简化 MLP expert。

### 8.2 联合 token 与 attention 语义

Action Expert 的逻辑序列为：

```text
VLM prefix
  └─ RGB / language / point summary

shared-expert suffix
  ├─ scene block: [Ego global scene token, World global scene token]
  └─ action block: [Ego action_0..T-1, World action_0..T-1]
```

- Ego/World 分别用 LitePT 注意力池化后的 `global_feat` 构造一个 scene token；
- 完整 LitePT 点 token 不直接进入共享 Action Expert，只在各自 PointAction 内与动作交互；
- Ego/World scene token 加入不同的可学习 branch-type embedding；
- Ego/World action token也加入不同的 branch-type embedding，不能只靠序列位置猜测坐标来源；
- 两组 scene token 位于同一个双向 block，可相互交换几何上下文；
- 两个完整 action chunk 位于同一个双向 block，所以任意 Ego action step 与任意 World action
  step 都可双向通信；
- scene block 不能读取后面的 action block；
- action block 可以读取全部 VLM prefix、全部 scene token 与两组全部有效 action token；
- padded scene/action token 同时受 key/query mask 排除；
- SmolVLA 原有 RoPE position id 仍在 VLM/Expert attention 内生效，PointAction 内仍保留动作
  step position embedding。

每个分支的信息路径都保持为原始模型的两级结构：

```text
完整 LitePT 点 token ── PointAction ── point-fused action token
          └──────── attention pooling ── 单个 global scene token
```

所以共享专家始终只接收两个场景级 token，不需要对点 token 额外采样或重汇聚。点级几何细节
通过 PointAction 写入动作 token，`global_feat` 则提供场景级上下文。该路径不使用前景概率、
role、PCA 或手工目标点。

这里的“causal”是 block 间的 prefix-causal，而不是 action chunk 内的自回归 causal。模型一次
生成整个 chunk，因此两组 action token 内部和彼此之间保持双向。

### 8.3 几何目标与损失

设当前末端位姿为 \(C=T_{W\leftarrow E_0}\)，未来绝对末端位姿为
\(A_t=T_{W\leftarrow E_t}\)。两个分支的目标分别是：

\[
B_t=C^{-1}A_t,\qquad G_t=A_tC^{-1}.
\]

二者通过 SE(3) 共轭关系严格连接：

\[
B_t=C^{-1}G_tC.
\]

关闭 `se3_enable` 时，World 分支对 \(G_t\) 做 pose9 Euclidean flow matching；开启
`se3_enable` 时，Ego 和 World 都改为 SE(3) 流形上的左平凡化 twist flow。两种模式都计算
endpoint SE(3) 几何损失；bridge loss 比较 \(C^{-1}\hat G_tC\) 与共享专家输出的 Ego endpoint。
随机坐标增广 \(S\) 同步作用于 World 点云、World flow 起点和监督标签：

\[
G'_t=SG_tS^{-1},\qquad \hat G'_t\approx S\hat G_tS^{-1}.
\]

总辅助损失仍由 `flow + SE(3) endpoint geometry + World–Ego bridge + conjugation
equivariance` 组成。完整 SE(3) 模式中的流路径为：

\[
H(t)=\operatorname{Exp}\!\left(t\operatorname{Log}(H_1H_0^{-1})\right)H_0,
\qquad
\xi(t)=\frac{\operatorname{Log}(H_1H(t)^{-1})}{1-t}.
\]

训练预测单位流时间下的 6D twist \(\xi\)，推理每一步执行
`H <- Exp(dt * xi) H`；旋转在训练插值和推理积分过程中都保持为合法 SO(3)。该实现不预测点流、
不运行 Kabsch、不使用 PCA，也不引入人工角色监督。

### 8.4 推理路径与 current pose 契约

推理不再绕过 World 分支。每次 policy call：

1. Ego flow 从原来的动作噪声初始化；
2. World flow 从合法 SE(3) 噪声初始化；`conjugate_ego` 时该噪声由 Ego 起点解析共轭得到；
3. PointSeg 选择 Ego 前景，再用当前 \(C\) 得到 World 前景；
4. 每个积分 step 重新生成两组 point-fused action token，经过共享专家得到两组 velocity；
5. 普通模式对两组 pose9 做 Euclidean Euler 积分；完整 SE(3) 模式分别对 Ego/World twist 做
   `Exp(dt*xi)` 群积分；最终只返回 Ego action。

因此开启 WorldFlow 的 checkpoint 在在线推理时必须提供
`worldflow.current_ee_pose: (B,9)`。它只是解析坐标变换载体，不作为 Ego robot-state prefix：

- `libero_pointcloud_eval.py` 从当前 simulator EEF pose 自动加入；
- 两份 `smolvla_model_inference.py` 优先读取显式字段；真实机器人 `single_inference` 可从原始
  `pose_eular`/state 自动构造；
- 普通 Ego `observation.state` 仍按 UMI 契约置为 identity，不会因此改回绝对位姿策略。

关键开关为：

```bash
--policy.worldflow_enable=true
--policy.worldflow_max_points=0
--policy.worldflow_loss_weight=0.05
--policy.worldflow_geo_loss_weight=0.05
--policy.worldflow_bridge_loss_weight=0.05
--policy.worldflow_equiv_loss_weight=0.02
--policy.se3_enable=false
```

`worldflow_max_points=0` 表示保留完整预测前景；正值只作为显存上限，不是额外前景筛选。
`worldflow_action_expert_layers` 和 `worldflow_action_expert_dropout` 只为兼容 v0.5 命令保留，
v2 中被忽略，因为只存在一套共享 Action Expert。WorldFlow 可以和 `se3_enable` 同时开启，但二者
都仍不能和 RTC 同时开启。完整四套件尚未完成同规模复验，因此通用默认仍是
`se3_enable=false`；但 8.8 的 `pose9_endpoint` 配置已经在 Spatial task 5 以真实随机采样达到
官方 10/10，不能再把所有完整 SE(3) 模式概括为“只有软件正确性”。

### 8.5 为什么仍默认关闭

v2 改变了 suffix 长度、共享专家的信息流和推理调用链，与 v0.5 auxiliary checkpoint 以及更早
Dense ObjectFlow checkpoint 都不具备功能等价性。旧 checkpoint 即使以非严格方式加载，也会
缺少 scene projection、branch-type embedding 和 World output projection，不能视为可用 v2
模型，必须重新训练。完成独立消融前仍默认：

```bash
--policy.worldflow_enable=false
--policy.worldflow_se3_head_enable=false
```

### 8.6 其他 SE(3) 参数

- `se3_enable=true`：把 Ego 动作流切换为完整的 SE(3)-twist geodesic flow；若同时开启
  WorldFlow，World 流也同步使用 6D twist 头和群积分，并通过共轭连接。
- `se3_noise_trans_scale`：Ego 起点平移李代数噪声标准差，单位米。
- `se3_noise_rot_scale`：Ego 起点旋转李代数噪声标准差，单位弧度。
- `se3_noise_gripper_scale`：夹爪宽度起点噪声标准差，单位米。
- `se3_twist_head_mode`：
  - `direct_twist` 新建 Ego 7D / World 6D 输出头，数学最直接，但旧 pose9 checkpoint 无法继承
    末端动作头知识；
  - `projected_pose9` 把旧 pose9 输出解释成瞬时矩阵导数并投影为 spatial twist；
  - `pose9_endpoint` 保留旧 flow head 的终点语义，先得到
    `x_hat_1=x_t+(1-t)v_pose9`，把 rotation-6D 投影到 SO(3)，再计算
    `Log(H_hat_1 H_t^-1)/(1-t)`。随机起点、训练路径和积分仍全部位于 SE(3)，同时最适合从已训练
    pose9 checkpoint warm-start；当前 10/10 候选使用该模式。
- `se3_enable=true` 与 `pose9_action_noise_enable=true` 互斥；前者已经包含完整流形 prior，不能再叠加
  pose9 prior。
- `worldflow_se3_head_enable`：历史兼容参数，当前忽略。
- `se3_final_correction_enable`：历史兼容参数，当前忽略。

### 8.7 随机动作流：SO(3) prior、旧版 Gaussian 与双流噪声契约

#### v0.4.2 原生普通噪声

v0.4.2 对全部 action channel 使用独立的 Euclidean Gaussian：

```python
noise = Normal(0, 0.1)  # xyz、rotation-6D、gripper 一视同仁
x_t = (1 - t) * noise + t * action
u_t = action - noise
```

优点是实现简单、与官方 normalized-action flow 形式一致、旧 checkpoint 可严格复现。缺点是本项目
使用 identity-normalized pose9：六个旋转 channel 的零均值随机向量通常不是正交旋转的前两列，
甚至在接近零时需要依靠后续 Gram–Schmidt 才能解释成旋转；0.1 对米、弧度和夹爪宽度也没有统一
物理含义。

#### 当前 SO(3)/SE(3) 合法随机 prior

开启 `pose9_action_noise_enable` 后，先在李代数中按物理单位采样：

\[
\xi_0=[v_0,\omega_0],\quad
v_0\sim\mathcal N(0,\sigma_t^2),\quad
\omega_0\sim\mathcal N(0,\sigma_r^2),\quad
H_0=\operatorname{Exp}(\xi_0),
\]

再把合法的 `H0` 转成 pose9；夹爪独立按米采样。当前用于研究的随机起点（不是已验证的生产默认值）：

```bash
--policy.pose9_action_noise_enable=true \
--policy.pose9_action_noise_trans_scale=0.15 \
--policy.pose9_action_noise_rot_scale=0.35 \
--policy.pose9_action_noise_gripper_scale=0.05
```

它保证随机起点旋转正交、尺度可解释，并避免 rotation-6D 零向量退化。但当前标准 Ego 路径仍在
pose9 的 Euclidean 表示中做 rectified-flow 直线插值和 Euler 积分；因此这是“合法 SE(3) 端点
prior + Euclidean pose9 flow”，不能宣称整条中间路径严格位于 SO(3) 流形。若需要整条路径都在
群上，应改用 `se3_enable=true`；该模式现在可以与 WorldFlow 同开，而且两路共轭 geodesic 在全部
训练时刻都保持同一个物理变换。

完整 SE(3)+WorldFlow 的已验收小先验配置：

```bash
--policy.se3_enable=true \
--policy.se3_twist_head_mode=pose9_endpoint \
--policy.pose9_action_noise_enable=false \
--policy.se3_noise_trans_scale=0.001 \
--policy.se3_noise_rot_scale=0.005 \
--policy.se3_noise_gripper_scale=0.0005 \
--policy.worldflow_enable=true \
--policy.worldflow_noise_coupling=conjugate_ego
```

三个尺度均为严格非零的物理随机量，分别对应 1 mm、约 0.286° 和 0.5 mm 标准差；它不是
identity prior，也没有绕过去噪积分。该尺度已在 task5 固定 seed 上闭环验收，但不能直接外推为
所有 LIBERO task 或真机的最优噪声。

对于小数据，随机 prior 比单位起点明显更难：模型必须覆盖一族初始状态，而不是记住一条从 identity
出发的确定性向量场。随机版通常需要更多数据/step，且应同时检查多 seed rollout；低 training loss
本身不能证明采样质量更好。它的潜在收益是保留多模态动作分布、降低对固定起点的过拟合，
并使推理时可通过 noise seed 产生不同候选轨迹；这些收益必须用充分训练后的多 seed
闭环结果证明，不能从 prior 的几何合法性直接推导。

#### 双流必须共享同一个物理随机轨迹

令 `C` 为当前 EEF-to-World，Ego 随机 body transform 为 `B0`。推荐模式不是再独立采一个 World
噪声，而是解析地构造：

\[
G_0=C B_0 C^{-1}.
\]

配置：

```bash
--policy.worldflow_enable=true \
--policy.worldflow_noise_coupling=conjugate_ego
```

`conjugate_ego` 下 `worldflow_noise_trans_scale/rot_scale` 被忽略；World prior 完全由 Ego prior 与
当前 `C` 决定。日志指标 `worldflow_noise_conjugacy_error` 应接近数值零。历史
`worldflow_noise_coupling=independent` 保留用于旧实验复现，但随机 Ego/World 起点描述的是两条不同
物理轨迹，会额外增加 bridge 学习难度。零噪声时两者都是 identity，因此旧问题会被掩盖。

严格对比噪声方案时，必须固定：数据、cache、初始化 checkpoint、offset、batch 顺序、训练 step、
PointSeg、WorldFlow 权重、推理 seed 和执行器。尤其不能再拿旧 `offset=0` 的随机实验与
`offset=1` 的确定性实验直接比较。

### 8.8 受控实验结论（2026-08-02）

在 LIBERO Spatial task 5 上，使用同一 dataset/cache、`action_chunk_start_offset=1`、固定推理
seed 和完全相同的无重试闭环执行器，完成了当前代码版本的最终验收：

| Ego/World flow | SE(3) 起点尺度（平移/旋转/夹爪） | 额外训练 | 训练/离线表现 | 官方固定初始状态闭环 |
| --- | --- | ---: | ---: | ---: |
| pose9 Euclidean + WorldFlow，identity prior | 0 / 0 / 0 | 0 | 确定性基线 | **10/10** |
| 完整 SE(3)，`direct_twist` 随机新头 | 0.03 m / 0.15 rad / 0.02 m | 1,000 | endpoint 50.29 mm / 3.05° | **0/3** |
| 完整 SE(3)，`projected_pose9` | 0.01 m / 0.05 rad / 0.005 m | 1,000 | 随机采样平移误差 16.14 mm | **2/3** |
| 完整 SE(3)，`pose9_endpoint` | 0.001 m / 0.005 rad / 0.0005 m | **250** | 随机采样平移误差 15.89 mm；训练 endpoint 约 6.8 mm / 0.85° | **10/10** |

各完整 SE(3) 候选的 `worldflow_path_conjugacy_error` 均维持在约 `1e-7` 或更低，独立真实模型
smoke test 中为 `3.35e-9`；说明两路随机起点、任意时刻路径和坐标共轭实现都达到浮点数值精度。
单元测试还验证了旋转正交性、行列式为 1、用目标 twist 恢复 endpoint、训练 mask 和三套
WorldFlow dataset wrapper 的动作 offset 一致性。

注意 `loss_action` 不是不同 prior 之间可以单独下结论的指标：它的监督目标和物理尺度会随 prior
改变。完整去噪 chunk 误差和不重试的闭环 rollout 才是最终判据。这次结果支持三个结论：

1. 完整 SE(3) 加噪、geodesic 插值、twist 监督和群积分真实执行，不是把噪声置零或绕过流。
2. `direct_twist` 的失败主要来自丢弃旧动作头并随机初始化新头，而非 SE(3) 共轭公式错误；
   `projected_pose9` 进一步证明把旧输出误解为瞬时导数仍有明显 warm-start 失配。
3. `pose9_endpoint` 保留旧 head 已学到的“预测 clean endpoint”语义，再把该 endpoint 转成严格
   SE(3) geodesic velocity，因此只需 250 个低学习率更新就恢复到官方 10/10。
4. 当前 task5 结论支持该模式进入扩大实验，但完整四套件与多随机 seed 尚未验收，所以仓库通用
   默认值仍保持 `se3_enable=false`。

最终候选与证据路径：

```text
checkpoint:
/tmp/temp/training/task5_fullse3_endpoint_tinyprior_from_win_000500/
  checkpoints/000250/pretrained_model

official 10-episode report:
/tmp/temp/evaluations/task5_fullse3_endpoint_tinyprior_000250_official_10eps/
  overall_report.json
```

验收协议为 `policy_noise_mode=sample`、`policy_noise_seed=0`、官方 init states 0--9、每个状态一次
不间断 rollout、无失败重试、无候选筛选、`action_index=0`、20 Hz、`exec_action_steps=20`。10 个
episode 全部成功，用时 76--103 control steps；因此该结果不是 oracle、dataset-domain 环境或
零噪声评测。

复现最终闭环：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
CUDA_VISIBLE_DEVICES=0 SONG_POINTSEG_REQUIRE_POINTOPS=1 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /path/to/fullse3_endpoint/checkpoints/000250/pretrained_model \
  --suite libero_spatial \
  --task-id 5 \
  --episodes 10 \
  --task-workers 1 \
  --isolated-policy-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --action-index 0 \
  --exec-action-steps 20 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --control-freq 20 \
  --no-use-suite-max-steps \
  --max-steps 1000 \
  --policy-noise-seed 0 \
  --policy-noise-mode sample \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /path/to/task5_fullse3_official_10eps
```

100,000 个 prior 的静态审计还显示：v0.4.2 raw Gaussian 的 rotation-6D 列范数误差均值
0.840，Gram–Schmidt 解码后旋转角中位数 132.6°；合法 SE(3) prior 的正交误差为
1e-8 级，旋转角中位数 30.8°。所以 raw Gaussian 虽然每个 channel 的标准差只有 0.1，
并不代表“只加了小旋转”。

几何限制现在按模式区分：`se3_enable=false` 的 pose9 Euclidean 双流仍只保证起点/终点共轭，
中间线性点不严格保持共轭；`se3_enable=true` 则使用两路共轭 geodesic，任意中间时刻都严格满足
`G(t)=C B(t) C^-1`。后者解决了表示问题，但本轮闭环结果证明，解决几何表示并不等于小数据下
立即获得更好的策略性能。

复现 World 分支因果消融时，对同一 checkpoint、episode 和 `--policy-noise-seed`
分别运行正常评测与：

```bash
python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  ... \
  --ablate-worldflow-tokens
```

该开关保持 token 布局、position id、Ego noise 和 World noise 不变，只屏蔽 World scene/action
token。“结果变了”证明 World 分支被实际使用；“成功率变好”才能证明它对当前任务有正收益。

---

## 9. LIBERO 数据转换：正确性优先

### 9.1 旧数据为什么会错

旧转换脚本的核心问题不是“所有 episode 看起来位置完全相同”，而是没有完整恢复每条 demo 的
环境定义。

LIBERO 官方 HDF5 为每条 demonstration 保存独立 `model_file`。其中包含柜体、把手、容器等
fixture 的 XML 布局。flattened simulator state 主要保存动态状态，不能可靠替代整份模型 XML。
旧脚本即使重置 state 后能看到不同物体位置，也可能仍沿用同一 task 环境的其他模型几何，
导致轨迹与柜体 / 把手出现毫米到厘米级错位。接触任务中，这足以让轨迹“差一点”或接触后滑走。

第二个问题是帧对齐。官方 LIBERO v1 文件中，当前观测与 state 存在固定一帧关系。实测：

- 用 `state[i+1]` 重建官方 `obs[i]`，EEF 平均位置误差约 0.304 mm；
- 用 `state[i]` 重建 `obs[i]`，平均误差约 7.48 mm。

但动作标签仍属于 step `i`，不能把所有数组一起简单平移。

第三个问题是动作语义。官方 HDF5 action 是归一化 delta OSC 命令，不是实际 EEF 位姿。将它直接
当绝对目标，或把实际 state 当 action，都会破坏控制目标。

### 9.2 当前修复

当前转换器逐 demo：

1. 恢复该 demo 自己的 `model_file`；
2. 清理并同步 controller goal history；
3. 用 `states[i+1]` 重建输出观测 `obs[i]`；
4. 仍以 `states[i]` 为 action `i` 的控制起点；
5. 调用 controller 的 `set_goal(raw_action[:6])`，保持 `use_delta=True`；
6. 读取控制器生成的绝对 target pose；
7. 保存实际状态与命令目标两种轨迹，供审计和训练分别使用。

当前字段：

| 字段 | 语义 |
|---|---|
| `observation.state` | 重建观测时的实际 EEF pose + 实际夹爪宽度 |
| `action` | 控制器绝对 target pose + 对应物理夹爪宽度 |
| `world_ee_poses/` | 实际 EEF 在模型 world（固定 overview 相机参考系）中的 pose |
| `action_target_ee_poses/` | 命令目标在模型 world 中的 pose |

实际 state 与 action 在接触、跟踪误差或控制延迟下本来就不同。这是正确数据，不应再次对齐为同值。

#### 9.2.1 `state_observation_offset`、`action_chunk_start_offset` 与 `action_index`

三者位于不同边界，不能互相替代：

| 参数 | 生效位置 | 修复后 LIBERO 设置 |
|---|---|---:|
| `state_observation_offset` | HDF5 重建 observation | `1`，用 `state[i+1]` 重建 `obs[i]` |
| `action_chunk_start_offset` | LeRobot DataLoader 读取监督 chunk | `1`，token 0 读取 `action[i+1]` |
| `action_index` | 在线评测从模型预测 chunk 选哪一行执行 | `0`，执行模型 token 0 |

这里看似出现两个 `+1`，但不是重复平移：前者修复 observation 对应的 simulator 快照，后者跳过已经
产生该 post-action observation 的旧控制命令。模型用 offset 1 训练后，预测 token 0 已经是第一条
因果未来命令，所以评测再设 `action_index=1` 会错误地多跳一帧。旧 offset0 checkpoint 临时使用
`action_index=1` 只能作为兼容诊断，不能替代重新训练，也不能写回新数据规范。

### 9.3 点云与 RGB

每帧按以下顺序处理：

1. 从 `agentview` 深度反投影，在 overview 相机坐标系生成场景点云；
2. 将仿真世界中的 EEF pose 通过仿真相机外参转换到同一相机坐标系；
3. 在该坐标系添加虚拟夹爪；
4. 保持总点数为 `num_points`，默认可使用 9,500 场景点 + 500 夹爪点；
5. 将合并点云转换到当前 EEF 坐标系；
6. 保存 RGB 图像、点云和动作。

仿真外参仅用于把仿真世界 EEF 转到 overview 相机系。它不意味着真机也需要相机到机器人世界外参。

### 9.4 四套件转换命令

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /path/to/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 10 \
  --num-points 10000 \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 1 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --vis-dir /path/to/libero_4suite_lerobot_dataset/visualizations \
  --overwrite \
  --output-root /path/to/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud
```

`--num-workers` 是 CPU / MuJoCo 进程并行，不是多 GPU 渲染。单次脚本只由
`MUJOCO_EGL_DEVICE_ID` 选择一张卡。若要使用四卡，最稳妥的方法是把四个 suite 拆为四条命令，
每条命令使用独立 GPU、`output-root` 和临时目录，最后再按确定顺序合并；不能让四个父进程同时
写同一个 LeRobot dataset。

`libero.json` 当前关键默认值是 256×256、10,000 点、500 个夹爪点、20 Hz、
`state_observation_offset=1`、`restore_demo_model=true`。命令行会覆盖 config。相对路径从
`benchmarks/song_real_libero` 解析。

### 9.5 输出与检查

```text
libero_4suite_lerobot_dataset/
├── data/
├── images/
├── meta/
├── point_clouds/episode_XXXXXX.zarr
├── world_ee_poses/episode_XXXXXX.npy
├── action_target_ee_poses/episode_XXXXXX.npy
└── libero_collect_summary.json
```

`--vis-count 0` 明确表示不生成可视化。要得到 PLY / 视频检查结果，使用非零
`--vis-count`，并建议显式传入 `--vis-dir`。转换后至少检查：

- 同一 demo 的重建视频与官方轨迹布局一致；
- 柜体、把手、目标物体没有固定偏移；
- `observation.state` 与 `action` 不再被错误地写成同值；
- 首个 UMI 动作不被强制为单位旋转；
- `libero_collect_summary.json` 中恢复模型和帧偏移均符合预期。

---

## 10. 真实机器人 HDF5 转换

真实 HDF5 已包含 XYZRGB，因此脚本不下载数据、不重播轨迹，也不从深度图重新计算点云。默认：

```text
cloud:       observations/cloud_rgb/<camera>
image:       observations/images/<camera>
pose:        observations/pose_eular
gripper:     observations/eff_angular
timestamp:   timestamp_ms
```

真实数据的末端 pose 默认已位于 overhead 相机坐标系。脚本：

1. 读取相机系 XYZRGB；
2. 在同一相机系添加虚拟夹爪；
3. 固定相机直接以 overhead 为 reference；移动相机按配置对齐到 episode 第一帧或 canonical
   overview 相机；
4. 将合并点云转换到当前 EEF；
5. 从人手示范 pose 构造 state/action；
6. 保存 RGB、时间戳和 sidecar pose。

当前真实采集没有单独记录控制器 command target，因此 state/action 都源自示范 pose。这与修复后的
LIBERO 数据语义不同，混合训练时必须意识到这一 domain 差异。

示例：

```bash
python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  --input-dir /path/to/humanhand_hdf5 \
  --output-root /path/to/real_adapter_lerobot_dataset \
  --repo-id real_adapter_lerobot_dataset \
  --camera overhead \
  --cloud-frame camera \
  --pose-format auto \
  --timestamp-mode source \
  --num-points 10000 \
  --add-gripper-cloud \
  --gripper-points 500 \
  --gripper-len 0.06 \
  --point-cloud-storage zarr \
  --workers 8 \
  --vis-count 2 \
  --overwrite
```

如果输入 HDF5 已包含夹爪点，改用 `--input-has-gripper-cloud`，不要重复添加。adapter 训练需要
RGB，不能使用 `--image-key none`。图像应按 LeRobot image feature 接口保存为路径；直接把
NumPy image 放入期望路径的统计代码会触发 PIL `seek/read` 错误。

时间戳用于定义 episode 内真实采样时间和数据同步。`--timestamp-mode source` 优先使用 HDF5
时间戳；缺失时才按 `fps` 构造均匀时间。

---

## 11. 生成 PointSeg cache

### 11.1 cache 存储什么

cache 是 index-only 结构，不复制完整点云。每个样本包含：

- 当前 / 未来点索引；
- soft foreground / background score；
- 训练权重与有效性；
- 三个运动证据通道；
- episode / frame 映射；
- manifest 与生成参数。

当源点数不超过 `current-points` 或 `future-points` 时保留全部点；超过时无放回采样到上限。

### 11.2 四卡命令

```bash
export SONG_POINTSEG_REQUIRE_POINTOPS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --output-dir=/path/to/song_pointseg_cache_v7 \
  --current-points=10000 \
  --future-points=10000 \
  --batch-size=24 \
  --num-workers=8 \
  --shard-size=4096 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4 \
  --overwrite
```

`SONG_POINTSEG_REQUIRE_POINTOPS=1` 要求 CUDA PointOps KNN，缺少时直接报错，避免无意落入极慢的
PyTorch fallback。调试机器没有 PointOps 时可设为 `0`，但速度会明显下降。

### 11.3 可视化

`--vis-count > 0` 会输出原点云、pseudo soft score 和关键时间百分位结果。benchmark cache
预览使用连续的蓝 → 黄 → 红热力图：蓝色为低前景概率，红色为高前景概率，不做硬阈值。独立
PointSeg 训练脚本生成的 `pred.ply` 与 `pseudo.ply` 则共享“原始 RGB 背景 + 绿色前景覆盖”的
着色规则。两类文件都保存完整原点坐标，不会只保存标签或原点。

episode 的 p25 / p50 / p75 / terminal 可视化用于检查交互前景连续性。例如挂杯任务在 p75 到
terminal 期间，杯子、杯把与杯架都应保留连续软概率。

### 11.4 何时必须重建

以下任一变化都必须重建 cache：

- 重新运行 dataset 转换；
- 修改 `num_points`、点云采样或夹爪点；
- 修改点云坐标系或 RGB / XYZ 通道；
- 修复 demo model、state offset 或 episode/frame 映射；
- 修改运动先验算法、时间偏移或 evidence channel；
- 合并、裁剪或重排 dataset。

当前 strict 检查能验证长度和 episode/frame 索引，但不能为每帧点云做完整内容哈希。因此“程序没有
报错”不代表旧 cache 与新 dataset 语义兼容。

### 11.5 无 cache 在线计算

不提供 `pointseg_sample_cache_dir` 时，benchmark 训练脚本可以按 batch 在线计算同一类双向软先验。
它适合验证算法，不适合大规模重复训练。相关环境变量：

```bash
export SONG_POINTSEG_ONLINE=1
export SONG_POINTSEG_ONLINE_DEVICE=cuda
export SONG_POINTSEG_ONLINE_CURRENT_POINTS=10000
export SONG_POINTSEG_ONLINE_FUTURE_POINTS=10000
export SONG_POINTSEG_ONLINE_NN_CHUNK_SIZE=1024
export SONG_POINTSEG_ONLINE_FUTURE_OFFSETS=1,2,4,8,16,31
```

在线 CUDA prior 与 fork DataLoader 不兼容时，训练器会降低或关闭 worker。正式实验优先使用离线
cache。`SONG_POINTSEG_CACHE_STRICT=0` 只用于诊断，不应用于正式训练。

---

## 12. VLM 离线权重

建议分开保存官方架构 / processor 与 SmolVLA policy 权重：

```bash
huggingface-cli download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir /path/to/hf_models/SmolVLM2-500M-Video-Instruct

huggingface-cli download \
  lerobot/smolvla_base \
  --local-dir /path/to/hf_models/smolvla_base
```

对应参数：

```bash
--policy.vlm_model_name=/path/to/hf_models/SmolVLM2-500M-Video-Instruct
--policy.vlm_weights_path=/path/to/hf_models/smolvla_base
--policy.load_vlm_weights=true
```

二者职责：

- `vlm_model_name`：提供完整 SmolVLM 架构、config、processor、tokenizer 和原始模型定义；
- `vlm_weights_path`：指定权重覆盖源，可以是 SmolVLA policy checkpoint、单个 safetensors，
  或完整 raw SmolVLM 目录。

当 `vlm_weights_path` 是 `lerobot/smolvla_base` 时，加载器从 policy state dict 中抽取
`vlm_with_expert.vlm` 参数，并严格检查当前保留的 VLM 层是否缺失。不要把只下载了
`smolvla_base` 的目录改名成 `SmolVLM2-500M-Video-Instruct` 后同时承担两种职责；其中可能没有
完整 raw Hugging Face processor / tokenizer 布局。

---

## 13. 训练

### 13.1 推荐 adapter 训练命令

```bash
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --multi_gpu --num_processes 4 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --pointseg_sample_cache_dir=/path/to/song_pointseg_cache_v7 \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name=/path/to/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/path/to/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
  --batch_size=48 \
  --steps=80000 \
  --log_freq=1 \
  --output_dir=/path/to/outputs/wep_vla_v04_adapter \
  --job_name=wep_vla_v04_adapter \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=4000 \
  --eval_freq=4000 \
  --num_workers=8 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.001 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=1500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.point_action_fusion_enable=true \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

`batch_size` 是每个进程的 batch。上例有效 batch 为 `48 × 4 = 192`。24 GB 显卡应从更小
batch 开始，避免某个 rank 被系统 `SIGKILL`。

#### 13.1.1 已通过 task5 闭环验收的 WorldFlow 流契约

在已有双流训练命令上，当前稳定设置为：

```bash
--policy.worldflow_enable=true \
--policy.pose9_action_noise_enable=true \
--policy.pose9_action_noise_trans_scale=0 \
--policy.pose9_action_noise_rot_scale=0 \
--policy.pose9_action_noise_gripper_scale=0 \
--policy.worldflow_noise_coupling=conjugate_ego \
--policy.worldflow_loss_weight=0.01 \
--policy.worldflow_geo_loss_weight=0.002 \
--policy.worldflow_bridge_loss_weight=0.005 \
--policy.worldflow_equiv_loss_weight=0.002 \
--policy.se3_enable=false
```

已有 10/10 checkpoint 的旧 config 没有保存 `worldflow_noise_coupling`，加载时回退为
`independent`；由于 Ego 的三个 noise scale 和 World 的两个 noise scale 都为零，两路起点都是 identity，与
`conjugate_ego` 在数值上相同。新实验仍建议显式写出 `conjugate_ego`，避免以后只调大噪声却忘记
同步双流物理契约。

从已验证 identity checkpoint 迁移到完整随机 SE(3) 时，推荐 task5 受控配置为：

```bash
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/path/to/identity_10_of_10/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/path/to/libero_dataset \
  --pointseg_sample_cache_dir=/path/to/pointseg_cache \
  --batch_size=5 \
  --steps=500 \
  --save_freq=250 \
  --num_workers=4 \
  --output_dir=/path/to/fullse3_endpoint_run \
  --wandb.enable=false \
  --policy.se3_enable=true \
  --policy.se3_twist_head_mode=pose9_endpoint \
  --policy.pose9_action_noise_enable=false \
  --policy.se3_noise_trans_scale=0.001 \
  --policy.se3_noise_rot_scale=0.005 \
  --policy.se3_noise_gripper_scale=0.0005 \
  --policy.worldflow_enable=true \
  --policy.worldflow_noise_coupling=conjugate_ego \
  --policy.flow_time_sampling=integration_grid \
  --policy.flow_time_zero_probability=0.5 \
  --policy.optimizer_lr=1e-6 \
  --policy.scheduler_decay_lr=5e-7 \
  --policy.scheduler_warmup_steps=50 \
  --policy.scheduler_decay_steps=500
```

task5 最终使用 `000250`，不是机械地选择最后 checkpoint。完整 SE(3) 必须作为 checkpoint config
保存后再评测；不能在已训练 pose9 checkpoint 的评测命令里临时切换，因为评测脚本只接收
`--policy.path`，更重要的是旧网络需要适配新的群路径输入分布。该配置已有单任务 10/10，但真机或
完整四套件仍应先做小规模安全验收。

### 13.2 warm start 与精确 resume

从已有模型权重开始新实验：

```bash
--policy.path=/path/to/checkpoints/last/pretrained_model
```

此时不要同时传 `--policy.type=smolvla`。这是 warm start：加载 policy 权重，但新建优化器和
学习率计划。

精确恢复训练状态应使用 checkpoint 保存的训练配置：

```bash
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --resume=true \
  --config_path=/path/to/checkpoint/train_config.json
```

恢复时优先相信 checkpoint 自带 policy config。再次传 adapter / VLM / PointSeg 参数只用于
显式验证，不应让命令覆盖成与 checkpoint 不同的结构。

### 13.3 两个训练入口的关系

- `train_song_benchmark.py` 是本项目正式入口；
- `src/lerobot/scripts/train_song.py` 使用相同 policy 核心，但包含不同 debug defaults、在线
  prior 点数和诊断代码；
- 在 dataset、cache、seed、batch、processor、policy config 和采样顺序完全相同时，两者前向
  应接近；正式可复现训练不要混用入口。

### 13.4 训练日志

重点监控：

| 指标 | 含义 |
|---|---|
| `loss_action` | 主动作目标；普通模式是 velocity MSE，完整 SE(3) 模式是 twist、endpoint 与夹爪目标的加权和，不是 rollout 成功率 |
| `loss_pointseg_aux` | soft BCE 与平滑项的加权辅助损失 |
| `loss_se3_twist` / `loss_se3_endpoint` | 完整 SE(3) 模式的 twist 监督与群 endpoint 几何误差 |
| `se3_action_trans_err` / `se3_action_rot_err_deg` | Ego endpoint 的米制平移误差与旋转角误差 |
| `loss_worldflow_flow/geo/bridge/equiv` | World twist/pose9 flow、终点几何、World–Ego 桥接与坐标增广等变损失 |
| `worldflow_noise_conjugacy_error` | Ego/World 随机起点是否满足 `G0=C B0 C^-1`；共轭模式应接近浮点零 |
| `worldflow_path_conjugacy_error` | 当前训练时刻的双流状态是否仍共轭；完整 SE(3) 模式应接近浮点零 |
| `pred_foreground_ratio` | 模型前景预测比例 |
| `pseudo_foreground_ratio` | 有效 soft prior 的前景比例 |
| `pseudo_valid_ratio` | 当前 batch 中 prior 有效监督覆盖率 |
| `pointseg_operation_prob_mean` | 平均前景概率 |

训练时 `shuffle=True`。dataset 通常按 task / episode / frame 连续存储，不 shuffle 的前 200 个
batch 可能只覆盖单一阶段或容易样本，因此不能把该 loss 与训练日志直接比较。

---

## 14. 只读离线评测

`src/lerobot/scripts/eval_song.py`：

- 使用 `model.eval()` 与 `torch.inference_mode()`；
- 不创建 optimizer，不执行 backward；
- 评测前后校验参数未改变；
- 输出逐 batch 指标与 `eval_metrics.json`。

示例：

```bash
python src/lerobot/scripts/eval_song.py \
  --policy.path=/path/to/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --pointseg_sample_cache_dir=/path/to/song_pointseg_cache_v7 \
  --batch_size=32 \
  --steps=200 \
  --seed=0 \
  --output_dir=/path/to/eval_loss \
  --wandb.enable=false
```

该脚本测量 flow 训练目标，不是闭环成功率。比较 checkpoint 时固定：

- `seed`；
- batch size；
- `steps`；
- dataset 顺序 / shuffle；
- cache；
- policy noise；
- 评测入口。

同一 checkpoint 在 batch=6 和有效 batch=192 下得到不同瞬时 loss 是正常的；如果用完全相同的
batch index、noise 和 \(t\) 仍明显不同，才应检查 processor、checkpoint config 或数值路径。

---

## 15. LIBERO rollout 评测

### 15.1 当前执行链

1. 创建与 task 对应的 LIBERO 环境；
2. 等待自由物体稳定，期间保持机械臂初始 pose；
3. 确保初始夹爪完全张开；
4. 读取 `agentview` RGB / depth，构造点云；
5. 在 overview 相机系加入虚拟夹爪并转换到当前 EEF；
6. adapter checkpoint 按 image feature 自动映射到 `agentview`；
7. 模型生成 UMI EEF-relative action chunk；
8. 还原为 absolute OSC target；
9. 按配置执行若干 chunk step；
10. 记录 success、轨迹、动作和实时 JSONL。

当前 absolute pose 执行器使用模型目标本身，不添加启发式目标超调量。标准 LIBERO baseline 通常
直接输出环境定义的 delta action；本项目使用 absolute target 接口，因此应报告该动作接口与执行
频率，但不应通过人为 overshoot 修复数据 / 控制语义错误。

### 15.2 公平串行评测

除了明确研究控制频率或最大时限的消融外，公平比较应：

- 固定同一 checkpoint、初始 state、env seed 与 policy noise seed；
- 不在失败后强制恢复初态重试；
- 每个初始状态只进行一次连续 rollout；
- 使用 suite 标准最大时限；
- 不启用 dataset-domain 环境或 oracle action；
- 固定 `action-index`、`exec-action-steps` 与 gripper control；
- 报告是否 batch 推理。

推荐基准命令：

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /path/to/checkpoints/last/pretrained_model \
  --suite libero_spatial \
  --all-tasks \
  --episodes 10 \
  --suite-gpu-ids 0 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --policy-noise-seed 0 \
  --env-seed 0 \
  --action-index 0 \
  --exec-action-steps 12 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --control-freq 20 \
  --use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /path/to/eval_serial
```

如果论文基线采用其他 `action-index`、执行步数或 5 Hz，必须作为独立 protocol 明确记录，不能与
20 Hz 结果混写。

### 15.3 四卡多 suite

四张 GPU、四个 suite：

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /path/to/checkpoints/last/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --task-workers 10 \
  --task-worker-backend process \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --inference-batch-wait-ms 5 \
  --action-index 0 \
  --exec-action-steps 12 \
  --control-freq 20 \
  --use-suite-max-steps \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /path/to/eval_parallel
```

`suite-gpu-ids` 的数量应与 suite 数量一致。它会为每个 suite 启动独立 GPU 进程，而不是在一个
Python 进程中做模型并行。

为了最大复现性，保持 `inference-batch-size=1`。共享 batch 推理虽然更省显存，但会改变稀疏点云
排序、CUDA kernel 和随机 flow 的数值路径，接触任务可能被微小轨迹差异放大。若追求速度并接受
轻微非确定性，可以逐步提高到 task worker 数。

`task-workers=10, episode-workers-per-task=1` 表示每个 suite 同时维护 10 个 task 环境。
把 `episode-workers-per-task` 提高到 4 会变成约 40 个 MuJoCo 环境 / suite；四 suite 可达到
160 个环境，42 核 / 120 GB 机器很容易 CPU 过载或被系统以 exit code `-9` 杀死。不要只根据
GPU 显存估算并行上限。

### 15.4 实时结果

评测会实时追加：

```text
evaluation_events.jsonl
```

并在任务、suite 和全局结束时生成 JSON report。进程异常退出时，JSONL 可用于恢复已经完成的
episode 记录。

串行交互模式：

- `n`：当前 episode 记为失败并进入下一个；
- `v`：保存当前动作轨迹 / 点云可视化；
- `r`：触发回滚诊断功能。

并行环境 worker 大于 1 时键盘控制自动关闭。

### 15.5 dataset-domain 与 oracle 诊断

```bash
--dataset-domain-env
--dataset-domain-demo-root /path/to/libero_demos
```

会恢复训练 demonstration 对应的 model XML 和初始 state，用于判断模型在数据域内是否工作。
它不是标准 benchmark。

进一步添加：

```bash
--dataset-domain-oracle-actions
```

会绕过模型，直接重放官方原始 action。它回答的是“当前环境重建和执行器能否重放 GT”，不是模型
成功率。若 oracle 都失败，应先修复数据对齐、controller 目标或 gripper 语义，不应继续通过降低
模型 loss 解决。

---

## 16. 真实机器人与本地推理

### 16.1 标准 observation

真实部署传入：

```python
cur_model_observation = {
    "joint_1": None,
    "joint_2": None,
    "joint_3": None,
    "joint_4": None,
    "joint_5": None,
    "joint_6": None,
    "joint_7": None,
    "gripper_width": gripper_width,
    "overhead": img_overhead_rgb,
    "hand": img_hand_rgb,
    "point_cloud": overhead_cloud_rgb,
    "pose_eular": policy_pose_eular,
}
```

要求：

- `point_cloud` 是 fixed overhead 相机系 XYZRGB；
- `pose_eular` 是同一相机系下的 EEF `[x,y,z,euler...]`；
- 图像为 RGB，不是 OpenCV BGR；
- adapter checkpoint 必须提供 checkpoint 声明的 image feature，真实数据通常是 `overhead`；
- 包装器在相机系加入夹爪，再把点云转换到当前 EEF，最后调用 policy。

如果相机移动，但 `pose_eular` 仍来自旧固定相机标定，点云与 EEF 不再共系，模型必然失效。

### 16.2 pickle OOD / 回放推理

部署侧可直接保存完整 observation：

```python
with open(f"/home/liusong/temp/ood_test_new{sno}.pkl", "wb") as file:
    pickle.dump(cur_model_observation, file)
```

推理：

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path /path/to/checkpoints/last/pretrained_model \
  --obs.path /home/liusong/temp/ood_test_new0.pkl \
  --device cuda
```

仅有 PLY 文件不足以完整测试 adapter checkpoint，因为 PLY 没有与采集同步的 RGB 和语言上下文。
PLY 模式只适合 point-only checkpoint，或作为完整 pickle 观测中的点云替换消融。

### 16.3 模态敏感性分析

推理脚本支持在固定初始 flow noise 下分别削弱 RGB、点云、语言或动作上下文：

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path /path/to/checkpoint \
  --obs.path /path/to/observation.pkl \
  --analyze-modalities \
  --analysis-seed 0
```

该结果衡量模型对某模态的敏感性，不自动等于该模态“提升了动作质量”。没有 GT 或 rollout 时，
只能用于定位 OOD 和依赖关系。

冻结 VLM 仍存在 RGB OOD 风险：新机械臂、光照和背景会改变 image token。点云前景分支可以降低
但不能完全消除此风险，因为 Action Expert 同时读取 VLM prefix。应通过真实 RGB 多样性、图像
增强、模态消融和闭环测试判断，而不是仅观察训练 loss。

### 16.4 当前随机流与 WorldFlow 改动对真机的影响

先给出部署结论：完整 SE(3) `pose9_endpoint` 候选已经在 LIBERO Spatial task 5 的官方
10 episodes 达到 10/10，证明新随机路径可用于闭环；但这不是对真机安全性、相机 OOD 或全部任务
的验收。真机正式运行前仍应先用离线 pickle、限速空载和小范围闭环逐级验证。不要在原本确定性
checkpoint 上只修改推理参数开启 SE(3)；应加载已经按同一 config 训练并保存的 checkpoint。

本轮改动没有改变 HDF5、LeRobot dataset、RGB、XYZRGB 点云或 PointSeg cache 的文件格式。
因此，仅从确定性 identity prior 改为合法 SE(3) 随机 prior，或者把双流噪声改为共轭耦合，
**不要求重新转换真机 dataset，也不要求重新生成 cache**。但 checkpoint 的训练目标发生了变化，
启用新随机 prior 后必须继续训练或重新训练；不能只改推理配置后期待旧权重自动适配。当前
`pose9_endpoint` 能继承旧 pose9 head，因此不需要随机新建输出头，但仍需要适配 SE(3) geodesic
状态分布；task5 候选使用 250 个低学习率更新完成该适配。

真机部署需要同步以下契约：

- checkpoint 必须携带最终生效的 `se3_enable`、`se3_twist_head_mode`、`se3_noise_*`、
  `worldflow_noise_coupling`、`action_chunk_start_offset` 和 `num_steps`，推理时不要改成另一套训练
  未见过的 prior；task5 验收值是 `pose9_endpoint` 与 `0.001 m / 0.005 rad / 0.0005 m`；
- 旧 checkpoint 没有 `worldflow_noise_coupling` 字段时会按 `independent` 加载。只要继续保持
  `se3_enable=false`，state dict 与旧 WorldFlow 路径兼容；切换 `direct_twist` 会新增 Ego/World
  输出头，不能把随机初始化缺失参数当成可用权重；`pose9_endpoint` 不新增输出头，而是复用
  `action_out_proj/world_action_out_proj` 并解析转换为群速度；
- `action_chunk_start_offset=1` 只决定训练标签从 dataset 的 `action[i+1]` 开始。真机仍执行预测
  chunk 的第 0 个 token；当前 `single_inference()` / action queue 已是这一语义，不应再跳过一行；
- `worldflow_enable=true` 时，部署必须提供与 overview 点云处于同一坐标系的当前 EEF pose9。
  当前包装器会从 `pose_eular + gripper_width` 构造 raw pose9：Ego state 仍置为 identity，
  但 `worldflow.current_ee_pose` 保留 raw 当前位姿，用来完成 Ego 点云到 World 点云的坐标变换和
  随机初值的共轭变换；
- 固定 overview 真机模式下，可继续把 overview 相机坐标系定义为 model world，不需要额外机器人
  base 外参；前提是点云和 `pose_eular` 都已表达在这个同一相机系。二者一旦不共系，WorldFlow
  的输入和 (G_0=C B_0 C^{-1}) 都会同时出错；
- 合法随机 prior 会使不同随机种子的 action chunk 略有差异。闭环复现实验必须固定 PyTorch/CUDA
  seed；正式部署可采样，但不能在同一个执行 chunk 中途因无关逻辑反复清空队列并重采样。当前
  10/10 证据只覆盖 `policy_noise_seed=0`，不同 seed 仍需单独验证；
- RGB adapter 输入、UMI Ego action 到 World 绝对目标的变换、夹爪点云加入方式和最后的
  `pose9 -> xyz/euler_zyx` 控制接口均未由本轮改动改变。

`deploy_savg_dp3_Desk_General.py` 当前导入的是
`src/lerobot/scripts/smolvla_model_inference.py`。该包装器已经能在 WorldFlow checkpoint 下从 raw
state 自动构造 `worldflow.current_ee_pose`，核心 policy 也会自动从同一个 Ego noise 推导 World
noise。benchmark 推理包装器也已修正：`conjugate_ego` 时不再额外传一份独立 World noise。
若要复现 LIBERO 的固定随机结果，仍需在部署侧显式固定每次重规划的 seed；当前 BestMan
脚本默认使用进程 RNG，并在每次重规划时清空 action queue；因此重启、重规划次数或
触发时刻变化时，随机版轨迹不保证逐位一致。确定性 identity prior 不受这一点影响。

当前两份推理包装器的夹爪点云职责不同，真机切换导入路径时不能混用：

- BestMan 在调用 `src/lerobot/scripts/smolvla_model_inference.py` 之前，已经手动把 500 个夹爪点
  与 49,500 个场景点合并；`src` 包装器只将整云从 overview/model-world 系变到
  current-Ego 系，不会再加一次夹爪。
- `benchmarks/song_real_libero/scripts/smolvla_model_inference.py` 则可在 `single_inference()`
  内部添加夹爪点。如果真机改为导入这一份，必须删除外部重复添加，或显式传
  `add_gripper_cloud=False`；否则点数、夹爪比例和 PointSeg 输入分布都会改变。

从代码变更范围看，本轮直接修改了核心 policy/config、训练/离线评测指标、LIBERO World-token
消融和 benchmark 推理噪声构造；没有修改 BestMan 控制器、相机采集、真机 HDF5/dataset 转换、
PointSeg cache 格式或 `src` 真机包装器的点云/动作坐标变换。因此，真机受到的主要影响是“加载
新 checkpoint 后的随机动作生成规约变化”，而不是传感器或控制接口变化。
`worldflow_enable=false` 时不要求 raw World pose；如果同时保持原 checkpoint 的 noise 配置，本轮
WorldFlow 共轭代码不会改变该真机路径。

具体到真机执行链，受影响与不受影响项如下：

| 环节 | 是否受影响 | 说明 |
| --- | --- | --- |
| RGB-D/IMU 采集、HDF5 字段 | 否 | 没有修改 `record_bestman_rgbd.py` 或真机原始格式 |
| HDF5 -> LeRobot dataset、PointSeg cache | 否 | 不因切换随机流重建；只有源 dataset/先验算法变化时才重建 |
| overview 点云、虚拟夹爪、UMI Ego 变换 | 否 | 输入仍是当前 EEF 系 XYZRGB；不能重复添加夹爪点 |
| `worldflow.current_ee_pose` | **是，WorldFlow 必需** | 必须保留 raw pose9，且与 overview/model-world 点云同坐标系 |
| policy checkpoint 解析 | **是** | 必须识别 `pose9_endpoint`、非零 `se3_noise_*` 和 `conjugate_ego` |
| action chunk 随机性 | **是** | 每次重规划会采 SE(3) 起点；固定 seed 才能逐次复现 |
| 输出动作格式 | 否 | 仍返回 `xyz + rotation-6D + gripper`，上层仍可转 `xyz/euler_zyx` |
| action queue / token 索引 | 契约不变 | offset 1 checkpoint 在线仍执行 token 0，不可二次跳到 action 1 |
| 机器人控制器、限位与碰撞安全 | 否 | 本轮未修改，随机 checkpoint 必须沿用原安全约束并先限速验证 |

---

## 17. 可视化与诊断顺序

出现“loss 很低但 rollout 很差”时，按以下顺序检查，避免先改模型结构。

### 17.1 数据正确性

1. dataset 视频是否与官方 demo 或真实采集一致；
2. LIBERO 是否逐 demo 恢复 `model_file`；
3. `state_observation_offset` 是否为 1；
4. action 是否为 controller target，而不是 achieved state；
5. 点云、EEF pose、虚拟夹爪是否在同一坐标系；
6. RGB camera 与 checkpoint image feature 是否匹配。

### 17.2 cache 正确性

1. cache 是否由当前 dataset 生成；
2. episode/frame mapping 是否一致；
3. p25/p50/p75/terminal 的杯子、杯把、杯架软概率是否连续；
4. pseudo PLY 是否包含原点云并正确着色；
5. PointSeg pred 与 pseudo 的差异是欠拟合还是标签本身缺失。

### 17.3 模型前向

1. checkpoint config 是否启用预期 adapter / PointSeg / fusion；
2. 同一 batch、同一 noise、同一 \(t\) 下两个入口 loss 是否一致；
3. action pad mask 是否正确；
4. 3,072 点与全景点是否按预期选点；
5. modality ablation 是否显示异常 RGB 或点云依赖；
6. rotation-6D 是否先解码再比较。

### 17.4 执行器

1. oracle raw action 能否完成 dataset-domain demo；
2. absolute target 是否按正确 controller frame 执行；
3. gripper width、qpos 和 delta-width 的符号 / 尺度是否一致；
4. control frequency、action index、chunk 执行步数是否与基线一致；
5. 是否存在未报告的 hold、release、adaptive steps 或失败重试。

### 17.5 最后才评估模型能力

只有数据、cache 和 oracle 执行都正确后，才根据以下现象调整模型：

- PointSeg 在终止阶段丢失目标；
- RGB OOD；
- XYZ feature 对视角 / 原点过敏；
- 动作只记住固定轨迹，未随物体姿态旋转；
- flow sampling 方差过大；
- chunk 开环执行过长。

---

## 18. 兼容性与版本迁移

| 来源 | 是否可直接用于当前版本 | 处理 |
|---|---|---|
| 修复前 LIBERO dataset | 否 | 用逐 demo model + offset 1 的脚本重建 |
| 修复前 dataset 对应 cache | 否 | dataset 重建后重新生成 |
| adapter 版真实 dataset，点云和 frame 未变化 | 条件兼容 | 核对 image key、点顺序、episode mapping；不确定时重建 cache |
| cache v7 + 完全相同 dataset | 是 | 使用 strict 检查 |
| v0.2 / v0.3 checkpoint | 条件兼容 | 以 checkpoint config 构造模型，不能强行开启新 fusion |
| v0.5 auxiliary / 更早 WorldFlow checkpoint + 当前 `worldflow_enable=true` | 否 | v2 改为共享专家和双分支推理；需重新训练 |
| point-only checkpoint + PLY | 是 | 不需要 RGB adapter |
| adapter checkpoint + 只有 PLY | 否 | 还需同步 RGB、task、EEF pose |

版本迁移的原则不是“shape 能对上就兼容”，而是数据与 token 的物理语义必须一致。

---

## 19. 常见问题

### 为什么 PointSeg 预测 3,072 个前景点仍可能漏掉杯架？

数量正确不代表组成正确。模型可能把高概率都分配给杯子、夹爪或近相机机械臂。应比较终止阶段
pseudo 与 pred 的空间组成，而不是只看前景点总数。

### 为什么机械臂靠近相机时更容易前景错误？

近距离机械臂占据更多点和更大 RGB 区域，XYZRGB 特征分布也发生变化。若训练中此类视角不足，
PointSeg 会把高显著性机械臂点当作操作前景。虚拟夹爪和真实机械臂外观不一致也会放大问题。

### 为什么训练 loss 约 0.0004，shuffle 离线评测却到 0.002？

训练日志是不同随机 batch、noise 与 \(t\) 上的瞬时 flow velocity MSE；未 shuffle 的连续切片可能
只覆盖很容易的 episode 阶段。先固定样本索引、noise、\(t\)、processor 和 checkpoint config，
再做逐样本比较。不能把 rollout action endpoint MSE 与训练 flow loss混为一谈。

### 为什么模型第一条 rotation-6D 不再是单位旋转？

当前原点是实际 `observation.state`，第一条 action 是相对于实际 EEF 的控制器目标。它本来可以
包含位移和旋转。旧版以 `action[0]` 为原点才会人为得到单位首动作。

### 为什么 gripper 设置更大的 `gripper_len` 仍解决不了开柜失败？

`gripper_len` 只改变虚拟点云几何，不修复 controller target、state/action 对齐、夹爪控制符号和
接触执行误差。若 oracle 重放都打不开，优先检查数据与执行器。

### 标准 LIBERO 评测是否需要人为超调 target？

不需要。标准评测执行策略动作。对于本项目 absolute target 接口，应正确重建并执行目标，而不是
添加经验偏置来掩盖转换错误。

### TensorFlow / TF-TRT warning 是否影响转换？

通常不影响。它们来自间接依赖初始化，只要实际 PyTorch / MuJoCo 流程正常即可。

### `exitcode=-9` 是什么？

通常是操作系统 OOM killer 或调度器强制杀进程，不是 Python/CUDA 可捕获异常。降低
`num_workers`、环境进程、episode 并行或 batch size，并查看系统日志与主机 RAM。

### EGL 下 Open3D 报 GLX `BadAccess` 怎么办？

MuJoCo EGL 与 Open3D GUI/GLX 可能冲突。服务器上保存 PLY/NPZ 后离线查看，不要在同一进程调用
`draw_geometries`。无 EGL 时可用 OSMesa CPU 离屏渲染，但会更慢。

### output directory 只有 wandb 仍报“已存在”怎么办？

正式训练使用新的 output path，或确认入口允许仅含 `wandb/` 的目录。不要并行启动多个 rank 前
由每个进程竞争创建 / 检查同一目录；应由 accelerate 主进程统一初始化。

---

## 20. 环境与最小校验

推荐 editable install：

```bash
pip install -e ".[smolvla,libero]"
```

若未安装：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

LIBERO 离屏：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

提交训练前至少执行：

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/smolvla/smolvlm_with_expert.py \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py
```

并保存以下 provenance：

```bash
git rev-parse HEAD
git status --short
sha256sum /path/to/checkpoint/model.safetensors
```

每个实验报告至少记录：

- branch / commit；
- dataset summary 和 cache manifest；
- checkpoint hash；
- 全部 policy flags；
- GPU 数与有效 batch；
- eval seed、频率、时限、action index、chunk 执行步数；
- 串行 / batch / 独立模型推理方式；
- 是否启用任何 dataset-domain 或 oracle 诊断。

---

## 21. 代码地图

### 模型

- [`configuration_smolvla.py`](../../src/lerobot/policies/smolvla/configuration_smolvla.py)：
  policy 配置、稳定 / 实验开关与兼容参数。
- [`modeling_smolvla.py`](../../src/lerobot/policies/smolvla/modeling_smolvla.py)：
  PointSeg conditioner、LitePT、Point–Action fusion、WorldFlow、flow matching。
- [`smolvlm_with_expert.py`](../../src/lerobot/policies/smolvla/smolvlm_with_expert.py)：
  SmolVLM / Action Expert、VLM 权重解析、RoPE 和联合注意力。
- [`song_pointseg.py`](../../src/lerobot/policies/smolvla/song_pointseg.py)：
  双向轨迹软先验与 PointSeg 网络。
- [`umi_processor.py`](../../src/lerobot/processor/umi_processor.py)：
  当前 EEF 坐标系下的 state/action 变换。

### 数据和 cache

- [`libero_hdf5_to_dataset.py`](scripts/libero_setting/libero_hdf5_to_dataset.py)：
  LIBERO model restore、帧对齐、controller target 重建、点云 / RGB 保存。
- [`real_hdf5_to_dataset.py`](scripts/real_setting/real_hdf5_to_dataset.py)：
  真实 XYZRGB HDF5 转换。
- [`song_cache_pointseg_samples.py`](scripts/song_cache_pointseg_samples.py)：
  多 GPU index-only cache v7。
- [`libero_pointcloud_utils.py`](scripts/libero_setting/libero_pointcloud_utils.py)：
  深度反投影、相机 / EEF 变换和虚拟夹爪。

### 训练和评测

- [`train_song_benchmark.py`](scripts/train_song_benchmark.py)：项目标准训练。
- [`eval_song.py`](../../src/lerobot/scripts/eval_song.py)：无梯度离线评测。
- [`libero_pointcloud_eval.py`](scripts/libero_setting/libero_pointcloud_eval.py)：
  串行 / 多 GPU rollout、数据域和 oracle 诊断。
- [`smolvla_model_inference.py`](scripts/smolvla_model_inference.py)：
  dataset、pickle、PLY 与模态分析。

---

## 22. 参考

- SmolVLA: [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)
- LIBERO: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
- Flow Matching: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Rotation 6D: [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)
- PointACT: [PointACT](https://arxiv.org/abs/2605.21414)

---

## 23. 后续开发检查表

每次恢复本项目时，先回答以下问题：

1. 当前 branch / commit 是否仍与文档基线一致？
2. dataset 是否由逐 demo `model_file`、offset 1、controller target 版本生成？
3. cache 是否由这份 dataset 重新生成？
4. checkpoint 内 adapter、PointSeg、Point–Action fusion 开关是什么？
5. WorldFlow/SE(3) 是否与 checkpoint 一致；若使用随机版，是否为 `pose9_endpoint`、
   `conjugate_ego`、非零物理尺度并在目标任务上完成闭环验收？
6. 训练与推理的 RGB key、点云坐标系和 EEF pose 是否一致？
7. 比较 loss 时是否固定了样本、noise、\(t\) 和 processor？
8. rollout 失败前，oracle action 能否通过同一执行器？
9. 评测协议是否明确频率、时限、chunk、gripper 和并行方式？
10. 新实验是否保存了 dataset/cache/checkpoint 的可追溯信息？

只要上述契约保持一致，模型架构、数据处理和评测结果才具有可比性。
