# WEP-VLA / Song Real-LIBERO 项目交接文档

更新时间：2026-08-14

## 当前强制覆盖项（2026-08-14）

- 此后所有 A800 测评只使用物理 GPU `3,4,5`；不得再按历史 6 卡配置启动新测评。
- A800 测评总环境并行上限为 `40`，推理 batch 固定为 `40`。
- LIBERO-10 的当前通用划分为：GPU 3 运行 tasks `0,3,6,9`，GPU 4 运行 tasks `1,4,7`，GPU 5 运行 tasks `2,5,8`；每个 task 使用 4 个 episode worker，总并行为 `16+12+12=40`。
- 当前入口脚本为实验根目录中的 `scripts/eval_libero10_v16_one_checkpoint_a800_gpu345_total40_b40_fixedbarrier.sh`；多 checkpoint 的 tmux 入口为 `scripts/launch_eval_libero10_v16_steps100_240_480_720_a800_gpu345_total40_b40_tmux.sh`。
- 同时筛选四个 checkpoint 时，使用实验根目录中的 `scripts/launch_eval_libero10_v17_steps100_240_480_720_a800_gpu345_concurrent4_total40_b40_tmux.sh`。该模式让 `100/240/480/720` 各使用 10 个环境并发，总并发仍为 40，推理 batch 参数仍为 40；不得与单 checkpoint 独占 40 环境的串行入口同时运行。
- v17 已完成一整个 `10 tasks x 50 episodes` 训练 epoch。本地 first-10 screen 的 `100/240/480/720` 为 `90/92/94/96`，但 step 720 的正式 500-state 流在 `267/286` 时出现第 19 个失败，最高只可能追平 `481/500`，已数学早停并保留全部输出。覆盖完整 episode 区间的固定分层集合 `0,5,...,45` 上，正常 WorldFlow 为 `92/100`，同 checkpoint 关闭 World-to-Ego 为 `96/100`，因果增益是 `-4`；此前 first-10 的 `+3` 是切片偏差，v17 已正式否决。
- v19 已从 exact-zero-residual、baseline-compatible 的 v14 起点启动，仅增加训练期 World-to-Ego stochastic depth `p=0.5`。它不增加参数、门控、辅助损失或推理分支；纯 Ego 和完整 World/Ego 样本共用原 action loss，且所有 Ego/World/残差参数保持非零学习率。训练使用 4 GPU、batch 48/GPU、worker 12、720 steps、W&B enabled，tmux 为 `wep_v043_libero10_v19_w2e_stochastic_depth_p50_720`。固定分层评测 waiter 为 `wep_v043_v19_stratified_worldflow_causal_after_train`，训练完成后自动评估 steps `100/240/480/720`，仅对超过同状态 baseline `95/100` 的 checkpoint 运行同 checkpoint 因果消融。
- v19 已完成并筛除。固定分层 episode IDs `0,5,...,45` 上，steps `100/240/480/720` 分别为 `92/93/95/92`，同 IDs baseline 为 `95/100`；没有 checkpoint 严格超过 baseline，因此 full-500 自动取消。补做的最佳 step480 同-checkpoint World-to-Ego 消融仍为 `95/100`，World 净因果增益为 `0`：World 正常臂独有成功 `(5,35),(6,25),(6,30)`，Ego-only 独有成功 `(1,45),(3,25),(9,30)`，共同失败 `(7,20),(9,25)`。v19 将 v17 的 broad World 因果效应从 `-4` 改善到 `0`，但尚未产生正增益。
- v20 residual boosting gradient isolation 已完成 720 步、138,240 个样本，即完整 `10 tasks x 50 episodes` 数据的一整个 epoch。它不增加参数、门控、辅助损失或推理分支；代码 commit 为 `fe99a71`，完整测试 `67/67` 通过，W&B 的 720 个 keep-ratio 均值为 `0.500752`。固定 broad steps `100/240/480/720` 为 `94/96/91/92`；step240 同-checkpoint 关闭 World-to-Ego 为 `93`，故 World 因果增益为 `+3`，是唯一候选。但正式 full-500 正常臂在 `259/279` 时已有 20 个失败，即便剩余 221 个全成功也最多 `480/500`，无法超过 `481/500` baseline，已数学早停。所有输出和 cache 均保留，matched baseline 与 full causal arm 未启动。v20 改善了 World 因果性但未达到最终目标，不能恢复多视角联合阶段。
- v21 已完成一整个 epoch。最终 runtime audit：720 条 W&B history 完整，World-correction keep ratio 均值 `0.245978`，三个 optimizer group 均保持非零 LR，78 个 World BN buffer 与 v14 逐位一致，4,326 个 residual 参数全部非零。固定分层 Broad 的 steps `100/240/480/720` 为 `95/89/95/97`；step720 同 checkpoint 关闭 World-to-Ego 为 `95`，所以 Broad World 因果增益为 `+2`。但正式 Full-500 在停止时为 `411/435`、24 个失败，即使余下65个全部成功也最多 `476/500`，无法超过 baseline `481/500`；v21 已正式否决，所有 partial 输出保留。
- v22 将 v21 step720 的547个 World-only参数保留、把1461个 baseline-common 参数逐位恢复，固定分层仅 `94/100`。这否定了“硬重置 shared/Ego 后直接拼接 World”的方案：Ego/shared 与 World 已产生共适应，不能硬拆。v23 对 v14→v21 的整套联合权重 delta 做统一 `alpha=0.50/0.75` 缩放，结果分别为 `95/100` 和 `94/100`，同样筛除；不再继续做 checkpoint 插值网格。
- v24 是当前 active 实验，用复制出的完整 v21 step720 模型/optimizer/scheduler/RNG 状态精确续训至 step1440。调度器仍保留原 `num_decay_steps=720`，所以 step721 起保持已有 floor LR，不重新 warmup或升高 LR；模型、数据、原 action loss、p=0.75 gradient routing、无门控/无 World 辅助损失等均不变。4-GPU、batch48/GPU、worker12、W&B enabled，tmux 为 `wep_v043_libero10_v24_exact_resume_v21step720_floorlr_to1440`；step721 已实际连续运行。Broad waiter `wep_v043_v24_floorlr_stratified_causal_after_train` 将筛 `960/1200/1440`，formal waiter `wep_v043_v24_full500_matched_causal_after_screen` 只在 absolute+causal Broad gate 通过后启动，并使用可强制终止的 Full-500 数学早停。
- 文档后文出现的 6-A800、80 workers 或 batch 80 均是历史实验记录，仅用于追溯结果，不再代表当前执行配置。

本文档用于在新 Codex 会话、换人维护或长时间中断后快速恢复项目上下文。内容区分为：

- 已由代码或实验确认的结论；
- 当前工作区状态；
- 尚未完成、不能当作结论的事项；
- 下一位执行者应立即开展的工作。

## 1. 项目背景与目标

项目基于 LeRobot/SmolVLA，主要研究以下内容：

1. 使用 RGB、语言、机器人状态和 3D 点云生成机器人动作 chunk；
2. 使用基于轨迹运动先验的软前景标签训练 PointSeg，从场景点云中提取操作相关物体；
3. 保持官方 SmolVLA/SmolVLM 的 VLM 结构，支持载入官方预训练权重；
4. 同时支持 LIBERO 仿真数据与真实 RGB-D/人手示教数据；
5. 探索 World/Ego、SE(3) 等变动作生成等辅助结构，但这些结构默认关闭；
6. 最终希望提高视角变化、物体位置变化、操作位姿变化和异构末端条件下的泛化能力。

整体数据流如下：

```text
LIBERO HDF5
  -> 恢复每条 demo 的 model_file 并重放 state
  -> RGB / overhead 点云 / 绝对末端动作 LeRobotDataset
  -> PointSeg 轨迹先验 cache
  -> WEP-VLA 训练
  -> 官方 LIBERO episode 评测

真实 RGB-D + 相机轨迹
  -> WiLoR/MANO 手部重建
  -> RGB-D 全局度量对齐
  -> handpose_wilor.jsonl
  -> 按独立 recording segment 切分 HDF5 episode
  -> LeRobotDataset / cache / 训练 / 真机推理
```

## 2. 当前代码快照

### 2.1 LeRobot 主仓库

- 路径：`/home/liusong/ProgramFiles/Huggingface/lerobot`
- 分支：`wep_vla_v0.4.3_multiview_doubleflow`
- 本轮 v24 启动前 HEAD：`93d7f85`；World residual gradient isolation 的代码 commit 为 `fe99a71`。
- 当前实验脚本、checkpoint、日志和 artifact 统一位于交接文档顶部所述 experiment root；不得删除 `/opt/data/private` 下任何文件。
- 临时实验统一使用 `/tmp/temp`，不要把临时诊断文件写入正式数据目录。

### 2.2 HandPoseExtraction 外部仓库

- 路径：`/home/liusong/ProgramFiles/HandPoseExtraction`
- 分支：`main`
- HEAD：`6f2fdc52427380d0e9bd51efc4b86f979c157663`
- 该仓库当前存在未提交改动，包括 RGB-D/MANO v5 对齐、无时序滤波 pipeline、gripper 构造和测试。
- `handpose_extraction/smoothing.py` 当前处于删除状态。
- 不允许整体 reset、checkout 或覆盖这些改动；必须先逐文件审查。
- LeRobot 工作区权限不一定允许直接写外部仓库。下一会话若要修复 HandPoseExtraction，需要显式取得该目录写权限。

## 3. 已确认结论

### 3.1 模型与 VLM adapter

- 官方 SmolVLA/SmolVLM 的视觉语言骨干结构必须保持不变，才能正确载入官方大规模图像/语言预训练权重。
- `vla_adapter_enable=true` 时，训练和推理 batch 必须包含 RGB 图像。
- `vlm_model_name` 用于确定模型架构和 processor；`vlm_weights_path` 可指向离线 SmolVLA/VLM 权重来源。
- adapter 模式通常使用：
  - `vla_adapter_enable=true`
  - `vla_adapter_freeze_vlm=true`
  - `load_vlm_weights=true`
- 冻结 VLM 不等于冻结整个策略。PointSeg、点云编码器、point-action fusion 和 Action Expert 仍可训练。
- 当前配置中 `point_action_fusion_enable=true`，动作 token 会在 Action Expert 之前与点 token 交互。
- `worldflow_enable`、`worldflow_se3_head_enable`、`se3_enable` 和 `se3_final_correction_enable` 默认均为 `false`。
- 当前常规基线在关闭 WorldFlow/SE3 时运行；不能把实验性辅助分支默认当作稳定增益。

关键文件：

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `benchmarks/song_real_libero/scripts/train_song_benchmark.py`
- `benchmarks/song_real_libero/scripts/smolvla_model_inference.py`

### 3.2 LIBERO 数据转换

已经确认旧数据转换的核心问题不是简单的“同一个场景位置被复用”，而是每条官方 demo 都携带独立 `model_file`。如果转换时忽略它，即使 flattened simulator state 正确，柜体、把手和可移动/可配置 fixture 仍可能与原 demo 存在毫米到厘米级偏差。

当前正确约定：

- `state_observation_offset=0`；
- 输出 observation 和 source action 使用同一个 source index；
- `restore_demo_model=true`；
- `require_source_fps_match=true`；
- overhead/agentview RGB 和点云必须来自恢复后的正确 demo 环境；
- action 最终转换为训练使用的绝对末端表示；
- `action_index=0` 是正式配置，历史上的 `action_index=1` 只用于定位旧时序/执行器问题。

配置位置：

- `benchmarks/song_real_libero/configs/libero.json`
- `benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py`

当前 LIBERO 数据转换与训练/评测命令记录在：

- `benchmarks/song_real_libero/WEPVAL_V042_GeneralDataset_ToolSeg.md`
- `benchmarks/song_real_libero/WEP_VLA_041_DATASET_FIXED--WEPVAL_V042.md`

### 3.3 LIBERO 评测

- 默认评测必须使用官方 LIBERO 测试 episode，而不是训练数据集中的 demo episode。
- `strict_official_init=true` 是标准 benchmark 路径。
- `dataset_domain_env` 与 `dataset_domain_oracle_actions` 仅用于诊断，结果不能作为标准 benchmark 成绩。
- 当前正式 `action_index=0`。
- 夹爪推荐配置为 `delta_width_initial_sync`，只在 episode 初始同步一次内部 gripper target，之后按 chunk 内相对宽度变化执行。
- `gripper_delta_alignment=current_minus_previous` 符合轨迹时间方向；`next_minus_current` 是旧的一行提前行为。
- 标准测评不能失败后强制恢复初始状态重试。
- 并行评测时每个隔离 policy worker 应拥有独立模型和噪声状态；共享 batch 推理可能改变 flow-matching 随机轨迹及调度时序。

关键文件：

- `benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py`
- `benchmarks/song_real_libero/configs/libero.json`

### 3.4 PointSeg 与 cache

- PointSeg 目标是前景/背景软二分类，不应依赖人工语义分割、人工指定目标点或真实点流。
- 轨迹先验应覆盖：
  - 末端附近与末端共同运动的工具/被抓物体；
  - 末端逐渐靠近的交互目标；
  - 接触末期虽然运动较小、但仍位于 gripper 附近的相关物体。
- 离线 cache 与无 cache 的在线 prior 计算必须使用同一套算法。
- cache 必须同时覆盖增广点云与原始非增广全场景点云，不能用固定点数阈值暗中判断数据类型。
- `current-points`/`future-points` 小于目标数时保留全部有效点，后续由 mask 处理变长输入。
- 当前 ToolSeg/PointSeg 的生产命令以 `WEPVAL_V042_GeneralDataset_ToolSeg.md` 为准；修改参数前先核对 checkpoint 的 `train_config.json`。

关键文件：

- `benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py`
- `benchmarks/song_real_libero/scripts/train_song_benchmark.py`
- `src/lerobot/policies/smolvla/song_pointseg.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`

### 3.5 真实 RGB-D 与 MANO mesh 对齐

当前真实数据目录：

```text
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/
data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking2
```

已确认：

- RealSense depth 已对齐到 color，使用的是对齐后 color camera intrinsics；
- Brown-Conrady 投影/反投影数值与 `pyrealsense2` 一致，误差不是主要来源；
- JSONL 的 `record_index` 与 RGB-D frame index 一致；
- mesh 旧版明显错位的主要原因是用投影顶点的 signed depth median 平移完整 MANO，容易被手中方块、指缝背景和局部近层误导；
- v5 使用检测框内 RGB-D 点云与完整 MANO 表面的粗到细 Chamfer 分位数搜索；
- v5 只允许一个全局平移同时作用于所有 joints/vertices，不进行逐点吸附、逐关节替换或时序轨迹滤波；
- 每个 recording segment 是独立 episode 的原始录制，任何时序状态都必须在 segment 边界 reset；
- 深度有效下界为 0.30 m。

当前正式 JSONL：

```text
.../Dynamic_Ego_CubeStacking2/handpose_wilor.jsonl
```

其状态：

- 256/256 帧版本为 `wilor_mano_mesh_rgbd_chamfer_v5`；
- 每帧都有一个有效 hand；
- 每帧都使用 `mano_mesh_rgbd_chamfer_translation`，没有降级；
- 旧 v4 备份为 `handpose_wilor.v4_global_median_backup.jsonl`；
- v5 mesh 比 v4 更符合真实深度，但在遮挡严重处仍存在 WiLoR 单目 MANO 形状/姿态误差；
- 已完成 115 帧 HDF5 smoke test，位置连续性阈值 0.05 m 下无断点；
- smoke test 输出在 `/tmp/temp/hdf5_v5_smoke_20260809`，不是生产训练数据。

关键文件：

- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/adapters/wilor.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/depth_fusion.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/gripper.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/pipeline.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py`
- `benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py`
- `benchmarks/song_real_libero/scripts/real_setting/README_CAMERA_MOTION.md`

## 4. 当前最高优先级未完成事项：episode 内旋转突变

### 4.1 已完成的初步定位

对当前 v5 `handpose_wilor.jsonl` 按独立 recording segment 使用旋转矩阵 geodesic angle 检查，结果为：

| Segment | 帧数 | 相邻旋转中位数 | P95 | 最大值 |
|---|---:|---:|---:|---:|
| 0 | 115 | 2.19° | 11.51° | 32.87° |
| 1 | 141 | 1.99° | 28.36° | 32.73° |

典型异常：

- frame `26 -> 27`：旋转约 31.09°，但 21 个 MANO joints 的中位移动仅约 1.59 mm；
- frame `117 -> 118`：旋转约 32.73°，joints 中位移动仅约 2.63 mm；
- frame `146 -> 147`：旋转约 29.81°，joints 中位移动仅约 2.79 mm。

因此可以确认：

- 旋转跳变真实存在于 HDF5 生成之前的 JSONL rotation matrix，不只是 Euler 的 `±π` 显示跳变；
- 跳变与 mesh 全局平移、相机外参或 ORB-SLAM 轨迹不是同一个问题；
- `pose_frame=camera` 时 HDF5 不应用相机外参，所以该路径不会制造上述原始旋转突变；
- `build_humanhand_hdf5_dataset.py` 的逐帧 `as_euler("zyx")` 可能另外引入 Euler 分支跳变，但不是当前 30° geodesic 跳变的根因。

### 4.2 当前最可能根因

`HandPosePipeline` 当前明确是 stateless：

```python
previous_rotation_camera_gripper=None
```

而 `pinch_gripper_from_hand` 使用多个 thumb/index finger segment 构造加权 forward/lateral axis。手指接近共线、抓住方块或 WiLoR 某个关节点轻微抖动时，候选轴的投影和权重关系会突然变化；由于没有 segment 内的方向对应关系，相邻帧可能选择不同坐标轴方向，造成 20–33° 的姿态跳变。

这不是要通过重度 EMA 掩盖的问题。正确修复应优先解决坐标系构造的唯一性和退化条件，然后才考虑是否需要非常轻量的连续性约束。

### 4.3 尚未完成的工作

1. 尚未把上述异常帧做 RGB、深度、mesh、joints、gripper axes 的并排可视化报告；
2. 尚未逐项分解 `x_up/y_right/z_forward` 哪个轴贡献了跳变；
3. 尚未实现 segment-scoped orientation correspondence；
4. 尚未添加 noisy-joint/degenerate-pinch 的回归测试；
5. 尚未重新生成修复后的完整 JSONL；
6. 尚未重新生成最终生产 HDF5、LeRobotDataset 和 cache。

## 5. 下一步行动

### P0：修复旋转突变

按以下顺序执行：

1. 在 `/tmp/temp` 写只读诊断脚本，列出每个 segment 的旋转 geodesic jump；
2. 对 top-k 异常 transition 保存：
   - 前后两帧 RGB；
   - MANO mesh/joints；
   - `x_up/y_right/z_forward`；
   - 每个候选 axis 及其权重；
   - handedness、bbox、depth correction source；
3. 判断跳变来自：
   - forward candidate 切换；
   - lateral axis 退化；
   - palm normal 符号变化；
   - WiLoR articulation 本身突变；
4. 优先采用通用、非滤波式修复：
   - 使用稳定 wrist/MCP palm frame 作为主姿态；
   - thumb/index 只决定 opening 和局部夹取方向，不让退化指尖主导完整姿态；
   - 对等价坐标轴表示做 segment 内最小 geodesic 对应；
   - 只有在当前帧几何退化时才使用上一帧方向消歧，不做位置/旋转 EMA；
   - segment 开始必须 reset；
5. 先用当前 256 帧离线重算并比较修复前后；
6. 目标验收建议：
   - 没有视觉真实大转动时，相邻旋转不应出现 20–33° 单帧跳变；
   - 修复不能改变 mesh/joints 的 3D 几何或全局位置；
   - segment 边界不纳入连续性约束；
7. 最后处理 Euler 存储：保留 quaternion/matrix 为真值；如果下游必须使用 Euler，则按 episode 做连续 branch unwrap，但不能用 unwrap 掩盖真实 matrix jump。

### P1：重新生成真实训练数据

P0 完成后：

1. 不使用 `--reuse-jsonl` 重新生成 `handpose_wilor.jsonl`；
2. 验证 pipeline version、frame count、无缺失 hand；
3. 按 `segments.txt` 生成最终 HDF5；
4. 检查 position、rotation geodesic、opening width、Euler branch 和相机轨迹；
5. 再生成 LeRobotDataset 与 PointSeg cache；
6. 旧 dataset/cache 不应与新 HDF5 混用。

### P2：剩余工程事项

- 解决当前环境 SciPy 对 NumPy `1.23.4` 的版本警告，建议升级到满足 SciPy 要求的 `>=1.23.5`，但变更环境前应锁定依赖并重跑测试；
- 将 HandPoseExtraction 当前未提交改动整理为小粒度 commit；
- 补充完整真实数据质量报告，不只检查位置 jump；
- 对 WorldFlow/SE3 分支继续保持默认关闭，除非单独消融实验确认收益。

## 6. 常用命令

### 6.1 查看当前 v5 JSONL

```bash
python /home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py \
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --input /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking2 \
  --jsonl /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking2/handpose_wilor.jsonl \
  --reuse-jsonl \
  --show3d
```

注意：`--reuse-jsonl` 只查看已有结果，不运行任何新算法。

### 6.2 重新推理手部姿态

```bash
python /home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py \
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --input "$DYNAMIC_OUTPUT_DIR" \
  --jsonl "$DYNAMIC_OUTPUT_DIR/handpose_wilor.jsonl" \
  --fast \
  --force-handedness right \
  --fusion-mode model-depth \
  --min-depth-m 0.30 \
  --max-depth-m 3.0 \
  --depth-window 5 \
  --min-valid-depth-points 6 \
  --include-mesh
```

不要在旋转修复完成前覆盖唯一结果；先输出到临时 JSONL 比较。

### 6.3 HDF5 连续性检查

```bash
python benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py \
  --hdf5_dir /path/to/hdf5_dir \
  --threshold 0.05
```

该脚本当前主要检查位置；旋转 geodesic 质量检查仍需补充。

### 6.4 回归测试

HandPoseExtraction：

```bash
cd /home/liusong/ProgramFiles/HandPoseExtraction
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/liusong/anaconda3/envs/reap/bin/python -m pytest \
  tests/test_geometry.py tests/test_offline_sequence.py -q
```

LeRobot：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/liusong/anaconda3/envs/reap/bin/python -m pytest \
  tests/scripts/test_real_camera_motion.py -q
```

最近通过结果：HandPose 26 项，LeRobot real-camera-motion 22 项。

## 7. 禁止事项与设计边界

- 不引入人工分割、人工指定目标点或人工光流监督；
- 不使用 PCA 等依赖场景分布、容易退化的几何捷径；
- 不逐点把 MANO mesh 吸附到 RGB-D 表面；
- 不用重度滤波掩盖错误坐标变换或错误姿态定义；
- 不跨 recording segment 继承任何时序状态；
- 不修改第三方 WiLoR/LitePT 源码来绕过本项目适配问题；
- 不整体覆盖或 reset 当前脏的 HandPoseExtraction 工作树；
- 不将 dataset-domain diagnostic 成绩当作官方 LIBERO benchmark 成绩。

## 8. 新会话启动提示词

在新对话中直接发送以下内容：

```text
读取并严格沿用：
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/PROJECT_HANDOFF_2026-08-11.md

不要询问我，直接执行其中 P0“修复 episode 内旋转突变”。先在 /tmp/temp 生成只读诊断，定位当前 Dynamic_Ego_CubeStacking2 中 20–33° 单帧旋转跳变究竟来自 x_up、y_right 还是 z_forward。修复必须通用、segment-scoped，不使用 PCA、人工监督、逐点 mesh 吸附或重度 EMA。修复后重新生成临时 JSONL，与现有 v5 做定量及可视化对比；通过测试后再更新正式文件。不要覆盖或 reset HandPoseExtraction 中其他未提交改动。
```
