# 五来源统一相机坐标系训练（2026-09-10）

代码位于本地 `wep_vla_v0.5.0`，基于 `origin/wep_vla_v0.5.0`
提交 `f74c275f7f72d65c61a02c7f60efb1d39b26a2cc` 增加本次统一数据训练适配。
仅适配训练/缓存读取入口；未修改数据集、原 checkpoint、评测代码或另一个 Molmo2 仓库。

## 运行

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot
bash benchmarks/song_real_libero/scripts/train_merged_camera_v050_7gpu.sh
```

完整环境变量和命令见该脚本。额外参数可追加到脚本末尾覆盖默认值。
这是从 policy 权重开始的新微调实验，不恢复旧优化器、scheduler 或旧训练数据配置。
checkpoint 路径中的 `*` 是目录名的真实字符，必须保留单引号，不能当作 shell 通配符。
该 checkpoint 实际是 SmolVLA/SmolVLM2-500M 双流模型，不是 Molmo2-7B。

## 数据与坐标约定

独立数据根目录：
`/home/liusong/datasets/merged_5datasets_rh20t_singleview_20260909_standalone`

| 来源 | 轨迹 | 帧 |
| --- | ---: | ---: |
| RH20T | 8,301 | 1,509,292 |
| LIBERO | 2,000 | 336,575 |
| RLBench | 1,000 | 155,174 |
| DynamicCube | 572 | 35,662 |
| StageGenMerged | 1,215 | 113,939 |
| 合计 | 13,088 | 2,150,642 |

不包含 StaticFrankaCubeStacking、StaticFrankaMugRack。
RGB 统一为 `observation.images.front`；输入点云仍是当前虚拟末端 ego 坐标。
WorldFlow 的 world 是每条轨迹固定的相机参考系，轴为右/下/前、单位米；
并非五来源共享同一个物理相机位置。`episode_reference_poses` 将首帧末端局部位姿映射到该参考系。
状态/动作是 10 维 pose9 + 真实夹爪宽度，保留 IDENTITY normalization，原 UMI 预处理只执行一次。
末尾 500 个虚拟夹爪点保持 RH20T 形状与真实宽度，不重写真实标签。

`world_ee_poses` 为当前位姿，`action_target_ee_poses` 为目标序列，
两者不能互相代替。DynamicCube/StageGen 的目标仍是明确标注的录制示范位姿序列，
不冒充控制器命令；另外三来源为 commanded_target。
使用 sidecar 的前提是 merge/alignment/provenance 的完整性、参考系、轴和单位校验全部通过。

20 FPS 仅作为已有的整数帧偏移网格使用，不重采样、不声称所有来源实际采集频率相同。
`chunk_size=32`、`action_chunk_start_offset=0` 沿用预训练配置，动作与 WorldFlow 采用相同边界补齐。

## 参数选择

- `camera_views=front`、`rgb_camera_views=front`；自动将 checkpoint 的外部相机特征名 agentview 映射到 front，模型结构/权重不变。
- `worldflow_reference_frame=pointcloud_reference_camera`，`worldflow_target_type=world_eef_trajectory`，必须使用目标 sidecar，禁止 robot_base 误解释。
- `worldflow_bootstrap_from_ego=false`：不覆盖已训练的 WorldFlow 分支。
- 保留原双流结构、冻结设置、loss 权重和 IDENTITY normalization；未重置专家。
- 每卡 batch 16，7 卡，累积 1，有效 batch 112；显存不足时降低每卡 batch 或增加梯度累积。
- 学习率 3e-5，warmup 500，60,000 optimizer steps，衰减到 3e-6：这是保守微调起点，不是已验证的最佳超参数。
- 默认五来源各约 20% 抽样，来源内按帧均匀；与按帧自然占比不同，小来源会重复更多。
  如需自然帧占比，设置 `--source_balanced_sampling=false`。不能同时打开 task_balanced_sampling。
- 不复用旧分来源 PointSeg 缓存。默认在线 CUDA 生成监督，因此 `num_workers=0`；仍有七个训练进程并行。
  在线监督比预生成缓存慢，后续可用适配后的 `song_cache_pointseg_samples.py` 对当前数据重新构建缓存。
  缓存必须带匹配的 `merged_dataset_contract`，不能手工给旧缓存补标记。
- 当前/未来点云保留原生点数并在 batch 内补齐；不额外设置 `SONG_POINTSEG_ONLINE_*_POINTS` 改变点数预算。
- `eval_freq=0`；本次没有同步修改旧 LIBERO eval 的坐标处理，不应直接认为旧评测入口与新模型输入一致。
- 迁移服务器后须修改 dataset/output/checkpoint 路径，并准备 checkpoint 配置引用的本地 VLM/tokenizer 权重目录及 pointops 环境。

## 验证

回归测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/liusong/anaconda3/envs/reap/bin/python -m pytest \
  tests/datasets/test_merged_camera.py tests/scripts/test_train_song_worldflow_dataset.py \
  tests/datasets/test_sampler.py tests/configs/test_train_config.py -q
```

50 项通过，覆盖旧相机/机器人基座逻辑、合并元数据拒绝条件、目标序列/补齐、
RGB 映射、来源采样、旧缓存拒绝与按轨迹读取真实 observation.state。

真实数据验证脚本 `scripts/validate_merged_camera_training.py`：
选择五来源的六条轨迹（额外覆盖 StageGen 的 50k/3072 点），各检查首/中/尾共 18 帧的
anchor-state-current、anchor-action-target 一致性，再对三个混合 batch 执行前向/反向，
`strict=True` 完整加载指定 checkpoint；普通与在线 PointSeg 模式均通过，无优化器更新。
报告：`/home/liusong/merge5_rh20t_20260909/v050_training_validation/{real_smoke,online_smoke}.json`。

七卡实际训练入口也已验证：六轨迹子集上，每卡 batch=1 和 batch=16 各运行两步，
在线 PointSeg、来源均衡采样、DDP 梯度同步及优化器更新正常，均 exit 0。
batch=16 有效 batch=112，两步日志 loss=0.469/0.451，梯度有限，无 OOM。
这只是有限样本的运行/容量检查，不代表全量长训收敛或评测性能已验证。
日志：`/home/liusong/merge5_rh20t_20260909/v050_training_validation/ddp7_batch16_smoke.log`。
两次均 `save_checkpoint=false`，没有保存新模型；正式 60k 训练尚未启动。
