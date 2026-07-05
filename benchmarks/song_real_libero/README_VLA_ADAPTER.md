# WEP-VLA: SmolVLA VLM + PointCloud Action-Expert Adapter

这个版本用于在当前点云策略上接入 SmolVLA / PointACT 风格的双系统结构：

- VLM 保持官方 SmolVLA 0.45B 的结构与权重载入方式，负责 RGB + language 的语义表征。
- VLM 默认冻结，只训练 action expert、点云处理模块、PointSeg、WorldFlow、以及新增的 action-side adapter。
- 点云不直接改 VLM backbone，而是在 action expert 侧通过 gated cross-attention 注入。
- 数据转换脚本同步支持 RGB 图像，训练时 batch 需要包含 `observation.images.*`。

参考论文：

- SmolVLA: https://arxiv.org/pdf/2506.01844
- VLA-Adapter: https://arxiv.org/html/2509.09372v2
- PointACT: https://arxiv.org/pdf/2605.21414

本文档中的 adapter / PointACT-style 表述指本仓库实现采用的“冻结 VLM + 可训练 action expert + 点云侧路注入”设计思路；VLM 结构与权重载入以 SmolVLA 论文和官方 checkpoint 兼容性为准。

## 1. 核心设计

### 1.1 VLM 保持官方 SmolVLA 0.45B 结构

SmolVLA 0.45B 论文设定是：

- VLM backbone 使用 SmolVLM-2。
- VLM 冻结。
- action expert 训练。
- 主模型约 450M 参数，其中约 100M 属于 action expert。
- VLM/LLM 部分使用前 16 层。

因此本版本不在 VLM 内部插入点云模块、不替换 VLM attention、不修改 VLM decoder block。新增的点云融合只发生在 action expert 侧。

默认保持：

```bash
--policy.num_vlm_layers=16
--policy.expert_width_multiplier=0.75
--policy.attention_mode=cross_attn
```

不要为了 adapter 模式把 `num_vlm_layers` 改成 `-1`，除非你明确想使用完整 SmolVLM 而不是 SmolVLA 0.45B 结构。

### 1.2 VLM 权重载入方式

新增参数：

```bash
--policy.vlm_weights_path=<path-or-hf-repo>
```

它支持两种来源：

1. 普通 VLM 权重目录或 repo，例如本地 `SmolVLM2-500M-Video-Instruct`。
2. SmolVLA policy checkpoint，例如 `lerobot/smolvla_base` 或本地保存的 `pretrained_model` 目录。

如果 `vlm_weights_path` 指向 SmolVLA policy checkpoint，代码会只抽取：

```text
model.vlm_with_expert.vlm.*
```

并加载到当前模型的 VLM 部分。原 checkpoint 里的旧 action expert 不会被加载；当前版本的点云 action expert / adapter 重新初始化并训练。

推荐离线写法：

```bash
--policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
--policy.vlm_weights_path=/opt/data/private/liusong/hf_models/smolvla_base \
--policy.load_vlm_weights=true \
--policy.num_vlm_layers=16
```

如果环境可访问 Hugging Face，也可以：

```bash
--policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
--policy.vlm_weights_path=lerobot/smolvla_base \
--policy.load_vlm_weights=true \
--policy.num_vlm_layers=16
```

### 1.3 冻结策略

开启 adapter 时推荐：

```bash
--policy.vla_adapter_enable=true \
--policy.vla_adapter_freeze_vlm=true \
--policy.load_vlm_weights=true \
--policy.train_expert_only=true
```

代码会在 `vla_adapter_enable=true` 且 `vla_adapter_freeze_vlm=true` 时自动保证：

- `load_vlm_weights=True`
- `train_expert_only=True`

`train_expert_only=True` 会冻结 `vlm_with_expert.vlm`，但不会冻结：

- action expert
- action input/output projection
- point cloud encoder / PointSeg conditioner
- point cloud expert-side projection
- VLA adapter bridge
- WorldFlow / SE(3) auxiliary head，如果开启

## 2. 数据要求

由于 VLM 需要 RGB 图像，新的 adapter 训练数据必须包含 image feature：

```text
observation.images.<camera>
```

例如：

```text
observation.images.overhead
```

如果 `--policy.vla_adapter_enable=true`，但 batch 中没有任何 `observation.images.*`，训练会直接报错，提示重新转换数据。

点云仍然通过 sidecar 或原有 point-cloud 读取逻辑进入：

```text
observation.point_cloud
```

### 2.1 原有 point 处理功能保持

这个版本没有移除原来的点云处理链路。无论是否开启 VLA adapter，训练 batch 仍然必须提供：

```text
observation.point_cloud
```

形状要求：

```text
(B, N, 6)
```

其中 6 个通道为：

```text
x, y, z, r, g, b
```

仍然支持可选 padding mask：

```text
observation.point_cloud_is_pad
```

也仍然支持 PointSeg / priorseg 相关 side-channel：

```text
pointseg.priors
pointseg.labels
pointseg.weights
pointseg.class_scores
pointseg.role_scores
pointseg.foreground_score
```

adapter 关闭时：

```text
point cloud -> LitePT / PointSeg -> point prefix token -> VLM/action expert
```

这保持原来的点云 prefix 路径。

adapter 开启且默认：

```bash
--policy.vla_adapter_point_prefix=false
```

时：

```text
point cloud -> LitePT / PointSeg -> expert-side point token -> action bridge -> action expert
```

也就是说，点云不会被塞进冻结 VLM 的 prefix，避免破坏官方 VLM 的图文输入分布；但点云特征仍然完整进入可训练 action expert。

如果需要 ablation，可设置：

```bash
--policy.vla_adapter_point_prefix=true
```

此时点云同时进入 VLM prefix 和 action bridge。

### 2.2 PointSeg / WorldFlow 保持

PointSeg 前景/背景选点仍使用向量化 top-k / gather，并处理以下边界情况：

- 点数少于目标点数时，自动重复已有候选，保持输出定长。
- 某个 batch 样本全 padding 时，fallback 到第一个点，避免 NaN 或空张量。
- 背景候选为空时，`pointseg_background_has_candidates=False`，并使用稳定 fallback。

WorldFlow 仍然使用：

```text
pointseg.role_scores
observation.point_cloud_is_pad
world_ee_poses sidecar
```

并在内部先通过 `select_objectflow_points` 选择 ObjectFlow 点，再做后续 dense flow / weighted Kabsch 处理。

### 2.3 点云数据转换保持

三个转换脚本仍保留原有点云功能：

- 真机 HDF5 转换仍支持已有 XYZRGB 点云、camera-frame gripper cloud、zarr/npy 存储、`world_ee_poses` sidecar。
- LIBERO 转换仍支持仿真外参转相机系末端位姿、相机坐标系下添加 gripper cloud、zarr/npy 存储、preview 导出。
- 通用 HDF5 转换仍保留 `downsample_point_clouds_keep_tail`，即 downsample 场景点时保留末端/gripper tail 点。

## 3. 数据转换脚本

### 3.1 真机 HDF5 转 LeRobot dataset

脚本：

```bash
benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py
```

默认会自动查找：

```text
observations/images/<camera>
```

常用参数：

```bash
python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  --input-dir /path/to/hdf5_raw \
  --output-root /path/to/real_lerobot_dataset \
  --camera overhead \
  --image-key observations/images/overhead \
  --point-cloud-key observations/cloud_rgb/overhead \
  --point-cloud-storage zarr \
  --overwrite
```

如果不想保存 RGB：

```bash
--image-key none
```

但这样不能用于 `vla_adapter_enable=true` 的 RGB-VLM 训练。

### 3.2 LIBERO HDF5 转 LeRobot dataset

脚本：

```bash
benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py
```

新增参数：

```bash
--save-rgb-images
--image-camera <camera_name>
```

示例：

```bash
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_object \
  --episodes 50 \
  --save-rgb-images \
  --image-camera agentview \
  --point-cloud-storage zarr \
  --overwrite
```

如果 `image_camera` 不在原始 `camera_names` 或 `pointcloud_camera_names` 中，脚本会自动加入实际 render camera 列表，避免 dataset schema 中有图像但 episode 缺图。

### 3.3 通用 HDF5 转换

脚本：

```bash
src/lerobot/scripts/song_lerobot_from_hdf5.py
```

新增参数：

```bash
--image-key observations/images/overhead
--image-feature-key observation.images.overhead
```

示例：

```bash
python src/lerobot/scripts/song_lerobot_from_hdf5.py \
  --hdf5-folder /path/to/hdf5 \
  --output-root /path/to/lerobot_dataset \
  --point-cloud-key observations/cloud_rgb/overhead \
  --image-key observations/images/overhead \
  --image-feature-key observation.images.overhead \
  --overwrite
```

## 4. 推荐训练命令

下面是 RGB + 点云 + frozen SmolVLA VLM + trainable action expert/adapter 的推荐模板：

```bash
export SONG_POINTSEG_REQUIRE_POINTOPS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset \
  --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/opt/data/private/liusong/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=16 \
  --policy.train_expert_only=true \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vla_adapter_point_prefix=false \
  --batch_size=48 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_adapter \
  --job_name=wep_vla_adapter \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=4000 \
  --eval_freq=4000 \
  --num_workers=12 \
  --policy.pointseg_enable=true \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/song_pointseg_sample_cache \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.002 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=4000 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=true \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

如果想更贴近 SmolVLA 官方“state 也作为 VLM prefix”的设置，可以额外开启：

```bash
--policy.encode_robot_state=true
```

## 5. Adapter 路径说明

开启：

```bash
--policy.vla_adapter_enable=true
```

后，数据流为：

```text
RGB image(s) + language tokens (+ optional state)
        │
        ▼
official SmolVLA / SmolVLM VLM backbone
        │ frozen VLM features
        ▼
action-side gated bridge  ◄── point-cloud tokens / PointSeg tokens
        │
        ▼
action expert flow matching
        │
        ▼
action chunk
```

默认：

```bash
--policy.vla_adapter_point_prefix=false
```

表示点云不作为 VLM prefix token 输入 VLM，而是在 action expert 侧注入。这可以最大限度避免破坏官方 VLM 的输入分布和结构。

如果设置：

```bash
--policy.vla_adapter_point_prefix=true
```

则点云 token 同时作为 VLM prefix 和 action bridge token 使用。这个模式更强，但也更可能改变 VLM prefix 分布，建议作为 ablation 使用。

## 6. WorldFlow / SE(3) 辅助项

当前推荐先保持：

```bash
--policy.worldflow_enable=true
--policy.worldflow_se3_head_enable=false
--policy.se3_enable=false
--policy.se3_final_correction_enable=false
```

其中：

- `worldflow_enable=true`：开启 world/ego 辅助的 dense ObjectFlow 监督。
- `worldflow_se3_head_enable=false`：旧 SE(3) head 只保留 CLI 兼容，不再作为主路径。
- `se3_enable=false`：动作生成仍使用原 flow matching action chunk。
- `se3_final_correction_enable=false`：不使用 PCA 风格 final correction。

## 7. 常见错误

### 7.1 batch 缺少 RGB

错误类似：

```text
vla_adapter_enable=True requires at least one RGB image feature in the batch.
Regenerate the LeRobot dataset with observation.images.* features.
```

解决：重新转换 dataset，并确认 meta features 中存在：

```text
observation.images.overhead
```

### 7.2 SmolVLA 0.45B VLM 权重没有加载

推荐显式指定：

```bash
--policy.vlm_model_name=/path/to/SmolVLM2-500M-Video-Instruct
--policy.vlm_weights_path=/path/to/smolvla_base
--policy.load_vlm_weights=true
--policy.num_vlm_layers=16
```

其中 `vlm_model_name` 用来创建 VLM 结构和 processor，`vlm_weights_path` 用来从官方 SmolVLA policy checkpoint 里抽取 VLM 子权重。

当 `vlm_weights_path` 指向 SmolVLA policy checkpoint 时，`vlm_model_name` 本地目录不需要包含完整 VLM 权重；只要有 `config.json` 和 processor/tokenizer 文件即可。代码会先从 `vlm_model_name` 构造官方 VLM 结构，再从 `vlm_weights_path/model.safetensors` 中加载 `model.vlm_with_expert.vlm.*` 子权重。

### 7.3 误设 `num_vlm_layers=-1`

如果目标是 SmolVLA 0.45B，请保持：

```bash
--policy.num_vlm_layers=16
```

`-1` 表示不截断 VLM 层数，可能对应完整 SmolVLM，而不是论文中 0.45B 的 SmolVLA 配置。

## 8. 本版本已做的轻量验证

已执行：

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/smolvla/smolvlm_with_expert.py \
  benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  src/lerobot/scripts/song_lerobot_from_hdf5.py
```

并验证：

- `observation.images.*` 会被 LeRobot 映射为 `FeatureType.VISUAL`。
- `vla_adapter_enable=true` 时 `load_vlm_weights/train_expert_only` 自动开启。
- `num_vlm_layers=16` 在 adapter 模式下保持不变。
- SmolVLA policy checkpoint 的 `model.vlm_with_expert.vlm.*` key 可以正确抽取为裸 VLM state dict。
- action bridge 支持前向和反向传播。

完整大模型训练仍需要用真实 VLM 权重和实际 RGB+点云 dataset 运行验证。
