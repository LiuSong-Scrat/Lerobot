# Frozen Full-Molmo2-ER WEP-VLA 设计合同

状态：架构设计冻结，尚未实现本版本代码  
目标基线：已复现的 500M WEP-VLA / v0.4.3 方法  
目标模型：完整冻结 Molmo2-ER + 从头训练的 Full SmolVLA-style Action Expert

## 1. 最终决策

新模型采用：

> 完整冻结的 Molmo2-ER VLM，加上与其 36 层一一对应、宽度为 VLM 0.75 倍的 36 层 SmolVLA-style Action Expert，并完整保留 500M WEP-VLA 的 PointSeg、FG/BG 全局点 token、局部 PointActionFusion、flow matching 目标和训练数据。

核心决定：

- 不再截断 Molmo2-ER 的文本层，完整使用 36/36 层。
- 不使用 MolmoAct、MolmoAct2 或 MolmoAct2-Pretrain 的任何 Action Expert/VLA 权重。
- Molmo2-ER 的 ViT、connector、token embedding、36 层 LLM 和 final norm 全部冻结。
- Action Expert、PointSeg、点投影、action/time 投影和 PointActionFusion 从头训练。
- Action Expert 继续使用 500M 的交替拓扑：偶数层 joint self-attention，奇数层 pure cross-attention。
- IMAGE 与 FG/BG SCENE token 构成双向感知块；文本保持 Molmo 原生 causal 语义。
- 使用真实二维 `agentview` RGB 和同视角 XYZRGB 点云。
- 保留 Molmo 原生视觉 processor、特殊 token 和 392 个视觉特征，不额外压缩成 64 个 token。
- 运行模型物理删除不参与 VLA forward 的 `lm_head`。

该模型不再是 3B。按实际参与模型注册的参数口径，它约为：

```text
6.261B total
1.799B trainable
4.462B frozen
```

## 2. 科学问题与控制原则

该实验检验：

> 在保持 500M WEP-VLA 的输入模态、点云双路接口、Action Expert 家族、动作目标和训练协议不变的前提下，将冻结的 VLM 替换成完整 Molmo2-ER，并按相同比例同步扩大 Action Expert，能否改善机器人策略能力。

必须保持不变：

- 数据集和 PointSeg cache；
- 单路 `agentview` RGB 和单路 `agentview` 点云；
- 一个 FG 和一个 BG 全局点 token；
- foreground local point 到 action 的直接融合；
- chunk、action 维度、flow noise、时间采样和损失；
- PointSeg 结构及辅助损失；
- state/worldflow/SE3 开关；
- global batch、seed、optimizer 和 scheduler 定义；
- VLM 全冻结、Expert 从头训练。

不可避免的 backbone-contract 差异必须公开报告：

- VLM 家族、预训练语料和 tokenizer 不同；
- 视觉塔和 connector 不同；
- 视觉 token 数不同；
- Molmo 使用原生 chat template 与 BOS；
- Molmo 的 embedding 不使用 SmolVLM 的 `sqrt(hidden_size)` 缩放；
- Molmo 的图像块双向、文本 causal；旧 500M 是整个条件 prefix 双向；
- Molmo 原生 Q/K norm、fused QKV 和 RoPE 参数不同。

因此，不能宣称唯一变化只是参数量。准确表述应为“在保持 VLA 外部接口和训练协议对齐的情况下替换并扩展冻结 backbone”。

## 3. 500M 与新模型总览

| 项目 | 旧 500M WEP-VLA | 新 Full-Molmo2-ER WEP-VLA |
|---|---|---|
| 冻结 VLM | SmolVLM2-500M 系列权重 | Molmo2-ER |
| 使用的文本层 | 前 16/32 层 | 完整 36/36 层 |
| VLM hidden / FFN | 960 / 2560 | 2560 / 9728 |
| VLM Q/KV/head dim | 15 / 5 / 64 | 32 / 8 / 128 |
| Expert 深度 | 16 | 36 |
| Expert hidden / FFN | 720 / 2048 | 1920 / 5120 |
| Expert 结构 | 偶数 joint-SA、奇数 pure-CA | 完全相同，仅加深加宽 |
| RGB 输入 | agentview 256² | agentview 256² |
| 视觉预处理 | 手工放大 512² | Molmo 原生 378² processor |
| 视觉输出 | 64 个 connector tokens | 392 个视觉特征、410 个 IMAGE 位置 |
| Scene tokens | 1 FG + 1 BG | 1 FG + 1 BG |
| Prefix mask | RGB+FG+BG+language 全双向 | IMAGE+SCENE 双向，TEXT causal |
| PointActionFusion | foreground local，4 heads | 完全保留 |
| Action | 32×10D，执行 16 步 | 完全保留 |
| 总参数 | 501,208,238 | 6,261,215,886 |
| 可训练参数 | 151,032,494 | 1,799,274,686 |

旧 500M 最终高分 checkpoint 是多阶段 warm-start 结果，不是从随机 Expert 开始的单阶段训练。新模型主实验应明确记录为 fresh trainable modules；若后续比较收敛速度，必须考虑初始化差异。

## 4. 输入数据合同

### 4.1 RGB

使用数据集中的：

```text
observation.images.agentview
shape: 3×256×256
range: [0, 1]
```

配置必须显式保证：

```text
camera_views=agentview
rgb_camera_views=agentview
```

### 4.2 点云

同一视角的点云为：

```text
shape: 10000×6
channels: X, Y, Z, R, G, B
```

仍使用当前 PointSeg cache 和相同坐标/UMI 处理。PointSeg cache 中的未来轨迹信息只用于离线生成软监督；模型在线输入仍只有当前帧点云、当前 RGB 和任务文本。

### 4.3 State 与 Action

```text
state_dim = 10
action_dim = 10
chunk_size = 32
n_action_steps = 16
encode_robot_state = false
```

State 仍参与 UMI 坐标变换，但不形成 VLM prefix token。

## 5. Molmo 原生视觉路径

必须使用本地 Molmo2-ER 的原生 processor：

```text
AutoProcessor.from_pretrained(
    Molmo2-ER,
    trust_remote_code=True,
    use_fast=False,
)
```

直接输入原始 256×256 RGB。禁止先经过旧 SmolVLA 路径的：

- `resize_with_pad(512)`；
- `[0,1] -> [-1,1]`；
- SmolVLM processor；
- 手工 4×4 token merge。

Molmo 原生流程：

```text
256×256 RGB
→ resize/crop 到 378×378
→ patch_size 14
→ 每个 crop 27×27 ViT patches
→ 原生 2×2 learned attention pooling
→ 每个 crop 14×14 = 196 visual features
```

当前 256×256 方形 agentview 会生成：

```text
global crop + 1×1 high-resolution crop
392 个 <im_patch> 视觉特征位置
18 个 image start/end/column 等结构位置
410 个 IMAGE 位置
```

每个位置先使用 Molmo token embedding，只在 `<im_patch>` 位置加上 vision connector feature。不能用 vision feature 替换整个 token embedding，也不能删除 image start/end/column token。

### 5.1 等价视觉加速

对当前固定 256×256 方图，global 和唯一 high-resolution crop 的像素逐元素相同。允许以下严格等价优化：

1. 运行时断言两个 crop 完全相同；
2. ViT 只编码第一个 crop；
3. 复用/复制第 `-3/-9` 层特征；
4. 继续生成完整两组视觉特征并填入原生 410-token 模板；
5. 断言失败时自动回退标准双 crop 编码。

该优化只减少视觉塔计算，不减少 36 层 LLM 的序列长度，也不改变模型输出。

## 6. Prefix 和三类 Token

最终顺序：

```text
[BOS]
[Molmo native IMAGE block]
[FG]
[BG]
[native chat/task TEXT]
[PAD]
```

展开为：

```text
Molmo native BOS
→ global/low-resolution image block
→ high-resolution image block
→ FG scene token
→ BG scene token
→ <|im_start|>user\n
→ task instruction
→ <|im_end|>\n
→ <|im_start|>assistant\n
→ right padding
```

FG/BG 插入位置必须由每个样本最后一个原生 IMAGE 位置动态确定，不能硬编码为固定 index。

显式定义：

```text
TEXT  = 0
IMAGE = 1
SCENE = 2
```

其中：

- `IMAGE`：Molmo processor 标记的所有 image patch/start/end/column token；
- `SCENE`：FG 和 BG 两个动态连续点云 token；
- `TEXT`：BOS、user/assistant role token 和自然语言；
- `PAD`：通过 validity mask 管理。

不要把 SCENE 伪装成 `<im_patch>` 或普通 IMAGE ID。只在 attention 规则中把 IMAGE 和 SCENE 归入同一个感知集合。

旧 `tokenizer_max_length=48` 只用于约束自然语言内容，绝不能把包含 410 个图像位置的整条 processor 序列截成 48。

## 7. IMAGE+SCENE 双向感知 Mask

定义：

```text
PERCEPTION = IMAGE ∪ SCENE
```

Prefix mask：

```text
allow(q, k) =
    valid(q) AND valid(k)
    AND [
        position(k) <= position(q)
        OR
        (type(q) ∈ PERCEPTION AND type(k) ∈ PERCEPTION)
    ]
```

分块关系：

| Query → Key | BOS | PERCEPTION | TEXT | ACTION |
|---|---:|---:|---:|---:|
| BOS | ✓ | ✗ | ✗ | ✗ |
| IMAGE/SCENE | ✓ | 全双向 | ✗ | ✗ |
| TEXT | ✓ | ✓ | causal | ✗ |
| ACTION | ✓ | ✓ | ✓ | 全双向 |

因此：

- IMAGE、FG 和 BG 相互双向可见；
- 感知块不能读取后置 instruction 或 action；
- 文本可以读取全部图像和 FG/BG；
- 文本内部严格 causal；
- action 可以读取整个 prefix 和全部有效 action；
- prefix 永远不能读取 action；
- padding query/key 全部屏蔽。

相对 Molmo 原生 mask，唯一修改是：

```text
原生：IMAGE ↔ IMAGE
新模型：IMAGE/SCENE ↔ IMAGE/SCENE
```

不能把 TEXT 并入双向块。

## 8. FG/BG 全局 Scene Tokens

保持 500M 的两个独立角色投影：

```text
foreground_global_feat: 64
→ Linear_fg(64, 2560)
→ 1 FG

background_global_feat: 64
→ Linear_bg(64, 2560)
→ 1 BG
```

约束：

- 始终恰好一个 FG 和一个 BG；
- 两个 Linear 不共享参数；
- 不额外增加 role token；独立投影本身区分角色；
- background 无候选时使用原有 `null_background_feat`，不能 mask 掉 BG；
- 不乘 SmolVLM 的 `sqrt(hidden_size)`；
- BF16 cast 必须保留对 FG/BG 输入的梯度；
- 初始化时使 FG/BG RMS 接近 Molmo 原生 IMAGE embedding RMS；
- 训练期间记录 IMAGE/FG/BG/TEXT RMS 和比例。

## 9. PointSeg 与局部 PointActionFusion

PointSeg 配置保持：

```text
pointseg_enable = true
pointseg_backbone_type = litept
pointseg_grid_size = 0.01
pointseg_feature_dim = 64
pointseg_foreground_ratio = 0.025
pointseg_background_ratio = 0.025
pointseg_min_foreground_points = 2500
pointseg_min_background_points = 0
pointseg_aux_loss_weight = 0.0005
pointseg_use_temporal_priors_as_input = false
pointseg_use_pseudo_selection = false
```

保留 500M 的局部路径：

```text
foreground local scene_tok1, mask
→ PointActionSelfAttention
→ action token residual
→ Expert layer 0
```

保持：

- 只有 foreground local tokens 直连 action；
- background local tokens 不直连 action；
- point hidden 为 64；
- action hidden 为 1920；
- 4 attention heads；
- dropout=0；
- 固定 32-step positional encoding；
- 整个 Expert 前只融合一次。

全局 FG/BG prefix 路和局部 foreground-action 路都必须保留，并在评估阶段分别做遮蔽消融。

## 10. 完整冻结的 Molmo2-ER

VLM 使用：

```text
text depth = 36
hidden = 2560
intermediate = 9728
Q heads = 32
KV heads = 8
head_dim = 128
rope_theta = 5,000,000
```

必须保留：

- 原生 fused QKV；
- 原生 Q/K RMSNorm；
- 原生 RoPE；
- 原生视觉 feature addition；
- 原生 tokenizer、BOS 和 chat template；
- 全部 text blocks 0..35；
- final norm。

冻结范围：

- ViT；
- vision connector/adapter；
- WTE 与所有特殊 token embedding；
- 36 层文本 block；
- final norm；
- `lm_head` 若保留在审计口径中也冻结。

所有冻结参数必须：

```text
requires_grad = false
module.eval()
not in optimizer
grad is None
training-before hash == training-after hash
```

### 10.1 冻结不等于切断计算图

ViT、connector 和原生输入 embedding 可以在 `no_grad` 中计算，因为其输入之前没有需要训练的模块。

插入可训练 FG/BG 后，完整 36 层 LLM 不能放进 `torch.no_grad()`，也不能 detach per-layer K/V。Action loss 必须经过冻结的 Molmo 运算回传到 FG/BG 投影和 PointSeg：

```text
flow loss
→ Action Expert
→ frozen Molmo operations
→ FG/BG projections
→ PointSeg
```

Molmo 参数本身始终没有梯度。

## 11. Action Expert：严格保持 500M 拓扑

不采用 MolmoAct2 的“每一层都先 SA 再 CA”的 DiT Expert，不使用 AdaRMS 或 MolmoAct2 Action Expert 权重。

新 Expert：

```text
depth = 36
hidden = 0.75 × 2560 = 1920
intermediate = 5120
Q heads = 32
KV heads = 8
head_dim = 128
self_attn_every_n_layers = 2
```

保留旧 action/time embedding MLP 和 flow time conditioning。

### 11.1 偶数层：Joint Self-Attention

层号：

```text
0, 2, 4, ..., 34
```

共 18 层。每层：

- VLM prefix 使用自己的 Q/K/V 投影；
- Expert action 使用自己的 Q/K/V 投影；
- 两侧 Q/K/V 沿 sequence 维组合后执行一次 attention；
- prefix 内部使用 IMAGE+SCENE/TEXT hybrid mask；
- prefix 看不到 action；
- action 能看全部 prefix 和完整 action chunk；
- 两侧分别做 residual 和 MLP。

### 11.2 奇数层：Pure Cross-Attention

层号：

```text
1, 3, 5, ..., 35
```

共 18 层。每层：

- Molmo prefix 正常做自己的 self-attention；
- Expert 只产生 action Query；
- 同层 Molmo K/V 经原有 K/V adapter 后供 action cross-attention；
- 该层不产生 action self-attention K/V；
- Expert 再做自己的 residual 和 MLP。

VLM 与 Expert 严格 36 层一一对应。不能减少 Expert 深度，也不能错位选取 VLM 层。

## 12. Flow、Action 与损失

保持 500M：

```text
action_dim = 10
chunk_size = 32
n_action_steps = 16
noise ~ N(0, 0.1²)
t = 0.999 × Beta(1.5, 1) + 0.001
x_t = (1-t) × noise + t × action
target = action - noise
```

Action loss：

```text
valid future step × 10D action 的平均 MSE
translation weight = 1
rotation6d weight = 1
gripper weight = 1
```

总损失：

```text
loss_total = loss_action + 0.0005 × loss_pointseg_aux
```

推理：

```text
10-step Euler flow integration
```

明确关闭：

```text
encode_robot_state = false
worldflow_enable = false
worldflow_bootstrap_from_ego = false
worldflow_se3_head_enable = false
se3_enable = false
se3_final_correction_enable = false
pose9_action_noise = false
RTC = false
```

`worldflow_bootstrap_from_ego=false` 在 WorldFlow 已关闭时不参与计算，但仍显式记录以避免配置漂移。

## 13. 参数量

### 13.1 新模型

| 模块 | 参数量 | 状态 |
|---|---:|---|
| Molmo ViT | 382,505,936 | frozen |
| Molmo connector | 56,611,328 | frozen |
| Molmo WTE | 389,283,840 | frozen |
| Molmo 36 text blocks | 3,633,509,376 | frozen |
| Molmo final norm | 2,560 | frozen |
| 冻结 Molmo 主干，无 lm_head | **4,461,913,040** | frozen |
| 18 个 Expert SA 层 | 884,809,728 | trainable |
| 18 个 Expert CA 层 | 851,779,584 | trainable |
| Expert final norm | 1,920 | trainable |
| Expert 合计 | **1,736,591,232** | trainable |
| PointSeg/WEP/投影/融合 | **62,711,614** | 其中 62,683,454 trainable |

主运行模型：

```text
total     = 6,261,215,886
trainable = 1,799,274,686
frozen    = 4,461,941,200
```

未使用的 Molmo `lm_head` 为 388,956,160 参数。若为完整 checkpoint 计数而保留：

```text
total     = 6,650,172,046
trainable = 1,799,274,686
frozen    = 4,850,897,360
```

模型实现应物理删除 `lm_head`，但实验报告同时给出 active-no-head 和 full-source 两种口径。

### 13.2 旧 500M

```text
total     = 501,208,238
trainable = 151,032,494
```

## 14. 初始化与训练协议

主实验只加载 Molmo2-ER 的冻结 VLM 权重。

不得加载：

- MolmoAct/MolmoAct2/MolmoAct2-Pretrain Action Expert；
- 旧 500M Action Expert；
- 当前 point-only 3B Action Expert；
- 其他机器人 VLA Action Expert 权重。

从头初始化：

- 36 层 Action Expert；
- action/time in/out projections；
- FG/BG projections；
- PointActionFusion；
- PointSeg/LitePT。

如果未来 warm-start 自己的 PointSeg，应作为单独消融，不能混入主实验。

训练配置：

```text
seed = 1000
global_batch = 192
optimizer = AdamW
optimizer_lr = 1e-4
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 1e-10
grad_clip_norm = 10
scheduler = cosine
scheduler_warmup_steps = 100
scheduler_decay_steps = 30000
scheduler_decay_lr = 2.5e-6
```

`3e-5` 是已有成熟 checkpoint 冷重启微调时采用的 floor，不适用于本模型随机 Action Expert 的主实验。为了对齐 500M fresh 训练，使用 `2.5e-6`。

建议先进行 30K pilot 并完成闭环评估，再决定是否继续到 80K。不能只根据训练 loss 选择 checkpoint。

## 15. 8×RTX 5090 运行预算

估计参数与训练状态显存：

```text
model weights: 11.78 GiB
gradients:      3.47 GiB
Adam moments:   6.94 GiB
steady state:  22.18 GiB
```

该估计沿用当前训练已经实证的 BF16 Expert Adam moments；若改成 FP32 moments，普通 replicated DDP 很可能放不下。

建议起点：

```text
num_processes = 8
microbatch_per_gpu = 1
gradient_accumulation_steps = 24
global_batch = 1 × 8 × 24 = 192
```

至少完成两个 optimizer steps，确保 Adam 状态已经创建后再判断显存。预计：

```text
B1 peak: 约 28–30 GiB
B2 peak: 约 31–32+ GiB，高风险
```

必要运行优化：

- selective BF16 load，删除 `lm_head`；
- 256²相同双 crop 的 exact single-encode reuse；
- ViT/connector `eval + no_grad`；
- LLM `eval + frozen`，但保留 input autograd；
- BF16 SDPA；
- 关闭 training cache、attention 输出和 hidden-state 收集；
- DDP `gradient_as_bucket_view=True`；
- accumulation microsteps 使用 `no_sync`；
- 固定图验证后使用 `find_unused_parameters=False` 和 `static_graph=True`；
- B1 仍 OOM 时启用逐层 activation checkpointing；
- optimizer 临时峰值过高时测试 `foreach=False`。

计算估计：

```text
约 7.51 TFLOPs / sample
global batch 192 ≈ 1.441 PFLOPs / optimizer step
预计 9–20 s / optimizer step
30K ≈ 3–7 天
80K ≈ 8–19 天
```

8 张 5090 无 NVLink，约 1.799B 可训练参数的梯度同步会是重要瓶颈。

## 16. 推理缓存

每次新观测：

1. 处理 agentview RGB；
2. 运行 PointSeg 并生成 FG/BG；
3. 一次性构造完整 `BOS+IMAGE+FG+BG+TEXT`；
4. 用 IMAGE+SCENE hybrid mask 跑完整 36 层；
5. 缓存各层 prefix K/V 和 validity mask；
6. 10 次 Euler 去噪复用同一份 prefix cache。

由于 IMAGE 可以反向读取 FG/BG，不能先单独 cache IMAGE 后再 causal append SCENE。IMAGE 和 SCENE 必须在同一次 prefill 中计算。

每个 Euler step 仍需重算 action self/cross-attention、time conditioning 和依赖当前 action 的 PointActionFusion。

## 17. 实现与烟测门禁

### 17.1 权重和结构

- ViT 和 connector 存在并严格加载；
- Molmo blocks 恰为 0..35；
- Expert 恰为 36 层；
- 偶数层全部 joint-SA，奇数层全部 pure-CA；
- 不存在 MolmoAct2 Action Expert checkpoint key；
- active 参数量精确匹配 6,261,215,886；
- trainable 参数量精确匹配 1,799,274,686；
- 源 Molmo shard hash 写入 experiment manifest。

### 17.2 Processor 与 token

- 每个样本恰好一个 agentview RGB；
- 256²输入恰好两个 crop；
- 恰好 392 个 `<im_patch>` 特征位置；
- 恰好 410 个 IMAGE 位置；
- FG/BG 恰好各一个；
- 插入点在最终 image end 后、文本前；
- BOS 不重复；
- 图像 token 未被 `max_length=48` 截断；
- batch 使用右 padding。

### 17.3 Mask 真值

必须逐项断言：

```text
IMAGE ↔ IMAGE = true
IMAGE ↔ SCENE = true
FG ↔ BG = true
PERCEPTION → TEXT = false
TEXT → PERCEPTION = true
TEXT → future TEXT = false
TEXT → past/self TEXT = true
ACTION → all prefix = true
ACTION ↔ ACTION = true
prefix → ACTION = false
PAD query/key = false
```

SCENE 关闭时，mask 和 Molmo native IMAGE/TEXT 路径必须退化并与官方实现对齐。

### 17.4 Freeze 与梯度

- 所有 Molmo 参数 `.grad is None`；
- optimizer 与 Molmo 参数 ID 交集为空；
- Molmo 训练前后逐 tensor/hash 相同；
- FG/BG projection、PointSeg、PointActionFusion 和 Expert 梯度 finite；
- PointSeg aux 从第一步产生梯度；
- 无 NaN/Inf；
- 记录 grad clipping 比例。

### 17.5 数值诊断

在层 0/5/11/17/23/29/35 记录：

- IMAGE RMS；
- FG RMS；
- BG RMS；
- TEXT RMS；
- scene/image 与 scene/text RMS ratio；
- hidden max；
- scene-token input gradient；
- Expert gradient norm；
- NaN/Inf 计数。

### 17.6 模态有效性

在固定输入、noise 和 t 下分别测试：

- RGB 置零/打乱；
- FG/BG 置零/交换；
- language 遮蔽；
- local PointActionFusion 关闭。

四种操作都应造成可测 action delta，避免某一路输入被模型实际忽略。

### 17.7 Cache 和 checkpoint

- non-cache forward 与 prefix-KV-cache forward 在 BF16 容差内一致；
- 10 次 Euler 复用 cache 与不复用结果一致；
- 保存后 strict reload；
- 固定 batch/noise/t 下动作输出一致；
- checkpoint 包含 FG/BG projections、PointActionFusion 和全部 Expert；
- checkpoint 不依赖宽松 missing/unexpected key 加载。

### 17.8 分布式烟测

- 单 GPU B1 完整 forward/backward；
- 至少两个 optimizer steps，覆盖 Adam 状态创建；
- save/reload/inference；
- 8 GPU、B1×accum24 两步；
- global batch 和 scheduler step 语义正确；
- 无 OOM、NCCL timeout、rank divergence 和非有限指标。

## 18. 评估协议

训练 loss 只能诊断优化稳定性，不能代表机器人成功率。模型选择必须使用与 500M 约 97% 结果相同的 LIBERO 闭环协议：

- 相同 40 个任务；
- 相同每任务 episode/seeds；
- 相同 agentview RGB 与点云构建；
- 相同执行 16 步和观测刷新逻辑；
- 按 suite、task 和总体报告成功率；
- 同时报告吞吐、显存、单步/闭环延迟；
- 固定 checkpoint 做 RGB、global scene、local point 和 language 消融。

推荐阶段：

1. 结构与数值门禁；
2. 单 batch 过拟合；
3. 8 GPU 100-step 稳定性；
4. 1K/5K pilot；
5. 30K checkpoint 完整 LIBERO 评估；
6. 只有闭环结果支持时再继续到 80K。

## 19. 论文/报告建议命名

建议名称：

```text
Frozen Full-Molmo2-ER WEP-VLA (6.26B)
```

建议方法描述：

> We retain the dual-path point-cloud interface, alternating self/cross-attention action expert, expert-to-VLM width ratio, flow-matching objective, and training protocol of the 500M WEP-VLA. We replace the frozen backbone with the full 36-layer Molmo2-ER and scale the randomly initialized action expert to 36 layers and 0.75 of the VLM hidden width. Molmo-native image processing and causal text semantics are preserved, while two point-conditioned foreground/background scene tokens are added to the bidirectional perceptual block. All Molmo parameters remain frozen and are verified unchanged throughout training.

不能表述为：

- “只改变了模型大小”；
- “使用了 MolmoAct2”；
- “Molmo forward 完全原样不变”；
- “所有 prefix 都是双向”；
- “VLM 在 `no_grad` 中运行”；
- “模型仍是 3B”。

