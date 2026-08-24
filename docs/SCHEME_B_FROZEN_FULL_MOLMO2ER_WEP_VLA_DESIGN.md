# Scheme B：Frozen Full-Molmo2-ER WEP-VLA 设计与运行合同

- 状态：方案 B 已按精确 DoubleFlow 基准重构；CPU 架构、冻结梯度、processor 与基准回归均通过，仍需在新代码进程中做真实 8-GPU 烟测。
- 生效仓库：`/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song`
- 唯一结构基准：`origin/wep_vla_v0.4.3_multiview_doubleflow`，提交 `da0ad03bf7eff2c6e9edcf04e1b324bebbdf93dd`。
- 被替代文档：`FROZEN_FULL_MOLMO2ER_WEP_VLA_DESIGN.md`（保留为历史方案 A，不再是启动依据）
- 分布式策略：普通 DDP；当前不使用 DeepSpeed/ZeRO。
- 物理设备合同：使用全部 8 张 A800 80GB，物理 GPU 0–7。

## 1. 唯一允许的架构

方案 B 只替换 SmolVLA 的冻结 VLM 条件分支，不改变 UMI、PointAction、flow matching 或 World–Ego DoubleFlow 的算法语义。

### 1.1 Molmo2-ER 是纯推理特征源

Molmo2-ER 的下列部分全部满足 `requires_grad=False`、`eval()`、不进入 optimizer：

- 原生 ViT 与 connector；
- WTE 与特殊 token embedding；
- 36 个文本 Transformer block；
- final norm。

`lm_head` 不实例化。

完整 VLM forward（包括 36 个文本 block）在 `torch.no_grad()` 内执行。每一层生成的 native IMAGE/TEXT K/V 被立刻 detach，并作为同层 Action Expert 的只读 attention memory。不得：

- 把 FG/BG token 插入 Molmo prefix；
- 为冻结 Molmo 保留 input autograd；
- 让 action loss 经过 Molmo 运算回传；
- 只冻结参数、但仍构建 36 层 Molmo backward graph。

这使梯度边界严格为：

```text
RGB + instruction
  -> Molmo ViT/WTE/36 blocks [eval + no_grad]
  -> 36-layer detached IMAGE/TEXT K/V memory
                                      |
PointSeg -> FG/BG projections --------+-> shared Action Expert -> flow loss
foreground local -> PointAction ------+
Ego/World scene/action tokens --------+
```

### 1.2 Native Molmo prefix

Molmo 只接收原生：

```text
[BOS] [410 IMAGE positions] [causal task/chat TEXT] [PAD]
```

- 410 个 IMAGE position 和 392 个视觉 feature 合同不变；
- IMAGE block 保持双向；
- TEXT 保持 causal；
- padding query/key 全屏蔽；
- prefix tensor、每层 K/V 均不携带梯度；
- FG/BG 不属于 Molmo token role。

### 1.2.1 批量矩阵化图像 fast path

固定的 256×256、单图、两 crop 合同使用 CPU batched Torch 路径：

- 一次 BCHW 378×378 resize；
- HWC float32 批量 normalize；
- 通过 `reshape + permute` 一次生成全部 27×27 patch；
- 预计算固定的 392×4 pooling index；
- 按 native `[global, local]` 顺序复制像素完全相同的两 crop。

fast path 只有在本地 processor 的 resize、patch、pooling、mean/std、placeholder 等字段全部匹配时才启用；任何合同漂移都会自动回退到官方 slow processor。真实本地 Molmo processor 的 B=3 CPU 对照中，所有 text fields、pixel values、pooling、grid 与 crop count 均逐位相等；隔离图像处理耗时由约 0.062 s 降至 0.029 s。该数字不是完整训练 `data_s`。

### 1.3 每层与 Expert 的对接

Action Expert 深度 36、hidden 1920、FFN 5120；与 Molmo 36 层严格一一对应。

- 偶数层 `0,2,...,34`：Expert query 使用本层 Expert Q；key/value 为本层 detached Molmo IMAGE/TEXT K/V 与 Expert scene/action K/V 的拼接。这保留 SmolVLA joint self-attention 的 trainable-side 交互，同时 VLM 不产生可训练 query/output。
- 奇数层 `1,3,...,35`：Expert query 只 cross-attend 同层 detached Molmo IMAGE/TEXT K/V；K/V adapter 仍可训练。该层不增加 Expert self-attention。
- Expert residual、MLP、final norm 全部可训练。

这里的“每层结果”在 attention 接口上具体实现为每层的 K/V memory，而不是只取最后一层 hidden state。

## 2. Trainable scene/action token 布局

共享 Expert suffix 为：

```text
scene block:
  [EGO_FOREGROUND] [EGO_BACKGROUND]
  [WORLD_FOREGROUND] [WORLD_BACKGROUND]

action block:
  [EGO_ACTION x 32] [WORLD_ACTION x 32]
```

这四个 scene token 与基准分支进入 VLM 前的 Ego/World 前景、背景语义逐一对应；不存在额外的 attention-pooled `EGO_SCENE` 或 `WORLD_SCENE` token。由于 Molmo 必须完全冻结，它们仅从原 VLM prefix 平移到 Action Expert 的非 action block。

- scene block 内双向；
- scene query 不能读取未来 action block；
- 两个 action stream 可读取全部 scene token、完整 action block和 36 层 detached Molmo memory；
- 奇数 pure-cross 层不做 trainable-side self-attention，保持原 SmolVLA 交替拓扑；
- 最终只执行 Ego action；World action 是同一物理轨迹的共轭监督/隐变量；
- 不使用 learned World-to-Ego gate，也不建立第二套 36 层 Expert。

FG/BG 投影目标维度从旧方案 A 的 VLM hidden 2560 改为 Expert hidden 1920。两个投影独立，null background 语义保持不变。

## 3. 与改版 SmolVLA 的严格对齐

### 3.1 UMI processor

`src/lerobot/processor/umi_processor.py` 与基准仓库对应文件保持逐字节一致。processor 顺序仍为：

```text
UMI feature construction -> policy normalizer -> model
```

没有把 UMI 坐标/动作转换移入 Molmo processor，也没有让图像 processor 修改 UMI 的 pose9/action 合同。

### 3.2 PointAction adapter

保持原实现：

- 只取 foreground local point tokens；
- `PointActionSelfAttention`，4 heads，dropout 0；
- 只在进入共享 Expert 前融合一次；
- background local points 不直连 action；
- padded point/action 不参与注意力；
- Ego 与 World 各自保留可训练 direct point path。

### 3.3 World–Ego DoubleFlow

保持原实现：

- Ego/World 表示同一条物理轨迹；
- commanded World target 来自 sidecar；
- World noise 采用精确分支注册的 `left_compose_ego`：由 Ego prior 与当前 EEF carrier 解析构造；
- Ego/World 共享单个 36 层 Expert；
- pose9 flow matching 两侧直接监督；
- 无 learned residual gate；
- 推理执行最终 Ego 10D action。

### 3.4 未改变的训练超参数

方案 B 不改变 seed、AdamW、LR、scheduler、chunk size、action dimension、PointSeg loss、flow time或数据合同。噪声耦合恢复为精确基准的 `left_compose_ego`。旧 Scheme B 曾额外注册 World 私有的 `152064 × 1920` language embedding；精确共享-Expert 基准不使用它，Ego/World 都读取同一份冻结 Molmo memory，因此该无效参数已移除。

当前 WorldFlow-on 精确合同：

```text
total     = 6,310,743,694
trainable = 1,823,342,324
frozen    = 4,487,401,370
```

相对 WorldFlow-off 的精确增量为：

```text
total     = 49,611,008
trainable = 36,880,923
frozen    = 12,730,085
```

当前 WorldFlow-off 精确合同：

```text
total     = 6,261,132,686
trainable = 1,786,461,401
frozen    = 4,474,671,285
```

训练前运行时审计会拒绝任何参数量、trainable allowlist、层数、冻结范围或 Scheme-B architecture contract 漂移。

## 4. DDP 与 batch 合同

当前保持 8-rank replicated DDP，不启用 DeepSpeed/ZeRO：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --multi_gpu --num_processes=8
find_unused_parameters = false
gradient_as_bucket_view = true
```

当前直接命令为每卡 microbatch 40、accumulation 1、`global_batch_size=320`，因此每次 optimizer step 的有效样本数为：

```text
40 × 8 × 1 = 320
```

`global_batch_size` 是 exact-global-batch 调度与审计目标，不是额外分配一份 batch。旧 Scheme B 的 b8/b24 测量基于已移除的 World 私有大词表嵌入，不能作为新结构的最终吞吐结论；新进程仍需重新记录 `data_s/train_s`、峰值显存和 samples/s。

## 5. 当前直接运行命令

在目标仓库执行：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song
```

随后直接复制执行 [WEPVLA_V043_DoubleFlow_MolmoER2.md](../benchmarks/song_real_libero/WEPVLA_V043_DoubleFlow_MolmoER2.md) 中的完整多行命令。该命令：

- 明确使用 `/home/liusong/anaconda3/envs/reap/bin/python`；
- 使用物理 GPU 0–7 和普通 DDP；
- 使用 `batch_size=40`、`global_batch_size=320`；
- 使用 `worldflow_noise_coupling=left_compose_ego`；
- 不需要也不调用任何 `.sh` 文件。

旧的已停止 tmux session 不是可恢复训练状态。是否 resume 只能依据目标输出目录中的 checkpoint 与 training state 完整性判断。

## 6. 强制验证

代码门禁必须同时证明：

1. `molmo_inference_only=true`；
2. 36 份 per-layer memory 全部 `requires_grad=False` 且无 `grad_fn`；
3. Molmo 输入、ViT/WTE/blocks/final norm 的 grad 均为 `None`；
4. FG/BG、PointSeg、PointAction、Ego/World scene/action、Expert 获得梯度；
5. suffix 顺序严格为 Ego FG/BG、World FG/BG + 两个成对 action stream；
6. UMI、PointAction、DoubleFlow 回归通过；
7. 普通 DDP launcher 不包含 DeepSpeed/ZeRO 参数；
8. 第一个 optimizer step 后所有 trainable tensor 都被 DDP 使用；
9. 参数 hash 在训练前后证明冻结 Molmo 未改变。

当前 b8、b24 的 8-GPU 两步烟测和 8-step benchmark 均已完成；绝对 OOM 上限未探测，也不作为固定 global batch 192 的吞吐目标。
