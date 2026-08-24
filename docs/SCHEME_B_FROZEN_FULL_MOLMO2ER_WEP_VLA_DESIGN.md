# WEP-Prefix Full-Molmo2-ER WEP-VLA 设计与运行合同

- 状态：方案 B 已按精确 DoubleFlow 基准重构；CPU 架构、冻结梯度、processor 与基准回归均通过，仍需在新代码进程中做真实 8-GPU 烟测。
- 生效仓库：`/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song`
- 唯一结构基准：`origin/wep_vla_v0.4.3_multiview_doubleflow`，提交 `da0ad03bf7eff2c6e9edcf04e1b324bebbdf93dd`。
- 被替代文档：`FROZEN_FULL_MOLMO2ER_WEP_VLA_DESIGN.md`（保留为历史方案 A，不再是启动依据）
- 分布式策略：普通 DDP；当前不使用 DeepSpeed/ZeRO。
- 物理设备合同：正式训练支持全部 8 张 A800 80GB（GPU 0–7）；当前文档中的短数据调试命令为了保留 GPU 0 做评测，显式使用 GPU 1–7。

## 1. 唯一允许的架构

方案 B 只替换 SmolVLA 的冻结 VLM 条件分支，不改变 UMI、PointAction、flow matching 或 World–Ego DoubleFlow 的算法语义。

### 1.1 Molmo2-ER 参数冻结，但 prefix 输入梯度不截断

Molmo2-ER 的下列部分全部满足 `requires_grad=False`、`eval()`、不进入 optimizer：

- 原生 ViT 与 connector；
- WTE 与特殊 token embedding；
- 36 个文本 Transformer block；
- final norm。

`lm_head` 不实例化。原生 ViT、connector 和 WTE 没有可训练输入，因此它们在 `no_grad` 下生成并 detach 原生图文 embedding；随后 Ego/World FG/BG 进入 VLM prefix。36 个 decoder block 不能使用 `no_grad`：它们虽不计算参数梯度，却必须计算 action loss 对 FG/BG 输入的 VJP。这与 WEPVLA 的冻结语义相同。

梯度边界为：

```text
RGB + instruction -> Molmo ViT/WTE [eval + no_grad + detach] --+
PointSeg -> Ego/World FG/BG projections -----------------------+-> 36-layer frozen Molmo prefix
                                                                  <-> 36-layer Action Expert
foreground local -> PointAction -> Ego/World Action suffix ------+-> flow loss

loss -> Expert -> frozen Molmo operations -> FG/BG projections
Molmo parameter.grad 始终为 None
```

### 1.2 Native Molmo prefix

送入 decoder 的完整 prefix 为：

```text
[BOS] [410 IMAGE] [EGO_FG] [EGO_BG] [WORLD_FG] [WORLD_BG] [task/chat TEXT] [right PAD]
```

Ego FG/BG 插在原生 IMAGE block 后；World FG/BG 紧随 Ego FG/BG，位于第一个后置 LANGUAGE token 之前。最终 position id 按有效 token 的物理顺序连续重算，不再把 World token 追加到语言后，也不再复用 Ego position id。PAD 始终被 key/query mask 屏蔽。

- 410 个 IMAGE position 和 392 个视觉 feature 合同不变；
- `IMAGE + Ego/World FG/BG` 属于同一个双向 perception block；
- BOS/前置文本与后置 LANGUAGE 保持 Molmo2-ER 原生 causal 语义：perception 不读取后置语言，语言可读取 perception 与此前语言；
- padding query/key 全屏蔽；
- native embedding 分支 detach，但拼入 FG/BG 后的完整 prefix 携带输入梯度；
- FG/BG 投影宽度为 Molmo hidden 2560。

因此当前 mask 精确实现：

```text
visible(q,k) = valid(q) & valid(k)
             & ((k <= q) | (q in PERCEPTION & k in PERCEPTION))
PERCEPTION   = IMAGE + EGO_FG/BG + WORLD_FG/BG
```

在四个 scene mask 全关闭时，这条路径退化为 Molmo2-ER 官方单 IMAGE block 的 `causal OR IMAGE↔IMAGE` 注意力，native language 的 position id 也保持不变。

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

- 偶数层 `0,2,...,34`：prefix 与 action 进行一次 joint MHA；mask 保证 prefix 只读 prefix，而 action 读取完整 prefix 与全部 Ego/World action。
- 奇数层 `1,3,...,35`：prefix 先做自身 self-attention；action query cross-attend 本层完整、持续演化的 IMAGE/LANGUAGE/FG/BG K/V，本层无 action-action attention。
- Expert residual、MLP、final norm 全部可训练。

因此 IMAGE 与四个 FG/BG 会双向共同演化；后置 LANGUAGE 以 causal 方式读取 perception，但 perception 不反向读取后置 LANGUAGE。后续 Action 读取每一层正在演化的完整 prefix，而不是预计算的静态 native K/V。该约束先保护 Molmo2-ER 原生图文先验，再保留 WEPVLA 的 Expert 对接方式。

## 2. Trainable scene/action token 布局

token 分区严格为：

```text
VLM prefix:
  [BOS] [IMAGE]
        [EGO_FOREGROUND] [EGO_BACKGROUND]
        [WORLD_FOREGROUND] [WORLD_BACKGROUND]
        [causal LANGUAGE]

Expert suffix:
  [EGO_ACTION x 32] [WORLD_ACTION x 32]
```

四个 scene token 与基准分支逐一对应；不存在额外 attention-pooled scene token，也不在 Expert suffix 中复制 FG/BG。

- perception block 内双向，LANGUAGE causal；任何 prefix query 都不能读取 noisy action；
- 偶数层两个 action stream 双向读取全部 prefix 和完整 action block；
- 奇数 pure-cross 层不做 trainable-side self-attention，保持原 SmolVLA 交替拓扑；
- 最终只执行 Ego action；World action 是同一物理轨迹的共轭监督/隐变量；
- 不使用 learned World-to-Ego gate，也不建立第二套 36 层 Expert。

FG/BG 投影目标维度为 VLM hidden 2560；两个投影独立，null background 语义保持不变。

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

本方案不改变 seed、AdamW、LR、scheduler、chunk size、action dimension、PointSeg loss、flow time或数据合同。噪声耦合恢复为精确基准的 `left_compose_ego`。旧 Scheme B 曾额外注册 World 私有的 `152064 × 1920` language embedding；精确共享-Expert 基准不使用它，Ego/World 都读取同一份冻结 Molmo memory，因此该无效参数已移除。

当前 WorldFlow-on 精确合同：

```text
total     = 6,310,910,734
trainable = 1,823,509,364
frozen    = 4,487,401,370
```

相对 WorldFlow-off 的精确增量为：

```text
total     = 49,694,848
trainable = 36,964,763
frozen    = 12,730,085
```

当前 WorldFlow-off 精确合同：

```text
total     = 6,261,215,886
trainable = 1,786,544,601
frozen    = 4,474,671,285
```

训练前运行时审计会拒绝任何参数量、trainable allowlist、层数、冻结范围或 token/attention contract 漂移。新 checkpoint 在 `config.json` 中持久化 `full_molmo_topology=molmo_native_hybrid_wepvla_expert_v4`。旧 detached-scene checkpoint 的 FG/BG 位于 1920-wide Expert 外部；旧 v3 虽已有 2560-wide VLM-prefix 参数，但它把整个 prefix 设为双向、把 World 放在语言后并复用 Ego position id。二者都不能作为 v4 full resume 或直接评测；必须从头训练，或另做显式 warm-start 转换。

## 4. DDP 与 batch 合同

当前保持 8-rank replicated DDP，不启用 DeepSpeed/ZeRO：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --multi_gpu --num_processes=8
find_unused_parameters = false
gradient_as_bucket_view = true
```

当前直接命令选择 7 张 GPU，每卡 microbatch 40、accumulation 1、`global_batch_size=280`，因此每次 optimizer step 的有效样本数为：

```text
40 × 7 × 1 = 280
```

`global_batch_size` 是 exact-global-batch 调度与审计目标，不是额外分配一份 batch。新结构默认启用 `molmo_gradient_checkpointing=true`，并以 2 个“Molmo prefix + Expert”耦合层为一个非重入 checkpoint segment。这与逐层 segment 数学等价；按 B40 张量尺寸估算，持久化的 segment 边界由约 3.56 GiB 降为 1.78 GiB。实际峰值还包含反向重算 workspace，必须以 A800 实测为准。训练不保存 36 层 K/V cache 或完整层内激活；Molmo 权重、ViT/WTE detach、训练 `use_cache=false`、固定单图/双 crop 的 pixel/pooling view 重用也继续减少不必要显存。prefix 输入梯度本身不可消除，否则会破坏 WEP 拓扑；旧 batch 40 峰值不能直接外推，需在新进程重新测量。

## 5. 当前直接运行命令

在目标仓库执行：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song
```

随后直接复制执行 [WEPVLA_V043_DoubleFlow_MolmoER2.md](../benchmarks/song_real_libero/WEPVLA_V043_DoubleFlow_MolmoER2.md) 中的完整多行命令。该命令：

- 明确使用 `/home/liusong/anaconda3/envs/reap/bin/python`；
- 使用命令中显式选择的物理 GPU 和普通 DDP；
- 当前使用 7 卡、`batch_size=40`、`global_batch_size=280`；
- 使用 `worldflow_noise_coupling=left_compose_ego`；
- 不需要也不调用任何 `.sh` 文件。

旧的已停止 tmux session 不是可恢复训练状态。是否 resume 只能依据目标输出目录中的 checkpoint 与 training state 完整性判断。

## 6. 强制验证

代码门禁必须同时证明：

1. `molmo_inference_only=false`，`molmo_gradient_checkpointing=true`；
2. prefix 顺序为 native BOS/IMAGE、Ego scene、World scene、causal LANGUAGE，suffix 只含两个成对 action stream；
3. Molmo/ViT 参数的 `grad` 均为 `None`，但 action loss 对 FG/BG projection 梯度非零；
4. checkpoint 开/关的固定输入前向、输入梯度和 Expert 参数梯度一致；
5. native hybrid mask、连续 position id、推理 prefix cache 与 non-cache joint forward 输出一致；
6. UMI、PointAction、DoubleFlow 回归通过；
7. 普通 DDP launcher 不包含 DeepSpeed/ZeRO 参数；
8. 第一个 optimizer step 后所有 trainable tensor 都被 DDP 使用；
9. 参数 hash 在训练前后证明冻结 Molmo 未改变。

旧 detached-scene-suffix 或 v3 全 prefix 双向拓扑的 b8、b24 8-GPU 结果不能作为 v4 的显存/吞吐证据。当前已完成 CPU 架构/梯度/cache 回归；真实 8-GPU 两步烟测与新拓扑的 batch 上限仍待实测。
