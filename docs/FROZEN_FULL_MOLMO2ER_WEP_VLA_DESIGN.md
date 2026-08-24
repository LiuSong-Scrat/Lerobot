# Frozen Full-Molmo2-ER WEP-VLA v5 详细设计合同

> 当前有效拓扑：`molmo_native_readonly_scene_shadow_wepvla_expert_v5`。
> 本文与 [SCHEME_B_FROZEN_FULL_MOLMO2ER_WEP_VLA_DESIGN.md](SCHEME_B_FROZEN_FULL_MOLMO2ER_WEP_VLA_DESIGN.md) 描述同一个 v5 实现；前者偏运行合同，本文偏逐层结构与验收。任何把 active FG/BG 插入原生 IMAGE block、令 native image/language 读取 FG/BG 的 v4 描述都已废止。

## 1. 目标与不可同时违反的两组约束

### 1.1 保护 Molmo2-ER 原生先验

激活点云后，Molmo2-ER 原生 image/language 路必须保持：

- 官方 processor token 序列；
- 官方 IMAGE/TEXT/PAD validity；
- 官方 position id；
- 官方 hybrid attention 语义；
- 36 层 frozen hidden/K/V 不读取任何 FG/BG/action；
- 同一数值精度与固定输入下，scene 开关不能改变 N 输出。

这要求 native stream 单独执行。把 S/A 拼到同一个大 attention，再把 N→S/A 设为 false，只能保证数学 mask，不保证 SDPA/GEMM 的 kernel shape 和舍入路径不变。

### 1.2 保留 WEPVLA 的 action 信息流

在不修改 N 的条件下，仍需保留：

- Ego/World FG/BG 是 action 之前的非 action 条件 token；
- scene token 逐层吸收图文信息；
- action 每一层都读取图文和 scene；
- 偶数 Action Expert 层包含 action-action 双向注意；
- 奇数 Action Expert 层是 pure cross-attention；
- local PointAction 只在进入 Expert 前融合一次；
- Ego/World action 共享同一个 Expert。

原 WEPVLA 允许冻结 VLM 的 image/language hidden 读取 FG/BG；这与“Molmo native hidden 完全不受点云影响”存在逻辑冲突。v5 明确选择保护 Molmo N，并用单向的 scene-shadow `N -> S` 补回 scene 的多模态条件化；不再声称保留 `S -> N`。

## 2. 三流物理布局

### 2.1 Native stream N

```text
N = Molmo processor 的完整官方 image/language 序列
  = [BOS/chat TEXT, IMAGE placeholders, LANGUAGE, right PAD]
```

对于当前 256×256 单张 `agentview`：

- processor 产生两个 378×378 crop；
- connector 产生 392 个视觉 feature；
- native 序列中有一段连续 410-position IMAGE block；
- feature 只加到对应的 native image patch embedding；
- 其余特殊 image token 和 language token 使用原生 WTE。

N 内没有 FG/BG、PointAction 或 noisy action。

### 2.2 Scene-shadow stream S

```text
S = [EGO_FG, EGO_BG, WORLD_FG, WORLD_BG]
```

- 每个 scene token 由 64D PointSeg 全局特征投影到 2560D Molmo hidden；
- Ego 与 World 使用各自的 foreground/background projection 和 type embedding；
- 无效 foreground/background 由 validity mask 屏蔽；
- S 是 prefix 条件流，但不是 native IMAGE/LANGUAGE 流；
- 四个 S token 在进入 backend 前物理追加在整个 N 后面，形成公共布局 `[N][S]`；
- backend 立即按已知 `scene_length=4` 拆成两个独立 forward。

### 2.3 Action stream A

```text
A = [EGO_ACTION_0 ... EGO_ACTION_31,
     WORLD_ACTION_0 ... WORLD_ACTION_31]
```

- 每个 action stream 32 token、Expert hidden 1920；
- Ego/World 同一时间步复用同一个相对 RoPE 位置；
- 两条流均带 stream/type 与 conjugate-motion 编码；
- 两条流共享一套 36 层 Action Expert；
- flow 输出分别投影回 Ego/World action velocity；
- 控制执行只取 Ego 10D action。

## 3. N 的官方只读执行

N 经完整 36 层 frozen Molmo decoder 单独执行：

```text
N^0 = official native embeddings
for l in 0..35:
    K_N^l, V_N^l = FrozenMolmoQKV_l(N^l)
    N^(l+1)       = FrozenMolmoBlock_l(N^l, official_native_mask)
```

执行合同：

- `requires_grad=False`：ViT、connector、WTE、36 blocks、final norm；
- `eval()`：所有 frozen Molmo module；
- `torch.no_grad()`：完整 N forward；
- `detach()`：每层 N K/V 和最终 N hidden；
- `lm_head` 不创建、不加载、不计入 active backend；
- S/A 的 tensor 从未作为 N attention 的列或行出现。

这里的“官方”指 Molmo2-ER checkpoint、processor、token/position 和 native hybrid attention 合同。由于当前完整模型有意使用 BF16，验收应在相同 dtype、相同 SDPA 环境下比较；不能把 FP32 与 BF16 的逐位差异误判为结构污染。

## 4. S 的 scene-shadow 执行

S 使用冻结 Molmo block 的参数和顺序，但只更新 S 自己：

```text
S^0 = projected Ego/World FG/BG
for l in 0..35:
    Q_S^l, K_S^l, V_S^l = FrozenMolmoQKV_l(S^l)
    S^(l+1) = FrozenMolmoBlock_l(
        query=S^l,
        key/value=[K_N^l,V_N^l ; K_S^l,V_S^l],
        mask=S -> valid N + all valid S,
    )
```

关键点：

- N query 不在 S forward 内，所以 S 不可能反向改变 N；
- 四个 S token 属于同一个双向 scene block；
- `K_S^l/V_S^l` 是进入第 l 层 attention 前的 K/V，与标准 transformer cache 语义一致；
- S 经过 frozen attention residual、MLP 和层间更新，因此第 l 层后的 S 已吸收前 l 层的 N/scene 信息；
- frozen 参数不求梯度，但 frozen 运算对 S 输入的 VJP 保留；
- `K_S/V_S` 不能 detach，否则 global FG/BG 路失去训练信号。

## 5. A 的 36 层 WEP 调度

Molmo N/S 使用 hidden 2560；Action Expert 使用 hidden 1920、36 层、FFN 5120、32 query heads、8 KV heads、head dim 128。每个 Expert 层与同编号的 N/S memory 一一对应。

### 5.1 偶数层：action joint/self attention

层号 `0,2,...,34`：

```text
Q_A, K_A, V_A = ExpertFusedQKV_l(A^l)
A_attn = SDPA(
    Q_A,
    K=[K_N^l ; K_S^l ; K_A],
    V=[V_N^l ; V_S^l ; V_A],
    mask=A -> valid N + valid S + all valid Ego/World A,
)
A^(l+1) = ExpertResidualMLP_l(A^l, A_attn)
```

N/S 的 GQA K/V 已是 8×128，与 Expert 的 8 KV heads/head dim 128 兼容，偶数层不需要额外 2560→1920 hidden projection；attention 只消费 K/V head tensor。全部 Ego/World action 在同一个双向 action block 内互读，不做时间 causal mask。

### 5.2 奇数层：pure cross-attention

层号 `1,3,...,35`：

```text
Q_A = ExpertCrossQ_l(A^l)
K_P, V_P = ExpertCrossKV_l([K_N^l ; K_S^l], [V_N^l ; V_S^l])
A_attn = SDPA(Q_A, K_P, V_P, mask=A -> valid N + valid S)
A^(l+1) = ExpertResidualMLP_l(A^l, A_attn)
```

奇数层没有 K_A/V_A，因此没有 action-action 注意。这不是 mask 偶然屏蔽，而是 pure cross-attention 的结构合同。

### 5.3 完整可见性矩阵

| Query \ Key | N | S | A（偶数） | A（奇数） |
|---|---:|---:|---:|---:|
| N | 官方 native mask | ✗ | ✗ | ✗ |
| S | ✓ | 双向 | ✗ | ✗ |
| A | ✓ | ✓ | 双向 | 不生成 A K/V |

这张表是 v5 的唯一 token 交互真值。不得用“FG/BG 与 IMAGE 同属双向 perception block”描述 v5。

## 6. Position id

### 6.1 N position

N 只按官方有效序列累计 position，scene 开关不改变任一 native token 的 position。

### 6.2 S position

```text
ego_fg_pos = last_native_image_pos + 1
ego_bg_pos = last_native_image_pos + 2
world_fg_pos = ego_fg_pos
world_bg_pos = ego_bg_pos
```

S 在物理 tensor 中位于完整 N 后，但使用与 WEP scene 配对一致的虚拟位置。scene position 可能与后置 native LANGUAGE position 重叠；N/S 是独立 query stream，这不会把 S 插入 N，也不会移动 N。

### 6.3 A position

Ego/World 同一 action step 共享相对 step position；base offset 使用全部有效 prefix position 的最大值加一。这样同时保留 N 原生 position 和 WEP paired-action position。

## 7. 梯度图与 optimizer allowlist

### 7.1 N 路

```text
RGB/language -> ViT/WTE -> 36 Molmo blocks -> N K/V
                no_grad, all detached
```

N 不保存参数梯度或输入梯度。

### 7.2 S/A 路

```text
PointSeg
  -> FG/BG global projections
  -> S through frozen Molmo operations
  -> differentiable per-layer S K/V
  -> A attention / Expert
  -> Ego + World flow losses
```

`loss -> frozen operation -> S input` 合法且必要；“冻结参数”不等于“禁止对输入求导”。

### 7.3 `molmo_inference_only=false`

此字段必须为 false。原因是 S 需要 input autograd，而不是 Molmo 参数可训练：

```text
N: inference-only / no_grad
S: frozen weights + differentiable input graph
A: trainable
whole backend: not inference-only
```

若设为 true，backend 会采用旧 detached-memory 路，global FG/BG 无法通过每层 scene memory 获得 action loss 梯度。

## 8. Activation checkpoint

训练默认：

```text
molmo_gradient_checkpointing=true
molmo_gradient_checkpointing_layers_per_segment=2
```

该开关只 checkpoint A：

1. N 在 forward 中计算一次并保存 detached 36 层 GQA K/V；
2. S 在 forward 中计算一次并保留四 token 的可微层间图及 K/V；
3. A 每两层形成一个 non-reentrant checkpoint segment；
4. backward 只重算该 A segment；
5. N、S、ViT、connector 不重算。

这与历史 v4 的“Molmo prefix + Expert 联合重算”不同。`layers_per_segment=2` 只控制 Action Expert segment 大小。

关闭 activation checkpoint 不改变 forward、mask、loss 或可训练参数，因此理论上不改变模型性能；它只在相同随机数、相同算子条件下改变激活保存/重算和浮点舍入调度。对性能结论应通过固定 seed 数值回归确认，不能用旧 v4 step time 外推。

## 9. 推理 cache

每个观测只进行一次 prefix prefill：

1. 原生 processor 生成 N；
2. ViT/connector/WTE 与 36 层 N 在 `no_grad` 下运行，保存 N K/V；
3. PointSeg 生成 S；
4. S 在 `no_grad` 下运行，保存 S K/V；推理无需保留 projection 梯度；
5. 偶数层 cache 保存直接可拼接的 N+S K/V；
6. 奇数层提前执行该层 trainable cross K/V projection，并保存 projected N+S K/V；
7. 10 次 Euler 去噪只重跑 A，复用同一份 prefix cache。

推理 cache 中 N/S 均 detach 是正确的，因为推理不做 backward；这不改变训练时 S K/V 必须可微的合同。

## 10. 与 WEPVLA 的同构边界

### 10.1 保持一致

- UMI processor、normalizer 与 pose9/action 数据合同；
- PointSeg 的 FG/BG 全局特征与 foreground local 特征；
- PointAction 仅在 Expert 前融合一次；
- action suffix 只含 Ego/World actions；
- 偶数 action-action 双向、奇数 pure cross；
- Ego/World shared Expert 与 paired positions；
- `left_compose_ego` DoubleFlow、sidecar target 和最终 Ego 执行；
- action/worldflow/pointseg loss 组成与权重由同一 CLI 控制。

### 10.2 有意差异

- VLM 从 SmolVLM 500M 替换为完整 Molmo2-ER；
- VLM/Expert 从 16/16、较窄 hidden 改为 36/36、2560/1920；
- N 不读取 S，以保护 Molmo 原生表征；
- S 是独立 scene-shadow，而不是 native IMAGE block 成员；
- Full-Molmo2-ER + Expert 主计算使用 BF16；
- N K/V 只算一次，activation checkpoint 只重算 A。

这些差异必须在实验报告中显式记录。尤其不能把“action 交互对齐 WEPVLA”误写成“所有 prefix hidden 与 WEPVLA 完全相同”。

## 11. BF16 精度合同

正式命令保留：

```text
accelerate --mixed_precision=no
policy.use_amp=false
```

与此同时：

- frozen Molmo 权重和计算为 BF16；
- trainable Action Expert 权重和计算为 BF16；
- PointSeg/部分 task projection 为 FP32；
- 进入 N/S/A backend 前显式、可微地 cast 到目标 dtype；
- 不启用 autocast/GradScaler。

BF16 是受显存和完整 36+36 层规模约束的有意数值差异。它不改变结构语义，但相较原 WEPVLA FP32 可能产生不同舍入、收敛噪声和最终指标；必须作为实验变量记录，不能承诺逐位等价。

## 12. Native vision fast path

固定 256×256 输入下，两个 378×378 crop 像素完全相同时，可只运行一次 ViT，再按官方 `(-3,-9)` 层特征复用到两 crop。运行时必须先断言相等；不相等时自动回退官方 multi-crop path。

CPU image fast path 也必须逐字段匹配 processor 合同。两类 fast path 都只能复用数值相同的中间量，不得改变 token template、connector、pooling 或 N mask。

## 13. Checkpoint schema 与 fresh train

以下旧 checkpoint 均不兼容 v5：

- detached scene/action suffix 版本；
- v3 全 prefix 双向或位置复用版本；
- `molmo_native_hybrid_wepvla_expert_v4`；
- 任何 `molmo_inference_only=true` 的旧 Full-Molmo checkpoint。

原因不只是 config 字符串不同：v4 的 N hidden/K/V 已被 FG/BG 条件化，scene/action 学到的是另一种每层 memory；v5 的 N/S 分流、position 和 cache 语义均改变。即使 key/shape 可以对上，也不能直接 resume 或把旧 optimizer state 当作 v5 状态。

v5 主实验必须：

```text
--policy.type=smolvla
--resume=false
--policy.full_molmo_topology=molmo_native_readonly_scene_shadow_wepvla_expert_v5
--policy.molmo_inference_only=false
```

不提供旧 `--policy.path`，从 pretrained Molmo2-ER + fresh trainable Expert/scene/point modules 开始。未来若做权重迁移，必须另设 topology/version、转换清单与独立消融。

## 14. 结构验收

### 14.1 N 不变量

同一输入比较 scene 全关与 scene 激活：

- N token ids/roles/validity 完全一致；
- N position ids 完全一致；
- N attention mask 完全一致；
- 每层 N hidden/K/V 在相同 dtype/kernel 下相等；
- N tensor 全部 `requires_grad=False`。

### 14.2 Mask 真值

必须逐层断言：

```text
N -> N = official native mask
N -> S = false
N -> A = false
S -> N = true for valid native keys
S -> S = bidirectional for valid scene keys
S -> A = false
A_even -> N/S/A = true for valid keys
A_odd  -> N/S = true for valid keys
A_odd  -> A = structurally absent
```

### 14.3 Freeze 与梯度

- 所有 Molmo parameter `requires_grad=False` 且 `grad is None`；
- N cache 不带梯度；
- S cache 带梯度；
- FG/BG projections、PointSeg、PointAction、Expert 梯度 finite；
- checkpoint on/off 的 fixed-batch forward 和梯度在 BF16 容差内一致。

### 14.4 Cache

- 推理 cache 与 non-cache action 输出在 BF16 容差内一致；
- 奇数层 cache 标记为 cross-projected；
- 每个新观测只 prefill 一次 N/S；
- 10-step denoise 不重新运行 N/S。

### 14.5 WEP 回归

- UMI processor 与基准一致；
- PointAction 仍只用 foreground local points；
- Ego/World action 同步位置、共享 Expert；
- flow/noise/sidecar/gripper 输出合同一致；
- DDP 第一个 optimizer step 后无意外 unused trainable tensor。

## 15. 启动依据

不要从本文拼接旧 shell 脚本。训练与评测只使用 [WEPVLA_V043_DoubleFlow_MolmoER2.md](../benchmarks/song_real_libero/WEPVLA_V043_DoubleFlow_MolmoER2.md) 中的直接多行命令。

本文只更新设计合同，没有启动训练。v4 的 batch、显存、step time、loss 或评测结果都不能直接作为 v5 的性能证据。
