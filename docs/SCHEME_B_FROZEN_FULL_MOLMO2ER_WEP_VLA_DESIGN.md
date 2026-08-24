# Scheme B v5：原生只读 Molmo2-ER + Scene Shadow + WEP Action Expert

- 当前拓扑标识：`molmo_native_readonly_scene_shadow_wepvla_expert_v5`
- 生效仓库：`/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song`
- WEP 结构基准：`origin/wep_vla_v0.4.3_multiview_doubleflow`，提交 `da0ad03bf7eff2c6e9edcf04e1b324bebbdf93dd`
- 分布式策略：普通 DDP；当前不使用 DeepSpeed/ZeRO
- 完整训练和评测命令：[WEPVLA_V043_DoubleFlow_MolmoER2.md](../benchmarks/song_real_libero/WEPVLA_V043_DoubleFlow_MolmoER2.md)

本文件是当前 Full-Molmo2-ER WEP-VLA 的运行合同。v5 只替换冻结 VLM 条件分支，并在不改变 Molmo2-ER 原生图文表征的前提下，对齐 WEPVLA 的 PointAction、偶/奇层 Action Expert 和 World–Ego DoubleFlow。历史 v4 的 active FG/BG-in-IMAGE 图不再有效。

## 1. 三条严格分离的流

### 1.1 N：Native Molmo image/language

`N` 只包含 Molmo processor 产生的官方图文序列：

```text
N = [official BOS / chat TEXT / 410 IMAGE positions / LANGUAGE / right PAD]
```

N 的物理 token 顺序、有效位、position id 和官方 hybrid attention mask 均保持不变。FG/BG 不插入 IMAGE block，也不移动 LANGUAGE/PAD。

N 独立执行完整 36 层 Molmo2-ER：

- 原生 ViT、connector、WTE、36 个 decoder block 和 final norm 全部冻结并处于 `eval()`；
- 整条 N forward 在 `torch.no_grad()` 下执行；
- 每层 N hidden/K/V 均不含 FG/BG 或 action 信息，且 `requires_grad=False`；
- N 的 attention kernel 只看到官方 N 序列，不通过“扩展序列后再 mask”来近似隔离；
- `lm_head` 不实例化。

因此激活 FG/BG 后，Molmo2-ER 的 image/language hidden 和 K/V 仍与该实现的原生 N 路完全相同。N 永远不读取 S 或 A。

### 1.2 S：Scene-shadow prefix

`S` 是非 action 的可训练 scene prefix：

```text
S = [EGO_FG] [EGO_BG] [WORLD_FG] [WORLD_BG]
```

四个 token 由 PointSeg 全局 FG/BG 特征分别投影到 Molmo hidden 2560。它们在公共 API 的 prefix 中逻辑排列为 `[N][S]`，但 backend 在进入 decoder 后将 N 与 S 拆开执行；S 不是原生 Molmo token，也不属于 IMAGE block。

S 在每一层复用同一个冻结 Molmo block：

- S query 读取该层 detached N K/V；
- 四个 S token 在同一个双向 scene block 内互读；
- S 不读取 action；
- S 的每层 K/V 不 detach，供所有 Action Expert 层使用；
- action loss 可穿过冻结 Molmo 运算更新 FG/BG projection 和上游 PointSeg，但不会产生 Molmo 参数梯度。

### 1.3 A：WEP paired action suffix

`A` 只包含成对的 noisy action token：

```text
A = [EGO_ACTION x 32] [WORLD_ACTION x 32]
```

Ego/World action 保持 WEPVLA 的共享 Expert、共轭运动编码和相同步号位置配对。最终只执行 Ego 10D action；World action 是同一物理轨迹的共轭监督/隐变量。

## 2. 每层可见性合同

三条流的固定信息方向是：

```text
N -> S -> A
\----------> A

禁止：S -> N、A -> N、A -> S
```

更精确地说：

| Query | 偶数层 `0,2,...,34` 可读 K/V | 奇数层 `1,3,...,35` 可读 K/V |
|---|---|---|
| N | 官方 N mask | 官方 N mask |
| S | 全部有效 N + 全部有效 S | 全部有效 N + 全部有效 S |
| A | 全部有效 N + S + Ego/World A | 全部有效 N + S |

偶数 Action 层使用 fused/joint MHA：全部 64 个 Ego/World action token 双向互读，同时读取 N+S。N/S 不参加 action residual 更新，也不能读取 noisy action。

奇数 Action 层使用 pure cross-attention：action query 只读取本层 N+S memory，不存在 action-action attention。这与 WEPVLA 的偶数 joint self-attention、奇数 pure cross-attention 调度一致。

逻辑 block mask 可写为：

```text
训练公共布局：[N][S][A]

N rows: [M_native, 0, 0]
S rows: [1,        1, 0]
A even: [1,        1, 1]
A odd : [1,        1, 0]
```

这里的 `1` 还要与每个样本的 validity mask 相与。实现上 N 必须作为独立 attention problem 运行；仅在更长 SDPA 中把 N→S/A mask 置零不足以保证 N kernel 形状和数值路径不变。

## 3. Position id 与 cache 语义

- N 使用官方连续 position id，LANGUAGE 不因 S 而平移；
- Ego FG/BG 使用最后一个 IMAGE position 后的 `[+1,+2]`；
- World FG/BG 复用 Ego FG/BG 的两个 position id，以保持 WEPVLA 的 Ego/World scene 配对；
- Ego/World action 的同一时间步复用相同相对位置；action 的 base offset 位于全部有效 prefix position 之后；
- scene position 与部分 native LANGUAGE position 可能数值重叠，这是在“不改变 N”与“保持 WEP scene 配对”两个约束下的有意选择，不表示 S 被插入 LANGUAGE 或 IMAGE 序列。

每层 cache 保存的是进入该层 attention 前的 K/V：

```text
native_cache[layer] = detached N K/V
scene_cache[layer]  = differentiable S K/V
```

偶数层直接拼接 N/S/A K/V；奇数层先用该层 Action Expert 的 cross K/V projection 将 N+S memory 投到 Expert cross-attention 空间。

## 4. 冻结与梯度边界

完整训练梯度路径为：

```text
RGB + instruction
  -> frozen Molmo N [eval + no_grad + detached per-layer K/V]
                                                     \
PointSeg -> FG/BG projections -> frozen Molmo S graph -> scene K/V -> Action Expert -> flow loss
foreground local points -> PointAction ---------------------------> Action Expert -> flow loss
```

必须同时满足：

1. Molmo ViT、connector、WTE、36 blocks 和 final norm 的 `parameter.grad` 始终为 `None`；
2. N hidden/K/V 的 `requires_grad` 始终为 false；
3. S hidden/K/V 保留输入梯度；
4. action loss 对 Ego/World FG/BG projection、PointSeg、PointAction 和 Action Expert 的梯度 finite 且非零。

`--policy.molmo_inference_only=false` 是强制值。它不表示 N 可训练；它表示整个 backend 不能被视为纯 inference-only，因为 S 仍需穿过冻结 Molmo block 保留输入梯度。将其设为 true 会切断 WEP scene 路径。

## 5. Activation checkpoint 与显存

当前强制建议：

```text
molmo_gradient_checkpointing=true
molmo_gradient_checkpointing_layers_per_segment=2
```

v5 的 checkpoint 只重算 A（Action Expert）segment：

- N 只前向一次，并保留 detached per-layer GQA K/V；
- S 只前向一次，四个 scene token 的小型可微图和 per-layer K/V 保留到 backward；
- backward 仅按 segment 重算 Action Expert；
- 不重算 ViT、N 或 S；
- 推理不使用 activation checkpoint。

因此 `molmo_gradient_checkpointing_layers_per_segment` 表示每个 Action Expert segment 的层数，不再表示“Molmo prefix + Expert”耦合 segment。关闭该开关不改变数学模型或 token 交互，只增加 A 激活显存、减少 backward 重算；batch 上限和 step time 必须在 A800 上实测。

## 6. BF16 是有意的数值差异

命令使用 `--mixed_precision=no`、`--policy.use_amp=false`，但 Full-Molmo2-ER 的冻结 Molmo 和 Action Expert 权重/主计算显式为 BF16。PointSeg 和若干 task projection 仍可保持 FP32，并在 N/S/A 边界通过可微 `.to(BF16)` 接入。

这意味着：

- 不使用 autocast 或 GradScaler；
- BF16 是为了容纳完整 36 层 Molmo2-ER 与 36 层 Expert 的有意实现选择；
- 它不会改变注意力拓扑、loss 定义或 trainable allowlist；
- 它与原 WEPVLA 的 FP32 数值路径不是逐位等价，比较结果时必须把 precision 记录为受控差异；
- 不应为追求表面“严格同构”而在未重新测量显存和稳定性前把整个模型切到 FP32。

## 7. 保持不变的 WEPVLA 前端与 DoubleFlow

### 7.1 UMI processor

`src/lerobot/processor/umi_processor.py` 继续保持：

```text
UMI feature construction -> policy normalizer -> model
```

Molmo processor 只负责原生 RGB/language，不接管 UMI pose9/action 坐标合同。

### 7.2 PointAction adapter

- 只取 foreground local point tokens；
- `PointActionSelfAttention` 为 4 heads、dropout 0；
- 在进入共享 Action Expert 前只融合一次；
- background local points 不直连 action；
- padded point/action 不参与注意力；
- Ego/World 各自保留 direct point path。

这条 local `[foreground point, action]` 融合属于 PointAction adapter，不是额外增加的一层 VLM/Expert joint MHA。进入 36 层 Expert 后，A 的交互严格按第 2 节执行。

### 7.3 World–Ego DoubleFlow

- Ego/World 表示同一条物理轨迹；
- commanded World target 来自 sidecar；
- World noise 使用基准的 `left_compose_ego`；
- 两条 action stream 共享单个 36 层 Expert；
- pose9 flow matching 两侧直接监督；
- 无 learned World-to-Ego gate；
- 推理执行最终 Ego 10D action。

## 8. Processor fast path

固定 256×256、单图、两 crop 合同可使用 CPU batched Torch fast path：一次 resize/normalize、矩阵化 patch、预计算 pooling index，并按官方顺序产生 native processor 字段。只有 resize、patch、pooling、mean/std、placeholder 等合同全部匹配时才启用；否则自动回退官方 slow processor。

fast path 只优化数据准备，不改变 N/S/A token、mask、position id、loss 或梯度图。完整训练的 `data_s` 还包含 DataLoader、视频/Parquet 读取、点云 cache 和 collate，不能用孤立 processor 耗时直接推断。

## 9. Checkpoint 兼容性

v5 必须 fresh train：

- 不允许从 `molmo_native_hybrid_wepvla_expert_v4` full checkpoint resume；
- 不允许从旧 detached-scene/suffix、v3 或 v4 checkpoint 直接评测；
- 即使部分参数名称和形状相同，旧权重是在不同的信息流、mask、position/cache 语义下优化的，不能视为 v5 warm start；
- policy `config.json` 必须保存 `full_molmo_topology=molmo_native_readonly_scene_shadow_wepvla_expert_v5`；
- 若未来研究显式权重迁移，必须作为独立实验实现转换器和消融，不能复用本合同的 fresh 训练结论。

旧 tmux session 或旧 optimizer state 也不是 v5 可恢复状态。当前命令使用 `--policy.type=smolvla --resume=false`，且不提供旧 `--policy.path`。

## 10. 启动与验证门禁

启动前必须确认：

1. `full_molmo_topology=molmo_native_readonly_scene_shadow_wepvla_expert_v5`；
2. `molmo_inference_only=false`；
3. `molmo_gradient_checkpointing=true` 且日志明确只 checkpoint Action；
4. N 的 token 顺序、mask、position、每层 hidden/K/V 在 S 开关前后不变；
5. `N -> S`、`N/S -> A` 存在，`S/A -> N` 和 `A -> S` 不存在；
6. 偶数 A 层可读 N+S+全部 A，奇数 A 层只读 N+S；
7. Molmo 参数无梯度，S projection 与 Action Expert 梯度非零；
8. checkpoint 开/关时固定输入的前向与梯度在 BF16 合理容差内一致；
9. UMI、PointAction 和 DoubleFlow 回归通过；
10. DDP 第一个 optimizer step 后不存在意外 unused trainable parameter。

当前文档只定义设计和直接命令，不代表已启动训练。真实 batch 上限、峰值显存和 step time 仍应以新的 v5 fresh process 实测，不能沿用 v4 结果。
