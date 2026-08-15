# WEP-VLA 当前对话归档（截至 2026-08-15）

本文是当前长对话的决策、实验事实和可恢复执行状态归档。它不是逐字聊天记录；其用途是让后续会话不依赖聊天上下文即可继续工作。未实际运行的内容会明确标为“未运行”，不得当作性能证据。

## 1. 最终目标

1. 先证明输入层双视角策略在 LIBERO-10 的 `10 tasks × 50 episodes` 上不低于、最好超过单视角基线。
2. 再证明开启 WorldFlow 能进一步提升同一策略。
3. 最终候选必须同时包含 multiview 与 WorldFlow，并通过完整 2×2 因果消融：
   - 双视角 + WorldFlow；
   - 主视角 + WorldFlow；
   - 双视角 + World→Ego disabled；
   - 主视角 + World→Ego disabled。
4. 最终正式成绩必须超过同协议 Ego-only baseline，并且 multiview 与 WorldFlow 两项简单效应均为正。
5. 禁止删除 `/opt/data/private` 下任何文件。

## 2. 用户明确的设计约束

- 双视角只能在输入点云阶段融合，不在模型层面增加“多视角模块”。模型始终只接收严格 `10,000` 个点，单/多视角天然兼容。
- 不采用固定 `99:1` 视角配额；不使用局部最近点一一交换、几何遮挡、人工语义、人工目标点或 task-specific 技巧。
- 方法要简洁、通用，不为某个 LIBERO task 写规则。
- 不引入任何 learned gate；`worldflow_ego_residual_gate_init` 必须为 `None`。
- 不冻结 Ego 分支；Ego 与 World 应共同训练。
- 不修改 LitePT 等外部函数库。
- 保持“双 point-action adapter + 单共享 point-action expert”总体框架。
- World 与 Ego 应描述同一物理轨迹并保持坐标共轭关系，World 信息必须能实际影响最终 Ego action。
- 不靠辅助损失或梯度补丁掩盖结构问题；最终仍使用原 action loss，重点解决输入表示和双流实际适配。
- 方法验证不能只看很少训练步；正式全任务微调采用 `10 tasks × 50 episodes`，至少覆盖一个完整 epoch。
- 训练使用本地 4 GPU；worker 为 `12`；以后训练全部启用 W&B。
- 长任务必须放在 tmux，避免网络断连造成进程退出。
- checkpoint 少于 100 步没有作为正式候选保存的必要。
- 所有实验脚本、日志、cache、checkpoint 和 artifact 统一放在本实验根目录下。

## 3. 评测协议与稳定性结论

### 3.1 权威固定动作协议

- suite：`libero_10`，10 tasks，正式 Full 为每 task 50 episodes，共 500 states。
- policy-noise seed：`0`；env seed：`7`。
- `inference_batching_mode=fixed_barrier`。
- 本地权威布局：4 GPU、总环境 worker `30`、inference batch `30`。
- 使用 canonical exact-action cache；复用同一 canonical cache 时，action 可逐值一致重放。
- 新建 CUDA canonical rollout 即使固定 seed/slot，也不能宣称跨独立新 cache 逐值完全一致；最强复现依据仍是复用 exact-action cache。

### 3.2 A800 约束

- 只使用物理 GPU `3,4,5`。
- 总环境并行上限 `40`，inference batch 固定 `40`。
- 单 GPU 命令如果只写 `--inference-batch-size 50`、没有显式 fixed barrier，实际是 dynamic batching；此前 baseline `99/100` 和 V32 的 `91/95/91` 都属于动态队列结果，不能覆盖权威 fixed-barrier Full500。

### 3.3 当前权威基线

- 同协议 Ego-only baseline：`472/500 = 94.4%`。
- 同一 canonical 结果的固定分层 IDs `0,5,10,...,45`：`95/100`。
- 历史 `481/500` 属于旧协议，只能追溯，不能作为当前 matched baseline。

## 4. WorldFlow 已达到的阶段性目标

权威通过版本为 V32 step100：

- 正常单视角 WorldFlow：`475/500 = 95.0%`；相对 matched baseline `+3`。
- 同 checkpoint 关闭 World→Ego：`469/500 = 93.8%`。
- WorldFlow 完整因果贡献：`+6`。
- 固定分层 Broad：正常/消融为 `96/100` 与 `93/100`。

V32 模型 checkpoint：

```text
/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811/singleview_worldflow/libero10_500ep/training/v32_from_v14_worldbnfixed_w2e_p75_anchor_residual_rate_coordframe_bodyframe_ego_tangent_common005_world02_residual4_4gpu_b32_w12_1080steps/checkpoints/000100/pretrained_model
```

`model.safetensors` SHA256：

```text
c258303b70d4cab64f89d93c825905813f6f49fbb08cf282b5d4321e1fdf1fb4
```

V32 是非对称但物理共轭的双流：每个去噪时刻都从当前 Ego flow state 通过 SE(3) 共轭得到 World state，World body-frame endpoint residual 再回写 Ego。它没有 learned gate，但训练中包含 stochastic depth、residual anchor、Ego-tangent shared-gradient protection 和坐标系增强。无论后续方案如何演进，当前 WorldFlow 性能事实以 `475/500` 与 causal `469/500` 为准。

权威 gate：

```text
singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v32_residual_rate_coordframe_bodyframe_ego_tangent_p75_local4_fixedbarrier_full500_matched_worldflow_gate_step000100_baseline472_recheck.json
```

## 5. 双视角输入探索的主要结论

### 5.1 已否决的通用输入方案

- V41 novelty-union-1cm：Broad `95/100`；Full500 在 `328/355`、27 failures 时已无法达到 V32 的 `475/500`，完成状态相对主视角为 `-10`。
- V43 union-FPS：Broad 最高 `93/100`。
- V44 novelty-3cm：Broad 最高 `93/100`。
- V45 voxel-cover-FPS-1cm：Broad 最高只能追平 V32 的 `96/100`，不能证明净提升。
- V46 multiscale novelty 1cm/3cm zero-shot：`93/98` 后第 5 个失败，最高 `95/100`；稳定恢复 V32 的 4 个失败，但新增 5 个失败。

FPS 被排除的核心原因不是“它不能产生多视角点”，而是全局 FPS 会重新分配大量主视角点，扰动已经训练成熟的单视角输入分布；历史 Broad 没有给出不降证据。当前保留的方法只增加主视角空间覆盖之外的副视角信息，并避免固定 quota。

### 5.2 V47/V48 训练结论

V47 使用完整 paired 数据和统一较高 LR 训练一整个 epoch：所有双视角 checkpoint 都未超过 Broad baseline；最终 step1564 的主视角 + World 也从 V32 的 `96/100` 降到 `93/100`，说明存在共同策略漂移/遗忘。

V48 将 point-input 路径设为高 LR、其他路径保持非零低 LR。用户在实际约 step1108 时要求停止，保存 checkpoint 为 `260/520/780/1040`。四个 checkpoint 全部未严格超过 `95/100`；最好 step520 为 `95/100`。

V48 step520 的三臂机制诊断：

- 双视角 + World：`95/100`；
- 主视角 + World：`97/100`；
- 双视角 + World→Ego disabled：`89/100`；
- multiview 因果贡献：`-2`；
- WorldFlow 因果贡献：`+6`。

更关键的代码审计发现：V48 所谓高 LR point-input 组只包含 Ego 点路径；World 的 `scene_encoder / scene_context_proj / point_action_adapter` 仍位于低 LR World 组。step260 时 World point-action adapter 漂移仅为 V47 的 `5.52%`，而 Ego adapter 为 `106.83%`。因此负交互的直接机制假设是：双视角改变了点云输入，但 World 点路径没有与 Ego 点路径对称适配。

### 5.3 V50 保守尺度 zero-shot 边界

- 主视角细覆盖尺度固定为 `1 cm`。
- 副视角新颖性粗尺度测试：scale4 与 scale5。
- scale4 在 `93/97` 时 4 failures，理论最高 `96/100`；同 97 states 的 V32 为 `94`，恢复 3 个 V32 失败但新增 4 个。
- scale5 在 `92/97` 时 5 failures，理论最高 `95/100`。
- scale4 是唯一仍可能追平 V32 Broad `96/100` 上界的保守输入，因此作为下一次充分训练的输入候选；zero-shot 本身没有通过。

## 6. 当前待运行方案 V51

### 6.1 实质内容

V51 不是新增模型 forward、门控或辅助 loss。它组合两个具有直接实验依据的改动：

1. 输入级保守 multiscale novelty：
   - cache 和主视角覆盖的基准体素为 `1 cm`；
   - 主视角先以 1 cm 网格保留空间覆盖；
   - 副视角只有在主视角之外形成 4 cm 粗网格新颖 cell 时才插入；
   - `4 cm` 是副视角新颖性阈值，不是 cache/PointSeg 的基础分辨率；
   - 不设视角 quota；输入严格为 `9,500 scene + 500 gripper = 10,000` 点。
2. Ego/World 直接点路径对称适配：
   - Ego 与 World 的直接 point path 都进入 `5e-8` 组；
   - 其他 action/language/shared/residual 路径以非零 `5e-9` 联合训练；
   - 物理梯度角色与 optimizer LR group 解耦，避免 World point tensor 被误当作 Ego/shared 而清掉 World 梯度；
   - 不冻结 Ego，不改变 V32 forward、模型架构或 action loss。

这属于针对输入域变化的对称路径适配，而不是新增推理规则。它是否有效必须由 rollout 证明，当前没有性能结论。

### 6.2 代码状态

```text
worktree: /home/liusong/ProgramFiles/Huggingface/lerobot_v51_conservative_symmetric_point_adaptation
branch:   wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation
HEAD:     93e2d8a3177a8addc40229da00962fcc2e7b7100
```

最近已记录的测试：

- SE3/WorldFlow：`78/78`；
- PointSeg/cache：`35/35`；
- training/cache/gradient entry：`16/16`。

真实 V32 optimizer/gradient role preflight：

- 1,243 个 trainable tensors，无重叠、无遗漏；
- 四组 tensor 数量：`153 / 1063 / 25 / 2`；
- 234 个 World-only point tensors 保留 World 梯度；
- 829 个 Ego point tensors 保持 Ego-tangent protection。

artifact：

```text
joint_multiview_worldflow/libero10_500ep/artifacts/v51_real_checkpoint_optimizer_and_gradient_role_preflight_v2.json
```

### 6.3 Cache 合同

目标目录：

```text
joint_multiview_worldflow/libero10_500ep/pointseg_cache_multiscale_novelty_union_1cm4cm
```

合同：

- cache schema version `12`；reader 兼容不可变 version 11 primary cache；
- `137,590` samples；4 GPU；36 shards；
- 基准 voxel size `0.01 m`；
- secondary novelty scale `4.0`，即 `0.04 m` 粗新颖性判断；
- current/future 都是严格 10,000 点；
- cache 完成后逐 shard 中点做 36-sample online/cache exact-index 对照；
- 审计未通过时训练在任何 update 前退出。

脚本：

```text
scripts/run_v51_multiscale_novelty_1cm4cm_cache_4gpu.sh
scripts/run_v51_cache_exact_index_audit.sh
scripts/launch_v51_cache_and_exact_audit_4gpu_tmux.sh
```

### 6.4 训练合同

- source：V32 step100；
- dataset：完整 `10 tasks × 50 episodes`；
- paired primary+dual coverage：`275,180` samples；
- 4 GPU；batch `44/GPU`；global batch `176`；worker `12`；
- `1,564` optimizer steps，处理 `275,264` samples，即完整 paired epoch + 84 samples；
- W&B enabled；
- checkpoint：`260/520/780/1040/1300/final1564`；不保存 `<100` checkpoint；
- cache v12、1 cm 基准/scale4 契约和 exact-index audit 是硬前置条件。

训练输出：

```text
joint_multiview_worldflow/libero10_500ep/training/v51_from_v32step100_multiscale_novelty1cm4cm_paired_symmetricpoint5e8_policy5e9_devicebound_4gpu_b44_w12_1564steps
```

脚本：

```text
scripts/run_v51_from_v32step100_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps.sh
scripts/launch_v51_paired_symmetricpoint_training_4gpu_tmux.sh
```

### 6.5 预注册筛选标准

Broad100 固定分层三臂：

- 双视角 + World 必须 `>95/100`；
- 严格高于同 checkpoint 主视角 + World；
- 严格高于同 checkpoint 双视角 + World→Ego disabled。

Full500 完整 2×2：

- 双视角 + World 必须 `>472/500`；
- 不低于 V32 的 `475/500`；
- 严格高于主视角 + World；
- 严格高于双视角 + World→Ego disabled。

预注册 artifact：

```text
joint_multiview_worldflow/libero10_500ep/artifacts/v51_preregistered_training_and_2x2_causal_gate_protocol.json
```

## 7. 本轮“恢复训练”后的实际状态

用户已明确说“恢复训练，尽量找到实质性的解决，而不是缝缝补补”。随后又明确要求“cache 以 1cm 为准”。当前解释和执行合同为：基础 cache/PointSeg voxel 固定 `1 cm`；`4 cm` 只作为副视角新颖性过滤尺度。

已完成的只读恢复核对：

- 完整读取 `PROJECT_HANDOFF_2026-08-11.md`；
- V51 worktree clean，HEAD 正确；
- 4 张 RTX 4090 空闲；
- V51 cache 目录不存在；
- V51 training output 不存在；
- V51 cache/training tmux 均不存在；
- cache generation 与 training 脚本 SHA256 和预注册一致；
- 底层 exact-index auditor SHA256 `9976f...` 和预注册一致；
- evaluation runner 实际对应 `eval_libero10_v12_one_checkpoint_4gpu_alltasks10ep_b30.sh`，SHA256 `d2685d...`，和预注册一致；
- `/opt/data/private` 可用空间约 13 TiB；历史 1cm/3cm cache 约 31 GiB。

刚启动的回归测试命令被用户切换到“归档对话”请求时中断，约运行 2.5 秒；没有得到新一轮 pytest 结果，不能把这次中断写成通过或失败。历史已通过的 `78+35+16` 结果仍有效，但下一次真正启动 cache 前应快速重跑。

截至本文写入时：

- 没有 V51 cache 生成进程；
- 没有 V51 训练进程；
- 没有 V51 评测进程；
- 没有创建 V51 cache/training tmux；
- 没有删除 `/opt/data/private` 下任何文件。

## 8. 下次继续执行的严格顺序

1. 在 V51 worktree 重跑三组回归测试，必须全部通过。
2. 用 tmux 启动 4-GPU cache 生成；先做 4-sample smoke，再生成 137,590-sample 正式 cache。
3. 自动执行 36-shard exact-index audit，确认 1 cm 基准、scale4 副视角新颖性、严格 10k、unique index 和 gripper exact 全部成立。
4. cache/audit 通过后，才用 tmux 启动 4-GPU、batch44/GPU、worker12、W&B 的 1,564-step 完整 paired epoch。
5. 训练期间只监控 OOM/NaN、LR/梯度角色和 W&B loss 健康度，不用很少步数 rollout 代替充分训练。
6. 训练完成后按预注册顺序做 Broad 三臂筛选；只有同时证明 multiview 和 WorldFlow 正贡献的 checkpoint 才进入 Full500 2×2。
7. Full500 若未同时满足 `>=475`、高于主视角臂、高于 World 消融臂，则 V51 不作为最终候选；保留全部输出，不删除任何 `/opt/data/private` 文件。

## 8.1 新容器实际恢复进展（2026-08-15）

- 三组回归重新通过：`78 + 35 + 16`。
- 4-sample smoke 与正式 v12 cache 已完成；正式 cache 为 `137590`
  samples、`36` shards、1 cm 基准、scale4 副视角新颖性。
- 36-shard online/cache exact-index audit 全部通过：严格 10k、unique
  indices、gripper exact 均为 true。
- 首次训练启动在 step 0 的 dataset 构建阶段停止：paired contract 将
  支持的 dual v12 / primary v11 storage schema 差异误判为语义不兼容；
  没有 optimizer update 或 checkpoint。
- 通用修复 commit 为
  `9db8504a2c575ad386f1d90efa98784c0ea8d701`，只从 paired semantic
  equality 中移除 schema version；两个 cache 各自仍由 reader 严格限制为
  supported versions `(11,12)`。新增测试后完整三组为 `130/130`。
- 原失败 output/log 保留；重启必须使用独立 `v51r1_schemafix` runtime
  入口、输出目录与 W&B run id，不覆盖原证据。

## 9. 主要权威文档

- 项目完整交接：`/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/PROJECT_HANDOFF_2026-08-11.md`
- 单视角 WorldFlow 长期进展：`SINGLEVIEW_WORLDFLOW_PROGRESS_2026-08-13.md`
- novelty-union 历史归档：`MULTIVIEW_NOVELTY_UNION_ARCHIVE_2026-08-13.md`
- 本文：`CURRENT_CONVERSATION_ARCHIVE_2026-08-15.md`
