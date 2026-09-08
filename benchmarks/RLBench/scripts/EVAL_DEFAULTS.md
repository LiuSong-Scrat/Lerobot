# RLBench 正式测评默认口径

更新日期：2026-08-30。

## 默认七任务

日常综合测评默认排除已经长期保持高成功率的三个任务：

- `close_box`
- `close_fridge`
- `toilet_seat_down`

默认测评以下七个任务：

- `close_laptop_lid`
- `phone_on_base`
- `stack_wine`
- `sweep_to_dustpan`
- `take_frame_off_hanger`
- `take_umbrella_out_of_umbrella_stand`
- `water_plants`

上述三个排除任务并未删除，完整十任务或专项回归时仍可显式指定。

## 模型输入

| 参数 | 默认值 |
|---|---|
| 总点数 | 20,000 |
| 虚拟夹爪点 | **250** |
| 场景点 | 19,750 |
| 虚拟夹爪模板 | REAP |
| 夹爪模板版本 | `libero_reap_four_box_physical_opening_geom0p06_rlbench_offset0p09_v4` |
| 夹爪局部偏移 | `[0, 0, -0.09]` m |
| 相机 | `front` RGB + point cloud |
| 图像大小 | 256 |

模型 checkpoint 自身配置继续决定 PointSeg/WorldFlow 网络参数。当前 vfinal checkpoint 060000 加载的是：

- `pointseg_min_foreground_points=5000`
- `pointseg_foreground_ratio=0.025`
- `pointseg_background_ratio=0.025`
- `worldflow_enable=true`

## 动作与夹爪协议

| 参数 | 默认值 |
|---|---|
| `gripper_mode` | `delta_width_initial_sync` |
| open delta threshold | **0.0025 m** |
| close delta threshold | **0.0025 m** |
| delta alignment | `current_minus_previous` |
| `action_index` | 0 |
| 模型 `n_action_steps` | 16 |
| `exec_action_steps` | 16 |
| 模型 action chunk | 32 |
| flow-matching denoising steps | 10 |
| `max_model_calls` | 20 |

电话 25 calls、浇花 15 calls、exec 24/32、denoising 20 steps 等均视为历史专项实验，不是当前全局默认。

## 仿真与控制

- CoppeliaSim 使用当前验证过的 legacy 环境。
- planner budget 默认 10 ms。
- `execution_mode=dataset_step`，arm action mode 为 planning。
- 保留 PointAct/PyRep 稳健性处理：
  - PointAct 工作空间裁剪；
  - Mover 最多重发 10 次；
  - 普通位置容差 0.05 m，夹爪位置容差 0.02 m；
  - 到位后再切换夹爪；
  - PyRep RML 兼容补丁；
  - 控制失败后丢弃旧 chunk 剩余动作并重新推理；
  - 默认关闭规划 collision checking，允许任务所需的主动接触。
- `take_umbrella_out_of_umbrella_stand` 特例：开启 collision checking。
- `water_plants` 特例：第一次闭爪后锁定关闭；植物碰撞 enabled，水滴碰撞 original。

1000 ms、100 ms、40 ms planner budget 均属于此前实验配置，不是当前默认。

## 随机种子与输出

- 默认 scene/episode seed 从 0 开始。
- 默认模型噪声 seed：`20260801`。
- episode 数量不永久固定，按实验 profile 或当次要求使用 10、25、50 等。
- 默认保存视频、`summary.json`、参数日志、控制日志、执行动作、模型 chunks 和 executed-action alignment。
- action 点云可视化与逐帧 PLY 默认关闭；专项诊断时显式开启。

## 当前七任务 profile

```text
0830_vfinal_10tasks_060000_gripper250_delta0025_calls20_planner10_7tasks_10eps
```

统一入口：

```bash
bash benchmarks/RLBench/scripts/s4_eval.sh \
  --checkpoint 0830_vfinal_10tasks_060000_gripper250_delta0025_calls20_planner10_7tasks_10eps \
  --tasks all
```

历史 checkpoint/profile 中显式登记的参数仍优先于这些共享 fallback，从而保证历史实验可复现。
