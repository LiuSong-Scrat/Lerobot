# MuJoCo 3.3.4 接触求解实验

本改动不是 MuJoCo 引擎补丁，也不宣称已经解决夹持掉落。
保留 3.3.4，不升级引擎、不修改安装包、场景 XML、摩擦系数、指垫几何、
执行器力限制、控制频率或 gripper 解码。

## 启用

在现有评测命令上增加：

```bash
--contact-solver-profile noslip_v1 \
--output-dir benchmarks/song_real_libero/outputs/eval_20hz_chunk24_thresh25_noslip_v1
```

`default` 为默认值，保持原求解参数。`noslip_v1` 固定将 `noslip_iterations`
设为 3，其余参数（包括 cone、impratio、timestep、solver、iterations、tolerance）不变。
要求实际引擎版本为 3.3.4，其他版本报错，避免无意混用新引擎。

## 生命周期与影响

- 原 reset、初始状态恢复、10 步 dummy action 和已有 warmup 全部使用原求解设置。
- 只在初始化结束、策略执行开始前切换；不额外推进仿真、不改状态、不重新采集首帧。
- 策略运行期间影响所有接触，而不只是夹爪；后续场景物理演化可能与原版不同。
- 退出/异常及复用环境时恢复原参数，防止污染下一 episode 的初始化。
- 结果的 `contact_solver_settings` 保存 before/after 和 changed_fields；配置及汇总保存 profile。
- `evaluation_protocol` 标记 `contact_solver_modified_rollout`、`benchmark_comparable=false`。
  原有其他诊断原因保存在 base_protocol 及原协议字段里。

NoSlip 用于抑制已存在接触上的滑移，不会修复漏检接触，也不能弥补错误抓取姿态或
超出摩擦能力的负载。求解设置改变也可能产生不稳定，故不能保证成功率提升。
不得把修正版结果与原环境的 baseline 直接宣称为同协议结果；应在同一设置下重测。

用户要求不再复现历史失败轨迹；此改动只做接口、初始化、渲染和短步进检查，
不会自动启动 400 条评测，也不以失败轨迹的成功率选择参数。

## 本次验证（2026-09-11）

- 接触求解开关、协议标记、已有输出/夹爪几何测试合计 24 项通过。
- MuJoCo 3.3.4 + EGL，Spatial task005 / initial state 0：原初始化完成后，
  启用前后 qpos、geom_pos、geom_friction、actuator_forcerange 和 256×256 RGB 完全一致。
- 该场景原设置为 elliptic cone、impratio=20、noslip_iterations=0；启用后唯一变化是
  noslip_iterations=3。10 步零动作下状态/深度有效，无新增 MuJoCo warning，退出恢复为 0。
- 未验证实际抓持改善或完整 400 条成功率。初始化相同不表示后续动力学相同。
