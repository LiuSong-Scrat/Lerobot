# RLBench WorldFlow Robot-Base Sidecar 生成说明

## 目标

在不重新采集 RLBench 数据的情况下，为现有 LeRobot 数据集生成两类固定机器人基座坐标系下的 EEF 位姿 sidecar：

```text
world_base_ee_poses/
world_base_action_target_ee_poses/
```

每条 episode 在两个目录中各对应一个 `(T, 9)` 的 NumPy 文件：

```text
world_base_ee_poses/episode_XXXXXX.npy
world_base_action_target_ee_poses/episode_XXXXXX.npy
```

其中 `T` 必须与该 episode 的 LeRobot 帧数完全一致。

## 坐标变换约定

本文统一使用以下记号：

```text
^A T_B：B 坐标系在 A 坐标系中的位姿；也就是将 B 系坐标变换到 A 系。
```

对应变量命名：

```text
T_base_world       = ^B T_W
T_world_ee         = ^W T_E
T_world_eef0       = ^W T_E0
T_eef0_target      = ^E0 T_target
```

如果已有矩阵是 `T_world_base = ^W T_B`，必须先求逆：

```python
T_base_world = np.linalg.inv(T_world_base)
```

不能将 `T_world_base` 直接左乘世界坐标位姿。

## 1. `world_base_ee_poses`

### 含义

当前 observation 时刻实际达到的 EEF 位姿，表达在固定机器人基座坐标系中。

### 数据来源

```text
world_ee_poses/episode_XXXXXX.npy
```

现有 `world_ee_poses[t]` 是 RLBench/CoppeliaSim 世界坐标系下的实际 EEF 位姿：

```text
T_world_ee[t] = ^W T_E(t)
```

### 计算方法

```text
T_base_ee[t] = T_base_world @ T_world_ee[t]
```

计算结果转换回 pose9 后保存：

```text
world_base_ee_poses/episode_XXXXXX.npy
```

## 2. `world_base_action_target_ee_poses`

### 含义

专家动作命令的目标 EEF 位姿，表达在固定机器人基座坐标系中。

它是 commanded target，不是仿真器执行后实际到达的 achieved pose。

### 数据来源

需要以下三项：

1. `action[t, :9]`：EEF0 坐标系下的专家目标 EEF pose9。
2. `world_ee_poses[0]`：该 episode 初始 EEF0 在世界坐标系中的位姿。
3. `T_base_world`：世界坐标系到机器人基座坐标系的固定变换。

对应矩阵为：

```text
T_eef0_target[t] = pose9_to_homo(action[t, :9])
T_world_eef0     = pose9_to_homo(world_ee_poses[0])
```

### 计算方法

先从 EEF0 变换到世界系，再从世界系变换到基座系：

```text
T_world_target[t] = T_world_eef0 @ T_eef0_target[t]

T_base_target[t] =
    T_base_world
    @ T_world_eef0
    @ T_eef0_target[t]
```

计算结果转换回 pose9 后保存：

```text
world_base_action_target_ee_poses/episode_XXXXXX.npy
```

## Episode 对齐语义

当前 RLBench 数据集使用 transition alignment：

```text
observation[t] + action[t] -> observation[t + 1]
```

因此同一个索引 `t` 的含义是：

```text
world_base_ee_poses[t]
    当前 observation[t] 的实际 EEF 位姿

world_base_action_target_ee_poses[t]
    action[t] 要求仿真器执行的命令目标 EEF 位姿
```

不能使用 `world_ee_poses[t + 1]` 替代 action target。前者是执行后的实际到达结果，可能因为 IK、控制误差或仿真碰撞而与命令目标不同。

## 目录结构

生成后的数据集应包含：

```text
DATASET_ROOT/
├── world_ee_poses/
│   ├── meta.json
│   ├── episode_000000.npy
│   └── ...
├── world_base_ee_poses/
│   ├── meta.json
│   ├── episode_000000.npy
│   └── ...
└── world_base_action_target_ee_poses/
    ├── meta.json
    ├── episode_000000.npy
    └── ...
```

## 元数据

`world_base_ee_poses/meta.json`：

```json
{
  "key": "worldflow.current_ee_pose",
  "shape": [9],
  "dtype": "float32",
  "layout": "episode_npy",
  "coordinate_frame": "robot_base",
  "target_semantics": "achieved_eef_pose",
  "path_format": "world_base_ee_poses/episode_{episode_index:06d}.npy"
}
```

`world_base_action_target_ee_poses/meta.json`：

```json
{
  "key": "worldflow.eef_trajectory",
  "shape": [9],
  "dtype": "float32",
  "layout": "episode_npy",
  "coordinate_frame": "robot_base",
  "target_semantics": "commanded_eef_pose",
  "alignment": "transition",
  "path_format": "world_base_action_target_ee_poses/episode_{episode_index:06d}.npy"
}
```

## 必做校验

### 1. 数量与形状

- 两个新目录的 episode 文件数量必须与数据集 episode 数量一致。
- 每个文件必须是 `(T, 9)`。
- 两个新文件、原 `world_ee_poses` 和对应 LeRobot episode 的长度必须一致。
- 所有数值必须有限，不能包含 `NaN` 或 `Inf`。

### 2. 旋转合法性

每个 pose9 恢复出的旋转矩阵都应满足：

```text
R.T @ R ~= I
det(R) ~= 1
```

### 3. Achieved pose 往返校验

```text
T_world_ee_recovered = inverse(T_base_world) @ T_base_ee
```

结果应与原始 `T_world_ee` 一致。

### 4. Action target 往返校验

```text
T_eef0_target_recovered =
    inverse(T_world_eef0)
    @ inverse(T_base_world)
    @ T_base_target
```

结果应与原始 `T_eef0_target` 一致。

### 5. 语义校验

- `world_base_ee_poses` 必须来自实际 observation 的 `world_ee_poses`。
- `world_base_action_target_ee_poses` 必须来自 action 标签。
- 不允许以后一帧 achieved EEF pose 冒充 commanded action target。
- 不允许只修改目录名或元数据，而不真正执行坐标变换。

## WorldFlow 训练配置

sidecar 生成并通过校验后，可使用正式 robot-base WorldFlow 配置：

```bash
--policy.worldflow_enable=true \
--policy.worldflow_target_type=world_eef_trajectory \
--policy.worldflow_reference_frame=robot_base \
--policy.worldflow_require_action_target_sidecar=true
```

该配置会读取：

```text
worldflow.current_ee_pose <- world_base_ee_poses
worldflow.eef_trajectory  <- world_base_action_target_ee_poses
```

