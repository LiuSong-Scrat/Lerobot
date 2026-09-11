# LIBERO 评测输出结构

参考文件：`/home/liusong/RE_rlbench_official_eval.py`。
实现入口：`scripts/libero_setting/libero_pointcloud_eval.py`；输出与绘图分别放在
`libero_eval_artifacts.py`、`libero_eval_visualization.py`。

本次仅调整输出，不调整模型输入、相机/虚拟夹爪策略、动作解码、控制、随机种子、成功判定或推理调用次数。
未恢复此前已撤销的成功率优化实验。

## 文件布局

保留 LIBERO 的 suite/task 外层，任务内采用参考脚本的分类布局。路径均相对于该任务目录：

```text
<output-dir>/
  config.json
  summary.json / overall.json / 现有进度与失败记录
  <suite>/task_000/
    config.json
    summary.json                     # task / episodes / successes / success_rate / results
    results/episode_000.json
    errors/episode_000.json           # 仅异常时
    videos/episode_000.mp4            # agentview 主视角
    videos/<其他 image_key>/episode_000.mp4
    actions/episode_000_actions.npy
    model_chunks/episode_000_model_chunks.npy
    action_chunks/episode_000/frame_000000_model_call_0001.npy
    umi_vis/episode_000/frame_000000_model_call_0001.ply  # 每次推理，需开启
    executed_action_alignment/
      episode_000_executed_model_actions_relative10.npy
      episode_000_executed_controller_actions_world7.npy
      episode_000_executed_model10_controller7.npy
      episode_000_execution_index.npy
      all_executed_model_actions_relative10.npy
      all_executed_controller_actions_world7.npy
      all_executed_model10_controller7.npy
      all_execution_index.npy
      manifest.json
      README.md
    action_visualizations/episode_000/
      frame_000000_model_call_0001_prob.ply
      frame_000000_model_call_0001.png
    frame_pointclouds/episode_000/frame_000000_model_input_raw.ply
    diagnostics/episode_000/          # 保留原 LIBERO actions.npz、goal_debug.json 等
```

已有旧输出不迁移、不清理。请使用新的 `--output-dir`，不要混用新旧布局。
已启动的进程不会可靠地热更新，应在下一次启动时使用新布局。

## 启用方式

在原评测命令后添加以下参数即可；训练/控制参数无需更改：

```bash
--save-video \
--save-action-records \
--save-action-chunks \
--save-action-visualizations \
--action-vis-point-mode prob \
--action-vis-every-n-frames 32 \
--action-vis-max-points 50000 \
--action-vis-image-width 768 \
--video-fps 20
```

若还需要观察每隔两步的原始模型点云，另加：

```bash
--save-frame-pointclouds --frame-pointcloud-every-n-frames 2
```

动作记录、完整 chunks 默认保存。PLY/PNG 轨迹图、逐帧点云默认关闭，避免增加常规评测的 IO。
`--save-video` 继续遵循原有 CLI/config 设置，不强制开启。
`--failure-artifacts-only` 只删除成功 episode 本次生成的轨迹 PLY/PNG；视频、动作和原始逐帧点云仍保留，语义与参考脚本一致。
所有新布尔选项支持 `--no-...`。

每个 episode 的每次推理都保存现有 `vis_umi_data` 的 PLY 结果：

```bash
--save_umi_vis
```

也支持 `--save-umi-vis`；默认关闭，关闭参数为 `--no-save-umi-vis`。
保存至 `umi_vis/episode_###/frame_######_model_call_####.ply`，episode JSON 中记录
`umi_vis` 文件列表与 `umi_vis_count`。直接使用本次预测 chunk 和本次模型输入点云，
显示 RGB 场景、当前虚拟夹爪、预测路径及姿态轴，与现有 `vis_umi_data(..., save_path=...)` 完全相同。
它是预测可视化，不是 GT，也不是概率着色 PLY；不打开窗口、不额外 forward。
不受 `action-vis-every-n-frames`、`save-action-visualizations`、`failure-artifacts-only` 影响，
成功/失败 episode 均保留所有推理文件（保存失败会记入 `artifact_errors`）。
推理缓存命中同样保存返回的 chunk。点数预算沿用 `--trajectory-vis-max-points`（默认 50000）。

视频始终保留渲染采集图像的 H×W，当前 256×256 采集对应 256×256 MP4。
旧 `--video-width` / `--video-height` 选项为兼容旧命令仍接受，但会提示已忽略，不能再触发 resize。

## 可视化语义

- 视频：任务、episode、控制帧号、相对物理帧号、模型调用号、执行 chunk 行号，以及最终 SUCCESS 标注。
  主视频仍使用原有 agentview 保存方向，不进行尺寸缩放；标注不作用于模型输入。
  编码逐帧进行，不额外缓存整段标注后的视频。
- PLY：当前虚拟末端坐标系下的场景/当前夹爪，以及完整预测轨迹和 XYZ 轴。
  轨迹从蓝/青→绿→红；`point_kind`、`action_index`、`action_phase` 和概率字段保存在顶点属性中。
  `full` 模式保留场景 RGB；`prob` 模式按当前模型的前景概率着色。
- PNG：将预测轨迹投影到 agentview RGB；当前计划执行部分用实线，未来部分用灰色虚线，白线标记末端 X 轴。
  只叠加当前虚拟夹爪采样点/线框，不生成未来每一帧的夹爪几何。
  投影使用实际相机外参，并将现有图像翻转纳入内参，不改动外参或训练图像约定。
- 图中的执行窗口是基于 `action_index` / `exec_action_steps` 的名义窗口。
  实际提前结束、hold、抓取/自适应扩展以 `execution_index` 为准。
- 概率来自同一次推理，按线程/进程的 batch 行独立传递；不会为了绘图额外 forward。
  推理缓存命中、未启用 pointseg 或点顺序不匹配时，概率标为 NaN，不伪造、不使用上一批结果。
- 原始逐帧 PLY 是以该控制步数为采样 seed、经过现有点云适配器得到的 XYZRGB，不附加不存在的逐帧前景概率。
  频繁保存会增加 CPU、磁盘和容量开销，并行运行时应按需启用。

## 与 RLBench 必须保留的区别

LIBERO 发出的绝对 OSC 命令是 **world7**：`xyz + rotvec + gripper command`。
参考 RLBench 是 world8（四元数）；不能仅为文件外观一致而补维或改控制。
模型 relative10 与 world7 每个实际执行步配对保存，因此 combined 为 17 维。
动作保存为 float32；hold 会重复对应模型行。手动回退/source-oracle 没有相应模型预测行，记录 NaN，不能当 GT。

`execution_index` 每行：
`[episode, model_call(1-based), chunk_row(0-based), phase, hold_attempt, environment_step(0-based)]`。
phase：0 普通预测执行，1 waypoint hold，2 手动回退，3 source oracle。
回退/oracle 的 chunk_row 为 -1、hold_attempt 为 0；初始化、reset、官方 10 步 warmup 不计入这些数组。
任务级 `all_*` 由汇总进程在合并 episode 分片后按 episode 排序生成，不跨任务混合。

RLBench 特有的传感器/控制诊断叠加不照搬到 LIBERO；原有 LIBERO 诊断文件保留在 diagnostics 中。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 SONG_LIBERO_ENV_WORKER=1 PYTHONPATH=src:. \
/home/liusong/anaconda3/envs/reap/bin/python -m pytest \
  tests/test_libero_eval_artifacts.py tests/test_rh20t_eval_gripper.py -q --noconftest
```

测试覆盖参数、保存目录、图像/输入不可变性、实际反投影的逆投影、执行数组和 hold 索引、
线程/进程前景快照隔离及缓存命中不复用旧概率。
本地真实环境短程输出样例位于 `outputs/output_layout_smoke_20260911`；其控制来自合成测试桩，
概率为合成 0.5，不能当作 checkpoint 的轨迹、分割或成功率结果。

首版输出适配的本地验证：13 项单元测试通过；CPU/OSMesa 真实 LIBERO 环境对照各执行 6 步、
调用测试桩 2 次，修改前后的控制动作、预测 chunks、末端状态和模型输入哈希完全相同。
新输出包含 7 帧 512×512 视频、2 组轨迹 PNG/PLY、4 个逐帧点云，以及逐步对齐数组；
详情见样例目录中的 `SMOKE_VALIDATION.json`。没有进行完整 checkpoint 的 400 条复评。

后续原尺寸视频/逐次 UMI 保存版本：15 项单元测试通过，增加旧视频尺寸选项不再缩放、
逐次 UMI PLY 与原函数字节一致、关闭开关不写文件、成功 episode 仍保留全部 UMI 的检查。
新版短程样例目录为 `outputs/output_layout_native_umi_smoke_20260911`。

## agentview 图像方向的来源

这里的“转换约定”指本仓库 `libero_hdf5_to_dataset.py` 的 `dataset_image_from_raw_obs()`：
对 `agentview_image` 执行 `image[:, ::-1]` 后再写入 dataset，即反转图像列顺序（左右镜像）。
不是上下翻转、不是旋转 180°，也不是新增的位姿变换。
训练直接读取已经保存的图像，不再重复镜像；eval 的同名函数对实时 RGB 做同样的一次镜像。
这是本项目现有转换器的存储规则，并非泛指所有 LIBERO 数据的统一标准。
本次视频尺寸/PLY 输出修改保留该输入处理，不修改训练集或模型输入方向。
