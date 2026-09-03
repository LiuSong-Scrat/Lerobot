#!/usr/bin/env bash

# 从指定的 LeRobot RLBench 数据集中读取一个 episode 的 action，然后在新建的
# RLBench 环境中逐帧执行。程序会保存 replay.mp4 和 result.json。
#
# task：当前数据集包含 close_laptop_lid、water_plants 和 sweep_to_dustpan。
# episode：任务内部的 episode 编号，范围是 0 到 149。也可以使用逗号分隔的
# 多个编号，例如 --episode 1,2,3,4,5,134；每条轨迹会分别重放和保存视频。
# mode=eef0_planning：把训练数据 parquet 中保存的 10D EEF0 action 当作
# waypoint，使用 RLBench 官方 EndEffectorPoseViaPlanning（IK + RRTConnect）。
# 需要 Jacobian 对照时，命令末尾显式追加 --mode eef0。
# action-source=parquet：确保重放的正是训练时读取的 action，而不是中间文件。
# replay 会先 task.reset() 创建任务对象，再恢复该 episode 的
# initial_task_states/episode_XXXXXX.npz，然后才读取恢复后的 EEF0 并执行动作。
# 旧数据集没有这个初始状态时程序会停止并要求重新采集，不会再偷偷使用随机布局。
# 只有诊断旧数据时才可以手动追加 --allow-fresh-reset；这种结果不是精确 replay。
# seed 只影响 task.reset() 的临时布局；成功恢复状态后，它不决定最终物体布局。
#
# 运行扫地 episode 0：
# bash benchmarks/RLBench/scripts/s1.1_RE_rlbench_dataset_action_replay.sh \
#     --task sweep_to_dustpan --episode 0

REPLAY_TASK="${TASK:-sweep_to_dustpan}"
REPLAY_MODE="${MODE:-eef0_planning}"
REPLAY_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_REPLAY_OUTPUT="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/eval/replays/${REPLAY_STAMP}__${REPLAY_TASK}__${REPLAY_MODE}_dataset_action_replay"

DISPLAY="${DISPLAY:-:99}" \
QT_X11_NO_MITSHM=1 \
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/CoppeliaSim}" \
LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}" \
"${PYTHON:-/home/liusong/miniconda3/envs/rlbench/bin/python}" \
    /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/scripts/RE_rlbench_dataset_action_replay.py \
    --dataset-root "${DATASET_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/rlbench_all_tasks_lerobot_20260804_211804}" \
    --artifact-root "${ARTIFACT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/rlbench_all_tasks_lerobot_20260804_211804_artifacts}" \
    --task "${REPLAY_TASK}" \
    --episode "${EPISODES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}" \
    --mode "${REPLAY_MODE}" \
    --action-source "${ACTION_SOURCE:-parquet}" \
    --variation "${VARIATION:-0}" \
    --seed "${SEED:-123}" \
    --image-size "${IMAGE_SIZE:-512}" \
    --video-fps "${VIDEO_FPS:-20}" \
    --gripper-open-width "${GRIPPER_OPEN_WIDTH:-0.04}" \
    --gripper-mode "${GRIPPER_MODE:-delta_width_initial_sync}" \
    --gripper-delta-threshold "${GRIPPER_DELTA_THRESHOLD:-0.002}" \
    --gripper-delta-alignment "${GRIPPER_DELTA_ALIGNMENT:-current_minus_previous}" \
    --output-root "${OUTPUT_ROOT:-${DEFAULT_REPLAY_OUTPUT}}" \
    "$@"
