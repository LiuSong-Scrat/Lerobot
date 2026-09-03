#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/liusong/miniconda3/envs/rlbench/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/benchmarks/RLBench/datasets/rlbench_all_tasks_100traj_lerobot_raw_20260807_123258_artifacts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmarks/RLBench/datasets/rlbench_close_laptop_lid_100traj_reap_v4_50000}"

export DISPLAY="${DISPLAY:-:99}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${REPO_ROOT}/benchmarks/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-${COPPELIASIM_ROOT}}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"

exec "${PYTHON}" "${SCRIPT_DIR}/RE_rlbench_rebuild_lerobot_from_artifacts.py" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --task "${TASK:-close_laptop_lid}" \
  --episodes "${EPISODES:-100}" \
  --num-points "${NUM_POINTS:-50000}" \
  --gripper-points "${GRIPPER_POINTS:-500}" \
  --image-size "${IMAGE_SIZE:-256}" \
  --fps "${FPS:-20}" \
  --variation "${VARIATION:-0}" \
  --seed "${SEED:-20260812}" \
  "${@}"
