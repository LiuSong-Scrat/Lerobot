#!/usr/bin/env bash
set -euo pipefail

# Dataset-state evaluation: the policy is evaluated online, but each episode
# starts from the exact RLBench reset state recorded for one dataset episode.
# The task-local episode indices are resolved through meta/episodes/*.parquet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/rlbench_all_tasks_lerobot_20260804_211804}"
TASK="${TASK:-close_fridge}"
DATASET_EPISODES="${DATASET_EPISODES:-0,1,2,3,4,5,6,7,8,9}"
POLICY_PATH="${POLICY_PATH:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_v041_rlbench_08051103/checkpoints/last/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/eval_dataset_state}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${OUTPUT_DIR}/serial_$(date +%Y%m%d_%H%M%S)}"

exec env \
    EVAL_GPU_IDS="${EVAL_GPU_IDS:-0}" \
    EVAL_DISPLAY_BASE="${EVAL_DISPLAY_BASE:-105}" \
    EVAL_LOG_DIR="${EVAL_LOG_DIR}" \
    bash "${SCRIPT_DIR}/s4_RE_rlbench_official_eval.sh" \
    --tasks "${TASK}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-episodes "${DATASET_EPISODES}" \
    --policy-path "${POLICY_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
