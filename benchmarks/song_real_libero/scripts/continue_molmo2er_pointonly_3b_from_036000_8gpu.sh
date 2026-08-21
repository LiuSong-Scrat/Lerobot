#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
LAUNCH_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/train_molmo2er_pointonly_3b_8gpu.sh"
SOURCE_CHECKPOINT="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_seed1000_8x5090_58c5f91/checkpoints/036000/pretrained_model"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTROL_DIR="${MOLMO_CONTINUATION_CONTROL_DIR:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_continue30k_launch_${STAMP}}"
FORMAL_OUTPUT_DIR="${MOLMO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_from036000_lr1e4_to3e5_30k_${STAMP}}"
SMOKE_OUTPUT_DIR="/tmp/molmo2er_pointonly_3b_from036000_smoke_${STAMP}"
LOCK_FILE="${MOLMO_QUEUE_LOCK_FILE:-/tmp/lerobot_molmo2er_pointonly_3b_8gpu.lock}"

if [[ ! -f "${SOURCE_CHECKPOINT}/model.safetensors" ]]; then
  echo "Missing source checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Molmo2-ER training run already holds ${LOCK_FILE}." >&2
  exit 73
fi

mkdir -p "${CONTROL_DIR}"
printf '%s\n' "$$" > "${CONTROL_DIR}/launcher.pid"
printf '%s\n' "${SOURCE_CHECKPOINT}" > "${CONTROL_DIR}/source_checkpoint.txt"
printf '%s\n' "${FORMAL_OUTPUT_DIR}" > "${CONTROL_DIR}/formal_output_dir.txt"

COMMON_ENV=(
  MOLMO_REQUIRE_IDLE_GPUS=1
  MOLMO_NUM_PROCESSES=8
  MOLMO_BATCH_SIZE=24
  MOLMO_ACCUMULATION_STEPS=1
  MOLMO_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MOLMO_POLICY_PATH="${SOURCE_CHECKPOINT}"
  MOLMO_SCHEDULER_DECAY_LR=0.00003
  MOLMO_NUM_WORKERS=14
  MOLMO_SAVE_FREQ=2000
  MOLMO_EVAL_FREQ=2000
  MOLMO_MAIN_PROCESS_PORT=29642
  WANDB_MODE=offline
)

echo "[continuation] source_checkpoint=${SOURCE_CHECKPOINT}"
echo "[continuation] optimizer and scheduler state will restart; this is not --resume"
echo "[continuation] point-only is retained; worldflow_enable=false makes bootstrap_from_ego inapplicable"
echo "[continuation] lr=1e-4 warmup=100 cosine_decay_steps=30000 decay_lr=3e-5"

echo "[smoke] $(date -u +%FT%TZ) strict checkpoint load and two DDP updates"
env "${COMMON_ENV[@]}" \
  MOLMO_OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
  MOLMO_JOB_NAME=molmo2er_pointonly_3b_from036000_smoke \
  MOLMO_STEPS=2 \
  MOLMO_NUM_WORKERS=0 \
  MOLMO_SAVE_CHECKPOINT=false \
  MOLMO_EVAL_FREQ=0 \
  MOLMO_WANDB_ENABLE=false \
  bash "${LAUNCH_SCRIPT}" > "${CONTROL_DIR}/smoke_8gpu.log" 2>&1
echo "[smoke] $(date -u +%FT%TZ) passed"

echo "[formal] $(date -u +%FT%TZ) starting 30k continuation"
exec env "${COMMON_ENV[@]}" \
  MOLMO_OUTPUT_DIR="${FORMAL_OUTPUT_DIR}" \
  MOLMO_JOB_NAME=molmo2er_pointonly_3b_from036000_lr1e4_to3e5_30k \
  MOLMO_STEPS=30000 \
  MOLMO_SAVE_CHECKPOINT=true \
  MOLMO_WANDB_ENABLE=true \
  bash "${LAUNCH_SCRIPT}" >> "${CONTROL_DIR}/formal_train.log" 2>&1
