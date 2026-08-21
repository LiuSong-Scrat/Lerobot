#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
LAUNCH_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/train_molmo2er_pointonly_3b_8gpu.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE_DIR="${MOLMO_QUEUE_DIR:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_launch_${STAMP}}"
FORMAL_OUTPUT_DIR="${MOLMO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_seed1000_8x5090_${STAMP}}"
SMOKE_OUTPUT_DIR="/tmp/molmo2er_pointonly_3b_8gpu_smoke_${STAMP}"
LOCK_FILE="${MOLMO_QUEUE_LOCK_FILE:-/tmp/lerobot_molmo2er_pointonly_3b_8gpu.lock}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Molmo2-ER 3B watcher/training run already holds ${LOCK_FILE}." >&2
  exit 73
fi

mkdir -p "${QUEUE_DIR}"
printf '%s\n' "$$" > "${QUEUE_DIR}/watcher.pid"
printf '%s\n' "${FORMAL_OUTPUT_DIR}" > "${QUEUE_DIR}/formal_output_dir.txt"

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF && $1 ~ /^[0-9]+$/ { count += 1 } END { print count + 0 }'
}

wait_for_all_gpus() {
  while true; do
    local count
    count="$(gpu_process_count)"
    if [[ "${count}" == "0" ]]; then
      echo "[wait] $(date -u +%FT%TZ) all GPUs appear idle; confirming for 10 seconds"
      sleep 10
      if [[ "$(gpu_process_count)" == "0" ]]; then
        return 0
      fi
    else
      echo "[wait] $(date -u +%FT%TZ) ${count} foreign GPU process(es) still active"
    fi
    sleep 20
  done
}

echo "[queue] control_dir=${QUEUE_DIR}"
echo "[queue] formal_output_dir=${FORMAL_OUTPUT_DIR}"
echo "[queue] lock_file=${LOCK_FILE}"
wait_for_all_gpus

echo "[smoke] $(date -u +%FT%TZ) starting exact 8-GPU/global-192 two-update smoke"
MOLMO_REQUIRE_IDLE_GPUS=1 \
MOLMO_NUM_PROCESSES=8 \
MOLMO_BATCH_SIZE=24 \
MOLMO_ACCUMULATION_STEPS=1 \
MOLMO_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOLMO_OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
MOLMO_JOB_NAME=molmo2er_pointonly_3b_8gpu_smoke \
MOLMO_STEPS=2 \
MOLMO_NUM_WORKERS=0 \
MOLMO_SAVE_CHECKPOINT=false \
MOLMO_SAVE_FREQ=2000 \
MOLMO_EVAL_FREQ=0 \
MOLMO_WANDB_ENABLE=false \
MOLMO_MAIN_PROCESS_PORT=29640 \
bash "${LAUNCH_SCRIPT}" > "${QUEUE_DIR}/smoke_8gpu.log" 2>&1
echo "[smoke] $(date -u +%FT%TZ) passed"

# A shared user may start another job in the short gap after the smoke exits.
# Re-enter the same non-overlap gate instead of racing or killing it.
wait_for_all_gpus

echo "[formal] $(date -u +%FT%TZ) starting 80k-update run"
printf '%s\n' "$$" > "${QUEUE_DIR}/formal_launcher.pid"
exec env \
  MOLMO_REQUIRE_IDLE_GPUS=1 \
  MOLMO_NUM_PROCESSES=8 \
  MOLMO_BATCH_SIZE=24 \
  MOLMO_ACCUMULATION_STEPS=1 \
  MOLMO_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MOLMO_OUTPUT_DIR="${FORMAL_OUTPUT_DIR}" \
  MOLMO_JOB_NAME=molmo2er_pointonly_3b_seed1000_8x5090 \
  MOLMO_STEPS=80000 \
  MOLMO_NUM_WORKERS=14 \
  MOLMO_SAVE_CHECKPOINT=true \
  MOLMO_SAVE_FREQ=2000 \
  MOLMO_EVAL_FREQ=2000 \
  MOLMO_WANDB_ENABLE=true \
  MOLMO_MAIN_PROCESS_PORT=29641 \
  WANDB_MODE=offline \
  bash "${LAUNCH_SCRIPT}" >> "${QUEUE_DIR}/formal_train.log" 2>&1
