#!/usr/bin/env bash
set -euo pipefail

# Avoid racing unrelated user jobs for the four GPUs. Launch only after all
# compute processes have been absent continuously for one minute.

REPO_ROOT=/home/liusong/ProgramFiles/Huggingface/lerobot
IDLE_POLLS_REQUIRED=6
IDLE_POLLS=0

cd "$REPO_ROOT"
echo "Waiting for all four GPUs to remain compute-idle for 60 seconds..."

while (( IDLE_POLLS < IDLE_POLLS_REQUIRED )); do
  COMPUTE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  if [[ -z "$COMPUTE_PIDS" ]]; then
    IDLE_POLLS=$((IDLE_POLLS + 1))
    echo "$(date '+%F %T') idle poll $IDLE_POLLS/$IDLE_POLLS_REQUIRED"
  else
    if (( IDLE_POLLS != 0 )); then
      echo "$(date '+%F %T') GPU use resumed; resetting idle window."
    fi
    IDLE_POLLS=0
  fi
  sleep 10
done

echo "$(date '+%F %T') GPUs are idle; launching input-only dual-view DoubleFlow training."
exec benchmarks/song_real_libero/scripts/run_v043_doubleflow_multiview_train_4gpu.sh
