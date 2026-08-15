#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v51}
session=wep_v043_v51r1egl0shared_all_checkpoints_stratified_3arm
worker="$root/scripts/wait_then_screen_v51r1_all_checkpoints_stratified_3arm.sh"
log="$root/joint_multiview_worldflow/libero10_500ep/logs/eval_v51r1egl0shared_all_checkpoints_stratified_3arm.log"

test -x "$worker"
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 3
fi
mkdir -p "$(dirname "$log")"
tmux new-session -d -s "$session" \
  "set -o pipefail; LEROBOT_REPO='$repo' EXPERIMENT_ROOT='$root' bash '$worker' 2>&1 | tee -a '$log'"
echo "launched tmux session: $session"
echo "waits for the complete V51R1 paired epoch, then screens steps 260/520/780/1040/1300/1564 with three causal arms"
