#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v51_conservative_symmetric_point_adaptation}
runner=$root/scripts/run_v51_multiscale_novelty_1cm4cm_cache_4gpu.sh
session=wep_v043_v51_multiscale_novelty_1cm4cm_cache_4gpu
log=$root/joint_multiview_worldflow/libero10_500ep/logs/cache_v51_multiscale_novelty_union_1cm4cm_4gpu.log

test -x "$runner"
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 3
fi
mkdir -p "$(dirname "$log")"
tmux new-session -d -s "$session" \
  "set -o pipefail; LEROBOT_REPO='$repo' EXPERIMENT_ROOT='$root' bash '$runner' 2>&1 | tee -a '$log'"
echo "launched tmux session: $session"
echo "log: $log"
