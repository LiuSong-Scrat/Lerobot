#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v52_consensus_multiscale}
workflow_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cache=$root/joint_multiview_worldflow/libero10_500ep/pointseg_cache_consensus_multiscale_novelty_union_1cm4cm
log=$root/joint_multiview_worldflow/libero10_500ep/logs/wait_then_audit_and_train_v52.log

while [[ ! -s "$cache/manifest.json" ]]; do
  printf '[%s] waiting for V52 cache manifest\n' "$(date --iso-8601=seconds)" >>"$log"
  sleep 30
done
while pgrep -f '[s]ong_cache_pointseg_samples.py.*pointseg_cache_consensus_multiscale_novelty_union_1cm4cm' >/dev/null; do
  printf '[%s] waiting for V52 cache workers to exit\n' "$(date --iso-8601=seconds)" >>"$log"
  sleep 10
done

EXPERIMENT_ROOT="$root" LEROBOT_REPO="$repo" \
  "$workflow_root/scripts/run_v52_cache_exact_index_audit.sh" 2>&1 | tee -a "$log"
EXPERIMENT_ROOT="$root" LEROBOT_REPO="$repo" \
  "$workflow_root/scripts/run_v52_from_v32step100_consensus1cm4cm_paired_symmetricpoint_4gpu.sh" \
  2>&1 | tee -a "$log"
