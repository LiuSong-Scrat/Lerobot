#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v52_consensus_multiscale}
dataset=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/libero_4suite_lerobot_dataset
cache=$root/joint_multiview_worldflow/libero10_500ep/pointseg_cache_consensus_multiscale_novelty_union_1cm4cm
output=$root/joint_multiview_worldflow/libero10_500ep/artifacts/v52_cache_online_exact_index_audit_36shards.json
workflow_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
auditor=$workflow_root/scripts/audit_v52_cache_exact_indices.py
python=/home/liusong/anaconda3/envs/reap/bin/python3.10
expected_head=9447a43a0e2ed3130a578618ac92225f71eb8a31

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain)"
test -s "$cache/manifest.json"
test ! -e "$output"
env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python" "$auditor" \
  --dataset-root "$dataset" \
  --cache-dir "$cache" \
  --output "$output" \
  --device cuda:0
