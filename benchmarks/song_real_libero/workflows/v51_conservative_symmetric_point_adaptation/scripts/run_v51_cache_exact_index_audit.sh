#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v51_conservative_symmetric_point_adaptation}
dataset=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/libero_4suite_lerobot_dataset
cache=$root/joint_multiview_worldflow/libero10_500ep/pointseg_cache_multiscale_novelty_union_1cm4cm
output=$root/joint_multiview_worldflow/libero10_500ep/artifacts/v51_cache_online_exact_index_audit_36shards.json
auditor=$root/scripts/audit_v46_cache_exact_indices.py
python=/home/liusong/anaconda3/envs/reap/bin/python3.10

test "$(git -C "$repo" rev-parse HEAD)" = 93e2d8a3177a8addc40229da00962fcc2e7b7100
test -z "$(git -C "$repo" status --porcelain)"
test -s "$cache/manifest.json"
test ! -e "$output"
env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python" "$auditor" \
  --dataset-root "$dataset" \
  --cache-dir "$cache" \
  --output "$output" \
  --device cuda:0 \
  --voxel-size 0.01 \
  --coarse-novelty-scale 4.0 \
  --expected-version 12 \
  --expected-samples 137590 \
  --expected-shards 36
