#!/usr/bin/env bash
set -euo pipefail

repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v52_consensus_multiscale}
root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
dataset=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/libero_4suite_lerobot_dataset
run_root=$root/joint_multiview_worldflow/libero10_500ep
smoke=$run_root/cache_smoke/v52_consensus_multiscale_novelty_union_1cm4cm_4gpu
cache=$run_root/pointseg_cache_consensus_multiscale_novelty_union_1cm4cm
python=/home/liusong/anaconda3/envs/reap/bin/python3.10
torchrun=/home/liusong/anaconda3/envs/reap/bin/torchrun
expected_head=9447a43a0e2ed3130a578618ac92225f71eb8a31

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain)"
test -s "$dataset/meta/info.json"
test ! -e "$smoke"
test ! -e "$cache"
mkdir -p "$(dirname "$smoke")"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
export OMP_NUM_THREADS=1

cd "$repo"
"$python" "$torchrun" --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id="$dataset" \
  --camera-views=agentview,robot0_eye_in_hand \
  --camera-view-fusion=consensus_multiscale_novelty_union \
  --camera-view-voxel-size=0.01 \
  --camera-view-coarse-novelty-scale=4.0 \
  --output-dir="$smoke" \
  --batch-size=2 --num-workers=0 --shard-size=4 \
  --storage-dtype=float16 --nn-chunk-size=1024 --vis-count=2 --smoke-test

"$python" - "$smoke/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["version"] == 12 and manifest["num_samples"] == 4
assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"]
assert manifest["camera_view_fusion"] == "consensus_multiscale_novelty_union"
assert manifest["camera_view_voxel_size"] == 0.01
assert manifest["camera_view_coarse_novelty_scale"] == 4.0
assert manifest["current_points"] == manifest["future_points"] == 10_000
assert manifest["gripper_points"] == 500
assert manifest["point_count_policy"] == (
    "fine_primary_voxel_consensus_medoid_plus_coarse_persistent_"
    "secondary_novel_voxels_preserve_primary_gripper"
)
PY

"$python" "$torchrun" --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id="$dataset" \
  --camera-views=agentview,robot0_eye_in_hand \
  --camera-view-fusion=consensus_multiscale_novelty_union \
  --camera-view-voxel-size=0.01 \
  --camera-view-coarse-novelty-scale=4.0 \
  --output-dir="$cache" \
  --batch-size=24 --num-workers=4 --shard-size=4096 \
  --storage-dtype=float16 --nn-chunk-size=1024 --vis-count=4

"$python" - "$cache/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["version"] == 12
assert manifest["num_samples"] == 137_590
assert sum(shard["length"] for shard in manifest["shards"]) == 137_590
assert len(manifest["shards"]) == 36
assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"]
assert manifest["camera_view_fusion"] == "consensus_multiscale_novelty_union"
assert manifest["camera_view_voxel_size"] == 0.01
assert manifest["camera_view_coarse_novelty_scale"] == 4.0
assert manifest["current_points"] == manifest["future_points"] == 10_000
assert manifest["gripper_points"] == 500
assert manifest["distributed"]["world_size"] == 4
assert manifest["point_count_policy"] == (
    "fine_primary_voxel_consensus_medoid_plus_coarse_persistent_"
    "secondary_novel_voxels_preserve_primary_gripper"
)
print("V52 full cache PASS", manifest["num_samples"], len(manifest["shards"]))
PY
