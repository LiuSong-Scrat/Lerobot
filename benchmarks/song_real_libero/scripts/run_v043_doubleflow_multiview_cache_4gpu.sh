#!/usr/bin/env bash
set -euo pipefail

# Build the input-only dual-view FPS cache. Two N-point camera clouds are
# combined outside SmolVLA and reduced back to exactly N points (10,000 for
# this dataset) before the policy receives them.

REPO_ROOT=/home/liusong/ProgramFiles/Huggingface/lerobot
PYTHON_BIN=/home/liusong/anaconda3/envs/reap/bin/python
DATASET_ROOT=${SONG_V043_DATASET_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_dataset}
CACHE_ROOT=${SONG_V043_MULTIVIEW_CACHE_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_union_fps_cache}

cd "$REPO_ROOT"

if [[ -f "$CACHE_ROOT/manifest.json" ]]; then
  test "$(jq -r '.num_samples' "$CACHE_ROOT/manifest.json")" = "336575"
  test "$(jq -r '.current_points' "$CACHE_ROOT/manifest.json")" = "10000"
  test "$(jq -r '.future_points' "$CACHE_ROOT/manifest.json")" = "10000"
  test "$(jq -r '.camera_view_fusion' "$CACHE_ROOT/manifest.json")" = "fps"
  test "$(jq -r '.camera_views | join(",")' "$CACHE_ROOT/manifest.json")" = "agentview,robot0_eye_in_hand"
  CACHE_WORLD_SIZE=$(jq -r '.distributed.world_size' "$CACHE_ROOT/manifest.json")
  test "$(find "$CACHE_ROOT/_dist_sync" -maxdepth 1 -name 'rank_*.done' | wc -l)" = "$CACHE_WORLD_SIZE"
  echo "Existing cache manifest and all rank completion markers pass the dual-view 10k contract: $CACHE_ROOT"
  exit 0
fi

if [[ -e "$CACHE_ROOT" ]]; then
  echo "Refusing to overwrite an incomplete cache directory: $CACHE_ROOT" >&2
  exit 1
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$REPO_ROOT/src"
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500

exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id="$DATASET_ROOT" \
  --camera-views=agentview,robot0_eye_in_hand \
  --camera-view-fusion=fps \
  --output-dir="$CACHE_ROOT" \
  --current-points=10000 \
  --future-points=10000 \
  --batch-size=16 \
  --num-workers=4 \
  --shard-size=4096 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4 \
  --rank-wait-timeout-sec=0
