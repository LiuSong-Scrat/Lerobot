#!/usr/bin/env bash
set -euo pipefail

# This host retains a stale zero-byte unversioned NVIDIA library symlink. Put
# the valid versioned driver directory first so PyTorch and NVML resolve the
# active 570 driver used by /dev/nvidia*.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

repo=${SONG_ABLATION_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot}
python=${SONG_ABLATION_PYTHON:-/home/liusong/anaconda3/envs/reap/bin/python}
dataset=${SONG_ABLATION_DATASET:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep}
cache=${SONG_ABLATION_CACHE:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache}
vlm=${SONG_ABLATION_VLM:-/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct}
vlm_weights=${SONG_ABLATION_VLM_WEIGHTS:-/opt/data/private/liusong/hf_models/smolvla_base}
root=${SONG_ABLATION_OUTPUT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/outputs/v043_cumulative_ablation_task6_task8_20260829}
steps=${SONG_ABLATION_STEPS:-30000}
save_freq=${SONG_ABLATION_SAVE_FREQ:-2000}
batch_size=${SONG_ABLATION_BATCH_SIZE:-8}
num_workers=${SONG_ABLATION_NUM_WORKERS:-4}
eval_episodes=${SONG_ABLATION_EVAL_EPISODES:-10}

variants=(
  smolvla_src
  smolvla_pointcloud
  smolvla_pointcloud_effseg
  smolvla_pointcloud_effseg_pointaction
)
train_gpus=(0 1 2 3)
eval_gpus=(4 5 6 7)

usage() {
  echo "usage: $0 {preflight|train|eval|all|summarize|status}"
}

require_inputs() {
  test -f "$dataset/meta/info.json"
  test -f "$vlm/config.json"
  test -f "$vlm_weights/model.safetensors"
  test -f "$cache/manifest.json"
  if (( steps < save_freq || steps % save_freq != 0 )); then
    echo "SONG_ABLATION_STEPS must be a positive multiple of SONG_ABLATION_SAVE_FREQ" >&2
    exit 2
  fi
}

preflight() {
  require_inputs
  cd "$repo"
  PYTHONPATH="$repo/src" "$python" - "$dataset" "$cache" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

dataset = Path(sys.argv[1])
cache = Path(sys.argv[2])
info = json.loads((dataset / "meta/info.json").read_text())
manifest = json.loads((cache / "manifest.json").read_text())
assert info["total_episodes"] == 100, info["total_episodes"]
assert info["total_tasks"] == 2, info["total_tasks"]
assert info["features"]["action"]["shape"] == [10]
assert manifest["num_samples"] == info["total_frames"]
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
expected = {
    "smolvla_src": (False, False, False),
    "smolvla_pointcloud": (True, False, False),
    "smolvla_pointcloud_effseg": (True, True, False),
    "smolvla_pointcloud_effseg_pointaction": (True, True, True),
}
for name, gates in expected.items():
    cfg = SmolVLAConfig(ablation_variant=name)
    actual = (cfg.pointcloud_enable, cfg.pointseg_enable, cfg.point_action_fusion_enable)
    assert actual == gates, (name, actual, gates)
    assert cfg.vla_adapter_enable and cfg.vla_adapter_freeze_vlm
    assert not cfg.worldflow_enable
    assert cfg.pointcloud_input_points == 10_000
print(f"ablation preflight PASS: frames={info['total_frames']} episodes={info['total_episodes']}")
PY
}

train_one() {
  local variant=$1 gpu=$2
  local output="$root/train/$variant"
  local log="$root/logs/train_${variant}.log"
  if [[ -s "$output/checkpoints/$(printf '%06d' "$steps")/pretrained_model/model.safetensors" ]]; then
    echo "[train] reuse completed $variant"
    return
  fi
  if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[train] incomplete output exists for $variant: $output" >&2
    return 1
  fi
  mkdir -p "$output" "$root/logs"
  local -a cache_args=()
  if [[ "$variant" == smolvla_pointcloud_effseg* ]]; then
    cache_args=(--pointseg_sample_cache_dir="$cache")
  fi
  cd "$repo"
  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo/src" SONG_POINTSEG_REQUIRE_POINTOPS=1 \
  "$python" benchmarks/song_real_libero/scripts/train_song_benchmark.py \
    --policy.type=smolvla --policy.push_to_hub=false \
    --dataset.repo_id="$dataset" "${cache_args[@]}" \
    --task_balanced_sampling=true \
    --batch_size="$batch_size" --gradient_accumulation_steps=1 \
    --steps="$steps" --save_freq="$save_freq" --eval_freq="$save_freq" \
    --log_freq=10 --num_workers="$num_workers" \
    --output_dir="$output" --job_name="v043_ablation_${variant}" \
    --policy.device=cuda --wandb.enable=true --wandb.disable_artifact=true \
    --policy.ablation_variant="$variant" \
    --policy.camera_views=agentview --policy.rgb_camera_views=agentview \
    --policy.vlm_model_name="$vlm" --policy.vlm_weights_path="$vlm_weights" \
    --policy.load_vlm_weights=true \
    --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps="$steps" --policy.scheduler_decay_lr=0.0000025 \
    --policy.pointseg_backbone_type=litept --policy.pointseg_grid_size=0.01 \
    --policy.pointseg_feature_dim=64 --policy.pointseg_aux_loss_weight=0.0005 \
    --policy.pointseg_foreground_ratio=0.025 --policy.pointseg_background_ratio=0.025 \
    --policy.pointseg_min_foreground_points=2500 --policy.pointseg_min_background_points=0 \
    --policy.pointseg_use_temporal_priors_as_input=false \
    --policy.pointseg_use_pseudo_selection=false \
    2>&1 | tee "$log"
}

train_all() {
  preflight
  mkdir -p "$root/logs" "$root/pids"
  local pids=()
  for i in "${!variants[@]}"; do
    train_one "${variants[$i]}" "${train_gpus[$i]}" &
    pids+=("$!")
    echo "$!" > "$root/pids/train_${variants[$i]}.pid"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

eval_one() {
  local variant=$1 gpu=$2 step=$3
  local step_tag
  printf -v step_tag '%06d' "$step"
  local checkpoint="$root/train/$variant/checkpoints/$step_tag/pretrained_model"
  local output="$root/eval/$variant/step$step_tag"
  local log="$root/logs/eval_${variant}_step${step_tag}.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "[eval] reuse $variant step=$step_tag"
    return
  fi
  while [[ ! -s "$checkpoint/model.safetensors" ]]; do
    local train_pid_file="$root/pids/train_${variant}.pid"
    if [[ -s "$train_pid_file" ]] && ! kill -0 "$(<"$train_pid_file")" 2>/dev/null; then
      echo "[eval] training ended before checkpoint appeared: $checkpoint" >&2
      return 1
    fi
    sleep 20
  done
  if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[eval] incomplete output exists: $output" >&2
    return 1
  fi
  mkdir -p "$output" "$root/logs"
  cd "$repo"
  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID="$gpu" PYTHONPATH="$repo/src" \
  "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path "$checkpoint" --device cuda \
    --num-points 10000 \
    --suite libero_10 --task-id 6 --task-id 8 --episodes "$eval_episodes" \
    --policy-noise-seed 0 --env-seed 7 --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 --isolated-policy-workers 1 \
    --task-workers 2 --episode-workers-per-task 2 --task-worker-backend process \
    --inference-batch-size 4 --inference-batching-mode fixed_barrier \
    --no-release-event-exec-enable --control-freq 20 --action-index 0 \
    --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
    --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
    --render-mode offscreen --no-visualize-foreground --no-save-video \
    --no-world-to-ego-causal-ablation --output-dir "$output" \
    2>&1 | tee "$log"
  test -s "$output/summary.json"
}

eval_watch_one() {
  local variant=$1 gpu=$2
  local step
  for ((step=save_freq; step<=steps; step+=save_freq)); do
    eval_one "$variant" "$gpu" "$step"
  done
}

eval_all() {
  require_inputs
  mkdir -p "$root/logs" "$root/pids"
  local pids=()
  for i in "${!variants[@]}"; do
    eval_watch_one "${variants[$i]}" "${eval_gpus[$i]}" &
    pids+=("$!")
    echo "$!" > "$root/pids/eval_${variants[$i]}.pid"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

summarize() {
  cd "$repo"
  "$python" benchmarks/song_real_libero/scripts/summarize_v043_ablation.py \
    --root "$root" --episodes-per-task "$eval_episodes"
}

status() {
  for variant in "${variants[@]}"; do
    local checkpoints summaries train_pid eval_pid
    checkpoints=$(find "$root/train/$variant/checkpoints" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    summaries=$(find "$root/eval/$variant" -name summary.json -type f 2>/dev/null | wc -l)
    train_pid=$(cat "$root/pids/train_${variant}.pid" 2>/dev/null || true)
    eval_pid=$(cat "$root/pids/eval_${variant}.pid" 2>/dev/null || true)
    echo "$variant checkpoints=$checkpoints evals=$summaries train_pid=${train_pid:-none} eval_pid=${eval_pid:-none}"
  done
}

case "${1:-}" in
  preflight) preflight ;;
  train) train_all ;;
  eval) eval_all ;;
  all)
    train_all & train_supervisor=$!
    eval_all & eval_supervisor=$!
    wait "$train_supervisor"
    wait "$eval_supervisor"
    summarize
    ;;
  summarize) summarize ;;
  status) status ;;
  *) usage; exit 2 ;;
esac
