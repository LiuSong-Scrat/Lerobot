#!/usr/bin/env bash
set -euo pipefail

# Prefer the valid versioned NVIDIA driver over this host's stale zero-byte
# unversioned symlink.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

repo=${SONG_ABLATION_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot}
python=${SONG_ABLATION_PYTHON:-/home/liusong/anaconda3/envs/reap/bin/python}
dataset=${SONG_ABLATION_DATASET:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep}
cache=${SONG_ABLATION_CACHE:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache}
vlm=${SONG_ABLATION_VLM:-/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct}
vlm_weights=${SONG_ABLATION_VLM_WEIGHTS:-/opt/data/private/liusong/hf_models/smolvla_base}
root=${SONG_ABLATION_OUTPUT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/outputs/v043_cumulative_ablation_task6_task8_20260829}
steps=${SONG_ABLATION_STEPS:-30000}
checkpoint_interval_s=${SONG_ABLATION_CHECKPOINT_INTERVAL_S:-3600}
batch_size=${SONG_ABLATION_BATCH_SIZE:-8}
num_workers=${SONG_ABLATION_NUM_WORKERS:-2}
eval_episodes=${SONG_ABLATION_EVAL_EPISODES:-50}
guard_sample_s=${SONG_ABLATION_GUARD_SAMPLE_S:-5}
guard_poll_s=${SONG_ABLATION_GUARD_POLL_S:-15}

variants=(
  smolvla_src
  smolvla_pointcloud
  smolvla_pointcloud_effseg
  smolvla_pointcloud_effseg_pointaction
)
train_gpus=(0 1 2 3)
eval_gpus=(4 5 6 7)
guard="$repo/benchmarks/song_real_libero/scripts/ablation_resource_guard.py"
summarizer="$repo/benchmarks/song_real_libero/scripts/summarize_v043_ablation.py"

usage() {
  echo "usage: $0 {preflight|train|eval|all|summarize|status}"
}

require_inputs() {
  test -f "$dataset/meta/info.json"
  test -f "$vlm/config.json"
  test -f "$vlm_weights/model.safetensors"
  test -f "$cache/manifest.json"
  test -f "$guard"
  if (( steps <= 0 || checkpoint_interval_s <= 0 )); then
    echo "steps and checkpoint interval must be positive" >&2
    exit 2
  fi
  if (( eval_episodes != 50 )); then
    echo "The fixed task 6/8 test protocol requires 50 episodes per task." >&2
    exit 2
  fi
  if (( num_workers > 2 )); then
    echo "Refusing num_workers=$num_workers: four trainers are capped at two workers each." >&2
    exit 2
  fi
}

guard_wait() {
  local gpu=$1
  "$python" "$guard" --root "$root" --gpu "$gpu" --wait \
    --sample-seconds "$guard_sample_s" --poll-seconds "$guard_poll_s" --consecutive 3
}

wait_for_training_gpu_allocation() {
  local gpu=$1 pid=$2
  local deadline=$((SECONDS + 600))
  while kill -0 "$pid" 2>/dev/null; do
    local used_mib
    used_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    used_mib=${used_mib//[[:space:]]/}
    if (( used_mib >= 2048 )); then
      echo "[train] gpu=$gpu initialized with ${used_mib} MiB"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "[train] timed out waiting for GPU $gpu initialization" >&2
      return 1
    fi
    sleep 5
  done
  echo "[train] process $pid ended before GPU $gpu initialization" >&2
  return 1
}

preflight() {
  require_inputs
  cd "$repo"
  local branch
  branch=$(git branch --show-current)
  if [[ "$branch" != "wep_vla_v0.4.3_multiview_doubleflow_ablation" ]]; then
    echo "Expected ablation branch, got ${branch:-detached HEAD}." >&2
    exit 2
  fi
  mkdir -p "$root/logs" "$root/pids" "$root/resource"
  PYTHONPATH="$repo/src" "$python" - "$dataset" "$cache" "$eval_episodes" <<'PY'
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

dataset = Path(sys.argv[1])
cache = Path(sys.argv[2])
eval_episodes = int(sys.argv[3])
info = json.loads((dataset / "meta/info.json").read_text())
manifest = json.loads((cache / "manifest.json").read_text())
assert info["total_episodes"] == 100, info["total_episodes"]
assert info["total_tasks"] == 2, info["total_tasks"]
assert info["features"]["action"]["shape"] == [10]
assert manifest["num_samples"] == info["total_frames"]
assert manifest["current_points"] == 10_000
assert eval_episodes == 50
episode_tasks = set()
for parquet_path in sorted((dataset / "data").rglob("*.parquet")):
    columns = pq.read_table(parquet_path, columns=["episode_index", "task_index"]).to_pydict()
    episode_tasks.update(zip(columns["episode_index"], columns["task_index"], strict=True))
assert Counter(task for _, task in episode_tasks) == {0: 50, 1: 50}, episode_tasks

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

gpu_lines = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
assert len(gpu_lines) == 8, gpu_lines
for line in gpu_lines:
    index, total, used = [int(value.strip()) for value in line.split(",")]
    assert total >= 24_000, (index, total)
    assert used < 2_048, (index, used)
print(
    f"ablation preflight PASS: frames={info['total_frames']} episodes={info['total_episodes']} "
    f"test_episodes={2 * eval_episodes} gpus={len(gpu_lines)}"
)
PY
  "$python" "$guard" --root "$root" --sample-seconds 1
}

train_one() {
  local variant=$1 gpu=$2
  local output="$root/train/$variant"
  local log="$root/logs/train_${variant}.log"
  local final_tag
  printf -v final_tag '%06d' "$steps"
  if [[ -s "$output/checkpoints/$final_tag/pretrained_model/model.safetensors" ]]; then
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
  exec env \
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo/src" SONG_POINTSEG_REQUIRE_POINTOPS=1 \
    "$python" benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --policy.type=smolvla --policy.push_to_hub=false \
      --dataset.repo_id="$dataset" "${cache_args[@]}" \
      --task_balanced_sampling=true \
      --batch_size="$batch_size" --gradient_accumulation_steps=1 \
      --steps="$steps" --save_freq="$steps" --save_interval_s="$checkpoint_interval_s" \
      --eval_freq=0 --log_freq=10 --num_workers="$num_workers" \
      --output_dir="$output" --job_name="v043_ablation_${variant}" \
      --policy.device=cuda --wandb.enable=false \
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
      >"$log" 2>&1
}

train_all() {
  preflight
  local pids=()
  for i in "${!variants[@]}"; do
    guard_wait "${train_gpus[$i]}"
    train_one "${variants[$i]}" "${train_gpus[$i]}" &
    pids+=("$!")
    echo "$!" > "$root/pids/train_${variants[$i]}.pid"
    wait_for_training_gpu_allocation "${train_gpus[$i]}" "$!"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

checkpoint_ready() {
  local checkpoint=$1
  test -s "$checkpoint/model.safetensors" \
    && test -s "$checkpoint/config.json" \
    && test -s "$checkpoint/policy_preprocessor.json" \
    && test -s "$checkpoint/policy_postprocessor.json"
}

eval_one() {
  local variant=$1 gpu=$2 step=$3 checkpoint=$4
  local step_tag
  printf -v step_tag '%06d' "$step"
  local output="$root/eval/$variant/step$step_tag"
  local log="$root/logs/eval_${variant}_step${step_tag}.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "[eval] reuse $variant step=$step_tag"
    return
  fi
  if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[eval] incomplete output exists: $output" >&2
    return 1
  fi
  guard_wait "$gpu"
  mkdir -p "$output" "$root/logs"
  cd "$repo"
  env \
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
      >"$log" 2>&1
  test -s "$output/summary.json"
}

eval_watch_one() {
  local variant=$1 gpu=$2
  local train_pid_file="$root/pids/train_${variant}.pid"
  while true; do
    local found_pending=0
    while IFS= read -r model_path; do
      local checkpoint step_tag step
      checkpoint=$(dirname "$model_path")
      step_tag=$(basename "$(dirname "$checkpoint")")
      [[ "$step_tag" =~ ^[0-9]+$ ]] || continue
      step=$((10#$step_tag))
      if [[ ! -s "$root/eval/$variant/step$step_tag/summary.json" ]]; then
        if checkpoint_ready "$checkpoint"; then
          eval_one "$variant" "$gpu" "$step" "$checkpoint"
          flock "$root/summary.lock" "$python" "$summarizer" \
            --root "$root" --episodes-per-task "$eval_episodes"
        else
          found_pending=1
        fi
      fi
    done < <(find "$root/train/$variant/checkpoints" -mindepth 3 -maxdepth 3 \
      -path '*/pretrained_model/model.safetensors' -type f 2>/dev/null | sort -V)

    local train_pid
    train_pid=$(cat "$train_pid_file" 2>/dev/null || true)
    if [[ -n "$train_pid" ]] && ! kill -0 "$train_pid" 2>/dev/null; then
      if (( found_pending == 0 )); then
        break
      fi
      echo "[eval] training ended with an incomplete checkpoint for $variant" >&2
      return 1
    fi
    sleep 30
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

resource_watch() {
  "$python" "$guard" --root "$root" --watch --sample-seconds 10 --poll-seconds 20 \
    >>"$root/logs/resource_watch.log" 2>&1
}

summarize() {
  cd "$repo"
  "$python" "$summarizer" --root "$root" --episodes-per-task "$eval_episodes"
}

pid_state() {
  local file=$1 pid
  pid=$(cat "$file" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid:running"
  else
    echo "${pid:-none}:stopped"
  fi
}

status() {
  for variant in "${variants[@]}"; do
    local checkpoints summaries
    checkpoints=$(find "$root/train/$variant/checkpoints" -path '*/pretrained_model/model.safetensors' -type f 2>/dev/null | wc -l || true)
    summaries=$(find "$root/eval/$variant" -name summary.json -type f 2>/dev/null | wc -l || true)
    echo "$variant checkpoints=$checkpoints evals=$summaries train=$(pid_state "$root/pids/train_${variant}.pid") eval=$(pid_state "$root/pids/eval_${variant}.pid")"
  done
  tail -1 "$root/resource/samples.jsonl" 2>/dev/null || true
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}

case "${1:-}" in
  preflight) preflight ;;
  train) train_all ;;
  eval) eval_all ;;
  all)
    mkdir -p "$root/logs" "$root/pids"
    resource_watch & resource_pid=$!
    echo "$resource_pid" > "$root/pids/resource_watch.pid"
    trap 'kill "$resource_pid" 2>/dev/null || true' EXIT
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
