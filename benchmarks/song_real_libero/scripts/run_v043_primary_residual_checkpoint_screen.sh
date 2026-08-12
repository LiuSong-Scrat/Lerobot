#!/usr/bin/env bash
set -euo pipefail

repo=/home/liusong/ProgramFiles/Huggingface/lerobot
root=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811
pilot="$root/pilots/libero10_task08_moka_50ep"
train=${SONG_V043_SCREEN_TRAIN_ROOT:-$pilot/training/stage1_primary_residual_twotimescale_4gpu_b48_w12_100steps}
python=/home/liusong/anaconda3/envs/reap/bin/python3.10
training_session=${SONG_V043_SCREEN_TRAINING_SESSION:-wep_v043_primary_residual_pilot}
tag_prefix=${SONG_V043_SCREEN_TAG_PREFIX:-primary_residual}

if [[ ! "$tag_prefix" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsafe checkpoint-screen tag prefix: $tag_prefix" >&2
  exit 1
fi

steps=(25 50 75 100)
gpus=(0 1 2 3)
workers=(8 8 7 7)

mkdir -p "$pilot/eval" "$pilot/logs"

all_checkpoints_ready() {
  local step checkpoint
  for step in "${steps[@]}"; do
    printf -v checkpoint '%s/checkpoints/%06d/pretrained_model/model.safetensors' "$train" "$step"
    [[ -s "$checkpoint" ]] || return 1
  done
}

while ! all_checkpoints_ready; do
  if ! tmux has-session -t "$training_session" 2>/dev/null; then
    echo "Training session ended before all checkpoints were written: $training_session" >&2
    exit 1
  fi
  sleep 20
done

# A checkpoint file can become visible just before Accelerate releases CUDA memory.
while pgrep -f "train_song_benchmark.py.*${train}" >/dev/null; do
  sleep 10
done

run_one() {
  local step=$1
  local gpu=$2
  local parallel_workers=$3
  local checkpoint tag output log
  printf -v checkpoint '%s/checkpoints/%06d/pretrained_model' "$train" "$step"
  printf -v tag '%s_step%06d_task08_full50_gpu%d_w%d_b%d' \
    "$tag_prefix" "$step" "$gpu" "$parallel_workers" "$parallel_workers"
  output="$pilot/eval/$tag"
  log="$pilot/logs/eval_${tag}.log"

  if [[ -e "$output" ]]; then
    echo "Evaluation output already exists; refusing to overwrite: $output" >&2
    return 1
  fi
  test -s "$checkpoint/config.json"
  test -s "$checkpoint/model.safetensors"

  cd "$repo"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$gpu" \
    MUJOCO_EGL_DEVICE_ID="$gpu" "$python" \
    benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path "$checkpoint" \
    --suite libero_10 --no-all-tasks --task-id 8 \
    --device cuda --render-gpu-device-id "$gpu" \
    --episodes 50 --policy-noise-seed 0 --env-seed 7 --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 --task-worker-backend process \
    --task-workers 1 --episode-workers-per-task "$parallel_workers" \
    --inference-batch-size "$parallel_workers" \
    --control-freq 20 --action-index 0 \
    --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
    --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
    --render-mode offscreen --no-visualize-foreground --no-save-video \
    --output-dir "$output" 2>&1 | tee -a "$log"
}

pids=()
for index in "${!steps[@]}"; do
  run_one "${steps[$index]}" "${gpus[$index]}" "${workers[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "At least one checkpoint evaluation failed; retained every artifact and log." >&2
  exit "$status"
fi

echo "$tag_prefix task-8 checkpoint screen complete (total episode workers: 30)"
find "$pilot/eval" -path "*${tag_prefix}_step*_task08_full50*/summary.json" -print -exec \
  "$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in ("successes","episodes","success_rate")})' {} \;
