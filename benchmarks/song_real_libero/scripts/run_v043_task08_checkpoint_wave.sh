#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT_A:GPU_A CHECKPOINT_B:GPU_B TAG_PREFIX" >&2
  exit 2
fi

root=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811
train="$root/pilots/libero10_task08_moka_50ep/training/stage1_union_fps_view_dropout_paired_cache_bnfixed_4gpu_b48_w12_lr25e6_100steps/checkpoints"
repo=/home/liusong/ProgramFiles/Huggingface/lerobot
tag_prefix=$3

run_one() {
  local specification=$1
  local checkpoint=${specification%%:*}
  local gpu=${specification##*:}
  local policy="$train/$checkpoint/pretrained_model"
  local tag="${tag_prefix}_step${checkpoint}_task08_50ep_gpu${gpu}_ew15_b15"
  local output="$root/eval/$tag"
  local log="$root/logs/eval_${tag}.log"
  test -s "$policy/model.safetensors"
  test ! -e "$output"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/liusong/anaconda3/envs/reap/bin/python3.10 \
    benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path "$policy" --device cuda --render-gpu-device-id "$gpu" \
    --suite libero_10 --task-id 8 --episodes 50 \
    --policy-noise-seed 0 --env-seed 7 --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 --task-worker-backend process \
    --task-workers 1 --episode-workers-per-task 15 --inference-batch-size 15 \
    --control-freq 20 --action-index 0 \
    --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
    --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
    --render-mode offscreen --no-visualize-foreground --no-save-video \
    --output-dir "$output" >"$log" 2>&1
}

cd "$repo"
run_one "$1" &
pid_a=$!
run_one "$2" &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
exit "$status"
