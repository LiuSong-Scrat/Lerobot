#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 POLICY_PATH TAG EPISODES EXPERIMENT_ROOT" >&2
  exit 2
fi
policy_path=$1
tag=$2
episodes=$3
root=$4
case "$episodes" in 10|50) ;; *) echo "EPISODES must be 10 or 50" >&2; exit 2 ;; esac
if [[ ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "TAG contains unsafe characters: $tag" >&2
  exit 2
fi

repo=/home/liusong/ProgramFiles/Huggingface/lerobot
output="$root/eval/$tag"
log="$root/logs/eval_${tag}.log"
merger="$root/scripts/merge_libero10_task_partitions.py"

test -s "$policy_path/model.safetensors"
test -s "$policy_path/config.json"
test -x "$merger"
test ! -e "$output"
mkdir -p "$output/parts"

cd "$repo"
task_groups=("0 4 8" "1 5 9" "2 6" "3 7")
task_workers=(3 3 2 2)
pids=()
for gpu in 0 1 2 3; do
  args=()
  for task in ${task_groups[$gpu]}; do args+=(--task-id "$task"); done
  part="$output/parts/gpu_${gpu}"
  part_log="$root/logs/eval_${tag}_gpu${gpu}.log"
  env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/liusong/anaconda3/envs/reap/bin/python3.10 \
    benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path "$policy_path" --device cuda --render-gpu-device-id "$gpu" \
    --suite libero_10 "${args[@]}" --episodes "$episodes" \
    --policy-noise-seed 0 --env-seed 7 --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 --task-worker-backend process \
    --task-workers "${task_workers[$gpu]}" --episode-workers-per-task 3 \
    --inference-batch-size 30 \
    --control-freq 20 --action-index 0 \
    --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
    --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
    --render-mode offscreen --no-visualize-foreground --no-save-video \
    --output-dir "$part" >"$part_log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then
  echo "One or more four-GPU evaluation partitions failed; inspect ${log%.log}_gpu*.log" \
    | tee -a "$log" >&2
  exit 1
fi

"$merger" --output-dir "$output" --episodes-per-task "$episodes" \
  "$output/parts/gpu_0/summary.json" "$output/parts/gpu_1/summary.json" \
  "$output/parts/gpu_2/summary.json" "$output/parts/gpu_3/summary.json" | tee -a "$log"
