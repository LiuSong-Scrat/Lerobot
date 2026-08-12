#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 POLICY_PATH TAG TASK_ID EPISODES EXPERIMENT_ROOT" >&2
  exit 2
fi
policy_path=$1
tag=$2
task_id=$3
episodes=$4
root=$5
case "$episodes" in 10|50) ;; *) echo "EPISODES must be 10 or 50" >&2; exit 2 ;; esac
if [[ ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "TAG contains unsafe characters: $tag" >&2
  exit 2
fi
if (( task_id < 0 || task_id > 9 )); then
  echo "TASK_ID must be in [0, 9]" >&2
  exit 2
fi

repo=/home/liusong/ProgramFiles/Huggingface/lerobot
output="$root/eval/$tag"
log="$root/logs/eval_${tag}.log"
merger="$root/scripts/merge_libero_task_episode_partitions.py"

test -s "$policy_path/model.safetensors"
test -s "$policy_path/config.json"
test -s "$merger"
test ! -e "$output"
mkdir -p "$output/parts"

if (( episodes == 50 )); then
  starts=(0 13 26 38)
  ends=(12 25 37 49)
  episode_workers=(8 8 7 7)
else
  starts=(0 3 6 8)
  ends=(2 5 7 9)
  episode_workers=(3 3 2 2)
fi

cd "$repo"
pids=()
for gpu in 0 1 2 3; do
  episode_args=()
  for ((episode_id=starts[gpu]; episode_id<=ends[gpu]; episode_id++)); do
    episode_args+=(--episode-id "$episode_id")
  done
  part="$output/parts/gpu_${gpu}"
  part_log="$root/logs/eval_${tag}_gpu${gpu}.log"
  env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/liusong/anaconda3/envs/reap/bin/python3.10 \
    benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path "$policy_path" --device cuda --render-gpu-device-id "$gpu" \
    --suite libero_10 --task-id "$task_id" --episodes "$episodes" "${episode_args[@]}" \
    --policy-noise-seed 0 --env-seed 7 --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 --task-worker-backend process \
    --task-workers 1 --episode-workers-per-task "${episode_workers[$gpu]}" \
    --inference-batch-size "${episode_workers[$gpu]}" \
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
  echo "One or more four-GPU episode partitions failed; inspect ${log%.log}_gpu*.log" \
    | tee -a "$log" >&2
  exit 1
fi

/home/liusong/anaconda3/envs/reap/bin/python3.10 "$merger" \
  --output-dir "$output" --task-id "$task_id" --episodes "$episodes" \
  "$output/parts/gpu_0/summary.json" "$output/parts/gpu_1/summary.json" \
  "$output/parts/gpu_2/summary.json" "$output/parts/gpu_3/summary.json" | tee -a "$log"
