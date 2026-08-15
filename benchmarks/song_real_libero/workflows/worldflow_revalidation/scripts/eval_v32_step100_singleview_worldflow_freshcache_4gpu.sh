#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ "$1" != normal && "$1" != disabled ]]; then
  echo "usage: $0 normal|disabled" >&2
  exit 2
fi
arm=$1

repo=/home/liusong/ProgramFiles/Huggingface/lerobot_worldflow_v32_revalidation
required_code_ancestor=dba65f7e0466364dcb43eaa016a7d7fb3c05c90d
python=/home/liusong/anaconda3/envs/reap/bin/python3.10
root=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811
checkpoint=$root/singleview_worldflow/libero10_500ep/training/v32_from_v14_worldbnfixed_w2e_p75_anchor_residual_rate_coordframe_bodyframe_ego_tangent_common005_world02_residual4_4gpu_b32_w12_1080steps/checkpoints/000100/pretrained_model
merger=$root/scripts/merge_libero10_task_partitions.py
run_id=v32step100_singleview_worldflow_revalidation_freshcache_seed0_childenvfix_${arm}
output=$root/singleview_worldflow/libero10_500ep/eval_4gpu_50ep/$run_id
cache=$root/singleview_worldflow/libero10_500ep/inference_cache/4gpu_total30_b30/$run_id
log_root=$root/singleview_worldflow/libero10_500ep/logs/4gpu_50ep/$run_id
artifact=$root/singleview_worldflow/libero10_500ep/artifacts/${run_id}.json

git -C "$repo" merge-base --is-ancestor "$required_code_ancestor" HEAD
[[ -z "$(git -C "$repo" status --short)" ]]
[[ "$(sha256sum "$repo/benchmarks/song_real_libero/scripts/smolvla_model_inference.py" | awk '{print $1}')" == 9e2d770bbff738d4aa883dd9dff156c3e204ee0e452c933b699d71a0d1ce95d0 ]]
[[ "$(sha256sum "$checkpoint/model.safetensors" | awk '{print $1}')" == c258303b70d4cab64f89d93c825905813f6f49fbb08cf282b5d4321e1fdf1fb4 ]]
test -x "$python"
test -x "$merger"
test ! -e "$output"
test ! -e "$cache"
test ! -e "$artifact"
mkdir -p "$log_root"
ulimit -n 65535 2>/dev/null || true

env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python" - "$checkpoint" <<'PY'
import pathlib
import sys
import lerobot.policies.smolvla.configuration_smolvla
from lerobot.configs.policies import PreTrainedConfig

c = PreTrainedConfig.from_pretrained(pathlib.Path(sys.argv[1]))
assert c.camera_views == "agentview"
assert c.camera_view_fusion == "legacy_budget"
assert c.worldflow_enable is True
assert c.worldflow_action_fusion == "endpoint_residual_boosting"
assert c.worldflow_noise_coupling == "projected_ego_path"
assert c.worldflow_frame_origin == "current_ee"
assert c.se3_enable is False
print("[preflight] exact V32 step100 single-view WorldFlow checkpoint PASS")
PY

ablation_args=(--no-world-to-ego-causal-ablation)
if [[ "$arm" == disabled ]]; then
  ablation_args=(--world-to-ego-causal-ablation)
fi

task_groups=("0 4 8" "1 5 9" "2 6" "3 7")
task_workers=(3 3 2 2)
parts=()
pids=()
for gpu in 0 1 2 3; do
  part=$output/parts/gpu_$gpu
  parts+=("$part/summary.json")
  mkdir -p "$(dirname "$part")"
  args=()
  for task in ${task_groups[$gpu]}; do
    args+=(--task-id "$task")
  done
  (
    cd "$repo"
    env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
      SONG_LIBERO_ENV_CUDA_VISIBLE_DEVICES=0 SONG_LIBERO_ENV_MUJOCO_EGL_DEVICE_ID=0 \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
      "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
        --config benchmarks/song_real_libero/configs/libero.json \
        --policy.path "$checkpoint" --device cuda --render-gpu-device-id "$gpu" \
        --suite libero_10 --no-all-tasks "${args[@]}" --episodes 50 \
        --policy-noise-seed 0 --env-seed 7 --strict-official-init \
        --gripper-control-mode delta_width_initial_sync --gripper-delta-threshold 0.002 \
        --gripper-delta-alignment current_minus_previous --waypoint-max-hold-steps 1 \
        --isolated-policy-workers 1 --task-worker-backend process \
        --task-workers "${task_workers[$gpu]}" --episode-workers-per-task 3 \
        --inference-batch-size 30 --inference-batching-mode fixed_barrier \
        --inference-cache-dir "$cache" --inference-cache-mode read_write \
        "${ablation_args[@]}" \
        --control-freq 20 --action-index 0 --exec-action-steps 24 \
        --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
        --render-mode offscreen --no-visualize-foreground --no-save-video \
        --output-dir "$part" >"$log_root/gpu_${gpu}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 ))

merge_args=()
if [[ "$arm" == disabled ]]; then
  merge_args+=(--allow-diagnostic-ablation)
fi
"$python" "$merger" \
  --output-dir "$output" --episodes-per-task 50 --partition-layout legacy4 \
  "${merge_args[@]}" "${parts[@]}" | tee "$log_root/merge.log"

"$python" - "$output/summary.json" "$artifact" "$arm" "$(git -C "$repo" rev-parse HEAD)" "$required_code_ancestor" <<'PY'
import hashlib
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
artifact_path = pathlib.Path(sys.argv[2])
arm = sys.argv[3]
commit = sys.argv[4]
code_ancestor = sys.argv[5]
d = json.loads(summary_path.read_text(encoding="utf-8"))
assert d["pointcloud_camera_names"] == ["agentview"]
assert d["image_camera_names"] == ["agentview"]
assert d["execution"]["inference_batching_mode"] == "fixed_barrier"
assert d["execution"]["inference_cache_mode"] == "read_write"
assert int(d["overall"]["episode_count"]) == 500
p = {
    "schema": "v32_singleview_worldflow_freshcache_revalidation_arm_v1",
    "arm": arm,
    "world_to_ego_causal_ablation": arm == "disabled",
    "checkpoint_step": 100,
    "git_commit": commit,
    "required_code_ancestor": code_ancestor,
    "evaluator_change_scope": "episode-child CUDA/EGL environment remapping only",
    "pointcloud_cameras": ["agentview"],
    "image_cameras": ["agentview"],
    "episode_count": int(d["overall"]["episode_count"]),
    "success_count": int(d["overall"]["success_count"]),
    "success_rate": float(d["overall"]["success_rate"]),
    "policy_noise_seed": 0,
    "env_seed": 7,
    "inference_batching_mode": "fixed_barrier",
    "hardware_schedule": "four RTX 4090 GPUs; 30 environment workers total; inference batch limit 30",
    "fresh_exact_action_cache": True,
    "inference_cache_dir": d["execution"]["inference_cache_dir"],
    "summary": str(summary_path),
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
}
artifact_path.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
print(json.dumps(p, indent=2))
PY
