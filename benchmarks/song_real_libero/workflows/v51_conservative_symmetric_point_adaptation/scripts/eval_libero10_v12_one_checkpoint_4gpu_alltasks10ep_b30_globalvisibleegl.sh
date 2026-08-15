#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 CHECKPOINT_STEP" >&2
  exit 2
fi
step=$1
episodes_per_task=${EVAL_EPISODES_PER_TASK:-10}
case "$episodes_per_task" in
  10) cache_episode_tag= ;;
  50) cache_episode_tag=_50ep ;;
  *) echo "EVAL_EPISODES_PER_TASK must be 10 or 50" >&2; exit 2 ;;
esac
episode_ids_csv=${EVAL_EPISODE_IDS:-}
episode_id_args=()
if [[ -n "$episode_ids_csv" ]]; then
  IFS=, read -r -a episode_ids <<< "$episode_ids_csv"
  if (( ${#episode_ids[@]} != episodes_per_task )); then
    echo "EVAL_EPISODE_IDS must contain exactly EVAL_EPISODES_PER_TASK indices" >&2
    exit 2
  fi
  declare -A seen_episode_ids=()
  for episode_id in "${episode_ids[@]}"; do
    [[ "$episode_id" =~ ^[0-9]+$ ]] && (( episode_id < 50 )) || {
      echo "EVAL_EPISODE_IDS entries must be unique integers in [0,49]" >&2
      exit 2
    }
    [[ -z "${seen_episode_ids[$episode_id]:-}" ]] || {
      echo "EVAL_EPISODE_IDS entries must be unique integers in [0,49]" >&2
      exit 2
    }
    seen_episode_ids[$episode_id]=1
    episode_id_args+=(--episode-id "$episode_id")
  done
fi
suffix=${EVAL_RUN_SUFFIX:-}
if [[ -n "$suffix" && ! "$suffix" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "invalid EVAL_RUN_SUFFIX" >&2
  exit 2
fi
if [[ -n "$episode_ids_csv" && -z "$suffix" ]]; then
  echo "EVAL_RUN_SUFFIX is required when EVAL_EPISODE_IDS is set" >&2
  exit 2
fi
tag_suffix=
[[ -n "$suffix" ]] && tag_suffix="_${suffix}"
batching_mode=${EVAL_INFERENCE_BATCHING_MODE:-dynamic}
case "$batching_mode" in
  dynamic) batching_tag= ;;
  fixed_barrier) batching_tag=_fixedbarrierv18 ;;
  *) echo "EVAL_INFERENCE_BATCHING_MODE must be dynamic or fixed_barrier" >&2; exit 2 ;;
esac
policy_noise_seed=${EVAL_POLICY_NOISE_SEED:-0}
[[ "$policy_noise_seed" =~ ^-?[0-9]+$ ]] || { echo "EVAL_POLICY_NOISE_SEED must be an integer" >&2; exit 2; }
expected_worldflow=${EVAL_EXPECT_WORLDFLOW:-1}
[[ "$expected_worldflow" == 0 || "$expected_worldflow" == 1 ]] || {
  echo "EVAL_EXPECT_WORLDFLOW must be 0 or 1" >&2
  exit 2
}
expected_worldflow_action_fusion=${EVAL_EXPECT_WORLDFLOW_ACTION_FUSION:-endpoint_residual_boosting}
expected_worldflow_joint_token_layout=${EVAL_EXPECT_WORLDFLOW_JOINT_TOKEN_LAYOUT:-}
expected_worldflow_bootstrap_from_ego=${EVAL_EXPECT_WORLDFLOW_BOOTSTRAP_FROM_EGO:-0}
expected_worldflow_loss_weight=${EVAL_EXPECT_WORLDFLOW_LOSS_WEIGHT:-0}
expected_worldflow_parallel_canonical_action_flow=${EVAL_EXPECT_WORLDFLOW_PARALLEL_CANONICAL_ACTION_FLOW:-}
expected_worldflow_noise_coupling=${EVAL_EXPECT_WORLDFLOW_NOISE_COUPLING:-projected_ego_path}
[[ "$expected_worldflow_bootstrap_from_ego" == 0 || "$expected_worldflow_bootstrap_from_ego" == 1 ]] || {
  echo "EVAL_EXPECT_WORLDFLOW_BOOTSTRAP_FROM_EGO must be 0 or 1" >&2
  exit 2
}
[[ -z "$expected_worldflow_parallel_canonical_action_flow" \
  || "$expected_worldflow_parallel_canonical_action_flow" == 0 \
  || "$expected_worldflow_parallel_canonical_action_flow" == 1 ]] || {
  echo "EVAL_EXPECT_WORLDFLOW_PARALLEL_CANONICAL_ACTION_FLOW must be empty, 0, or 1" >&2
  exit 2
}
seed_tag=
(( policy_noise_seed == 0 )) || seed_tag="_seed${policy_noise_seed}"
inference_cache_mode=${EVAL_INFERENCE_CACHE_MODE:-off}
case "$inference_cache_mode" in
  off|read_write|readonly) ;;
  *) echo "EVAL_INFERENCE_CACHE_MODE must be off, read_write, or readonly" >&2; exit 2 ;;
esac
world_to_ego_causal_ablation=${EVAL_WORLD_TO_EGO_CAUSAL_ABLATION:-0}
[[ "$world_to_ego_causal_ablation" == 0 || "$world_to_ego_causal_ablation" == 1 ]] || {
  echo "EVAL_WORLD_TO_EGO_CAUSAL_ABLATION must be 0 or 1" >&2
  exit 2
}
ablation_tag=
ablation_args=(--no-world-to-ego-causal-ablation)
if [[ "$world_to_ego_causal_ablation" == 1 ]]; then
  ablation_tag=_worldtoegoablated
  ablation_args=(--world-to-ego-causal-ablation)
fi
pointcloud_camera_mode=${EVAL_POINTCLOUD_CAMERA_MODE:-checkpoint}
camera_tag=
camera_args=()
case "$pointcloud_camera_mode" in
  checkpoint) ;;
  primary_only)
    camera_tag=_primaryonly
    camera_args=(--camera agentview)
    ;;
  *)
    echo "EVAL_POINTCLOUD_CAMERA_MODE must be checkpoint or primary_only" >&2
    exit 2
    ;;
esac

ulimit -n 65535 2>/dev/null || true
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot}
required_commit=fbfacd7d8a0b8c666b66a870ddd3cad4e9e8f5d8
root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
train=${EVAL_TRAIN_DIR:-$root/singleview_worldflow/libero10_500ep/training/worldflow_endpoint_residual_taskbalanced_v12_actiononly_from_v9step75_4gpu_b48_accum1_w12_720steps}
python=/home/liusong/anaconda3/envs/reap/bin/python3.10
merger="$root/scripts/merge_libero10_task_partitions.py"
printf -v step6 '%06d' "$step"
checkpoint=${EVAL_CHECKPOINT_OVERRIDE:-$train/checkpoints/$step6/pretrained_model}
run_label=${EVAL_RUN_LABEL:-v12_actiononly}
artifact_label=${EVAL_ARTIFACT_LABEL:-v12_actiononly}
if [[ ! "$run_label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || [[ ! "$artifact_label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "EVAL_RUN_LABEL and EVAL_ARTIFACT_LABEL must be safe path components" >&2
  exit 2
fi
base_tag="libero10_worldflow_endpoint_residual_${run_label}_step${step6}_all10tasks${episodes_per_task}ep_4x4090_total30_b30_codefbfacd7${batching_tag}${seed_tag}"
tag="${base_tag}${camera_tag}${ablation_tag}${tag_suffix}"
output="$root/singleview_worldflow/libero10_500ep/eval_4gpu_${episodes_per_task}ep/$tag"
log_root="$root/singleview_worldflow/libero10_500ep/logs/4gpu_${episodes_per_task}ep"
artifact="$root/singleview_worldflow/libero10_500ep/artifacts/taskbalanced_${artifact_label}_step${step6}_4gpu_total30_b30_alltasks${episodes_per_task}ep_codefbfacd7${batching_tag}${seed_tag}${camera_tag}${ablation_tag}${tag_suffix}.json"
inference_cache_dir=${EVAL_INFERENCE_CACHE_DIR:-$root/singleview_worldflow/libero10_500ep/inference_cache/4gpu_total30_b30/${artifact_label}_step${step6}${batching_tag}${seed_tag}${camera_tag}${cache_episode_tag}}

test -s "$checkpoint/model.safetensors"
test -x "$python"
test -x "$merger"
git -C "$repo" cat-file -e "$required_commit^{commit}"
git -C "$repo" merge-base --is-ancestor "$required_commit" HEAD
mkdir -p "$log_root" "$(dirname "$artifact")"

env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python" - \
  "$repo" "$checkpoint" "${EVAL_CHECKPOINT_OVERRIDE:+override}" "$expected_worldflow" \
  "$expected_worldflow_action_fusion" "$expected_worldflow_joint_token_layout" \
  "$expected_worldflow_bootstrap_from_ego" "$expected_worldflow_loss_weight" \
  "$expected_worldflow_parallel_canonical_action_flow" "$expected_worldflow_noise_coupling" <<'PY'
import pathlib,sys
repo=pathlib.Path(sys.argv[1]).resolve(); checkpoint=pathlib.Path(sys.argv[2]).resolve(); override=bool(sys.argv[3]); expected_worldflow=bool(int(sys.argv[4]))
expected_action_fusion=sys.argv[5]; expected_joint_layout=sys.argv[6]
expected_bootstrap=bool(int(sys.argv[7])); expected_loss_weight=float(sys.argv[8])
expected_parallel_canonical=sys.argv[9]
expected_noise_coupling=sys.argv[10]
from lerobot.configs.policies import PreTrainedConfig
import lerobot.policies.smolvla.configuration_smolvla as cm
import lerobot.policies.smolvla.modeling_smolvla as mm
for path in (pathlib.Path(cm.__file__).resolve(),pathlib.Path(mm.__file__).resolve()):
    assert repo/"src" in path.parents, f"stale import: {path}"
c=PreTrainedConfig.from_pretrained(checkpoint)
assert bool(c.worldflow_enable) is expected_worldflow
assert c.se3_enable is False
if expected_worldflow:
    assert c.worldflow_action_fusion==expected_action_fusion
    if expected_joint_layout:
        assert c.worldflow_joint_token_layout==expected_joint_layout
    if expected_parallel_canonical:
        assert bool(c.worldflow_parallel_canonical_action_flow) is bool(int(expected_parallel_canonical))
    assert c.worldflow_noise_coupling==expected_noise_coupling
    assert c.worldflow_ego_residual_gate_init is None
    if not override:
        assert c.worldflow_bootstrap_from_ego is expected_bootstrap
        assert float(c.worldflow_loss_weight)==expected_loss_weight
        assert all(getattr(c,k)==0 for k in ("worldflow_geo_loss_weight","worldflow_bridge_loss_weight","worldflow_equiv_loss_weight"))
print(f"[code-preflight] PASS {checkpoint}")
PY

task_groups=("0 4 8" "1 5 9" "2 6" "3 7")
task_workers=(3 3 2 2)
parts=(); pids=()
for gpu in 0 1 2 3; do
  part="$output/parts/gpu_${gpu}"
  summary="$part/summary.json"
  parts+=("$summary")
  if [[ -s "$summary" ]]; then
    "$python" - "$summary" "$checkpoint" "$episodes_per_task" "$episode_ids_csv" ${task_groups[$gpu]} <<'PY'
import json,pathlib,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); ckpt=pathlib.Path(sys.argv[2]).resolve(); episodes=int(sys.argv[3]); episode_ids=[int(value) for value in sys.argv[4].split(",") if value]; ids=sorted(map(int,sys.argv[5:]))
assert pathlib.Path(d["policy_path"]).resolve()==ckpt
assert sorted(int(t["task_id"]) for t in d["results"])==ids
assert all(len(t["episodes"])==episodes for t in d["results"])
if episode_ids:
    assert all(sorted(int(e["episode_index"]) for e in t["episodes"])==sorted(episode_ids) for t in d["results"])
print(f"[reuse] {sys.argv[1]}")
PY
    continue
  fi
  test ! -e "$part"
  mkdir -p "$(dirname "$part")"
  args=(); for task in ${task_groups[$gpu]}; do args+=(--task-id "$task"); done
  cache_args=()
  if [[ "$inference_cache_mode" != off ]]; then
    cache_args+=(--inference-cache-dir "$inference_cache_dir" --inference-cache-mode "$inference_cache_mode")
  fi
  part_log="$log_root/eval_${tag}_gpu${gpu}.log"
  (
    cd "$repo"
    env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES=0,1,2,3 MUJOCO_EGL_DEVICE_ID="$gpu" \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
        --config benchmarks/song_real_libero/configs/libero.json \
        --policy.path "$checkpoint" --device "cuda:$gpu" --render-gpu-device-id "$gpu" \
        --suite libero_10 --no-all-tasks "${args[@]}" --episodes "$episodes_per_task" \
        "${episode_id_args[@]}" \
        "${camera_args[@]}" \
        --policy-noise-seed "$policy_noise_seed" --env-seed 7 --strict-official-init \
        --gripper-control-mode delta_width_initial_sync --gripper-delta-threshold 0.002 \
        --gripper-delta-alignment current_minus_previous --waypoint-max-hold-steps 1 \
        --isolated-policy-workers 1 --task-worker-backend process \
        --task-workers "${task_workers[$gpu]}" --episode-workers-per-task 3 \
        --inference-batch-size 30 --inference-batching-mode "$batching_mode" \
        "${cache_args[@]}" \
        "${ablation_args[@]}" \
        --control-freq 20 --action-index 0 \
        --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
        --render-mode offscreen --no-visualize-foreground --no-save-video \
        --output-dir "$part" >"$part_log" 2>&1
  ) & pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; (( status==0 ))

if [[ ! -s "$output/summary.json" ]]; then
  merge_episode_args=()
  if [[ -n "$episode_ids_csv" ]]; then
    merge_episode_args+=(--episode-ids "$episode_ids_csv")
  fi
  if [[ "$world_to_ego_causal_ablation" == 1 ]]; then
    "$python" "$merger" --output-dir "$output" --episodes-per-task "$episodes_per_task" \
      "${merge_episode_args[@]}" --partition-layout legacy4 --allow-diagnostic-ablation "${parts[@]}" \
      | tee "$log_root/merge_${tag}.log"
  else
    "$python" "$merger" --output-dir "$output" --episodes-per-task "$episodes_per_task" \
      "${merge_episode_args[@]}" --partition-layout legacy4 "${parts[@]}" \
      | tee "$log_root/merge_${tag}.log"
  fi
fi
"$python" - "$output/summary.json" "$artifact" "$step" "$suffix" "$batching_mode" "$policy_noise_seed" "$world_to_ego_causal_ablation" "$episode_ids_csv" "$pointcloud_camera_mode" <<'PY'
import json,pathlib,sys
summary=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); step=int(sys.argv[3]); suffix=sys.argv[4]
batching_mode=sys.argv[5]; policy_noise_seed=int(sys.argv[6])
world_to_ego_causal_ablation=bool(int(sys.argv[7]))
episode_ids=[int(value) for value in sys.argv[8].split(",") if value]
pointcloud_camera_mode=sys.argv[9]
d=json.load(summary.open(encoding="utf-8")); p={"step":step,"success_count":int(d["overall"]["success_count"]),
 "episode_count":int(d["overall"]["episode_count"]),"success_rate":float(d["overall"]["success_rate"]),
 "hardware_schedule":"four RTX 4090 GPUs; 30 environment workers total; inference batch limit 30",
 "inference_batching_mode":batching_mode,"policy_noise_seed":policy_noise_seed,
 "inference_cache_mode":d["execution"].get("inference_cache_mode","off"),
 "inference_cache_dir":d["execution"].get("inference_cache_dir"),
 "summary":str(summary)}
if world_to_ego_causal_ablation:
    p["world_to_ego_causal_ablation"]=True
    p["benchmark_comparable"]=False
if pointcloud_camera_mode != "checkpoint":
    p["pointcloud_camera_mode"]=pointcloud_camera_mode
    p["benchmark_comparable"]=False
if episode_ids:
    p["episode_ids"]=episode_ids
    p["benchmark_comparable"]=False
if suffix:p["run_suffix"]=suffix
if out.exists(): assert json.load(out.open(encoding="utf-8"))==p
else: out.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")
print(json.dumps(p,indent=2))
PY
