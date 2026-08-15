#!/usr/bin/env bash
set -euo pipefail

repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v52_consensus_multiscale}
root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
python=${PYTHON_BIN:-/home/liusong/anaconda3/envs/reap/bin/python3.10}
workflow_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runner="$workflow_root/scripts/eval_libero10_v52_one_checkpoint_4gpu_childenvfix.sh"
train="$root/joint_multiview_worldflow/libero10_500ep/training/v52_from_v32step100_consensus_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps"
artifact_root="$root/singleview_worldflow/libero10_500ep/artifacts"
joint_artifact_root="$root/joint_multiview_worldflow/libero10_500ep/artifacts"
step=${V52_STEP:-1564}
[[ "$step" =~ ^(260|520|780|1040|1300|1564)$ ]] || { echo "invalid V52_STEP=$step" >&2; exit 2; }
printf -v step6 '%06d' "$step"
checkpoint="$train/checkpoints/$step6/pretrained_model"
aggregate=${V52_STEP_SCREEN_ARTIFACT:-"$joint_artifact_root/v52childenvfix_step${step6}_stratified_3arm_multiview_worldflow_screen.json"}
convergence_artifact="$joint_artifact_root/v52_training_convergence_complete.json"
drift_artifact="$joint_artifact_root/v52_parameter_drift_source_v32step100_to_step001564.json"
episode_ids=0,5,10,15,20,25,30,35,40,45
label=v52childenvfix_v32_consensus1cm4cm_paired_symmetricpoint5e8_policy5e9_b44
train_session=wep_v043_v52_audit_then_paired_train
cache_session=wep_v043_v52_consensus_cache_audit
cache_waiter_session=wep_v043_v52_audit_then_paired_train
monitor_session=wep_v043_v52_convergence_monitor

test -x "$runner"
test -x "$python"
mkdir -p "$joint_artifact_root"

if [[ -s "$aggregate" ]]; then
  "$python" - "$aggregate" <<'PY'
import json,sys
print(json.dumps(json.load(open(sys.argv[1],encoding="utf-8")),indent=2))
PY
  exit 0
fi

while [[ ! -s "$checkpoint/model.safetensors" ]]; do
  if ! tmux has-session -t "=$train_session" 2>/dev/null \
      && ! tmux has-session -t "=$cache_session" 2>/dev/null \
      && ! tmux has-session -t "=$cache_waiter_session" 2>/dev/null; then
    echo "V52 pipeline exited without checkpoint: $checkpoint" >&2
    exit 4
  fi
  sleep 30
done
while tmux has-session -t "=$train_session" 2>/dev/null; do sleep 20; done

while [[ ! -s "$convergence_artifact" || ! -s "$drift_artifact" ]]; do
  if ! tmux has-session -t "=$monitor_session" 2>/dev/null; then
    echo "V52 convergence/parameter monitor exited without required final artifacts" >&2
    exit 5
  fi
  sleep 10
done

"$python" - "$convergence_artifact" "$drift_artifact" <<'PY'
import json,sys
convergence=json.load(open(sys.argv[1],encoding="utf-8"))
drift=json.load(open(sys.argv[2],encoding="utf-8"))
assert convergence["expected_final_step"]==1564
assert convergence["maximum_parsed_step"]==1564
assert convergence["training_complete"] is True
assert convergence["heuristic_only_not_a_performance_gate"] is True
assert drift["expected_step"]==1564
assert drift["exact_architecture_key_match"] is True
assert drift["all_optimizer_groups_changed"] is True
assert drift["all_architecture_roles_changed"] is True
assert drift["symmetric_world_point_path"] is True
assert drift["passes_optimizer_state"] is True
assert drift["passes_parameter_drift_audit"] is True
print("[V52 final training evidence] convergence and parameter-drift audits PASS")
PY

"$python" - "$repo" "$checkpoint" <<'PY'
import json,pathlib,sys
repo=pathlib.Path(sys.argv[1]).resolve()
p=pathlib.Path(sys.argv[2]).resolve()
c=json.load((p/"config.json").open(encoding="utf-8"))
assert c["camera_views"] in ("agentview,robot0_eye_in_hand",["agentview","robot0_eye_in_hand"])
assert c["rgb_camera_views"] in ("agentview",["agentview"])
assert c["camera_view_fusion"]=="consensus_multiscale_novelty_union"
assert c["camera_view_voxel_size"]==0.01
assert c["camera_view_coarse_novelty_scale"]==4.0
assert c["camera_view_fps_target_points"]==10_000
assert c["camera_view_fps_gripper_points"]==500
assert c["multiview_input_view_dropout_enable"] is True
assert c["multiview_input_view_dropout_paired_coverage"] is True
assert c["multiview_input_symmetric_point_path_adaptation"] is True
assert c["worldflow_enable"] is True
assert c["worldflow_action_fusion"]=="endpoint_residual_boosting"
assert c["worldflow_noise_coupling"]=="projected_ego_path"
assert c["worldflow_ego_residual_gate_init"] is None
assert c["se3_enable"] is False
assert all(c[k]==0 for k in (
    "worldflow_loss_weight","worldflow_geo_loss_weight",
    "worldflow_bridge_loss_weight","worldflow_equiv_loss_weight"))
assert (repo/"src/lerobot/policies/smolvla/song_pointseg.py").is_file()
print(f"[V52 stratified preflight] PASS {p}")
PY

common_env=(
  LEROBOT_REPO="$repo"
  EXPERIMENT_ROOT="$root"
  EVAL_TRAIN_DIR="$train"
  EVAL_CHECKPOINT_OVERRIDE="$checkpoint"
  EVAL_EPISODES_PER_TASK=10
  EVAL_EPISODE_IDS="$episode_ids"
  EVAL_INFERENCE_BATCHING_MODE=fixed_barrier
  EVAL_POLICY_NOISE_SEED=0
  EVAL_INFERENCE_CACHE_MODE=read_write
  EVAL_EXPECT_WORLDFLOW=1
  EVAL_EXPECT_WORLDFLOW_ACTION_FUSION=endpoint_residual_boosting
  EVAL_EXPECT_WORLDFLOW_BOOTSTRAP_FROM_EGO=0
  EVAL_EXPECT_WORLDFLOW_LOSS_WEIGHT=0
  EVAL_EXPECT_WORLDFLOW_NOISE_COUPLING=projected_ego_path
  EVAL_RUN_LABEL="$label"
  EVAL_ARTIFACT_LABEL="$label"
)

run_arm() {
  local camera_mode=$1
  local world_ablation=$2
  local suffix=$3
  local cache_name=$4
  env "${common_env[@]}" \
    EVAL_POINTCLOUD_CAMERA_MODE="$camera_mode" \
    EVAL_WORLD_TO_EGO_CAUSAL_ABLATION="$world_ablation" \
    EVAL_RUN_SUFFIX="$suffix" \
    EVAL_INFERENCE_CACHE_DIR="$root/joint_multiview_worldflow/libero10_500ep/inference_cache/4gpu_total30_b30/$cache_name" \
    bash "$runner" "$step"
}

run_arm_background() {
  local camera_mode=$1
  local world_ablation=$2
  local suffix=$3
  local cache_name=$4
  setsid env "${common_env[@]}" \
    EVAL_POINTCLOUD_CAMERA_MODE="$camera_mode" \
    EVAL_WORLD_TO_EGO_CAUSAL_ABLATION="$world_ablation" \
    EVAL_RUN_SUFFIX="$suffix" \
    EVAL_INFERENCE_CACHE_DIR="$root/joint_multiview_worldflow/libero10_500ep/inference_cache/4gpu_total30_b30/$cache_name" \
    bash "$runner" "$step" &
  arm_pid=$!
}

read_progress() {
  "$python" - "$1" <<'PY'
import glob,json,pathlib,sys
rows=[]
for path in glob.glob(str(pathlib.Path(sys.argv[1])/"parts/gpu_*/progress.json")):
    try: rows.append(json.load(open(path,encoding="utf-8")))
    except Exception: pass
print(sum(int(r.get("completed_episode_count",0)) for r in rows),
      sum(int(r.get("success_count",0)) for r in rows),
      sum(int(r.get("failure_count",0)) for r in rows))
PY
}

stop_arm() {
  kill -TERM -- "-$arm_pid" 2>/dev/null || kill -TERM "$arm_pid" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "$arm_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$arm_pid" 2>/dev/null; then
    kill -KILL -- "-$arm_pid" 2>/dev/null || kill -KILL "$arm_pid" 2>/dev/null || true
  fi
  wait "$arm_pid" || true
  sleep 2
}

dual_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks10ep_codefbfacd7_fixedbarrierv18_stratified_step5_dual_world.json"
primary_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks10ep_codefbfacd7_fixedbarrierv18_primaryonly_stratified_step5_primary_world.json"
causal_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks10ep_codefbfacd7_fixedbarrierv18_worldtoegoablated_stratified_step5_dual_worldablated.json"
dual_progress="$root/singleview_worldflow/libero10_500ep/eval_4gpu_10ep/libero10_worldflow_endpoint_residual_${label}_step${step6}_all10tasks10ep_4x4090_total30_b30_codefbfacd7_fixedbarrierv18_stratified_step5_dual_world"

if [[ ! -s "$dual_artifact" ]]; then
  run_arm_background checkpoint 0 stratified_step5_dual_world "v52childenvfix_step${step6}_dual_world_stratified_step5"
  rejected=0
  while kill -0 "$arm_pid" 2>/dev/null; do
    sleep 10
    read -r completed successes failures < <(read_progress "$dual_progress")
    if (( failures >= 5 )); then rejected=1; stop_arm; break; fi
  done
  if (( ! rejected )); then wait "$arm_pid"; fi
  if (( rejected )); then
    "$python" - "$aggregate" "$dual_progress" "$step" <<'PY'
import glob,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); progress=pathlib.Path(sys.argv[2]); step=int(sys.argv[3]); rows=[]
for path in glob.glob(str(progress/"parts/gpu_*/progress.json")):
    try: rows.append(json.load(open(path,encoding="utf-8")))
    except Exception: pass
c=sum(int(r.get("completed_episode_count",0)) for r in rows)
s=sum(int(r.get("success_count",0)) for r in rows)
f=sum(int(r.get("failure_count",0)) for r in rows)
maximum=s+100-c
p={
  "status":"mathematical_early_stop_below_broad_baseline",
  "step":step,"episode_count":100,"completed_episode_count":c,
  "dual_world_successes":s,"dual_world_failure_count":f,
  "maximum_possible_dual_world_successes":maximum,
  "episode_ids":[0,5,10,15,20,25,30,35,40,45],
  "protocol":"fixed-barrier-v18; four RTX4090; total30; inference batch30; policy-noise seed0",
  "broad_baseline_successes":95,"v32_worldflow_successes_on_same_ids":96,
  "primary_world_successes":None,"dual_world_ablated_successes":None,
  "delta_over_broad_baseline":None,"delta_over_v32_worldflow":None,
  "multiview_causal_delta_successes":None,"worldflow_causal_delta_successes":None,
  "passes_absolute_screen":False,"passes_multiview_causal_screen":False,
  "passes_worldflow_causal_screen":False,"passes_joint_screen":False,
  "same_checkpoint":True,"dual_world_artifact":None,"primary_world_artifact":None,
  "dual_world_ablated_artifact":None,"partial_progress_root":str(progress),
  "stopped_without_deleting_outputs":True,
  "next_action":"do not run causal screen arms or Full500 for this checkpoint",
}
assert f>=5 and maximum<=95, p
out.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")
print(json.dumps(p,indent=2))
PY
    exit 0
  fi
fi
dual_successes=$(jq -r '.success_count' "$dual_artifact")

primary_successes=null
causal_successes=null
if (( dual_successes > 95 )); then
  if [[ ! -s "$primary_artifact" ]]; then
    run_arm primary_only 0 stratified_step5_primary_world "v52childenvfix_step${step6}_primary_world_stratified_step5"
  fi
  primary_successes=$(jq -r '.success_count' "$primary_artifact")
fi
if [[ "$primary_successes" != null ]] && (( dual_successes > primary_successes )); then
  if [[ ! -s "$causal_artifact" ]]; then
    run_arm checkpoint 1 stratified_step5_dual_worldablated "v52childenvfix_step${step6}_dual_worldablated_stratified_step5"
  fi
  causal_successes=$(jq -r '.success_count' "$causal_artifact")
fi

"$python" - "$aggregate" "$dual_artifact" "$primary_artifact" "$causal_artifact" \
  "$dual_successes" "$primary_successes" "$causal_successes" "$step" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
d=int(sys.argv[5])
p=None if sys.argv[6]=="null" else int(sys.argv[6])
a=None if sys.argv[7]=="null" else int(sys.argv[7])
step=int(sys.argv[8])
absolute=d>95
multiview=p is not None and d>p
worldflow=a is not None and d>a
passed=absolute and multiview and worldflow
payload={
  "status":"candidate" if passed else "screened_out",
  "step":step,
  "episode_count":100,
  "episode_ids":[0,5,10,15,20,25,30,35,40,45],
  "protocol":"fixed-barrier-v18; four RTX4090; total30; inference batch30; policy-noise seed0",
  "broad_baseline_successes":95,
  "v32_worldflow_successes_on_same_ids":96,
  "dual_world_successes":d,
  "primary_world_successes":p,
  "dual_world_ablated_successes":a,
  "delta_over_broad_baseline":d-95,
  "delta_over_v32_worldflow":d-96,
  "multiview_causal_delta_successes":None if p is None else d-p,
  "worldflow_causal_delta_successes":None if a is None else d-a,
  "passes_absolute_screen":absolute,
  "passes_multiview_causal_screen":multiview,
  "passes_worldflow_causal_screen":worldflow,
  "passes_joint_screen":passed,
  "same_checkpoint":True,
  "dual_world_artifact":sys.argv[2],
  "primary_world_artifact":sys.argv[3] if p is not None else None,
  "dual_world_ablated_artifact":sys.argv[4] if a is not None else None,
  "next_action":"run Full500 2x2 causal gate" if passed else "do not start Full500 for V52",
}
out.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
