#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v51}
python=${PYTHON_BIN:-/home/liusong/anaconda3/envs/reap/bin/python3.10}
runner="$root/scripts/eval_libero10_v12_one_checkpoint_4gpu_alltasks10ep_b30_sharedegl0.sh"
train="$root/joint_multiview_worldflow/libero10_500ep/training/v51r1_from_v32step100_multiscale_novelty1cm4cm_paired_symmetricpoint5e8_policy5e9_devicebound_schemafix_4gpu_b44_w12_1564steps"
artifact_root="$root/singleview_worldflow/libero10_500ep/artifacts"
joint_artifact_root="$root/joint_multiview_worldflow/libero10_500ep/artifacts"
step=${V51R1_STEP:-1564}
[[ "$step" =~ ^(260|520|780|1040|1300|1564)$ ]] || { echo "invalid V51R1_STEP=$step" >&2; exit 2; }
printf -v step6 '%06d' "$step"
checkpoint="$train/checkpoints/$step6/pretrained_model"
screen_session=wep_v043_v51r1egl0shared_all_checkpoints_stratified_3arm
screen=${V51R1_SCREEN_ARTIFACT:-"$joint_artifact_root/v51r1egl0shared_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json"}
gate=${V51R1_STEP_GATE_ARTIFACT:-"$joint_artifact_root/v51r1egl0shared_step${step6}_full500_2x2_multiview_worldflow_causal_gate.json"}
baseline_artifact="$artifact_root/taskbalanced_baseline_v042_fixed_action_step030000_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_canonical_fixed_action_v34protocol.json"
v32_gate="$artifact_root/taskbalanced_v32_residual_rate_coordframe_bodyframe_ego_tangent_p75_local4_fixedbarrier_full500_matched_worldflow_gate_step000100_baseline472_recheck.json"
label=v51r1egl0shared_v32_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_schemafix_b44
baseline_successes=472
v32_worldflow_successes=475
required_dual_successes=475
normal_failure_stop=$(( 500 - required_dual_successes + 1 ))

test -x "$runner"
test -x "$python"
test -s "$baseline_artifact"
test -s "$v32_gate"
mkdir -p "$joint_artifact_root"

if [[ -s "$gate" ]]; then
  "$python" - "$gate" <<'PY'
import json,sys
print(json.dumps(json.load(open(sys.argv[1],encoding="utf-8")),indent=2))
PY
  exit 0
fi

while [[ ! -s "$screen" ]]; do
  if ! tmux has-session -t "=$screen_session" 2>/dev/null; then
    echo "V51R1 all-checkpoint screen exited without aggregate: $screen" >&2
    exit 4
  fi
  sleep 20
done
if ! "$python" - "$screen" "$step" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); step=int(sys.argv[2])
row=next((r for r in p.get("rows",[]) if int(r["step"])==step),None)
assert row is not None
raise SystemExit(0 if row.get("passes_joint_screen") is True else 1)
PY
then
  "$python" - "$gate" "$screen" "$step" <<'PY'
import json,pathlib,sys
out,screen=map(pathlib.Path,sys.argv[1:3]); step=int(sys.argv[3])
p={"status":"not_started","step":step,"reason":"V51R1 stratified three-arm screen did not pass for this checkpoint","screen_artifact":str(screen),"full500_arms_started":[]}
out.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")
print(json.dumps(p,indent=2))
PY
  exit 0
fi

test -s "$checkpoint/model.safetensors"
"$python" - "$checkpoint" "$baseline_artifact" "$v32_gate" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); c=json.load((p/"config.json").open())
b=json.load(open(sys.argv[2])); v=json.load(open(sys.argv[3]))
assert c["camera_view_fusion"]=="multiscale_novelty_union"
assert c["camera_view_fps_target_points"]==10_000
assert c["camera_view_fps_gripper_points"]==500
assert c["multiview_input_view_dropout_paired_coverage"] is True
assert c["multiview_input_symmetric_point_path_adaptation"] is True
assert c["worldflow_enable"] is True
assert c["worldflow_action_fusion"]=="endpoint_residual_boosting"
assert c["worldflow_noise_coupling"]=="projected_ego_path"
assert c["worldflow_ego_residual_gate_init"] is None
assert c["se3_enable"] is False
assert all(c[k]==0 for k in ("worldflow_loss_weight","worldflow_geo_loss_weight","worldflow_bridge_loss_weight","worldflow_equiv_loss_weight"))
assert b["success_count"]==472 and b["episode_count"]==500
assert v["passes_joint_gate"] is True and v["normal_successes"]==475 and v["world_to_ego_causal_ablation_successes"]==469
print("[V51R1 Full500 2x2 preflight] PASS")
PY

common_env=(
  LEROBOT_REPO="$repo"
  EXPERIMENT_ROOT="$root"
  EVAL_TRAIN_DIR="$train"
  EVAL_CHECKPOINT_OVERRIDE="$checkpoint"
  EVAL_EPISODES_PER_TASK=50
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

base_output="$root/singleview_worldflow/libero10_500ep/eval_4gpu_50ep/libero10_worldflow_endpoint_residual_${label}_step${step6}_all10tasks50ep_4x4090_total30_b30_codefbfacd7_fixedbarrierv18"
dual_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_full500_dual_world.json"
primary_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_primaryonly_full500_primary_world.json"
dual_ablated_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_worldtoegoablated_full500_dual_worldablated.json"
primary_ablated_artifact="$artifact_root/taskbalanced_${label}_step${step6}_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_primaryonly_worldtoegoablated_full500_primary_worldablated.json"

dual_progress="${base_output}_full500_dual_world"
if [[ ! -s "$dual_artifact" ]]; then
  run_arm_background checkpoint 0 full500_dual_world "v51r1egl0shared_step${step6}_full500_dual_world"
  rejected=0
  while kill -0 "$arm_pid" 2>/dev/null; do
    sleep 10
    read -r completed successes failures < <(read_progress "$dual_progress")
    if (( failures >= normal_failure_stop )); then rejected=1; stop_arm; break; fi
  done
  if (( ! rejected )); then wait "$arm_pid"; fi
  if (( rejected )); then
    "$python" - "$gate" "$screen" "$dual_progress" "$baseline_successes" "$v32_worldflow_successes" "$required_dual_successes" "$step" <<'PY'
import glob,json,pathlib,sys
out,screen,progress=map(pathlib.Path,sys.argv[1:4]); b,v,required=map(int,sys.argv[4:7])
step=int(sys.argv[7])
rows=[]
for path in glob.glob(str(progress/"parts/gpu_*/progress.json")):
    try: rows.append(json.load(open(path,encoding="utf-8")))
    except Exception: pass
c=sum(int(r.get("completed_episode_count",0)) for r in rows); s=sum(int(r.get("success_count",0)) for r in rows); f=sum(int(r.get("failure_count",0)) for r in rows)
p={"status":"mathematical_early_stop_dual_below_v32_nondegradation","step":step,"episode_count_target":500,"completed_episode_count":c,"success_count":s,"failure_count":f,"remaining_episode_count":500-c,"maximum_possible_final_successes":s+500-c,"baseline_successes":b,"v32_worldflow_successes":v,"required_dual_successes":required,"screen_artifact":str(screen),"partial_progress_root":str(progress),"full500_arms_started":["dual_world"],"stopped_without_deleting_outputs":True}
assert p["maximum_possible_final_successes"]<required
out.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8"); print(json.dumps(p,indent=2))
PY
    exit 0
  fi
fi
dual_successes=$(jq -r '.success_count' "$dual_artifact")
if (( dual_successes < required_dual_successes )); then
  echo "completed dual arm unexpectedly failed the pre-registered non-degradation threshold" >&2
  exit 5
fi

primary_progress="${base_output}_primaryonly_full500_primary_world"
if [[ ! -s "$primary_artifact" ]]; then
  run_arm_background primary_only 0 full500_primary_world "v51r1egl0shared_step${step6}_full500_primary_world"
  rejected=0
  while kill -0 "$arm_pid" 2>/dev/null; do
    sleep 10
    read -r completed successes failures < <(read_progress "$primary_progress")
    if (( successes >= dual_successes )); then rejected=1; stop_arm; break; fi
  done
  if (( ! rejected )); then wait "$arm_pid"; fi
  if (( rejected )); then
    "$python" - "$gate" "$screen" "$dual_artifact" "$primary_progress" "$dual_successes" "$step" <<'PY'
import glob,json,pathlib,sys
out,screen,dual,progress=map(pathlib.Path,sys.argv[1:5]); d=int(sys.argv[5]); step=int(sys.argv[6]); rows=[]
for path in glob.glob(str(progress/"parts/gpu_*/progress.json")):
    try: rows.append(json.load(open(path,encoding="utf-8")))
    except Exception: pass
c=sum(int(r.get("completed_episode_count",0)) for r in rows); s=sum(int(r.get("success_count",0)) for r in rows)
p={"status":"mathematical_early_stop_no_positive_multiview_delta","step":step,"dual_world_successes":d,"primary_world_partial_successes":s,"primary_completed_episode_count":c,"screen_artifact":str(screen),"dual_world_artifact":str(dual),"primary_partial_progress_root":str(progress),"full500_arms_started":["dual_world","primary_world"],"stopped_without_deleting_outputs":True}
assert s>=d
out.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8"); print(json.dumps(p,indent=2))
PY
    exit 0
  fi
fi
primary_successes=$(jq -r '.success_count' "$primary_artifact")
if (( dual_successes <= primary_successes )); then
  echo "completed primary arm unexpectedly failed the multiview-positive threshold" >&2
  exit 6
fi

dual_ablated_progress="${base_output}_worldtoegoablated_full500_dual_worldablated"
if [[ ! -s "$dual_ablated_artifact" ]]; then
  run_arm_background checkpoint 1 full500_dual_worldablated "v51r1egl0shared_step${step6}_full500_dual_worldablated"
  rejected=0
  while kill -0 "$arm_pid" 2>/dev/null; do
    sleep 10
    read -r completed successes failures < <(read_progress "$dual_ablated_progress")
    if (( successes >= dual_successes )); then rejected=1; stop_arm; break; fi
  done
  if (( ! rejected )); then wait "$arm_pid"; fi
  if (( rejected )); then
    "$python" - "$gate" "$screen" "$dual_artifact" "$primary_artifact" "$dual_ablated_progress" "$dual_successes" "$primary_successes" "$step" <<'PY'
import glob,json,pathlib,sys
out,screen,dual,primary,progress=map(pathlib.Path,sys.argv[1:6]); d,p=map(int,sys.argv[6:8]); step=int(sys.argv[8]); rows=[]
for path in glob.glob(str(progress/"parts/gpu_*/progress.json")):
    try: rows.append(json.load(open(path,encoding="utf-8")))
    except Exception: pass
c=sum(int(r.get("completed_episode_count",0)) for r in rows); s=sum(int(r.get("success_count",0)) for r in rows)
x={"status":"mathematical_early_stop_no_positive_worldflow_delta","step":step,"dual_world_successes":d,"primary_world_successes":p,"dual_world_ablated_partial_successes":s,"dual_world_ablated_completed_episode_count":c,"screen_artifact":str(screen),"dual_world_artifact":str(dual),"primary_world_artifact":str(primary),"dual_world_ablated_partial_progress_root":str(progress),"full500_arms_started":["dual_world","primary_world","dual_world_ablated"],"stopped_without_deleting_outputs":True}
assert s>=d
out.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8"); print(json.dumps(x,indent=2))
PY
    exit 0
  fi
fi
dual_ablated_successes=$(jq -r '.success_count' "$dual_ablated_artifact")
if (( dual_successes <= dual_ablated_successes )); then
  echo "completed causal arm unexpectedly failed the World-positive threshold" >&2
  exit 7
fi

if [[ ! -s "$primary_ablated_artifact" ]]; then
  run_arm_background primary_only 1 full500_primary_worldablated "v51r1egl0shared_step${step6}_full500_primary_worldablated"
  wait "$arm_pid"
fi
primary_ablated_successes=$(jq -r '.success_count' "$primary_ablated_artifact")

"$python" - "$gate" "$screen" "$dual_artifact" "$primary_artifact" "$dual_ablated_artifact" "$primary_ablated_artifact" \
  "$dual_successes" "$primary_successes" "$dual_ablated_successes" "$primary_ablated_successes" "$baseline_successes" "$v32_worldflow_successes" "$step" <<'PY'
import json,pathlib,sys
out,screen,dual,primary,dual_a,primary_a=map(pathlib.Path,sys.argv[1:7])
d,p,da,pa,b,v=map(int,sys.argv[7:13])
step=int(sys.argv[13])
passed=d>b and d>=v and d>p and d>da
x={
 "status":"passed" if passed else "failed_joint_gate",
 "step":step,
 "episode_count":500,
 "dual_world_successes":d,
 "primary_world_successes":p,
 "dual_world_ablated_successes":da,
 "primary_world_ablated_successes":pa,
 "baseline_successes":b,
 "v32_worldflow_successes":v,
 "delta_over_baseline":d-b,
 "delta_over_v32_worldflow":d-v,
 "multiview_causal_delta_successes":d-p,
 "worldflow_causal_delta_successes":d-da,
 "worldflow_delta_with_primary_input":p-pa,
 "multiview_delta_with_world_ablated":da-pa,
 "causal_interaction_successes":d-p-da+pa,
 "same_checkpoint":True,
 "same_fixed_barrier_protocol":True,
 "same_policy_noise_seed":True,
 "passes_absolute_baseline_gate":d>b,
 "passes_v32_nondegradation_gate":d>=v,
 "passes_multiview_causal_gate":d>p,
 "passes_worldflow_causal_gate":d>da,
 "passes_joint_gate":passed,
 "screen_artifact":str(screen),
 "dual_world_artifact":str(dual),
 "primary_world_artifact":str(primary),
 "dual_world_ablated_artifact":str(dual_a),
 "primary_world_ablated_artifact":str(primary_a),
 "full500_arms_started":["dual_world","primary_world","dual_world_ablated","primary_world_ablated"],
}
out.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8"); print(json.dumps(x,indent=2))
PY
