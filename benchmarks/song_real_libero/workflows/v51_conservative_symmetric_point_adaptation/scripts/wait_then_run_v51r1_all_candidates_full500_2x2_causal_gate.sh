#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v51}
python=${PYTHON_BIN:-/home/liusong/anaconda3/envs/reap/bin/python3.10}
worker="$root/scripts/wait_then_run_v51r1_checkpoint_full500_2x2_causal_gate.sh"
artifact_root="$root/joint_multiview_worldflow/libero10_500ep/artifacts"
screen="$artifact_root/v51r1eglglobal_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json"
screen_session=wep_v043_v51r1eglglobal_all_checkpoints_stratified_3arm
aggregate="$artifact_root/v51r1eglglobal_all_candidates_full500_2x2_multiview_worldflow_causal_gate.json"

test -x "$worker"
test -x "$python"
mkdir -p "$artifact_root"

if [[ -s "$aggregate" ]]; then
  "$python" - "$aggregate" <<'PY'
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

mapfile -t candidate_steps < <("$python" - "$screen" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
for step in p.get("candidate_steps_ranked",[]):
    print(int(step))
PY
)

if (( ${#candidate_steps[@]} == 0 )); then
  "$python" - "$aggregate" "$screen" <<'PY'
import json,pathlib,sys
out,screen=map(pathlib.Path,sys.argv[1:3])
p={"status":"screened_out","screen_artifact":str(screen),"candidate_steps_ranked":[],"attempted_step_gates":[],"winning_step":None,"passes_joint_gate":False}
out.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")
print(json.dumps(p,indent=2))
PY
  exit 0
fi

step_gates=()
winner=
for step in "${candidate_steps[@]}"; do
  printf -v step6 '%06d' "$step"
  step_gate="$artifact_root/v51r1eglglobal_step${step6}_full500_2x2_multiview_worldflow_causal_gate.json"
  step_gates+=("$step_gate")
  env LEROBOT_REPO="$repo" EXPERIMENT_ROOT="$root" \
    V51R1_STEP="$step" V51R1_SCREEN_ARTIFACT="$screen" V51R1_STEP_GATE_ARTIFACT="$step_gate" \
    bash "$worker"
  status=$("$python" - "$step_gate" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])
PY
  )
  if [[ "$status" == passed ]]; then
    winner=$step
    break
  fi
done

"$python" - "$aggregate" "$screen" "${winner:-none}" "${step_gates[@]}" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); screen=pathlib.Path(sys.argv[2]); winner=None if sys.argv[3]=="none" else int(sys.argv[3])
gate_paths=[pathlib.Path(value) for value in sys.argv[4:]]
gates=[json.load(path.open(encoding="utf-8")) for path in gate_paths]
passed=winner is not None
payload={
    "status":"passed" if passed else "failed_all_ranked_candidates",
    "screen_artifact":str(screen),
    "candidate_steps_ranked":json.load(screen.open(encoding="utf-8"))["candidate_steps_ranked"],
    "attempted_step_gates":[str(path) for path in gate_paths],
    "attempted_results":gates,
    "winning_step":winner,
    "winning_step_gate":str(gate_paths[-1]) if passed else None,
    "passes_joint_gate":passed,
    "stops_after_first_full500_pass":True,
    "preserves_all_partial_outputs":True,
}
temporary=out.with_suffix(out.suffix+".tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(out)
print(json.dumps(payload,indent=2))
PY
