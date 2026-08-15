#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v52_consensus_multiscale}
python=${PYTHON_BIN:-/home/liusong/anaconda3/envs/reap/bin/python3.10}
workflow_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
worker="$workflow_root/scripts/wait_then_screen_v52_checkpoint_stratified_3arm.sh"
artifact_root="$root/joint_multiview_worldflow/libero10_500ep/artifacts"
aggregate="$artifact_root/v52childenvfix_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json"
steps=(260 520 780 1040 1300 1564)

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

# Phase 1: finish the dual+World 100-episode initial screen for every checkpoint
# before starting any new causal-control arm.
step_artifacts=()
for step in "${steps[@]}"; do
  printf -v step6 '%06d' "$step"
  step_artifact="$artifact_root/v52childenvfix_step${step6}_stratified_3arm_multiview_worldflow_screen.json"
  step_artifacts+=("$step_artifact")
  env LEROBOT_REPO="$repo" EXPERIMENT_ROOT="$root" \
    V52_STEP="$step" V52_STEP_SCREEN_ARTIFACT="$step_artifact" V52_SCREEN_PHASE=dual_only \
    bash "$worker"
done

# Phase 2: only checkpoints above 95/100 proceed to primary+World and then,
# when still eligible, dual+World->Ego-disabled causal screening.
for index in "${!steps[@]}"; do
  step=${steps[$index]}
  step_artifact=${step_artifacts[$index]}
  env LEROBOT_REPO="$repo" EXPERIMENT_ROOT="$root" \
    V52_STEP="$step" V52_STEP_SCREEN_ARTIFACT="$step_artifact" V52_SCREEN_PHASE=causal_only \
    bash "$worker"
done

"$python" - "$aggregate" "${step_artifacts[@]}" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
rows=[]
for value in sys.argv[2:]:
    path=pathlib.Path(value)
    row=json.load(path.open(encoding="utf-8"))
    assert int(row["step"]) in {260,520,780,1040,1300,1564}
    row["step_screen_artifact"]=str(path)
    rows.append(row)
assert sorted(int(row["step"]) for row in rows)==[260,520,780,1040,1300,1564]
candidates=[row for row in rows if row.get("passes_joint_screen") is True]
candidates.sort(key=lambda row:(
    -int(row["dual_world_successes"]),
    -min(int(row["multiview_causal_delta_successes"]),int(row["worldflow_causal_delta_successes"])),
    int(row["step"]),
))
payload={
    "status":"candidate" if candidates else "screened_out",
    "training_completed_through_step":1564,
    "screened_steps":[260,520,780,1040,1300,1564],
    "episode_count_per_arm":100,
    "protocol":"fixed-barrier-v18; four RTX4090; total30; inference batch30; policy-noise seed0; stratified episode IDs 0,5,...45",
    "broad_baseline_successes":95,
    "v32_worldflow_successes_on_same_ids":96,
    "candidate_steps_ranked":[int(row["step"]) for row in candidates],
    "candidate_count":len(candidates),
    "rows":rows,
    "next_action":"run ranked Full500 2x2 causal gates" if candidates else "archive V52 as screened out",
}
temporary=out.with_suffix(out.suffix+".tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(out)
print(json.dumps(payload,indent=2))
PY
