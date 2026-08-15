#!/usr/bin/env bash
set -euo pipefail

root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
run_root="$root/joint_multiview_worldflow/libero10_500ep"
train="$run_root/training/v52_from_v32step100_consensus_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps"
log="$run_root/logs/train_v52_from_v32step100_consensus_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps.log"
auditor="$root/scripts/audit_worldflow_training_convergence.py"
drift_auditor="$root/scripts/audit_v47_parameter_drift.py"
artifact_root="$run_root/artifacts"
python=${PYTHON_BIN:-/home/liusong/anaconda3/envs/reap/bin/python3.10}
source_policy="$root/singleview_worldflow/libero10_500ep/training/v32_from_v14_worldbnfixed_w2e_p75_anchor_residual_rate_coordframe_bodyframe_ego_tangent_common005_world02_residual4_4gpu_b32_w12_1080steps/checkpoints/000100/pretrained_model"
pipeline_session=wep_v043_v52_audit_then_paired_train
cache_session=wep_v043_v52_consensus_cache_audit
steps=(260 520 780 1040 1300 1564)
boundaries=260,520,780,1040,1300,1564

test -s "$auditor"
test -s "$drift_auditor"
test -s "$source_policy/model.safetensors"
test -x "$python"
mkdir -p "$artifact_root"

for step in "${steps[@]}"; do
  printf -v step6 '%06d' "$step"
  checkpoint="$train/checkpoints/$step6"
  while [[ ! -s "$checkpoint/pretrained_model/model.safetensors" ]]; do
    if ! tmux has-session -t "=$pipeline_session" 2>/dev/null \
        && ! tmux has-session -t "=$cache_session" 2>/dev/null; then
      echo "V52 pipeline exited before complete step${step6}" >&2
      exit 4
    fi
    sleep 30
  done
  test -s "$log"
  "$python" "$auditor" \
    --log "$log" \
    --output "$artifact_root/v52_training_convergence_through_step${step6}.json" \
    --expected-step "$step" \
    --maximum-step "$step" \
    --require-complete \
    --boundaries "$boundaries" \
    --tail-window 200 \
    --long-tail-window 782
  if [[ "$step" == 260 || "$step" == 1564 ]]; then
    "$python" "$drift_auditor" \
      --source "$source_policy" \
      --target "$checkpoint/pretrained_model" \
      --output "$artifact_root/v52_parameter_drift_source_v32step100_to_step${step6}.json" \
      --expected-step "$step" \
      --require-all-groups-changed \
      --require-architecture-roles-changed \
      --require-optimizer-state \
      --symmetric-world-point-path \
      --expected-initial-lrs 5e-9,5e-8,5e-9,5e-9
  fi
done

cp "$artifact_root/v52_training_convergence_through_step001564.json" \
  "$artifact_root/v52_training_convergence_complete.json"
echo "V52 convergence checkpoint monitor complete"
