#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the persistent experiment directory}

install_if_missing_or_identical() {
  local source=$1
  local target=$2
  local mode=$3
  mkdir -p "$(dirname "$target")"
  if [[ -e "$target" ]]; then
    if cmp -s "$source" "$target"; then
      echo "already identical: $target"
      return 0
    fi
    echo "refusing to overwrite different existing file: $target" >&2
    return 2
  fi
  install -m "$mode" "$source" "$target"
  echo "installed: $target"
}

for name in \
  run_v51_multiscale_novelty_1cm4cm_cache_4gpu.sh \
  run_v51_cache_exact_index_audit.sh \
  launch_v51_cache_and_exact_audit_4gpu_tmux.sh \
  launch_v51_multiscale_novelty_1cm4cm_cache_4gpu_tmux.sh \
  run_v51_from_v32step100_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps.sh \
  launch_v51_paired_symmetricpoint_training_4gpu_tmux.sh \
  run_v51r1_from_v32step100_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_schemafix_4gpu_b44_w12_1564steps.sh \
  launch_v51r1_schemafix_paired_symmetricpoint_training_4gpu_tmux.sh \
  wait_then_screen_v51r1_checkpoint_stratified_3arm.sh \
  wait_then_screen_v51r1_all_checkpoints_stratified_3arm.sh \
  launch_wait_then_screen_v51r1_all_checkpoints_stratified_3arm_tmux.sh \
  wait_then_run_v51r1_checkpoint_full500_2x2_causal_gate.sh \
  wait_then_run_v51r1_all_candidates_full500_2x2_causal_gate.sh \
  launch_wait_then_run_v51r1_all_candidates_full500_2x2_causal_gate_tmux.sh \
  audit_v46_cache_exact_indices.py \
  audit_v49_real_checkpoint_optimizer_preflight.py \
  eval_libero10_v12_one_checkpoint_4gpu_alltasks10ep_b30_sharedegl0.sh \
  eval_libero10_v12_one_checkpoint_4gpu_alltasks10ep_b30_globalvisibleegl.sh \
  eval_libero10_v12_one_checkpoint_4gpu_alltasks10ep_b30_localvisibleegl.sh \
  eval_libero10_v12_one_checkpoint_4gpu_alltasks10ep_b30.sh; do
  install_if_missing_or_identical "$bundle_dir/scripts/$name" "$root/scripts/$name" 0755
done

artifact_dir=$root/joint_multiview_worldflow/libero10_500ep/artifacts
for name in \
  v51_preregistered_training_and_2x2_causal_gate_protocol.json \
  v51r1_shared_egl0_runtime_contract.json \
  v51r1_egl_global_visibility_correction.json \
  v51r1_egl_visible_device_mapping_correction.json \
  v51_real_checkpoint_optimizer_and_gradient_role_preflight_v2.json; do
  install_if_missing_or_identical "$bundle_dir/artifacts/$name" "$artifact_dir/$name" 0644
done

install_if_missing_or_identical \
  "$bundle_dir/CURRENT_CONVERSATION_ARCHIVE_2026-08-15.md" \
  "$root/CURRENT_CONVERSATION_ARCHIVE_2026-08-15_V51R1.md" 0644

echo "V51 runtime bundle is present under: $root"
