#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
POLL_SECONDS="${MOLMO_MONITOR_INTERVAL_SECONDS:-600}"
ORCHESTRATOR_ROOT="${MOLMO_ORCHESTRATOR_ROOT:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_stage3_and_eval_control}"
ORCHESTRATOR_LOG="${ORCHESTRATOR_ROOT}/monitor.log"
LOCK_FILE="${ORCHESTRATOR_ROOT}/orchestrator.lock"

STAGE2_OUTPUT="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_from036000_lr1e4_to3e5_30k_20260819T101948Z"
STAGE2_LOG="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_continue30k_launch_20260819T101948Z/formal_train.log"
CHECKPOINT_036000="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_seed1000_8x5090_58c5f91/checkpoints/036000/pretrained_model"
CHECKPOINT_066000="${STAGE2_OUTPUT}/checkpoints/030000/pretrained_model"

STAGE3_CONTROL="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_stage3_from066000_control"
STAGE3_OUTPUT="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_stage3_from066000_lr1e4_to3e5_30k"
CHECKPOINT_096000="${STAGE3_OUTPUT}/checkpoints/030000/pretrained_model"
CONTINUATION_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/continue_molmo2er_pointonly_3b_30k_8gpu.sh"

EVAL_ROOT="${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_libero_eval_three_checkpoints_v043_protocol"
EVAL_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/eval_molmo2er_pointonly_3b_three_checkpoints.sh"

mkdir -p "${ORCHESTRATOR_ROOT}"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  echo "Another stage3/evaluation orchestrator holds ${LOCK_FILE}." >&2
  exit 73
fi
printf '%s\n' "$$" > "${ORCHESTRATOR_ROOT}/orchestrator.pid"

log() {
  printf '[orchestrator] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${ORCHESTRATOR_LOG}"
}

count_loss_records() {
  local log_path="$1"
  if [[ -f "${log_path}" ]]; then
    grep -c 'loss_action:' "${log_path}" || true
  else
    printf '0\n'
  fi
}

checkpoint_is_complete() {
  local checkpoint="$1"
  local expected_step="$2"
  local checkpoint_root="${checkpoint%/pretrained_model}"
  [[ -f "${checkpoint}/model.safetensors" ]] && \
    [[ -f "${checkpoint}/config.json" ]] && \
    [[ -f "${checkpoint}/policy_preprocessor.json" ]] && \
    [[ -f "${checkpoint}/policy_postprocessor.json" ]] && \
    [[ "$(jq -r '.step // empty' "${checkpoint_root}/training_state/training_step.json" 2>/dev/null || true)" == "${expected_step}" ]]
}

stage2_is_running() {
  pgrep -f "train_song_benchmark.py.*--output_dir=${STAGE2_OUTPUT}" >/dev/null 2>&1
}

log "waiting for stage2 30k training; poll_interval=${POLL_SECONDS}s"
while stage2_is_running; do
  step_count="$(count_loss_records "${STAGE2_LOG}")"
  error_count="$(grep -Eic 'Traceback|CUDA out of memory|NCCL.*(error|timeout)|non.?finite loss' "${STAGE2_LOG}" 2>/dev/null || true)"
  log "stage2 running step_records=${step_count}/30000 error_matches=${error_count}"
  sleep "${POLL_SECONDS}"
done

stage2_records="$(count_loss_records "${STAGE2_LOG}")"
if [[ "${stage2_records}" != "30000" ]] || ! checkpoint_is_complete "${CHECKPOINT_066000}" 30000; then
  log "stage2 failed completion audit: records=${stage2_records}, checkpoint=${CHECKPOINT_066000}"
  exit 2
fi
log "stage2 complete and checkpoint_066000 audited"

if checkpoint_is_complete "${CHECKPOINT_096000}" 30000; then
  log "stage3 checkpoint already complete; skipping training"
else
  if [[ -e "${STAGE3_OUTPUT}" || -e "${STAGE3_CONTROL}" ]]; then
    log "refusing ambiguous partial stage3 output; inspect ${STAGE3_OUTPUT} and ${STAGE3_CONTROL}"
    exit 2
  fi
  log "starting stage3 smoke and 30k training from checkpoint_066000"
  env \
    MOLMO_CONTINUATION_SOURCE_CHECKPOINT="${CHECKPOINT_066000}" \
    MOLMO_CONTINUATION_SOURCE_LABEL=from066000_lr1e4_to3e5 \
    MOLMO_CONTINUATION_CONTROL_DIR="${STAGE3_CONTROL}" \
    MOLMO_OUTPUT_DIR="${STAGE3_OUTPUT}" \
    MOLMO_MAIN_PROCESS_PORT=29643 \
    bash "${CONTINUATION_SCRIPT}" > "${ORCHESTRATOR_ROOT}/stage3_driver.log" 2>&1 &
  stage3_pid=$!
  printf '%s\n' "${stage3_pid}" > "${ORCHESTRATOR_ROOT}/stage3_driver.pid"
  while kill -0 "${stage3_pid}" 2>/dev/null; do
    stage3_records="$(count_loss_records "${STAGE3_CONTROL}/formal_train.log")"
    error_count="$(grep -Eic 'Traceback|CUDA out of memory|NCCL.*(error|timeout)|non.?finite loss' "${STAGE3_CONTROL}/formal_train.log" 2>/dev/null || true)"
    log "stage3 active step_records=${stage3_records}/30000 error_matches=${error_count}"
    sleep "${POLL_SECONDS}"
  done
  set +e
  wait "${stage3_pid}"
  stage3_rc=$?
  set -e
  if [[ "${stage3_rc}" != "0" ]]; then
    log "stage3 driver failed rc=${stage3_rc}; inspect ${ORCHESTRATOR_ROOT}/stage3_driver.log"
    exit "${stage3_rc}"
  fi
fi

stage3_records="$(count_loss_records "${STAGE3_CONTROL}/formal_train.log")"
if [[ "${stage3_records}" != "30000" ]] || ! checkpoint_is_complete "${CHECKPOINT_096000}" 30000; then
  log "stage3 failed completion audit: records=${stage3_records}, checkpoint=${CHECKPOINT_096000}"
  exit 2
fi
log "stage3 complete and checkpoint_096000 audited"

if [[ -f "${EVAL_ROOT}/comparison.json" ]]; then
  log "three-checkpoint evaluation summary already exists; skipping evaluation"
else
  log "starting sequential three-checkpoint LIBERO evaluation"
  env \
    MOLMO_THREE_CHECKPOINT_EVAL_ROOT="${EVAL_ROOT}" \
    MOLMO_CHECKPOINT_036000="${CHECKPOINT_036000}" \
    MOLMO_CHECKPOINT_066000="${CHECKPOINT_066000}" \
    MOLMO_CHECKPOINT_096000="${CHECKPOINT_096000}" \
    bash "${EVAL_SCRIPT}" > "${ORCHESTRATOR_ROOT}/evaluation_driver.log" 2>&1 &
  eval_pid=$!
  printf '%s\n' "${eval_pid}" > "${ORCHESTRATOR_ROOT}/evaluation_driver.pid"
  while kill -0 "${eval_pid}" 2>/dev/null; do
    completed=0
    successes=0
    while IFS= read -r progress_path; do
      current_completed="$(jq -r '.completed_episode_count // 0' "${progress_path}" 2>/dev/null || printf '0')"
      current_successes="$(jq -r '.success_count // 0' "${progress_path}" 2>/dev/null || printf '0')"
      completed=$((completed + current_completed))
      successes=$((successes + current_successes))
    done < <(find "${EVAL_ROOT}" -mindepth 2 -maxdepth 2 -name progress.json -type f 2>/dev/null | sort)
    log "evaluation active completed_episodes=${completed}/6000 successes_so_far=${successes}"
    sleep "${POLL_SECONDS}"
  done
  set +e
  wait "${eval_pid}"
  eval_rc=$?
  set -e
  if [[ "${eval_rc}" != "0" ]]; then
    log "evaluation driver failed rc=${eval_rc}; inspect ${ORCHESTRATOR_ROOT}/evaluation_driver.log"
    exit "${eval_rc}"
  fi
fi

if [[ ! -f "${EVAL_ROOT}/comparison.json" ]]; then
  log "evaluation finished without comparison.json"
  exit 2
fi
log "all work complete; summary=${EVAL_ROOT}/comparison.json"
