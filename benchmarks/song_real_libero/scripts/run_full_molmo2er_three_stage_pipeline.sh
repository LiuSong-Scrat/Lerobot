#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
PYTHON_BIN="${EXPERIMENT_ROOT}/.venv-smol5090/bin/python"
TRAIN_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/train_full_molmo2er_wep_vla_8gpu.sh"
EVAL_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/eval_full_molmo2er_wep_vla_two_checkpoints.sh"
RUN_ROOT="${FULL_MOLMO2ER_PIPELINE_ROOT:-${EXPERIMENT_ROOT}/outputs/full_molmo2er_wep_vla_three_stage}"
CONTROL_ROOT="${RUN_ROOT}/control"
PIPELINE_LOG="${CONTROL_ROOT}/pipeline_monitor.log"
LOCK_FILE="${FULL_MOLMO2ER_PIPELINE_LOCK_FILE:-/tmp/lerobot_full_molmo2er_wep_vla_three_stage.lock}"
POLL_SECONDS="${FULL_MOLMO2ER_MONITOR_INTERVAL_SECONDS:-600}"
CHECKPOINT_INTERVAL_STEPS="${FULL_MOLMO2ER_CHECKPOINT_INTERVAL_STEPS:-100}"
RECOVERY_CHECKPOINTS_TO_KEEP="${FULL_MOLMO2ER_RECOVERY_CHECKPOINTS_TO_KEEP:-1}"
TRAINING_MIN_GPU_FREE_MIB="${FULL_MOLMO2ER_TRAINING_MIN_GPU_FREE_MIB:-31500}"

STAGE1_OUTPUT="${RUN_ROOT}/stage1_fresh_036000"
STAGE2_OUTPUT="${RUN_ROOT}/stage2_from036000_to066000"
STAGE3_OUTPUT="${RUN_ROOT}/stage3_from066000_to096000"
CHECKPOINT_036000="${STAGE1_OUTPUT}/checkpoints/036000/pretrained_model"
CHECKPOINT_066000="${STAGE2_OUTPUT}/checkpoints/030000/pretrained_model"
CHECKPOINT_096000="${STAGE3_OUTPUT}/checkpoints/030000/pretrained_model"
EVAL_ROOT="${RUN_ROOT}/libero_eval_066000_096000_v043_protocol"

STAGE1_LOG="${CONTROL_ROOT}/stage1_fresh_036000.log"
STAGE2_LOG="${CONTROL_ROOT}/stage2_from036000_to066000.log"
STAGE3_LOG="${CONTROL_ROOT}/stage3_from066000_to096000.log"
EVAL_DRIVER_LOG="${CONTROL_ROOT}/libero_eval_066000_096000.log"
SMOKE_LOG="${CONTROL_ROOT}/fresh_two_step_smoke.log"
SMOKE_MARKER="${CONTROL_ROOT}/fresh_two_step_smoke.complete.json"
SMOKE_OUTPUT_DIR="/tmp/full_molmo2er_wep_vla_pipeline_smoke_${UID}_$$"
SMOKE_LAUNCH_RECORD_DIR="${SMOKE_OUTPUT_DIR}.launch"
COMPLETION_AUDIT="${RUN_ROOT}/completion_audit.json"

ACTIVE_PID=""

for command_name in jq nvidia-smi flock realpath find sed grep awk setsid; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done
if [[ ! -x "${PYTHON_BIN}" || ! -f "${TRAIN_SCRIPT}" || ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Missing Full-Molmo2-ER training or evaluation launcher." >&2
  exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 10 )); then
  echo "FULL_MOLMO2ER_MONITOR_INTERVAL_SECONDS must be an integer of at least 10." >&2
  exit 2
fi
if [[ ! "${CHECKPOINT_INTERVAL_STEPS}" =~ ^[0-9]+$ ]] || \
  (( CHECKPOINT_INTERVAL_STEPS < 1 )) || \
  (( 36000 % CHECKPOINT_INTERVAL_STEPS != 0 )) || \
  (( 30000 % CHECKPOINT_INTERVAL_STEPS != 0 )); then
  echo "FULL_MOLMO2ER_CHECKPOINT_INTERVAL_STEPS must positively divide both 36000 and 30000." >&2
  exit 2
fi
if [[ ! "${RECOVERY_CHECKPOINTS_TO_KEEP}" =~ ^[0-9]+$ ]] || \
  (( RECOVERY_CHECKPOINTS_TO_KEEP < 1 )); then
  echo "FULL_MOLMO2ER_RECOVERY_CHECKPOINTS_TO_KEEP must be at least 1." >&2
  exit 2
fi
if [[ ! "${TRAINING_MIN_GPU_FREE_MIB}" =~ ^[0-9]+$ ]] || \
  (( TRAINING_MIN_GPU_FREE_MIB < 1 )); then
  echo "FULL_MOLMO2ER_TRAINING_MIN_GPU_FREE_MIB must be a positive integer." >&2
  exit 2
fi

mkdir -p "${CONTROL_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Full-Molmo2-ER three-stage pipeline holds ${LOCK_FILE}." >&2
  exit 73
fi
printf '%s\n' "$$" > "${CONTROL_ROOT}/pipeline.pid"

log() {
  printf '[pipeline] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${PIPELINE_LOG}"
}

cleanup_smoke_output() {
  local cleanup_target
  for cleanup_target in "${SMOKE_OUTPUT_DIR}" "${SMOKE_LAUNCH_RECORD_DIR}"; do
    if [[ ! -e "${cleanup_target}" ]]; then
      continue
    fi
    case "${cleanup_target}" in
      /tmp/full_molmo2er_wep_vla_pipeline_smoke_*) ;;
      *)
        log "refusing unsafe smoke cleanup target: ${cleanup_target}"
        return 2
        ;;
    esac
    find "${cleanup_target}" -depth -delete
    log "deleted temporary smoke artifact with find -depth -delete: ${cleanup_target}"
  done
}

terminate_active_process_group() {
  local reason="$1"
  local wait_iteration
  local child_pid="${ACTIVE_PID}"
  [[ -n "${child_pid}" ]] || return 0
  if kill -0 -- "-${child_pid}" 2>/dev/null || kill -0 "${child_pid}" 2>/dev/null; then
    log "${reason}; terminating active child process group pgid=${child_pid}"
    kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true
    for ((wait_iteration = 1; wait_iteration <= 30; wait_iteration++)); do
      if ! kill -0 -- "-${child_pid}" 2>/dev/null && ! kill -0 "${child_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 -- "-${child_pid}" 2>/dev/null || kill -0 "${child_pid}" 2>/dev/null; then
      log "${reason}; escalating active child process group pgid=${child_pid} to SIGKILL"
      kill -KILL -- "-${child_pid}" 2>/dev/null || kill -KILL "${child_pid}" 2>/dev/null || true
    fi
    wait "${child_pid}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
}

handle_signal() {
  local signal_name="$1"
  terminate_active_process_group "received ${signal_name}"
  exit 130
}

handle_exit() {
  local pipeline_rc="$?"
  trap - EXIT
  terminate_active_process_group "pipeline exiting rc=${pipeline_rc}"
  cleanup_smoke_output || true
  exit "${pipeline_rc}"
}

trap handle_exit EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal HUP' HUP

gpu_snapshot() {
  local snapshot
  snapshot="$(
    nvidia-smi \
      --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>&1 \
      | awk -F',' '{gsub(/ /, "", $0); printf "%s%s", (NR == 1 ? "" : ";"), $0}'
  )"
  printf '%s\n' "${snapshot:-unavailable}"
}

gpu_capacity_ready() {
  local minimum_free_mib="$1"
  local gpu_index
  local -a free_mib
  mapfile -t free_mib < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ "${#free_mib[@]}" != "8" ]]; then
    return 2
  fi
  for gpu_index in "${!free_mib[@]}"; do
    if [[ ! "${free_mib[gpu_index]}" =~ ^[0-9]+$ ]] || \
      (( free_mib[gpu_index] < minimum_free_mib )); then
      return 1
    fi
  done
}

wait_for_training_capacity() {
  local phase="$1"
  while true; do
    if gpu_capacity_ready "${TRAINING_MIN_GPU_FREE_MIB}"; then
      log "${phase} GPU capacity available; confirming for 1 second; gpu(index,used_MiB,free_MiB,util_pct)=$(gpu_snapshot)"
      sleep 1
      if gpu_capacity_ready "${TRAINING_MIN_GPU_FREE_MIB}"; then
        return 0
      fi
    else
      capacity_rc="$?"
      if [[ "${capacity_rc}" == "2" ]]; then
        log "${phase} expected exactly 8 readable GPUs; gpu=$(gpu_snapshot)"
        exit 2
      fi
      log "${phase} waiting for >=${TRAINING_MIN_GPU_FREE_MIB} MiB free on every GPU; gpu(index,used_MiB,free_MiB,util_pct)=$(gpu_snapshot)"
    fi
    sleep "${POLL_SECONDS}"
  done
}

error_count() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    printf '0\n'
    return 0
  fi
  grep -Eic \
    'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|NCCL[^[:cntrl:]]*(error|timeout)|non.?finite loss|ChildFailedError|Segmentation fault|Killed$' \
    "${log_path}" 2>/dev/null || true
}

latest_training_fields() {
  local log_path="$1"
  local metric_line=""
  local action_line=""
  local step="0"
  local loss="NA"
  local loss_action="NA"
  local learning_rate="NA"
  if [[ -f "${log_path}" ]]; then
    metric_line="$(tr '\r' '\n' < "${log_path}" | grep -E 'step:[0-9]+ .*loss:' | tail -n 1 || true)"
    action_line="$(tr '\r' '\n' < "${log_path}" | grep 'loss_action:' | tail -n 1 || true)"
  fi
  if [[ -n "${metric_line}" ]]; then
    step="$(sed -n 's/.*step:\([0-9][0-9]*\).*/\1/p' <<< "${metric_line}")"
    loss="$(sed -n 's/.* loss:\([^[:space:]]*\).*/\1/p' <<< "${metric_line}")"
    learning_rate="$(sed -n 's/.* lr:\([^[:space:]]*\).*/\1/p' <<< "${metric_line}")"
  fi
  if [[ -n "${action_line}" ]]; then
    loss_action="$(sed -n 's/.*loss_action:\([^[:space:]]*\).*/\1/p' <<< "${action_line}")"
  fi
  printf 'step=%s loss=%s loss_action=%s lr=%s' \
    "${step:-0}" "${loss:-NA}" "${loss_action:-NA}" "${learning_rate:-NA}"
}

training_snapshot() {
  local phase="$1"
  local log_path="$2"
  log "${phase} pid=${ACTIVE_PID} $(latest_training_fields "${log_path}") errors=$(error_count "${log_path}") gpu(index,used_MiB,free_MiB,util_pct)=$(gpu_snapshot)"
}

evaluation_snapshot() {
  local completed=0
  local successes=0
  local progress_path
  local progress_completed
  local progress_successes
  while IFS= read -r progress_path; do
    progress_completed="$(jq -r '.completed_episode_count // 0' "${progress_path}" 2>/dev/null || printf '0')"
    progress_successes="$(jq -r '.success_count // 0' "${progress_path}" 2>/dev/null || printf '0')"
    [[ "${progress_completed}" =~ ^[0-9]+$ ]] || progress_completed=0
    [[ "${progress_successes}" =~ ^[0-9]+$ ]] || progress_successes=0
    completed=$((completed + progress_completed))
    successes=$((successes + progress_successes))
  done < <(
    find "${EVAL_ROOT}" -mindepth 4 -maxdepth 4 -type f -name progress.json 2>/dev/null | sort
  )
  log "evaluation pid=${ACTIVE_PID} completed_episodes=${completed}/4000 successes_so_far=${successes} errors=$(error_count "${EVAL_DRIVER_LOG}") gpu(index,used_MiB,free_MiB,util_pct)=$(gpu_snapshot)"
}

sleep_until_poll_or_exit() {
  local pid="$1"
  local remaining="${POLL_SECONDS}"
  local sleep_chunk
  while (( remaining > 0 )) && kill -0 "${pid}" 2>/dev/null; do
    if (( remaining > 10 )); then
      sleep_chunk=10
    else
      sleep_chunk="${remaining}"
    fi
    sleep "${sleep_chunk}"
    remaining=$((remaining - sleep_chunk))
  done
}

monitor_training_child() {
  local phase="$1"
  local log_path="$2"
  local output_dir="${3:-}"
  local expected_steps="${4:-}"
  local expected_decay_lr="${5:-}"
  local child_pid="${ACTIVE_PID}"
  while kill -0 "${child_pid}" 2>/dev/null; do
    training_snapshot "${phase}" "${log_path}"
    if [[ -n "${output_dir}" ]] && \
      ! prune_recovery_checkpoints "${output_dir}" "${expected_steps}" "${expected_decay_lr}"; then
      log "${phase} recovery-checkpoint retention audit failed; no unchecked path was removed"
    fi
    sleep_until_poll_or_exit "${child_pid}"
  done
  set +e
  wait "${child_pid}"
  local child_rc="$?"
  set -e
  training_snapshot "${phase} final" "${log_path}"
  if [[ -n "${output_dir}" ]] && \
    ! prune_recovery_checkpoints "${output_dir}" "${expected_steps}" "${expected_decay_lr}"; then
    log "${phase} final recovery-checkpoint retention audit failed"
  fi
  ACTIVE_PID=""
  return "${child_rc}"
}

monitor_evaluation_child() {
  local child_pid="${ACTIVE_PID}"
  while kill -0 "${child_pid}" 2>/dev/null; do
    evaluation_snapshot
    sleep_until_poll_or_exit "${child_pid}"
  done
  set +e
  wait "${child_pid}"
  local child_rc="$?"
  set -e
  evaluation_snapshot
  ACTIVE_PID=""
  return "${child_rc}"
}

safetensors_files_are_structurally_valid() {
  "${PYTHON_BIN}" - "$@" >/dev/null 2>&1 <<'PY'
import sys

from safetensors import safe_open

for path in sys.argv[1:]:
    with safe_open(path, framework="numpy") as handle:
        keys = list(handle.keys())
        if not keys:
            raise SystemExit(f"empty safetensors file: {path}")
PY
}

committed_checkpoint_root() {
  local output_dir="$1"
  local last_checkpoint="${output_dir}/checkpoints/last"
  local committed_root
  local checkpoint_parent

  checkpoint_output_is_scoped "${output_dir}" || return 1
  [[ -L "${last_checkpoint}" ]] || return 1
  committed_root="$(realpath -e "${last_checkpoint}" 2>/dev/null)" || return 1
  [[ -d "${committed_root}" && ! -L "${committed_root}" ]] || return 1
  [[ "$(basename "${committed_root}")" =~ ^[0-9]{6}$ ]] || return 1
  checkpoint_parent="$(realpath -e "$(dirname "${committed_root}")")" || return 1
  [[ "${checkpoint_parent}" == "$(realpath -e "${output_dir}/checkpoints")" ]] || return 1
  printf '%s\n' "${committed_root}"
}

checkpoint_is_committed() {
  local output_dir="$1"
  local checkpoint_name="$2"
  local committed_root
  committed_root="$(committed_checkpoint_root "${output_dir}")" || return 1
  [[ "$(basename "${committed_root}")" == "${checkpoint_name}" ]]
}

repair_last_checkpoint_pointer() {
  local output_dir="$1"
  local checkpoint_dir="$2"
  local checkpoint_name
  local checkpoints_root
  local temporary_link
  local last_checkpoint="${output_dir}/checkpoints/last"

  checkpoint_output_is_scoped "${output_dir}" || return 2
  [[ -d "${checkpoint_dir}" && ! -L "${checkpoint_dir}" ]] || return 2
  checkpoint_name="$(basename "${checkpoint_dir}")"
  [[ "${checkpoint_name}" =~ ^[0-9]{6}$ ]] || return 2
  checkpoints_root="$(realpath -e "${output_dir}/checkpoints")" || return 2
  [[ "$(realpath -e "$(dirname "${checkpoint_dir}")")" == "${checkpoints_root}" ]] || return 2
  if [[ -e "${last_checkpoint}" && ! -L "${last_checkpoint}" ]]; then
    log "refusing to replace non-symlink checkpoint commit pointer: ${last_checkpoint}"
    return 2
  fi

  temporary_link="${output_dir}/checkpoints/.last.repair.$$"
  [[ ! -e "${temporary_link}" && ! -L "${temporary_link}" ]] || return 2
  ln -s "${checkpoint_name}" "${temporary_link}" || return 2
  if ! mv -Tf "${temporary_link}" "${last_checkpoint}"; then
    log "failed atomic checkpoint commit-pointer repair for ${checkpoint_dir}"
    return 2
  fi
  checkpoint_is_committed "${output_dir}" "${checkpoint_name}" || return 2
  log "atomically repaired checkpoint commit pointer after explicit resume authorization: ${last_checkpoint} -> ${checkpoint_name}"
}

checkpoint_layout_is_final_only() {
  local output_dir="$1"
  local checkpoint_name="$2"
  local -a checkpoint_directories
  [[ -d "${output_dir}/checkpoints/${checkpoint_name}" ]] || return 1
  mapfile -t checkpoint_directories < <(
    find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  )
  [[ "${#checkpoint_directories[@]}" == "1" ]] && \
    [[ "${checkpoint_directories[0]}" == "${checkpoint_name}" ]]
}

checkpoint_artifact_is_complete() {
  local output_dir="$1"
  local checkpoint_name="$2"
  local expected_steps="$3"
  local expected_decay_lr="$4"
  local checkpoint_root="${output_dir}/checkpoints/${checkpoint_name}"
  local checkpoint="${checkpoint_root}/pretrained_model"
  local recorded_step
  local saved_output_dir
  local output_canonical
  local required_file
  local allowed_accumulation_a=24
  local allowed_accumulation_b=24

  if [[ "${output_dir}" == "${STAGE1_OUTPUT}" ]]; then
    allowed_accumulation_b=48
  fi

  [[ "${checkpoint_name}" =~ ^[0-9]{6}$ ]] || return 1
  for required_file in \
    config.json model.safetensors policy_preprocessor.json policy_postprocessor.json train_config.json; do
    [[ -s "${checkpoint}/${required_file}" ]] || return 1
  done
  for required_file in \
    optimizer_param_groups.json optimizer_state.safetensors rng_state.safetensors \
    scheduler_state.json training_step.json; do
    [[ -s "${checkpoint_root}/training_state/${required_file}" ]] || return 1
  done
  for required_file in \
    config.json policy_preprocessor.json policy_postprocessor.json train_config.json; do
    jq -e 'type == "object"' "${checkpoint}/${required_file}" >/dev/null 2>&1 || return 1
  done
  jq -e 'type == "array" and length > 0' \
    "${checkpoint_root}/training_state/optimizer_param_groups.json" >/dev/null 2>&1 || return 1
  jq -e 'type == "object"' \
    "${checkpoint_root}/training_state/scheduler_state.json" >/dev/null 2>&1 || return 1
  jq -e 'type == "object"' \
    "${checkpoint_root}/training_state/training_step.json" >/dev/null 2>&1 || return 1
  safetensors_files_are_structurally_valid \
    "${checkpoint}/model.safetensors" \
    "${checkpoint_root}/training_state/optimizer_state.safetensors" \
    "${checkpoint_root}/training_state/rng_state.safetensors" || return 1
  recorded_step="$(jq -r '.step // empty' "${checkpoint_root}/training_state/training_step.json" 2>/dev/null || true)"
  [[ "${recorded_step}" =~ ^[0-9]+$ ]] || return 1
  (( 10#${checkpoint_name} == recorded_step )) || return 1
  (( recorded_step > 0 && recorded_step <= expected_steps )) || return 1
  jq -e --argjson recorded_step "${recorded_step}" '
    (.last_epoch == $recorded_step)
    and (._step_count == ($recorded_step + 1))
    and ((._last_lr | type) == "array")
    and ((.base_lrs | type) == "array")
  ' "${checkpoint_root}/training_state/scheduler_state.json" >/dev/null 2>&1 || return 1
  output_canonical="$(realpath -e "${output_dir}" 2>/dev/null)" || return 1
  saved_output_dir="$(jq -r '.output_dir // empty' "${checkpoint}/train_config.json" 2>/dev/null || true)"
  [[ -n "${saved_output_dir}" ]] || return 1
  [[ "$(realpath -m "${saved_output_dir}")" == "${output_canonical}" ]] || return 1

  jq -e \
    --argjson expected_steps "${expected_steps}" \
    --argjson expected_decay_lr "${expected_decay_lr}" \
    --argjson allowed_accumulation_a "${allowed_accumulation_a}" \
    --argjson allowed_accumulation_b "${allowed_accumulation_b}" '
      (.steps == $expected_steps)
      and (.batch_size == 1)
      and (
        (.gradient_accumulation_steps == $allowed_accumulation_a)
        or (.gradient_accumulation_steps == $allowed_accumulation_b)
      )
      and (.save_checkpoint == true)
      and ((.save_freq | type) == "number")
      and (.save_freq >= 1)
      and (.eval_freq == 0)
      and ((.resume == false) or (.resume == true))
      and (.seed == 1000)
      and (.policy.vlm_backend == "molmo2_full")
      and (.policy.num_vlm_layers == 36)
      and (.policy.num_expert_layers == 36)
      and (.policy.optimizer_lr == 0.0001)
      and (.policy.optimizer_betas == [0.9, 0.95])
      and (.policy.optimizer_eps == 0.00000001)
      and (.policy.optimizer_weight_decay == 0.0000000001)
      and (.policy.optimizer_grad_clip_norm == 10)
      and (.policy.scheduler_warmup_steps == 100)
      and (.policy.scheduler_decay_steps == 30000)
      and (.policy.scheduler_decay_lr == $expected_decay_lr)
      and (.policy.worldflow_enable == false)
      and (.policy.worldflow_bootstrap_from_ego == false)
      and (.optimizer.type == "adamw")
      and (.optimizer.lr == 0.0001)
      and (.optimizer.betas == [0.9, 0.95])
      and (.optimizer.eps == 0.00000001)
      and (.optimizer.weight_decay == 0.0000000001)
      and (.optimizer.grad_clip_norm == 10)
      and (.scheduler.type == "cosine_decay_with_warmup")
      and (.scheduler.num_warmup_steps == 100)
      and (.scheduler.num_decay_steps == 30000)
      and (.scheduler.peak_lr == 0.0001)
      and (.scheduler.decay_lr == $expected_decay_lr)
    ' "${checkpoint}/train_config.json" >/dev/null 2>&1
}

training_stage_is_complete() {
  local output_dir="$1"
  local checkpoint_name="$2"
  local expected_steps="$3"
  local expected_decay_lr="$4"
  local last_checkpoint="${output_dir}/checkpoints/last"

  checkpoint_artifact_is_complete \
    "${output_dir}" "${checkpoint_name}" "${expected_steps}" "${expected_decay_lr}" || return 1
  [[ "$(jq -r '.step // empty' "${output_dir}/checkpoints/${checkpoint_name}/training_state/training_step.json" 2>/dev/null || true)" == "${expected_steps}" ]] || return 1
  checkpoint_layout_is_final_only "${output_dir}" "${checkpoint_name}" || return 1
  [[ -L "${last_checkpoint}" ]] || return 1
  [[ "$(realpath -e "${last_checkpoint}" 2>/dev/null || true)" == \
    "$(realpath -e "${output_dir}/checkpoints/${checkpoint_name}" 2>/dev/null || true)" ]]
}

latest_recoverable_checkpoint() {
  local output_dir="$1"
  local expected_steps="$2"
  local expected_decay_lr="$3"
  local checkpoint_dir
  local checkpoint_name
  local recorded_step

  checkpoint_dir="$(committed_checkpoint_root "${output_dir}")" || return 1
  checkpoint_name="$(basename "${checkpoint_dir}")"
  checkpoint_artifact_is_complete \
    "${output_dir}" "${checkpoint_name}" "${expected_steps}" "${expected_decay_lr}" || return 1
  recorded_step="$(jq -r '.step' "${checkpoint_dir}/training_state/training_step.json")"
  (( recorded_step < expected_steps )) || return 1
  printf '%s\n' "${checkpoint_dir}"
}

latest_structurally_complete_checkpoint() {
  local output_dir="$1"
  local expected_steps="$2"
  local expected_decay_lr="$3"
  local checkpoint_dir
  local checkpoint_name

  [[ -d "${output_dir}/checkpoints" ]] || return 1
  while IFS= read -r checkpoint_dir; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    if checkpoint_artifact_is_complete \
      "${output_dir}" "${checkpoint_name}" "${expected_steps}" "${expected_decay_lr}"; then
      printf '%s\n' "${checkpoint_dir}"
      return 0
    fi
  done < <(
    find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
      -regextype posix-extended -regex '.*/[0-9]{6}' -print 2>/dev/null | sort -r
  )
  return 1
}

checkpoint_output_is_scoped() {
  local output_dir="$1"
  case "${output_dir}" in
    "${STAGE1_OUTPUT}"|"${STAGE2_OUTPUT}"|"${STAGE3_OUTPUT}") ;;
    *) return 1 ;;
  esac
  [[ ! -L "${output_dir}" ]] || return 1
  if [[ -e "${output_dir}/checkpoints" ]]; then
    [[ -d "${output_dir}/checkpoints" && ! -L "${output_dir}/checkpoints" ]] || return 1
  fi
}

delete_scoped_checkpoint_directory() {
  local output_dir="$1"
  local checkpoint_dir="$2"
  local checkpoint_name
  local checkpoint_parent

  checkpoint_output_is_scoped "${output_dir}" || return 2
  [[ -d "${checkpoint_dir}" && ! -L "${checkpoint_dir}" ]] || return 2
  checkpoint_name="$(basename "${checkpoint_dir}")"
  [[ "${checkpoint_name}" =~ ^[0-9]{6}$ ]] || return 2
  checkpoint_parent="$(realpath -e "$(dirname "${checkpoint_dir}")")" || return 2
  [[ "${checkpoint_parent}" == "$(realpath -e "${output_dir}/checkpoints")" ]] || return 2
  if ! find "${checkpoint_dir}" -depth -delete; then
    log "checkpoint pruning failed for scoped path: ${checkpoint_dir}"
    return 2
  fi
  log "pruned recoverable numeric checkpoint after scope audit: ${checkpoint_dir}"
}

prune_recovery_checkpoints() {
  local output_dir="$1"
  local expected_steps="$2"
  local expected_decay_lr="$3"
  local checkpoint_dir
  local checkpoint_name
  local committed_root
  local committed_step
  local current_committed_root
  local retained=0
  local -a valid_checkpoints=()

  [[ -d "${output_dir}/checkpoints" ]] || return 0
  checkpoint_output_is_scoped "${output_dir}" || return 2
  committed_root="$(committed_checkpoint_root "${output_dir}")" || return 0
  checkpoint_name="$(basename "${committed_root}")"
  checkpoint_artifact_is_complete \
    "${output_dir}" "${checkpoint_name}" "${expected_steps}" "${expected_decay_lr}" || return 0
  committed_step=$((10#${checkpoint_name}))
  while IFS= read -r checkpoint_dir; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    if (( 10#${checkpoint_name} > committed_step )); then
      continue
    fi
    if checkpoint_artifact_is_complete \
      "${output_dir}" "${checkpoint_name}" "${expected_steps}" "${expected_decay_lr}"; then
      valid_checkpoints+=("${checkpoint_dir}")
    fi
  done < <(
    find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
      -regextype posix-extended -regex '.*/[0-9]{6}' -print 2>/dev/null | sort -r
  )

  for checkpoint_dir in "${valid_checkpoints[@]}"; do
    retained=$((retained + 1))
    if (( retained <= RECOVERY_CHECKPOINTS_TO_KEEP )); then
      continue
    fi
    current_committed_root="$(committed_checkpoint_root "${output_dir}" || true)"
    if [[ "${current_committed_root}" != "${committed_root}" ]]; then
      log "checkpoint commit pointer changed during retention; skipping the rest of this pruning pass"
      return 0
    fi
    delete_scoped_checkpoint_directory "${output_dir}" "${checkpoint_dir}" || return 2
  done
}

prune_stage_to_final_checkpoint() {
  local output_dir="$1"
  local final_checkpoint_name="$2"
  local expected_steps="$3"
  local expected_decay_lr="$4"
  local checkpoint_dir
  local checkpoint_name

  checkpoint_artifact_is_complete \
    "${output_dir}" "${final_checkpoint_name}" "${expected_steps}" "${expected_decay_lr}" || return 1
  checkpoint_is_committed "${output_dir}" "${final_checkpoint_name}" || return 1
  [[ "$(jq -r '.step' "${output_dir}/checkpoints/${final_checkpoint_name}/training_state/training_step.json")" == \
    "${expected_steps}" ]] || return 1
  checkpoint_output_is_scoped "${output_dir}" || return 2

  while IFS= read -r checkpoint_dir; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    if [[ "${checkpoint_name}" == "${final_checkpoint_name}" ]]; then
      continue
    fi
    checkpoint_is_committed "${output_dir}" "${final_checkpoint_name}" || return 1
    delete_scoped_checkpoint_directory "${output_dir}" "${checkpoint_dir}" || return 2
  done < <(
    find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
      -regextype posix-extended -regex '.*/[0-9]{6}' -print 2>/dev/null | sort
  )
}

record_source_checkpoint() {
  local record_path="$1"
  local checkpoint="$2"
  local canonical_checkpoint
  canonical_checkpoint="$(realpath -e "${checkpoint}")"
  if [[ -f "${record_path}" ]]; then
    if [[ "$(head -n 1 "${record_path}")" != "${canonical_checkpoint}" ]]; then
      log "source checkpoint contract mismatch: ${record_path}"
      exit 2
    fi
  else
    printf '%s\n' "${canonical_checkpoint}" > "${record_path}"
  fi
}

stage_source_canonical() {
  local source_checkpoint="$1"
  if [[ -n "${source_checkpoint}" ]]; then
    realpath -e "${source_checkpoint}"
  else
    printf '\n'
  fi
}

stage_state_matches_contract() {
  local state_path="$1"
  local label="$2"
  local output_dir="$3"
  local steps="$4"
  local decay_lr="$5"
  local source_checkpoint="$6"
  local source_canonical
  source_canonical="$(stage_source_canonical "${source_checkpoint}")" || return 1
  [[ -s "${state_path}" ]] || return 1
  jq -e \
    --arg label "${label}" \
    --arg output "$(realpath -m "${output_dir}")" \
    --arg source "${source_canonical}" \
    --argjson steps "${steps}" \
    --argjson decay_lr "${decay_lr}" \
    --argjson checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS}" '
      (.label == $label)
      and (.output == $output)
      and (.source_checkpoint == $source)
      and (.steps == $steps)
      and (.scheduler_decay_lr == $decay_lr)
      and (.global_batch_size == 192)
      and (.checkpoint_interval_steps == $checkpoint_interval_steps)
      and ((.status == "running") or (.status == "complete"))
    ' "${state_path}" >/dev/null 2>&1
}

write_stage_state() {
  local state_path="$1"
  local label="$2"
  local output_dir="$3"
  local steps="$4"
  local decay_lr="$5"
  local source_checkpoint="$6"
  local status="$7"
  local attempt_mode="$8"
  local resume_checkpoint="$9"
  local attempt_log="${10}"
  local source_canonical
  local state_tmp="${state_path}.tmp.$$"
  local history_suffix

  source_canonical="$(stage_source_canonical "${source_checkpoint}")"
  jq -n \
    --arg updated_at "$(date -u +%FT%TZ)" \
    --arg label "${label}" \
    --arg output "$(realpath -m "${output_dir}")" \
    --arg source_checkpoint "${source_canonical}" \
    --arg status "${status}" \
    --arg attempt_mode "${attempt_mode}" \
    --arg resume_checkpoint "${resume_checkpoint}" \
    --arg attempt_log "${attempt_log}" \
    --argjson steps "${steps}" \
    --argjson decay_lr "${decay_lr}" \
    --argjson checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS}" '
      {
        updated_at: $updated_at,
        label: $label,
        output: $output,
        source_checkpoint: $source_checkpoint,
        status: $status,
        attempt_mode: $attempt_mode,
        resume_checkpoint: $resume_checkpoint,
        attempt_log: $attempt_log,
        steps: $steps,
        scheduler_decay_lr: $decay_lr,
        checkpoint_interval_steps: $checkpoint_interval_steps,
        global_batch_size: 192
      }
    ' > "${state_tmp}"
  if [[ -e "${state_path}" ]]; then
    history_suffix="$(date -u +%Y%m%dT%H%M%S.%NZ)_pid$$"
    cp -p "${state_path}" "${state_path}.history.${history_suffix}"
  fi
  mv -f "${state_tmp}" "${state_path}"
}

read_recovery_action() {
  local action_path="$1"
  local action=""
  if [[ -f "${action_path}" ]]; then
    action="$(head -n 1 "${action_path}" | tr -d '[:space:]')"
  fi
  case "${action}" in
    ""|resume|restart) printf '%s\n' "${action}" ;;
    *) return 2 ;;
  esac
}

consume_recovery_action() {
  local action_path="$1"
  local action="$2"
  local used_path
  [[ -e "${action_path}" ]] || return 0
  used_path="${action_path}.used.$(date -u +%Y%m%dT%H%M%S.%NZ)_${action}_pid$$"
  mv "${action_path}" "${used_path}"
  log "consumed explicit recovery action '${action}'; preserved marker at ${used_path}"
}

archive_incomplete_stage_for_restart() {
  local label="$1"
  local output_dir="$2"
  local state_path="$3"
  local launch_record_dir="${output_dir}.launch"
  local archive_suffix
  local pid_path="${CONTROL_ROOT}/${label}.pid"
  local recorded_pid=""

  checkpoint_output_is_scoped "${output_dir}" || return 2
  if [[ -s "${pid_path}" ]]; then
    recorded_pid="$(head -n 1 "${pid_path}" | tr -d '[:space:]')"
    if [[ "${recorded_pid}" =~ ^[0-9]+$ ]] && kill -0 "${recorded_pid}" 2>/dev/null; then
      log "refusing restart archive because recorded ${label} pid=${recorded_pid} is still alive"
      return 2
    fi
  fi

  archive_suffix="failed_$(date -u +%Y%m%dT%H%M%S.%NZ)_pid$$"
  if [[ -e "${output_dir}" ]]; then
    if ! mv "${output_dir}" "${output_dir}.${archive_suffix}"; then
      log "failed to preserve incomplete stage output: ${output_dir}"
      return 2
    fi
    log "preserved incomplete stage output at ${output_dir}.${archive_suffix}"
  fi
  if [[ -e "${launch_record_dir}" ]]; then
    if ! mv "${launch_record_dir}" "${launch_record_dir}.${archive_suffix}"; then
      log "failed to preserve incomplete launch provenance: ${launch_record_dir}"
      return 2
    fi
    log "preserved incomplete launch provenance at ${launch_record_dir}.${archive_suffix}"
  fi
  if [[ -e "${state_path}" ]]; then
    if ! mv "${state_path}" "${state_path}.${archive_suffix}"; then
      log "failed to preserve incomplete stage-state marker: ${state_path}"
      return 2
    fi
    log "preserved incomplete stage-state marker at ${state_path}.${archive_suffix}"
  fi
}

prepare_attempt_log() {
  local log_path="$1"
  local archived_log
  if [[ -e "${log_path}" ]]; then
    archived_log="${log_path}.previous.$(date -u +%Y%m%dT%H%M%S.%NZ)_pid$$"
    mv "${log_path}" "${archived_log}"
    log "preserved previous attempt log at ${archived_log}"
  fi
}

last_logged_step() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    printf '0\n'
    return 0
  fi
  tr '\r' '\n' < "${log_path}" \
    | grep -Eo 'step:[0-9]+' \
    | tail -n 1 \
    | sed 's/step://' || true
}

run_smoke_if_needed() {
  if [[ -f "${SMOKE_MARKER}" ]] && jq -e \
    '.status == "complete" and .optimizer_steps == 2 and .batch_size == 1 and .gradient_accumulation_steps == 24' \
    "${SMOKE_MARKER}" >/dev/null 2>&1; then
    log "fresh two-step smoke already completed; skipping"
    return 0
  fi
  if [[ -e "${SMOKE_OUTPUT_DIR}" ]]; then
    log "unexpected pre-existing per-process smoke path: ${SMOKE_OUTPUT_DIR}"
    exit 2
  fi
  wait_for_training_capacity "fresh smoke"
  log "starting 8-GPU two-optimizer-step smoke (batch=1, accumulation=24, global_batch=192)"
  env \
    MOLMO_NUM_PROCESSES=8 \
    MOLMO_BATCH_SIZE=1 \
    MOLMO_ACCUMULATION_STEPS=24 \
    MOLMO_STEPS=2 \
    MOLMO_NUM_WORKERS=0 \
    MOLMO_SAVE_CHECKPOINT=false \
    MOLMO_SAVE_FREQ=2 \
    MOLMO_WANDB_ENABLE=false \
    MOLMO_POLICY_PATH= \
    MOLMO_SCHEDULER_DECAY_LR=2.5e-6 \
    MOLMO_OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
    MOLMO_LAUNCH_RECORD_DIR="${SMOKE_LAUNCH_RECORD_DIR}" \
    MOLMO_JOB_NAME=full_molmo2er_wep_vla_fresh_two_step_smoke \
    MOLMO_MAIN_PROCESS_PORT=29650 \
    MOLMO_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    WANDB_MODE=offline \
    setsid bash "${TRAIN_SCRIPT}" > "${SMOKE_LOG}" 2>&1 &
  ACTIVE_PID="$!"
  printf '%s\n' "${ACTIVE_PID}" > "${CONTROL_ROOT}/smoke.pid"
  if monitor_training_child smoke "${SMOKE_LOG}"; then
    :
  else
    smoke_rc="$?"
    log "smoke failed rc=${smoke_rc}; inspect ${SMOKE_LOG}"
    exit "${smoke_rc}"
  fi
  if [[ "$(last_logged_step "${SMOKE_LOG}")" != "2" ]] || \
    [[ "$(error_count "${SMOKE_LOG}")" != "0" ]]; then
    log "smoke returned success but failed the two-step/error audit"
    exit 2
  fi
  cleanup_smoke_output
  smoke_marker_tmp="${SMOKE_MARKER}.tmp.$$"
  jq -n \
    --arg completed_at "$(date -u +%FT%TZ)" \
    '{
      status: "complete",
      completed_at: $completed_at,
      gpu_processes: 8,
      optimizer_steps: 2,
      batch_size: 1,
      gradient_accumulation_steps: 24,
      global_batch_size: 192,
      temporary_output_deleted: true
    }' > "${smoke_marker_tmp}"
  mv -f "${smoke_marker_tmp}" "${SMOKE_MARKER}"
  log "fresh smoke passed and temporary output was deleted"
}

run_training_stage() {
  local label="$1"
  local output_dir="$2"
  local checkpoint_name="$3"
  local steps="$4"
  local decay_lr="$5"
  local source_checkpoint="$6"
  local log_path="$7"
  local port="$8"
  local state_path="${CONTROL_ROOT}/${label}.stage_state.json"
  local action_path="${CONTROL_ROOT}/${label}.recovery_action"
  local recovery_action=""
  local resume_checkpoint=""
  local resume_config_path=""
  local resume_mode="false"
  local resume_step=""
  local attempt_mode=""
  local state_contract_known="false"
  local launch_policy_path="${source_checkpoint}"
  local allow_resume_world_size_change="false"

  if training_stage_is_complete "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}"; then
    log "${label} already complete; final-only checkpoint audit passed"
    return 0
  fi
  if [[ -n "${source_checkpoint}" ]]; then
    record_source_checkpoint "${CONTROL_ROOT}/${label}_source_checkpoint.txt" "${source_checkpoint}"
  fi

  # A SIGKILL can land after the final checkpoint is durable but before the
  # retention/audit code runs. In that case, finish the idempotent pruning and
  # accept the stage without launching another optimizer step.
  if checkpoint_artifact_is_complete \
    "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}" && \
    checkpoint_is_committed "${output_dir}" "${checkpoint_name}" && \
    [[ "$(jq -r '.step' "${output_dir}/checkpoints/${checkpoint_name}/training_state/training_step.json")" == "${steps}" ]]; then
    if ! prune_stage_to_final_checkpoint \
      "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}"; then
      log "${label} has a durable final checkpoint but final-only pruning failed"
      exit 2
    fi
    if ! training_stage_is_complete "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}"; then
      log "${label} post-SIGKILL final checkpoint failed the final-only audit"
      exit 2
    fi
    write_stage_state \
      "${state_path}" "${label}" "${output_dir}" "${steps}" "${decay_lr}" \
      "${source_checkpoint}" complete recovered_after_final_checkpoint "" "${log_path}"
    log "${label} recovered after final checkpoint save; final-only audit passed"
    return 0
  fi

  if ! recovery_action="$(read_recovery_action "${action_path}")"; then
    log "invalid recovery action in ${action_path}; first line must be exactly resume or restart"
    exit 2
  fi

  if [[ -e "${output_dir}" || -e "${output_dir}.launch" ]]; then
    if stage_state_matches_contract \
      "${state_path}" "${label}" "${output_dir}" "${steps}" "${decay_lr}" "${source_checkpoint}"; then
      state_contract_known="true"
    fi

    if [[ "${recovery_action}" == "restart" ]]; then
      archive_incomplete_stage_for_restart "${label}" "${output_dir}" "${state_path}" || exit 2
      consume_recovery_action "${action_path}" restart
    else
      resume_checkpoint="$(latest_recoverable_checkpoint "${output_dir}" "${steps}" "${decay_lr}" || true)"
      if [[ -z "${resume_checkpoint}" ]]; then
        if [[ "${recovery_action}" == "resume" ]]; then
          resume_checkpoint="$(latest_structurally_complete_checkpoint "${output_dir}" "${steps}" "${decay_lr}" || true)"
          if [[ -n "${resume_checkpoint}" ]]; then
            repair_last_checkpoint_pointer "${output_dir}" "${resume_checkpoint}" || exit 2
          else
            log "${label} was explicitly marked resume, but no structurally complete optimizer/scheduler checkpoint exists"
            exit 2
          fi
        else
          log "${label} has no committed recovery checkpoint; use an explicit 'resume' marker to deep-validate/adopt an uncommitted checkpoint, or 'restart' to preserve and restart: ${action_path}"
          exit 2
        fi
      fi
      if [[ "${state_contract_known}" != "true" && "${recovery_action}" != "resume" ]]; then
        log "${label} output has a recovery checkpoint but no matching stage-state marker; to adopt it, run: printf 'resume\\n' > ${action_path}"
        exit 2
      fi
      resume_checkpoint="$(realpath -e "${resume_checkpoint}")"
      resume_step="$(jq -r '.step' "${resume_checkpoint}/training_state/training_step.json")"
      if [[ "${resume_step}" == "${steps}" ]]; then
        consume_recovery_action "${action_path}" resume
        prune_stage_to_final_checkpoint \
          "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}" || exit 2
        training_stage_is_complete \
          "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}" || exit 2
        write_stage_state \
          "${state_path}" "${label}" "${output_dir}" "${steps}" "${decay_lr}" \
          "${source_checkpoint}" complete explicit_final_commit_repair "${resume_checkpoint}" "${log_path}"
        log "${label} accepted explicitly repaired final checkpoint; final-only audit passed"
        return 0
      fi
      resume_config_path="${resume_checkpoint}/pretrained_model/train_config.json"
      resume_mode="true"
      launch_policy_path=""
      attempt_mode="resume_in_stage"
      if [[ "${label}" == "stage1_fresh_036000" ]] && \
        [[ "$(jq -r '.gradient_accumulation_steps' "${resume_config_path}")" == "48" ]]; then
        allow_resume_world_size_change="true"
        attempt_mode="resume_in_stage_4gpu_to_8gpu"
      fi
      if [[ "${recovery_action}" == "resume" ]]; then
        consume_recovery_action "${action_path}" resume
      fi
      log "${label} selected exact in-stage resume checkpoint=${resume_checkpoint} step=${resume_step}"
    fi
  elif [[ "${recovery_action}" == "resume" ]]; then
    log "${label} cannot resume because output directory does not exist: ${output_dir}"
    exit 2
  elif [[ "${recovery_action}" == "restart" ]]; then
    consume_recovery_action "${action_path}" restart
  fi

  if [[ "${resume_mode}" == "false" ]]; then
    if [[ -n "${source_checkpoint}" ]]; then
      attempt_mode="warm_start_resume_false"
    else
      attempt_mode="fresh_resume_false"
    fi
  fi

  prepare_attempt_log "${log_path}"
  write_stage_state \
    "${state_path}" "${label}" "${output_dir}" "${steps}" "${decay_lr}" \
    "${source_checkpoint}" running "${attempt_mode}" "${resume_checkpoint}" "${log_path}"
  wait_for_training_capacity "${label}"
  log "starting ${label}: mode=${attempt_mode} steps=${steps} source=${source_checkpoint:-fresh} resume_checkpoint=${resume_checkpoint:-none} lr=0.0001 warmup=100 decay_steps=30000 decay_lr=${decay_lr} save_freq=${CHECKPOINT_INTERVAL_STEPS}"
  env \
    MOLMO_NUM_PROCESSES=8 \
    MOLMO_BATCH_SIZE=1 \
    MOLMO_ACCUMULATION_STEPS=24 \
    MOLMO_STEPS="${steps}" \
    MOLMO_NUM_WORKERS=8 \
    MOLMO_SAVE_CHECKPOINT=true \
    MOLMO_SAVE_FREQ="${CHECKPOINT_INTERVAL_STEPS}" \
    MOLMO_WANDB_ENABLE=true \
    MOLMO_WANDB_RUN_MODE=offline \
    MOLMO_POLICY_PATH="${launch_policy_path}" \
    MOLMO_RESUME="${resume_mode}" \
    MOLMO_RESUME_CONFIG_PATH="${resume_config_path}" \
    MOLMO_RESUME_ALLOW_WORLD_SIZE_CHANGE="${allow_resume_world_size_change}" \
    MOLMO_SCHEDULER_DECAY_LR="${decay_lr}" \
    MOLMO_OUTPUT_DIR="${output_dir}" \
    MOLMO_LAUNCH_RECORD_DIR="${output_dir}.launch" \
    MOLMO_JOB_NAME="full_molmo2er_wep_vla_${label}" \
    MOLMO_MAIN_PROCESS_PORT="${port}" \
    MOLMO_MIN_GPU_FREE_MIB="${TRAINING_MIN_GPU_FREE_MIB}" \
    MOLMO_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    WANDB_MODE=offline \
    setsid bash "${TRAIN_SCRIPT}" > "${log_path}" 2>&1 &
  ACTIVE_PID="$!"
  printf '%s\n' "${ACTIVE_PID}" > "${CONTROL_ROOT}/${label}.pid"
  if monitor_training_child "${label}" "${log_path}" "${output_dir}" "${steps}" "${decay_lr}"; then
    :
  else
    stage_rc="$?"
    log "${label} failed rc=${stage_rc}; inspect ${log_path}"
    exit "${stage_rc}"
  fi
  if [[ "$(error_count "${log_path}")" != "0" ]]; then
    log "${label} log contains fatal error signatures despite a zero exit status"
    exit 2
  fi
  if [[ "$(last_logged_step "${log_path}")" != "${steps}" ]]; then
    log "${label} final logged step is $(last_logged_step "${log_path}") instead of ${steps}"
    exit 2
  fi
  if ! checkpoint_artifact_is_complete \
    "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}"; then
    log "${label} final checkpoint is incomplete before retention pruning"
    exit 2
  fi
  if ! prune_stage_to_final_checkpoint \
    "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}"; then
    log "${label} failed safe final-only checkpoint pruning"
    exit 2
  fi
  if ! training_stage_is_complete "${output_dir}" "${checkpoint_name}" "${steps}" "${decay_lr}"; then
    log "${label} failed checkpoint/config/final-only completion audit"
    exit 2
  fi
  write_stage_state \
    "${state_path}" "${label}" "${output_dir}" "${steps}" "${decay_lr}" \
    "${source_checkpoint}" complete "${attempt_mode}" "${resume_checkpoint}" "${log_path}"
  log "${label} complete; audited checkpoint=${output_dir}/checkpoints/${checkpoint_name}/pretrained_model"
}

write_pipeline_manifest() {
  local manifest_path="${CONTROL_ROOT}/pipeline_contract.json"
  local temporary_path="${manifest_path}.tmp.$$"
  local archived_manifest
  jq -n \
    --arg created_at "$(date -u +%FT%TZ)" \
    --arg stage1_output "${STAGE1_OUTPUT}" \
    --arg stage2_output "${STAGE2_OUTPUT}" \
    --arg stage3_output "${STAGE3_OUTPUT}" \
    --arg eval_root "${EVAL_ROOT}" \
    --argjson monitor_interval_seconds "${POLL_SECONDS}" \
    --argjson checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS}" \
    --argjson recovery_checkpoints_to_keep "${RECOVERY_CHECKPOINTS_TO_KEEP}" '
      {
        created_at: $created_at,
        architecture: "Frozen Full-Molmo2-ER WEP-VLA",
        gpu_processes: 8,
        batch_size_per_process: 1,
        gradient_accumulation_steps: 24,
        global_batch_size: 192,
        optimizer_lr: 0.0001,
        scheduler_warmup_steps: 100,
        scheduler_decay_steps: 30000,
        save_policy: {
          interval_steps: $checkpoint_interval_steps,
          rolling_complete_checkpoints_to_keep: $recovery_checkpoints_to_keep,
          after_stage_completion: "final checkpoint only"
        },
        monitor_interval_seconds: $monitor_interval_seconds,
        stages: [
          {label: "checkpoint_036000", steps: 36000, initialization: "fresh", scheduler_decay_lr: 0.0000025, output: $stage1_output},
          {label: "checkpoint_066000", steps: 30000, cumulative_steps: 66000, initialization: "warm-start checkpoint_036000, resume=false", scheduler_decay_lr: 0.00003, output: $stage2_output},
          {label: "checkpoint_096000", steps: 30000, cumulative_steps: 96000, initialization: "warm-start checkpoint_066000, resume=false", scheduler_decay_lr: 0.00003, output: $stage3_output}
        ],
        evaluation: {checkpoints: [66000, 96000], root: $eval_root, tasks: 40, episodes_per_task: 50}
      }
    ' > "${temporary_path}"
  if [[ -e "${manifest_path}" ]]; then
    archived_manifest="${manifest_path}.previous.$(date -u +%Y%m%dT%H%M%S.%NZ)_pid$$"
    cp -p "${manifest_path}" "${archived_manifest}"
  fi
  mv -f "${temporary_path}" "${manifest_path}"
}

write_pipeline_manifest
log "pipeline lock acquired; monitor_interval=${POLL_SECONDS}s checkpoint_interval=${CHECKPOINT_INTERVAL_STEPS} keep_recovery=${RECOVERY_CHECKPOINTS_TO_KEEP} run_root=${RUN_ROOT}"

if training_stage_is_complete "${STAGE1_OUTPUT}" 036000 36000 2.5e-6; then
  log "stage1_fresh_036000 already complete; smoke and stage1 are re-entrant skips"
else
  run_smoke_if_needed
fi

run_training_stage \
  stage1_fresh_036000 "${STAGE1_OUTPUT}" 036000 36000 2.5e-6 "" "${STAGE1_LOG}" 29651
run_training_stage \
  stage2_from036000_to066000 "${STAGE2_OUTPUT}" 030000 30000 3e-5 "${CHECKPOINT_036000}" "${STAGE2_LOG}" 29652
run_training_stage \
  stage3_from066000_to096000 "${STAGE3_OUTPUT}" 030000 30000 3e-5 "${CHECKPOINT_066000}" "${STAGE3_LOG}" 29653

record_source_checkpoint "${CONTROL_ROOT}/evaluation_checkpoint_066000.txt" "${CHECKPOINT_066000}"
record_source_checkpoint "${CONTROL_ROOT}/evaluation_checkpoint_096000.txt" "${CHECKPOINT_096000}"
log "starting/re-auditing two-checkpoint LIBERO evaluation"
env \
  FULL_MOLMO2ER_EVAL_ROOT="${EVAL_ROOT}" \
  FULL_MOLMO2ER_CHECKPOINT_066000="${CHECKPOINT_066000}" \
  FULL_MOLMO2ER_CHECKPOINT_096000="${CHECKPOINT_096000}" \
  FULL_MOLMO2ER_EVAL_MIN_GPU_FREE_MIB=20480 \
  setsid bash "${EVAL_SCRIPT}" > "${EVAL_DRIVER_LOG}" 2>&1 &
ACTIVE_PID="$!"
printf '%s\n' "${ACTIVE_PID}" > "${CONTROL_ROOT}/evaluation.pid"
if monitor_evaluation_child; then
  :
else
  eval_rc="$?"
  log "LIBERO evaluation failed rc=${eval_rc}; inspect ${EVAL_DRIVER_LOG}"
  exit "${eval_rc}"
fi

if [[ "$(error_count "${EVAL_DRIVER_LOG}")" != "0" ]]; then
  log "evaluation driver log contains fatal error signatures"
  exit 2
fi
if [[ ! -s "${EVAL_ROOT}/comparison.json" ]] || ! jq -e \
  --arg checkpoint_066000 "$(realpath -e "${CHECKPOINT_066000}")" \
  --arg checkpoint_096000 "$(realpath -e "${CHECKPOINT_096000}")" '
    (.checkpoint_066000.path == $checkpoint_066000)
    and (.checkpoint_096000.path == $checkpoint_096000)
    and (.checkpoint_066000.episode_count == 2000)
    and (.checkpoint_096000.episode_count == 2000)
    and (.protocol.task_count == 40)
    and (.protocol.episodes_per_task == 50)
    and (.protocol.exec_action_steps == 24)
    and (.protocol.adaptive_exec_max_steps == 24)
    and (.protocol.grasp_exec_steps == 24)
  ' "${EVAL_ROOT}/comparison.json" >/dev/null; then
  log "evaluation completion audit failed: ${EVAL_ROOT}/comparison.json"
  exit 2
fi

if ! training_stage_is_complete "${STAGE1_OUTPUT}" 036000 36000 2.5e-6 || \
  ! training_stage_is_complete "${STAGE2_OUTPUT}" 030000 30000 3e-5 || \
  ! training_stage_is_complete "${STAGE3_OUTPUT}" 030000 30000 3e-5; then
  log "final training re-audit failed"
  exit 2
fi

completion_tmp="${COMPLETION_AUDIT}.tmp.$$"
jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg checkpoint_036000 "$(realpath -e "${CHECKPOINT_036000}")" \
  --arg checkpoint_066000 "$(realpath -e "${CHECKPOINT_066000}")" \
  --arg checkpoint_096000 "$(realpath -e "${CHECKPOINT_096000}")" \
  --slurpfile comparison "${EVAL_ROOT}/comparison.json" '
    {
      status: "complete",
      completed_at: $completed_at,
      training: {
        global_batch_size: 192,
        final_checkpoint_only: true,
        checkpoint_036000: {path: $checkpoint_036000, stage_steps: 36000, scheduler_decay_lr: 0.0000025},
        checkpoint_066000: {path: $checkpoint_066000, stage_steps: 30000, cumulative_steps: 66000, resume: false, scheduler_decay_lr: 0.00003},
        checkpoint_096000: {path: $checkpoint_096000, stage_steps: 30000, cumulative_steps: 96000, resume: false, scheduler_decay_lr: 0.00003}
      },
      evaluation: $comparison[0]
    }
  ' > "${completion_tmp}"
mv -f "${completion_tmp}" "${COMPLETION_AUDIT}"

log "all three training stages and both LIBERO evaluations completed; audit=${COMPLETION_AUDIT}"
