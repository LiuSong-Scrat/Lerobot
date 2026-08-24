#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
PYTHON_BIN="${MOLMO_PYTHON_BIN:-${EXPERIMENT_ROOT}/.venv-smol5090/bin/python}"
TOOLS="${SCRIPT_DIR}/full_molmo2er_worldflow_tools.py"
TRAIN_LAUNCHER="${SCRIPT_DIR}/train_full_molmo2er_worldflow_8gpu.sh"
EVAL_LAUNCHER="${SCRIPT_DIR}/eval_full_molmo2er_worldflow_8gpu.sh"
RUN_ROOT="${FULL_MOLMO2ER_WORLD_PIPELINE_ROOT:-${EXPERIMENT_ROOT}/outputs/full_molmo2er_worldflow_three_stage}"
CONTROL_ROOT="${RUN_ROOT}.control"
LOCK_FILE="${FULL_MOLMO2ER_WORLD_PIPELINE_LOCK_FILE:-${CONTROL_ROOT}/pipeline.lock}"
STATE_FILE="${CONTROL_ROOT}/pipeline_state.json"
DATA_AUDIT_PATH="${CONTROL_ROOT}/data_semantic_audit.json"
MODEL_AUDIT_PATH="${CONTROL_ROOT}/model_source_audit.json"
SMOKE_ROOT="${CONTROL_ROOT}/two_step_smoke"
SMOKE_MARKER="${SMOKE_ROOT}/complete.json"
SAVE_FREQ="${FULL_MOLMO2ER_SAVE_FREQ:-2000}"
RUN_EVALUATION="${FULL_MOLMO2ER_RUN_EVALUATION:-true}"
DRY_RUN="${FULL_MOLMO2ER_DRY_RUN:-false}"
readonly BATCH_PROFILE="${FULL_MOLMO2ER_BATCH_PROFILE:-b24}"

MODEL_DIR="${MOLMO2_ER_MODEL_DIR:-${EXPERIMENT_ROOT}/Molmo2-ER}"
DATA_ROOT="${MOLMO_DATA_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data}"
DATASET_DIR="${MOLMO_DATASET_DIR:-${DATA_ROOT}/libero_4suite_lerobot_dataset}"
CACHE_DIR="${MOLMO_POINTSEG_CACHE_DIR:-${DATA_ROOT}/libero_4suite_lerobot_toolseg_cache}"

STAGE_LABELS=(
  stage1_fresh_000000_to036000
  stage2_finetune_036000_to066000
  stage3_finetune_066000_to096000
)
STAGE_KINDS=(fresh finetune finetune)
STAGE_STEPS=(36000 30000 30000)
STAGE_STARTS=(0 36000 66000)
STAGE_ENDS=(36000 66000 96000)
STAGE_FLOORS=(2.5e-6 3e-5 3e-5)
STAGE_PORTS=(29670 29671 29672)

fail() {
  printf '[full-molmo2er-pipeline] %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[full-molmo2er-pipeline] %s %s\n' "$(date -u +%FT%TZ)" "$*"
}

[[ -x "${PYTHON_BIN}" ]] || fail "Pipeline Python is missing or not executable: ${PYTHON_BIN}"
for required_file in "${TOOLS}" "${TRAIN_LAUNCHER}" "${EVAL_LAUNCHER}"; do
  [[ -f "${required_file}" ]] || fail "Missing pipeline component: ${required_file}"
done
[[ "${RUN_EVALUATION}" == "true" || "${RUN_EVALUATION}" == "false" ]] || fail "FULL_MOLMO2ER_RUN_EVALUATION must be true or false."
[[ "${BATCH_PROFILE}" == "b4" || "${BATCH_PROFILE}" == "b8" || "${BATCH_PROFILE}" == "b16" || "${BATCH_PROFILE}" == "b24" ]] || fail "FULL_MOLMO2ER_BATCH_PROFILE must be b4, b8, b16, or b24."
[[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "false" ]] || fail "FULL_MOLMO2ER_DRY_RUN must be true or false."
[[ "${SAVE_FREQ}" =~ ^[0-9]+$ ]] && (( SAVE_FREQ > 0 )) || fail "FULL_MOLMO2ER_SAVE_FREQ must be positive."
(( 36000 % SAVE_FREQ == 0 && 30000 % SAVE_FREQ == 0 )) || fail "Save frequency must divide both 36000 and 30000."

mkdir -p "${CONTROL_ROOT}"
if ! command -v flock >/dev/null 2>&1; then
  fail "The pipeline requires flock for cross-process duplicate-launch protection."
fi
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  fail "Another three-stage Full-Molmo2-ER WorldFlow pipeline holds ${LOCK_FILE}."
fi
printf '%s\n' "$$" > "${CONTROL_ROOT}/pipeline.pid"

if [[ "${DRY_RUN}" == "true" ]]; then
  log "dry-run plan: physical GPUs 0-7, 8 ranks, profile=${BATCH_PROFILE}, exact gradient global batch 192"
  log "planned gate: fresh Full+RGB+WorldFlow two-optimizer-step smoke with post-Adam eight-rank memory audit"
  for stage_index in "${!STAGE_LABELS[@]}"; do
    log "planned ${STAGE_LABELS[stage_index]}: ${STAGE_STEPS[stage_index]} steps, cumulative ${STAGE_STARTS[stage_index]}..${STAGE_ENDS[stage_index]}, floor=${STAGE_FLOORS[stage_index]}"
  done
  log "planned evaluation: cumulative checkpoints 066000 and 096000, 40 tasks x 50 episodes"
  exit 0
fi

[[ -d "${MODEL_DIR}" ]] || fail "Molmo2-ER model directory is missing: ${MODEL_DIR}"
"${PYTHON_BIN}" "${TOOLS}" model-audit \
  --model-dir "${MODEL_DIR}" \
  --output "${MODEL_AUDIT_PATH}" >/dev/null
[[ -d "${DATASET_DIR}" && -d "${CACHE_DIR}" ]] || fail "Dataset or PointSeg cache is missing."

if [[ -s "${DATA_AUDIT_PATH}" ]]; then
  EXPECTED_DATA_HASH="$("${PYTHON_BIN}" "${TOOLS}" json-get --file "${DATA_AUDIT_PATH}" --path semantic_hash)"
  "${PYTHON_BIN}" "${TOOLS}" data-audit \
    --dataset "${DATASET_DIR}" \
    --cache "${CACHE_DIR}" \
    --expected-hash "${EXPECTED_DATA_HASH}" \
    --output "${DATA_AUDIT_PATH}" >/dev/null
else
  "${PYTHON_BIN}" "${TOOLS}" data-audit \
    --dataset "${DATASET_DIR}" \
    --cache "${CACHE_DIR}" \
    --output "${DATA_AUDIT_PATH}" >/dev/null
  EXPECTED_DATA_HASH="$("${PYTHON_BIN}" "${TOOLS}" json-get --file "${DATA_AUDIT_PATH}" --path semantic_hash)"
fi
"${PYTHON_BIN}" "${TOOLS}" contract --batch-profile "${BATCH_PROFILE}" > "${CONTROL_ROOT}/formal_contract.json"
"${PYTHON_BIN}" "${TOOLS}" pipeline-event \
  --output "${STATE_FILE}" --status running --stage initialization \
  --message "locked exact global-batch/data contracts" >/dev/null

mkdir -p "${RUN_ROOT}/cumulative_checkpoints" "${RUN_ROOT}/logs"

run_two_step_smoke() {
  mkdir -p "${SMOKE_ROOT}/attempts"
  if [[ -s "${SMOKE_MARKER}" ]] && "${PYTHON_BIN}" "${TOOLS}" validate-smoke \
    --manifest "${SMOKE_MARKER}" --expected-data-hash "${EXPECTED_DATA_HASH}" \
    --batch-profile "${BATCH_PROFILE}" >/dev/null 2>&1; then
    log "preserving audited two-step Full+RGB+WorldFlow smoke marker: ${SMOKE_MARKER}"
    return 0
  fi

  local attempt_id
  local attempt_root
  local smoke_output
  local launch_dir
  local smoke_log
  attempt_id="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
  attempt_root="${SMOKE_ROOT}/attempts/${attempt_id}"
  smoke_output="${attempt_root}/output"
  launch_dir="${attempt_root}/launch"
  smoke_log="${attempt_root}/train.log"
  mkdir -p "${attempt_root}"
  "${PYTHON_BIN}" "${TOOLS}" pipeline-event \
    --output "${STATE_FILE}" --status smoke --stage two_step_smoke \
    --message "starting mandatory 8-rank Full+RGB+WorldFlow exact-192 smoke" >/dev/null
  log "starting mandatory two-step smoke; evidence will remain under ${attempt_root}"

  MOLMO_RUN_MODE=smoke \
  MOLMO_EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  MOLMO_PYTHON_BIN="${PYTHON_BIN}" \
  MOLMO2_ER_MODEL_DIR="${MODEL_DIR}" \
  MOLMO_DATASET_DIR="${DATASET_DIR}" \
  MOLMO_POINTSEG_CACHE_DIR="${CACHE_DIR}" \
  MOLMO_STAGE_LABEL=preflight_full_worldflow_two_step \
  MOLMO_STAGE_KIND=fresh \
  MOLMO_STEPS=2 \
  MOLMO_CUMULATIVE_START_STEP=0 \
  MOLMO_SCHEDULER_DECAY_LR=2.5e-6 \
  MOLMO_POLICY_PATH= \
  MOLMO_RESUME_CONFIG_PATH= \
  MOLMO_OUTPUT_DIR="${smoke_output}" \
  MOLMO_LAUNCH_RECORD_DIR="${launch_dir}" \
  MOLMO_DATA_AUDIT_PATH="${attempt_root}/data_semantic_audit.json" \
  MOLMO_EXPECTED_DATA_SEMANTIC_HASH="${EXPECTED_DATA_HASH}" \
  MOLMO_SAVE_FREQ=2 \
  MOLMO_SAVE_CHECKPOINT=false \
  MOLMO_NUM_WORKERS=0 \
  MOLMO_WANDB_ENABLE=false \
  MOLMO_HASH_MODEL_SHARDS=false \
  MOLMO_JOB_NAME=full_molmo2er_worldflow_preflight_two_step_seed1000_8xa800 \
  MOLMO_MAIN_PROCESS_PORT=29660 \
  MOLMO_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MOLMO_NUM_PROCESSES=8 \
  MOLMO_BATCH_PROFILE="${BATCH_PROFILE}" \
  bash "${TRAIN_LAUNCHER}" 2>&1 | tee "${smoke_log}"

  "${PYTHON_BIN}" "${TOOLS}" write-smoke-manifest \
    --output "${SMOKE_MARKER}" \
    --smoke-output "${smoke_output}" \
    --launch-manifest "${launch_dir}/launch_manifest.json" \
    --log "${smoke_log}" \
    --batch-profile "${BATCH_PROFILE}" \
    --expected-data-hash "${EXPECTED_DATA_HASH}" >/dev/null
  "${PYTHON_BIN}" "${TOOLS}" validate-smoke \
    --manifest "${SMOKE_MARKER}" --expected-data-hash "${EXPECTED_DATA_HASH}" \
    --batch-profile "${BATCH_PROFILE}" >/dev/null
  "${PYTHON_BIN}" "${TOOLS}" pipeline-event \
    --output "${STATE_FILE}" --status running --stage two_step_smoke \
    --message "mandatory smoke passed with eight-rank post-Adam CUDA memory and frozen-hash audits" >/dev/null
  log "mandatory two-step smoke passed: ${SMOKE_MARKER}"
}


run_stage() {
  local stage_index="$1"
  local label="${STAGE_LABELS[stage_index]}"
  local kind="${STAGE_KINDS[stage_index]}"
  local steps="${STAGE_STEPS[stage_index]}"
  local cumulative_start="${STAGE_STARTS[stage_index]}"
  local cumulative_end="${STAGE_ENDS[stage_index]}"
  local floor="${STAGE_FLOORS[stage_index]}"
  local port="${STAGE_PORTS[stage_index]}"
  local local_step_name
  local cumulative_name
  local_step_name="$(printf '%06d' "${steps}")"
  cumulative_name="$(printf '%06d' "${cumulative_end}")"
  local output_dir="${RUN_ROOT}/${label}"
  local launch_dir="${output_dir}.launch"
  local final_checkpoint="${output_dir}/checkpoints/${local_step_name}/pretrained_model"
  local cumulative_alias="${RUN_ROOT}/cumulative_checkpoints/${cumulative_name}"
  local alias_manifest="${CONTROL_ROOT}/checkpoint_${cumulative_name}.json"
  local log_path="${RUN_ROOT}/logs/${label}.log"
  local source_checkpoint=""

  if (( stage_index > 0 )); then
    source_checkpoint="${RUN_ROOT}/cumulative_checkpoints/$(printf '%06d' "${STAGE_ENDS[stage_index - 1]}")"
    "${PYTHON_BIN}" "${TOOLS}" checkpoint-audit --checkpoint "${source_checkpoint}" >/dev/null \
      || fail "Previous cumulative checkpoint is not usable: ${source_checkpoint}"
  fi

  if "${PYTHON_BIN}" "${TOOLS}" checkpoint-audit \
    --checkpoint "${final_checkpoint}" --expected-step "${steps}" >/dev/null 2>&1; then
    log "${label} is already complete; preserving its audited final checkpoint"
  else
    local resume_config=""
    local launch_policy_path="${source_checkpoint}"
    if [[ -d "${output_dir}" ]]; then
      if resume_config="$("${PYTHON_BIN}" "${TOOLS}" find-resume --output-dir "${output_dir}" --target-step "${steps}")"; then
        launch_policy_path=""
        log "${label} will resume in-stage from ${resume_config}"
      else
        local resume_status="$?"
        if (( resume_status == 4 )); then
          fail "${label} has an incomplete output directory but no complete recovery checkpoint: ${output_dir}"
        fi
        fail "Could not audit recovery checkpoints for ${label}."
      fi
    fi

    "${PYTHON_BIN}" "${TOOLS}" pipeline-event \
      --output "${STATE_FILE}" --status running --stage "${label}" \
      --message "$([[ -n "${resume_config}" ]] && printf in-stage-resume || printf fresh-stage-start)" >/dev/null
    log "starting ${label}: steps=${steps}, cumulative=${cumulative_start}..${cumulative_end}, floor=${floor}, source=${launch_policy_path:-resume-state}"

    MOLMO_RUN_MODE=formal \
    MOLMO_BATCH_PROFILE="${BATCH_PROFILE}" \
    MOLMO_EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
    MOLMO_PYTHON_BIN="${PYTHON_BIN}" \
    MOLMO2_ER_MODEL_DIR="${MODEL_DIR}" \
    MOLMO_DATASET_DIR="${DATASET_DIR}" \
    MOLMO_POINTSEG_CACHE_DIR="${CACHE_DIR}" \
    MOLMO_STAGE_LABEL="${label}" \
    MOLMO_STAGE_KIND="${kind}" \
    MOLMO_STEPS="${steps}" \
    MOLMO_CUMULATIVE_START_STEP="${cumulative_start}" \
    MOLMO_SCHEDULER_DECAY_LR="${floor}" \
    MOLMO_POLICY_PATH="${launch_policy_path}" \
    MOLMO_RESUME_CONFIG_PATH="${resume_config}" \
    MOLMO_OUTPUT_DIR="${output_dir}" \
    MOLMO_LAUNCH_RECORD_DIR="${launch_dir}" \
    MOLMO_DATA_AUDIT_PATH="${launch_dir}/data_semantic_audit.json" \
    MOLMO_EXPECTED_DATA_SEMANTIC_HASH="${EXPECTED_DATA_HASH}" \
    MOLMO_SAVE_FREQ="${SAVE_FREQ}" \
    MOLMO_SAVE_CHECKPOINT=true \
    MOLMO_MAIN_PROCESS_PORT="${port}" \
    MOLMO_JOB_NAME="full_molmo2er_worldflow_${label}_seed1000_8xa800" \
    bash "${TRAIN_LAUNCHER}" 2>&1 | tee -a "${log_path}"

    "${PYTHON_BIN}" "${TOOLS}" checkpoint-audit \
      --checkpoint "${final_checkpoint}" --expected-step "${steps}" \
      --output "${CONTROL_ROOT}/${label}_final_checkpoint_audit.json" >/dev/null \
      || fail "${label} returned successfully but its final checkpoint failed audit."
  fi

  "${PYTHON_BIN}" "${TOOLS}" publish-alias \
    --source "${final_checkpoint}" \
    --source-step "${steps}" \
    --cumulative-step "${cumulative_end}" \
    --alias "${cumulative_alias}" \
    --output "${alias_manifest}" >/dev/null
  "${PYTHON_BIN}" "${TOOLS}" pipeline-event \
    --output "${STATE_FILE}" --status running --stage "${label}" \
    --message "stage complete and cumulative checkpoint published" \
    --checkpoint "${cumulative_alias}" >/dev/null
  log "completed ${label}; cumulative checkpoint ${cumulative_name} -> ${final_checkpoint}"
}

run_two_step_smoke

for stage_index in "${!STAGE_LABELS[@]}"; do
  run_stage "${stage_index}"
done

CHECKPOINT_066000="${RUN_ROOT}/cumulative_checkpoints/066000"
CHECKPOINT_096000="${RUN_ROOT}/cumulative_checkpoints/096000"
if [[ "${RUN_EVALUATION}" == "true" ]]; then
  "${PYTHON_BIN}" "${TOOLS}" pipeline-event \
    --output "${STATE_FILE}" --status evaluating --stage libero_eval \
    --message "starting automatic 66k/96k LIBERO evaluation" >/dev/null
  FULL_MOLMO2ER_WORLD_PIPELINE_ROOT="${RUN_ROOT}" \
  FULL_MOLMO2ER_CHECKPOINT_066000="${CHECKPOINT_066000}" \
  FULL_MOLMO2ER_CHECKPOINT_096000="${CHECKPOINT_096000}" \
  MOLMO_EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  MOLMO_PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVAL_LAUNCHER}" 2>&1 | tee -a "${RUN_ROOT}/logs/libero_eval_066000_096000.log"
fi

"${PYTHON_BIN}" "${TOOLS}" pipeline-event \
  --output "${STATE_FILE}" --status complete --stage done \
  --message "$([[ "${RUN_EVALUATION}" == "true" ]] && printf 'three stages and evaluation complete' || printf 'three stages complete; evaluation explicitly skipped')" \
  --checkpoint "${CHECKPOINT_096000}" >/dev/null
log "formal three-stage pipeline complete: ${RUN_ROOT}"
