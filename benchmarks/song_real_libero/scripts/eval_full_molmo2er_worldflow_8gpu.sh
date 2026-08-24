#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
PYTHON_BIN="${MOLMO_PYTHON_BIN:-${EXPERIMENT_ROOT}/.venv-smol5090/bin/python}"
TOOLS="${SCRIPT_DIR}/full_molmo2er_worldflow_tools.py"
EVAL_SCRIPT="${SCRIPT_DIR}/libero_setting/libero_pointcloud_eval.py"
EVAL_CONFIG="${FULL_MOLMO2ER_EVAL_CONFIG:-${EXPERIMENT_ROOT}/benchmarks/song_real_libero/configs/libero.json}"
PIPELINE_ROOT="${FULL_MOLMO2ER_WORLD_PIPELINE_ROOT:-${EXPERIMENT_ROOT}/outputs/full_molmo2er_worldflow_three_stage}"
CHECKPOINT_066000="${FULL_MOLMO2ER_CHECKPOINT_066000:-${PIPELINE_ROOT}/cumulative_checkpoints/066000}"
CHECKPOINT_096000="${FULL_MOLMO2ER_CHECKPOINT_096000:-${PIPELINE_ROOT}/cumulative_checkpoints/096000}"
EVAL_ROOT="${FULL_MOLMO2ER_EVAL_ROOT:-${PIPELINE_ROOT}/libero_eval_066000_096000}"
LOCK_FILE="${FULL_MOLMO2ER_EVAL_LOCK_FILE:-${EVAL_ROOT}.lock}"
MIN_GPU_FREE_MIB="${FULL_MOLMO2ER_EVAL_MIN_GPU_FREE_MIB:-40000}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
readonly PHYSICAL_CUDA_DEVICES="0,1,2,3,4,5,6,7"

ACTIVE_PIDS=()
SHARD_ROWS=()

fail() {
  printf '[full-molmo2er-eval] %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[full-molmo2er-eval] %s %s\n' "$(date -u +%FT%TZ)" "$*"
}

terminate_workers() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap terminate_workers INT TERM HUP

[[ -x "${PYTHON_BIN}" ]] || fail "Evaluation Python is missing or not executable: ${PYTHON_BIN}"
for required_file in "${TOOLS}" "${EVAL_SCRIPT}" "${EVAL_CONFIG}" "${NVIDIA_EGL_VENDOR_JSON}"; do
  [[ -f "${required_file}" ]] || fail "Missing evaluation dependency: ${required_file}"
done
[[ "${MIN_GPU_FREE_MIB}" =~ ^[0-9]+$ ]] && (( MIN_GPU_FREE_MIB > 0 )) \
  || fail "FULL_MOLMO2ER_EVAL_MIN_GPU_FREE_MIB must be positive."
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "${PHYSICAL_CUDA_DEVICES}" ]]; then
  fail "Evaluation CUDA_VISIBLE_DEVICES must be exactly physical GPUs 0-7."
fi

"${PYTHON_BIN}" "${TOOLS}" checkpoint-audit --checkpoint "${CHECKPOINT_066000}" >/dev/null
"${PYTHON_BIN}" "${TOOLS}" checkpoint-audit --checkpoint "${CHECKPOINT_096000}" >/dev/null
CHECKPOINT_066000="$(cd -- "${CHECKPOINT_066000}" && pwd -P)"
CHECKPOINT_096000="$(cd -- "${CHECKPOINT_096000}" && pwd -P)"
mapfile -t SHARD_ROWS < <("${PYTHON_BIN}" "${TOOLS}" eval-plan)
(( ${#SHARD_ROWS[@]} == 8 )) || fail "Formal static evaluation plan must contain exactly eight shards."

mkdir -p "${EVAL_ROOT}"
if ! command -v flock >/dev/null 2>&1; then
  fail "The evaluator requires flock for duplicate-launch protection."
fi
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  fail "Another 66k/96k LIBERO evaluation holds ${LOCK_FILE}."
fi
printf '%s\n' "$$" > "${EVAL_ROOT}/evaluation.pid"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_CUDA_DEVICES}"
"${PYTHON_BIN}" "${TOOLS}" gpu-audit \
  --min-free-mib "${MIN_GPU_FREE_MIB}" \
  --output "${EVAL_ROOT}/gpu_audit.json" >/dev/null
"${PYTHON_BIN}" "${TOOLS}" write-eval-manifest \
  --output "${EVAL_ROOT}/evaluation_manifest.json" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --checkpoint-066000 "${CHECKPOINT_066000}" \
  --checkpoint-096000 "${CHECKPOINT_096000}" >/dev/null

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EXPERIMENT_ROOT}/src:${EXPERIMENT_ROOT}"
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"

run_shard() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$3"
  local shard_name="$4"
  local physical_gpu="$5"
  local suite_name="$6"
  local task_csv="$7"
  local shard_dir="${output_dir}/shards/${shard_name}"
  local report_path="${shard_dir}/overall_report.json"

  if [[ -s "${report_path}" ]] && "${PYTHON_BIN}" "${TOOLS}" validate-shard \
    --report "${report_path}" --suite "${suite_name}" --task-ids "${task_csv}" --episodes 50; then
    log "preserving audited shard ${label}/${shard_name}"
    return 0
  fi

  mkdir -p "${shard_dir}"
  local task_args=()
  local task_id
  IFS=',' read -r -a task_ids <<< "${task_csv}"
  for task_id in "${task_ids[@]}"; do
    task_args+=(--task-id "${task_id}")
  done
  log "starting ${label}/${shard_name} on physical GPU ${physical_gpu}"
  env \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    MUJOCO_EGL_DEVICE_ID="${physical_gpu}" \
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --config "${EVAL_CONFIG}" \
    --policy.path "${checkpoint}" \
    --suite "${suite_name}" \
    "${task_args[@]}" \
    --episodes 50 \
    --policy-noise-seed 0 \
    --env-seed 7 \
    --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 \
    --task-workers 5 \
    --episode-workers-per-task 4 \
    --inference-batch-size 80 \
    --no-release-event-exec-enable \
    --control-freq 20 \
    --action-index 0 \
    --exec-action-steps 16 \
    --adaptive-exec-max-steps 16 \
    --grasp-exec-steps 16 \
    --max-steps 1000 \
    --no-use-suite-max-steps \
    --recreate-env-per-episode \
    --render-mode offscreen \
    --no-visualize-foreground \
    --no-save-video \
    --output-dir "${shard_dir}" \
    >> "${shard_dir}/launcher.log" 2>&1

  "${PYTHON_BIN}" "${TOOLS}" validate-shard \
    --report "${report_path}" --suite "${suite_name}" --task-ids "${task_csv}" --episodes 50
  log "finished ${label}/${shard_name}"
}

run_gpu_worker() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$3"
  local assigned_gpu="$4"
  local row
  for row in "${SHARD_ROWS[@]}"; do
    local shard_name physical_gpu suite_name task_csv
    IFS=$'\t' read -r shard_name physical_gpu suite_name task_csv <<< "${row}"
    if [[ "${physical_gpu}" == "${assigned_gpu}" ]]; then
      run_shard "${label}" "${checkpoint}" "${output_dir}" \
        "${shard_name}" "${physical_gpu}" "${suite_name}" "${task_csv}"
    fi
  done
}

evaluate_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="${EVAL_ROOT}/${label}"
  local report_path="${output_dir}/overall_report.json"
  local requested_checkpoint_path="${output_dir}/requested_checkpoint.txt"

  mkdir -p "${output_dir}/shards"
  if [[ -s "${requested_checkpoint_path}" ]]; then
    local stored_checkpoint
    stored_checkpoint="$(head -n 1 "${requested_checkpoint_path}")"
    stored_checkpoint="$(cd -- "${stored_checkpoint}" 2>/dev/null && pwd -P)" \
      || fail "Stored checkpoint provenance is invalid: ${requested_checkpoint_path}"
    [[ "${stored_checkpoint}" == "${checkpoint}" ]] \
      || fail "Checkpoint provenance collision for ${output_dir}."
  else
    printf '%s\n' "${checkpoint}" > "${requested_checkpoint_path}"
  fi

  if [[ -s "${report_path}" ]] && "${PYTHON_BIN}" "${TOOLS}" validate-overall --report "${report_path}"; then
    log "${label} already has an audited 40-task/2000-episode report"
    return 0
  fi

  log "launching eight static workers for ${label}"
  ACTIVE_PIDS=()
  local physical_gpu
  for physical_gpu in 0 1 2 3 4 5 6 7; do
    run_gpu_worker "${label}" "${checkpoint}" "${output_dir}" "${physical_gpu}" &
    ACTIVE_PIDS+=("$!")
  done
  local failed=0
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  (( failed == 0 )) || fail "At least one static evaluation worker failed for ${label}."

  local merge_args=()
  local row
  for row in "${SHARD_ROWS[@]}"; do
    local shard_name _physical_gpu _suite_name _task_csv
    IFS=$'\t' read -r shard_name _physical_gpu _suite_name _task_csv <<< "${row}"
    merge_args+=(--shard "${output_dir}/shards/${shard_name}/overall_report.json")
  done
  "${PYTHON_BIN}" "${TOOLS}" merge-eval --output "${report_path}" "${merge_args[@]}"
  "${PYTHON_BIN}" "${TOOLS}" validate-overall --report "${report_path}"
  log "completed ${label}: $("${PYTHON_BIN}" "${TOOLS}" json-get --file "${report_path}" --path overall.success_count)/2000 successes"
}

evaluate_checkpoint checkpoint_066000 "${CHECKPOINT_066000}"
evaluate_checkpoint checkpoint_096000 "${CHECKPOINT_096000}"

"${PYTHON_BIN}" "${TOOLS}" write-comparison \
  --output "${EVAL_ROOT}/comparison.json" \
  --tsv-output "${EVAL_ROOT}/comparison.tsv" \
  --checkpoint-066000 "${CHECKPOINT_066000}" \
  --checkpoint-096000 "${CHECKPOINT_096000}" \
  --report-066000 "${EVAL_ROOT}/checkpoint_066000/overall_report.json" \
  --report-096000 "${EVAL_ROOT}/checkpoint_096000/overall_report.json"
log "both checkpoints are complete: ${EVAL_ROOT}/comparison.json"
