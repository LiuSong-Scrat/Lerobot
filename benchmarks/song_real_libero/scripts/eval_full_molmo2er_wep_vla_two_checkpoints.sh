#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
PYTHON_BIN="${EXPERIMENT_ROOT}/.venv-smol5090/bin/python"
EVAL_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py"
EVAL_CONFIG="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/configs/libero.json"
PIPELINE_ROOT="${EXPERIMENT_ROOT}/outputs/full_molmo2er_wep_vla_three_stage"
EVAL_ROOT="${FULL_MOLMO2ER_EVAL_ROOT:-${PIPELINE_ROOT}/libero_eval_066000_096000_v043_protocol}"
CHECKPOINT_066000="${FULL_MOLMO2ER_CHECKPOINT_066000:-${PIPELINE_ROOT}/stage2_from036000_to066000/checkpoints/030000/pretrained_model}"
CHECKPOINT_096000="${FULL_MOLMO2ER_CHECKPOINT_096000:-${PIPELINE_ROOT}/stage3_from066000_to096000/checkpoints/030000/pretrained_model}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
MIN_GPU_FREE_MIB="${FULL_MOLMO2ER_EVAL_MIN_GPU_FREE_MIB:-20480}"
LOCK_FILE="${FULL_MOLMO2ER_EVAL_LOCK_FILE:-${EVAL_ROOT}/evaluation.lock}"

labels=(checkpoint_066000 checkpoint_096000)
checkpoints=("${CHECKPOINT_066000}" "${CHECKPOINT_096000}")
required_checkpoint_files=(config.json model.safetensors policy_preprocessor.json policy_postprocessor.json)
gpu_ids=(0 1 2 3 4 5 6 7)
suite_names=(
  libero_spatial libero_spatial
  libero_object libero_object
  libero_goal libero_goal
  libero_10 libero_10
)
task_groups=(
  "0 1 2 3 4" "5 6 7 8 9"
  "0 1 2 3 4" "5 6 7 8 9"
  "0 1 2 3 4" "5 6 7 8 9"
  "0 1 2 3 4" "5 6 7 8 9"
)
task_starts=(0 5 0 5 0 5 0 5)
shard_names=(
  gpu0_libero_spatial_tasks0-4 gpu1_libero_spatial_tasks5-9
  gpu2_libero_object_tasks0-4 gpu3_libero_object_tasks5-9
  gpu4_libero_goal_tasks0-4 gpu5_libero_goal_tasks5-9
  gpu6_libero_10_tasks0-4 gpu7_libero_10_tasks5-9
)
active_pids=()

log() {
  printf '[eval] %s %s\n' "$(date -u +%FT%TZ)" "$*"
}

terminate_children() {
  local pid
  for pid in "${active_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap terminate_children INT TERM HUP

for command_name in jq nvidia-smi flock realpath; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing evaluation Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_SCRIPT}" || ! -f "${EVAL_CONFIG}" ]]; then
  echo "Missing LIBERO evaluator or configuration." >&2
  exit 2
fi
if [[ ! -f "${NVIDIA_EGL_VENDOR_JSON}" ]]; then
  echo "Missing NVIDIA EGL vendor manifest: ${NVIDIA_EGL_VENDOR_JSON}" >&2
  exit 2
fi
if [[ ! "${MIN_GPU_FREE_MIB}" =~ ^[0-9]+$ ]] || (( MIN_GPU_FREE_MIB < 20480 )); then
  echo "FULL_MOLMO2ER_EVAL_MIN_GPU_FREE_MIB must be an integer of at least 20480." >&2
  exit 2
fi
for checkpoint in "${checkpoints[@]}"; do
  for filename in "${required_checkpoint_files[@]}"; do
    if [[ ! -s "${checkpoint}/${filename}" ]]; then
      echo "Missing or empty checkpoint file: ${checkpoint}/${filename}" >&2
      exit 2
    fi
  done
done

mkdir -p "${EVAL_ROOT}"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  echo "Another Full-Molmo2-ER evaluation holds ${LOCK_FILE}." >&2
  exit 73
fi
printf '%s\n' "$$" > "${EVAL_ROOT}/evaluation.pid"

check_gpu_memory() {
  local phase="$1"
  local gpu_index
  local -a free_mib
  mapfile -t free_mib < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ "${#free_mib[@]}" != "${#gpu_ids[@]}" ]]; then
    echo "Expected exactly ${#gpu_ids[@]} physical GPUs, found ${#free_mib[@]}." >&2
    return 2
  fi
  for gpu_index in "${!gpu_ids[@]}"; do
    if [[ ! "${free_mib[gpu_index]}" =~ ^[0-9]+$ ]]; then
      echo "Could not read free memory for GPU ${gpu_ids[gpu_index]}." >&2
      return 2
    fi
    if (( free_mib[gpu_index] < MIN_GPU_FREE_MIB )); then
      echo "GPU ${gpu_ids[gpu_index]} has ${free_mib[gpu_index]} MiB free during ${phase}; need at least ${MIN_GPU_FREE_MIB} MiB. Existing light GPU processes are allowed, but this memory floor is strict." >&2
      return 75
    fi
  done
  log "${phase} GPU memory gate passed (per-GPU free MiB: ${free_mib[*]}; minimum=${MIN_GPU_FREE_MIB})"
}

shard_report_is_valid() {
  local report_path="$1"
  local expected_suite="$2"
  local expected_start="$3"
  jq -e \
    --arg expected_suite "${expected_suite}" \
    --argjson expected_start "${expected_start}" '
      def episodes: [.suites[].tasks[].episodes[]];
      (.overall.task_count == 5)
      and (.overall.episode_count == 250)
      and (.suites | length == 1)
      and (.suites[0].suite == $expected_suite)
      and (.suites[0].task_count == 5)
      and (.suites[0].episode_count == 250)
      and (([.suites[0].tasks[].task_id] | sort) == [range($expected_start; $expected_start + 5)])
      and (all(.suites[0].tasks[];
        .suite == $expected_suite
        and .episode_count == 50
        and (.episodes | length == 50)
        and (([.episodes[].episode_index] | sort) == [range(0; 50)])
      ))
      and ((episodes | length) == 250)
      and (all(episodes[];
        (.error? == null)
        and ((.success | type) == "boolean")
        and ((.steps // 0) > 0)
        and ((.model_call_count // 0) > 0)
        and ((.policy_forward_call_count // 0) > 0)
        and (.strict_official_init == true)
        and (.gripper_control_mode == "delta_width_initial_sync")
        and (.gripper_delta_threshold == 0.002)
        and (.gripper_delta_alignment == "current_minus_previous")
        and (.waypoint_max_hold_steps == 1)
        and (.adaptive_exec_max_steps == 24)
        and (.grasp_exec_steps == 24)
        and (.release_event_exec_enable == false)
        and (.max_steps == 1000)
        and (.evaluation_protocol.benchmark_comparable == true)
      ))
      and (.overall.success_count == ([episodes[] | select(.success == true)] | length))
    ' "${report_path}" >/dev/null 2>&1
}

overall_report_is_valid() {
  local report_path="$1"
  jq -e '
      def episodes: [.suites[].tasks[].episodes[]];
      ["libero_spatial", "libero_object", "libero_goal", "libero_10"] as $suite_order
      | (.overall.task_count == 40)
      and (.overall.episode_count == 2000)
      and ((.suites | length) == 4)
      and (([.suites[].suite]) == $suite_order)
      and (([.suites[] | .suite as $suite | .tasks[] | "\($suite):\(.task_id)"])
        == [$suite_order[] as $suite | range(0; 10) as $task_id | "\($suite):\($task_id)"])
      and (all(.suites[];
        .task_count == 10
        and .episode_count == 500
        and (.tasks | length == 10)
      ))
      and (all(.suites[].tasks[];
        .episode_count == 50
        and (.episodes | length == 50)
        and (([.episodes[].episode_index] | sort) == [range(0; 50)])
      ))
      and ((episodes | length) == 2000)
      and (all(episodes[];
        (.error? == null)
        and ((.success | type) == "boolean")
        and ((.steps // 0) > 0)
        and ((.model_call_count // 0) > 0)
        and ((.policy_forward_call_count // 0) > 0)
        and (.strict_official_init == true)
        and (.gripper_control_mode == "delta_width_initial_sync")
        and (.gripper_delta_threshold == 0.002)
        and (.gripper_delta_alignment == "current_minus_previous")
        and (.waypoint_max_hold_steps == 1)
        and (.adaptive_exec_max_steps == 24)
        and (.grasp_exec_steps == 24)
        and (.release_event_exec_enable == false)
        and (.max_steps == 1000)
        and (.evaluation_protocol.benchmark_comparable == true)
      ))
      and (.overall.success_count == ([episodes[] | select(.success == true)] | length))
    ' "${report_path}" >/dev/null 2>&1
}

merge_shard_reports() {
  local output_path="$1"
  shift
  local temporary_path="${output_path}.tmp.$$"
  jq -s '
      ["libero_spatial", "libero_object", "libero_goal", "libero_10"] as $suite_order
      | [.[] | .suites[].tasks[]] as $all_tasks
      | [
          $suite_order[] as $suite_name
          | ($all_tasks | map(select(.suite == $suite_name)) | sort_by(.task_id)) as $tasks
          | ([$tasks[].episodes[]] | length) as $episode_count
          | ([$tasks[].episodes[] | select(.success == true)] | length) as $success_count
          | {
              suite: $suite_name,
              task_count: ($tasks | length),
              episode_count: $episode_count,
              success_count: $success_count,
              success_rate: ($success_count / $episode_count),
              task_success_rate_mean: ([$tasks[].success_rate] | add / length),
              tasks: $tasks
            }
        ] as $suites
      | ([$suites[].tasks[]]) as $tasks
      | ([$tasks[].episodes[]] | length) as $episode_count
      | ([$tasks[].episodes[] | select(.success == true)] | length) as $success_count
      | {
          overall: {
            task_count: ($tasks | length),
            episode_count: $episode_count,
            success_count: $success_count,
            success_rate: ($success_count / $episode_count),
            task_success_rate_mean: ([$tasks[].success_rate] | add / length)
          },
          suites: $suites
        }
    ' "$@" > "${temporary_path}"
  if ! overall_report_is_valid "${temporary_path}"; then
    echo "Merged report failed strict 40-task/2000-episode audit: ${temporary_path}" >&2
    return 2
  fi
  mv -f "${temporary_path}" "${output_path}"
}

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

egl_preflight_done=0
for checkpoint_index in "${!labels[@]}"; do
  label="${labels[checkpoint_index]}"
  checkpoint="$(realpath -e "${checkpoints[checkpoint_index]}")"
  output_dir="${EVAL_ROOT}/${label}"
  report_path="${output_dir}/overall_report.json"
  requested_checkpoint_path="${output_dir}/requested_checkpoint.txt"

  if [[ -d "${output_dir}" ]]; then
    if [[ ! -s "${requested_checkpoint_path}" ]]; then
      echo "Existing evaluation directory lacks checkpoint provenance: ${output_dir}" >&2
      exit 2
    fi
    stored_checkpoint="$(realpath -e "$(head -n 1 "${requested_checkpoint_path}")" 2>/dev/null || true)"
    if [[ "${stored_checkpoint}" != "${checkpoint}" ]]; then
      echo "Checkpoint provenance mismatch in ${output_dir}: stored=${stored_checkpoint:-<invalid>} requested=${checkpoint}" >&2
      exit 2
    fi
  else
    mkdir -p "${output_dir}/shards"
    printf '%s\n' "${checkpoint}" > "${requested_checkpoint_path}"
  fi

  if [[ -f "${report_path}" ]]; then
    if overall_report_is_valid "${report_path}"; then
      log "${label} already complete and strictly audited; preserving it"
      continue
    fi
    echo "Existing aggregate report is invalid; refusing to overwrite it: ${report_path}" >&2
    exit 2
  fi

  check_gpu_memory "before ${label}"
  sleep 10
  check_gpu_memory "${label} 10-second confirmation"
  if (( egl_preflight_done == 0 )); then
    for gpu_id in "${gpu_ids[@]}"; do
      CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" "${PYTHON_BIN}" - <<'PY'
from mujoco.egl import GLContext

context = GLContext(64, 64)
context.make_current()
context.free()
PY
    done
    egl_preflight_done=1
    log "NVIDIA EGL preflight passed on GPUs ${gpu_ids[*]}"
  fi

  log "starting ${label}: ${checkpoint}"
  pids=()
  shard_reports=()
  launched_shard_indices=()
  for shard_index in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[shard_index]}"
    suite_name="${suite_names[shard_index]}"
    shard_name="${shard_names[shard_index]}"
    shard_dir="${output_dir}/shards/${shard_name}"
    shard_report="${shard_dir}/overall_report.json"
    shard_reports+=("${shard_report}")

    if [[ -f "${shard_report}" ]] && shard_report_is_valid \
      "${shard_report}" "${suite_name}" "${task_starts[shard_index]}"; then
      log "preserving audited shard ${label}/${shard_name}"
      continue
    fi
    if [[ -e "${shard_dir}" ]]; then
      echo "Incomplete or invalid shard directory requires manual inspection: ${shard_dir}" >&2
      exit 2
    fi

    task_args=()
    for task_id in ${task_groups[shard_index]}; do
      task_args+=(--task-id "${task_id}")
    done
    mkdir -p "${shard_dir}"
    env CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
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
      --exec-action-steps 24 \
      --adaptive-exec-max-steps 24 \
      --grasp-exec-steps 24 \
      --max-steps 1000 \
      --no-use-suite-max-steps \
      --recreate-env-per-episode \
      --render-mode offscreen \
      --no-visualize-foreground \
      --save-video \
      --output-dir "${shard_dir}" \
      > "${shard_dir}/launcher.log" 2>&1 &
    pid="$!"
    pids+=("${pid}")
    active_pids+=("${pid}")
    launched_shard_indices+=("${shard_index}")
    log "launched ${label}/${shard_name} pid=${pid} GPU=${gpu_id}"
  done

  shard_failed=0
  for launched_index in "${!pids[@]}"; do
    shard_index="${launched_shard_indices[launched_index]}"
    if wait "${pids[launched_index]}"; then
      log "finished ${label}/${shard_names[shard_index]}"
    else
      return_code="$?"
      echo "[eval] failed ${label}/${shard_names[shard_index]} rc=${return_code}" >&2
      shard_failed=1
    fi
  done
  active_pids=()
  if (( shard_failed != 0 )); then
    exit 2
  fi

  for shard_index in "${!shard_reports[@]}"; do
    if ! shard_report_is_valid "${shard_reports[shard_index]}" \
      "${suite_names[shard_index]}" "${task_starts[shard_index]}"; then
      echo "Shard failed strict report audit: ${shard_reports[shard_index]}" >&2
      exit 2
    fi
  done
  merge_shard_reports "${report_path}" "${shard_reports[@]}"
  successes="$(jq -r '.overall.success_count' "${report_path}")"
  log "completed ${label}: ${successes}/2000"
done

for checkpoint_index in "${!labels[@]}"; do
  if ! overall_report_is_valid "${EVAL_ROOT}/${labels[checkpoint_index]}/overall_report.json"; then
    echo "Final report audit failed for ${labels[checkpoint_index]}." >&2
    exit 2
  fi
done

comparison_tmp="${EVAL_ROOT}/comparison.json.tmp.$$"
jq -n \
  --arg generated_at "$(date -u +%FT%TZ)" \
  --arg checkpoint_066000 "$(realpath -e "${CHECKPOINT_066000}")" \
  --arg checkpoint_096000 "$(realpath -e "${CHECKPOINT_096000}")" \
  --arg reference_path '/raid5/rongshengwang/Lerobot/eval_FULL4-9705*' \
  --slurpfile report_066000 "${EVAL_ROOT}/checkpoint_066000/overall_report.json" \
  --slurpfile report_096000 "${EVAL_ROOT}/checkpoint_096000/overall_report.json" '
    {
      generated_at: $generated_at,
      protocol: {
        reference: "500M v0.4.3 97.05% protocol",
        suites: ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        task_count: 40,
        episodes_per_task: 50,
        episode_count_per_checkpoint: 2000,
        gpu_shards: 8,
        tasks_per_shard: 5,
        episode_workers_per_task: 4,
        inference_batch_size: 80,
        exec_action_steps: 24,
        adaptive_exec_max_steps: 24,
        grasp_exec_steps: 24,
        strict_official_init: true
      },
      reference_500m: {
        path: $reference_path,
        episode_count: 2000,
        success_count: 1941,
        success_rate: 0.9705
      },
      checkpoint_066000: ($report_066000[0].overall + {path: $checkpoint_066000}),
      checkpoint_096000: ($report_096000[0].overall + {path: $checkpoint_096000})
    }
  ' > "${comparison_tmp}"
jq -e '
  (.checkpoint_066000.episode_count == 2000)
  and (.checkpoint_096000.episode_count == 2000)
  and (.protocol.exec_action_steps == 24)
  and (.protocol.adaptive_exec_max_steps == 24)
  and (.protocol.grasp_exec_steps == 24)
' "${comparison_tmp}" >/dev/null
mv -f "${comparison_tmp}" "${EVAL_ROOT}/comparison.json"

comparison_tsv_tmp="${EVAL_ROOT}/comparison.tsv.tmp.$$"
printf 'label\tcheckpoint\tepisodes\tsuccesses\tsuccess_rate\n' > "${comparison_tsv_tmp}"
printf '500m_v043_reference\t%s\t2000\t1941\t0.970500000\n' \
  '/raid5/rongshengwang/Lerobot/eval_FULL4-9705*' >> "${comparison_tsv_tmp}"
for checkpoint_index in "${!labels[@]}"; do
  label="${labels[checkpoint_index]}"
  report_path="${EVAL_ROOT}/${label}/overall_report.json"
  printf '%s\t%s\t%s\t%s\t%.9f\n' \
    "${label}" \
    "$(realpath -e "${checkpoints[checkpoint_index]}")" \
    "$(jq -r '.overall.episode_count' "${report_path}")" \
    "$(jq -r '.overall.success_count' "${report_path}")" \
    "$(jq -r '.overall.success_rate' "${report_path}")" \
    >> "${comparison_tsv_tmp}"
done
mv -f "${comparison_tsv_tmp}" "${EVAL_ROOT}/comparison.tsv"

log "both checkpoints complete and strictly audited: ${EVAL_ROOT}/comparison.json"
