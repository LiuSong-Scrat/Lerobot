#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
PYTHON_BIN="${EXPERIMENT_ROOT}/.venv-smol5090/bin/python"
EVAL_SCRIPT="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py"
EVAL_CONFIG="${EXPERIMENT_ROOT}/benchmarks/song_real_libero/configs/libero.json"
EVAL_ROOT="${MOLMO_THREE_CHECKPOINT_EVAL_ROOT:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_libero_eval_three_checkpoints_v043_protocol}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
CHECKPOINT_036000="${MOLMO_CHECKPOINT_036000:?Set MOLMO_CHECKPOINT_036000}"
CHECKPOINT_066000="${MOLMO_CHECKPOINT_066000:?Set MOLMO_CHECKPOINT_066000}"
CHECKPOINT_096000="${MOLMO_CHECKPOINT_096000:?Set MOLMO_CHECKPOINT_096000}"

labels=(checkpoint_036000 checkpoint_066000 checkpoint_096000)
checkpoints=("${CHECKPOINT_036000}" "${CHECKPOINT_066000}" "${CHECKPOINT_096000}")
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
shard_names=(
  gpu0_libero_spatial_tasks0-4 gpu1_libero_spatial_tasks5-9
  gpu2_libero_object_tasks0-4 gpu3_libero_object_tasks5-9
  gpu4_libero_goal_tasks0-4 gpu5_libero_goal_tasks5-9
  gpu6_libero_10_tasks0-4 gpu7_libero_10_tasks5-9
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing evaluation Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${NVIDIA_EGL_VENDOR_JSON}" ]]; then
  echo "Missing NVIDIA EGL vendor manifest: ${NVIDIA_EGL_VENDOR_JSON}" >&2
  exit 2
fi
for checkpoint in "${checkpoints[@]}"; do
  for filename in "${required_checkpoint_files[@]}"; do
    if [[ ! -f "${checkpoint}/${filename}" ]]; then
      echo "Missing checkpoint file: ${checkpoint}/${filename}" >&2
      exit 2
    fi
  done
done

gpu_is_busy() {
  local gpu_id="$1"
  nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

for gpu_id in "${gpu_ids[@]}"; do
  if gpu_is_busy "${gpu_id}"; then
    echo "GPU ${gpu_id} is busy; refusing to overlap the controlled LIBERO evaluation." >&2
    exit 75
  fi
done
sleep 10
for gpu_id in "${gpu_ids[@]}"; do
  if gpu_is_busy "${gpu_id}"; then
    echo "GPU ${gpu_id} became busy during idle confirmation." >&2
    exit 75
  fi
done

mkdir -p "${EVAL_ROOT}"
printf 'label\tcheckpoint\tepisodes\tsuccesses\tsuccess_rate\n' > "${EVAL_ROOT}/comparison.tsv"
printf '500m_v043_reference\t%s\t2000\t1941\t0.970500000\n' \
  '/raid5/rongshengwang/Lerobot/eval_FULL4-9705*' >> "${EVAL_ROOT}/comparison.tsv"

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

for gpu_id in "${gpu_ids[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" "${PYTHON_BIN}" - <<'PY'
from mujoco.egl import GLContext

context = GLContext(64, 64)
context.make_current()
context.free()
PY
done
echo "[eval] NVIDIA EGL preflight passed on GPUs ${gpu_ids[*]}"

report_is_valid() {
  local report_path="$1"
  local expected_episodes="$2"
  jq -e --argjson expected_episodes "${expected_episodes}" '
    .overall.episode_count == $expected_episodes
    and ([.suites[].tasks[].episodes[]] | length == $expected_episodes)
    and (
      [.suites[].tasks[].episodes[]
       | select(
           (.error? != null)
           or ((.steps // 0) <= 0)
           or ((.model_call_count // 0) <= 0)
           or ((.policy_forward_call_count // 0) <= 0)
         )]
      | length == 0
    )
  ' "${report_path}" >/dev/null
}

for index in "${!labels[@]}"; do
  label="${labels[index]}"
  checkpoint="${checkpoints[index]}"
  output_dir="${EVAL_ROOT}/${label}"
  report_path="${output_dir}/overall_report.json"

  if [[ -f "${report_path}" ]] && report_is_valid "${report_path}" 2000; then
    echo "[eval] $(date -u +%FT%TZ) ${label} already complete; preserving existing result"
  else
    if [[ -e "${output_dir}" ]]; then
      echo "Refusing to overwrite incomplete evaluation directory: ${output_dir}" >&2
      exit 2
    fi
    mkdir -p "${output_dir}"
    printf '%s\n' "${checkpoint}" > "${output_dir}/requested_checkpoint.txt"
    echo "[eval] $(date -u +%FT%TZ) starting ${label}: ${checkpoint}"
    mkdir -p "${output_dir}/shards"
    pids=()
    shard_reports=()
    for shard_index in "${!gpu_ids[@]}"; do
      gpu_id="${gpu_ids[shard_index]}"
      suite_name="${suite_names[shard_index]}"
      shard_name="${shard_names[shard_index]}"
      shard_dir="${output_dir}/shards/${shard_name}"
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
      pids+=("$!")
      shard_reports+=("${shard_dir}/overall_report.json")
      echo "[eval] launched ${label}/${shard_name} pid=${pids[-1]} GPU=${gpu_id}"
    done

    shard_failed=0
    for shard_index in "${!pids[@]}"; do
      if wait "${pids[shard_index]}"; then
        echo "[eval] finished ${label}/${shard_names[shard_index]}"
      else
        return_code="$?"
        echo "[eval] failed ${label}/${shard_names[shard_index]} rc=${return_code}" >&2
        shard_failed=1
      fi
    done
    if (( shard_failed != 0 )); then
      exit 2
    fi
    for shard_report in "${shard_reports[@]}"; do
      if ! report_is_valid "${shard_report}" 250; then
        echo "Invalid shard report: ${shard_report}" >&2
        exit 2
      fi
    done

    "${PYTHON_BIN}" - "${report_path}" "${shard_reports[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
reports = []
for report_path in sys.argv[2:]:
    with open(report_path, encoding="utf-8") as report_file:
        reports.append(json.load(report_file))

tasks = [
    task
    for report in reports
    for suite in report["suites"]
    for task in suite["tasks"]
]
suite_order = {name: index for index, name in enumerate(
    ("libero_spatial", "libero_object", "libero_goal", "libero_10")
)}
tasks.sort(key=lambda task: (suite_order[task["suite"]], int(task["task_id"])))
task_keys = [(task["suite"], int(task["task_id"])) for task in tasks]
expected_keys = [(suite, task_id) for suite in suite_order for task_id in range(10)]
if task_keys != expected_keys:
    raise RuntimeError(f"Unexpected merged task keys: {task_keys}")

episode_count = sum(int(task["episode_count"]) for task in tasks)
success_count = sum(
    sum(bool(episode.get("success", False)) for episode in task["episodes"])
    for task in tasks
)
if episode_count != 2000:
    raise RuntimeError(f"Expected 2000 merged episodes, got {episode_count}")
merged_suites = []
for suite_name in suite_order:
    suite_tasks = [task for task in tasks if task["suite"] == suite_name]
    suite_episode_count = sum(int(task["episode_count"]) for task in suite_tasks)
    suite_success_count = sum(
        sum(bool(episode.get("success", False)) for episode in task["episodes"])
        for task in suite_tasks
    )
    merged_suites.append({
        "suite": suite_name,
        "task_count": len(suite_tasks),
        "episode_count": suite_episode_count,
        "success_count": suite_success_count,
        "success_rate": suite_success_count / suite_episode_count,
        "task_success_rate_mean": (
            sum(float(task["success_rate"]) for task in suite_tasks) / len(suite_tasks)
        ),
        "tasks": suite_tasks,
    })
merged = {
    "overall": {
        "task_count": len(tasks),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count,
        "task_success_rate_mean": sum(float(task["success_rate"]) for task in tasks) / len(tasks),
    },
    "suites": merged_suites,
}
temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
with open(temporary_path, "w", encoding="utf-8") as output_file:
    json.dump(merged, output_file, indent=2, ensure_ascii=False)
    output_file.write("\n")
os.replace(temporary_path, output_path)
PY
  fi

  episodes="$(jq -r '.overall.episode_count' "${report_path}")"
  successes="$(jq -r '.overall.success_count' "${report_path}")"
  success_rate="$(jq -r '.overall.success_rate' "${report_path}")"
  if [[ "${episodes}" != "2000" ]]; then
    echo "Incomplete ${label} evaluation: ${episodes}/2000 episodes" >&2
    exit 2
  fi
  if ! report_is_valid "${report_path}" 2000; then
    echo "Invalid ${label} evaluation: one or more episodes have an error, zero steps, or no model call" >&2
    exit 2
  fi
  printf '%s\t%s\t%s\t%s\t%.9f\n' \
    "${label}" "${checkpoint}" "${episodes}" "${successes}" "${success_rate}" \
    >> "${EVAL_ROOT}/comparison.tsv"
  echo "[eval] $(date -u +%FT%TZ) completed ${label}: ${successes}/${episodes} (${success_rate})"
done

jq -n \
  --arg protocol '500M 97.05% protocol: four suites, 40 tasks x 50 episodes; eight GPU shards, five tasks/shard, four episode workers/task' \
  --arg reference_path '/raid5/rongshengwang/Lerobot/eval_FULL4-9705*' \
  --slurpfile ckpt036 "${EVAL_ROOT}/checkpoint_036000/overall_report.json" \
  --slurpfile ckpt066 "${EVAL_ROOT}/checkpoint_066000/overall_report.json" \
  --slurpfile ckpt096 "${EVAL_ROOT}/checkpoint_096000/overall_report.json" \
  '{
    protocol: $protocol,
    reference_500m: {path: $reference_path, episode_count: 2000, success_count: 1941, success_rate: 0.9705},
    checkpoint_036000: $ckpt036[0].overall,
    checkpoint_066000: $ckpt066[0].overall,
    checkpoint_096000: $ckpt096[0].overall
  }' > "${EVAL_ROOT}/comparison.json"

echo "[eval] all three checkpoints complete: ${EVAL_ROOT}/comparison.json"
