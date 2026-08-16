#!/usr/bin/env bash
set -euo pipefail

# Evaluate as many checkpoints concurrently as there are GPUs.  Each evaluator
# loads its checkpoint once, keeps it resident on its assigned GPU, and runs
# task 6 followed by task 8 in the same process.

repo=${WORLD_EEF_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot}
python=${WORLD_EEF_PYTHON:-/home/liusong/anaconda3/envs/reap/bin/python}
training=${WORLD_EEF_TRAINING_DIR:?WORLD_EEF_TRAINING_DIR is required}
experiment=${WORLD_EEF_EXPERIMENT_DIR:?WORLD_EEF_EXPERIMENT_DIR is required}
steps_text=${WORLD_EEF_STEPS:-000260 000520 000780 001040 001300}
gpu_ids_text=${WORLD_EEF_GPU_IDS:-0 1 2 3}
total_episode_workers=${WORLD_EEF_TOTAL_EPISODE_WORKERS:-28}
episodes=${WORLD_EEF_EPISODES:-50}
dry_run=${WORLD_EEF_DRY_RUN:-0}

read -r -a steps <<<"$steps_text"
read -r -a gpu_ids <<<"${gpu_ids_text//,/ }"
if ((${#steps[@]} == 0)); then
    echo "WORLD_EEF_STEPS resolved to an empty list" >&2
    exit 2
fi
if ((${#gpu_ids[@]} == 0)); then
    echo "WORLD_EEF_GPU_IDS resolved to an empty list" >&2
    exit 2
fi
if ((total_episode_workers < 1 || episodes < 1)); then
    echo "worker and episode counts must be positive" >&2
    exit 2
fi

eval_root="$experiment/eval_multicheckpoint"
log_dir="$experiment/logs_multicheckpoint"
mkdir -p "$eval_root" "$log_dir"

child_pids=()
cleanup_children() {
    local pid
    for pid in "${child_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_children INT TERM

run_checkpoint() {
    local step=${1:?step required}
    local gpu_id=${2:?gpu id required}
    local episode_workers=${3:?episode worker count required}
    local checkpoint="$training/checkpoints/$step/pretrained_model"
    local output="$eval_root/step${step}_tasks6_8_${episodes}ep"
    local summary="$output/summary.json"
    local log="$log_dir/eval_step${step}_tasks6_8_${episodes}ep_gpu${gpu_id}.log"

    test -s "$checkpoint/model.safetensors"
    if [[ -s "$summary" ]]; then
        echo "[multi-checkpoint] reuse completed step=$step summary=$summary"
        return
    fi
    if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "incomplete evaluation output exists: $output" >&2
        return 1
    fi

    local -a command=(
        "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py
        --config benchmarks/song_real_libero/configs/libero.json
        --policy.path "$checkpoint"
        --suite libero_10 --task-id 6 --task-id 8 --episodes "$episodes"
        --policy-noise-seed 0 --env-seed 7 --strict-official-init
        --gripper-control-mode delta_width_initial_sync
        --gripper-delta-threshold 0.002
        --gripper-delta-alignment current_minus_previous
        --waypoint-max-hold-steps 1
        --isolated-policy-workers 1 --task-workers 1
        --episode-workers-per-task "$episode_workers" --task-worker-backend process
        --inference-batch-size 80 --inference-batching-mode fixed_barrier
        --no-release-event-exec-enable
        --control-freq 20 --action-index 0
        --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode
        --render-mode offscreen --no-visualize-foreground --no-save-video
        --no-world-to-ego-causal-ablation --output-dir "$output"
    )

    echo "[multi-checkpoint] step=$step gpu=$gpu_id episode_workers=$episode_workers tasks=6,8"
    if [[ "$dry_run" == 1 ]]; then
        printf 'env CUDA_VISIBLE_DEVICES=%q MUJOCO_EGL_DEVICE_ID=%q ' "$gpu_id" "$gpu_id"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi

    mkdir -p "$output"
    cd "$repo"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    PYTHONPATH="$repo/src" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES="$gpu_id" MUJOCO_EGL_DEVICE_ID="$gpu_id" \
    "${command[@]}" 2>&1 | tee -a "$log"
    test -s "$summary"
}

for ((group_start = 0; group_start < ${#steps[@]}; group_start += ${#gpu_ids[@]})); do
    group_size=$((${#steps[@]} - group_start))
    if ((group_size > ${#gpu_ids[@]})); then
        group_size=${#gpu_ids[@]}
    fi
    workers_per_checkpoint=$((total_episode_workers / group_size))
    if ((workers_per_checkpoint < 1)); then
        workers_per_checkpoint=1
    fi

    child_pids=()
    group_steps=()
    for ((slot = 0; slot < group_size; slot++)); do
        step=${steps[group_start + slot]}
        gpu_id=${gpu_ids[slot]}
        group_steps+=("$step")
        run_checkpoint "$step" "$gpu_id" "$workers_per_checkpoint" &
        child_pids+=("$!")
    done

    group_failed=0
    for pid in "${child_pids[@]}"; do
        if ! wait "$pid"; then
            group_failed=1
        fi
    done
    child_pids=()
    if ((group_failed)); then
        echo "checkpoint group failed: ${group_steps[*]}" >&2
        exit 1
    fi
done

if [[ "$dry_run" == 1 ]]; then
    exit 0
fi

"$python" - "$eval_root" "$experiment/checkpoint_selection_multicheckpoint.json" "$episodes" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
expected_episodes = int(sys.argv[3])
baseline = {6: 46, 8: 45}
records = []
for summary_path in sorted(root.glob("step*_tasks6_8_*ep/summary.json")):
    summary = json.loads(summary_path.read_text())
    tag = summary_path.parent.name.split("_tasks", 1)[0]
    tasks = {}
    for task_id in (6, 8):
        matches = [
            item for item in summary["results"]
            if item["suite"] == "libero_10" and int(item["task_id"]) == task_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one task {task_id} record in {summary_path}")
        episodes = matches[0]["episodes"]
        if len(episodes) != expected_episodes:
            raise RuntimeError(
                f"Expected {expected_episodes} task {task_id} episodes in {summary_path}, "
                f"got {len(episodes)}"
            )
        failures = sorted(
            int(episode["episode_index"])
            for episode in episodes
            if not bool(episode["success"])
        )
        successes = expected_episodes - len(failures)
        tasks[str(task_id)] = {
            "successes": successes,
            "baseline_successes": baseline[task_id],
            "delta": successes - baseline[task_id],
            "failure_indices": failures,
        }
    qualified = all(tasks[str(task_id)]["successes"] > baseline[task_id] for task_id in (6, 8))
    total = sum(item["successes"] for item in tasks.values())
    records.append({
        "tag": tag,
        "tasks": tasks,
        "total_successes": total,
        "qualified_both_tasks_improve": qualified,
        "strong_gate": qualified and tasks["6"]["successes"] >= 48
        and tasks["8"]["successes"] >= 48 and total >= 96,
        "summary": str(summary_path),
    })
records.sort(
    key=lambda item: (item["qualified_both_tasks_improve"], item["total_successes"], item["tag"]),
    reverse=True,
)
report = {
    "baseline": {"task6": 46, "task8": 45, "total": 91},
    "execution": "up to four GPU-resident checkpoints concurrently; tasks 6 and 8 share one load",
    "qualified_tags": [item["tag"] for item in records if item["qualified_both_tasks_improve"]],
    "strong_tags": [item["tag"] for item in records if item["strong_gate"]],
    "ranked_results": records,
}
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
