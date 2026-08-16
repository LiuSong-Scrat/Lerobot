#!/usr/bin/env bash
set -euo pipefail

repo=/home/liusong/ProgramFiles/Huggingface/lerobot
python=/home/liusong/anaconda3/envs/reap/bin/python
training=${WORLD_EEF_TRAINING_DIR:?WORLD_EEF_TRAINING_DIR is required}
experiment=${WORLD_EEF_EXPERIMENT_DIR:?WORLD_EEF_EXPERIMENT_DIR is required}
log_dir="$experiment/logs"
eval_root="$experiment/eval"
mkdir -p "$log_dir" "$eval_root"

task_successes() {
    "$python" - "$1" "$2" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
task_id = int(sys.argv[2])
records = [
    record for record in summary["results"]
    if record["suite"] == "libero_10" and int(record["task_id"]) == task_id
]
if len(records) != 1:
    raise RuntimeError(f"Expected one task {task_id} record, got {len(records)}")
episodes = records[0]["episodes"]
if len(episodes) != 50:
    raise RuntimeError(f"Expected 50 task {task_id} episodes, got {len(episodes)}")
print(sum(bool(episode["success"]) for episode in episodes))
PY
}

eval_task() {
    local step=${1:?step required}
    local task_id=${2:?task id required}
    local checkpoint="$training/checkpoints/$step/pretrained_model"
    local output="$eval_root/step${step}_task${task_id}_50ep"
    local summary="$output/summary.json"
    test -s "$checkpoint/model.safetensors"
    if [[ -s "$summary" ]]; then
        task_successes "$summary" "$task_id"
        return
    fi
    if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "incomplete evaluation output exists: $output" >&2
        exit 1
    fi
    mkdir -p "$output"
    cd "$repo"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    PYTHONPATH="$repo/src" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
        --config benchmarks/song_real_libero/configs/libero.json \
        --policy.path "$checkpoint" \
        --suite libero_10 --task-id "$task_id" --episodes 50 \
        --policy-noise-seed 0 --env-seed 7 --strict-official-init \
        --gripper-control-mode delta_width_initial_sync \
        --gripper-delta-threshold 0.002 \
        --gripper-delta-alignment current_minus_previous \
        --waypoint-max-hold-steps 1 \
        --isolated-policy-workers 1 --task-workers 1 \
        --episode-workers-per-task 28 --task-worker-backend process \
        --inference-batch-size 80 --inference-batching-mode fixed_barrier \
        --no-release-event-exec-enable \
        --control-freq 20 --action-index 0 \
        --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
        --render-mode offscreen --no-visualize-foreground --no-save-video \
        --no-world-to-ego-causal-ablation --output-dir "$output" \
        2>&1 | tee -a "$log_dir/eval_step${step}_task${task_id}_50ep.log" >&2
    test -s "$summary"
    task_successes "$summary" "$task_id"
}

steps=(000260 000520 000780 001040 001300)
for step in "${steps[@]}"; do
    task6_successes=$(eval_task "$step" 6)
    echo "[staged-screen] step=$step task6=$task6_successes/50"
    if ((task6_successes <= 46)); then
        echo "[staged-screen] step=$step rejected before task8 (baseline task6=46/50)"
        continue
    fi
    task8_successes=$(eval_task "$step" 8)
    echo "[staged-screen] step=$step task8=$task8_successes/50"
done

"$python" - "$eval_root" "$experiment/checkpoint_selection.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
baseline = {6: 46, 8: 45}
records = []
for task6_summary in sorted(root.glob("step*_task6_50ep/summary.json")):
    tag = task6_summary.parent.name.split("_task", 1)[0]
    tasks = {}
    for task_id in (6, 8):
        path = root / f"{tag}_task{task_id}_50ep" / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        result = next(
            item for item in summary["results"]
            if item["suite"] == "libero_10" and int(item["task_id"]) == task_id
        )
        policy_failures = sorted(
            int(episode["episode_index"])
            for episode in result["episodes"]
            if not bool(episode["success"]) and not episode.get("error")
        )
        cancelled = sorted(
            int(episode["episode_index"])
            for episode in result["episodes"]
            if not bool(episode["success"]) and episode.get("error")
        )
        observed_successes = sum(
            bool(episode["success"]) for episode in result["episodes"]
        )
        completed_rollouts = len(result["episodes"]) - len(cancelled)
        maximum_possible_successes = observed_successes + len(cancelled)
        tasks[str(task_id)] = {
            "status": "complete" if not cancelled else "stopped_early",
            "successes": observed_successes if not cancelled else None,
            "observed_successes": observed_successes,
            "completed_rollouts": completed_rollouts,
            "maximum_possible_successes": maximum_possible_successes,
            "baseline_successes": baseline[task_id],
            "delta": observed_successes - baseline[task_id] if not cancelled else None,
            "failure_indices": policy_failures,
            "policy_failure_indices": policy_failures,
            "cancelled_episode_indices": cancelled,
            "summary": str(path),
        }
    qualified = set(tasks) == {"6", "8"} and all(
        tasks[str(task_id)]["status"] == "complete"
        and tasks[str(task_id)]["successes"] > baseline[task_id]
        for task_id in (6, 8)
    )
    total = sum(item["observed_successes"] for item in tasks.values())
    records.append({
        "tag": tag,
        "tasks": tasks,
        "evaluated_total_successes": total,
        "qualified_both_tasks_improve": qualified,
        "strong_gate": qualified and tasks["6"]["successes"] >= 48
        and tasks["8"]["successes"] >= 48 and total >= 96,
    })
records.sort(
    key=lambda item: (
        item["qualified_both_tasks_improve"],
        item["evaluated_total_successes"],
        item["tag"],
    ),
    reverse=True,
)
report = {
    "baseline": {"task6": 46, "task8": 45, "total": 91},
    "screen_order": "task6 first; task8 only when task6 > 46/50",
    "same_task_episode_workers": 28,
    "qualified_tags": [item["tag"] for item in records if item["qualified_both_tasks_improve"]],
    "strong_tags": [item["tag"] for item in records if item["strong_gate"]],
    "ranked_results": records,
}
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
