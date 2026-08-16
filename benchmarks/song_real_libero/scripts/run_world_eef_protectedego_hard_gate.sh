#!/usr/bin/env bash
set -euo pipefail

repo=/home/liusong/ProgramFiles/Huggingface/lerobot
runner="$repo/benchmarks/song_real_libero/scripts/run_world_eef_task6_task8_focused.sh"
training=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/world_eef_task6_task8_100ep_protectedego_4gpu_b24_1564steps
experiment=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/libero_setting/world_eef_task6_task8_100ep_protectedego_20260816

if (($# == 0)); then
    echo "usage: $0 STEP [STEP ...]" >&2
    exit 2
fi

compact_progress() {
    local progress=$1
    jq -c '{
        task6: {
            completed: (.tasks["6"].completed_episode_count // 0),
            success: (.tasks["6"].success_count // 0),
            failure: ((.tasks["6"].completed_episode_count // 0) - (.tasks["6"].success_count // 0))
        },
        task8: {
            completed: (.tasks["8"].completed_episode_count // 0),
            success: (.tasks["8"].success_count // 0),
            failure: ((.tasks["8"].completed_episode_count // 0) - (.tasks["8"].success_count // 0))
        }
    }' "$progress"
}

stop_process_group() {
    local leader=$1
    kill -INT -- "-$leader" 2>/dev/null || true
    local attempt
    for attempt in {1..30}; do
        if ! kill -0 "$leader" 2>/dev/null; then
            return
        fi
        sleep 1
    done
    kill -TERM -- "-$leader" 2>/dev/null || true
}

run_step() {
    local step=$1
    local padded
    printf -v padded '%06d' "$step"
    local tag="step${padded}"
    local checkpoint="$training/checkpoints/$padded/pretrained_model"
    local output="$experiment/eval/${tag}_dual_50ep"
    local progress="$output/libero_10/progress.json"
    test -s "$checkpoint/model.safetensors"
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite existing output: $output" >&2
        return 2
    fi

    echo "[hard-gate] starting $tag"
    cd "$repo"
    setsid env \
        WORLD_EEF_TRAINING_DIR="$training" \
        WORLD_EEF_EXPERIMENT_DIR="$experiment" \
        WORLD_EEF_FREEZE_PRETRAINED_EGO=true \
        bash "$runner" eval "$checkpoint" "$tag" 50 false false &
    local leader=$!
    local reason=
    local c6=0 s6=0 f6=0 c8=0 s8=0 f8=0
    while kill -0 "$leader" 2>/dev/null; do
        if [[ -s "$progress" ]]; then
            c6=$(jq -r '.tasks["6"].completed_episode_count // 0' "$progress")
            s6=$(jq -r '.tasks["6"].success_count // 0' "$progress")
            c8=$(jq -r '.tasks["8"].completed_episode_count // 0' "$progress")
            s8=$(jq -r '.tasks["8"].success_count // 0' "$progress")
            f6=$((c6 - s6))
            f8=$((c8 - s8))
            printf '[hard-gate] %s task6=%s/%s (%s fail) task8=%s/%s (%s fail)\n' \
                "$tag" "$s6" "$c6" "$f6" "$s8" "$c8" "$f8"
            if ((f6 >= 4)); then
                reason=task6
                stop_process_group "$leader"
                break
            fi
            if ((c6 == 50 && f8 >= 5)); then
                reason=task8
                stop_process_group "$leader"
                break
            fi
        fi
        sleep 10
    done
    wait "$leader" || true

    if [[ -n "$reason" ]]; then
        local renamed
        if [[ "$reason" == task6 ]]; then
            renamed="$experiment/eval/DISQUALIFIED_PARTIAL_${tag}_dual_task6_${s6}of${c6}_${f6}fail"
        else
            renamed="$experiment/eval/DISQUALIFIED_PARTIAL_${tag}_dual_task6_${s6}of${c6}_task8_${s8}of${c8}_${f8}fail"
        fi
        mv "$output" "$renamed"
        echo "[hard-gate] disqualified $tag at $reason; preserved=$renamed"
        return
    fi

    test -s "$output/summary.json"
    compact_progress "$progress"
    echo "[hard-gate] completed $tag"
}

for step in "$@"; do
    run_step "$step"
done
