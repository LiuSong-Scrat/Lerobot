close_box：9
close_fridge：9
close_laptop_lid：9
umbrella：9
toilet：9
stack_wine：9
sweep：9
take_frame_off_hanger：7



water_plants：2
phone_on_base：4




COPPELIASIM_ROOT="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/CoppeliaSim" \
LD_LIBRARY_PATH="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM=xcb \
QT_QPA_PLATFORM_PLUGIN_PATH="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/CoppeliaSim" \
QT_PLUGIN_PATH= \
QT_XCB_GL_INTEGRATION= \
QT_X11_NO_MITSHM=1 \
CUDA_VISIBLE_DEVICES=0 \
RLBENCH_PLANNER_MAX_TIME_MS=50 \
xvfb-run -a -s "-screen 0 1280x1024x24" \
/home/liusong/miniconda3/envs/rlbench/bin/python \
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/scripts/RE_rlbench_official_eval.py \
  --task water_plants \
  --policy-path "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_vfinal-20000+20000_5000+2_fixed_gripper_10tasks_0829/checkpoints/103000/pretrained_model" \
  --episodes 20 \
  --variation 0 \
  --seed 100 \
  --model-noise-seed 20260801 \
  --num-points 20000 \
  --gripper-points 500 \
  --add-gripper-cloud \
  --gripper-template reap \
  --action-index 0 \
  --exec-action-steps 28 \
  --planner-max-time-ms 50 \
  --execution-mode dataset_step \
  --arm-action-mode planning \
  --gripper-mode delta_width_initial_sync \
  --gripper-delta-open-threshold 0.0025 \
  --gripper-delta-close-threshold 0.003 \
  --gripper-delta-alignment current_minus_previous \
  --no-gripper-lock-after-close \
  --no-collision-checking \
  --image-size 256 \
  --gripper-len 0.11 \
  --max-model-calls 10 \
  --max-model-call-compensations 10 \
  --simulator-robustness-optimizations-song \
  --gripper-close-require-reach \
  --continue-chunk-on-mover-unreached \
  --mover-position-tolerance 0.001 \
  --mover-rotation-tolerance 0.05491 \
  --mover-gripper-position-tolerance 0.001 \
  --mover-gripper-rotation-tolerance 0.05491 \
  --save-video \
  --save-action-visualizations \
  --failure-artifacts-only \
  --no-save-action-chunks \
  --run-dir "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/eval/0903_SONG_TestScene_thresh3_phone_on_base_gripper_len11_seed101_episode20"


# ############# FULL TEST ###########
set -uo pipefail

EVAL_EPISODES="${EVAL_EPISODES:-100}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

if (( EVAL_EPISODES < 1 || EVAL_EPISODES > 100 )); then
    echo "EVAL_EPISODES 必须在 1~100 之间" >&2
    exit 2
fi

BASE="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench"
COPPELIA="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/CoppeliaSim"
PYTHON="/home/liusong/miniconda3/envs/rlbench/bin/python"
EVAL_PY="${BASE}/scripts/RE_rlbench_official_eval.py"
POLICY="${BASE}/outputs/wep_vla_vfinal-20000+20000_5000+2_fixed_gripper_10tasks_0829/checkpoints/103000/pretrained_model"
DATASET="${BASE}/datasets/rlbench_10tasks_100traj_raw_expert_target_reap_v4_20k_20260823_fixed_gripper_flow_20260827"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${BASE}/eval/Song_Full/0903_SONG_TrainScene_thresh3_10tasks_gripper_len11_chunk28_${EVAL_EPISODES}eps_${RUN_ID}_fix_water_seed100_fixed_virtual_gripper_FULL"

DATASET_EPISODES=""
for ((episode_index=0; episode_index<EVAL_EPISODES; episode_index++)); do
    if [[ -n "${DATASET_EPISODES}" ]]; then
        DATASET_EPISODES+=","
    fi
    DATASET_EPISODES+="${episode_index}"
done

# 慢任务优先，减少最后等待单个慢任务的尾部时间。
TASKS=(
    phone_on_base
    water_plants
    take_frame_off_hanger
    stack_wine
    close_laptop_lid
    sweep_to_dustpan
    take_umbrella_out_of_umbrella_stand
    close_fridge
    close_box
    toilet_seat_down
)

mkdir -p "${OUTPUT_ROOT}/launch_logs"

run_task() (
    task="$1"
    task_root="${OUTPUT_ROOT}/${task}"
    launch_log="${OUTPUT_ROOT}/launch_logs/${task}.log"

    echo "[start] task=${task} episodes=${EVAL_EPISODES}"

    env \
        PYTHONHASHSEED=0 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        MALLOC_ARENA_MAX=2 \
        COPPELIASIM_ROOT="${COPPELIA}" \
        LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}" \
        QT_QPA_PLATFORM=xcb \
        QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIA}" \
        QT_PLUGIN_PATH= \
        QT_XCB_GL_INTEGRATION= \
        QT_X11_NO_MITSHM=1 \
        CUDA_VISIBLE_DEVICES=0 \
        RLBENCH_PLANNER_MAX_TIME_MS=50 \
        xvfb-run -a \
        -s "-screen 0 1280x1024x24 -ac +extension GLX +render -noreset" \
        "${PYTHON}" "${EVAL_PY}" \
        --task "${task}" \
        --policy-path "${POLICY}" \
        --episodes "${EVAL_EPISODES}" \
        --variation 0 \
        --seed 100 \
        --model-noise-seed 20260801 \
        --num-points 20000 \
        --gripper-points 500 \
        --add-gripper-cloud \
        --gripper-template reap \
        --action-index 0 \
        --exec-action-steps 28 \
        --planner-max-time-ms 50 \
        --execution-mode dataset_step \
        --arm-action-mode planning \
        --gripper-mode delta_width_initial_sync \
        --gripper-delta-threshold 0.003 \
        --gripper-delta-open-threshold 0.0025 \
        --gripper-delta-close-threshold 0.003 \
        --gripper-delta-alignment current_minus_previous \
        --no-gripper-lock-after-close \
        --no-collision-checking \
        --image-size 256 \
        --gripper-len 0.11 \
        --max-model-calls 10 \
        --max-model-call-compensations 10 \
        --simulator-robustness-optimizations-song \
        --gripper-close-require-reach \
        --continue-chunk-on-mover-unreached \
        --mover-position-tolerance 0.001 \
        --mover-rotation-tolerance 0.05491 \
        --mover-gripper-position-tolerance 0.001 \
        --mover-gripper-rotation-tolerance 0.05491 \
        --save-video \
        --save-action-visualizations \
        --failure-artifacts-only \
        --no-save-action-chunks \
        --run-dir "${task_root}" \
        >"${launch_log}" 2>&1

    status=$?
    echo "[done] task=${task} status=${status} log=${launch_log}"
    exit "${status}"
)

active=0
overall_status=0

echo "[eval-start] tasks=10 episodes_per_task=${EVAL_EPISODES} total_episodes=$((10 * EVAL_EPISODES)) max_concurrent=${MAX_CONCURRENT}"
echo "[eval-output] ${OUTPUT_ROOT}"

for task in "${TASKS[@]}"; do
    while (( active >= MAX_CONCURRENT )); do
        if ! wait -n; then
            overall_status=1
        fi
        active=$((active - 1))
    done

    run_task "${task}" &
    active=$((active + 1))
done

while (( active > 0 )); do
    if ! wait -n; then
        overall_status=1
    fi
    active=$((active - 1))
done

echo "[eval-finished] status=${overall_status} output=${OUTPUT_ROOT}"
exit "${overall_status}"
