#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
PYTHON_BIN="${EXPERIMENT_ROOT}/.venv-smol5090/bin/python"
MODEL_DIR="/raid5/rongshengwang/Lerobot/Molmo2-ER"
DATA_ROOT="${MOLMO_DATA_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data}"
DATASET_DIR="${DATA_ROOT}/libero_4suite_lerobot_dataset"
CACHE_DIR="${DATA_ROOT}/libero_4suite_lerobot_toolseg_cache"

NUM_PROCESSES="${MOLMO_NUM_PROCESSES:-${NUM_PROCESSES:-8}}"
BATCH_SIZE="${MOLMO_BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${MOLMO_ACCUMULATION_STEPS:-}"
SELECTED_CUDA_DEVICES="${MOLMO_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
STEPS="${MOLMO_STEPS:-36000}"
NUM_WORKERS="${MOLMO_NUM_WORKERS:-8}"
SAVE_CHECKPOINT="${MOLMO_SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${MOLMO_SAVE_FREQ:-${STEPS}}"
WANDB_ENABLE="${MOLMO_WANDB_ENABLE:-false}"
WANDB_RUN_MODE="${MOLMO_WANDB_RUN_MODE:-offline}"
POLICY_PATH="${MOLMO_POLICY_PATH:-}"
RESUME_MODE="${MOLMO_RESUME:-false}"
RESUME_CONFIG_PATH="${MOLMO_RESUME_CONFIG_PATH:-}"
ALLOW_RESUME_WORLD_SIZE_CHANGE="${MOLMO_RESUME_ALLOW_WORLD_SIZE_CHANGE:-false}"
SCHEDULER_DECAY_LR="${MOLMO_SCHEDULER_DECAY_LR:-2.5e-6}"
OUTPUT_DIR="${MOLMO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/full_molmo2er_wep_vla_stage_$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_RECORD_DIR="${MOLMO_LAUNCH_RECORD_DIR:-${OUTPUT_DIR}.launch}"
JOB_NAME="${MOLMO_JOB_NAME:-full_molmo2er_wep_vla_seed1000_${NUM_PROCESSES}x5090}"
MAIN_PROCESS_PORT="${MOLMO_MAIN_PROCESS_PORT:-29650}"
MIN_GPU_FREE_MIB="${MOLMO_MIN_GPU_FREE_MIB:-31500}"

for command_name in git jq nvidia-smi realpath sha256sum find sort xargs cmp; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done
CODE_COMMIT="$(git -C "${EXPERIMENT_ROOT}" rev-parse HEAD)"

write_training_source_manifest() {
  local manifest_path="$1"
  (
    cd "${EXPERIMENT_ROOT}"
    find src benchmarks/song_real_libero/scripts \
      -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) > "${manifest_path}"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Persistent training Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}" || ! -d "${DATASET_DIR}" || ! -d "${CACHE_DIR}" ]]; then
  echo "Molmo weights, dataset, or PointSeg cache is missing." >&2
  exit 2
fi
if [[ "${RESUME_MODE}" != "true" && "${RESUME_MODE}" != "false" ]]; then
  echo "MOLMO_RESUME must be exactly true or false, got ${RESUME_MODE}." >&2
  exit 2
fi
if [[ "${ALLOW_RESUME_WORLD_SIZE_CHANGE}" != "true" && "${ALLOW_RESUME_WORLD_SIZE_CHANGE}" != "false" ]]; then
  echo "MOLMO_RESUME_ALLOW_WORLD_SIZE_CHANGE must be exactly true or false, got ${ALLOW_RESUME_WORLD_SIZE_CHANGE}." >&2
  exit 2
fi
for integer_value in "${NUM_PROCESSES}" "${BATCH_SIZE}" "${STEPS}" "${SAVE_FREQ}" "${MIN_GPU_FREE_MIB}"; do
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]] || (( integer_value < 1 )); then
    echo "GPU/process/batch/step/save-frequency values must be positive integers." >&2
    exit 2
  fi
done
if [[ -z "${ACCUMULATION_STEPS}" ]]; then
  if (( 192 % (NUM_PROCESSES * BATCH_SIZE) != 0 )); then
    echo "Cannot derive integral gradient accumulation for global batch 192: processes=${NUM_PROCESSES} batch=${BATCH_SIZE}." >&2
    exit 2
  fi
  ACCUMULATION_STEPS=$((192 / (NUM_PROCESSES * BATCH_SIZE)))
elif [[ ! "${ACCUMULATION_STEPS}" =~ ^[0-9]+$ ]] || (( ACCUMULATION_STEPS < 1 )); then
  echo "MOLMO_ACCUMULATION_STEPS must be a positive integer, got ${ACCUMULATION_STEPS}." >&2
  exit 2
fi
if [[ ! "${SELECTED_CUDA_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "CUDA device selection must be a comma-separated list of physical GPU indices, got ${SELECTED_CUDA_DEVICES}." >&2
  exit 2
fi
IFS=',' read -r -a SELECTED_GPU_IDS <<< "${SELECTED_CUDA_DEVICES}"
if (( ${#SELECTED_GPU_IDS[@]} != NUM_PROCESSES )); then
  echo "Selected ${#SELECTED_GPU_IDS[@]} physical GPUs (${SELECTED_CUDA_DEVICES}) but NUM_PROCESSES=${NUM_PROCESSES}." >&2
  exit 2
fi
declare -A SEEN_GPU_IDS=()
for physical_gpu_id in "${SELECTED_GPU_IDS[@]}"; do
  if [[ -n "${SEEN_GPU_IDS[${physical_gpu_id}]:-}" ]]; then
    echo "CUDA device selection contains duplicate physical GPU ${physical_gpu_id}: ${SELECTED_CUDA_DEVICES}." >&2
    exit 2
  fi
  SEEN_GPU_IDS["${physical_gpu_id}"]=1
done
if [[ -n "${POLICY_PATH}" ]]; then
  for filename in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -s "${POLICY_PATH}/${filename}" ]]; then
      echo "Continuation checkpoint is incomplete: ${POLICY_PATH}/${filename}" >&2
      exit 2
    fi
  done
fi
if (( NUM_PROCESSES * BATCH_SIZE * ACCUMULATION_STEPS != 192 )); then
  echo "Refusing global-batch drift: ${NUM_PROCESSES}*${BATCH_SIZE}*${ACCUMULATION_STEPS} != 192" >&2
  exit 2
fi

RESUME_CHECKPOINT_ROOT=""
RESUME_STEP=""
RESUME_SOURCE_NUM_PROCESSES="${NUM_PROCESSES}"
RESUME_SOURCE_ACCUMULATION_STEPS="${ACCUMULATION_STEPS}"
if [[ "${RESUME_MODE}" == "true" ]]; then
  if [[ -n "${POLICY_PATH}" ]]; then
    echo "MOLMO_POLICY_PATH must be empty for an in-stage resume; use MOLMO_RESUME_CONFIG_PATH." >&2
    exit 2
  fi
  if [[ ! -s "${RESUME_CONFIG_PATH}" || "$(basename "${RESUME_CONFIG_PATH}")" != "train_config.json" ]]; then
    echo "Resume config is missing or is not train_config.json: ${RESUME_CONFIG_PATH:-<empty>}" >&2
    exit 2
  fi
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Resume output directory does not exist: ${OUTPUT_DIR}" >&2
    exit 2
  fi

  resume_config_canonical="$(realpath -e "${RESUME_CONFIG_PATH}")"
  output_canonical="$(realpath -e "${OUTPUT_DIR}")"
  resume_pretrained_dir="$(dirname "${resume_config_canonical}")"
  RESUME_CHECKPOINT_ROOT="$(dirname "${resume_pretrained_dir}")"
  resume_checkpoint_name="$(basename "${RESUME_CHECKPOINT_ROOT}")"
  case "${resume_config_canonical}" in
    "${output_canonical}"/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]/pretrained_model/train_config.json) ;;
    *)
      echo "Resume config is not a numeric checkpoint inside OUTPUT_DIR: ${resume_config_canonical}" >&2
      exit 2
      ;;
  esac

  for filename in \
    pretrained_model/config.json \
    pretrained_model/model.safetensors \
    pretrained_model/policy_preprocessor.json \
    pretrained_model/policy_postprocessor.json \
    training_state/optimizer_param_groups.json \
    training_state/optimizer_state.safetensors \
    training_state/rng_state.safetensors \
    training_state/scheduler_state.json \
    training_state/training_step.json; do
    if [[ ! -s "${RESUME_CHECKPOINT_ROOT}/${filename}" ]]; then
      echo "Resume checkpoint is incomplete: ${RESUME_CHECKPOINT_ROOT}/${filename}" >&2
      exit 2
    fi
  done
  if ! "${PYTHON_BIN}" - \
    "${RESUME_CHECKPOINT_ROOT}/pretrained_model/model.safetensors" \
    "${RESUME_CHECKPOINT_ROOT}/training_state/optimizer_state.safetensors" \
    "${RESUME_CHECKPOINT_ROOT}/training_state/rng_state.safetensors" \
    >/dev/null 2>&1 <<'PY'
import sys

from safetensors import safe_open

for path in sys.argv[1:]:
    with safe_open(path, framework="numpy") as handle:
        if not list(handle.keys()):
            raise SystemExit(f"empty safetensors file: {path}")
PY
  then
    echo "Resume checkpoint contains a structurally invalid safetensors file: ${RESUME_CHECKPOINT_ROOT}" >&2
    exit 2
  fi
  RESUME_STEP="$(jq -r '.step // empty' "${RESUME_CHECKPOINT_ROOT}/training_state/training_step.json")"
  if [[ ! "${RESUME_STEP}" =~ ^[0-9]+$ ]] || \
    (( 10#${resume_checkpoint_name} != RESUME_STEP )) || \
    (( RESUME_STEP <= 0 || RESUME_STEP >= STEPS )); then
    echo "Invalid resume step/checkpoint name: name=${resume_checkpoint_name} step=${RESUME_STEP:-<missing>} target=${STEPS}" >&2
    exit 2
  fi
  if ! jq -e --argjson resume_step "${RESUME_STEP}" '
    (.last_epoch == $resume_step)
    and (._step_count == ($resume_step + 1))
    and ((._last_lr | type) == "array")
    and ((.base_lrs | type) == "array")
  ' "${RESUME_CHECKPOINT_ROOT}/training_state/scheduler_state.json" >/dev/null || \
    ! jq -e 'type == "array" and length > 0' \
      "${RESUME_CHECKPOINT_ROOT}/training_state/optimizer_param_groups.json" >/dev/null; then
    echo "Resume optimizer/scheduler control state is invalid for step ${RESUME_STEP}." >&2
    exit 2
  fi
  saved_output_dir="$(jq -r '.output_dir // empty' "${resume_config_canonical}")"
  if [[ -z "${saved_output_dir}" ]] || \
    [[ "$(realpath -m "${saved_output_dir}")" != "${output_canonical}" ]]; then
    echo "Resume checkpoint output_dir does not match ${output_canonical}." >&2
    exit 2
  fi
  saved_batch_size="$(jq -r '.batch_size // empty' "${resume_config_canonical}")"
  RESUME_SOURCE_ACCUMULATION_STEPS="$(jq -r '.gradient_accumulation_steps // empty' "${resume_config_canonical}")"
  if [[ ! "${saved_batch_size}" =~ ^[0-9]+$ ]] || (( saved_batch_size < 1 )) || \
    [[ ! "${RESUME_SOURCE_ACCUMULATION_STEPS}" =~ ^[0-9]+$ ]] || \
    (( RESUME_SOURCE_ACCUMULATION_STEPS < 1 )) || \
    (( 192 % (saved_batch_size * RESUME_SOURCE_ACCUMULATION_STEPS) != 0 )); then
    echo "Resume checkpoint has no valid global-batch-192 distributed contract." >&2
    exit 2
  fi
  RESUME_SOURCE_NUM_PROCESSES=$((192 / (saved_batch_size * RESUME_SOURCE_ACCUMULATION_STEPS)))
  if (( RESUME_SOURCE_NUM_PROCESSES * saved_batch_size * RESUME_SOURCE_ACCUMULATION_STEPS != 192 )); then
    echo "Resume source global batch is not exactly 192." >&2
    exit 2
  fi
  if (( RESUME_SOURCE_NUM_PROCESSES != NUM_PROCESSES )); then
    supported_elastic_transition=false
    if (( saved_batch_size == 1 && BATCH_SIZE == 1 )) && \
      { \
        (( RESUME_SOURCE_NUM_PROCESSES == 4 && NUM_PROCESSES == 8 && \
           RESUME_SOURCE_ACCUMULATION_STEPS == 48 && ACCUMULATION_STEPS == 24 )) || \
        (( RESUME_SOURCE_NUM_PROCESSES == 8 && NUM_PROCESSES == 4 && \
           RESUME_SOURCE_ACCUMULATION_STEPS == 24 && ACCUMULATION_STEPS == 48 )); \
      }; then
      supported_elastic_transition=true
    fi
    if [[ "${ALLOW_RESUME_WORLD_SIZE_CHANGE}" != "true" ]] || \
      [[ "${supported_elastic_transition}" != "true" ]]; then
      echo "Refusing unsupported resume world-size change: source=${RESUME_SOURCE_NUM_PROCESSES}x${saved_batch_size}x${RESUME_SOURCE_ACCUMULATION_STEPS}, target=${NUM_PROCESSES}x${BATCH_SIZE}x${ACCUMULATION_STEPS}." >&2
      exit 2
    fi
  elif (( RESUME_SOURCE_ACCUMULATION_STEPS != ACCUMULATION_STEPS )) || \
    (( saved_batch_size != BATCH_SIZE )); then
    echo "Refusing resume batch/accumulation drift at unchanged world size." >&2
    exit 2
  fi
  if ! jq -e \
    --argjson steps "${STEPS}" \
    --argjson batch_size "${BATCH_SIZE}" \
    --argjson decay_lr "${SCHEDULER_DECAY_LR}" '
      (.steps == $steps)
      and (.batch_size == $batch_size)
      and (.seed == 1000)
      and (.save_checkpoint == true)
      and (.eval_freq == 0)
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
      and (.policy.scheduler_decay_lr == $decay_lr)
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
      and (.scheduler.decay_lr == $decay_lr)
    ' "${resume_config_canonical}" >/dev/null; then
    echo "Resume checkpoint violates the locked training contract: ${resume_config_canonical}" >&2
    exit 2
  fi
else
  if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
    echo "MOLMO_RESUME_CONFIG_PATH is only valid when MOLMO_RESUME=true." >&2
    exit 2
  fi
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse output directory for a fresh/warm-start launch: ${OUTPUT_DIR}" >&2
    exit 2
  fi
  if [[ -e "${LAUNCH_RECORD_DIR}" ]]; then
    echo "Refusing to reuse launch-record directory for a fresh/warm-start launch: ${LAUNCH_RECORD_DIR}" >&2
    exit 2
  fi
fi

for physical_gpu_id in "${SELECTED_GPU_IDS[@]}"; do
  if ! gpu_query_result="$(
    nvidia-smi \
      --id="${physical_gpu_id}" \
      --query-gpu=index,memory.free \
      --format=csv,noheader,nounits
  )"; then
    echo "Failed to query selected physical GPU ${physical_gpu_id}." >&2
    exit 2
  fi
  if [[ ! "${gpu_query_result}" =~ ^[[:space:]]*([0-9]+),[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
    echo "Physical GPU ${physical_gpu_id} returned invalid index/free-memory data: ${gpu_query_result:-<empty>}." >&2
    exit 2
  fi
  queried_gpu_id="${BASH_REMATCH[1]}"
  gpu_free_mib="${BASH_REMATCH[2]}"
  if (( 10#${queried_gpu_id} != 10#${physical_gpu_id} )); then
    echo "Requested physical GPU ${physical_gpu_id}, but nvidia-smi returned GPU ${queried_gpu_id}." >&2
    exit 2
  fi
  if (( gpu_free_mib < MIN_GPU_FREE_MIB )); then
    echo "Physical GPU ${physical_gpu_id} has only ${gpu_free_mib} MiB free; need at least ${MIN_GPU_FREE_MIB} MiB." >&2
    exit 75
  fi
done

# TrainPipelineConfig intentionally rejects non-empty fresh output roots. Keep
# immutable launch provenance in an adjacent directory so the training process
# itself remains the sole creator of OUTPUT_DIR. Resume attempts receive their
# own immutable subdirectory and never overwrite the original launch record.
if [[ "${RESUME_MODE}" == "true" ]]; then
  resume_attempt_id="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
  ATTEMPT_RECORD_DIR="${LAUNCH_RECORD_DIR}/resume_attempts/${resume_attempt_id}"
  mkdir -p "${ATTEMPT_RECORD_DIR}"
  printf '%s\n' "$(realpath -e "${RESUME_CONFIG_PATH}")" > "${ATTEMPT_RECORD_DIR}/resume_config_path.txt"
  printf '%s\n' "${RESUME_STEP}" > "${ATTEMPT_RECORD_DIR}/resume_step.txt"
  sha256sum \
    "${RESUME_CONFIG_PATH}" \
    "${RESUME_CHECKPOINT_ROOT}/training_state/training_step.json" \
    "${RESUME_CHECKPOINT_ROOT}/training_state/scheduler_state.json" \
    "${RESUME_CHECKPOINT_ROOT}/training_state/optimizer_param_groups.json" \
    > "${ATTEMPT_RECORD_DIR}/resume_control_state_sha256.txt"
  write_training_source_manifest "${ATTEMPT_RECORD_DIR}/training_source_sha256.txt"
  if [[ ! -s "${LAUNCH_RECORD_DIR}/training_source_sha256.txt" ]] || \
    ! cmp -s \
      "${LAUNCH_RECORD_DIR}/training_source_sha256.txt" \
      "${ATTEMPT_RECORD_DIR}/training_source_sha256.txt"; then
    echo "Refusing resume because the training source tree differs from the initial launch manifest." >&2
    exit 2
  fi
  if [[ ! -s "${LAUNCH_RECORD_DIR}/distributed_contract.json" ]] || ! jq -e \
    --argjson source_processes "${RESUME_SOURCE_NUM_PROCESSES}" \
    --argjson source_batch "${saved_batch_size}" \
    --argjson source_accumulation "${RESUME_SOURCE_ACCUMULATION_STEPS}" '
      (.global_batch_size == 192)
      and (.num_processes == $source_processes)
      and (.batch_size_per_process == $source_batch)
      and (.gradient_accumulation_steps == $source_accumulation)
    ' "${LAUNCH_RECORD_DIR}/distributed_contract.json" >/dev/null; then
    echo "Refusing resume because initial distributed provenance disagrees with the checkpoint." >&2
    exit 2
  fi
else
  ATTEMPT_RECORD_DIR="${LAUNCH_RECORD_DIR}"
  mkdir -p "${ATTEMPT_RECORD_DIR}"
  sha256sum "${MODEL_DIR}"/model-*.safetensors > "${ATTEMPT_RECORD_DIR}/molmo_source_sha256.txt"
  write_training_source_manifest "${ATTEMPT_RECORD_DIR}/training_source_sha256.txt"
fi
jq -n \
  --arg created_at "$(date -u +%FT%TZ)" \
  --arg mode "$([[ "${RESUME_MODE}" == "true" ]] && printf resume || printf fresh_or_warm_start)" \
  --argjson num_processes "${NUM_PROCESSES}" \
  --argjson batch_size "${BATCH_SIZE}" \
  --argjson accumulation_steps "${ACCUMULATION_STEPS}" \
  --argjson source_num_processes "${RESUME_SOURCE_NUM_PROCESSES}" \
  --argjson source_accumulation_steps "${RESUME_SOURCE_ACCUMULATION_STEPS}" \
  --argjson allow_world_size_change "${ALLOW_RESUME_WORLD_SIZE_CHANGE}" '
    {
      created_at: $created_at,
      mode: $mode,
      num_processes: $num_processes,
      batch_size_per_process: $batch_size,
      gradient_accumulation_steps: $accumulation_steps,
      global_batch_size: ($num_processes * $batch_size * $accumulation_steps),
      source_num_processes: $source_num_processes,
      source_gradient_accumulation_steps: $source_accumulation_steps,
      allow_world_size_change: $allow_world_size_change
    }
  ' > "${ATTEMPT_RECORD_DIR}/distributed_contract.json"
git -C "${EXPERIMENT_ROOT}" status --short > "${ATTEMPT_RECORD_DIR}/code_status_at_launch.txt"
git -C "${EXPERIMENT_ROOT}" diff --binary > "${ATTEMPT_RECORD_DIR}/code_diff_at_launch.patch"
printf '%s\n' "${CODE_COMMIT}" > "${ATTEMPT_RECORD_DIR}/code_commit.txt"

cd "${EXPERIMENT_ROOT}"
export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_DEVICES}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EXPERIMENT_ROOT}/src:${EXPERIMENT_ROOT}"
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${MOLMO_OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MOLMO_FULL_CUDA_LEASE_ENABLE="${MOLMO_FULL_CUDA_LEASE_ENABLE:-1}"
export MOLMO_FULL_CUDA_LEASE_TARGET_GIB="${MOLMO_FULL_CUDA_LEASE_TARGET_GIB:-30}"
export MOLMO_FULL_CUDA_LEASE_CHUNK_MIB="${MOLMO_FULL_CUDA_LEASE_CHUNK_MIB:-1024}"
export MOLMO_FULL_CUDA_LEASE_HEADROOM_MIB="${MOLMO_FULL_CUDA_LEASE_HEADROOM_MIB:-512}"
export MOLMO_CHECKPOINTS_TO_KEEP="${MOLMO_CHECKPOINTS_TO_KEEP:-1}"

echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] launch_record_dir=${ATTEMPT_RECORD_DIR}"
echo "[launch] code_commit=${CODE_COMMIT} (working-tree snapshot saved in adjacent launch record)"
echo "[launch] physical_cuda_devices=${CUDA_VISIBLE_DEVICES}"
echo "[launch] processes=${NUM_PROCESSES} batch=${BATCH_SIZE} accumulation=${ACCUMULATION_STEPS} global_batch=192"
echo "[launch] steps=${STEPS} seed=1000 native_rgb=agentview full_vlm=36/36 expert=36"
echo "[launch] policy_path=${POLICY_PATH:-<fresh>}"
echo "[launch] resume=${RESUME_MODE} resume_step=${RESUME_STEP:-<none>} resume_checkpoint=${RESUME_CHECKPOINT_ROOT:-<none>}"
echo "[launch] resume_distribution=${RESUME_SOURCE_NUM_PROCESSES}x${BATCH_SIZE}x${RESUME_SOURCE_ACCUMULATION_STEPS}->${NUM_PROCESSES}x${BATCH_SIZE}x${ACCUMULATION_STEPS} allow_world_size_change=${ALLOW_RESUME_WORLD_SIZE_CHANGE}"
echo "[launch] lr=0.0001 warmup=100 decay_steps=30000 decay_lr=${SCHEDULER_DECAY_LR}"
echo "[launch] cuda_allocator_lease=${MOLMO_FULL_CUDA_LEASE_ENABLE} target=${MOLMO_FULL_CUDA_LEASE_TARGET_GIB}GiB chunk=${MOLMO_FULL_CUDA_LEASE_CHUNK_MIB}MiB headroom=${MOLMO_FULL_CUDA_LEASE_HEADROOM_MIB}MiB"
echo "[launch] rolling_checkpoint_retention=${MOLMO_CHECKPOINTS_TO_KEEP} (committed numeric checkpoints per stage)"

CONFIG_SOURCE=()
if [[ "${RESUME_MODE}" == "true" ]]; then
  CONFIG_SOURCE=(--config_path="${RESUME_CONFIG_PATH}")
  POLICY_SOURCE=()
  RESUME_ARGUMENT=(--resume=true)
elif [[ -n "${POLICY_PATH}" ]]; then
  POLICY_SOURCE=(--policy.path="${POLICY_PATH}")
  RESUME_ARGUMENT=(--resume=false)
else
  POLICY_SOURCE=(--policy.type=smolvla)
  RESUME_ARGUMENT=(--resume=false)
fi

exec "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes="${NUM_PROCESSES}" \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  "${CONFIG_SOURCE[@]}" \
  "${POLICY_SOURCE[@]}" \
  "${RESUME_ARGUMENT[@]}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_DIR}" \
  --pointseg_sample_cache_dir="${CACHE_DIR}" \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=16 \
  --policy.action_chunk_start_offset=0 \
  --policy.max_state_dim=10 \
  --policy.max_action_dim=10 \
  --policy.camera_views=agentview \
  --policy.rgb_camera_views=agentview \
  --policy.empty_cameras=0 \
  --policy.tokenizer_max_length=48 \
  --policy.num_steps=10 \
  --policy.flow_time_sampling=beta \
  --policy.flow_time_zero_probability=0.0 \
  --policy.use_cache=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.encode_robot_state=false \
  --policy.vla_adapter_enable=false \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_backend=molmo2_full \
  --policy.vlm_model_name="${MODEL_DIR}" \
  --policy.vlm_weights_path="${MODEL_DIR}" \
  --policy.load_vlm_weights=true \
  --policy.load_action_expert_weights=false \
  --policy.load_action_expert_projection_weights=false \
  --policy.num_vlm_layers=36 \
  --policy.num_expert_layers=36 \
  --policy.expert_width_multiplier=0.75 \
  --policy.self_attn_every_n_layers=2 \
  --policy.attention_mode=cross_attn \
  --policy.add_image_special_tokens=false \
  --policy.prefix_length=-1 \
  --policy.pad_language_to=longest \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.0005 \
  --policy.pointseg_foreground_ratio=0.025 \
  --policy.pointseg_background_ratio=0.025 \
  --policy.pointseg_min_foreground_points=2500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.point_action_fusion_enable=true \
  --policy.point_action_fusion_heads=4 \
  --policy.point_action_fusion_dropout=0.0 \
  --policy.action_loss_translation_weight=1.0 \
  --policy.action_loss_rotation_weight=1.0 \
  --policy.action_loss_gripper_weight=1.0 \
  --policy.pose9_action_noise_enable=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.worldflow_bootstrap_from_ego=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false \
  --policy.optimizer_lr=0.0001 \
  --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=1e-10 \
  --policy.optimizer_grad_clip_norm=10 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}" \
  --policy.compile_model=false \
  --policy.use_amp=false \
  --policy.use_peft=false \
  --batch_size="${BATCH_SIZE}" \
  --gradient_accumulation_steps="${ACCUMULATION_STEPS}" \
  --steps="${STEPS}" \
  --seed=1000 \
  --log_freq=1 \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq=0 \
  --num_workers="${NUM_WORKERS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.mode="${WANDB_RUN_MODE}" \
  --wandb.disable_artifact=true
