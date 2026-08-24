#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
TOOLS="${SCRIPT_DIR}/full_molmo2er_worldflow_tools.py"
PYTHON_BIN="${MOLMO_PYTHON_BIN:-${EXPERIMENT_ROOT}/.venv-smol5090/bin/python}"
MODEL_DIR="${MOLMO2_ER_MODEL_DIR:-${EXPERIMENT_ROOT}/Molmo2-ER}"
DATA_ROOT="${MOLMO_DATA_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data}"
DATASET_DIR="${MOLMO_DATASET_DIR:-${DATA_ROOT}/libero_4suite_lerobot_dataset}"
CACHE_DIR="${MOLMO_POINTSEG_CACHE_DIR:-${DATA_ROOT}/libero_4suite_lerobot_toolseg_cache}"

# The physical device and exact-global-192 contracts are immutable. Only the
# four audited microbatch/accumulation profiles below are selectable.
readonly PHYSICAL_CUDA_DEVICES="0,1,2,3,4,5,6,7"
readonly NUM_PROCESSES=8
readonly BATCH_PROFILE="${MOLMO_BATCH_PROFILE:-b24}"
case "${BATCH_PROFILE}" in
  b4)
    BATCH_SIZE=4
    ACCUMULATION_STEPS=6
    ;;
  b8)
    BATCH_SIZE=8
    ACCUMULATION_STEPS=3
    ;;
  b16)
    BATCH_SIZE=16
    ACCUMULATION_STEPS=2
    ;;
  b24)
    BATCH_SIZE=24
    ACCUMULATION_STEPS=1
    ;;
  *)
    printf '[full-molmo2er-train] MOLMO_BATCH_PROFILE must be b4, b8, b16, or b24; got %s.\n' "${BATCH_PROFILE}" >&2
    exit 2
    ;;
esac
readonly BATCH_SIZE
readonly ACCUMULATION_STEPS
readonly GLOBAL_BATCH_SIZE=192
readonly SAMPLES_PER_FULL_MICROSTEP="$((NUM_PROCESSES * BATCH_SIZE))"
readonly FULL_MICROSTEPS="$((GLOBAL_BATCH_SIZE / SAMPLES_PER_FULL_MICROSTEP))"
readonly PARTIAL_SAMPLES="$((GLOBAL_BATCH_SIZE % SAMPLES_PER_FULL_MICROSTEP))"
(( PARTIAL_SAMPLES % BATCH_SIZE == 0 )) || {
  printf '[full-molmo2er-train] Internal exact-batch profile is not rank-microbatch representable.\n' >&2
  exit 2
}
readonly PARTIAL_ACTIVE_RANKS="$((PARTIAL_SAMPLES / BATCH_SIZE))"
readonly PHYSICAL_FORWARD_SAMPLES="$((SAMPLES_PER_FULL_MICROSTEP * ACCUMULATION_STEPS))"
readonly DISCARDED_FOR_GRADIENT="$((PHYSICAL_FORWARD_SAMPLES - GLOBAL_BATCH_SIZE))"
readonly VALID_LOSS_SCALE_FRACTION="$((NUM_PROCESSES * BATCH_SIZE))/$GLOBAL_BATCH_SIZE"
readonly REQUIRED_NVML_LIBRARY="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.107.02"
readonly NVML_LIBRARY="${MOLMO_NVML_LIBRARY:-${REQUIRED_NVML_LIBRARY}}"

RUN_MODE="${MOLMO_RUN_MODE:-formal}"
DEFAULT_STEPS=36000
DEFAULT_SAVE_FREQ=2000
DEFAULT_SAVE_CHECKPOINT=true
DEFAULT_WANDB_ENABLE=true
if [[ "${RUN_MODE}" == "benchmark" ]]; then
  DEFAULT_STEPS=8
  DEFAULT_SAVE_FREQ=8
  DEFAULT_SAVE_CHECKPOINT=false
  DEFAULT_WANDB_ENABLE=false
fi
STAGE_LABEL="${MOLMO_STAGE_LABEL:-stage1_fresh_000000_to036000}"
STAGE_KIND="${MOLMO_STAGE_KIND:-fresh}"
STEPS="${MOLMO_STEPS:-${DEFAULT_STEPS}}"
CUMULATIVE_START_STEP="${MOLMO_CUMULATIVE_START_STEP:-0}"
SCHEDULER_DECAY_LR="${MOLMO_SCHEDULER_DECAY_LR:-2.5e-6}"
POLICY_PATH="${MOLMO_POLICY_PATH:-}"
RESUME_CONFIG_PATH="${MOLMO_RESUME_CONFIG_PATH:-}"
OUTPUT_DIR="${MOLMO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/full_molmo2er_worldflow_three_stage/${STAGE_LABEL}}"
LAUNCH_RECORD_DIR="${MOLMO_LAUNCH_RECORD_DIR:-${OUTPUT_DIR}.launch}"
DATA_AUDIT_PATH="${MOLMO_DATA_AUDIT_PATH:-${LAUNCH_RECORD_DIR}/data_semantic_audit.json}"
EXPECTED_DATA_HASH="${MOLMO_EXPECTED_DATA_SEMANTIC_HASH:-}"
SAVE_FREQ="${MOLMO_SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}"
SAVE_CHECKPOINT="${MOLMO_SAVE_CHECKPOINT:-${DEFAULT_SAVE_CHECKPOINT}}"
NUM_WORKERS="${MOLMO_NUM_WORKERS:-8}"
MAIN_PROCESS_PORT="${MOLMO_MAIN_PROCESS_PORT:-29670}"
MIN_GPU_FREE_MIB="${MOLMO_MIN_GPU_FREE_MIB:-60000}"
WANDB_ENABLE="${MOLMO_WANDB_ENABLE:-${DEFAULT_WANDB_ENABLE}}"
WANDB_RUN_MODE="${MOLMO_WANDB_RUN_MODE:-offline}"
JOB_NAME="${MOLMO_JOB_NAME:-full_molmo2er_worldflow_${STAGE_LABEL}_seed1000_8xa800}"
HASH_MODEL_SHARDS="${MOLMO_HASH_MODEL_SHARDS:-true}"
DRY_RUN="${MOLMO_DRY_RUN:-false}"

fail() {
  printf '[full-molmo2er-train] %s\n' "$*" >&2
  exit 2
}

is_bool() {
  [[ "$1" == "true" || "$1" == "false" ]]
}

[[ "${NVML_LIBRARY}" == "${REQUIRED_NVML_LIBRARY}" ]] \
  || fail "MOLMO_NVML_LIBRARY must remain the kernel-matched ${REQUIRED_NVML_LIBRARY}."
[[ -s "${NVML_LIBRARY}" ]] || fail "Kernel-matched NVML library is missing: ${NVML_LIBRARY}"
export LD_PRELOAD="${NVML_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"

for required_file in "${TOOLS}" "${SCRIPT_DIR}/train_song_benchmark.py"; do
  [[ -f "${required_file}" ]] || fail "Missing required source file: ${required_file}"
done
[[ -x "${PYTHON_BIN}" ]] || fail "Training Python is missing or not executable: ${PYTHON_BIN}"
[[ -d "${MODEL_DIR}" ]] || fail "Molmo2-ER model directory is missing: ${MODEL_DIR}"
"${PYTHON_BIN}" "${TOOLS}" model-audit --model-dir "${MODEL_DIR}" >/dev/null
[[ -d "${DATASET_DIR}" ]] || fail "LeRobot dataset directory is missing: ${DATASET_DIR}"
[[ -d "${CACHE_DIR}" ]] || fail "PointSeg cache directory is missing: ${CACHE_DIR}"
[[ -d "${DATASET_DIR}/world_ee_poses" ]] || fail "Achieved World EEF sidecars are missing."
[[ -d "${DATASET_DIR}/action_target_ee_poses" ]] || fail "Commanded World EEF target sidecars are required."

if [[ -n "${MOLMO_CUDA_VISIBLE_DEVICES:-}" && "${MOLMO_CUDA_VISIBLE_DEVICES}" != "${PHYSICAL_CUDA_DEVICES}" ]]; then
  fail "MOLMO_CUDA_VISIBLE_DEVICES cannot override the fixed physical GPU 0-7 contract."
fi
if [[ -n "${MOLMO_NUM_PROCESSES:-}" && "${MOLMO_NUM_PROCESSES}" != "${NUM_PROCESSES}" ]]; then
  fail "MOLMO_NUM_PROCESSES must remain ${NUM_PROCESSES}."
fi
if [[ -n "${MOLMO_BATCH_SIZE:-}" && "${MOLMO_BATCH_SIZE}" != "${BATCH_SIZE}" ]]; then
  fail "MOLMO_BATCH_SIZE must remain ${BATCH_SIZE}."
fi
if [[ -n "${MOLMO_ACCUMULATION_STEPS:-}" && "${MOLMO_ACCUMULATION_STEPS}" != "${ACCUMULATION_STEPS}" ]]; then
  fail "MOLMO_ACCUMULATION_STEPS must remain ${ACCUMULATION_STEPS}."
fi
if [[ -n "${MOLMO_GLOBAL_BATCH_SIZE:-}" && "${MOLMO_GLOBAL_BATCH_SIZE}" != "${GLOBAL_BATCH_SIZE}" ]]; then
  fail "MOLMO_GLOBAL_BATCH_SIZE must remain ${GLOBAL_BATCH_SIZE}."
fi

[[ "${STAGE_LABEL}" =~ ^[a-zA-Z0-9_.-]+$ ]] || fail "Unsafe stage label: ${STAGE_LABEL}"
[[ "${STAGE_KIND}" == "fresh" || "${STAGE_KIND}" == "finetune" ]] || fail "MOLMO_STAGE_KIND must be fresh or finetune."
for integer_value in "${STEPS}" "${CUMULATIVE_START_STEP}" "${SAVE_FREQ}" "${NUM_WORKERS}" "${MAIN_PROCESS_PORT}" "${MIN_GPU_FREE_MIB}"; do
  [[ "${integer_value}" =~ ^[0-9]+$ ]] || fail "Expected a non-negative integer, got ${integer_value}."
done
(( STEPS > 0 && SAVE_FREQ > 0 && STEPS % SAVE_FREQ == 0 )) || fail "MOLMO_SAVE_FREQ must positively divide MOLMO_STEPS."
(( NUM_WORKERS >= 0 && MIN_GPU_FREE_MIB > 0 )) || fail "Worker count/free-memory floor is invalid."
for bool_value in "${WANDB_ENABLE}" "${HASH_MODEL_SHARDS}" "${SAVE_CHECKPOINT}" "${DRY_RUN}"; do
  is_bool "${bool_value}" || fail "Boolean environment values must be exactly true or false."
done
[[ "${RUN_MODE}" == "formal" || "${RUN_MODE}" == "smoke" || "${RUN_MODE}" == "benchmark" ]] || fail "MOLMO_RUN_MODE must be formal, smoke, or benchmark."

if [[ "${RUN_MODE}" == "smoke" ]]; then
  [[ "${STAGE_KIND}" == "fresh" ]] || fail "Smoke mode must construct a fresh policy."
  [[ "${STEPS}" == "2" ]] || fail "Smoke mode must contain exactly two optimizer steps."
  [[ "${CUMULATIVE_START_STEP}" == "0" ]] || fail "Smoke mode must start at step 0."
  [[ "${SCHEDULER_DECAY_LR}" == "2.5e-6" ]] || fail "Smoke mode must use the fresh LR floor."
  [[ "${SAVE_CHECKPOINT}" == "false" ]] || fail "Smoke mode must not write a giant policy checkpoint."
  [[ -z "${POLICY_PATH}" && -z "${RESUME_CONFIG_PATH}" ]] || fail "Smoke mode cannot load a policy or resume."
elif [[ "${RUN_MODE}" == "benchmark" ]]; then
  [[ -n "${MOLMO_BATCH_PROFILE:-}" ]] || fail "Benchmark mode requires an explicit MOLMO_BATCH_PROFILE."
  [[ -n "${MOLMO_OUTPUT_DIR:-}" && -n "${MOLMO_LAUNCH_RECORD_DIR:-}" ]] \
    || fail "Benchmark mode requires unique explicit output and launch-record directories."
  [[ "${STAGE_KIND}" == "fresh" && "${STEPS}" == "8" && "${CUMULATIVE_START_STEP}" == "0" ]] \
    || fail "Benchmark mode is fixed to a fresh eight-optimizer-step run from step 0."
  [[ "${SCHEDULER_DECAY_LR}" == "2.5e-6" ]] || fail "Benchmark mode must use the fresh LR floor."
  [[ "${SAVE_CHECKPOINT}" == "false" ]] || fail "Benchmark mode must not save policy checkpoints."
  [[ "${WANDB_ENABLE}" == "false" ]] || fail "Benchmark mode must keep W&B disabled."
  [[ -z "${POLICY_PATH}" && -z "${RESUME_CONFIG_PATH}" ]] || fail "Benchmark mode cannot load a policy or resume."
else
  [[ "${SAVE_CHECKPOINT}" == "true" ]] || fail "Formal stages must save checkpoints."
  if [[ "${STAGE_KIND}" == "fresh" ]]; then
    [[ "${STEPS}" == "36000" ]] || fail "Fresh stage must contain exactly 36000 optimizer steps."
    [[ "${CUMULATIVE_START_STEP}" == "0" ]] || fail "Fresh stage must start at cumulative step 0."
    [[ "${SCHEDULER_DECAY_LR}" == "2.5e-6" ]] || fail "Fresh stage LR floor must be 2.5e-6."
    [[ -z "${POLICY_PATH}" ]] || fail "Fresh stage cannot warm-start from a policy checkpoint."
  else
    [[ "${STEPS}" == "30000" ]] || fail "Fine-tuning stages must contain exactly 30000 optimizer steps."
    [[ "${SCHEDULER_DECAY_LR}" == "3e-5" ]] || fail "Fine-tuning stage LR floor must be 3e-5."
    if [[ -z "${RESUME_CONFIG_PATH}" ]]; then
      [[ -n "${POLICY_PATH}" ]] || fail "A fine-tuning stage requires MOLMO_POLICY_PATH."
    fi
  fi
fi

CONFIG_SOURCE=()
POLICY_SOURCE=()
RESUME_ARGUMENT=(--resume=false)
ATTEMPT_RECORD_DIR="${LAUNCH_RECORD_DIR}"
if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  [[ -z "${POLICY_PATH}" ]] || fail "In-stage resume cannot also specify MOLMO_POLICY_PATH."
  [[ -s "${RESUME_CONFIG_PATH}" && "$(basename -- "${RESUME_CONFIG_PATH}")" == "train_config.json" ]] \
    || fail "MOLMO_RESUME_CONFIG_PATH must name a complete train_config.json."
  [[ -d "${OUTPUT_DIR}" ]] || fail "Resume output directory does not exist: ${OUTPUT_DIR}"
  resume_checkpoint="$(dirname -- "${RESUME_CONFIG_PATH}")"
  "${PYTHON_BIN}" "${TOOLS}" checkpoint-audit --checkpoint "${resume_checkpoint}" --training-state >/dev/null
  CONFIG_SOURCE=(--config_path="${RESUME_CONFIG_PATH}")
  RESUME_ARGUMENT=(--resume=true)
  ATTEMPT_RECORD_DIR="${LAUNCH_RECORD_DIR}/resume_attempts/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
else
  [[ ! -e "${OUTPUT_DIR}" ]] || fail "Refusing to overwrite an existing fresh/warm-start output: ${OUTPUT_DIR}"
  [[ ! -e "${LAUNCH_RECORD_DIR}" ]] || fail "Refusing to overwrite an existing launch record: ${LAUNCH_RECORD_DIR}"
  if [[ -n "${POLICY_PATH}" ]]; then
    "${PYTHON_BIN}" "${TOOLS}" checkpoint-audit --checkpoint "${POLICY_PATH}" >/dev/null
    POLICY_SOURCE=(--policy.path="${POLICY_PATH}")
  else
    POLICY_SOURCE=(--policy.type=smolvla)
  fi
fi

mkdir -p "${ATTEMPT_RECORD_DIR}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_CUDA_DEVICES}"
if [[ "${DRY_RUN}" != "true" ]]; then
  "${PYTHON_BIN}" "${TOOLS}" gpu-audit \
    --min-free-mib "${MIN_GPU_FREE_MIB}" \
    --output "${ATTEMPT_RECORD_DIR}/gpu_audit.json" >/dev/null
fi

DATA_AUDIT_ARGS=(
  data-audit
  --dataset "${DATASET_DIR}"
  --cache "${CACHE_DIR}"
  --output "${DATA_AUDIT_PATH}"
)
if [[ -n "${EXPECTED_DATA_HASH}" ]]; then
  DATA_AUDIT_ARGS+=(--expected-hash "${EXPECTED_DATA_HASH}")
fi
"${PYTHON_BIN}" "${TOOLS}" "${DATA_AUDIT_ARGS[@]}" >/dev/null

TRAIN_COMMAND=(
  "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch
  --multi_gpu
  --num_processes="${NUM_PROCESSES}"
  --num_machines=1
  --mixed_precision=no
  --dynamo_backend=no
  --main_process_port="${MAIN_PROCESS_PORT}"
  benchmarks/song_real_libero/scripts/train_song_benchmark.py
  "${CONFIG_SOURCE[@]}"
  "${POLICY_SOURCE[@]}"
  "${RESUME_ARGUMENT[@]}"
  --policy.push_to_hub=false
  --dataset.repo_id="${DATASET_DIR}"
  --pointseg_sample_cache_dir="${CACHE_DIR}"
  --policy.n_obs_steps=1
  --policy.chunk_size=32
  --policy.n_action_steps=16
  --policy.action_chunk_start_offset=0
  --policy.max_state_dim=10
  --policy.max_action_dim=10
  --policy.camera_views=agentview
  --policy.rgb_camera_views=agentview
  --policy.empty_cameras=0
  --policy.tokenizer_max_length=48
  --policy.num_steps=10
  --policy.flow_time_sampling=beta
  --policy.flow_time_zero_probability=0.0
  --policy.use_cache=true
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.train_state_proj=true
  --policy.encode_robot_state=false
  --policy.vla_adapter_enable=false
  --policy.vla_adapter_freeze_vlm=true
  --policy.vlm_backend=molmo2_full
  --policy.full_molmo_topology=molmo_native_hybrid_wepvla_expert_v4
  --policy.molmo_inference_only=false
  --policy.molmo_gradient_checkpointing=true
  --policy.molmo_gradient_checkpointing_layers_per_segment=2
  --policy.vlm_model_name="${MODEL_DIR}"
  --policy.vlm_weights_path="${MODEL_DIR}"
  --policy.load_vlm_weights=true
  --policy.load_action_expert_weights=false
  --policy.load_action_expert_projection_weights=false
  --policy.num_vlm_layers=36
  --policy.num_expert_layers=36
  --policy.expert_width_multiplier=0.75
  --policy.self_attn_every_n_layers=2
  --policy.attention_mode=cross_attn
  --policy.add_image_special_tokens=false
  --policy.prefix_length=-1
  --policy.pad_language_to=longest
  --policy.pointseg_enable=true
  --policy.pointseg_checkpoint_path=null
  --policy.pointseg_backbone_type=litept
  --policy.pointseg_grid_size=0.01
  --policy.pointseg_feature_dim=64
  --policy.pointseg_aux_loss_weight=0.0005
  --policy.pointseg_foreground_ratio=0.025
  --policy.pointseg_background_ratio=0.025
  --policy.pointseg_min_foreground_points=2500
  --policy.pointseg_min_background_points=0
  --policy.pointseg_use_temporal_priors_as_input=false
  --policy.pointseg_use_pseudo_selection=false
  --policy.pointseg_freeze_batchnorm_stats=true
  --policy.point_action_fusion_enable=true
  --policy.point_action_fusion_heads=4
  --policy.point_action_fusion_dropout=0.0
  --policy.action_loss_translation_weight=1.0
  --policy.action_loss_rotation_weight=1.0
  --policy.action_loss_gripper_weight=1.0
  --policy.pose9_action_noise_enable=false
  --policy.pose9_action_noise_trans_scale=0.15
  --policy.pose9_action_noise_rot_scale=0.35
  --policy.pose9_action_noise_gripper_scale=0.05
  --policy.worldflow_enable=true
  --policy.worldflow_target_type=world_eef_trajectory
  --policy.worldflow_world_eef_velocity_mode=base_pose9_euclidean
  --policy.worldflow_reference_frame=robot_base
  --policy.worldflow_frame_origin=global
  --policy.worldflow_scene_frame_origin=global
  --policy.worldflow_action_fusion=point_action_expert_conjugate_bridge
  --policy.worldflow_action_expert_mode=shared
  --policy.worldflow_current_ee_pose_token=false
  --policy.worldflow_freeze_pretrained_ego=false
  --policy.worldflow_training_coordinate_frame_augmentation=false
  --policy.worldflow_pretrained_lr_multiplier=1.0
  --policy.worldflow_new_lr_multiplier=1.0
  --policy.worldflow_eef_probe_radius_m=0.10
  --policy.worldflow_bootstrap_from_ego=false
  --policy.worldflow_ego_residual_gate_init=null
  --policy.worldflow_noise_coupling=left_compose_ego
  --policy.worldflow_require_action_target_sidecar=true
  --policy.worldflow_feature_dim=64
  --policy.worldflow_grid_size=0.01
  --policy.worldflow_max_points=2048
  --policy.worldflow_loss_weight=1.0
  --policy.worldflow_geo_loss_weight=0.0
  --policy.worldflow_bridge_loss_weight=0.0
  --policy.worldflow_equiv_loss_weight=0.0
  --policy.worldflow_trans_weight=1.0
  --policy.worldflow_rot_weight=1.0
  --policy.worldflow_noise_trans_scale=0.15
  --policy.worldflow_noise_rot_scale=0.20
  --policy.worldflow_augmentation_trans_scale=0.20
  --policy.worldflow_augmentation_rot_scale=0.75
  --policy.worldflow_action_expert_layers=-1
  --policy.worldflow_action_expert_dropout=0.0
  --policy.worldflow_min_transport_points=3
  --policy.worldflow_transport_score_threshold=0.05
  --policy.worldflow_se3_head_enable=false
  --policy.se3_enable=false
  --policy.se3_final_correction_enable=false
  --policy.optimizer_lr=0.0001
  --policy.optimizer_betas='[0.9,0.95]'
  --policy.optimizer_eps=1e-8
  --policy.optimizer_weight_decay=1e-10
  --policy.optimizer_grad_clip_norm=10
  --policy.scheduler_warmup_steps=100
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
  --policy.compile_model=false
  --policy.use_amp=false
  --policy.use_peft=false
  --batch_size="${BATCH_SIZE}"
  --gradient_accumulation_steps="${ACCUMULATION_STEPS}"
  --global_batch_size="${GLOBAL_BATCH_SIZE}"
  --steps="${STEPS}"
  --seed=1000
  --log_freq=1
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq=0
  --num_workers="${NUM_WORKERS}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --policy.device=cuda
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.mode="${WANDB_RUN_MODE}"
  --wandb.disable_artifact=true
)

HASH_FLAG=--hash-model-shards
if [[ "${HASH_MODEL_SHARDS}" == "false" ]]; then
  HASH_FLAG=--no-hash-model-shards
fi
MANIFEST_SOURCE_ARGS=()
if [[ -n "${POLICY_PATH}" ]]; then
  MANIFEST_SOURCE_ARGS+=(--policy-path "${POLICY_PATH}")
fi
if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  MANIFEST_SOURCE_ARGS+=(--resume-config "${RESUME_CONFIG_PATH}")
fi
"${PYTHON_BIN}" "${TOOLS}" write-stage-manifest \
  --output "${ATTEMPT_RECORD_DIR}/launch_manifest.json" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --model-dir "${MODEL_DIR}" \
  --data-audit "${DATA_AUDIT_PATH}" \
  --stage-label "${STAGE_LABEL}" \
  --stage-kind "${STAGE_KIND}" \
  --steps "${STEPS}" \
  --cumulative-start-step "${CUMULATIVE_START_STEP}" \
  --decay-lr "${SCHEDULER_DECAY_LR}" \
  --batch-profile "${BATCH_PROFILE}" \
  "${MANIFEST_SOURCE_ARGS[@]}" \
  "${HASH_FLAG}" \
  --command "${TRAIN_COMMAND[@]}" >/dev/null

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EXPERIMENT_ROOT}/src:${EXPERIMENT_ROOT}"
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_RUN_MODE}"
export OMP_NUM_THREADS="${MOLMO_OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MOLMO_FULL_CUDA_LEASE_ENABLE="${MOLMO_FULL_CUDA_LEASE_ENABLE:-0}"
export MOLMO_CHECKPOINTS_TO_KEEP="${MOLMO_CHECKPOINTS_TO_KEEP:-1}"

printf '[full-molmo2er-train] stage=%s kind=%s output=%s\n' "${STAGE_LABEL}" "${STAGE_KIND}" "${OUTPUT_DIR}"
printf '[full-molmo2er-train] profile=%s physical_gpus=%s processes=%d microbatch=%d microsteps=%d full_microsteps=%d partial_active_ranks=%d exact_global_batch=%d physical_forwards=%d discarded_for_gradient=%d valid_scale=%s\n' \
  "${BATCH_PROFILE}" "${PHYSICAL_CUDA_DEVICES}" "${NUM_PROCESSES}" "${BATCH_SIZE}" "${ACCUMULATION_STEPS}" "${FULL_MICROSTEPS}" \
  "${PARTIAL_ACTIVE_RANKS}" "${GLOBAL_BATCH_SIZE}" "${PHYSICAL_FORWARD_SAMPLES}" "${DISCARDED_FOR_GRADIENT}" "${VALID_LOSS_SCALE_FRACTION}"
printf '[full-molmo2er-train] steps=%d cumulative=%d..%d lr=1e-4 warmup=100 decay_steps=30000 decay_lr=%s bootstrap=false\n' \
  "${STEPS}" "${CUMULATIVE_START_STEP}" "$((CUMULATIVE_START_STEP + STEPS))" "${SCHEDULER_DECAY_LR}"

cd -- "${EXPERIMENT_ROOT}"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[full-molmo2er-train] dry-run command:'
  printf ' %q' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${TRAIN_COMMAND[@]}"
