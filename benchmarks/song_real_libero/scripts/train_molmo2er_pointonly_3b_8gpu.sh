#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_ROOT="${MOLMO_EXPERIMENT_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2}"
PYTHON_BIN="${EXPERIMENT_ROOT}/.venv-smol5090/bin/python"
MODEL_DIR="/raid5/rongshengwang/Lerobot/Molmo2-ER"
DATA_ROOT="${MOLMO_DATA_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data}"
DATASET_DIR="${DATA_ROOT}/libero_4suite_lerobot_dataset"
CACHE_DIR="${DATA_ROOT}/libero_4suite_lerobot_toolseg_cache"

NUM_PROCESSES="${MOLMO_NUM_PROCESSES:-8}"
BATCH_SIZE="${MOLMO_BATCH_SIZE:-24}"
ACCUMULATION_STEPS="${MOLMO_ACCUMULATION_STEPS:-1}"
STEPS="${MOLMO_STEPS:-80000}"
NUM_WORKERS="${MOLMO_NUM_WORKERS:-14}"
SAVE_CHECKPOINT="${MOLMO_SAVE_CHECKPOINT:-true}"
SAVE_FREQ="${MOLMO_SAVE_FREQ:-2000}"
EVAL_FREQ="${MOLMO_EVAL_FREQ:-2000}"
WANDB_ENABLE="${MOLMO_WANDB_ENABLE:-true}"
WANDB_RUN_MODE="${MOLMO_WANDB_RUN_MODE:-offline}"
POLICY_PATH="${MOLMO_POLICY_PATH:-}"
SCHEDULER_DECAY_LR="${MOLMO_SCHEDULER_DECAY_LR:-2.5e-6}"
OUTPUT_DIR="${MOLMO_OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/molmo2er_pointonly_3b_seed1000_8x5090_$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_NAME="${MOLMO_JOB_NAME:-molmo2er_pointonly_3b_seed1000_8x5090}"
MAIN_PROCESS_PORT="${MOLMO_MAIN_PROCESS_PORT:-29640}"
CODE_COMMIT="$(git -C "${EXPERIMENT_ROOT}" rev-parse HEAD)"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Persistent training Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}" || ! -d "${DATASET_DIR}" || ! -d "${CACHE_DIR}" ]]; then
  echo "Molmo weights, dataset, or PointSeg cache is missing." >&2
  exit 2
fi
if [[ -n "${POLICY_PATH}" && ! -f "${POLICY_PATH}/model.safetensors" ]]; then
  echo "Continuation policy checkpoint is incomplete: ${POLICY_PATH}" >&2
  exit 2
fi
if (( NUM_PROCESSES * BATCH_SIZE * ACCUMULATION_STEPS != 192 )); then
  echo "Refusing to change the controlled global batch: ${NUM_PROCESSES}*${BATCH_SIZE}*${ACCUMULATION_STEPS} != 192" >&2
  exit 2
fi
if [[ "${MOLMO_ALLOW_DIRTY_CODE:-0}" != "1" ]] && \
  [[ -n "$(git -C "${EXPERIMENT_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to train from a dirty experiment worktree; commit the exact code first." >&2
  exit 2
fi
if [[ "${MOLMO_REQUIRE_IDLE_GPUS:-1}" == "1" ]]; then
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "At least one GPU already has a compute process; refusing to overlap the formal run." >&2
    exit 75
  fi
  sleep 10
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "A GPU process appeared during the 10-second idle confirmation; refusing to overlap it." >&2
    exit 75
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${EXPERIMENT_ROOT}"

export CUDA_VISIBLE_DEVICES="${MOLMO_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${EXPERIMENT_ROOT}/src:${EXPERIMENT_ROOT}"
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] code_commit=${CODE_COMMIT}"
echo "[launch] processes=${NUM_PROCESSES} batch=${BATCH_SIZE} accumulation=${ACCUMULATION_STEPS} global_batch=192"
echo "[launch] steps=${STEPS} mixed_precision=no seed=1000"
echo "[launch] policy_path=${POLICY_PATH:-<fresh>}"
echo "[launch] lr=0.0001 warmup=100 decay_steps=30000 decay_lr=${SCHEDULER_DECAY_LR}"

if [[ -n "${POLICY_PATH}" ]]; then
  POLICY_SOURCE=(--policy.path="${POLICY_PATH}")
else
  POLICY_SOURCE=(--policy.type=smolvla)
fi

exec "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes="${NUM_PROCESSES}" \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  "${POLICY_SOURCE[@]}" \
  --resume=false \
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
  --policy.vlm_backend=molmo2_text \
  --policy.vlm_model_name="${MODEL_DIR}" \
  --policy.vlm_weights_path="${MODEL_DIR}" \
  --policy.load_vlm_weights=true \
  --policy.load_action_expert_weights=false \
  --policy.load_action_expert_projection_weights=false \
  --policy.num_vlm_layers=18 \
  --policy.num_expert_layers=18 \
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
  --eval_freq="${EVAL_FREQ}" \
  --num_workers="${NUM_WORKERS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.mode="${WANDB_RUN_MODE}" \
  --wandb.disable_artifact=true
