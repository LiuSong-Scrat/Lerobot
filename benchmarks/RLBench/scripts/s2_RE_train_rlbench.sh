#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/rlbench_water_lerobot_20260803_123657}"
POINTSEG_CACHE_DIR="${POINTSEG_CACHE_DIR:-${DATASET_ROOT}_pointseg_cache}"

"${PYTHON}" "${SCRIPT_DIR}/RE_rlbench_validate_reap_dataset.py" \
    "${DATASET_ROOT}" --cache-dir "${POINTSEG_CACHE_DIR}"

exec "${PYTHON}" /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/train_song_benchmark.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --pointseg_sample_cache_dir="${POINTSEG_CACHE_DIR}" \
    --dataset.repo_id="${DATASET_ROOT}" \
    --policy.vla_adapter_enable=true \
    --policy.vla_adapter_freeze_vlm=true \
    --policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
    --policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base \
    --policy.load_vlm_weights=true \
    --batch_size=32 \
    --steps=8000 \
    --log_freq=1 \
    --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_v041_rlbench_water_0803 \
    --job_name=wep_vla_v041_rlbench_water \
    --policy.device=cuda \
    --wandb.enable=false \
    --wandb.disable_artifact=true \
    --save_freq=1000 \
    --eval_freq=2000 \
    --num_workers=8 \
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
    --policy.worldflow_enable=false \
    --policy.worldflow_se3_head_enable=false \
    --policy.se3_enable=false \
    --policy.se3_final_correction_enable=false
