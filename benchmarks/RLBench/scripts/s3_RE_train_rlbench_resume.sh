#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/rlbench_box_tasks_100traj_lerobot_raw_expert_target_20260810_173629}"
POINTSEG_CACHE_DIR="${POINTSEG_CACHE_DIR:-${DATASET_ROOT}_pointseg_cache_new}"

"${PYTHON}" "${SCRIPT_DIR}/RE_rlbench_validate_reap_dataset.py" \
    "${DATASET_ROOT}" --cache-dir "${POINTSEG_CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch --num_processes=1 /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/train_song_benchmark.py \
    --policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_v041_rlbench_10tasks_0808/checkpoints/022000/pretrained_model \
    --policy.push_to_hub=false \
    --resume=true \
    --config_path=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_v041_rlbench_10tasks_0808/checkpoints/022000/pretrained_model/train_config.json \
    --dataset.repo_id="${DATASET_ROOT}" \
    --pointseg_sample_cache_dir="${POINTSEG_CACHE_DIR}" \
    --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_v041_rlbench_10tasks_0808/checkpoints \
    --policy.vla_adapter_enable=true \
    --policy.vla_adapter_freeze_vlm=true \
    --policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
    --policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base \
    --policy.load_vlm_weights=true \
    --batch_size=200 \
    --steps=35000 \
    --log_freq=1 \
    --job_name=wep_vla_v041_rlbench \
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
