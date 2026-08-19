#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the current DoubleFlow architecture with input-only dual-view FPS.
# The cache contract guarantees that the model sees 10,000 points, not the
# 19,500-point scene union used transiently by the input adapter.

REPO_ROOT=/home/liusong/ProgramFiles/Huggingface/lerobot
PYTHON_BIN=/home/liusong/anaconda3/envs/reap/bin/python
DATASET_ROOT=${SONG_V043_DATASET_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_dataset}
CACHE_ROOT=${SONG_V043_MULTIVIEW_CACHE_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_union_fps_cache}
BASE_POLICY=${SONG_V043_DOUBLEFLOW_BASE_POLICY:-/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wepvla_v043_doubleflow_multiview_v03_after_3k_libero/checkpoints/002500/pretrained_model}
OUTPUT_ROOT=${SONG_V043_DOUBLEFLOW_MULTIVIEW_OUTPUT:-/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v043_doubleflow_dualview_fps_4gpu_b32_w6_r2_30k_20260819}
JOB_NAME=${SONG_V043_DOUBLEFLOW_MULTIVIEW_JOB_NAME:-wep_vla_v043_doubleflow_dualview_fps_4gpu_b32_w6_r2_30k_20260819}

cd "$REPO_ROOT"

test -f "$BASE_POLICY/config.json"
test -f "$CACHE_ROOT/manifest.json"
test "$(jq -r '.current_points' "$CACHE_ROOT/manifest.json")" = "10000"
test "$(jq -r '.future_points' "$CACHE_ROOT/manifest.json")" = "10000"
test "$(jq -r '.camera_view_fusion' "$CACHE_ROOT/manifest.json")" = "fps"
test "$(jq -r '.camera_views | join(",")' "$CACHE_ROOT/manifest.json")" = "agentview,robot0_eye_in_hand"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite an existing training output: $OUTPUT_ROOT" >&2
  echo "Set SONG_V043_DOUBLEFLOW_MULTIVIEW_OUTPUT and SONG_V043_DOUBLEFLOW_MULTIVIEW_JOB_NAME for a new run." >&2
  exit 1
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$REPO_ROOT/src"
export SONG_POINTSEG_REQUIRE_POINTOPS=1

# 24 DataLoader workers plus four train ranks stay below the 30-process CPU budget.
exec "$PYTHON_BIN" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=0 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path="$BASE_POLICY" \
  --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET_ROOT" \
  --pointseg_sample_cache_dir="$CACHE_ROOT" \
  --task_balanced_sampling=true \
  --batch_size=32 \
  --gradient_accumulation_steps=1 \
  --steps=30000 \
  --save_freq=2000 \
  --log_freq=1 \
  --eval_freq=2000 \
  --num_workers=6 \
  --output_dir="$OUTPUT_ROOT" \
  --job_name="$JOB_NAME" \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=0.00003 \
  --policy.camera_views=agentview,robot0_eye_in_hand \
  --policy.camera_view_fusion=fps \
  --policy.rgb_camera_views=agentview \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/opt/data/private/liusong/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
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
  --policy.pointseg_freeze_batchnorm_stats=true \
  --policy.point_action_fusion_enable=true \
  --policy.worldflow_enable=true \
  --policy.worldflow_target_type=world_eef_trajectory \
  --policy.worldflow_world_eef_velocity_mode=base_pose9_euclidean \
  --policy.worldflow_reference_frame=robot_base \
  --policy.worldflow_frame_origin=global \
  --policy.worldflow_scene_frame_origin=global \
  --policy.worldflow_noise_coupling=left_compose_ego \
  --policy.worldflow_action_fusion=point_action_expert_conjugate_bridge \
  --policy.worldflow_action_expert_mode=shared \
  --policy.worldflow_current_ee_pose_token=false \
  --policy.worldflow_bootstrap_from_ego=true \
  --policy.worldflow_freeze_pretrained_ego=false \
  --policy.worldflow_feature_dim=64 \
  --policy.worldflow_grid_size=0.01 \
  --policy.worldflow_max_points=2048 \
  --policy.worldflow_loss_weight=1.0 \
  --policy.worldflow_geo_loss_weight=0.0 \
  --policy.worldflow_bridge_loss_weight=0.0 \
  --policy.worldflow_equiv_loss_weight=0.0 \
  --policy.worldflow_training_coordinate_frame_augmentation=false \
  --policy.worldflow_pretrained_lr_multiplier=1.0 \
  --policy.worldflow_new_lr_multiplier=1.0 \
  --policy.worldflow_trans_weight=1.0 \
  --policy.worldflow_rot_weight=1.0 \
  --policy.worldflow_eef_probe_radius_m=0.10 \
  --policy.worldflow_require_action_target_sidecar=true \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
