# MultiView DataCollection
ulimit -n 65535
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root benchmarks/song_real_libero/data/libero_setting/libero_demos \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 14 \
  --num-points 10000 \
  --camera agentview \
  --camera robot0_eye_in_hand \
  --image-camera agentview \
  --image-camera robot0_eye_in_hand \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 0 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --overwrite \
  --vis-dir benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_overhead_rgb/libero_4suite_lerobot_dataset/visualizations \
  --output-root benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_overhead_rgb/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud


# fused multiview cache
SONG_CAMERA_VIEWS=agentview,robot0_eye_in_hand \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
torchrun --standalone --nproc_per_node=6 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_overhead_rgb/libero_4suite_lerobot_dataset \
  --output-dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_overhead_rgb/libero_4suite_lerobot_cache \
  --batch-size=24 \
  --num-workers=4 \
  --shard-size=4096 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4 \
  --overwrite


# Train
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_overhead_rgb/libero_4suite_lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_overhead_rgb/libero_4suite_lerobot_cache \
  --policy.camera_views=agentview,robot0_eye_in_hand \
  --policy.rgb_camera_views=agentview \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
  --batch_size=4 \
  --steps=80000 \
  --log_freq=1 \
  --output_dir=benchmarks/song_real_libero/outputs/wep_vla_v043_multiview_overhead_rgb \
  --job_name=wep_vla_v043_multiview_overhead_rgb \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=2000 \
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


# Optional: enable joint World/Ego double flow
# Leave --policy.worldflow_enable=false above to keep the original v0.4.3
# module structure and checkpoint path. Enabling it adds an independent World
# LitePT + PointAction adapter; Ego and World action tokens then share the
# existing Action Expert.
#
# The dataset must contain world_ee_poses/ and, preferably,
# action_target_ee_poses/ sidecars produced by the current converters.
# Replace the WorldFlow flags in the training command with:
#
#   --policy.point_action_fusion_enable=true \
#   --policy.worldflow_enable=true \
#   --policy.worldflow_feature_dim=64 \
#   --policy.worldflow_grid_size=0.01 \
#   --policy.worldflow_loss_weight=0.01 \
#   --policy.worldflow_geo_loss_weight=0.002 \
#   --policy.worldflow_bridge_loss_weight=0.005 \
#   --policy.worldflow_equiv_loss_weight=0.002 \
#   --policy.worldflow_trans_weight=1.0 \
#   --policy.worldflow_rot_weight=1.0 \
#   --policy.worldflow_max_points=0 \
#   --policy.worldflow_require_action_target_sidecar=true \
#   --policy.pose9_action_noise_enable=true \
#   --policy.worldflow_noise_coupling=conjugate_ego \
#   --policy.worldflow_augmentation_trans_scale=0.05 \
#   --policy.worldflow_augmentation_rot_scale=0.2 \
#   --policy.worldflow_se3_head_enable=false \
#   --policy.se3_enable=false \
#   --policy.se3_final_correction_enable=false
#
# This keeps the original pose9 Action Expert while making the World and Ego
# random flow origins describe the same physical transform. For the legacy
# ablation use pose9_action_noise_enable=false and
# worldflow_noise_coupling=independent; that path remains supported but is not
# a geometrically coupled double flow.
#
# Full manifold ablation (larger behavior change): replace the pose9/se3 flags
# above with:
#   --policy.pose9_action_noise_enable=false \
#   --policy.se3_enable=true \
#   --policy.se3_twist_head_mode=pose9_endpoint \
#   --policy.worldflow_noise_coupling=conjugate_ego


# Real RGB-D collection and moving-camera processing
# See scripts/real_setting/README_CAMERA_MOTION.md. Main entrypoint:
#
#   bash benchmarks/song_real_libero/scripts/real_setting/song_rgbd_pipeline.sh --help
