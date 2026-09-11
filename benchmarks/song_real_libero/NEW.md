# Cache
torchrun --standalone --nproc_per_node=4   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/libero_data/lerobot_dataset   --output-dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/libero_data/lerobot_cache   --batch-size=32   --num-workers=8   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=4  --overwrite



# 訓練
cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song

PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/src:/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
TOKENIZERS_PARALLELISM=false \
WANDB_MODE=offline \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MOLMO_FULL_CUDA_LEASE_ENABLE=0 \
MOLMO_CHECKPOINTS_TO_KEEP=1 \
/home/liusong/anaconda3/envs/reap/bin/python \
  -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes=7 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=29670 \
benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/wepvla_v043_doubleflow_multiview_long68/checkpoints/last/pretrained_model \
  --resume=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache \
  --batch_size=40 \
  --gradient_accumulation_steps=1 \
  --global_batch_size=280 \
  --steps=4500 \
  --seed=1000 \
  --save_checkpoint=true \
  --save_freq=1500 \
  --eval_freq=1500 \
  --log_freq=1 \
  --num_workers=12 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/wepvla_v043_doubleflow_multiview_long68_after_1500 \
  --job_name=stage1_wepprefix_v3_long68_fresh_after_1500 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
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
  --policy.full_molmo_topology=wepvla_scene_in_vlm_prefix_v3 \
  --policy.molmo_inference_only=false \
  --policy.molmo_gradient_checkpointing=true \
  --policy.molmo_gradient_checkpointing_layers_per_segment=2 \
  --policy.molmo_image_fast_path=true \
  --policy.vlm_model_name=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/Molmo2-ER \
  --policy.vlm_weights_path=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/Molmo2-ER \
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
  --policy.pointseg_checkpoint_path=null \
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
  --policy.pointseg_freeze_batchnorm_stats=true \
  --policy.pose9_action_noise_trans_scale=0.15 \
  --policy.pose9_action_noise_rot_scale=0.35 \
  --policy.pose9_action_noise_gripper_scale=0.05 \
  --policy.worldflow_enable=true \
  --policy.worldflow_target_type=world_eef_trajectory \
  --policy.worldflow_world_eef_velocity_mode=base_pose9_euclidean \
  --policy.worldflow_reference_frame=robot_base \
  --policy.worldflow_frame_origin=global \
  --policy.worldflow_scene_frame_origin=global \
  --policy.worldflow_action_fusion=point_action_expert_conjugate_bridge \
  --policy.worldflow_action_expert_mode=shared \
  --policy.worldflow_current_ee_pose_token=false \
  --policy.worldflow_freeze_pretrained_ego=false \
  --policy.worldflow_training_coordinate_frame_augmentation=false \
  --policy.worldflow_pretrained_lr_multiplier=1.0 \
  --policy.worldflow_new_lr_multiplier=1.0 \
  --policy.worldflow_eef_probe_radius_m=0.10 \
  --policy.worldflow_bootstrap_from_ego=false \
  --policy.worldflow_ego_residual_gate_init=null \
  --policy.worldflow_noise_coupling=left_compose_ego \
  --policy.worldflow_require_action_target_sidecar=true \
  --policy.worldflow_feature_dim=64 \
  --policy.worldflow_grid_size=0.01 \
  --policy.worldflow_max_points=2048 \
  --policy.worldflow_loss_weight=1.0 \
  --policy.worldflow_geo_loss_weight=0.0 \
  --policy.worldflow_bridge_loss_weight=0.0 \
  --policy.worldflow_equiv_loss_weight=0.0 \
  --policy.worldflow_trans_weight=1.0 \
  --policy.worldflow_rot_weight=1.0 \
  --policy.worldflow_noise_trans_scale=0.15 \
  --policy.worldflow_noise_rot_scale=0.20 \
  --policy.worldflow_augmentation_trans_scale=0.20 \
  --policy.worldflow_augmentation_rot_scale=0.75 \
  --policy.worldflow_action_expert_layers=-1 \
  --policy.worldflow_action_expert_dropout=0.0 \
  --policy.worldflow_min_transport_points=3 \
  --policy.worldflow_transport_score_threshold=0.05 \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false \
  --policy.optimizer_lr=0.0001 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=1e-10 \
  --policy.optimizer_grad_clip_norm=10 \
  --policy.scheduler_warmup_steps=50 \
  --policy.scheduler_decay_steps=4500 \
  --policy.scheduler_decay_lr=3e-5 \
  --policy.compile_model=false \
  --policy.use_amp=false \
  --policy.use_peft=false


# 新數據訓練
cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song

PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=7 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/src:/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song \
HF_HUB_OFFLINE=1 \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/liusong/anaconda3/envs/reap/bin/python \
  -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes=7 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=29682 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_4suite40tasks_from055000_step0_100k_manual/checkpoints/090000/pretrained_model \
  --resume=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/libero_data/lerobot_dataset \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/libero_data/lerobot_cache \
  --batch_size=20 \
  --gradient_accumulation_steps=1 \
  --global_batch_size=140 \
  --task_balanced_sampling=true \
  --steps=60000 \
  --seed=1000 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --eval_freq=2000 \
  --log_freq=1 \
  --num_workers=12 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_4suite40tasks_from55k_after_90k_after50k_NEWDATA \
  --job_name=molmo_wepvla_4suite40tasks_from55k_after_90k_after50k_NEWDATA \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=0 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=0.000001 \
  --policy.camera_views=agentview \
  --policy.rgb_camera_views=agentview




# 测试

  cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song

  PYTHONHASHSEED=0 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  MALLOC_ARENA_MAX=2 \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/src:/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song \
  /home/liusong/anaconda3/envs/reap/bin/python \
    benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_4suite40tasks_from55k_after_90k_after28k/checkpoints/last/pretrained_model/ \
    --device cuda \
    --suite-gpu-ids 0,1,2,3 \
    --suite libero_spatial \
    --suite libero_object \
    --suite libero_goal \
    --suite libero_10 \
    --all-tasks \
    --episodes 50 \
    --policy-noise-seed 0 \
    --env-seed 7 \
    --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 \
    --task-workers 10 \
    --episode-workers-per-task 5 \
    --task-worker-backend process \
    --inference-batch-size 200 \
    --inference-batching-mode fixed_barrier \
    --no-release-event-exec-enable \
    --control-freq 20 \
    --action-index 0 \
    --exec-action-steps 16 \
    --adaptive-exec-max-steps 16 \
    --grasp-exec-steps 16 \
    --max-steps 1000 \
    --no-use-suite-max-steps \
    --recreate-env-per-episode \
    --render-mode offscreen \
    --no-visualize-foreground \
    --save-video \
    --no-world-to-ego-causal-ablation \
    --output-dir benchmarks/song_real_libero/outputs/full_molmo2er_worldflow_after5w5_FULL_afert_90k_after28k_after50k_chunk16_FULL





# 数据集虚拟gripper到人手的标定
cd /home/liusong/ProgramFiles/Huggingface/lerobot
BASE=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/real_data/static_human_data
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src:/home/liusong/ProgramFiles/Huggingface/lerobot \
/home/liusong/anaconda3/envs/reap/bin/python \
  benchmarks/song_real_libero/scripts/real_setting/preview_virtual_eef_transform_ply.py \
  --input-dataset "$BASE/wepvla_v043_doubleflow_static_franka_fold_trash_src_with_stagegen_data/lerobot_dataset" \
  --output-dir "/home/liusong/temp/virtual_eef_custom_preview/umi_vis" \
  --episodes 0 113 215 715 \
  --frames 76 33 70 42 \
  --x 0.03 \
  --y 0.02 \
  --z -0.02 \
  --rotx 15 \
  --roty -30 \
  --rotz 10