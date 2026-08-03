# SCAI SERVER

## Libero Benchmark
## 1.准备WEP-VLA Lerobot格式Benchmark数据集
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_demos/libero_spatial \
  --suite libero_spatial \
  --all-tasks \
  --episodes 50 \
  --num-workers 25 \
  --num-points 10000 \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 1 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --overwrite \
  --vis-dir /home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_dataset/libero_spatial_lerobot_dataset/visualizations \
  --output-root /home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_dataset/libero_spatial_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud


torchrun --standalone --nproc_per_node=4   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_dataset/libero_spatial_lerobot_dataset   --output-dir=/home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_dataset/libero_spatial_lerobot_dataset/libero_temp_lerobot_cache   --batch-size=24   --num-workers=4   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=4  --overwrite


#export CUDA_VISIBLE_DEVICES=1
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4   benchmarks/song_real_libero/scripts/train_song_benchmark.py \
    --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k/checkpoints/008000/pretrained_model \
    --policy.push_to_hub=false \
    --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v041_dataset_fixed_v1/libero_4suite_lerobot_dataset  \
    --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v041_dataset_fixed_v1/libero_4suite_lerobot_cache \
    --policy.vla_adapter_enable=true \
    --policy.vla_adapter_freeze_vlm=true \
    --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
    --policy.vlm_weights_path=/opt/data/private/liusong/hf_models/smolvla_base \
    --policy.load_vlm_weights=true \
    --batch_size=48 \
    --steps=80000 \
    --log_freq=1 \
    --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k_after8k_after10k \
    --job_name=wep_vla_v041_libero_dataset_fixed_v1_after4k_after8k_after10k  \
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
    --policy.pointseg_min_background_points=0  \
    --policy.pointseg_use_temporal_priors_as_input=false  \
    --policy.pointseg_use_pseudo_selection=false \
    --policy.worldflow_enable=false \
    --policy.worldflow_se3_head_enable=false \
    --policy.se3_enable=false \
    --policy.se3_final_correction_enable=false \


# DOUBLE FLOW
cd /home/liusong/ProgramFiles/Huggingface/lerobot
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1
CUDA_VISIBLE_DEVICES=0 \
python  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  \
  --policy.path=/tmp/temp/training/task5_double_flow_se3/checkpoints/016000/pretrained_model \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  \
  --dataset.repo_id=/home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_cache/ \
  --dataset.use_imagenet_stats=true \
  --dataset.image_transforms.enable=false \
  --dataset.video_backend=pyav \
  --output_dir=/tmp/temp/training/task5_double_flow_se3_after_16k \
  --job_name=task5_double_flow_se3_after_16k \
  --seed=1000 \
  --num_workers=8 \
  --batch_size=32 \
  --steps=40000 \
  --resume=false \
  --save_checkpoint=true \
  --save_freq=4000 \
  --eval_freq=4000 \
  --log_freq=1 \
  --optimizer.type=adamw \
  --optimizer.lr=1e-5 \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10.0 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.peak_lr=1e-5 \
  --scheduler.decay_lr=1e-6 \
  --scheduler.num_warmup_steps=100 \
  --scheduler.num_decay_steps=20000 \
  --use_policy_training_preset=true \
  --policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=16 \
  --policy.freeze_vision_encoder=true \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.load_action_expert_weights=false \
  --policy.load_action_expert_projection_weights=false \
  --policy.attention_mode=cross_attn \
  --policy.self_attn_every_n_layers=2 \
  --policy.expert_width_multiplier=0.75 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=16 \
  --policy.action_chunk_start_offset=1 \
  --policy.max_action_dim=10 \
  --policy.max_state_dim=10 \
  --policy.n_obs_steps=1 \
  --policy.num_steps=10 \
  --policy.flow_time_sampling=integration_grid \
  --policy.flow_time_zero_probability=0.9 \
  --policy.action_loss_translation_weight=1.0 \
  --policy.action_loss_rotation_weight=1.0 \
  --policy.action_loss_gripper_weight=1.0 \
  --policy.se3_enable=true \
  --policy.se3_twist_head_mode=direct_twist \
  --policy.se3_pose_loss_weight=1.0 \
  --policy.se3_endpoint_loss_weight=0.25 \
  --policy.se3_gripper_loss_weight=1.0 \
  --policy.se3_equivariance_loss_weight=0.0 \
  --policy.se3_final_correction_enable=false \
  --policy.se3_noise_trans_scale=0.0 \
  --policy.se3_noise_rot_scale=0.0 \
  --policy.se3_noise_gripper_scale=0.0 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_foreground_ratio=0.025 \
  --policy.pointseg_background_ratio=0.025 \
  --policy.pointseg_min_foreground_points=2500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_aux_loss_weight=0.001 \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.point_action_fusion_enable=true \
  --policy.point_action_fusion_heads=4 \
  --policy.point_action_fusion_dropout=0.0 \
  --policy.worldflow_enable=true \
  --policy.worldflow_feature_dim=64 \
  --policy.worldflow_grid_size=0.01 \
  --policy.worldflow_loss_weight=0.01 \
  --policy.worldflow_bridge_loss_weight=0.005 \
  --policy.worldflow_equiv_loss_weight=0.002 \
  --policy.worldflow_geo_loss_weight=0.002 \
  --policy.worldflow_trans_weight=1.0 \
  --policy.worldflow_rot_weight=1.0 \
  --policy.worldflow_noise_coupling=conjugate_ego \
  --policy.worldflow_augmentation_trans_scale=0.05 \
  --policy.worldflow_augmentation_rot_scale=0.2 \
  --policy.worldflow_min_transport_points=3 \
  --policy.worldflow_transport_score_threshold=0.05 \
  --policy.worldflow_se3_head_enable=false \
  --policy.worldflow_action_expert_dropout=0.0 \
  --policy.pose9_action_noise_enable=false \
  --policy.compile_model=false \
  --policy.use_amp=false \
  --policy.use_cache=true \
  --wandb.enable=true \
  --wandb.disable_artifact=true \


## EVAL


  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /tmp/temp/training/task5_double_flow_se3_after_16k/checkpoints/016000/pretrained_model \
  --suite libero_spatial \
  --task-id 5 \
  --episodes 30 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 10 \
  --inference-batch-size 10 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.004 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 16 \
  --adaptive-exec-max-steps 16 \
  --grasp-exec-steps 16 \
  --max-steps 600 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --action-index 0 \
  --gripper-delta-alignment current_minus_previous \
  --policy-noise-mode sample \
  --observation-height 256 \
  --observation-width 256 \
  --num-points 10000 \
  --no-ablate-worldflow-tokens \
  --point-cloud-seed-base 0 \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_task5_double_flow_se3_16k_after5k_after16k





cd /home/liusong/ProgramFiles/Huggingface/lerobot
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/tmp/temp/training/task5_causal_offset1_from_det4k_003000/checkpoints/005000/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v050_data/libero_temp_lerobot_cache/ \
  --batch_size=40 \
  --steps=50000 \
  --log_freq=1 \
  --output_dir=/tmp/temp/training/task5_after_5k \
  --job_name=task5_after_5k \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=5000 \
  --eval_freq=5000 \
  --num_workers=8 \
  --policy.action_chunk_start_offset=1 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=16 \
  --policy.num_steps=10 \
  --policy.flow_time_sampling=integration_grid \
  --policy.flow_time_zero_probability=0.5 \
  --policy.optimizer_lr=5e-6 \
  --policy.optimizer_weight_decay=1e-10 \
  --policy.optimizer_grad_clip_norm=10.0 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=60000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_foreground_ratio=0.025 \
  --policy.pointseg_background_ratio=0.025 \
  --policy.pointseg_min_foreground_points=2500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_aux_loss_weight=0.0001 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.point_action_fusion_enable=true \
  --policy.action_loss_translation_weight=10.0 \
  --policy.action_loss_rotation_weight=1.0 \
  --policy.action_loss_gripper_weight=10.0 \
  --policy.pose9_action_noise_enable=true \
  --policy.pose9_action_noise_trans_scale=0.0 \
  --policy.pose9_action_noise_rot_scale=0.0 \
  --policy.pose9_action_noise_gripper_scale=0.0 \
  --policy.worldflow_enable=true \
  --policy.worldflow_feature_dim=64 \
  --policy.worldflow_grid_size=0.01 \
  --policy.worldflow_loss_weight=0.01 \
  --policy.worldflow_geo_loss_weight=0.002 \
  --policy.worldflow_bridge_loss_weight=0.005 \
  --policy.worldflow_equiv_loss_weight=0.002 \
  --policy.worldflow_trans_weight=1.0 \
  --policy.worldflow_rot_weight=1.0 \
  --policy.worldflow_max_points=0 \
  --policy.worldflow_action_expert_layers=-1 \
  --policy.worldflow_action_expert_dropout=0.0 \
  --policy.worldflow_noise_trans_scale=0.0 \
  --policy.worldflow_noise_rot_scale=0.0 \
  --policy.worldflow_augmentation_trans_scale=0.05 \
  --policy.worldflow_augmentation_rot_scale=0.2 \
  --policy.worldflow_min_transport_points=3 \
  --policy.worldflow_transport_score_threshold=0.05 \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false

# REAL SETTING
  ## RawDataFromCamera 
    python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
    --camera L515 \
    --output benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video \
    --num-frames 0 \
    --storage video \
    --video-fps 15 \
    --space-toggle-recording 

  ## HDF5FromRawData  Raw--->HDF5
    ###############################
    #inference可视化  --run-inference or --show-inference
    #切片可视化  --show 
    #真机数据保持在 overhead 相机坐标系，不使用 camera_to_world 外参。
    #如果要一键离线推理再交互切片：
    python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
      --input benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube \
      --output-dir benchmarks/song_real_libero/data/real_setting/humanhand_demo_video_ego_cube \
      --run-inference \ 
      --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
      --fast \
      --force-handedness right \
      --fusion-mode model-depth \
      --camera-names overhead,hand 
      

    #直接用推理后的结果交互切片
    python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
      --input benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube \
      --output-dir benchmarks/song_real_libero/data/real_setting/humanhand_demo_video_ego_cube   

    #离线推理后切分点切片
    python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
      --input benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube \
      --jsonl benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube/handpose_wilor.jsonl \
      --output-dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_cube_demo \
      --no-interactive \
      --pose-frame camera \
      --segments "$(cat benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube/segments.txt)" \
      --max-points 50000 \
      --segment-workers 16

  ## 预处理---Continuous---HalfReduce---AddGripper---MixedStageGen
    ## Continuous----Check HDF5 Quality
      python benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py  --hdf5_dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo
    ## Continuous----Check HDF5 Quality

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/hdf5_edit_reduce.py

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/add_gripper_cloud_to_hdf5.py

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py

  ## LeRobotDatasetFromRealHDF5  HDF5-->(Add Gripper-->)Current EEF/UMI-->Zarr-->Dataset
    # 输入 HDF5 已经包含 observations/cloud_rgb/<camera>，不会下载 LIBERO 数据、
    # 重播轨迹、渲染深度图或从深度图反投影点云。
    # 如果 HDF5 已经由 add_gripper_cloud_to_hdf5.py 加过末端点云（StageGen Mixed），追加： --input-has-gripper-cloud 
    # 如果无虚拟末端点云，使用 --gripper-points 500 --gripper-max-width 0.08 
    # 当使用变长点云（适配StageGen Mixed）训练，使用 --num-points 0 
    #  --task "Place the Red Cube on the Blue Cube"  使用固定任务编码
    export HDF5_USE_FILE_LOCKING=FALSE
    python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
      --input-dir /opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/code_0611/humanhand_offline_demo \
      --output-root /opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/newmix_data/real_lerobot_dataset_new_priorseg \
      --repo-id song_real_pointcloud \
      --fps 15 \
      --num-points 0 \
      --input-has-gripper-cloud \
      --point-cloud-storage zarr \
      --workers 14 \
      --vis-count 2 \
      --overwrite \
      --task "Place the Red Cube on the Blue Cube"

  ## PriorCache ###########################   MultiGPU  --nproc_per_node=4
  torchrun --standalone --nproc_per_node=4   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/newmix_data/real_lerobot_dataset_new_priorseg   --output-dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/newmix_data/song_pointseg_sample_cache   --current-points 50000 --future-points 16384  --batch-size=24   --num-workers=14   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=0   --overwrite

  ## TrainCode ###########################   MultiGPU CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu --num_processes 4 python benchmarks/song_real_libero/scripts/train_song_benchmark.py
export SONG_POINTSEG_REQUIRE_POINTOPS=1 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4   benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/ep_vla_v0.2_new_priorseg_mixed_task/checkpoints/014000/pretrained_model \
      --policy.push_to_hub=false \
      --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/newmix_data/real_lerobot_dataset_new_priorseg \
      --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/newmix_data/song_pointseg_sample_cache \
      --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
      --policy.load_vlm_weights=false \
      --batch_size=48 \
      --steps=500000 \
      --log_freq=1 \
      --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/ep_vla_v0.2_new_priorseg_mixed_task1 \
      --job_name=ep_vla_v0.2_new_priorseg_mixed_task  \
      --policy.device=cuda \
      --wandb.enable=true \
      --wandb.disable_artifact=true \
      --save_freq=2000 \
      --eval_freq=2000 \
      --num_workers=12 \
      --policy.pointseg_enable=true \
      --policy.pointseg_backbone_type=litept \
      --policy.pointseg_grid_size=0.01 \
      --policy.pointseg_feature_dim=64 \
      --policy.pointseg_aux_loss_weight=0.002 \
      --policy.pointseg_foreground_ratio=0.08 \
      --policy.pointseg_background_ratio=0.08 \
      --policy.pointseg_min_foreground_points=4000 \
      --policy.pointseg_min_background_points=0  \
      --policy.pointseg_use_temporal_priors_as_input=false  \
      --policy.pointseg_use_pseudo_selection=false \
      --policy.worldflow_enable=false \
      --policy.worldflow_se3_head_enable=false \
      --policy.se3_enable=false \
      --policy.se3_final_correction_enable=false
