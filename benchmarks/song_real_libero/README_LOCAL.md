
# LOCAL SERVER

## Libero Benchmark
## 1.准备WEP-VLA Lerobot格式Benchmark数据集
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_demos \
  --suite libero_spatial \
  --task-id 5 \
  --episodes 10 \
  --num-workers 10 \
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
  --vis-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wepvla_v5_libero_4suite_data/libero_4suite_lerobot_dataset/visualizations \
  --output-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wepvla_v5_libero_4suite_data/libero_4suite_lerobot_dataset/ \
  --repo-id song_libero_4suite_pointcloud

torchrun --standalone --nproc_per_node=4   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wepvla_v5_libero_4suite_data/libero_4suite_lerobot_dataset/   --output-dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wepvla_v5_libero_4suite_data/libero_4suite_lerobot_cache   --batch-size=24   --num-workers=4   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=4  --overwrite


#export CUDA_VISIBLE_DEVICES=1
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1 
python  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wepvla_v5_libero_4suite_data/libero_4suite_lerobot_dataset/  \
    --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/wepvla_v5_libero_4suite_data/libero_4suite_lerobot_cache \
    --policy.vla_adapter_enable=true \
    --policy.vla_adapter_freeze_vlm=true \
    --policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
    --policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base \
    --policy.load_vlm_weights=true \
    --batch_size=2 \
    --steps=80000 \
    --log_freq=1 \
    --output_dir=benchmarks/song_real_libero/outputs/wep_vla_v050_doubleflow \
    --job_name=wep_vla_v050_doubleflow  \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --save_freq=200 \
    --eval_freq=200 \
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
    --policy.worldflow_enable=true \
    --policy.worldflow_feature_dim=64 \
    --policy.worldflow_grid_size=0.01 \
    --policy.worldflow_max_points=0 \
    --policy.point_action_fusion_enable=true \
    --policy.point_action_fusion_heads=4 \
    --policy.worldflow_loss_weight=1.0 \
    --policy.worldflow_geo_loss_weight=0.05 \
    --policy.worldflow_bridge_loss_weight=0.05 \
    --policy.worldflow_equiv_loss_weight=0.02 \
    --policy.worldflow_trans_weight=1.0 \
    --policy.worldflow_rot_weight=1.0 \
    --policy.worldflow_augmentation_trans_scale=0.20 \
    --policy.worldflow_augmentation_rot_scale=0.75 \
    --policy.worldflow_se3_head_enable=false \
    --policy.se3_enable=false \
    --policy.se3_final_correction_enable=false

#### doubleflow
--policy.worldflow_enable=true
--policy.worldflow_max_points=0
--policy.worldflow_loss_weight=0.05
--policy.worldflow_geo_loss_weight=0.05
--policy.worldflow_bridge_loss_weight=0.05
--policy.worldflow_equiv_loss_weight=0.02
--policy.se3_enable=false

#### auxiliary
--policy.worldflow_enable=true \
--policy.worldflow_action_expert_layers=-1 \
--policy.worldflow_max_points=0 \
--policy.worldflow_loss_weight=0.05 \
--policy.worldflow_geo_loss_weight=0.05 \
--policy.worldflow_bridge_loss_weight=0.05 \
--policy.worldflow_equiv_loss_weight=0.02 \
--policy.worldflow_se3_head_enable=false \
--policy.se3_enable=false \
--policy.se3_final_correction_enable=false


## BenchmarkEval ###########################  
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_10 \
  --task-id 3 \
  --task-id 4 \
  --task-id 6 \
  --task-id 9 \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 4 \
  --episode-workers-per-task 1 \
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/10k_dataset_fixed_v1_new_long_thresh2_chunk24
  
  
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_goal \
  --task-id 3 \
  --task-id 5 \
  --task-id 9 \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 3 \
  --episode-workers-per-task 2 \
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/10k_eval_new_goal_dataset_fixed_v1_thresh2_chunk24



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
    export HDF5_USE_FILE_LOCKING=FALSE
    python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
      --input-dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo \
      --output-root benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset \
      --repo-id song_real_pointcloud \
      --fps 15 \
      --num-points 0 \
      --input-has-gripper-cloud \
      --point-cloud-storage zarr \
      --workers 6 \
      --vis-count 2 \
      --overwrite \
      --task  "Place the Red Cube on the Blue Cube"


  ## PriorCache ###########################   MultiGPU  --nproc_per_node=4
  torchrun --standalone --nproc_per_node=1   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset   --output-dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache   --current-points 50000 --future-points 16384  --batch-size=24   --num-workers=14   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=0   --overwrite

  ## TrainCode ###########################   MultiGPU CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu --num_processes 4 python benchmarks/song_real_libero/scripts/train_song_benchmark.py
    export SONG_POINTSEG_REQUIRE_POINTOPS=1 
    python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --policy.type=smolvla \
      --policy.push_to_hub=false \
      --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset \
      --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache \
      --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
      --policy.load_vlm_weights=false \
      --batch_size=4 \
      --steps=500000 \
      --log_freq=1 \
      --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/ep-vla \
      --job_name=ep-vla  \
      --policy.device=cuda \
      --wandb.enable=true \
      --wandb.disable_artifact=true \
      --save_freq=5000 \
      --eval_freq=5000 \
      --num_workers=0 \
      --policy.pointseg_enable=true \
      --policy.pointseg_backbone_type=litept \
      --policy.pointseg_grid_size=0.01 \
      --policy.pointseg_feature_dim=64 \
      --policy.pointseg_aux_loss_weight=0.002 \
      --policy.pointseg_foreground_ratio=0.08 \
      --policy.pointseg_background_ratio=0.08 \
      --policy.pointseg_min_foreground_points=500 \
      --policy.pointseg_min_background_points=0 \
      --policy.worldflow_enable=true \
      --policy.worldflow_se3_head_enable=false \
      --policy.se3_enable=false \
      --policy.se3_final_correction_enable=false

  ## BenchmarkEval ###########################  
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python   benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py   --config benchmarks/song_real_libero/configs/libero.json   --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model  --suite libero_spatial  --all-tasks   --episodes 10  --action-index 0   --exec-action-steps 12   --save-video --render-mode offscreen   --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/10w_eval_spatial_temp


