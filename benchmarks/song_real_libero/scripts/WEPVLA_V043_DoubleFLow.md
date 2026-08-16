# WEPVLA V0.4.3 DoubleFlow

Mujoco==3.3.4  
Robosuite==1.4.0

以下命令在仓库根目录执行：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot
```

## 1. 准备 Dataset

生成 LIBERO-10 task 6、task 8 各 50 episodes，共 100 episodes。数据同时保存：

- 以 episode 首帧末端为原点的 Ego action/state；
- 机器人基座坐标系下的当前末端位姿；
- 机器人基座坐标系下的 commanded EEF target；
- agentview RGB 与 XYZRGB 点云。

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/libero_demos \
  --suite libero_10 \
  --task-id 6 \
  --task-id 8 \
  --episodes 50 \
  --num-workers 10 \
  --worker-scope episode \
  --num-points 10000 \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 0 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --no-save-video \
  --vis-count 2 \
  --resume-temp-artifacts \
  --overwrite \
  --vis-dir /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep/visualizations \
  --output-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep \
  --repo-id song_libero10_task6_task8_world_eef_doubleflow
```

## 2. 生成 Cache

4 个进程、每个进程 6 个 DataLoader workers，适配最多 30 个 CPU 线程：

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
torchrun --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep \
  --camera-views=agentview \
  --camera-view-fusion=legacy_budget \
  --output-dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache \
  --current-points=10000 \
  --future-points=10000 \
  --batch-size=24 \
  --num-workers=6 \
  --shard-size=2048 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4 \
  --overwrite
```

## 3. 训练 DoubleFlow

当前正式结构：

- Ego：当前末端坐标系下的完整 EEF trajectory；
- World：机器人基座坐标系下的完整 EEF trajectory；
- Ego/World 使用相同类型的 LitePTEncoder 与 PointActionAdapter；
- 两个流进入同一个 PointActionExpert；
- 使用成对位置编码、共轭运动编码和 block-causal 可见性；
- 只冻结 VLM，其余有效模块均参与训练。

```bash
CUDA_VISIBLE_DEVICES=3,4,5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes=3 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=0 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache \
  --task_balanced_sampling=true \
  --batch_size=160 \
  --gradient_accumulation_steps=1 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=1 \
  --eval_freq=2000 \
  --num_workers=14 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/a800_world_eef_task6_task8_joint_pae_blockcausal_3gpu_b160_1300steps_20260816 \
  --job_name=a800_world_eef_task6_task8_joint_pae_blockcausal_3gpu_b160_1300steps_20260816 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.camera_views=agentview \
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
  --policy.worldflow_geo_loss_weight=1.0 \
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
```

长时间运行建议放入 tmux：

```bash
tmux new-session -s wepvla_v043_doubleflow
# 在 tmux 中执行上述训练命令
```

## 4. 测试

下面以 step 1300 为例，同时测试 LIBERO-10 task 6 和 task 8，各 50 episodes。若物理误差门槛选择了其他 checkpoint，只替换 `--policy.path` 和 `--output-dir`。

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=5 \
MUJOCO_EGL_DEVICE_ID=5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/a800_world_eef_task6_task8_joint_pae_blockcausal_3gpu_b160_1300steps_20260816/checkpoints/002000/pretrained_model \
  --suite libero_10 \
  --task-id 6 \
  --task-id 8 \
  --episodes 50 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --strict-official-init \
  --gripper-control-mode delta_width_initial_sync \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --waypoint-max-hold-steps 1 \
  --isolated-policy-workers 1 \
  --task-workers 2 \
  --episode-workers-per-task 25 \
  --task-worker-backend process \
  --inference-batch-size 50 \
  --inference-batching-mode fixed_barrier \
  --no-release-event-exec-enable \
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
  --no-save-video \
  --no-world-to-ego-causal-ablation \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_temp_double_flow_long68

