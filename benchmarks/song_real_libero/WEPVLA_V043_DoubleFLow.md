# WEPVLA V0.4.3 DoubleFlow

Mujoco==3.3.4  
Robosuite==1.4.0

以下命令在仓库根目录执行：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot

git rev-parse HEAD
```

本文对应的模型架构提交为：

```text
d21767a809f760a737ed07c7067f92d635361c38
```

## 1. 准备 Dataset

生成 LIBERO-10 task 6、task 8 各 50 episodes，共 100 episodes。数据同时保存：

- 以 episode 首帧末端为原点的 Ego action/state；
- 机器人基座坐标系下的当前末端位姿；
- 机器人基座坐标系下的 commanded EEF target；
- agentview RGB 与 XYZRGB 点云。

如果目标 Dataset 已经生成并通过检查，不需要重复执行本节。下面的
`--overwrite` 会覆盖同名输出。

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=5 \
MUJOCO_EGL_DEVICE_ID=5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
/home/liusong/anaconda3/envs/reap/bin/python \
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

3 个进程、每个进程 8 个 DataLoader workers，共 24 个 DataLoader workers。
如果 Cache 已经完整生成并通过检查，不需要重复执行本节。下面的
`--overwrite` 会覆盖同名输出。

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
CUDA_VISIBLE_DEVICES=3,4,5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
/home/liusong/anaconda3/envs/reap/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=3 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep \
  --camera-views=agentview \
  --camera-view-fusion=legacy_budget \
  --output-dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache \
  --current-points=10000 \
  --future-points=10000 \
  --batch-size=24 \
  --num-workers=8 \
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
- 原训练好的 Ego foreground/background token 原位保留在 VLM prefix；
- World foreground/background 作为两个新 token 追加到 VLM prefix；
- World object/background projection 复制训练好的 Ego projection 初始化，随后独立训练；
- shared PointActionExpert suffix 只包含 32 个 Ego action 与 32 个 World action token；
- Ego/World 对应动作使用成对位置编码、共轭运动编码和 block-causal 可见性；
- World 使用完整 pose9+gripper 10D flow，与 Ego 使用同形的逐通道 MSE；
- 只冻结 VLM，其余有效模块均参与训练。

3 张 GPU、每张卡 batch 32，global batch 为 `3 * 32 = 96`，与原正式
配置 `4 * 24 = 96` 相同。

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
CUDA_VISIBLE_DEVICES=3,4,5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
/home/liusong/anaconda3/envs/reap/bin/python \
  -m accelerate.commands.launch \
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
  --batch_size=32 \
  --gradient_accumulation_steps=1 \
  --steps=1300 \
  --save_freq=1300 \
  --save_steps='[100,260,520,780,1040,1300]' \
  --log_freq=1 \
  --eval_freq=1300 \
  --num_workers=8 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/a800_doubleflow_v0base_preservedprefix_pose9gripper_3gpu_b32_1300steps_20260817 \
  --job_name=a800_doubleflow_v0base_preservedprefix_pose9gripper_3gpu_b32_1300steps_20260817 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=50 \
  --policy.scheduler_decay_steps=1300 \
  --policy.scheduler_decay_lr=0.00001 \
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
```

长时间运行建议放入 tmux：

```bash
tmux new-session -s wepvla_v043_doubleflow
# 在 tmux 中执行上述训练命令
```

## 4. 测试

下面以 step 1300 为例，同时测试 LIBERO-10 task 6 和 task 8，各 50
episodes，共 100 episodes。A800 上使用 `2 * 25 = 50` 个环境 worker，
推理使用固定 batch 50。若测试其他 checkpoint，只替换 `--policy.path`
和 `--output-dir`。

当前分支按要求保留 v0.0 原生评测器和原生 LitePT，没有包含后来加入的
canonical singleton 或确定性 LitePT 修改。因此本节是高并行成功率评测，
不能宣称不同运行之间的 action 或成功 episode 集合 bitwise 完全一致。

```bash
PYTHONHASHSEED=0 \
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
/home/liusong/anaconda3/envs/reap/bin/python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/a800_doubleflow_v0base_preservedprefix_pose9gripper_3gpu_b32_1300steps_20260817/checkpoints/001300/pretrained_model \
  --device cuda \
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
  --output-dir /opt/data/private/liusong/benchmarks/song_real_libero/outputs/libero_setting/a800_doubleflow_v0base_preservedprefix_pose9gripper_step001300_100ep
```





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
MUJOCO_EGL_DEVICE_ID=0 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
/home/liusong/anaconda3/envs/reap/bin/python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v043_multiview_doubleflow_finetune_4G/checkpoints/030000/pretrained_model \
  --device cuda \
  --suite-gpu-ids 0,1,2,3 \
  --suite libero_10 \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
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
  --episode-workers-per-task 8 \
  --task-worker-backend process \
  --inference-batch-size 80 \
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
  --save-video \
  --no-world-to-ego-causal-ablation \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_temp_wep_vla_v043_multiview_doubleflow_finetune_4G_30k_libero_FULL

  