  # 1....  From Zero
  --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual_after2500/checkpoints/013500/pretrained_model \  
  
  # 2....  RESUME
  --config_path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual/checkpoints/002500/pretrained_model/train_config.json \
  --resume=true \
  --resume_restart_scheduler=true \
  --resume_scheduler_start_lr=0.00007 \
  --resume_scheduler_end_lr=0.00002 \
  --resume_scheduler_decay_steps=30000 \
  # 3.... LORA
  --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual_after2500/checkpoints/013500/pretrained_model \
  --resume=false \
  --steps=30000 \
  --policy.optimizer_lr=0.00007 \
  --policy.scheduler_warmup_steps=0 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=0.00003 \
  --policy.molmo_lora_enable=true \
  --policy.molmo_lora_rank=8 \
  --policy.molmo_lora_alpha=8.0 \
  --policy.molmo_lora_dropout=0.0 \
  --policy.molmo_lora_lr_multiplier=0.1 \
  --policy.use_peft=false \




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
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache \
  --config_path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual_after33000/checkpoints/033500/pretrained_model/train_config.json \
  --resume=true \
  --resume_restart_scheduler=true \
  --resume_scheduler_phase_start_step=33500 \
  --resume_scheduler_start_lr=1e-7 \
  --resume_scheduler_end_lr=1e-10 \
  --resume_scheduler_decay_steps=3000 \
  --policy.optimizer_lr=0.00006 \
  --policy.use_peft=false \
  --policy.molmo_gradient_checkpointing=false \
  --batch_size=60 \
  --global_batch_size=420 \
  --steps=38000 \
  --save_freq=250 \
  --eval_freq=250 \
  --log_freq=1 \
  --num_workers=12 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual_after33500 \
  --job_name=v3_feature_align_long68_fresh_language_casual_after33500 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.device=cuda \
  --policy.rgb_camera_views=agentview \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.vlm_backend=molmo2_full \
  --policy.vlm_model_name=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/Molmo2-ER \
  --policy.vlm_weights_path=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/Molmo2-ER \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=36 \
  --policy.num_expert_layers=36 \
  --policy.expert_width_multiplier=0.28125 \
  --policy.pointseg_enable=true \
  --policy.pointseg_aux_loss_weight=0.0005 \
  --policy.pointseg_foreground_ratio=0.025 \
  --policy.pointseg_background_ratio=0.025 \
  --policy.pointseg_min_foreground_points=2500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.pointseg_freeze_batchnorm_stats=true \
  --policy.worldflow_enable=true \
  --policy.worldflow_target_type=world_eef_trajectory \
  --policy.worldflow_world_eef_velocity_mode=base_pose9_euclidean \
  --policy.worldflow_reference_frame=robot_base \
  --policy.worldflow_scene_frame_origin=global \
  --policy.worldflow_action_fusion=point_action_expert_conjugate_bridge \
  --policy.worldflow_noise_coupling=left_compose_ego \
  --policy.worldflow_require_action_target_sidecar=true \
  --policy.worldflow_max_points=2048 \
  --policy.worldflow_loss_weight=1.0 \
  --policy.worldflow_geo_loss_weight=0.0 \
  --policy.worldflow_bridge_loss_weight=0.0 \
  --policy.worldflow_equiv_loss_weight=0.0 








# TEST



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
  PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song/src:/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song \
  /home/liusong/anaconda3/envs/reap/bin/python \
    benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
    --config benchmarks/song_real_libero/configs/libero.json \
    --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual_after33000/checkpoints/last/pretrained_model \
    --device cuda \
    --suite libero_10 \
    --task-id 6 \
    --episodes 50 \
    --policy-noise-seed 0 \
    --env-seed 7 \
    --strict-official-init \
    --gripper-control-mode delta_width_initial_sync \
    --gripper-delta-threshold 0.002 \
    --gripper-delta-alignment current_minus_previous \
    --waypoint-max-hold-steps 1 \
    --isolated-policy-workers 1 \
    --task-workers 1 \
    --episode-workers-per-task 50 \
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
    --save-video \
    --no-world-to-ego-causal-ablation \
    --output-dir benchmarks/song_real_libero/outputs/v3_feature_align_casual_after29250_nograd_2000_long68_31




可以，但当前仓库有大量 staged、unstaged 和 untracked 内容，不能直接执行 `git switch`。那样可能报冲突，也可能改变远超这几个文件的内容。

最符合你要求的是“运行时代码文件覆盖”：当前分支仍保持 `molmo2-full-local`，只临时替换8个实际影响训练的文件，训练结束后原样恢复。

## 切换到 language-casual 版本

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song

REPO=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song
CASUAL_COMMIT=2265510

CASUAL_FILES=(
  benchmarks/song_real_libero/scripts/train_song_benchmark.py
  src/lerobot/configs/policies.py
  src/lerobot/policies/factory.py
  src/lerobot/policies/smolvla/configuration_smolvla.py
  src/lerobot/policies/smolvla/constants.py
  src/lerobot/policies/smolvla/modeling_smolvla.py
  src/lerobot/policies/smolvla/molmo2_full_with_expert.py
  src/lerobot/policies/smolvla/molmo2_with_expert.py
)

mkdir -p /home/liusong/.cache
BACKUP_DIR=$(mktemp -d /home/liusong/.cache/molmo2-full-local-before-language-casual.XXXXXX)

tar -C "$REPO" -cpf "$BACKUP_DIR/runtime-files.tar" "${CASUAL_FILES[@]}"

(
  cd "$REPO"
  sha256sum "${CASUAL_FILES[@]}"
) > "$BACKUP_DIR/runtime-files.sha256"

git restore \
  --source="$CASUAL_COMMIT" \
  --worktree \
  -- "${CASUAL_FILES[@]}"

echo "BACKUP_DIR=$BACKUP_DIR"
git branch --show-current
git diff --exit-code "$CASUAL_COMMIT" -- "${CASUAL_FILES[@]}"
```

最后一条 `git diff` 没有输出，说明这8个文件已经与 `language_casual` 提交完全一致。

此时：

```bash
git branch --show-current
```

仍会显示：

```text
molmo2-full-local
```

这是正常的：我们只覆盖了运行时代码，没有修改分支、HEAD 或 index，其他本地文件也没有变化。

训练时应使用新输出目录，并从头训练，不能 resume 旧 Feature-Align checkpoint。

## 训练完成后恢复 molmo2-full-local 文件

在同一个终端中执行：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song

tar -C "$REPO" -xpf "$BACKUP_DIR/runtime-files.tar"

(
  cd "$REPO"
  sha256sum -c "$BACKUP_DIR/runtime-files.sha256"
)

git branch --show-current
```

校验结果应全部显示：

```text
OK
```

这表示8个文件已经逐字节恢复到消融训练之前的状态，原来的 staged 状态也没有被修改。

如果换了终端，需要先将上面打印出来的路径重新赋值：

```bash
BACKUP_DIR=/home/liusong/.cache/molmo2-full-local-before-language-casual.xxxxxx
REPO=/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song
```

注意：`language_casual` 训练得到的 checkpoint 带有新 topology marker。恢复旧代码后加载它会被门禁拒绝，这是预期行为；后续评测该 checkpoint 时需要再次应用上述覆盖，或使用独立 worktree。