# Song Real + LIBERO Benchmark

This project bundles the benchmark workflow for the local point-cloud SmolVLA policy:

1. record BestMan RGB-D sequences
2. slice them into HDF5 episodes
3. add a gripper point cloud while keeping 50,000 total points
4. filter discontinuous trajectories
5. convert HDF5 episodes to a LeRobot dataset
6. cache pointseg foreground/background indices and pseudo labels
7. train or resume SmolVLA
8. run local point-cloud inference
9. run LIBERO online point-cloud inference/evaluation

The scripts are copied into this benchmark directory so the workflow is reproducible without relying on personal debug entrypoints. The model and dataset classes are still imported from the current `lerobot` source tree.

## Quick Start

Dry-run the default local pipeline:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py \
  --config benchmarks/song_real_libero/configs/local.json \
  --stage all \
  --dry-run
```

Run one stage:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py \
  --config benchmarks/song_real_libero/configs/local.json \
  --stage convert
```

`all` runs `collect -> hdf5 -> gripper -> check -> convert -> cache -> train`. Inference and LIBERO are explicit because they usually need a specific observation, PLY, checkpoint, or simulator setup.

## Real-Robot Data Pipeline

Record RGB-D:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py --stage collect
```

Build HDF5 from a recorded sequence:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py \
  --stage hdf5 \
  --segments 0:120
```

Add gripper points, filter continuity, convert to LeRobot, and build cache:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py --stage gripper
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py --stage check
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py --stage convert
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py --stage cache
```

Expected converted dataset outputs:

```text
data/lerobot_dataset/
  point_clouds/episode_000000.zarr
  world_ee_poses/episode_000000.npy
```

Real-robot HDF5 conversion and LIBERO collection use compressed zarr point clouds by default:

```text
data/libero_lerobot_dataset/
  point_clouds/episode_000000.zarr
  world_ee_poses/episode_000000.npy
```

Pointseg cache v2 is index-only: each shard stores `point_indices`, pseudo labels, weights, and scores. It does not duplicate `observation.point_cloud`; training reconstructs the cached sample from the dataset's episode point cloud storage. Motion priors are recomputed online only when explicitly needed.

When `--policy.pointseg_enable=true` and `--pointseg_sample_cache_dir` is omitted or points to a missing cache, training now computes the same motion-prior pseudo labels online once per DataLoader batch from current/future point clouds. This fallback uses CUDA by default when available; if CUDA is used, the training script forces DataLoader `num_workers=0` to avoid CUDA initialization inside forked worker processes. Set `SONG_POINTSEG_ONLINE=0` to disable this fallback, or set `SONG_POINTSEG_ONLINE_DEVICE=cpu` to keep multi-worker CPU loading. Tune `SONG_POINTSEG_ONLINE_CURRENT_POINTS`, `SONG_POINTSEG_ONLINE_FUTURE_POINTS`, and `SONG_POINTSEG_ONLINE_NN_CHUNK_SIZE` for debugging.

## Training

Resume from the local `ep_vla` checkpoint with the benchmark defaults:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/run_pipeline.py --stage train
```

Equivalent explicit command for `train_song_benchmark.py`:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/ep_vla/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/pointseg_cache \
  --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=false \
  --batch_size=4 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train \
  --job_name=song_real_libero_pointseg \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=10000 \
  --eval_freq=1000 \
  --num_workers=8 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.002 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=4000 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

For the generated LIBERO dataset, replace the dataset, cache, and output paths:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/ep_vla/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_pointseg_cache \
  --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=false \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh \
  --job_name=song_libero_pointseg \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=10000 \
  --eval_freq=1000 \
  --num_workers=8 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.002 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=4000 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

Fresh LIBERO training without loading a policy checkpoint:

```bash
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_pointseg_cache \
  --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=false \
  --batch_size=4 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh \
  --job_name=song_libero_pointseg_fresh \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=1000 \
  --eval_freq=1000 \
  --num_workers=4 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.002 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=4000 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

The default benchmark training path enables pointseg and action flow matching only:

```text
pointseg_enable=true
worldflow_enable=false
worldflow_se3_head_enable=false
se3_enable=false
```

Large W&B artifacts are disabled with `wandb.disable_artifact=true`.

## Local Inference

For `libero_object --task-id 0`, the task language is:

```text
pick up the alphabet soup and place it in the basket
```

Run from a real-robot observation pickle:

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model \
  --obs.path /home/liusong/temp/obs_dict_umi_trash.pkl \
  --task "pick up the alphabet soup and place it in the basket" \
  --num-points 50000 \
  --add-gripper-cloud \
  --gripper-points 500 \
  --gripper-template reap \
  --output-dir benchmarks/song_real_libero/outputs/infer_obs
```

Run from a LIBERO LeRobot dataset frame:

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model \
  --dataset.repo_id benchmarks/song_real_libero/data/libero_lerobot_dataset \
  --index 0 \
  --task "pick up the alphabet soup and place it in the basket" \
  --output-dir benchmarks/song_real_libero/outputs/infer_libero_dataset
```

Run from a local PLY point cloud:

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model \
  --ply.path benchmarks/song_real_libero/outputs/libero_collect_vis/episode_000000_demo_0/frame_0000_point_cloud_eff.ply \
  --num-points 50000 \
  --add-gripper-cloud \
  --gripper-points 500 \
  --gripper-template reap \
  --task "pick up the alphabet soup and place it in the basket" \
  --output-dir benchmarks/song_real_libero/outputs/infer_ply
```

For `obs.path`, the script adds the virtual gripper in the world frame using the current end-effector pose, then converts the merged cloud to the current end-effector frame. For `ply.path`, the input PLY is assumed to already be in the current end-effector frame. Dataset-frame inference uses the point cloud already stored in the LeRobot dataset. Outputs include `action.npy`, `result.json`, `trajectory.ply`, and, for observation/PLY inference, `model_point_cloud.npy`.

## LIBERO

Install LIBERO dependencies first:

```bash
pip install -e ".[libero]"
```

Generate a LIBERO point-cloud LeRobot dataset from LIBERO demonstration HDF5 files:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n reap python \
  benchmarks/song_real_libero/scripts/libero_collect_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_object \
  --task-id 0 \
  --episodes 1 \
  --num-workers 1 \
  --point-cloud-storage zarr
```

The benchmark scripts create `data/libero_config/config.yaml` automatically before importing LIBERO, so LIBERO's first-run interactive dataset prompt is bypassed.

Equivalent runner form:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n reap python \
  benchmarks/song_real_libero/scripts/run_pipeline.py \
  --stage libero_collect \
  --task-id 0 \
  --episodes 1
```

For faster collection, use episode-level multiprocessing:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_collect_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_object \
  --task-id 0 \
  --episodes 1 \
  --num-workers 1 \
  --save-video  \
  --vis-count 1
```
### ALL Object Data One Episode
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_collect_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_object \
  --episodes 5 \
  --num-workers 10 \
  --save-video  \
  --vis-count 1
```

### Four-Suite Training Dataset

If the demo HDF5 files are arranged under:

```text
benchmarks/song_real_libero/data/
  libero_spatial/*.hdf5
  libero_object/*.hdf5
  libero_goal/*.hdf5
  libero_10/*.hdf5
```

convert all tasks from the four suites into one LeRobot point-cloud dataset:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python \
  benchmarks/song_real_libero/scripts/libero_collect_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 5 \
  --num-workers 10 \
  --point-cloud-storage zarr \
  --output-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud \
  --save-video \
  --vis-count 2
```

Each episode keeps the task text from its own LIBERO task: LeRobot `task` uses `task.language`, and `libero_collect_summary.json` records `suite`, `task_id`, `task_name`, `task_language`, `problem_folder`, and `bddl_file`.

Build pointseg cache for the four-suite dataset:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id song_libero_4suite_pointcloud \
  --dataset.root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset \
  --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_pointseg_cache \
  --overwrite
```

The generated cache is compact: it saves point indices and pseudo-label tensors only. If you have an older cache with `point_cloud.npy` inside each shard, rebuild it with the command above to reclaim disk space.

Train on the generated four-suite dataset by replacing only the dataset/cache/output paths:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=song_libero_4suite_pointcloud \
  --dataset.root=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset \
  --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_pointseg_cache \
  --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=false \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_4suite_fresh \
  --job_name=song_libero_4suite_pointseg \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=10000 \
  --eval_freq=1000 \
  --num_workers=8 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.002 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=4000 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

Run online evaluation on the four suites with each task's own LIBERO language prompt:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python \
  benchmarks/song_real_libero/scripts/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_4suite_fresh/checkpoints/000100/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 10 \
  --action-index 0 \
  --exec-action-steps 16 \
  --no-replan-every-step \
  --gripper-wait-until-reached \
  --gripper-wait-tolerance 0.004 \
  --gripper-wait-max-steps 12 \
  --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/eval_libero_4suite
```


Each worker creates its own LIBERO/robosuite environment and writes lightweight temporary episode artifacts. The main process moves the final point-cloud arrays into the LeRobot dataset sequentially, so dataset writes remain deterministic without duplicating large point clouds through pickle files.

The generated dataset stores the LeRobot `task` field as the LIBERO task language. `libero_collect_summary.json` also includes `task_id`, `task_name`, `task_language`, `problem_folder`, and `bddl_file`.

`libero_collect_dataset.py` finds LIBERO demo HDF5 files under `libero.get_libero_path("datasets")` by default. If your demos are elsewhere, pass `--demo-root /path/to/libero/datasets` or `--demo-file /path/to/task_demo.hdf5`.

If no local demo HDF5 files are found, the default config downloads the matching official LIBERO demo package from Hugging Face. Disable this with `--no-download-demos`.

The generated training dataset is written to `dataset_output_root` in `configs/libero.json` and contains:

```text
data/libero_lerobot_dataset/
  point_clouds/episode_000000.zarr
  world_ee_poses/episode_000000.npy
  libero_collect_summary.json
```

The default config disables preview outputs for faster, smaller collection. To save previews, pass `--vis-count <N>` and/or `--save-video`; each preview episode can contain:

```text
frame_XXXX_point_cloud_eff.ply
frame_XXXX_umi_action_frame.ply
umi_action_trajectory.ply
world_ee_trajectory.ply
agentview_image.mp4
robot0_eye_in_hand_image.mp4
preview.json
```

`camera_names` controls RGB-D rendering and MP4 previews. `pointcloud_camera_names` controls which rendered cameras are fused into `observation.point_cloud`; the default is only `["agentview"]` because directly mixing the static scene camera with the wrist camera can produce duplicate crossed scenes if the camera frames are not perfectly aligned.

The LIBERO collector mirrors the real-robot point-cloud contract: it samples the scene cloud to `num_points`, adds the REAP-style virtual gripper at the current world end-effector pose, converts the merged cloud to the current end-effector frame, replaces the last `gripper_points` entries, and keeps the final shape at `(num_points, 6)`. With the default config this is `49500` scene points plus `500` virtual gripper points. The virtual gripper shape and offset follow the benchmark `add_gripper_cloud_to_hdf5.py` REAP template.

Export previews from an already generated dataset without recollecting:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/visualize_lerobot_pointcloud_dataset.py \
  --dataset-root benchmarks/song_real_libero/data/libero_lerobot_dataset \
  --episode 0 \
  --stride 20 \
  --count 8
```

Build pointseg cache:

```bash
conda run -n reap python benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id song_libero_pointcloud \
  --dataset.root benchmarks/song_real_libero/data/libero_lerobot_dataset \
  --output-dir benchmarks/song_real_libero/data/libero_pointseg_cache \
  --overwrite
```

Run online LIBERO inference/evaluation:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n reap python \
  benchmarks/song_real_libero/scripts/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model \
  --suite libero_object \
  --task-id 0 \
  --episodes 1
```

This online runner resets a LIBERO/robosuite environment, reads RGB-D observations each step, builds the same `50000 = 49500 scene + 500 gripper` point cloud, predicts `pose9 + gripper`, converts it to LIBERO's 7D relative action, steps the simulator, and reports success/reward.

## Notes

- `data/` and `outputs/` are ignored by git.
- The benchmark defaults use local `/home/liusong/...` paths. Copy and edit `configs/local.json` for server runs.
- If pointseg cache OOMs on an 8GB GPU, reduce `cache.batch_size`, `cache.future_points`, or `cache.nn_chunk_size`.




## ###########Data Collection################
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_collect_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --suite libero_object \
  --episodes 5 \
  --num-workers 10 \
  --point-cloud-storage zarr \
  --save-video  \
  --vis-count 1
```
## ###########Prior Cache################
```bash
conda run -n reap python benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id song_libero_pointcloud \
  --dataset.root benchmarks/song_real_libero/data/libero_lerobot_dataset \
  --output-dir benchmarks/song_real_libero/data/libero_pointseg_cache \
  --overwrite
```


## ###########Train Policy################   Resume
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset \
  --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=false \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh_post \
  --job_name=song_libero_pointseg_fresh1 \
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
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false

## ###########Train Policy################ Full Train
```bash
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset \
  --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=false \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=1 \
  --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh \
  --job_name=song_libero_pointseg_fresh \
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
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

### #######################
### #######################Data Colletct#############


```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python \
  benchmarks/song_real_libero/scripts/libero_collect_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --all-tasks \
  --episodes 25 \
  --num-workers 4 \
  --point-cloud-storage zarr \
  --output-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/temp_dataset \
  --repo-id song_libero_4suite_pointcloud \
  --save-video \
  --vis-count 2 
```
### #######################Prior Cache Generation#############

python benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id song_libero_pointcloud \
  --dataset.root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_4suite_lerobot_dataset \
  --output-dir benchmarks/song_real_libero/data/libero_pointseg_cache \
  --overwrite


### #######################Benchmark Eval#############
  ```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python \
  benchmarks/song_real_libero/scripts/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10  \
  --all-tasks \
  --episodes 10 \
  --action-index 0 \
  --exec-action-steps 16 \
  --no-replan-every-step \
  --gripper-wait-until-reached \
  --gripper-wait-tolerance 0.004 \
  --gripper-wait-max-steps 12 \
  --save-video \
  --render-mode viewer3d \
  --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/eval_libero_4suite
```


`libero_pointcloud_eval.py` saves rollout videos by default. Add `--no-save-video` to disable video output for faster evaluation.
By default evaluation predicts one SmolVLA action chunk and executes the first 16 actions before replanning. Gripper execution follows the predicted continuous gripper width from the same action row as the arm target, with no smoothing, thresholding, deadband, or artificial open/close labels. When the gripper target width changes, the runner repeats that same action row until the measured gripper width reaches `gripper_wait_tolerance` or `gripper_wait_max_steps` is exhausted, so the arm does not advance to the next chunk row before the close/open command has been applied. Each episode saves `model_pose_actions.npy`, `gripper_targets.npy`, `gripper_actuals.npy`, `gripper_width_errors.npy`, `gripper_wait_flags.npy`, and `gripper_progress_indices.npy` for debugging this alignment.
During interactive evaluation, press `v` in the terminal to save the latest predicted UMI action chunk visualization under `<output-dir>/keyboard_vis/`. The default mode writes PLY/NPZ files and is safe for `MUJOCO_GL=egl`; use `--keyboard-vis-mode window` only on a local desktop session with working Open3D/GLX.

For local 3D debugging on a desktop session, use the MuJoCo viewer mode instead of EGL:

`--headed` is equivalent to `--render-mode viewer3d`. Use `--render-mode onscreen` only if you want robosuite's OpenCV camera window instead of the interactive MuJoCo 3D viewer.
Do not set `PYOPENGL_PLATFORM=glfw`; robosuite may still initialize an EGL context for offscreen RGB-D, and that combination causes an import error.
