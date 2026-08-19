cd /home/liusong/ProgramFiles/Huggingface/lerobot

PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
/home/liusong/anaconda3/envs/reap/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=6 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_dataset \
  --camera-views=agentview,robot0_eye_in_hand \
  --camera-view-fusion=fps \
  --output-dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_union_fps_cache \
  --current-points=10000 \
  --future-points=10000 \
  --batch-size=24 \
  --num-workers=12 \
  --shard-size=4096 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4




/home/liusong/anaconda3/envs/reap/bin/python - <<'PY'
import json
from pathlib import Path

cache = Path(
    "/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_union_fps_cache"
)

with open(cache / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

print("num_samples:", manifest["num_samples"])
print("camera_views:", manifest["camera_views"])
print("camera_view_fusion:", manifest["camera_view_fusion"])
print("camera_view_weights:", manifest.get("camera_view_weights"))
print("gripper_points:", manifest["gripper_points"])

assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"]
assert manifest["camera_view_fusion"] == "fps"
assert manifest.get("camera_view_weights") is None
assert manifest["gripper_points"] == 500
PY



cd /home/liusong/ProgramFiles/Huggingface/lerobot

PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=/home/liusong/ProgramFiles/Huggingface/lerobot/src \
SONG_POINTSEG_REQUIRE_POINTOPS=1 \
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
/home/liusong/anaconda3/envs/reap/bin/python \
  -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=0 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_dataset \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_union_fps_cache \
  --task_balanced_sampling=true \
  --batch_size=160 \
  --gradient_accumulation_steps=1 \
  --steps=20000 \
  --save_freq=1000 \
  --log_freq=1 \
  --eval_freq=1000 \
  --num_workers=12 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v043_dualview_fps_3gpu_b48_lr5e6_8k_20260819 \
  --job_name=wep_vla_v043_dualview_fps_3gpu_b48_lr5e6_8k_20260819 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=20000 \
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
  --policy.pointseg_freeze_batchnorm_stats=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false