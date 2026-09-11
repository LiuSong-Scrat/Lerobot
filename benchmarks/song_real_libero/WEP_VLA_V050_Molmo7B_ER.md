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
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_dataset \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v043_multiview_data/libero_4suite_lerobot_cache \
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
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_4suite40tasks_from55k_after_90k \
  --job_name=molmo_wepvla_4suite40tasks_from55k_after_90k \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=0 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=0.000001 \
  --policy.camera_views=agentview \
  --policy.rgb_camera_views=agentview


# Resume

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
  --resume=true \
  --config_path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_4suite40tasks_from55k_after_90k_after50k_NEWDATA/checkpoints/backpack048000/pretrained_model/train_config.json \
  --policy.push_to_hub=false \
  --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/libero_data/lerobot_dataset \
  --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/mixture_setting/libero_data/lerobot_cache \
  --batch_size=20 \
  --gradient_accumulation_steps=1 \
  --global_batch_size=140 \
  --task_balanced_sampling=true \
  --num_workers=14 \
  --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_4suite40tasks_from55k_after_90k_after50k_NEWDATA_after48k \
  --job_name=molmo_wepvla_4suite40tasks_from55k_after_90k_after50k_NEWDATA_after48k \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=0 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=0.000001 \
  --policy.camera_views=agentview \
  --policy.rgb_camera_views=agentview



 