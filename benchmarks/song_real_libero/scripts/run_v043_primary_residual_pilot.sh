#!/usr/bin/env bash
set -euo pipefail

repo=/home/liusong/ProgramFiles/Huggingface/lerobot
root=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811
pilot="$root/pilots/libero10_task08_moka_50ep"
dataset=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/libero_4suite_lerobot_dataset
policy="$root/training/stage1_primary_residual_controls/alpha_0p00_zero_init/pretrained_model"
cache="$pilot/cache/pointseg_primary_residual"
smoke="$pilot/artifacts/training_smoke_primary_residual_4gpu_b48_w12_2steps"
train="$pilot/training/stage1_primary_residual_twotimescale_4gpu_b48_w12_100steps"
python=/home/liusong/anaconda3/envs/reap/bin/python3.10
torchrun=/home/liusong/anaconda3/envs/reap/bin/torchrun
vlm=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct
vlm_weights=/opt/data/private/liusong/hf_models/smolvla_base
episodes="[$(seq -s, 400 449)]"

mkdir -p "$pilot/cache" "$pilot/logs" "$pilot/artifacts" "$pilot/training"
cd "$repo"

if [[ ! -s "$cache/manifest.json" ]]; then
  if [[ -n "$(find "$cache" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Incomplete cache exists; refusing overwrite: $cache" >&2
    exit 1
  fi
  mkdir -p "$cache"
  SONG_POINTCLOUD_GRIPPER_POINTS=500 "$python" "$torchrun" --standalone --nproc_per_node=4 \
    benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
    --dataset.repo_id="$dataset" --episodes=400:450 \
    --camera-views=agentview,robot0_eye_in_hand \
    --camera-view-fusion=primary_residual \
    --output-dir="$cache" --batch-size=24 --num-workers=4 \
    --shard-size=2048 --storage-dtype=float16 --nn-chunk-size=1024 --vis-count=4 \
    2>&1 | tee -a "$pilot/logs/cache_task08_50ep_primary_residual.log"
fi

PYTHONPATH=src "$python" - "$cache" <<'PY'
import json, sys
from pathlib import Path
from lerobot.policies.smolvla.song_pointseg import SongPointSegCachedDataset
root = Path(sys.argv[1])
manifest = json.load(open(root / "manifest.json"))
assert manifest["num_samples"] == 20744
assert manifest["camera_view_fusion"] == "primary_residual"
assert manifest["point_count_policy"] == "primary_exact_labels_secondary_residual_raw_union"
assert manifest["primary_residual_contract"]["model_input_points"] == 19500
cache = SongPointSegCachedDataset(root)
for index in (0, len(cache) // 2, len(cache) - 1):
    item = cache[index]
    assert item["pointseg.labels"].numel() == 10000
print(f"primary residual cache PASS: samples={len(cache)}")
PY

train_once() {
  local output=$1
  local steps=$2
  local save_freq=$3
  local tag=$4
  if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Training output is not empty; refusing overwrite: $output" >&2
    exit 1
  fi
  mkdir -p "$output"
  ulimit -n 65535
  SONG_POINTSEG_REQUIRE_POINTOPS=1 OMP_NUM_THREADS=1 "$python" -m accelerate.commands.launch \
    --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=no \
    --dynamo_backend=no --main_process_port=0 \
    benchmarks/song_real_libero/scripts/train_song_benchmark.py \
    --policy.path="$policy" --policy.push_to_hub=false \
    --dataset.repo_id="$dataset" --dataset.episodes="$episodes" \
    --pointseg_sample_cache_dir="$cache" \
    --policy.camera_views=agentview,robot0_eye_in_hand \
    --policy.camera_view_fusion=primary_residual --policy.rgb_camera_views=agentview \
    --policy.vla_adapter_enable=true --policy.vla_adapter_freeze_vlm=true \
    --policy.vlm_model_name="$vlm" --policy.vlm_weights_path="$vlm_weights" \
    --policy.load_vlm_weights=true --batch_size=48 --steps="$steps" \
    --log_freq=1 --output_dir="$output" --job_name="$tag" \
    --policy.device=cuda --wandb.enable=false --wandb.disable_artifact=true \
    --save_freq="$save_freq" --eval_freq="$save_freq" --num_workers=12 \
    --policy.optimizer_lr=0.000025 --policy.scheduler_warmup_steps=10 \
    --policy.scheduler_decay_steps="$steps" --policy.scheduler_decay_lr=0.0000025 \
    --policy.multiview_pretrained_lr_multiplier=0.1 \
    --policy.multiview_residual_lr_multiplier=1.0 \
    --policy.pointseg_enable=true --policy.pointseg_backbone_type=litept \
    --policy.pointseg_grid_size=0.01 --policy.pointseg_feature_dim=64 \
    --policy.pointseg_aux_loss_weight=0.0005 \
    --policy.pointseg_foreground_ratio=0.025 --policy.pointseg_background_ratio=0.025 \
    --policy.pointseg_min_foreground_points=2500 --policy.pointseg_min_background_points=0 \
    --policy.pointseg_use_temporal_priors_as_input=false \
    --policy.pointseg_use_pseudo_selection=false \
    --policy.worldflow_enable=false --policy.worldflow_se3_head_enable=false \
    --policy.se3_enable=false --policy.se3_final_correction_enable=false \
    2>&1 | tee -a "$pilot/logs/train_${tag}.log"
}

train_once "$smoke" 2 1 task08_primary_residual_smoke_4gpu_b48_w12_2steps
test -s "$smoke/checkpoints/000002/pretrained_model/model.safetensors"
train_once "$train" 100 25 task08_primary_residual_twotimescale_4gpu_b48_w12_100steps
test -s "$train/checkpoints/000100/pretrained_model/model.safetensors"
echo "primary residual task08 pilot training complete"
