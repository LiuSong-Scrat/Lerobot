#!/usr/bin/env bash
set -euo pipefail

repo=/home/liusong/ProgramFiles/Huggingface/lerobot_singleview_object_worldflow
python=/home/liusong/anaconda3/envs/reap/bin/python
torchrun=/home/liusong/anaconda3/envs/reap/bin/torchrun
dataset=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep
cache=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache
baseline=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model
training=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/world_eef_task6_task8_100ep_bootstrap_4gpu_b24_1564steps
experiment=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/libero_setting/world_eef_task6_task8_100ep_20260816
log_dir="$experiment/logs"
mkdir -p "$log_dir"

audit_dataset() {
    PYTHONPATH="$repo/src" "$python" - "$dataset" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
summary = json.loads((root / "libero_collect_summary.json").read_text())
episodes = summary["episodes"]
assert info["total_episodes"] == 100, info["total_episodes"]
assert summary["num_episodes"] == 100, summary["num_episodes"]
assert Counter(int(item["task_id"]) for item in episodes) == {6: 50, 8: 50}
assert summary["robot_base_worldflow"] == {
    "coordinate_frame": "robot_base",
    "target": "commanded model-EEF trajectory",
    "implicit_point_flow": True,
    "explicit_object_pose_supervision": False,
}

current_meta = json.loads((root / "world_base_ee_poses/meta.json").read_text())
target_meta = json.loads((root / "world_base_action_target_ee_poses/meta.json").read_text())
assert current_meta["coordinate_frame"] == "robot_base"
assert target_meta["coordinate_frame"] == "robot_base"
assert target_meta["target_semantics"] == "commanded_eef_pose"

for episode_index, episode in enumerate(episodes):
    current = np.load(root / f"world_base_ee_poses/episode_{episode_index:06d}.npy")
    target = np.load(root / f"world_base_action_target_ee_poses/episode_{episode_index:06d}.npy")
    expected_frames = int(episode["frames"])
    assert current.shape == (expected_frames, 9), (episode_index, current.shape, expected_frames)
    assert target.shape == (expected_frames, 9), (episode_index, target.shape, expected_frames)
    assert np.isfinite(current).all(), episode_index
    assert np.isfinite(target).all(), episode_index
    assert (root / f"point_clouds/episode_{episode_index:06d}.zarr").is_dir(), episode_index

print(
    "dataset audit PASS: "
    f"episodes={info['total_episodes']} frames={info['total_frames']} "
    "task_counts={6: 50, 8: 50} frame=robot_base target=commanded_eef_pose"
)
PY
}

build_cache() {
    audit_dataset
    if [[ -s "$cache/manifest.json" ]]; then
        echo "cache already complete: $cache"
        return
    fi
    if [[ -n "$(find "$cache" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "incomplete cache exists; refusing to overwrite: $cache" >&2
        exit 1
    fi
    mkdir -p "$cache"
    cd "$repo"
    SONG_POINTCLOUD_GRIPPER_POINTS=500 OMP_NUM_THREADS=1 "$torchrun" \
        --standalone --nproc_per_node=4 \
        benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
        --dataset.repo_id="$dataset" \
        --camera-views=agentview \
        --camera-view-fusion=legacy_budget \
        --output-dir="$cache" \
        --current-points=10000 --future-points=10000 \
        --batch-size=24 --num-workers=4 \
        --shard-size=2048 --storage-dtype=float16 \
        --nn-chunk-size=1024 --vis-count=4 \
        2>&1 | tee -a "$log_dir/cache.log"
    test -s "$cache/manifest.json"
}

audit_cache() {
    audit_dataset
    PYTHONPATH="$repo/src" "$python" - "$dataset" "$cache" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.policies.smolvla.song_pointseg import SongPointSegCachedDataset

dataset_root = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
info = json.loads((dataset_root / "meta/info.json").read_text())
manifest = json.loads((cache_root / "manifest.json").read_text())
assert manifest["num_samples"] == info["total_frames"], (manifest["num_samples"], info["total_frames"])
assert manifest["camera_views"] == ["agentview"], manifest["camera_views"]
cache = SongPointSegCachedDataset(cache_root)
assert len(cache) == info["total_frames"]
for index in (0, len(cache) // 2, len(cache) - 1):
    item = cache[index]
    assert item["pointseg.labels"].numel() == 10000, index
print(f"cache audit PASS: samples={len(cache)} camera_views={manifest['camera_views']}")
PY
}

train() {
    audit_cache
    if [[ -n "$(find "$training" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "training output is not empty; refusing to overwrite: $training" >&2
        exit 1
    fi
    mkdir -p "$training"
    cd "$repo"
    ulimit -n 65535
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    SONG_POINTSEG_REQUIRE_POINTOPS=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$python" -m accelerate.commands.launch \
        --multi_gpu --num_processes=4 --num_machines=1 \
        --mixed_precision=no --dynamo_backend=no --main_process_port=0 \
        benchmarks/song_real_libero/scripts/train_song_benchmark.py \
        --policy.path="$baseline" --policy.push_to_hub=false \
        --dataset.repo_id="$dataset" \
        --pointseg_sample_cache_dir="$cache" \
        --task_balanced_sampling=true \
        --batch_size=24 --gradient_accumulation_steps=1 \
        --steps=1564 --save_freq=1564 \
        --save_steps='[100,260,520,780,1040,1300,1564]' \
        --log_freq=1 --eval_freq=1564 --num_workers=12 \
        --output_dir="$training" \
        --job_name=world_eef_task6_task8_100ep_bootstrap_4gpu_b24_1564steps \
        --policy.device=cuda --wandb.enable=false --wandb.disable_artifact=true \
        --policy.optimizer_lr=0.000025 \
        --policy.scheduler_warmup_steps=50 \
        --policy.scheduler_decay_steps=1564 \
        --policy.scheduler_decay_lr=0.0000025 \
        --policy.camera_views=agentview --policy.rgb_camera_views=agentview \
        --policy.vla_adapter_enable=true --policy.vla_adapter_freeze_vlm=true \
        --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
        --policy.vlm_weights_path=/opt/data/private/liusong/hf_models/smolvla_base \
        --policy.load_vlm_weights=true \
        --policy.pointseg_enable=true --policy.pointseg_backbone_type=litept \
        --policy.pointseg_grid_size=0.01 --policy.pointseg_feature_dim=64 \
        --policy.pointseg_aux_loss_weight=0.0005 \
        --policy.pointseg_foreground_ratio=0.025 --policy.pointseg_background_ratio=0.025 \
        --policy.pointseg_min_foreground_points=2500 --policy.pointseg_min_background_points=0 \
        --policy.pointseg_use_temporal_priors_as_input=false \
        --policy.pointseg_use_pseudo_selection=false \
        --policy.point_action_fusion_enable=true \
        --policy.worldflow_enable=true \
        --policy.worldflow_target_type=world_eef_trajectory \
        --policy.worldflow_reference_frame=robot_base \
        --policy.worldflow_frame_origin=global \
        --policy.worldflow_scene_frame_origin=global \
        --policy.worldflow_noise_coupling=independent \
        --policy.worldflow_action_fusion=cross_attention \
        --policy.worldflow_bootstrap_from_ego=true \
        --policy.worldflow_feature_dim=64 --policy.worldflow_grid_size=0.01 \
        --policy.worldflow_max_points=2048 \
        --policy.worldflow_loss_weight=0.02 \
        --policy.worldflow_geo_loss_weight=0.002 \
        --policy.worldflow_bridge_loss_weight=0.0 \
        --policy.worldflow_equiv_loss_weight=0.0 \
        --policy.worldflow_training_coordinate_frame_augmentation=false \
        --policy.worldflow_pretrained_lr_multiplier=0.2 \
        --policy.worldflow_new_lr_multiplier=1.0 \
        --policy.worldflow_trans_weight=1.0 --policy.worldflow_rot_weight=1.0 \
        --policy.worldflow_require_action_target_sidecar=true \
        --policy.worldflow_se3_head_enable=false \
        --policy.se3_enable=false --policy.se3_final_correction_enable=false \
        2>&1 | tee -a "$log_dir/train.log"
    test -s "$training/checkpoints/001564/pretrained_model/model.safetensors"
}

eval_checkpoint() {
    local checkpoint=${2:?usage: $0 eval CHECKPOINT [TAG] [EPISODES] [ABLATED]}
    local tag=${3:-$(basename "$(dirname "$(dirname "$checkpoint")")")}
    local episodes=${4:-50}
    local ablated=${5:-false}
    local output="$experiment/eval/${tag}_dual_${episodes}ep"
    local ablation_flag=--no-world-to-ego-causal-ablation
    if [[ "$ablated" == true ]]; then
        output="$experiment/eval/${tag}_world_to_ego_disabled_${episodes}ep"
        ablation_flag=--world-to-ego-causal-ablation
    fi
    if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "evaluation output is not empty; refusing to overwrite: $output" >&2
        exit 1
    fi
    mkdir -p "$output"
    cd "$repo"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
        --config benchmarks/song_real_libero/configs/libero.json \
        --policy.path "$checkpoint" \
        --suite libero_10 --task-id 6 --task-id 8 --episodes "$episodes" \
        --policy-noise-seed 0 --env-seed 7 --strict-official-init \
        --gripper-control-mode delta_width_initial_sync \
        --gripper-delta-threshold 0.002 \
        --gripper-delta-alignment current_minus_previous \
        --waypoint-max-hold-steps 1 \
        --isolated-policy-workers 1 --task-workers 10 \
        --episode-workers-per-task 2 --task-worker-backend process \
        --inference-batch-size 80 --inference-batching-mode fixed_barrier \
        --no-release-event-exec-enable \
        --control-freq 20 --action-index 0 \
        --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
        --render-mode offscreen --no-visualize-foreground --save-video \
        "$ablation_flag" --output-dir "$output" \
        2>&1 | tee -a "$log_dir/eval_${tag}_${episodes}ep_${ablated}.log"
}

case "${1:-}" in
    audit) audit_dataset ;;
    cache) build_cache; audit_cache ;;
    train) train ;;
    pipeline) build_cache; audit_cache; train ;;
    eval) eval_checkpoint "$@" ;;
    *)
        echo "usage: $0 {audit|cache|train|pipeline|eval CHECKPOINT [TAG] [EPISODES] [ABLATED]}" >&2
        exit 2
        ;;
esac
