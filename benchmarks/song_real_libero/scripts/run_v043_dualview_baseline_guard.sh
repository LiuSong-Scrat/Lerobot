#!/usr/bin/env bash
# Managed entrypoint for the guarded dual-view -> WorldFlow experiment.
# Every generated artifact and log is rooted below SONG_V043_EXPERIMENT_ROOT.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
env_file="$repo_root/benchmarks/song_real_libero/configs/v043_dualview_baseline_guard.env"

if [[ ! -f "$env_file" ]]; then
    echo "Missing experiment environment file: $env_file" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$env_file"

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory is missing: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file is missing: $1" >&2
        exit 1
    fi
}

prepare_runtime_dirs() {
    mkdir -p \
        "$SONG_V043_CACHE_ROOT" \
        "$SONG_V043_TRAIN_ROOT" \
        "$SONG_V043_EVAL_ROOT" \
        "$SONG_V043_EXPERIMENT_ROOT/artifacts" \
        "$SONG_V043_EXPERIMENT_ROOT/checkpoints" \
        "$SONG_V043_EXPERIMENT_ROOT/logs" \
        "$WANDB_DIR"
}

check_gpu_idle() {
    if [[ "${SONG_V043_ALLOW_BUSY_GPU:-0}" == "1" ]]; then
        return
    fi
    local threshold_mib=${SONG_V043_GPU_IDLE_THRESHOLD_MIB:-2048}
    local used_mib
    while IFS= read -r used_mib; do
        used_mib=${used_mib//[[:space:]]/}
        if (( used_mib > threshold_mib )); then
            echo "GPU memory use (${used_mib} MiB) exceeds the ${threshold_mib} MiB safety threshold." >&2
            echo "Wait for the existing job, or explicitly set SONG_V043_ALLOW_BUSY_GPU=1." >&2
            exit 1
        fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
}

run_accelerate_4gpu() {
    OMP_NUM_THREADS=1 "$SONG_V043_PYTHON" -m accelerate.commands.launch \
        --multi_gpu \
        --num_processes=4 \
        --num_machines=1 \
        --mixed_precision=no \
        --dynamo_backend=no \
        --main_process_port=0 \
        "$@"
}

preflight() {
    prepare_runtime_dirs
    require_dir "$SONG_V043_DATASET_ROOT"
    require_file "$SONG_V043_DATASET_ROOT/meta/info.json"
    require_dir "$SONG_V043_DATASET_ROOT/point_clouds"
    require_dir "$SONG_V043_DATASET_ROOT/point_clouds_robot0_eye_in_hand"
    require_dir "$SONG_V043_BASELINE_CKPT"
    require_file "$SONG_V043_BASELINE_CKPT/config.json"
    require_file "$SONG_V043_BASELINE_CKPT/model.safetensors"
    require_dir "$SONG_V043_VLM_MODEL"
    require_dir "$SONG_V043_VLM_WEIGHTS"
    require_file "$SONG_V043_PYTHON"
    require_file "$SONG_V043_TORCHRUN"
    require_file "$repo_root/benchmarks/song_real_libero/configs/v043_dualview_baseline_guard_manifest.json"
    require_file "$repo_root/benchmarks/song_real_libero/scripts/probe_v043_dualview_action_drift.py"

    local expected_sha actual_sha
    expected_sha=$("$SONG_V043_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline"]["model_sha256"])' \
        "$repo_root/benchmarks/song_real_libero/configs/v043_dualview_baseline_guard_manifest.json")
    actual_sha=$(sha256sum "$SONG_V043_BASELINE_CKPT/model.safetensors" | awk '{print $1}')
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "Baseline checkpoint hash mismatch: expected=$expected_sha actual=$actual_sha" >&2
        exit 1
    fi

    PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$SONG_V043_PYTHON" - "$SONG_V043_BASELINE_CKPT" <<'PY'
import json
import sys
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.song_pointseg import parse_camera_view_fusion, parse_camera_views

with open(f"{sys.argv[1]}/config.json", encoding="utf-8") as stream:
    payload = json.load(stream)
payload.pop("type", None)
payload.update(
    camera_views="agentview,robot0_eye_in_hand",
    camera_view_weights=None,
    camera_view_fusion="fps",
    rgb_camera_views="agentview",
)
cfg = SmolVLAConfig(**payload)
views = parse_camera_views(cfg.camera_views)
fusion = parse_camera_view_fusion(cfg.camera_view_fusion)
assert views == ("agentview", "robot0_eye_in_hand"), views
assert fusion == "fps", fusion
assert cfg.camera_view_weights is None
assert not cfg.worldflow_enable
assert not cfg.se3_enable
print(f"preflight config: views={views} fusion={fusion} worldflow={cfg.worldflow_enable} se3={cfg.se3_enable}")
PY
    echo "preflight baseline_sha256=$actual_sha"
    echo "preflight experiment_root=$SONG_V043_EXPERIMENT_ROOT"
}

run_cache() {
    preflight
    check_gpu_idle
    if [[ -n "$(find "$SONG_V043_CACHE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Cache directory is not empty; refusing an implicit rebuild: $SONG_V043_CACHE_ROOT" >&2
        exit 1
    fi
    local log_file="$SONG_V043_EXPERIMENT_ROOT/logs/stage1_cache_fps_union.log"
    cd "$repo_root"
    SONG_POINTCLOUD_GRIPPER_POINTS=500 \
        "$SONG_V043_PYTHON" "$SONG_V043_TORCHRUN" --standalone --nproc_per_node=4 \
        benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
        --dataset.repo_id="$SONG_V043_DATASET_ROOT" \
        --camera-views=agentview,robot0_eye_in_hand \
        --camera-view-fusion=fps \
        --output-dir="$SONG_V043_CACHE_ROOT" \
        --batch-size=24 \
        --num-workers=4 \
        --shard-size=4096 \
        --storage-dtype=float16 \
        --nn-chunk-size=1024 \
        --vis-count=4 2>&1 | tee -a "$log_file"
}

validate_cache() {
    require_file "$SONG_V043_CACHE_ROOT/manifest.json"
    "$SONG_V043_PYTHON" - "$SONG_V043_CACHE_ROOT/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"], manifest["camera_views"]
assert manifest["camera_view_weights"] is None, manifest["camera_view_weights"]
assert manifest["camera_view_fusion"] == "fps", manifest["camera_view_fusion"]
assert manifest["gripper_points"] == 500, manifest["gripper_points"]
assert manifest["num_samples"] == 137590, manifest["num_samples"]
assert manifest["trajectory_offset_filtering"] == "relative_frame_offsets", manifest.get("trajectory_offset_filtering")
assert manifest["shards"], "cache manifest has no shards"
print(f"cache manifest: samples={manifest['num_samples']} shards={len(manifest['shards'])}")
PY
}

run_train() {
    preflight
    validate_cache
    check_gpu_idle
    local output_root=${SONG_V043_STAGE1_TRAIN_OUTPUT_ROOT:-$SONG_V043_STAGE1_TRAIN_ROOT}
    local steps=${SONG_V043_STAGE1_TRAIN_STEPS:-2000}
    local save_freq=${SONG_V043_STAGE1_TRAIN_SAVE_FREQ:-200}
    local eval_freq=${SONG_V043_STAGE1_TRAIN_EVAL_FREQ:-200}
    local num_workers=${SONG_V043_STAGE1_TRAIN_NUM_WORKERS:-12}
    local batch_size_per_gpu=${SONG_V043_STAGE1_TRAIN_BATCH_SIZE_PER_GPU:-48}
    local freeze_pointseg_bn_stats=${SONG_V043_STAGE1_TRAIN_FREEZE_POINTSEG_BN_STATS:-false}
    local optimizer_lr=${SONG_V043_STAGE1_TRAIN_LR:-0.0001}
    local scheduler_warmup_steps=${SONG_V043_STAGE1_TRAIN_WARMUP_STEPS:-100}
    local scheduler_decay_steps=${SONG_V043_STAGE1_TRAIN_DECAY_STEPS:-30000}
    local scheduler_decay_lr=${SONG_V043_STAGE1_TRAIN_DECAY_LR:-0.0000025}
    local run_tag=${SONG_V043_STAGE1_TRAIN_TAG:-wep_vla_v043_dualview_fps_union_4gpu_b48_baseline_guard}
    if [[ ! "$steps" =~ ^[0-9]+$ ]] || (( steps < 1 )); then
        echo "SONG_V043_STAGE1_TRAIN_STEPS must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "$save_freq" =~ ^[0-9]+$ ]] || (( save_freq < 1 )); then
        echo "SONG_V043_STAGE1_TRAIN_SAVE_FREQ must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "$eval_freq" =~ ^[0-9]+$ ]] || (( eval_freq < 1 )); then
        echo "SONG_V043_STAGE1_TRAIN_EVAL_FREQ must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "$num_workers" =~ ^[0-9]+$ ]] || (( num_workers < 1 )); then
        echo "SONG_V043_STAGE1_TRAIN_NUM_WORKERS must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "$batch_size_per_gpu" =~ ^[0-9]+$ ]] || (( batch_size_per_gpu < 1 )); then
        echo "SONG_V043_STAGE1_TRAIN_BATCH_SIZE_PER_GPU must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "$run_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsafe Stage-1 training tag: $run_tag" >&2
        exit 1
    fi
    if [[ "$freeze_pointseg_bn_stats" != "true" && "$freeze_pointseg_bn_stats" != "false" ]]; then
        echo "SONG_V043_STAGE1_TRAIN_FREEZE_POINTSEG_BN_STATS must be true or false." >&2
        exit 1
    fi
    if [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "Training directory is not empty; refusing to mix or overwrite runs: $output_root" >&2
        exit 1
    fi
    mkdir -p "$output_root"
    local log_file="$SONG_V043_EXPERIMENT_ROOT/logs/stage1_train_dualview_fps_union_4gpu_b48.log"
    cd "$repo_root"
    ulimit -n 65535
    export SONG_POINTSEG_REQUIRE_POINTOPS=1
    run_accelerate_4gpu benchmarks/song_real_libero/scripts/train_song_benchmark.py \
        --policy.path="$SONG_V043_BASELINE_CKPT" \
        --policy.push_to_hub=false \
        --dataset.repo_id="$SONG_V043_DATASET_ROOT" \
        --pointseg_sample_cache_dir="$SONG_V043_CACHE_ROOT" \
        --policy.camera_views=agentview,robot0_eye_in_hand \
        --policy.camera_view_fusion=fps \
        --policy.rgb_camera_views=agentview \
        --policy.vla_adapter_enable=true \
        --policy.vla_adapter_freeze_vlm=true \
        --policy.vlm_model_name="$SONG_V043_VLM_MODEL" \
        --policy.vlm_weights_path="$SONG_V043_VLM_WEIGHTS" \
        --policy.load_vlm_weights=true \
        --batch_size="$batch_size_per_gpu" \
        --steps="$steps" \
        --log_freq=1 \
        --output_dir="$output_root" \
        --job_name="$run_tag" \
        --policy.device=cuda \
        --wandb.enable=false \
        --wandb.disable_artifact=true \
        --save_freq="$save_freq" \
        --eval_freq="$eval_freq" \
        --num_workers="$num_workers" \
        --policy.optimizer_lr="$optimizer_lr" \
        --policy.scheduler_warmup_steps="$scheduler_warmup_steps" \
        --policy.scheduler_decay_steps="$scheduler_decay_steps" \
        --policy.scheduler_decay_lr="$scheduler_decay_lr" \
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
        --policy.pointseg_freeze_batchnorm_stats="$freeze_pointseg_bn_stats" \
        --policy.worldflow_enable=false \
        --policy.worldflow_se3_head_enable=false \
        --policy.se3_enable=false \
        --policy.se3_final_correction_enable=false 2>&1 | tee -a "$log_file"
}

run_train_worldflow() {
    preflight
    validate_cache
    check_gpu_idle
    local stage1_ckpt=${SONG_V043_STAGE1_CKPT:-}
    local output_root=${SONG_V043_STAGE2_TRAIN_OUTPUT_ROOT:-$SONG_V043_STAGE2_TRAIN_ROOT}
    local steps=${SONG_V043_STAGE2_TRAIN_STEPS:-2000}
    local save_freq=${SONG_V043_STAGE2_TRAIN_SAVE_FREQ:-200}
    local eval_freq=${SONG_V043_STAGE2_TRAIN_EVAL_FREQ:-200}
    local num_workers=${SONG_V043_STAGE2_TRAIN_NUM_WORKERS:-12}
    local batch_size_per_gpu=${SONG_V043_STAGE2_TRAIN_BATCH_SIZE_PER_GPU:-24}
    local gradient_accumulation_steps=${SONG_V043_STAGE2_TRAIN_GRAD_ACCUM_STEPS:-2}
    local warmup_steps=${SONG_V043_STAGE2_TRAIN_WARMUP_STEPS:-50}
    local optimizer_lr=${SONG_V043_STAGE2_TRAIN_OPTIMIZER_LR:-0.0000025}
    local scheduler_decay_lr=${SONG_V043_STAGE2_TRAIN_DECAY_LR:-0.00000025}
    local pretrained_lr_multiplier=${SONG_V043_STAGE2_PRETRAINED_LR_MULTIPLIER:-0.2}
    local new_lr_multiplier=${SONG_V043_STAGE2_NEW_LR_MULTIPLIER:-4.0}
    local worldflow_loss_weight=${SONG_V043_STAGE2_FLOW_LOSS_WEIGHT:-0.02}
    local worldflow_geo_loss_weight=${SONG_V043_STAGE2_GEO_LOSS_WEIGHT:-0.002}
    local worldflow_bridge_loss_weight=${SONG_V043_STAGE2_BRIDGE_LOSS_WEIGHT:-0.005}
    local worldflow_equiv_loss_weight=${SONG_V043_STAGE2_EQUIV_LOSS_WEIGHT:-0.001}
    local bootstrap_from_ego=${SONG_V043_STAGE2_BOOTSTRAP_FROM_EGO:-false}
    local world_se3_head_enable=${SONG_V043_STAGE2_WORLD_SE3_HEAD_ENABLE:-false}
    local action_fusion=${SONG_V043_STAGE2_ACTION_FUSION:-conjugate_residual_consensus}
    local run_tag=${SONG_V043_STAGE2_TRAIN_TAG:-wep_vla_v043_libero10_worldego_conjugate_residual_consensus_4gpu_b24_accum2}
    if [[ -z "$stage1_ckpt" ]]; then
        echo "Set SONG_V043_STAGE1_CKPT to an immutable Stage 1 checkpoint that passed evaluation." >&2
        exit 1
    fi
    require_dir "$stage1_ckpt"
    require_file "$stage1_ckpt/config.json"
    require_file "$stage1_ckpt/model.safetensors"
    if [[ ! "$run_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsafe WorldFlow training tag: $run_tag" >&2
        exit 1
    fi
    if [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "WorldFlow directory is not empty; refusing to mix or overwrite runs: $output_root" >&2
        exit 1
    fi
    mkdir -p "$output_root"
    local log_file="$SONG_V043_EXPERIMENT_ROOT/logs/train_${run_tag}.log"
    cd "$repo_root"
    ulimit -n 65535
    export SONG_POINTSEG_REQUIRE_POINTOPS=1
    run_accelerate_4gpu benchmarks/song_real_libero/scripts/train_song_benchmark.py \
        --policy.path="$stage1_ckpt" \
        --policy.push_to_hub=false \
        --dataset.repo_id="$SONG_V043_DATASET_ROOT" \
        --pointseg_sample_cache_dir="$SONG_V043_CACHE_ROOT" \
        --policy.camera_views=agentview,robot0_eye_in_hand \
        --policy.camera_view_fusion=fps \
        --policy.rgb_camera_views=agentview \
        --policy.vla_adapter_enable=true \
        --policy.vla_adapter_freeze_vlm=true \
        --policy.vlm_model_name="$SONG_V043_VLM_MODEL" \
        --policy.vlm_weights_path="$SONG_V043_VLM_WEIGHTS" \
        --policy.load_vlm_weights=true \
        --batch_size="$batch_size_per_gpu" \
        --gradient_accumulation_steps="$gradient_accumulation_steps" \
        --steps="$steps" \
        --log_freq=1 \
        --output_dir="$output_root" \
        --job_name="$run_tag" \
        --policy.device=cuda \
        --wandb.enable=false \
        --wandb.disable_artifact=true \
        --save_freq="$save_freq" \
        --eval_freq="$eval_freq" \
        --num_workers="$num_workers" \
        --policy.optimizer_lr="$optimizer_lr" \
        --policy.scheduler_warmup_steps="$warmup_steps" \
        --policy.scheduler_decay_steps="$steps" \
        --policy.scheduler_decay_lr="$scheduler_decay_lr" \
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
        --policy.worldflow_bootstrap_from_ego="$bootstrap_from_ego" \
        --policy.worldflow_feature_dim=64 \
        --policy.worldflow_grid_size=0.01 \
        --policy.worldflow_loss_weight="$worldflow_loss_weight" \
        --policy.worldflow_geo_loss_weight="$worldflow_geo_loss_weight" \
        --policy.worldflow_bridge_loss_weight="$worldflow_bridge_loss_weight" \
        --policy.worldflow_equiv_loss_weight="$worldflow_equiv_loss_weight" \
        --policy.worldflow_pretrained_lr_multiplier="$pretrained_lr_multiplier" \
        --policy.worldflow_new_lr_multiplier="$new_lr_multiplier" \
        --policy.worldflow_trans_weight=1.0 \
        --policy.worldflow_rot_weight=1.0 \
        --policy.worldflow_max_points=0 \
        --policy.worldflow_require_action_target_sidecar=true \
        --policy.pose9_action_noise_enable=false \
        --policy.worldflow_noise_coupling=conjugate_ego \
        --policy.worldflow_frame_origin=current_ee \
        --policy.worldflow_action_fusion="$action_fusion" \
        --policy.worldflow_augmentation_trans_scale=0.05 \
        --policy.worldflow_augmentation_rot_scale=0.2 \
        --policy.worldflow_se3_head_enable="$world_se3_head_enable" \
        --policy.se3_enable=true \
        --policy.se3_twist_head_mode=pose9_chart_endpoint \
        --policy.se3_noise_trans_scale=0.10 \
        --policy.se3_noise_rot_scale=0.10 \
        --policy.se3_noise_gripper_scale=0.10 \
        --policy.flow_time_sampling=integration_grid \
        --policy.flow_time_zero_probability=0.25 \
        --policy.se3_final_correction_enable=false 2>&1 | tee -a "$log_file"
}

run_eval() {
    preflight
    check_gpu_idle
    local checkpoint=${SONG_V043_EVAL_CKPT:-}
    local tag=${SONG_V043_EVAL_TAG:-}
    local task_workers=${SONG_V043_EVAL_TASK_WORKERS:-3}
    local episode_workers=${SONG_V043_EVAL_EPISODE_WORKERS_PER_TASK:-10}
    local batch_size=${SONG_V043_EVAL_BATCH_SIZE:-30}
    if [[ -z "$checkpoint" || -z "$tag" ]]; then
        echo "Set SONG_V043_EVAL_CKPT and SONG_V043_EVAL_TAG for an immutable evaluation." >&2
        exit 1
    fi
    if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "SONG_V043_EVAL_TAG contains unsafe characters: $tag" >&2
        exit 1
    fi
    if [[ ! "$task_workers" =~ ^[0-9]+$ ]] || [[ ! "$episode_workers" =~ ^[0-9]+$ ]] || \
        (( task_workers < 1 || episode_workers < 1 || task_workers * episode_workers > 30 )); then
        echo "Evaluation task_workers * episode_workers_per_task must be in [1, 30]." >&2
        exit 1
    fi
    if [[ ! "$batch_size" =~ ^[0-9]+$ ]] || (( batch_size < 1 || batch_size > 30 )); then
        echo "SONG_V043_EVAL_BATCH_SIZE must be in [1, 30]." >&2
        exit 1
    fi
    require_dir "$checkpoint"
    require_file "$checkpoint/config.json"
    require_file "$checkpoint/model.safetensors"
    local output_dir="$SONG_V043_EVAL_ROOT/$tag"
    if [[ -e "$output_dir" ]]; then
        echo "Evaluation output already exists; refusing to overwrite: $output_dir" >&2
        exit 1
    fi
    local log_file="$SONG_V043_EXPERIMENT_ROOT/logs/eval_${tag}.log"
    cd "$repo_root"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$SONG_V043_PYTHON" \
        benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
        --config benchmarks/song_real_libero/configs/libero.json \
        --policy.path "$checkpoint" \
        --suite-gpu-ids 0,1,2,3 \
        --suite libero_spatial --suite libero_object --suite libero_goal --suite libero_10 \
        --all-tasks --episodes 50 \
        --policy-noise-seed 0 --env-seed 7 --strict-official-init \
        --gripper-control-mode delta_width_initial_sync \
        --gripper-delta-threshold 0.002 \
        --gripper-delta-alignment current_minus_previous \
        --waypoint-max-hold-steps 1 \
        --isolated-policy-workers 1 --task-worker-backend process \
        --task-workers "$task_workers" --episode-workers-per-task "$episode_workers" \
        --inference-batch-size "$batch_size" \
        --control-freq 20 --action-index 0 \
        --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
        --render-mode offscreen --no-visualize-foreground --save-video \
        --output-dir "$output_dir" 2>&1 | tee -a "$log_file"
}

pilot_prepare_dirs() {
    mkdir -p \
        "$SONG_V043_PILOT_ROOT/artifacts" \
        "$SONG_V043_PILOT_ROOT/cache" \
        "$SONG_V043_PILOT_ROOT/eval" \
        "$SONG_V043_PILOT_ROOT/logs" \
        "$SONG_V043_PILOT_ROOT/tmux" \
        "$SONG_V043_PILOT_ROOT/training"
}

pilot_preflight() {
    preflight
    pilot_prepare_dirs
    require_dir "$SONG_V043_PILOT_DATASET_ROOT"
    require_file "$SONG_V043_PILOT_DATASET_ROOT/meta/info.json"
    "$SONG_V043_PYTHON" - "$SONG_V043_PILOT_DATASET_ROOT" <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import sys

root = sys.argv[1]
meta = LeRobotDatasetMetadata(root, root=root)
selected = [meta.episodes[index] for index in range(400, 450)]
assert len(selected) == 50
assert {record["tasks"][0] for record in selected} == {"put both moka pots on the stove"}
assert sum(int(record["length"]) for record in selected) == 20744
print("pilot preflight: task=libero_10/8 episodes=400:450 samples=20744")
PY
}

run_pilot_cache() {
    pilot_preflight
    check_gpu_idle
    if [[ -n "$(find "$SONG_V043_PILOT_CACHE_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "Pilot cache directory is not empty: $SONG_V043_PILOT_CACHE_ROOT" >&2
        exit 1
    fi
    mkdir -p "$SONG_V043_PILOT_CACHE_ROOT"
    local log_file="$SONG_V043_PILOT_ROOT/logs/cache_task08_50ep_fps_union.log"
    cd "$repo_root"
    SONG_POINTCLOUD_GRIPPER_POINTS=500 \
        "$SONG_V043_PYTHON" "$SONG_V043_TORCHRUN" --standalone --nproc_per_node=4 \
        benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
        --dataset.repo_id="$SONG_V043_PILOT_DATASET_ROOT" \
        --episodes="$SONG_V043_PILOT_EPISODE_RANGE" \
        --camera-views=agentview,robot0_eye_in_hand \
        --camera-view-fusion=fps \
        --output-dir="$SONG_V043_PILOT_CACHE_ROOT" \
        --batch-size=24 \
        --num-workers=4 \
        --shard-size=2048 \
        --storage-dtype=float16 \
        --nn-chunk-size=1024 \
        --vis-count=4 2>&1 | tee -a "$log_file"
}

run_pilot_cache_smoke_distributed() {
    pilot_preflight
    check_gpu_idle
    if [[ -n "$(find "$SONG_V043_PILOT_CACHE_SMOKE_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "Distributed pilot cache smoke directory is not empty: $SONG_V043_PILOT_CACHE_SMOKE_ROOT" >&2
        exit 1
    fi
    mkdir -p "$SONG_V043_PILOT_CACHE_SMOKE_ROOT"
    local log_file="$SONG_V043_PILOT_ROOT/logs/cache_smoke_fps_union_4gpu_b24_96.log"
    cd "$repo_root"
    SONG_POINTCLOUD_GRIPPER_POINTS=500 \
        "$SONG_V043_PYTHON" "$SONG_V043_TORCHRUN" --standalone --nproc_per_node=4 \
        benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
        --dataset.repo_id="$SONG_V043_PILOT_DATASET_ROOT" \
        --episodes="$SONG_V043_PILOT_EPISODE_RANGE" \
        --camera-views=agentview,robot0_eye_in_hand \
        --camera-view-fusion=fps \
        --output-dir="$SONG_V043_PILOT_CACHE_SMOKE_ROOT" \
        --batch-size=24 \
        --num-workers=4 \
        --shard-size=24 \
        --max-samples=96 \
        --storage-dtype=float16 \
        --nn-chunk-size=1024 \
        --vis-count=0 2>&1 | tee -a "$log_file"

    "$SONG_V043_PYTHON" - "$SONG_V043_PILOT_CACHE_SMOKE_ROOT/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
assert manifest["num_samples"] == 96
assert manifest["distributed"]["world_size"] == 4
assert manifest["camera_view_fusion"] == "fps"
assert manifest["point_count_policy"] == "fps_scene_union_preserve_primary_gripper"
assert manifest["fps_contract"] == {
    "backend": "pointops_cuda",
    "target_points": 10000,
    "target_scene_points": 9500,
    "preserved_gripper_points": 500,
}
assert len(manifest["shards"]) == 4
print("distributed FPS cache smoke: PASS samples=96 world_size=4 batch_per_gpu=24")
PY
}

validate_pilot_cache() {
    require_file "$SONG_V043_PILOT_CACHE_ROOT/manifest.json"
    PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$SONG_V043_PYTHON" - "$SONG_V043_PILOT_CACHE_ROOT" <<'PY'
from pathlib import Path
import json
import sys

from lerobot.policies.smolvla.song_pointseg import SongPointSegCachedDataset

root = Path(sys.argv[1])
with open(root / "manifest.json", encoding="utf-8") as stream:
    manifest = json.load(stream)
assert manifest["num_samples"] == 20744
assert manifest["camera_views"] == ["agentview", "robot0_eye_in_hand"]
assert manifest["camera_view_weights"] is None
assert manifest["camera_view_fusion"] == "fps"
assert manifest["gripper_points"] == 500
assert manifest["trajectory_offset_filtering"] == "relative_frame_offsets"
assert manifest["args"]["episodes"] == list(range(400, 450))
cache = SongPointSegCachedDataset(root)
assert len(cache) == 20744
for index in (0, len(cache) // 2, len(cache) - 1):
    item = cache[index]
    assert int(item["dataset_index"]) == index
    assert item["pointseg.labels"].numel() == 10000
print(f"pilot cache: PASS samples={len(cache)} shards={len(manifest['shards'])}")
PY
}

run_pilot_train() {
    pilot_preflight
    validate_pilot_cache
    check_gpu_idle
    local output_root=${SONG_V043_PILOT_TRAIN_OUTPUT_ROOT:-$SONG_V043_PILOT_TRAIN_ROOT}
    local steps=${SONG_V043_PILOT_TRAIN_STEPS:-100}
    local save_freq=${SONG_V043_PILOT_TRAIN_SAVE_FREQ:-25}
    local eval_freq=${SONG_V043_PILOT_TRAIN_EVAL_FREQ:-25}
    local num_workers=${SONG_V043_PILOT_TRAIN_NUM_WORKERS:-12}
    local batch_size_per_gpu=${SONG_V043_PILOT_TRAIN_BATCH_SIZE_PER_GPU:-48}
    local warmup_steps=${SONG_V043_PILOT_TRAIN_WARMUP_STEPS:-10}
    local freeze_pointseg_bn_stats=${SONG_V043_PILOT_TRAIN_FREEZE_POINTSEG_BN_STATS:-true}
    local run_tag=${SONG_V043_PILOT_TRAIN_TAG:-task08_50ep_fps_union_4gpu_b48_100steps}
    if [[ ! "$run_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsafe pilot training tag: $run_tag" >&2
        exit 1
    fi
    if [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "Pilot training directory is not empty: $output_root" >&2
        exit 1
    fi
    mkdir -p "$output_root"
    local episode_list
    episode_list="[$(seq -s, 400 449)]"
    local log_file="$SONG_V043_PILOT_ROOT/logs/train_${run_tag}.log"
    cd "$repo_root"
    ulimit -n 65535
    export SONG_POINTSEG_REQUIRE_POINTOPS=1
    run_accelerate_4gpu benchmarks/song_real_libero/scripts/train_song_benchmark.py \
        --policy.path="$SONG_V043_BASELINE_CKPT" \
        --policy.push_to_hub=false \
        --dataset.repo_id="$SONG_V043_PILOT_DATASET_ROOT" \
        --dataset.episodes="$episode_list" \
        --pointseg_sample_cache_dir="$SONG_V043_PILOT_CACHE_ROOT" \
        --policy.camera_views=agentview,robot0_eye_in_hand \
        --policy.camera_view_fusion=fps \
        --policy.rgb_camera_views=agentview \
        --policy.vla_adapter_enable=true \
        --policy.vla_adapter_freeze_vlm=true \
        --policy.vlm_model_name="$SONG_V043_VLM_MODEL" \
        --policy.vlm_weights_path="$SONG_V043_VLM_WEIGHTS" \
        --policy.load_vlm_weights=true \
        --batch_size="$batch_size_per_gpu" \
        --steps="$steps" \
        --log_freq=1 \
        --output_dir="$output_root" \
        --job_name="wep_vla_v043_${run_tag}" \
        --policy.device=cuda \
        --wandb.enable=false \
        --save_freq="$save_freq" \
        --eval_freq="$eval_freq" \
        --num_workers="$num_workers" \
        --policy.optimizer_lr=0.000025 \
        --policy.scheduler_warmup_steps="$warmup_steps" \
        --policy.scheduler_decay_steps="$steps" \
        --policy.scheduler_decay_lr=0.0000025 \
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
        --policy.pointseg_freeze_batchnorm_stats="$freeze_pointseg_bn_stats" \
        --policy.worldflow_enable=false \
        --policy.worldflow_se3_head_enable=false \
        --policy.se3_enable=false \
        --policy.se3_final_correction_enable=false 2>&1 | tee -a "$log_file"
}

run_pilot_train_worldflow() {
    pilot_preflight
    validate_pilot_cache
    check_gpu_idle
    local stage1_ckpt=${SONG_V043_PILOT_STAGE1_CKPT:-}
    local output_root=${SONG_V043_PILOT_WORLD_TRAIN_OUTPUT_ROOT:-$SONG_V043_PILOT_STAGE2_TRAIN_ROOT}
    local steps=${SONG_V043_PILOT_WORLD_TRAIN_STEPS:-100}
    local save_freq=${SONG_V043_PILOT_WORLD_TRAIN_SAVE_FREQ:-25}
    local eval_freq=${SONG_V043_PILOT_WORLD_TRAIN_EVAL_FREQ:-25}
    local num_workers=${SONG_V043_PILOT_WORLD_TRAIN_NUM_WORKERS:-12}
    local batch_size_per_gpu=${SONG_V043_PILOT_WORLD_TRAIN_BATCH_SIZE_PER_GPU:-24}
    local gradient_accumulation_steps=${SONG_V043_PILOT_WORLD_TRAIN_GRAD_ACCUM_STEPS:-2}
    local warmup_steps=${SONG_V043_PILOT_WORLD_TRAIN_WARMUP_STEPS:-10}
    local optimizer_lr=${SONG_V043_PILOT_WORLD_TRAIN_OPTIMIZER_LR:-0.000025}
    local scheduler_decay_lr=${SONG_V043_PILOT_WORLD_TRAIN_DECAY_LR:-0.0000025}
    local pretrained_lr_multiplier=${SONG_V043_PILOT_WORLD_PRETRAINED_LR_MULTIPLIER:-0.2}
    local new_lr_multiplier=${SONG_V043_PILOT_WORLD_NEW_LR_MULTIPLIER:-1.0}
    local worldflow_loss_weight=${SONG_V043_PILOT_WORLD_FLOW_LOSS_WEIGHT:-0.0005}
    local worldflow_geo_loss_weight=${SONG_V043_PILOT_WORLD_GEO_LOSS_WEIGHT:-0.0001}
    local worldflow_bridge_loss_weight=${SONG_V043_PILOT_WORLD_BRIDGE_LOSS_WEIGHT:-0.0002}
    local worldflow_equiv_loss_weight=${SONG_V043_PILOT_WORLD_EQUIV_LOSS_WEIGHT:-0.0001}
    local bootstrap_from_ego=${SONG_V043_PILOT_WORLD_BOOTSTRAP_FROM_EGO:-false}
    local world_se3_head_enable=${SONG_V043_PILOT_WORLD_SE3_HEAD_ENABLE:-false}
    local action_fusion=${SONG_V043_PILOT_WORLD_ACTION_FUSION:-conjugate_residual}
    local run_tag=${SONG_V043_PILOT_WORLD_TRAIN_TAG:-task08_50ep_worldego_joint_se3_chart_conjugate_4gpu_b24_accum2_100steps}
    if [[ -z "$stage1_ckpt" ]]; then
        echo "Set SONG_V043_PILOT_STAGE1_CKPT to the immutable passed Stage 1 checkpoint." >&2
        exit 1
    fi
    if [[ ! "$run_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsafe pilot WorldFlow training tag: $run_tag" >&2
        exit 1
    fi
    require_dir "$stage1_ckpt"
    require_file "$stage1_ckpt/config.json"
    require_file "$stage1_ckpt/model.safetensors"
    if [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "Pilot WorldFlow training directory is not empty: $output_root" >&2
        exit 1
    fi
    mkdir -p "$output_root"
    local episode_list
    episode_list="[$(seq -s, 400 449)]"
    local log_file="$SONG_V043_PILOT_ROOT/logs/train_${run_tag}.log"
    cd "$repo_root"
    ulimit -n 65535
    export SONG_POINTSEG_REQUIRE_POINTOPS=1
    run_accelerate_4gpu benchmarks/song_real_libero/scripts/train_song_benchmark.py \
        --policy.path="$stage1_ckpt" \
        --policy.push_to_hub=false \
        --dataset.repo_id="$SONG_V043_PILOT_DATASET_ROOT" \
        --dataset.episodes="$episode_list" \
        --pointseg_sample_cache_dir="$SONG_V043_PILOT_CACHE_ROOT" \
        --policy.camera_views=agentview,robot0_eye_in_hand \
        --policy.camera_view_fusion=fps \
        --policy.rgb_camera_views=agentview \
        --policy.vla_adapter_enable=true \
        --policy.vla_adapter_freeze_vlm=true \
        --policy.vlm_model_name="$SONG_V043_VLM_MODEL" \
        --policy.vlm_weights_path="$SONG_V043_VLM_WEIGHTS" \
        --policy.load_vlm_weights=true \
        --batch_size="$batch_size_per_gpu" \
        --gradient_accumulation_steps="$gradient_accumulation_steps" \
        --steps="$steps" \
        --log_freq=1 \
        --output_dir="$output_root" \
        --job_name="wep_vla_v043_${run_tag}" \
        --policy.device=cuda \
        --wandb.enable=false \
        --save_freq="$save_freq" \
        --eval_freq="$eval_freq" \
        --num_workers="$num_workers" \
        --policy.optimizer_lr="$optimizer_lr" \
        --policy.scheduler_warmup_steps="$warmup_steps" \
        --policy.scheduler_decay_steps="$steps" \
        --policy.scheduler_decay_lr="$scheduler_decay_lr" \
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
        --policy.worldflow_bootstrap_from_ego="$bootstrap_from_ego" \
        --policy.worldflow_feature_dim=64 \
        --policy.worldflow_grid_size=0.01 \
        --policy.worldflow_loss_weight="$worldflow_loss_weight" \
        --policy.worldflow_geo_loss_weight="$worldflow_geo_loss_weight" \
        --policy.worldflow_bridge_loss_weight="$worldflow_bridge_loss_weight" \
        --policy.worldflow_equiv_loss_weight="$worldflow_equiv_loss_weight" \
        --policy.worldflow_pretrained_lr_multiplier="$pretrained_lr_multiplier" \
        --policy.worldflow_new_lr_multiplier="$new_lr_multiplier" \
        --policy.worldflow_trans_weight=1.0 \
        --policy.worldflow_rot_weight=1.0 \
        --policy.worldflow_max_points=0 \
        --policy.worldflow_require_action_target_sidecar=true \
        --policy.pose9_action_noise_enable=false \
        --policy.worldflow_noise_coupling=conjugate_ego \
        --policy.worldflow_frame_origin=current_ee \
        --policy.worldflow_action_fusion="$action_fusion" \
        --policy.worldflow_augmentation_trans_scale=0.05 \
        --policy.worldflow_augmentation_rot_scale=0.2 \
        --policy.worldflow_se3_head_enable="$world_se3_head_enable" \
        --policy.se3_enable=true \
        --policy.se3_twist_head_mode=pose9_chart_endpoint \
        --policy.se3_noise_trans_scale=0.10 \
        --policy.se3_noise_rot_scale=0.10 \
        --policy.se3_noise_gripper_scale=0.10 \
        --policy.flow_time_sampling=integration_grid \
        --policy.flow_time_zero_probability=0.25 \
        --policy.se3_final_correction_enable=false 2>&1 | tee -a "$log_file"
}

run_pilot_eval() {
    if [[ "${SONG_V043_PILOT_EVAL_SKIP_PREFLIGHT:-0}" != "1" ]]; then
        pilot_preflight
    fi
    check_gpu_idle
    local checkpoint=${SONG_V043_PILOT_EVAL_CKPT:-}
    local tag=${SONG_V043_PILOT_EVAL_TAG:-}
    local episode_count=${SONG_V043_PILOT_EVAL_EPISODES:-20}
    local gpu=${SONG_V043_PILOT_EVAL_GPU:-0}
    local episode_ids_csv=${SONG_V043_PILOT_EVAL_EPISODE_IDS:-}
    local parallel_workers=${SONG_V043_PILOT_EVAL_PARALLEL_WORKERS:-30}
    local inference_batch_size=${SONG_V043_PILOT_EVAL_INFERENCE_BATCH_SIZE:-$parallel_workers}
    local world_to_ego_causal_ablation=${SONG_V043_PILOT_EVAL_WORLD_TO_EGO_CAUSAL_ABLATION:-0}
    local secondary_view_causal_ablation=${SONG_V043_PILOT_EVAL_SECONDARY_VIEW_CAUSAL_ABLATION:-0}
    if [[ -z "$checkpoint" || -z "$tag" ]]; then
        echo "Set SONG_V043_PILOT_EVAL_CKPT and SONG_V043_PILOT_EVAL_TAG." >&2
        exit 1
    fi
    if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsafe pilot eval tag: $tag" >&2
        exit 1
    fi
    if [[ ! "$parallel_workers" =~ ^[0-9]+$ ]] || \
        (( parallel_workers < 1 || parallel_workers > 30 )); then
        echo "SONG_V043_PILOT_EVAL_PARALLEL_WORKERS must be in [1, 30]." >&2
        exit 1
    fi
    if [[ ! "$inference_batch_size" =~ ^[0-9]+$ ]] || \
        (( inference_batch_size < 1 || inference_batch_size > 30 )); then
        echo "SONG_V043_PILOT_EVAL_INFERENCE_BATCH_SIZE must be in [1, 30]." >&2
        exit 1
    fi
    require_dir "$checkpoint"
    require_file "$checkpoint/config.json"
    require_file "$checkpoint/model.safetensors"
    local -a episode_id_args=()
    local -a causal_ablation_args=()
    if [[ "$secondary_view_causal_ablation" == "1" ]]; then
        causal_ablation_args+=(--secondary-view-causal-ablation)
    elif [[ "$secondary_view_causal_ablation" != "0" ]]; then
        echo "SONG_V043_PILOT_EVAL_SECONDARY_VIEW_CAUSAL_ABLATION must be 0 or 1." >&2
        exit 1
    fi
    if [[ "$world_to_ego_causal_ablation" == "1" ]]; then
        causal_ablation_args+=(--world-to-ego-causal-ablation)
    elif [[ "$world_to_ego_causal_ablation" != "0" ]]; then
        echo "SONG_V043_PILOT_EVAL_WORLD_TO_EGO_CAUSAL_ABLATION must be 0 or 1." >&2
        exit 1
    fi
    if [[ -n "$episode_ids_csv" ]]; then
        local -A seen_episode_ids=()
        local -a requested_episode_ids=()
        local episode_id
        IFS=',' read -r -a requested_episode_ids <<< "$episode_ids_csv"
        for episode_id in "${requested_episode_ids[@]}"; do
            if [[ ! "$episode_id" =~ ^[0-9]+$ ]] || (( episode_id >= 50 )); then
                echo "Invalid pilot episode id: $episode_id" >&2
                exit 1
            fi
            if [[ -n "${seen_episode_ids[$episode_id]:-}" ]]; then
                echo "Duplicate pilot episode id: $episode_id" >&2
                exit 1
            fi
            seen_episode_ids[$episode_id]=1
            episode_id_args+=(--episode-id "$episode_id")
        done
    fi
    local output_dir="$SONG_V043_PILOT_EVAL_ROOT/$tag"
    if [[ -e "$output_dir" ]]; then
        echo "Pilot evaluation output already exists: $output_dir" >&2
        exit 1
    fi
    local log_file="$SONG_V043_PILOT_ROOT/logs/eval_${tag}.log"
    cd "$repo_root"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$gpu" \
        MUJOCO_EGL_DEVICE_ID="$gpu" "$SONG_V043_PYTHON" \
        benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
        --config benchmarks/song_real_libero/configs/libero.json \
        --policy.path "$checkpoint" \
        --suite libero_10 --no-all-tasks --task-id 8 \
        --device cuda --render-gpu-device-id "$gpu" \
        --episodes "$episode_count" \
        "${episode_id_args[@]}" \
        "${causal_ablation_args[@]}" \
        --policy-noise-seed 0 --env-seed 7 --strict-official-init \
        --gripper-control-mode delta_width_initial_sync \
        --gripper-delta-threshold 0.002 \
        --gripper-delta-alignment current_minus_previous \
        --waypoint-max-hold-steps 1 \
        --isolated-policy-workers 1 --task-worker-backend process \
        --task-workers 1 --episode-workers-per-task "$parallel_workers" \
        --inference-batch-size "$inference_batch_size" \
        --control-freq 20 --action-index 0 \
        --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
        --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
        --render-mode offscreen --no-visualize-foreground --no-save-video \
        --output-dir "$output_dir" 2>&1 | tee -a "$log_file"
}

usage() {
    echo "Usage: $0 {preflight|cache|validate-cache|train|train-worldflow|eval|pilot-preflight|pilot-cache-smoke-distributed|pilot-cache|pilot-validate-cache|pilot-train|pilot-train-worldflow|pilot-eval}" >&2
}

case "${1:-}" in
    preflight) preflight ;;
    cache) run_cache ;;
    validate-cache) validate_cache ;;
    train) run_train ;;
    train-worldflow) run_train_worldflow ;;
    eval) run_eval ;;
    pilot-preflight) pilot_preflight ;;
    pilot-cache-smoke-distributed) run_pilot_cache_smoke_distributed ;;
    pilot-cache) run_pilot_cache ;;
    pilot-validate-cache) validate_pilot_cache ;;
    pilot-train) run_pilot_train ;;
    pilot-train-worldflow) run_pilot_train_worldflow ;;
    pilot-eval) run_pilot_eval ;;
    *) usage; exit 2 ;;
esac
