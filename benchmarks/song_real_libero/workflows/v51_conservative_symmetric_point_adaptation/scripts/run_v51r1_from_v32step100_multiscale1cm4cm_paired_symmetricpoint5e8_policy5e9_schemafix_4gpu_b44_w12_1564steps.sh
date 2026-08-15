#!/usr/bin/env bash
set -euo pipefail

repo=${LEROBOT_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot_v51_conservative_symmetric_point_adaptation}
root=${EXPERIMENT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811}
dataset=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/libero_4suite_lerobot_dataset
run_root=$root/joint_multiview_worldflow/libero10_500ep
dual_cache=$run_root/pointseg_cache_multiscale_novelty_union_1cm4cm
primary_cache=$root/pointseg_cache_agentview_primary_view_dropout
source_policy=$root/singleview_worldflow/libero10_500ep/training/v32_from_v14_worldbnfixed_w2e_p75_anchor_residual_rate_coordframe_bodyframe_ego_tangent_common005_world02_residual4_4gpu_b32_w12_1080steps/checkpoints/000100/pretrained_model
source_gate=$root/singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v32_residual_rate_coordframe_bodyframe_ego_tangent_p75_local4_fixedbarrier_full500_matched_worldflow_gate_step000100_baseline472_recheck.json
cache_audit=$run_root/artifacts/v51_cache_online_exact_index_audit_36shards.json
optimizer_preflight=$run_root/artifacts/v51r1_schemafix_real_checkpoint_optimizer_and_gradient_role_preflight.json
optimizer_preflight_script=$root/scripts/audit_v49_real_checkpoint_optimizer_preflight.py
output=$run_root/training/v51r1_from_v32step100_multiscale_novelty1cm4cm_paired_symmetricpoint5e8_policy5e9_devicebound_schemafix_4gpu_b44_w12_1564steps
log=$run_root/logs/train_v51r1_from_v32step100_multiscale_novelty1cm4cm_paired_symmetricpoint5e8_policy5e9_devicebound_schemafix_4gpu_b44_w12_1564steps.log
wandb_dir=$run_root/wandb
vlm_model=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct
vlm_weights=/opt/data/private/liusong/hf_models/smolvla_base
python=${PYTHON_BIN:-/home/liusong/anaconda3/envs/reap/bin/python3.10}
expected_head=9db8504a2c575ad386f1d90efa98784c0ea8d701

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain)"
test -s "$dataset/meta/info.json"
test -s "$dual_cache/manifest.json"
test -s "$primary_cache/manifest.json"
test -s "$source_policy/model.safetensors"
test -s "$source_gate"
test -s "$cache_audit"
test -s "$optimizer_preflight_script"
test ! -e "$output"
test "$(sha256sum "$source_policy/model.safetensors" | awk '{print $1}')" = c258303b70d4cab64f89d93c825905813f6f49fbb08cf282b5d4321e1fdf1fb4
mkdir -p "$(dirname "$log")" "$wandb_dir"

env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python" - \
  "$dataset/meta/info.json" "$dual_cache/manifest.json" "$primary_cache/manifest.json" \
  "$source_policy/config.json" "$source_gate" "$cache_audit" "$repo" <<'PY'
import json,sys
from pathlib import Path

from lerobot.policies.smolvla import configuration_smolvla as smolvla_config_module

info=json.load(open(sys.argv[1],encoding="utf-8"))
dual=json.load(open(sys.argv[2],encoding="utf-8"))
primary=json.load(open(sys.argv[3],encoding="utf-8"))
config=json.load(open(sys.argv[4],encoding="utf-8"))
gate=json.load(open(sys.argv[5],encoding="utf-8"))
cache_audit=json.load(open(sys.argv[6],encoding="utf-8"))
expected_src=(Path(sys.argv[7]).resolve()/"src")
imported_config=Path(smolvla_config_module.__file__).resolve()
assert imported_config.is_relative_to(expected_src), (
    f"V51 imported the wrong lerobot tree: {imported_config} is not under {expected_src}"
)
fields=smolvla_config_module.SmolVLAConfig.__dataclass_fields__
assert "multiview_input_symmetric_point_path_adaptation" in fields
assert "camera_view_coarse_novelty_scale" in fields
assert info["total_episodes"]==500 and info["total_frames"]==137_590
assert dual["version"]==12 and dual["num_samples"]==137_590
assert primary["version"]==11 and primary["num_samples"]==137_590
assert dual["camera_views"]==["agentview","robot0_eye_in_hand"]
assert dual["camera_view_fusion"]=="multiscale_novelty_union"
assert dual["camera_view_voxel_size"]==0.01
assert dual["camera_view_coarse_novelty_scale"]==4.0
assert dual["current_points"]==dual["future_points"]==10_000
assert dual["gripper_points"]==500
assert dual["point_count_policy"]=="fine_primary_voxel_cover_plus_coarse_persistent_secondary_novel_voxels_preserve_primary_gripper"
assert primary["camera_views"]==["agentview"]
assert primary["camera_view_fusion"]=="legacy_budget"
assert primary["current_points"]==primary["future_points"]==10_000
assert config["worldflow_enable"] is True
assert config["worldflow_action_fusion"]=="endpoint_residual_boosting"
assert config["worldflow_noise_coupling"]=="projected_ego_path"
assert config["worldflow_ego_residual_gate_init"] is None
assert config["worldflow_freeze_batchnorm_stats"] is True
assert config["pointseg_freeze_batchnorm_stats"] is True
assert config["se3_enable"] is False
assert all(config[k]==0 for k in ("worldflow_loss_weight","worldflow_geo_loss_weight","worldflow_bridge_loss_weight","worldflow_equiv_loss_weight"))
assert gate["passes_joint_gate"] is True
assert gate["normal_successes"]==475 and gate["matched_baseline_successes"]==472
assert gate["world_to_ego_causal_ablation_successes"]==469
assert cache_audit["manifest_version"]==12
assert cache_audit["coarse_novelty_scale"]==4.0
assert cache_audit["manifest_num_samples"]==137_590
assert cache_audit["manifest_shard_count"]==cache_audit["audited_samples"]==36
assert cache_audit["all_exact_online_cache_index_match"] is True
assert cache_audit["all_exact_10000"] is True
assert cache_audit["all_unique_indices"] is True
assert cache_audit["all_gripper_exact"] is True
paired=2*info["total_frames"]
global_batch=4*44
assert 1563*global_batch < paired <= 1564*global_batch
assert 1564*global_batch >= paired
print("V51 preflight PASS: full paired epoch, exact10k fine1cm/coarse4cm input, symmetric Ego/World point paths 5e-8 and policy paths 5e-9")
PY

CUDA_VISIBLE_DEVICES='' env PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python" \
  "$optimizer_preflight_script" \
  --source "$source_policy" \
  --output "$optimizer_preflight" \
  --expected-repo-head "$expected_head"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
export OMP_NUM_THREADS=1
export WANDB_DIR="$wandb_dir"
export WANDB_RESUME=allow
export WANDB_INIT_TIMEOUT=300
if [[ -z ${WANDB_MODE:-} ]]; then
  if timeout 20 "$python" -m wandb login --verify >>"$run_root/logs/wandb_v51_connectivity_preflight.log" 2>&1; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi

cd "$repo"
ulimit -n 65535
exec "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=no \
  --dynamo_backend=no --main_process_port=0 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path="$source_policy" --policy.push_to_hub=false \
  --dataset.repo_id="$dataset" \
  --pointseg_sample_cache_dir="$dual_cache" \
  --pointseg_primary_sample_cache_dir="$primary_cache" \
  --policy.camera_views=agentview,robot0_eye_in_hand \
  --policy.rgb_camera_views=agentview \
  --policy.camera_view_fusion=multiscale_novelty_union \
  --policy.camera_view_voxel_size=0.01 \
  --policy.camera_view_coarse_novelty_scale=4.0 \
  --policy.camera_view_fps_target_points=10000 \
  --policy.camera_view_fps_gripper_points=500 \
  --policy.multiview_input_view_dropout_enable=true \
  --policy.multiview_input_view_dropout_paired_coverage=true \
  --policy.multiview_input_pretrained_lr_multiplier=0.005 \
  --policy.multiview_input_point_lr_multiplier=0.05 \
  --policy.multiview_input_symmetric_point_path_adaptation=true \
  --policy.vla_adapter_enable=true --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name="$vlm_model" --policy.vlm_weights_path="$vlm_weights" \
  --policy.load_vlm_weights=true --task_balanced_sampling=true \
  --batch_size=44 --gradient_accumulation_steps=1 --num_workers=12 \
  --steps=1564 --log_freq=1 --output_dir="$output" \
  --job_name=wep_vla_v043_v51r1_v32step100_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_devicebound_schemafix_b44_1564 \
  --policy.device=cuda --eval_freq=0 \
  --save_freq=1564 --save_steps='[260,520,780,1040,1300]' \
  --wandb.enable=true --wandb.project=wep_vla_v043_libero10 \
  --wandb.run_id=joint-v51r1-v32step100-multiscale1cm4cm-paired-symmetricpoint5e8-policy5e9-devicebound-schemafix-b44-20260815 \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=0.000001 --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=1564 --policy.scheduler_decay_lr=0.0000001 \
  --policy.pointseg_enable=true --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.0005 \
  --policy.pointseg_foreground_ratio=0.025 --policy.pointseg_background_ratio=0.025 \
  --policy.pointseg_min_foreground_points=2500 --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false --policy.pointseg_freeze_batchnorm_stats=true \
  --policy.point_action_fusion_enable=true \
  --policy.worldflow_enable=true --policy.worldflow_bootstrap_from_ego=false \
  --policy.worldflow_feature_dim=64 --policy.worldflow_grid_size=0.01 \
  --policy.worldflow_freeze_batchnorm_stats=true \
  --policy.worldflow_loss_weight=0.0 --policy.worldflow_geo_loss_weight=0.0 \
  --policy.worldflow_bridge_loss_weight=0.0 --policy.worldflow_equiv_loss_weight=0.0 \
  --policy.worldflow_pretrained_lr_multiplier=0.005 \
  --policy.worldflow_new_lr_multiplier=0.005 \
  --policy.worldflow_residual_lr_multiplier=0.005 \
  --policy.worldflow_training_world_to_ego_dropout_probability=0.75 \
  --policy.worldflow_training_residual_anchor_stop_gradient=true \
  --policy.worldflow_training_ego_priority_gradient_projection=false \
  --policy.worldflow_training_shared_gradient_ego_tangent_projection=true \
  --policy.worldflow_endpoint_residual_rate_parameterization=true \
  --policy.worldflow_endpoint_residual_ego_frame_parameterization=false \
  --policy.worldflow_endpoint_residual_body_frame_parameterization=true \
  --policy.worldflow_training_coordinate_frame_augmentation=true \
  --policy.worldflow_augmentation_trans_scale=0.05 \
  --policy.worldflow_augmentation_rot_scale=0.2 \
  --policy.worldflow_trans_weight=1.0 --policy.worldflow_rot_weight=1.0 \
  --policy.worldflow_max_points=0 --policy.worldflow_require_action_target_sidecar=true \
  --policy.pose9_action_noise_enable=false \
  --policy.worldflow_noise_coupling=projected_ego_path \
  --policy.worldflow_frame_origin=current_ee \
  --policy.worldflow_action_fusion=endpoint_residual_boosting \
  --policy.worldflow_se3_head_enable=false --policy.se3_enable=false \
  --policy.se3_twist_head_mode=pose9_chart_endpoint \
  --policy.se3_noise_trans_scale=0.10 --policy.se3_noise_rot_scale=0.10 \
  --policy.se3_noise_gripper_scale=0.10 \
  --policy.flow_time_sampling=integration_grid --policy.flow_time_zero_probability=0.25 \
  --policy.se3_final_correction_enable=false 2>&1 | tee "$log"

