from __future__ import annotations

import copy
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "benchmarks" / "song_real_libero" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import full_molmo2er_worldflow_tools as tools  # noqa: E402


TRAIN = SCRIPT_DIR / "train_full_molmo2er_worldflow_8gpu.sh"
PIPELINE = SCRIPT_DIR / "run_full_molmo2er_worldflow_three_stage_8gpu.sh"
EVAL = SCRIPT_DIR / "eval_full_molmo2er_worldflow_8gpu.sh"
HELPER = SCRIPT_DIR / "full_molmo2er_worldflow_tools.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _memory_audit(batch_profile: str = "b24") -> dict[str, object]:
    contract = tools.global_batch_contract(batch_profile)
    ranks = []
    mapping = []
    for rank in range(8):
        allocated = 1_000 + rank
        reserved = 2_000 + rank
        peak_allocated = 3_000 + rank
        peak_reserved = 4_000 + rank
        record = {
            "version": 1,
            "mode": "exact_global_batch_post_first_optimizer_step_cuda_memory",
            "completed_optimizer_step": 1,
            "global_rank": rank,
            "local_rank": rank,
            "world_size": 8,
            "hostname": "test-a800",
            "accelerator_device": f"cuda:{rank}",
            "logical_cuda_index": rank,
            "physical_cuda_device": str(rank),
            "cuda_visible_devices": [str(item) for item in range(8)],
            "global_batch_size": contract["global_batch_size"],
            "physical_forward_samples_per_optimizer_step": contract["physical_forward_samples"],
            "discarded_for_gradient_samples_per_optimizer_step": contract["discarded_for_gradient"],
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "optimizer_state_tensor_count": 12,
            "optimizer_state_total_numel": 345,
        }
        ranks.append(record)
        mapping.append(
            {
                "global_rank": rank,
                "local_rank": rank,
                "logical_cuda_index": rank,
                "physical_cuda_device": str(rank),
            }
        )
    return {
        "version": 1,
        "mode": "exact_global_batch_post_first_optimizer_step_cuda_memory",
        "completed_optimizer_step": 1,
        "world_size": 8,
        "global_batch_size": contract["global_batch_size"],
        "rank_count": 8,
        "max_allocated_bytes": 1_007,
        "max_reserved_bytes": 2_007,
        "max_peak_allocated_bytes": 3_007,
        "max_peak_reserved_bytes": 4_007,
        "logical_to_physical_cuda_mapping": mapping,
        "ranks": ranks,
    }


def _frozen_hash_audit() -> dict[str, object]:
    snapshot = {
        "algorithm": "sha256",
        "hash_scheme": "length-prefixed metadata then bytes",
        "parameter_order": "lexicographic fully-qualified name",
        "chunk_bytes": 16 * 1024 * 1024,
        "parameter_count": 100,
        "total_numel": 1_000,
        "total_bytes": 2_000,
        "sha256": "a" * 64,
    }
    return {
        "version": 1,
        "mode": "full_molmo2er_frozen_live_parameter_before_after_hash",
        "comparison_pass": True,
        "before": snapshot,
        "after": copy.deepcopy(snapshot),
    }


def _exact_batch_audit() -> dict[str, object]:
    return {
        "version": 1,
        "mode": "exact_global_batch",
        **tools.exact_global_batch_runtime_contract("b24"),
    }


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("b4", (4, 6, 6, 0, 192, 0, "32/192")),
        ("b8", (8, 3, 3, 0, 192, 0, "64/192")),
        ("b16", (16, 2, 1, 4, 256, 64, "128/192")),
        ("b24", (24, 1, 1, 0, 192, 0, "192/192")),
    ],
)
def test_exact_global_192_batch_profiles_are_fully_derived(
    profile: str, expected: tuple[object, ...]
) -> None:
    contract = tools.global_batch_contract(profile)
    assert (
        contract["microbatch_per_rank"],
        contract["microsteps_per_optimizer_step"],
        contract["full_microsteps"],
        contract["partial_microstep_active_ranks"],
        contract["physical_forward_samples"],
        contract["discarded_for_gradient"],
        contract["partial_microstep_valid_loss_scale"],
    ) == expected
    runtime = tools.exact_global_batch_runtime_contract(profile)
    assert runtime["global_batch_size"] == 192
    assert runtime["micro_batch_size_per_rank"] == expected[0]
    assert runtime["gradient_accumulation_steps"] == expected[1]
    assert runtime["partial_micro_step_index"] == (expected[2] if expected[3] else None)
    assert runtime["partial_active_ranks"] == expected[3]


def test_unknown_batch_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown batch profile"):
        tools.global_batch_contract("b20")


def test_shell_contract_is_full_vision_worldflow_exact192_and_three_stage() -> None:
    for path in (TRAIN, PIPELINE, EVAL):
        subprocess.run(["bash", "-n", str(path)], check=True)

    train = TRAIN.read_text(encoding="utf-8")
    assert 'readonly PHYSICAL_CUDA_DEVICES="0,1,2,3,4,5,6,7"' in train
    assert "readonly NUM_PROCESSES=8" in train
    assert 'readonly BATCH_PROFILE="${MOLMO_BATCH_PROFILE:-b24}"' in train
    assert "b4)" in train and "b8)" in train and "b16)" in train and "b24)" in train
    assert all(value in train for value in ("BATCH_SIZE=4", "BATCH_SIZE=8", "BATCH_SIZE=16", "BATCH_SIZE=24"))
    assert "ACCUMULATION_STEPS=6" in train and "ACCUMULATION_STEPS=3" in train
    assert "ACCUMULATION_STEPS=2" in train
    assert "ACCUMULATION_STEPS=1" in train
    assert "readonly GLOBAL_BATCH_SIZE=192" in train
    assert '--global_batch_size="${GLOBAL_BATCH_SIZE}"' in train
    assert "--policy.vlm_backend=molmo2_full" in train
    assert "--policy.full_molmo_topology=v3_feature_align_language_casual" in train
    assert "--multi_gpu" in train
    assert "--deepspeed" not in train
    assert "zero_stage" not in train
    assert "--policy.num_vlm_layers=36" in train
    assert "--policy.num_expert_layers=36" in train
    assert "--policy.camera_views=agentview" in train
    assert "--policy.rgb_camera_views=agentview" in train
    assert "--policy.freeze_vision_encoder=true" in train
    assert "--policy.train_expert_only=true" in train
    assert "--policy.molmo_inference_only=false" in train
    assert "--policy.molmo_gradient_checkpointing=true" in train
    assert "--policy.molmo_gradient_checkpointing_layers_per_segment=2" in train
    assert "--policy.worldflow_enable=true" in train
    assert "--policy.worldflow_target_type=world_eef_trajectory" in train
    assert "--policy.worldflow_world_eef_velocity_mode=base_pose9_euclidean" in train
    assert "--policy.worldflow_reference_frame=robot_base" in train
    assert "--policy.worldflow_frame_origin=global" in train
    assert "--policy.worldflow_scene_frame_origin=global" in train
    assert "--policy.worldflow_action_fusion=point_action_expert_conjugate_bridge" in train
    assert "--policy.worldflow_action_expert_mode=shared" in train
    assert "--policy.worldflow_current_ee_pose_token=false" in train
    assert "--policy.worldflow_freeze_pretrained_ego=false" in train
    assert "--policy.worldflow_training_coordinate_frame_augmentation=false" in train
    assert "--policy.worldflow_pretrained_lr_multiplier=1.0" in train
    assert "--policy.worldflow_new_lr_multiplier=1.0" in train
    assert "--policy.worldflow_eef_probe_radius_m=0.10" in train
    assert "--policy.worldflow_bootstrap_from_ego=false" in train
    assert "--policy.worldflow_ego_residual_gate_init=null" in train
    assert "--policy.pointseg_checkpoint_path=null" in train
    assert "--policy.pointseg_freeze_batchnorm_stats=true" in train
    assert "--policy.pose9_action_noise_enable=false" in train
    assert "--policy.scheduler_warmup_steps=100" in train
    assert "--policy.scheduler_decay_steps=30000" in train
    assert "--policy.optimizer_lr=0.0001" in train
    assert 'SAVE_FREQ="${MOLMO_SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}"' in train
    assert '"${RUN_MODE}" == "benchmark"' in train
    assert '"${STEPS}" == "8"' in train
    assert '[[ "${SAVE_CHECKPOINT}" == "false" ]]' in train
    assert '[[ "${WANDB_ENABLE}" == "false" ]]' in train
    assert '--batch-profile "${BATCH_PROFILE}"' in train
    assert 'readonly REQUIRED_NVML_LIBRARY="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.107.02"' in train
    assert 'export LD_PRELOAD="${NVML_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"' in train
    assert 'MODEL_DIR="${MOLMO2_ER_MODEL_DIR:-${EXPERIMENT_ROOT}/Molmo2-ER}"' in train
    assert "=None" not in train

    pipeline = PIPELINE.read_text(encoding="utf-8")
    assert "STAGE_STEPS=(36000 30000 30000)" in pipeline
    assert "STAGE_ENDS=(36000 66000 96000)" in pipeline
    assert "STAGE_FLOORS=(2.5e-6 3e-5 3e-5)" in pipeline
    assert 'SAVE_FREQ="${FULL_MOLMO2ER_SAVE_FREQ:-2000}"' in pipeline
    assert "MOLMO_STEPS=2 \\" in pipeline
    assert "MOLMO_SAVE_CHECKPOINT=false \\" in pipeline
    assert 'readonly BATCH_PROFILE="${FULL_MOLMO2ER_BATCH_PROFILE:-b24}"' in pipeline
    assert 'MOLMO_BATCH_PROFILE="${BATCH_PROFILE}" \\' in pipeline
    assert pipeline.index('"${PYTHON_BIN}" "${TOOLS}" model-audit') < pipeline.index(
        "data-audit"
    )
    smoke_call = pipeline.rindex("\nrun_two_step_smoke\n")
    stage_loop = pipeline.index('\nfor stage_index in "${!STAGE_LABELS[@]}"; do')
    assert smoke_call < stage_loop
    assert 'CHECKPOINT_066000="${RUN_ROOT}/cumulative_checkpoints/066000"' in pipeline
    assert 'CHECKPOINT_096000="${RUN_ROOT}/cumulative_checkpoints/096000"' in pipeline
    assert 'bash "${EVAL_LAUNCHER}"' in pipeline

    evaluator = EVAL.read_text(encoding="utf-8")
    assert "for physical_gpu in 0 1 2 3 4 5 6 7; do" in evaluator
    assert "--episodes 50" in evaluator
    assert "evaluate_checkpoint checkpoint_066000" in evaluator
    assert "evaluate_checkpoint checkpoint_096000" in evaluator
    for shell_text in (train, pipeline, evaluator):
        assert "nvidia-smi" not in shell_text
        assert "jq" not in shell_text


def test_eval_shards_cover_40_tasks_once_across_gpu0_to_gpu7() -> None:
    tools._assert_eval_plan()
    assignments = [
        (shard.suite, task_id)
        for shard in tools.EVAL_SHARDS
        for task_id in shard.task_ids
    ]
    assert len(assignments) == 40
    assert len(set(assignments)) == 40
    assert {shard.physical_gpu for shard in tools.EVAL_SHARDS} == set(range(8))


def test_checkpoint_audit_uses_pretrained_model_parent_as_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "030000" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "config.json", tools.POLICY_CONTRACT)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    _write_json(checkpoint / "policy_preprocessor.json", {})
    _write_json(checkpoint / "policy_postprocessor.json", {})

    state = checkpoint.parent / "training_state"
    state.mkdir()
    for name in (
        "optimizer_param_groups.json",
        "optimizer_state.safetensors",
        "rng_state.safetensors",
        "scheduler_state.json",
    ):
        (state / name).write_bytes(b"x")
    _write_json(state / "training_step.json", {"step": 30_000})

    audit = tools.audit_checkpoint(checkpoint, expected_step=30_000, training_state=True)
    assert audit["checkpoint_step"] == 30_000
    assert audit["training_state"]["step"] == 30_000


def test_smoke_manifest_requires_eight_rank_memory_and_frozen_hash(
    tmp_path: Path,
) -> None:
    smoke_output = tmp_path / "smoke_output"
    smoke_output.mkdir()
    launch = tmp_path / "launch_manifest.json"
    log = tmp_path / "train.log"
    marker = tmp_path / "complete.json"
    data_hash = "data-semantic-hash"

    _write_json(
        launch,
        {
            "policy_contract": tools.POLICY_CONTRACT,
            "global_batch_contract": tools.global_batch_contract("b24"),
            "data_audit": {"semantic_hash": data_hash},
        },
    )
    _write_json(smoke_output / "exact_global_batch_manifest.json", _exact_batch_audit())
    _write_json(
        smoke_output / "full_molmo2er_parameter_audit.json",
        {
            "backend": "molmo2_full",
            "worldflow_enabled": True,
            "vlm_layers": 36,
            "expert_layers": 36,
            "vision_backbone_present": True,
            "molmo_frozen": True,
            "trainable_allowlist_pass": True,
        },
    )
    memory_path = smoke_output / "exact_global_batch_post_adam_cuda_memory_audit.json"
    _write_json(memory_path, _memory_audit())
    _write_json(
        smoke_output / "full_molmo2er_frozen_parameter_hash_audit.json",
        _frozen_hash_audit(),
    )
    log.write_text(
        "step:1 skipped_nonfinite_loss:0 skipped_nonfinite_grad:0\n"
        "step:2 skipped_nonfinite_loss:0 skipped_nonfinite_grad:0\n",
        encoding="utf-8",
    )

    args = Namespace(
        output=str(marker),
        smoke_output=str(smoke_output),
        launch_manifest=str(launch),
        log=str(log),
        expected_data_hash=data_hash,
    )
    assert tools.command_write_smoke_manifest(args) == 0
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["optimizer_steps"] == 2
    assert payload["memory_observed_after_optimizer_step"] == 1
    assert payload["post_adam_cuda_memory_audit"]["rank_count"] == 8
    assert payload["frozen_parameter_hash_audit"]["comparison_pass"] is True
    assert set(payload["artifacts"]) == {
        "launch_manifest",
        "training_log",
        "exact_global_batch_manifest",
        "parameter_audit",
        "post_adam_cuda_memory_audit",
        "frozen_parameter_hash_audit",
    }
    tools._validate_smoke_manifest_payload(payload, expected_data_hash=data_hash)

    memory_path.write_text(memory_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash drifted"):
        tools._validate_smoke_manifest_payload(payload, expected_data_hash=data_hash)


def test_eight_rank_memory_validator_rejects_gpu_mapping_drift() -> None:
    audit = _memory_audit()
    tools._validate_post_adam_memory_audit(audit)
    audit["ranks"][0]["cuda_visible_devices"] = ["0"]
    with pytest.raises(ValueError, match="visibility contract"):
        tools._validate_post_adam_memory_audit(audit)


def _make_data_contract(tmp_path: Path) -> tuple[Path, Path]:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    episodes_dir = dataset / "meta" / "episodes"
    episodes_dir.mkdir(parents=True)
    _write_json(
        dataset / "meta" / "info.json",
        {"total_tasks": 40, "total_episodes": 2, "total_frames": 4},
    )
    table = pa.table(
        {
            "episode_index": [0, 1],
            "length": [2, 2],
            "dataset_from_index": [0, 2],
            "dataset_to_index": [2, 4],
        }
    )
    parquet.write_table(table, episodes_dir / "chunk-000.parquet")

    for directory in (
        dataset / "world_ee_poses",
        dataset / "action_target_ee_poses",
    ):
        directory.mkdir()
        np.save(directory / "episode_000000.npy", np.zeros((2, 9), dtype=np.float32))
        np.save(directory / "episode_000001.npy", np.ones((2, 9), dtype=np.float32))

    shard = cache / "shard_000000"
    shard.mkdir(parents=True)
    np.save(shard / "sample_offsets.npy", np.asarray([0, 2, 4, 6, 8], dtype=np.int64))
    np.save(shard / "point_indices.npy", np.arange(8, dtype=np.int64))
    np.save(shard / "dataset_index.npy", np.arange(4, dtype=np.int64))
    np.save(shard / "episode_index.npy", np.asarray([0, 0, 1, 1], dtype=np.int64))
    np.save(shard / "frame_index.npy", np.asarray([0, 1, 0, 1], dtype=np.int64))
    _write_json(
        cache / "manifest.json",
        {
            "num_samples": 4,
            "current_points": 10_000,
            "cache_mode": "agentview",
            "args": {"dataset_repo_id": str(dataset.resolve())},
            "fields": [
                "point_indices",
                "dataset_index",
                "episode_index",
                "frame_index",
            ],
            "shards": [
                {
                    "path": "shard_000000",
                    "start": 0,
                    "length": 4,
                    "num_points": 8,
                }
            ],
        },
    )
    return dataset, cache


def test_data_audit_checks_finite_sidecars_and_cache_index_alignment(
    tmp_path: Path,
) -> None:
    dataset, cache = _make_data_contract(tmp_path)
    output = tmp_path / "audit.json"
    args = Namespace(
        dataset=str(dataset),
        cache=str(cache),
        expected_hash=None,
        output=str(output),
    )
    assert tools.command_data_audit(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all_pose9_values_finite"] is True
    assert payload["episode_boundaries_verified"] is True
    assert payload["cache_dataset_index_alignment_verified"] is True

    commanded = dataset / "action_target_ee_poses" / "episode_000000.npy"
    bad_pose = np.zeros((2, 9), dtype=np.float32)
    bad_pose[0, 0] = np.nan
    np.save(commanded, bad_pose)
    with pytest.raises(ValueError, match="non-finite"):
        tools.command_data_audit(args)

    np.save(commanded, np.zeros((2, 9), dtype=np.float32))
    np.save(
        cache / "shard_000000" / "dataset_index.npy",
        np.asarray([0, 1, 3, 2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="not contiguous"):
        tools.command_data_audit(args)
