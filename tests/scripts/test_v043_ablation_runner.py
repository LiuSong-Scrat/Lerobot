import json
import os
import shlex
import subprocess
from pathlib import Path

RUNNER = Path("benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh").resolve()


def _write_nvidia_smi(tmp_path: Path, used_by_gpu: dict[int, int]) -> Path:
    executable = tmp_path / "nvidia-smi"
    lines = [f"{index}, 24564, {used_by_gpu.get(index, 0)}" for index in range(8)]
    executable.write_text("#!/bin/sh\nprintf '%s\\n' " + " ".join(map(shlex.quote, lines)) + "\n")
    executable.chmod(0o755)
    return executable


def test_gpu_preflight_ignores_busy_unrequested_gpu(tmp_path: Path) -> None:
    _write_nvidia_smi(tmp_path, {0: 3487})
    command = f"""
set -euo pipefail
export PATH={shlex.quote(str(tmp_path))}:$PATH
source {shlex.quote(str(RUNNER))}
gpu_preflight 1,2,3
"""

    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    assert "required=[1, 2, 3]" in result.stdout


def test_gpu_preflight_rejects_busy_requested_gpu(tmp_path: Path) -> None:
    _write_nvidia_smi(tmp_path, {2: 3487})
    command = f"""
set -euo pipefail
export PATH={shlex.quote(str(tmp_path))}:$PATH
source {shlex.quote(str(RUNNER))}
gpu_preflight 1,2,3
"""

    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "(2, 3487)" in result.stderr


def _run_admission_probe(
    tmp_path: Path, *, trainer_running: bool, training_eval_slots: int = 1
) -> bool:
    root = tmp_path / "output"
    (root / "pids").mkdir(parents=True)
    pid = os.getpid() if trainer_running else 999_999_999
    (root / "pids" / "train_smolvla_src.pid").write_text(str(pid))
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
export SONG_ABLATION_EVAL_LOCK={shlex.quote(str(tmp_path / 'eval.lock'))}
export SONG_ABLATION_GUARD_POLL_S=0.05
export SONG_ABLATION_POST_TRAINING_EVAL_SLOTS=2
export SONG_ABLATION_TRAINING_EVAL_SLOTS={training_eval_slots}
export SONG_ABLATION_POST_TRAINING_EVAL_STAGGER_S=0
export SONG_ABLATION_TRAINING_EVAL_STAGGER_S=0
source {shlex.quote(str(RUNNER))}
eval_result_valid() {{ [[ -f "$1/done" ]]; }}
run_eval_command() {{
  local output=$3
  local probe_fd probe_slot
  for probe_slot in 0 1; do
    exec {{probe_fd}}>{shlex.quote(str(tmp_path))}/probe_$probe_slot.lock
    if flock -n "$probe_fd"; then
      break
    fi
    exec {{probe_fd}}>&-
    probe_fd=
  done
  if [[ -z "$probe_fd" ]]; then
    touch "$root/over_capacity"
  fi
  if ! mkdir "$root/active_eval" 2>/dev/null; then
    touch "$root/overlap"
  fi
  sleep 0.2
  mkdir -p "$output"
  touch "$output/done"
  rmdir "$root/active_eval" 2>/dev/null || true
}}
eval_one smolvla_src 4 1 /checkpoint/a & first=$!
eval_one smolvla_pointcloud 5 1 /checkpoint/b & second=$!
eval_one smolvla_pointcloud_effseg 6 1 /checkpoint/c & third=$!
wait "$first"
wait "$second"
wait "$third"
[[ -f "$root/eval/smolvla_src/step000001/done" ]]
[[ -f "$root/eval/smolvla_pointcloud/step000001/done" ]]
[[ -f "$root/eval/smolvla_pointcloud_effseg/step000001/done" ]]
if [[ -f "$root/overlap" ]]; then
  echo overlap
else
  echo serial
fi
[[ ! -f "$root/over_capacity" ]]
"""
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    return result.stdout.strip() == "overlap"


def test_evaluations_are_serial_while_a_trainer_is_running(tmp_path: Path) -> None:
    assert not _run_admission_probe(tmp_path, trainer_running=True)


def test_evaluations_can_overlap_while_training_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    assert _run_admission_probe(
        tmp_path, trainer_running=True, training_eval_slots=2
    )


def test_evaluations_can_overlap_after_training_stops(tmp_path: Path) -> None:
    assert _run_admission_probe(tmp_path, trainer_running=False)


def test_eval_parallelism_uses_resource_phase_defaults() -> None:
    command = f"""
set -euo pipefail
source {shlex.quote(str(RUNNER))}
[[ "$training_eval_episode_workers_per_task" == 1 ]]
[[ "$training_eval_inference_batch_size" == 3 ]]
[[ "$training_eval_episode_workers_by_task" == "6=1,8=2" ]]
[[ "$single_trainer_eval_episode_workers_per_task" == 2 ]]
[[ "$single_trainer_eval_inference_batch_size" == 4 ]]
[[ "$single_trainer_eval_episode_workers_by_task" == "6=2,8=2" ]]
[[ "$training_eval_stagger_s" == 120 ]]
[[ "$post_training_eval_episode_workers_per_task" == 3 ]]
[[ "$post_training_eval_inference_batch_size" == 6 ]]
[[ "$post_training_eval_episode_workers_by_task" == "6=2,8=4" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_eval_parallelism_rejects_undersized_fixed_batch(tmp_path: Path) -> None:
    command = f"""
set -euo pipefail
export SONG_ABLATION_TRAINING_EVAL_EPISODE_WORKERS_PER_TASK=2
export SONG_ABLATION_TRAINING_EVAL_EPISODE_WORKERS_BY_TASK=
export SONG_ABLATION_TRAINING_EVAL_INFERENCE_BATCH_SIZE=3
source {shlex.quote(str(RUNNER))}
status=0
validate_eval_parallelism || status=$?
[[ "$status" == 2 ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_eval_parallelism_rejects_undersized_asymmetric_training_batch() -> None:
    command = f"""
set -euo pipefail
export SONG_ABLATION_TRAINING_EVAL_EPISODE_WORKERS_BY_TASK=6=1,8=2
export SONG_ABLATION_TRAINING_EVAL_INFERENCE_BATCH_SIZE=2
source {shlex.quote(str(RUNNER))}
status=0
validate_eval_parallelism || status=$?
[[ "$status" == 2 ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_eval_parallelism_rejects_undersized_single_trainer_batch() -> None:
    command = f"""
set -euo pipefail
export SONG_ABLATION_SINGLE_TRAINER_EVAL_EPISODE_WORKERS_BY_TASK=6=2,8=2
export SONG_ABLATION_SINGLE_TRAINER_EVAL_INFERENCE_BATCH_SIZE=3
source {shlex.quote(str(RUNNER))}
status=0
validate_eval_parallelism || status=$?
[[ "$status" == 2 ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def _run_parallelism_phase_probe(
    tmp_path: Path, *, trainer_running: bool, trainer_count: int = 1
) -> str:
    root = tmp_path / "output"
    (root / "pids").mkdir(parents=True)
    trainer_pid = os.getpid() if trainer_running else 999_999_999
    (root / "pids" / "train_smolvla_src.pid").write_text(str(trainer_pid))
    if trainer_running and trainer_count > 1:
        (root / "pids" / "train_smolvla_pointcloud.pid").write_text(str(trainer_pid))
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
export SONG_ABLATION_EVAL_LOCK={shlex.quote(str(tmp_path / 'eval.lock'))}
export SONG_ABLATION_TRAINING_EVAL_SLOTS=2
export SONG_ABLATION_POST_TRAINING_EVAL_STAGGER_S=0
export SONG_ABLATION_TRAINING_EVAL_STAGGER_S=0
source {shlex.quote(str(RUNNER))}
eval_result_valid() {{ [[ -f "$1/done" ]]; }}
run_eval_command() {{
  printf '%s %s %s %s\n' "$5" "$6" "$7" "${{8:-none}}" >"$root/observed_parallelism"
  mkdir -p "$3"
  touch "$3/done"
}}
eval_one smolvla_src 4 1 /checkpoint
cat "$root/observed_parallelism"
"""
    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_eval_command_receives_training_parallelism(tmp_path: Path) -> None:
    assert _run_parallelism_phase_probe(tmp_path, trainer_running=True) == "false 2 4 6=2,8=2"


def test_eval_command_uses_lower_parallelism_with_multiple_trainers(tmp_path: Path) -> None:
    assert (
        _run_parallelism_phase_probe(tmp_path, trainer_running=True, trainer_count=2)
        == "false 1 3 6=1,8=2"
    )


def test_eval_command_receives_post_training_parallelism(tmp_path: Path) -> None:
    assert _run_parallelism_phase_probe(tmp_path, trainer_running=False) == "true 3 6 6=2,8=4"


def test_checkpoint_is_staged_locally_with_bandwidth_limit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "config.json").write_text(
        '{"vlm_model_name":"remote-architecture","vlm_weights_path":"remote-weights"}'
    )
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    (vlm / "config.json").write_text("{}")
    (vlm / "tokenizer.json").write_text("{}")
    (vlm / "model.safetensors").write_bytes(b"excluded")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    (weights / "model.safetensors").write_bytes(b"weights")
    stage_root = tmp_path / "stages"
    command = f"""
set -euo pipefail
export SONG_ABLATION_CHECKPOINT_STAGE_ROOT={shlex.quote(str(stage_root))}
export SONG_ABLATION_CHECKPOINT_STAGE_BWLIMIT_KIB=1024
export SONG_ABLATION_VLM={shlex.quote(str(vlm))}
export SONG_ABLATION_VLM_WEIGHTS={shlex.quote(str(weights))}
source {shlex.quote(str(RUNNER))}
stage=$(stage_checkpoint_locally {shlex.quote(str(checkpoint))} variant 000001 "$$")
cmp {shlex.quote(str(checkpoint / 'model.safetensors'))} "$stage/pretrained_model/model.safetensors"
cmp {shlex.quote(str(weights / 'model.safetensors'))} "$stage/vlm_weights/model.safetensors"
[[ ! -e "$stage/vlm_architecture/model.safetensors" ]]
jq -e --arg stage "$stage" \
  '.vlm_model_name == ($stage + "/vlm_architecture") and .vlm_weights_path == ($stage + "/vlm_weights")' \
  "$stage/pretrained_model/config.json"
[[ "$stage" == {shlex.quote(str(stage_root))}/* ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_stale_checkpoint_stages_are_reclaimed_without_touching_live_owner(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stages"
    command = f"""
set -euo pipefail
export SONG_ABLATION_CHECKPOINT_STAGE_ROOT={shlex.quote(str(stage_root))}
source {shlex.quote(str(RUNNER))}
mkdir -p "$checkpoint_stage_root/dead" "$checkpoint_stage_root/live"
printf '%s\n' 999999999 > "$checkpoint_stage_root/dead/.stage_owner_pid"
printf '%s\n' "$$" > "$checkpoint_stage_root/live/.stage_owner_pid"
cleanup_stale_stage_directories
[[ ! -e "$checkpoint_stage_root/dead" ]]
[[ -d "$checkpoint_stage_root/live" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_resource_interruption_waits_and_requests_retry(tmp_path: Path) -> None:
    root = tmp_path / "output"
    marker = root / "resource" / "EVAL_TERMINATED_RESOURCE_LIMIT"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
eval_one() {{ return 143; }}
guard_wait() {{ touch "$root/guard_wait_called"; rm -f {shlex.quote(str(marker))}; }}
status=0
eval_one_resilient smolvla_src 4 1 /checkpoint || status=$?
[[ "$status" == 75 ]]
[[ -f "$root/guard_wait_called" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_non_resource_eval_failure_is_not_retried(tmp_path: Path) -> None:
    root = tmp_path / "output"
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
eval_one() {{ return 23; }}
guard_wait() {{ touch "$root/guard_wait_called"; }}
status=0
eval_one_resilient smolvla_src 4 1 /checkpoint || status=$?
[[ "$status" == 23 ]]
[[ ! -e "$root/guard_wait_called" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_zero_episode_dead_claim_is_cleared_and_restarted(tmp_path: Path) -> None:
    root = tmp_path / "output"
    output = root / "eval" / "smolvla_src" / "step000001"
    claim = output / ".evaluation_run.claim"
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text(json.dumps({"pid": 999_999_999}))
    (output / "failed_episodes.json").write_text(json.dumps({"failure_count": 0}))
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
export SONG_ABLATION_TRAINING_EVAL_SLOTS=2
export SONG_ABLATION_TRAINING_EVAL_STAGGER_S=0
source {shlex.quote(str(RUNNER))}
training_is_running() {{ :; }}
acquire_training_eval_slot() {{ :; }}
eval_result_valid() {{ [[ -f "$1/done" ]]; }}
run_eval_command() {{ mkdir -p "$3"; touch "$3/done"; }}
eval_one smolvla_src 4 1 /checkpoint
[[ -f {shlex.quote(str(output / 'done'))} ]]
[[ ! -e {shlex.quote(str(claim))} ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_zero_episode_live_claim_is_not_cleared(tmp_path: Path) -> None:
    root = tmp_path / "output"
    output = root / "eval" / "smolvla_src" / "step000001"
    claim = output / ".evaluation_run.claim"
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text(json.dumps({"pid": os.getpid()}))
    (output / "failed_episodes.json").write_text(json.dumps({"failure_count": 0}))
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
eval_result_valid() {{ return 1; }}
run_eval_command() {{ touch "$root/incorrectly_started"; }}
status=0
eval_one smolvla_src 4 1 /checkpoint || status=$?
[[ "$status" == 1 ]]
[[ -f {shlex.quote(str(claim / 'owner.json'))} ]]
[[ ! -e "$root/incorrectly_started" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_unstable_variants_reads_only_false_entries(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "stability.json").write_text(
        '{"variants": {'
        '"smolvla_src": {"stable": true}, '
        '"smolvla_pointcloud": {"stable": false}, '
        '"smolvla_pointcloud_effseg": {"stable": true}, '
        '"smolvla_pointcloud_effseg_pointaction": {"stable": false}'
        '}}'
    )
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
unstable_variants
"""

    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    assert result.stdout.splitlines() == [
        "smolvla_pointcloud",
        "smolvla_pointcloud_effseg_pointaction",
    ]


def test_latest_resumable_checkpoint_requires_complete_training_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    checkpoints = root / "train" / "smolvla_pointcloud" / "checkpoints"
    incomplete = checkpoints / "040000"
    complete = checkpoints / "030000"
    for checkpoint in (incomplete, complete):
        (checkpoint / "pretrained_model").mkdir(parents=True)
        (checkpoint / "training_state").mkdir()
        (checkpoint / "pretrained_model" / "train_config.json").write_text("{}")
        (checkpoint / "training_state" / "training_step.json").write_text("{}")
    (complete / "training_state" / "optimizer_state.safetensors").write_text("state")
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
latest_resumable_checkpoint smolvla_pointcloud
"""

    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    assert Path(result.stdout.strip()) == complete


def test_extension_prevalidates_all_candidates_before_starting(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
preflight() {{ :; }}
unstable_variants() {{ printf '%s\n' smolvla_pointcloud smolvla_pointcloud_effseg_pointaction; }}
variant_evaluations_complete() {{ [[ "$1" == smolvla_pointcloud ]]; }}
extend_train_one() {{ touch "$root/training_started"; }}
status=0
extend_unstable || status=$?
[[ "$status" == 1 ]]
[[ ! -e "$root/training_started" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)


def test_extension_starts_matching_evaluation_watcher(tmp_path: Path) -> None:
    root = tmp_path / "output"
    (root / "pids").mkdir(parents=True)
    command = f"""
set -euo pipefail
export SONG_ABLATION_OUTPUT_ROOT={shlex.quote(str(root))}
source {shlex.quote(str(RUNNER))}
preflight() {{ :; }}
unstable_variants() {{ printf '%s\n' smolvla_pointcloud; }}
variant_evaluations_complete() {{ :; }}
guard_wait() {{ :; }}
wait_for_training_gpu_allocation() {{ :; }}
extend_train_one() {{ touch "$root/training_started"; }}
eval_watch_one() {{ printf '%s %s\n' "$1" "$2" >"$root/eval_watcher_started"; }}
summarize() {{ touch "$root/summarized"; }}
extend_unstable
[[ -f "$root/training_started" ]]
[[ "$(cat "$root/eval_watcher_started")" == "smolvla_pointcloud 5" ]]
[[ -f "$root/summarized" ]]
"""

    subprocess.run(["bash", "-c", command], check=True)
