import os
import shlex
import subprocess
from pathlib import Path

RUNNER = Path("benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh").resolve()


def _run_admission_probe(tmp_path: Path, *, trainer_running: bool) -> bool:
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
export SONG_ABLATION_POST_TRAINING_EVAL_STAGGER_S=0
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


def test_evaluations_can_overlap_after_training_stops(tmp_path: Path) -> None:
    assert _run_admission_probe(tmp_path, trainer_running=False)


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
stage=$(stage_checkpoint_locally {shlex.quote(str(checkpoint))} variant 000001)
cmp {shlex.quote(str(checkpoint / 'model.safetensors'))} "$stage/pretrained_model/model.safetensors"
cmp {shlex.quote(str(weights / 'model.safetensors'))} "$stage/vlm_weights/model.safetensors"
[[ ! -e "$stage/vlm_architecture/model.safetensors" ]]
jq -e --arg stage "$stage" \
  '.vlm_model_name == ($stage + "/vlm_architecture") and .vlm_weights_path == ($stage + "/vlm_weights")' \
  "$stage/pretrained_model/config.json"
[[ "$stage" == {shlex.quote(str(stage_root))}/* ]]
"""

    subprocess.run(["bash", "-c", command], check=True)
