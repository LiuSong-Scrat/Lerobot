#!/usr/bin/env bash
set -euo pipefail

# Prefer the valid versioned NVIDIA driver over this host's stale zero-byte
# unversioned symlink.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

repo=${SONG_ABLATION_REPO:-/home/liusong/ProgramFiles/Huggingface/lerobot}
python=${SONG_ABLATION_PYTHON:-/home/liusong/anaconda3/envs/reap/bin/python}
dataset=${SONG_ABLATION_DATASET:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep}
cache=${SONG_ABLATION_CACHE:-/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep_pointseg_cache}
vlm=${SONG_ABLATION_VLM:-/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct}
vlm_weights=${SONG_ABLATION_VLM_WEIGHTS:-/opt/data/private/liusong/hf_models/smolvla_base}
root=${SONG_ABLATION_OUTPUT_ROOT:-/opt/data/private/liusong/benchmarks/song_real_libero/outputs/v043_cumulative_ablation_task6_task8_20260829}
steps=${SONG_ABLATION_STEPS:-30000}
extension_steps=${SONG_ABLATION_EXTENSION_STEPS:-60000}
checkpoint_interval_s=${SONG_ABLATION_CHECKPOINT_INTERVAL_S:-3600}
batch_size=${SONG_ABLATION_BATCH_SIZE:-8}
num_workers=${SONG_ABLATION_NUM_WORKERS:-2}
eval_episodes=${SONG_ABLATION_EVAL_EPISODES:-10}
guard_sample_s=${SONG_ABLATION_GUARD_SAMPLE_S:-5}
guard_poll_s=${SONG_ABLATION_GUARD_POLL_S:-15}
eval_lock=${SONG_ABLATION_EVAL_LOCK:-/tmp/song_real_libero_v043_eval.lock}
summary_lock=${SONG_ABLATION_SUMMARY_LOCK:-/tmp/song_real_libero_v043_summary.lock}
post_training_eval_slots=${SONG_ABLATION_POST_TRAINING_EVAL_SLOTS:-1}
training_eval_slots=${SONG_ABLATION_TRAINING_EVAL_SLOTS:-1}
post_training_eval_stagger_s=${SONG_ABLATION_POST_TRAINING_EVAL_STAGGER_S:-60}
training_eval_episode_workers_per_task=${SONG_ABLATION_TRAINING_EVAL_EPISODE_WORKERS_PER_TASK:-1}
post_training_eval_episode_workers_per_task=${SONG_ABLATION_POST_TRAINING_EVAL_EPISODE_WORKERS_PER_TASK:-4}
training_eval_inference_batch_size=${SONG_ABLATION_TRAINING_EVAL_INFERENCE_BATCH_SIZE:-$((2 * training_eval_episode_workers_per_task))}
post_training_eval_inference_batch_size=${SONG_ABLATION_POST_TRAINING_EVAL_INFERENCE_BATCH_SIZE:-$((2 * post_training_eval_episode_workers_per_task))}
checkpoint_stage_root=${SONG_ABLATION_CHECKPOINT_STAGE_ROOT:-/tmp/song_real_libero_v043_checkpoints}
checkpoint_stage_bwlimit_kib=${SONG_ABLATION_CHECKPOINT_STAGE_BWLIMIT_KIB:-102400}

variants=(
  smolvla_src
  smolvla_pointcloud
  smolvla_pointcloud_effseg
  smolvla_pointcloud_effseg_pointaction
)
train_gpus=(0 1 2 3)
eval_gpus=(4 5 6 7)
guard="$repo/benchmarks/song_real_libero/scripts/ablation_resource_guard.py"
cache_reclaimer="$repo/benchmarks/song_real_libero/scripts/checkpoint_cache_reclaimer.py"
summarizer="$repo/benchmarks/song_real_libero/scripts/summarize_v043_ablation.py"

usage() {
  echo "usage: $0 {preflight|train|extend|eval|all|summarize|status}"
}

validate_eval_parallelism() {
  if (( training_eval_episode_workers_per_task < 1 || training_eval_episode_workers_per_task > eval_episodes )); then
    echo "Training-time episode workers per task must be between one and $eval_episodes." >&2
    return 2
  fi
  if (( post_training_eval_episode_workers_per_task < 1 || post_training_eval_episode_workers_per_task > eval_episodes )); then
    echo "Post-training episode workers per task must be between one and $eval_episodes." >&2
    return 2
  fi
  if (( training_eval_inference_batch_size < 2 * training_eval_episode_workers_per_task )); then
    echo "Training-time fixed-barrier batch must cover both tasks' episode workers." >&2
    return 2
  fi
  if (( post_training_eval_inference_batch_size < 2 * post_training_eval_episode_workers_per_task )); then
    echo "Post-training fixed-barrier batch must cover both tasks' episode workers." >&2
    return 2
  fi
}

require_inputs() {
  test -f "$dataset/meta/info.json"
  test -f "$vlm/config.json"
  test -f "$vlm_weights/model.safetensors"
  test -f "$cache/manifest.json"
  test -f "$guard"
  test -f "$cache_reclaimer"
  if (( steps <= 0 || checkpoint_interval_s <= 0 )); then
    echo "steps and checkpoint interval must be positive" >&2
    exit 2
  fi
  if (( eval_episodes != 10 )); then
    echo "The fixed task 6/8 test protocol requires 10 episodes per task." >&2
    exit 2
  fi
  if (( num_workers > 2 )); then
    echo "Refusing num_workers=$num_workers: four trainers are capped at two workers each." >&2
    exit 2
  fi
  if (( post_training_eval_slots < 1 || post_training_eval_slots > 2 )); then
    echo "Post-training evaluation concurrency must be one or two." >&2
    exit 2
  fi
  if (( training_eval_slots < 1 || training_eval_slots > 2 )); then
    echo "Training-time evaluation concurrency must be one or two." >&2
    exit 2
  fi
  if (( post_training_eval_stagger_s < 0 )); then
    echo "Post-training evaluation staggering must be non-negative." >&2
    exit 2
  fi
  validate_eval_parallelism
  if (( checkpoint_stage_bwlimit_kib <= 0 )); then
    echo "Checkpoint staging bandwidth must be positive." >&2
    exit 2
  fi
  command -v rsync >/dev/null
}

guard_wait() {
  local gpu=$1
  "$python" "$guard" --root "$root" --gpu "$gpu" --wait \
    --sample-seconds "$guard_sample_s" --poll-seconds "$guard_poll_s" --consecutive 3 \
    --recover-hard-marker-after-soft
}

wait_for_training_gpu_allocation() {
  local gpu=$1 pid=$2
  local deadline=$((SECONDS + 600))
  while kill -0 "$pid" 2>/dev/null; do
    local used_mib
    used_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    used_mib=${used_mib//[[:space:]]/}
    if (( used_mib >= 2048 )); then
      echo "[train] gpu=$gpu initialized with ${used_mib} MiB"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "[train] timed out waiting for GPU $gpu initialization" >&2
      return 1
    fi
    sleep 5
  done
  echo "[train] process $pid ended before GPU $gpu initialization" >&2
  return 1
}

gpu_preflight() {
  local required_gpu_csv=${1:-0,1,2,3}
  "$python" - "$required_gpu_csv" <<'PY'
import subprocess
import sys

required_gpu_ids = {int(value) for value in sys.argv[1].split(",") if value}
gpu_lines = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
assert len(gpu_lines) == 8, gpu_lines
inventory = {}
for line in gpu_lines:
    index, total, used = [int(value.strip()) for value in line.split(",")]
    assert total >= 24_000, (index, total)
    inventory[index] = used
assert required_gpu_ids.issubset(inventory), (required_gpu_ids, inventory.keys())
for index in sorted(required_gpu_ids):
    assert inventory[index] < 2_048, (index, inventory[index])
print(f"gpu preflight PASS: gpus={len(gpu_lines)} required={sorted(required_gpu_ids)}")
PY
}

preflight() {
  local required_gpu_csv=${1:-0,1,2,3}
  local required_gpu
  local -a required_gpu_ids=()
  require_inputs
  cd "$repo"
  local branch
  branch=$(git branch --show-current)
  if [[ "$branch" != "wep_vla_v0.4.3_multiview_doubleflow_ablation" ]]; then
    echo "Expected ablation branch, got ${branch:-detached HEAD}." >&2
    exit 2
  fi
  mkdir -p "$root/logs" "$root/pids" "$root/resource"
  PYTHONPATH="$repo/src" "$python" - "$dataset" "$cache" "$eval_episodes" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

dataset = Path(sys.argv[1])
cache = Path(sys.argv[2])
eval_episodes = int(sys.argv[3])
info = json.loads((dataset / "meta/info.json").read_text())
manifest = json.loads((cache / "manifest.json").read_text())
assert info["total_episodes"] == 100, info["total_episodes"]
assert info["total_tasks"] == 2, info["total_tasks"]
assert info["features"]["action"]["shape"] == [10]
assert manifest["num_samples"] == info["total_frames"]
assert manifest["current_points"] == 10_000
assert eval_episodes == 10
episode_tasks = set()
for parquet_path in sorted((dataset / "data").rglob("*.parquet")):
    columns = pq.read_table(parquet_path, columns=["episode_index", "task_index"]).to_pydict()
    episode_tasks.update(zip(columns["episode_index"], columns["task_index"], strict=True))
assert Counter(task for _, task in episode_tasks) == {0: 50, 1: 50}, episode_tasks

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
expected = {
    "smolvla_src": (False, False, False),
    "smolvla_pointcloud": (True, False, False),
    "smolvla_pointcloud_effseg": (True, True, False),
    "smolvla_pointcloud_effseg_pointaction": (True, True, True),
}
for name, gates in expected.items():
    cfg = SmolVLAConfig(ablation_variant=name)
    actual = (cfg.pointcloud_enable, cfg.pointseg_enable, cfg.point_action_fusion_enable)
    assert actual == gates, (name, actual, gates)
    assert cfg.vla_adapter_enable and cfg.vla_adapter_freeze_vlm
    assert not cfg.worldflow_enable
    assert cfg.pointcloud_input_points == 10_000

print(
    f"ablation preflight PASS: frames={info['total_frames']} episodes={info['total_episodes']} "
    f"test_episodes={2 * eval_episodes}"
)
PY
  gpu_preflight "$required_gpu_csv"
  IFS=, read -ra required_gpu_ids <<<"$required_gpu_csv"
  for required_gpu in "${required_gpu_ids[@]}"; do
    "$python" "$guard" --root "$root" --gpu "$required_gpu" --sample-seconds 1
  done
}

train_one() {
  local variant=$1 gpu=$2
  local output="$root/train/$variant"
  local log="$root/logs/train_${variant}.log"
  local final_tag
  printf -v final_tag '%06d' "$steps"
  if [[ -s "$output/checkpoints/$final_tag/pretrained_model/model.safetensors" ]]; then
    echo "[train] reuse completed $variant"
    return
  fi
  if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[train] incomplete output exists for $variant: $output" >&2
    return 1
  fi
  mkdir -p "$output" "$root/logs"
  local -a cache_args=()
  if [[ "$variant" == smolvla_pointcloud_effseg* ]]; then
    cache_args=(--pointseg_sample_cache_dir="$cache")
  fi
  cd "$repo"
  exec env \
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo/src" SONG_POINTSEG_REQUIRE_POINTOPS=1 \
    "$python" benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --policy.type=smolvla --policy.push_to_hub=false \
      --dataset.repo_id="$dataset" "${cache_args[@]}" \
      --task_balanced_sampling=true \
      --batch_size="$batch_size" --gradient_accumulation_steps=1 \
      --steps="$steps" --save_freq="$steps" --save_interval_s="$checkpoint_interval_s" \
      --eval_freq=0 --log_freq=10 --num_workers="$num_workers" \
      --output_dir="$output" --job_name="v043_ablation_${variant}" \
      --policy.device=cuda --wandb.enable=false \
      --policy.ablation_variant="$variant" \
      --policy.camera_views=agentview --policy.rgb_camera_views=agentview \
      --policy.vlm_model_name="$vlm" --policy.vlm_weights_path="$vlm_weights" \
      --policy.load_vlm_weights=true \
      --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 \
      --policy.scheduler_decay_steps="$steps" --policy.scheduler_decay_lr=0.0000025 \
      --policy.pointseg_backbone_type=litept --policy.pointseg_grid_size=0.01 \
      --policy.pointseg_feature_dim=64 --policy.pointseg_aux_loss_weight=0.0005 \
      --policy.pointseg_foreground_ratio=0.025 --policy.pointseg_background_ratio=0.025 \
      --policy.pointseg_min_foreground_points=2500 --policy.pointseg_min_background_points=0 \
      --policy.pointseg_use_temporal_priors_as_input=false \
      --policy.pointseg_use_pseudo_selection=false \
      >"$log" 2>&1
}

train_all() {
  preflight
  local pids=()
  for i in "${!variants[@]}"; do
    guard_wait "${train_gpus[$i]}"
    train_one "${variants[$i]}" "${train_gpus[$i]}" &
    pids+=("$!")
    echo "$!" > "$root/pids/train_${variants[$i]}.pid"
    wait_for_training_gpu_allocation "${train_gpus[$i]}" "$!"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

unstable_variants() {
  "$python" - "$root/stability.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Missing stability.json; finish and summarize current evaluations first")
payload = json.loads(path.read_text())
for variant, result in payload.get("variants", {}).items():
    if result.get("stable") is False:
        print(variant)
PY
}

variant_evaluations_complete() {
  local variant=$1 model_path checkpoint step_tag
  while IFS= read -r model_path; do
    checkpoint=$(dirname "$model_path")
    step_tag=$(basename "$(dirname "$checkpoint")")
    if ! eval_result_valid "$root/eval/$variant/step$step_tag"; then
      return 1
    fi
  done < <(find "$root/train/$variant/checkpoints" -mindepth 3 -maxdepth 3 \
    -path '*/pretrained_model/model.safetensors' -type f 2>/dev/null | sort -V)
}

latest_resumable_checkpoint() {
  local variant=$1 checkpoint
  while IFS= read -r checkpoint; do
    if [[ -s "$checkpoint/pretrained_model/train_config.json" \
      && -s "$checkpoint/training_state/training_step.json" \
      && -s "$checkpoint/training_state/optimizer_state.safetensors" ]]; then
      printf '%s\n' "$checkpoint"
      return 0
    fi
  done < <(find "$root/train/$variant/checkpoints" -mindepth 1 -maxdepth 1 \
    -type d 2>/dev/null | sort -Vr)
  return 1
}

extend_train_one() {
  local variant=$1 gpu=$2 checkpoint output log target_tag
  output="$root/train/$variant"
  log="$root/logs/train_${variant}_extend_to_${extension_steps}.log"
  printf -v target_tag '%06d' "$extension_steps"
  if [[ -s "$output/checkpoints/$target_tag/pretrained_model/model.safetensors" ]]; then
    echo "[extend] reuse completed $variant target=$target_tag"
    return 0
  fi
  checkpoint=$(latest_resumable_checkpoint "$variant") || {
    echo "[extend] no resumable checkpoint for $variant" >&2
    return 1
  }
  cd "$repo"
  exec env \
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo/src" SONG_POINTSEG_REQUIRE_POINTOPS=1 \
    "$python" benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --resume=true \
      --config_path="$checkpoint/pretrained_model/train_config.json" \
      --steps="$extension_steps" --save_freq="$extension_steps" \
      --save_interval_s="$checkpoint_interval_s" --eval_freq=0 --log_freq=10 \
      --num_workers="$num_workers" --wandb.enable=false \
      --output_dir="$output" --job_name="v043_ablation_${variant}_extended" \
      >>"$log" 2>&1
}

extend_unstable() {
  local variant index gpu pid failed=0 required_gpu_csv
  local candidates=() required_gpus=() train_pids=() eval_pids=()
  if (( extension_steps <= steps )); then
    echo "[extend] extension_steps must exceed the original training steps" >&2
    return 2
  fi
  mapfile -t candidates < <(unstable_variants)
  if (( ${#candidates[@]} == 0 )); then
    echo "[extend] all variants are already stable"
    return 0
  fi
  for variant in "${candidates[@]}"; do
    if ! variant_evaluations_complete "$variant"; then
      echo "[extend] pending evaluations remain for $variant; refusing early extension" >&2
      return 1
    fi
    index=$(variant_index "$variant")
    required_gpus+=("${train_gpus[$index]}")
  done
  required_gpu_csv=$(IFS=,; echo "${required_gpus[*]}")
  preflight "$required_gpu_csv"
  for variant in "${candidates[@]}"; do
    index=$(variant_index "$variant")
    gpu=${train_gpus[$index]}
    guard_wait "$gpu"
    extend_train_one "$variant" "$gpu" &
    pid=$!
    train_pids+=("$pid")
    printf '%s\n' "$pid" >"$root/pids/train_${variant}.pid"
    wait_for_training_gpu_allocation "$gpu" "$pid"
    eval_watch_one "$variant" "${eval_gpus[$index]}" &
    pid=$!
    eval_pids+=("$pid")
    printf '%s\n' "$pid" >"$root/pids/eval_${variant}.pid"
  done
  for pid in "${train_pids[@]}"; do
    wait "$pid" || failed=1
  done
  for pid in "${eval_pids[@]}"; do
    wait "$pid" || failed=1
  done
  summarize
  return "$failed"
}

checkpoint_ready() {
  local checkpoint=$1
  test -s "$checkpoint/model.safetensors" \
    && test -s "$checkpoint/config.json" \
    && test -s "$checkpoint/policy_preprocessor.json" \
    && test -s "$checkpoint/policy_postprocessor.json"
}

eval_result_valid() {
  local output=$1
  "$python" - "$output/summary.json" "$eval_episodes" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
episodes_per_task = int(sys.argv[2])
if not summary_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text())
expected_total = 2 * episodes_per_task
if summary.get("overall", {}).get("episode_count") != expected_total:
    raise SystemExit(1)
tasks = [task for suite in summary.get("suite_reports", []) for task in suite.get("tasks", [])]
if {task.get("task_id") for task in tasks} != {6, 8}:
    raise SystemExit(1)
episodes = [episode for task in tasks for episode in task.get("episodes", [])]
if len(episodes) != expected_total:
    raise SystemExit(1)
if any(episode.get("error") not in (None, "") for episode in episodes):
    raise SystemExit(1)
if any(int(episode.get("model_call_count", 0)) <= 0 for episode in episodes):
    raise SystemExit(1)
PY
}

eval_output_resumable() {
  local output=$1
  "$python" - "$output" "$eval_episodes" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
episodes_per_task = int(sys.argv[2])
suite_dir = output / "libero_10"
progress_path = suite_dir / "progress.json"
if not progress_path.is_file():
    raise SystemExit(1)
try:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected_total = 2 * episodes_per_task
if progress.get("suite") != "libero_10":
    raise SystemExit(1)
if progress.get("expected_task_ids") != [6, 8]:
    raise SystemExit(1)
if progress.get("episodes_per_task") != episodes_per_task:
    raise SystemExit(1)
if progress.get("expected_episode_count") != expected_total:
    raise SystemExit(1)

expected_protocol = {
    "name": "single_uninterrupted_rollout",
    "rollouts_per_initial_state": 1,
    "retry_failed_rollout": False,
    "action_samples_per_model_call": 1,
    "action_sample_selection": "none",
    "initial_state_source": "task_suite.get_task_init_states",
    "fixture_reset_sequence": "seeded_serial_episode_index",
    "benchmark_comparable": True,
}
expected_paths = {
    suite_dir / f"task_{task_id:03d}" / f"episode_{episode_index:03d}" / "result.json"
    for task_id in (6, 8)
    for episode_index in range(episodes_per_task)
}
actual_paths = set(suite_dir.glob("task_*/episode_*/result.json"))
if not actual_paths.issubset(expected_paths):
    raise SystemExit(1)
for result_path in actual_paths:
    try:
        record = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1)
    expected_index = int(result_path.parent.name.removeprefix("episode_"))
    if record.get("episode_index") != expected_index:
        raise SystemExit(1)
    if record.get("error") not in (None, ""):
        raise SystemExit(1)
    if int(record.get("steps", 0) or 0) <= 0:
        raise SystemExit(1)
    if int(record.get("max_steps", -1)) != 1000:
        raise SystemExit(1)
    if int(record.get("model_call_count", 0) or 0) <= 0:
        raise SystemExit(1)
    if int(record.get("policy_forward_call_count", 0) or 0) <= 0:
        raise SystemExit(1)
    if record.get("action_source") != "policy_flow_matching_sample":
        raise SystemExit(1)
    if record.get("evaluation_protocol") != expected_protocol:
        raise SystemExit(1)
    if record.get("policy_noise_seed_base") != 0:
        raise SystemExit(1)
    if record.get("strict_official_init") is not True:
        raise SystemExit(1)
    alignment = record.get("environment_alignment")
    if not isinstance(alignment, dict) or alignment.get("benchmark_comparable") is not True:
        raise SystemExit(1)
    action_npz = Path(str(record.get("action_npz", "")))
    if action_npz != result_path.parent / "actions.npz":
        raise SystemExit(1)
    if not action_npz.is_file() or action_npz.stat().st_size <= 0:
        raise SystemExit(1)
PY
}

training_is_running() {
  local variant pid
  for variant in "${variants[@]}"; do
    pid=$(cat "$root/pids/train_${variant}.pid" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

variant_index() {
  local requested=$1 index
  for index in "${!variants[@]}"; do
    if [[ "${variants[$index]}" == "$requested" ]]; then
      echo "$index"
      return 0
    fi
  done
  return 1
}

acquire_post_training_eval_slot() {
  acquire_eval_slot "$1" "$post_training_eval_slots"
}

acquire_training_eval_slot() {
  acquire_eval_slot "$1" "$training_eval_slots"
}

acquire_eval_slot() {
  local output_variable=$1 slot_count=$2 slot candidate_fd
  while true; do
    for ((slot = 0; slot < slot_count; slot++)); do
      exec {candidate_fd}>"${eval_lock}.slot${slot}"
      if flock -n "$candidate_fd"; then
        printf -v "$output_variable" '%s' "$candidate_fd"
        return 0
      fi
      exec {candidate_fd}>&-
    done
    sleep "$guard_poll_s"
  done
}

cleanup_stage_directory() {
  local stage_dir=$1
  if [[ -z "$stage_dir" || ! -d "$stage_dir" || "$stage_dir" != "$checkpoint_stage_root/"* ]]; then
    return 2
  fi
  find "$stage_dir" -depth -delete
}

stage_checkpoint_locally() {
  local checkpoint=$1 variant=$2 step=$3 stage_dir stage_checkpoint stage_vlm stage_vlm_weights
  mkdir -p "$checkpoint_stage_root"
  stage_dir=$(mktemp -d "$checkpoint_stage_root/${variant}_step${step}.XXXXXX")
  stage_checkpoint="$stage_dir/pretrained_model"
  stage_vlm="$stage_dir/vlm_architecture"
  stage_vlm_weights="$stage_dir/vlm_weights"
  mkdir -p "$stage_checkpoint" "$stage_vlm" "$stage_vlm_weights"
  if ! rsync --archive --bwlimit="$checkpoint_stage_bwlimit_kib" \
    "$checkpoint/" "$stage_checkpoint/" || \
    ! rsync --archive --bwlimit="$checkpoint_stage_bwlimit_kib" \
      "$vlm_weights/" "$stage_vlm_weights/" || \
    ! rsync --archive --bwlimit="$checkpoint_stage_bwlimit_kib" \
      --exclude=model.safetensors --exclude=onnx/ --exclude=.git/ \
      "$vlm/" "$stage_vlm/"; then
    cleanup_stage_directory "$stage_dir"
    return 1
  fi
  if [[ ! -s "$stage_checkpoint/model.safetensors" || ! -s "$stage_checkpoint/config.json" ]]; then
    cleanup_stage_directory "$stage_dir"
    return 1
  fi
  "$python" - "$stage_checkpoint/config.json" "$stage_vlm" "$stage_vlm_weights" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
original_weights = config.get("vlm_weights_path")
config["vlm_model_name"] = sys.argv[2]
config["vlm_weights_path"] = sys.argv[3]
if config.get("action_expert_weights_path") == original_weights:
    config["action_expert_weights_path"] = sys.argv[3]
path.write_text(json.dumps(config, indent=4) + "\n")
PY
  echo "$stage_dir"
}

run_eval_command() {
  local gpu=$1 checkpoint=$2 output=$3 log=$4 stage_checkpoint=${5:-false}
  local episode_workers_per_task=${6:-$training_eval_episode_workers_per_task}
  local inference_batch_size=${7:-$training_eval_inference_batch_size}
  local eval_checkpoint=$checkpoint stage_dir= stage_reclaimer_pid= status=0
  guard_wait "$gpu"
  if [[ "$stage_checkpoint" == true ]]; then
    stage_dir=$(stage_checkpoint_locally \
      "$checkpoint" "$(basename "$(dirname "$output")")" "$(basename "$output")")
    eval_checkpoint="$stage_dir/pretrained_model"
  fi
  mkdir -p "$output" "$root/logs"
  if [[ -n "$stage_dir" ]]; then
    (
      while [[ ! -s "$output/progress.json" ]]; do
        sleep 1
      done
      "$python" "$cache_reclaimer" --checkpoint "$stage_dir" >/dev/null 2>&1 || true
      echo "[eval] released staged checkpoint cache: $stage_dir"
    ) &
    stage_reclaimer_pid=$!
  fi
  cd "$repo"
  env \
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$gpu" \
    MUJOCO_EGL_DEVICE_ID="$gpu" SONG_LIBERO_ENV_CUDA_VISIBLE_DEVICES="$gpu,0" \
    SONG_LIBERO_ENV_MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH="$repo/src" \
    "$python" benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
      --config benchmarks/song_real_libero/configs/libero.json \
      --policy.path "$eval_checkpoint" --device cuda \
      --num-points 10000 \
      --suite libero_10 --task-id 6 --task-id 8 --episodes "$eval_episodes" \
      --policy-noise-seed 0 --env-seed 7 --strict-official-init \
      --gripper-control-mode delta_width_initial_sync \
      --gripper-delta-threshold 0.002 \
      --gripper-delta-alignment current_minus_previous \
      --waypoint-max-hold-steps 1 --isolated-policy-workers 1 \
      --task-workers 2 --episode-workers-per-task "$episode_workers_per_task" \
      --task-worker-backend process \
      --inference-batch-size "$inference_batch_size" \
      --inference-batching-mode fixed_barrier \
      --no-release-event-exec-enable --control-freq 20 --action-index 0 \
      --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
      --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
      --render-mode offscreen --no-visualize-foreground --no-save-video \
      --no-world-to-ego-causal-ablation --output-dir "$output" \
      >"$log" 2>&1 || status=$?
  if [[ -n "$stage_reclaimer_pid" ]]; then
    kill "$stage_reclaimer_pid" 2>/dev/null || true
    wait "$stage_reclaimer_pid" 2>/dev/null || true
  fi
  if [[ -n "$stage_dir" ]]; then
    "$python" "$cache_reclaimer" --checkpoint "$stage_dir" >/dev/null 2>&1 || true
    cleanup_stage_directory "$stage_dir"
  fi
  return "$status"
}

eval_one() {
  local variant=$1 gpu=$2 step=$3 checkpoint=$4
  local step_tag
  printf -v step_tag '%06d' "$step"
  local output="$root/eval/$variant/step$step_tag"
  local log="$root/logs/eval_${variant}_step${step_tag}.log"
  if eval_result_valid "$output"; then
    echo "[eval] reuse $variant step=$step_tag"
    return
  fi
  if [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if ! eval_output_resumable "$output"; then
      echo "[eval] incomplete output is not safely resumable: $output" >&2
      return 1
    fi
    echo "[eval] resume incomplete output: $output"
  fi
  (
    if training_is_running && (( training_eval_slots > 1 )); then
      local training_slot_fd
      acquire_training_eval_slot training_slot_fd
      run_eval_command "$gpu" "$checkpoint" "$output" "$log" false \
        "$training_eval_episode_workers_per_task" "$training_eval_inference_batch_size"
      eval_result_valid "$output"
      exit
    fi

    exec 9>"$eval_lock"
    while training_is_running; do
      if flock -w "$guard_poll_s" 9; then
        if training_is_running; then
          run_eval_command "$gpu" "$checkpoint" "$output" "$log" false \
            "$training_eval_episode_workers_per_task" "$training_eval_inference_batch_size"
          eval_result_valid "$output"
          exit
        fi
        flock -u 9
      fi
    done

    # With no trainers resident, use bounded slots and stagger model loads.
    # The default remains one because evaluator process trees are host-memory
    # bound on the 60 GiB benchmark machine.
    local index slot_fd
    index=$(variant_index "$variant")
    sleep "$((index * post_training_eval_stagger_s))"
    acquire_post_training_eval_slot slot_fd
    run_eval_command "$gpu" "$checkpoint" "$output" "$log" true \
      "$post_training_eval_episode_workers_per_task" "$post_training_eval_inference_batch_size"
    eval_result_valid "$output"
  )
}

eval_one_resilient() {
  local variant=$1 gpu=$2 step=$3 checkpoint=$4 status=0
  eval_one "$variant" "$gpu" "$step" "$checkpoint" || status=$?
  if (( status == 0 )); then
    return 0
  fi
  if [[ ! -f "$root/resource/EVAL_TERMINATED_RESOURCE_LIMIT" ]]; then
    return "$status"
  fi
  echo "[eval] resource guard interrupted $variant step=$step; waiting to resume"
  guard_wait "$gpu"
  return 75
}

eval_watch_one() {
  local variant=$1 gpu=$2
  local train_pid_file="$root/pids/train_${variant}.pid"
  local stopped_idle_scans=0
  while true; do
    local found_pending=0 resource_interrupted=0
    while IFS= read -r model_path; do
      local checkpoint step_tag step eval_status=0
      checkpoint=$(dirname "$model_path")
      step_tag=$(basename "$(dirname "$checkpoint")")
      [[ "$step_tag" =~ ^[0-9]+$ ]] || continue
      step=$((10#$step_tag))
      if ! eval_result_valid "$root/eval/$variant/step$step_tag"; then
        if checkpoint_ready "$checkpoint"; then
          eval_one_resilient "$variant" "$gpu" "$step" "$checkpoint" \
            || eval_status=$?
          if (( eval_status == 75 )); then
            resource_interrupted=1
            break
          fi
          if (( eval_status != 0 )); then
            return "$eval_status"
          fi
          flock "$summary_lock" "$python" "$summarizer" \
            --root "$root" --episodes-per-task "$eval_episodes"
        else
          found_pending=1
        fi
      fi
    done < <(find "$root/train/$variant/checkpoints" -mindepth 3 -maxdepth 3 \
      -path '*/pretrained_model/model.safetensors' -type f 2>/dev/null | sort -V)

    if (( resource_interrupted != 0 )); then
      continue
    fi

    local train_pid
    train_pid=$(cat "$train_pid_file" 2>/dev/null || true)
    if [[ -n "$train_pid" ]] && ! kill -0 "$train_pid" 2>/dev/null; then
      stopped_idle_scans=$((stopped_idle_scans + 1))
      # Final checkpoints can become visible shortly after the trainer PID
      # exits on NFS. Require a quiet window before declaring the queue done.
      if (( stopped_idle_scans >= 3 )); then
        if (( found_pending == 0 )); then
          break
        fi
        echo "[eval] training ended with an incomplete checkpoint for $variant" >&2
        return 1
      fi
    else
      stopped_idle_scans=0
    fi
    sleep 30
  done
}

eval_all() {
  require_inputs
  mkdir -p "$root/logs" "$root/pids"
  local pids=()
  for i in "${!variants[@]}"; do
    eval_watch_one "${variants[$i]}" "${eval_gpus[$i]}" &
    pids+=("$!")
    echo "$!" > "$root/pids/eval_${variants[$i]}.pid"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

resource_watch() {
  "$python" "$guard" --root "$root" --watch --sample-seconds 2 --poll-seconds 2 \
    --terminate-eval-memory-gib 57 \
    >>"$root/logs/resource_watch.log" 2>&1
}

checkpoint_cache_watch() {
  "$python" "$cache_reclaimer" --train-root "$root/train" --watch \
    --poll-seconds 1 --readvise-seconds 60 \
    >>"$root/logs/checkpoint_cache_reclaimer.log" 2>&1
}

summarize() {
  cd "$repo"
  "$python" "$summarizer" --root "$root" --episodes-per-task "$eval_episodes"
}

pid_state() {
  local file=$1 pid
  pid=$(cat "$file" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid:running"
  else
    echo "${pid:-none}:stopped"
  fi
}

status() {
  for variant in "${variants[@]}"; do
    local checkpoints summaries
    checkpoints=$(find "$root/train/$variant/checkpoints" -path '*/pretrained_model/model.safetensors' -type f 2>/dev/null | wc -l || true)
    summaries=$(find "$root/eval/$variant" -name summary.json -type f 2>/dev/null | wc -l || true)
    echo "$variant checkpoints=$checkpoints evals=$summaries train=$(pid_state "$root/pids/train_${variant}.pid") eval=$(pid_state "$root/pids/eval_${variant}.pid")"
  done
  tail -1 "$root/resource/samples.jsonl" 2>/dev/null || true
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}

main() {
  case "${1:-}" in
    preflight) preflight ;;
    train) train_all ;;
    extend) extend_unstable ;;
    eval) eval_all ;;
    all)
      mkdir -p "$root/logs" "$root/pids"
      resource_watch & resource_pid=$!
      echo "$resource_pid" > "$root/pids/resource_watch.pid"
      checkpoint_cache_watch & cache_reclaimer_pid=$!
      echo "$cache_reclaimer_pid" > "$root/pids/checkpoint_cache_reclaimer.pid"
      trap 'kill "$resource_pid" "$cache_reclaimer_pid" 2>/dev/null || true' EXIT
      train_all & train_supervisor=$!
      eval_all & eval_supervisor=$!
      wait "$train_supervisor"
      wait "$eval_supervisor"
      summarize
      ;;
    summarize) summarize ;;
    status) status ;;
    *) usage; return 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
