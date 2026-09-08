#!/usr/bin/env bash
set -euo pipefail

# Build the PointSeg cache for a selected subset of episodes from the existing
# multi-task RLBench LeRobot dataset. Keep the original RLBench cache entrypoint
# as the default so preview PLY files retain the original heatmap colors.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/liusong/miniconda3/envs/rlbench/bin/python}"
CACHE_SCRIPT="${CACHE_SCRIPT:-${SCRIPT_DIR}/RE_rlbench_cache_pointseg_samples.py}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/benchmarks/RLBench/datasets/box/rlbench_box_tasks_100traj_lerobot_raw_expert_target_20260810_173629}"
POINT_CLOUD_DIR="${POINT_CLOUD_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmarks/RLBench/datasets/box/rlbench_box_tasks_100traj_lerobot_raw_expert_target_20260810_173629_pointseg_cache_new}"
TASKS_RAW="${TASKS:-}"
PARAM_SET="${PARAM_SET:-raw}"
ALL_TASKS=false

CACHE_EXTRA_ARGS=()
usage() {
  cat <<'EOF'
Usage:
  bash s1.2_RE_rlbench_cache_pointseg_tasks.sh [options]

Select tasks with canonical names joined by commas or spaces:
  --tasks water_plants,sweep_to_dustpan
  TASKS="water_plants sweep_to_dustpan" bash s1.2_RE_rlbench_cache_pointseg_tasks.sh

Options:
  --dataset-root PATH       Existing 10-task LeRobot dataset.
  --point-cloud-dir PATH    Point-cloud directory; defaults to DATASET_ROOT/point_clouds.
  --output-dir PATH         Cache output; defaults to DATASET_ROOT_pointseg_cache_PARAM_SET.
  --tasks TASKS             Task names, comma-separated or space-separated.
  --param-set NAME          Foreground parameter set: 27e (default) or raw.
  --all-tasks               Explicitly cache all tasks. This is the default.
  --python PATH             Python interpreter used for the cache job.
  --overwrite               Replace a non-empty output directory.
  --help                    Show this help.

The remaining cache-specific arguments can be passed after '--', for example:
  ... --tasks water_plants -- --batch-size 4 --num-workers 8 --device cuda

The default foreground parameters are the 27e configuration. Each can be
overridden with its matching environment variable, for example:
  CONTACT_RADIUS=0.18 TASKS=water_plants bash s1.2_RE_rlbench_cache_pointseg_tasks.sh
EOF
}

while (($# > 0)); do
  case "$1" in
    --dataset-root)
      [[ $# -ge 2 ]] || { echo "--dataset-root requires a value" >&2; exit 2; }
      DATASET_ROOT="$2"
      shift 2
      ;;
    --point-cloud-dir)
      [[ $# -ge 2 ]] || { echo "--point-cloud-dir requires a value" >&2; exit 2; }
      POINT_CLOUD_DIR="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --tasks)
      [[ $# -ge 2 ]] || { echo "--tasks requires a value" >&2; exit 2; }
      TASKS_RAW="$2"
      shift 2
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TASKS_RAW+=" $1"
        shift
      done
      ;;
    --param-set)
      [[ $# -ge 2 ]] || { echo "--param-set requires 27e or raw" >&2; exit 2; }
      PARAM_SET="$2"
      shift 2
      ;;
    --all-tasks)
      ALL_TASKS=true
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      PYTHON="$2"
      shift 2
      ;;
    --overwrite)
      CACHE_EXTRA_ARGS+=(--overwrite)
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      CACHE_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown wrapper argument: $1. Put cache-specific arguments after --." >&2
      echo "Use --help for usage." >&2
      exit 2
      ;;
  esac
done

case "${PARAM_SET}" in
  27e)
    MOTION_ROTATION_RADIUS="${MOTION_ROTATION_RADIUS:-0.08}"
    MOTION_BASELINE_THRESHOLD="${MOTION_BASELINE_THRESHOLD:-0.015}"
    MOTION_BASELINE_TEMPERATURE="${MOTION_BASELINE_TEMPERATURE:-0.005}"
    MOTION_RELATIVE_MARGIN="${MOTION_RELATIVE_MARGIN:-0.10}"
    MOTION_RELATIVE_TAU="${MOTION_RELATIVE_TAU:-0.10}"
    TRAJECTORY_SIGMA="${TRAJECTORY_SIGMA:-0.13}"
    CONTACT_RADIUS="${CONTACT_RADIUS:-0.22}"
    CONTACT_TEMPERATURE="${CONTACT_TEMPERATURE:-0.02}"
    APPROACH_MARGIN="${APPROACH_MARGIN:-0.005}"
    APPROACH_TAU="${APPROACH_TAU:-0.025}"
    BACKGROUND_TRAJECTORY_SIGMA="${BACKGROUND_TRAJECTORY_SIGMA:-0.20}"
    ;;
  raw)
    MOTION_ROTATION_RADIUS="${MOTION_ROTATION_RADIUS:-0.18}"
    MOTION_BASELINE_THRESHOLD="${MOTION_BASELINE_THRESHOLD:-0.010}"
    MOTION_BASELINE_TEMPERATURE="${MOTION_BASELINE_TEMPERATURE:-0.006}"
    MOTION_RELATIVE_MARGIN="${MOTION_RELATIVE_MARGIN:-0.05}"
    MOTION_RELATIVE_TAU="${MOTION_RELATIVE_TAU:-0.08}"
    TRAJECTORY_SIGMA="${TRAJECTORY_SIGMA:-0.22}"
    CONTACT_RADIUS="${CONTACT_RADIUS:-0.22}"
    CONTACT_TEMPERATURE="${CONTACT_TEMPERATURE:-0.055}"
    APPROACH_MARGIN="${APPROACH_MARGIN:-0.0}"
    APPROACH_TAU="${APPROACH_TAU:-0.04}"
    BACKGROUND_TRAJECTORY_SIGMA="${BACKGROUND_TRAJECTORY_SIGMA:-0.32}"
    ;;
  *)
    echo "PARAM_SET must be 27e or raw, got: ${PARAM_SET}" >&2
    exit 2
    ;;
esac

if [[ -n "${TASKS_RAW}" && "${ALL_TASKS}" == "true" ]]; then
  echo "Use either --tasks/TASKS or --all-tasks, not both." >&2
  exit 2
fi

if [[ -z "${POINT_CLOUD_DIR}" ]]; then
  POINT_CLOUD_DIR="${DATASET_ROOT}/point_clouds"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${DATASET_ROOT}_pointseg_cache_${PARAM_SET}"
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi

# Standalone cache generation must not silently consume a legacy gripper
# cloud. The main collector performs the same check through its v4 signature.
"${PYTHON}" "${SCRIPT_DIR}/RE_rlbench_validate_reap_dataset.py" "${DATASET_ROOT}"

SELECTED_TASKS=()
if [[ -n "${TASKS_RAW}" ]]; then
  TASKS_NORMALIZED="${TASKS_RAW//,/ }"
  read -r -a SELECTED_TASKS <<< "${TASKS_NORMALIZED}"
  if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then
    echo "No task was provided in TASKS/--tasks." >&2
    exit 2
  fi
fi

EPISODE_INDICES=""
if [[ ${#SELECTED_TASKS[@]} -gt 0 ]]; then
  EPISODE_INDICES="$(${PYTHON} - "${DATASET_ROOT}" "${SELECTED_TASKS[@]}" <<'PY'
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq


def normalize(value: str) -> str:
    value = value.strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


# The LeRobot metadata stores RLBench task descriptions with spaces and, for
# two tasks, a slightly different wording from the shell-friendly task IDs.
aliases = {
    "close_box": ("close box",),
    "close_fridge": ("close fridge",),
    "close_laptop_lid": ("close laptop lid",),
    "phone_on_base": ("phone on base", "put the phone on the base"),
    "stack_wine": ("stack wine",),
    "sweep_to_dustpan": ("sweep dirt to dustpan", "sweep to dustpan"),
    "take_frame_off_hanger": ("take frame off hanger",),
    "take_umbrella_out_of_umbrella_stand": (
        "take umbrella out of umbrella stand",
    ),
    "toilet_seat_down": ("toilet seat down",),
    "water_plants": ("water plant", "water plants"),
}

canonical_by_alias = {}
for canonical, values in aliases.items():
    for value in (canonical, *values):
        canonical_by_alias[normalize(value)] = canonical

requested = []
for raw in sys.argv[2:]:
    key = normalize(raw)
    canonical = canonical_by_alias.get(key)
    if canonical is None:
        valid = ", ".join(sorted(aliases))
        raise SystemExit(f"Unknown task '{raw}'. Valid task IDs: {valid}")
    if canonical not in requested:
        requested.append(canonical)

root = Path(sys.argv[1])
episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
if not episode_paths:
    raise SystemExit(f"No episode metadata parquet files found under {root / 'meta' / 'episodes'}")

columns = ["episode_index", "tasks", "dataset_from_index", "dataset_to_index"]
episodes = []
for episode_path in episode_paths:
    episodes.extend(pq.read_table(episode_path, columns=columns).to_pylist())
selected = []
counts = {task: 0 for task in requested}
for episode in episodes:
    metadata_tasks = {normalize(str(task)) for task in (episode.get("tasks") or [])}
    matched = next(
        (task for task in requested if metadata_tasks.intersection(normalize(alias) for alias in aliases[task])),
        None,
    )
    if matched is None:
        continue
    selected.append(int(episode["episode_index"]))
    counts[matched] += 1

missing = [task for task, count in counts.items() if count == 0]
if missing:
    raise SystemExit(f"Requested task(s) have no episodes in {root}: {', '.join(missing)}")

print(",".join(str(index) for index in selected))
print(
    "[task-select] "
    + " ".join(f"{task}={counts[task]} episodes" for task in requested),
    file=sys.stderr,
)
print(f"[task-select] total episodes={len(selected)}", file=sys.stderr)
PY
  )"
fi

CACHE_ARGS=(
  --dataset.repo_id "${DATASET_ROOT}"
  --dataset.root "${DATASET_ROOT}"
  --point-cloud-dir "${POINT_CLOUD_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --current-points "${CURRENT_POINTS:-10000}"
  --future-points "${FUTURE_POINTS:-10000}"
  --batch-size "${CACHE_BATCH_SIZE:-16}"
  --num-workers "${CACHE_NUM_WORKERS:-8}"
  --shard-size "${CACHE_SHARD_SIZE:-256}"
  --device "${CACHE_DEVICE:-cuda}"
  --seed "${CACHE_SEED:-1000}"
  --vis-one-episode-per-task
  --motion-rotation-radius "${MOTION_ROTATION_RADIUS}"
  --motion-baseline-threshold "${MOTION_BASELINE_THRESHOLD}"
  --motion-baseline-temperature "${MOTION_BASELINE_TEMPERATURE}"
  --motion-relative-margin "${MOTION_RELATIVE_MARGIN}"
  --motion-relative-tau "${MOTION_RELATIVE_TAU}"
  --trajectory-sigma "${TRAJECTORY_SIGMA}"
  --contact-radius "${CONTACT_RADIUS}"
  --contact-temperature "${CONTACT_TEMPERATURE}"
  --approach-margin "${APPROACH_MARGIN}"
  --approach-tau "${APPROACH_TAU}"
  --background-trajectory-sigma "${BACKGROUND_TRAJECTORY_SIGMA}"
)
if [[ ${#SELECTED_TASKS[@]} -gt 0 ]]; then
  CACHE_ARGS+=(--episode-indices "${EPISODE_INDICES}")
else
  echo "[task-select] all tasks; --episode-indices omitted" >&2
fi

if [[ ${#CACHE_EXTRA_ARGS[@]} -gt 0 ]]; then
  CACHE_ARGS+=("${CACHE_EXTRA_ARGS[@]}")
fi

echo "[cache-start] dataset=${DATASET_ROOT}" >&2
echo "[param-set] ${PARAM_SET}" >&2
echo "[cache-start] output=${OUTPUT_DIR}" >&2
echo "[cache-start] point_cloud_dir=${POINT_CLOUD_DIR}" >&2
echo "[cache-start] script=${CACHE_SCRIPT}" >&2
exec "${PYTHON}" "${CACHE_SCRIPT}" "${CACHE_ARGS[@]}"
