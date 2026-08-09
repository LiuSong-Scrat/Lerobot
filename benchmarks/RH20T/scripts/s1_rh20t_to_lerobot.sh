#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
stamp="$(date +%Y%m%d_%H%M%S)"
output_root="${RH20T_OUTPUT_ROOT:-${repo_root}/lerobot/benchmarks/RH20T/lerobot_dataset/${stamp}}"
input_root="${RH20T_ROOT:-/opt/data/private/liusong/RH20T/extracted}"
camera_serial="${RH20T_CAMERA_SERIAL:-cam_037522061512}"
max_frames_arg=()
if [[ -n "${RH20T_MAX_FRAMES:-}" ]]; then
  max_frames_arg=(--max-frames "${RH20T_MAX_FRAMES}")
fi

cd "${repo_root}"
exec python3 lerobot/benchmarks/RH20T/scripts/convert_rh20t_to_lerobot.py \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --cfgs 7 \
  --camera-serial "${camera_serial}" \
  --workers "${RH20T_NUM_WORKERS:-1}" \
  "${max_frames_arg[@]}" \
  "$@"
