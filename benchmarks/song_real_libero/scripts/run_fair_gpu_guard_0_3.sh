#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GUARD_LAUNCHER="${SCRIPT_DIR}/run_gpu_idle_memory_guard.sh"
STATE_DIR="${GPU_FAIR_GUARD_STATE_DIR:-/raid5/rongshengwang/.lerobot_gpu_guards/molmo2er/gpu_fair_guard_0_3}"
STATE_PATH="${STATE_DIR}/fair_guard_status.json"
LOG_PATH="${STATE_DIR}/fair_guard.log"
LEASE_STATUS_DIR="${STATE_DIR}/lease"
LOCK_PATH="${GPU_FAIR_GUARD_LOCK_PATH:-/tmp/lerobot_fair_gpu_guard_0_3.lock}"
GRACE_SECONDS="${GPU_FAIR_GUARD_GRACE_SECONDS:-180}"
POLL_SECONDS="${GPU_FAIR_GUARD_POLL_SECONDS:-3}"
MIN_FREE_MIB="${GPU_FAIR_GUARD_MIN_FREE_MIB:-31500}"
GUARD_PID=""
IDLE_SINCE=0
LAST_HEARTBEAT=0

mkdir -p "${STATE_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "Another fair GPU 0-3 guard holds ${LOCK_PATH}." >&2
  exit 73
fi

log() {
  printf '[fair-gpu-guard] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG_PATH}"
}

gpu_snapshot() {
  nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' '$1 + 0 < 4 {gsub(/ /, "", $0); printf "%s%s", (seen++ ? ";" : ""), $0}'
}

write_state() {
  local state="$1"
  local message="$2"
  local now_epoch
  local elapsed=0
  local temporary_path="${STATE_PATH}.tmp.$$"
  now_epoch="$(date -u +%s)"
  if (( IDLE_SINCE > 0 )); then
    elapsed=$((now_epoch - IDLE_SINCE))
  fi
  jq -n \
    --arg state "${state}" \
    --arg message "${message}" \
    --arg updated_at "$(date -u +%FT%TZ)" \
    --arg snapshot "$(gpu_snapshot 2>&1 || printf unavailable)" \
    --argjson pid "$$" \
    --argjson guard_pid "${GUARD_PID:-0}" \
    --argjson grace_seconds "${GRACE_SECONDS}" \
    --argjson idle_elapsed_seconds "${elapsed}" \
    '{
      state: $state,
      message: $message,
      updated_at: $updated_at,
      pid: $pid,
      guard_pid: (if $guard_pid == 0 then null else $guard_pid end),
      grace_seconds: $grace_seconds,
      idle_elapsed_seconds: $idle_elapsed_seconds,
      gpu_snapshot: $snapshot
    }' > "${temporary_path}"
  mv -f "${temporary_path}" "${STATE_PATH}"
}

all_four_gpus_idle() {
  local gpu_index
  local free_mib
  for gpu_index in 0 1 2 3; do
    free_mib="$(
      nvidia-smi --id="${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' '
    )" || return 1
    [[ "${free_mib}" =~ ^[0-9]+$ ]] || return 1
    (( free_mib >= MIN_FREE_MIB )) || return 1
  done
}

cleanup() {
  local exit_code="$?"
  trap - EXIT INT TERM HUP
  if [[ -n "${GUARD_PID}" ]] && kill -0 "${GUARD_PID}" 2>/dev/null; then
    log "stopping only the child guard created by this coordinator: pid=${GUARD_PID}"
    kill -TERM "${GUARD_PID}" 2>/dev/null || true
    wait "${GUARD_PID}" 2>/dev/null || true
  fi
  GUARD_PID=""
  write_state stopped "fair guard stopped; owned CUDA reservations released"
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM HUP

log "policy active: wait for GPUs 0-3 to remain fully idle for ${GRACE_SECONDS}s; any use resets the grace window"
while true; do
  now_epoch="$(date -u +%s)"
  if all_four_gpus_idle; then
    if (( IDLE_SINCE == 0 )); then
      IDLE_SINCE="${now_epoch}"
      log "GPUs 0-3 became idle; starting the three-minute courtesy window; gpu=$(gpu_snapshot)"
    fi
    idle_elapsed=$((now_epoch - IDLE_SINCE))
    if (( idle_elapsed >= GRACE_SECONDS )); then
      log "courtesy window completed without another job; permanently claiming GPUs 0-3 until explicitly stopped"
      write_state launching_guard "three-minute courtesy window completed; launching persistent memory guard"
      bash "${GUARD_LAUNCHER}" \
        --gpus 0 1 2 3 \
        --min-free-mib "${MIN_FREE_MIB}" \
        --reserve-mib 30000 \
        --allocation-headroom-mib 512 \
        --chunk-mib 512 \
        --poll-seconds 3 \
        --heartbeat-seconds 180 \
        --status-dir "${LEASE_STATUS_DIR}" >> "${LOG_PATH}" 2>&1 &
      GUARD_PID="$!"
      write_state guard_running "persistent GPU 0-3 guard launched after courtesy window"
      if wait "${GUARD_PID}"; then
        guard_exit_code=0
      else
        guard_exit_code="$?"
      fi
      log "persistent child guard exited rc=${guard_exit_code}; restarting the courtesy policy"
      GUARD_PID=""
      IDLE_SINCE=0
    else
      write_state courtesy_wait "GPUs 0-3 are idle; waiting three minutes before claiming"
    fi
  else
    if (( IDLE_SINCE > 0 )); then
      log "another job appeared within the courtesy window; yielding GPUs 0-3 and resetting the timer; gpu=$(gpu_snapshot)"
    fi
    IDLE_SINCE=0
    write_state yielding "GPUs 0-3 are in use; no CUDA context or reservation created"
  fi

  if (( now_epoch - LAST_HEARTBEAT >= 180 )); then
    log "heartbeat state=$(jq -r .state "${STATE_PATH}") gpu=$(gpu_snapshot)"
    LAST_HEARTBEAT="${now_epoch}"
  fi
  sleep "${POLL_SECONDS}"
done
