#!/usr/bin/env bash

set -euo pipefail

readonly REPO_DIR="/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song"
readonly PYTHON_BIN="/home/liusong/anaconda3/envs/reap/bin/python"
readonly CONFIG_PATH="/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_doubleflow/molmo2_er_after1500_after4500/checkpoints/014000/pretrained_model/train_config.json"

export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes=7 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=29671 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --config_path="${CONFIG_PATH}" \
  --resume=true
