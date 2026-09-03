#!/usr/bin/env bash
set -euo pipefail

cd /home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

checkpoint_root=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/molmo_wepvla_contract_18l_native64_after2w6/checkpoints/018000

exec /home/liusong/anaconda3/envs/reap/bin/python -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes=7 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=29681 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --config_path="$checkpoint_root/pretrained_model/train_config.json" \
  --resume=true
