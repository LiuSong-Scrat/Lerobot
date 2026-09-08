#!/usr/bin/env bash

python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/scripts/RE_rlbench_fix_checkpoint_paths.py \
    --checkpoint-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/wep_vla_vfinal-20000+20000_5000+2_fixed_gripper_10tasks_0829/checkpoints/150000/pretrained_model \
    --vlm-model-path /home/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
    --vlm-weights-path /home/liusong/hf_models/smolvla_base