#!/usr/bin/env python

import sys
import os
sys.path.insert(0, '/home/liusong/ProgramFiles/Huggingface/lerobot/src')

import torch
import numpy as np
from lerobot.processor.umi_processor import UMIProcessor

def test_umi_processor():
    print("Testing UMIProcessor...")

    # Create a mock batch
    batch_size = 2
    seq_len = 10
    state_dim = 9  # pose9
    action_dim = 10  # pose9 + gripper
    point_cloud_size = 1024

    batch = {
        'observation': {
            'state': torch.randn(batch_size, seq_len, state_dim),
            'point_cloud': torch.randn(batch_size, seq_len, point_cloud_size, 6)
        },
        'action': torch.randn(batch_size, seq_len, action_dim)
    }

    print(f"Input batch shapes:")
    print(f"  state: {batch['observation']['state'].shape}")
    print(f"  point_cloud: {batch['observation']['point_cloud'].shape}")
    print(f"  action: {batch['action'].shape}")

    # Create processor
    processor = UMIProcessor()
    print("UMIProcessor created successfully")

    # Process batch
    try:
        processed_batch = processor(batch)
        print("Batch processed successfully")

        print(f"Output batch shapes:")
        print(f"  state: {processed_batch['observation']['state'].shape}")
        print(f"  point_cloud: {processed_batch['observation']['point_cloud'].shape}")
        print(f"  action: {processed_batch['action'].shape}")

        print("Test passed!")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_umi_processor()