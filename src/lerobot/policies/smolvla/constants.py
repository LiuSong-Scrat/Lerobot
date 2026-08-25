# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

"""Stable identifiers shared by SmolVLA config and checkpoint loaders."""


# Shape-incompatible with the original 1920-wide V3 checkpoint and
# semantics-incompatible with the earlier globally-bidirectional feature-align
# prefix. Native Molmo IMAGE/LANGUAGE queries keep the official causal-text +
# bidirectional-image mask and their original RoPE positions. Trainable scene
# readouts may read the complete native stream, but native tokens never read a
# scene token; actions read both domains from the Expert suffix.
FULL_MOLMO2ER_NATIVE_READOUT_TOPOLOGY = "v3_feature_align_language_casual"
# Historical import-compatible name. Its value is intentionally the new marker
# so checkpoints from the globally-bidirectional prefix cannot load silently.
FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY = FULL_MOLMO2ER_NATIVE_READOUT_TOPOLOGY
FULL_MOLMO2ER_EXPERT_HIDDEN_SIZE = 720
FULL_MOLMO2ER_EXPERT_INTERMEDIATE_SIZE = 2048
FULL_MOLMO2ER_EXPERT_WIDTH_MULTIPLIER = 720 / 2560
