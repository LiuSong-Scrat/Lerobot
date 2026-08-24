# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

"""Stable identifiers shared by SmolVLA config and checkpoint loaders."""


# Shape-incompatible with the original 1920-wide V3 checkpoint.  The token
# reachability graph is unchanged, but the existing Action Expert trunk is
# narrowed to the WEPVLA width and its cross-attention context projections are
# shared across layers in the same spirit as MolmoAct2.
FULL_MOLMO2ER_WEP_PREFIX_TOPOLOGY = (
    "wepvla_scene_in_vlm_prefix_v3_feature_align"
)
FULL_MOLMO2ER_EXPERT_HIDDEN_SIZE = 720
FULL_MOLMO2ER_EXPERT_INTERMEDIATE_SIZE = 2048
FULL_MOLMO2ER_EXPERT_WIDTH_MULTIPLIER = 720 / 2560
