# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 32 #50
    n_action_steps: int = 16 #50
    # Some recorder formats store observation[i] after action[i] has already
    # been applied.  Their first causal control target is therefore
    # action[i + 1].  Shift action chunk lookup at the dataset boundary rather
    # than training a stale token and compensating with action_index at runtime.
    action_chunk_start_offset: int = 0

    # Shorter state and action vectors will be padded
    max_state_dim: int = 10
    max_action_dim: int = 10

    # Normalization mapping for Flow Matching (required for action normalization)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Point-cloud and RGB views are independently selectable. None preserves
    # legacy checkpoints by coupling RGB selection to point-cloud selection.
    camera_views: str = "agentview"
    rgb_camera_views: str | None = None

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    # Number of denoising steps during flow matching inference.
    # Recommended values: 8-10 (fast), 20-30 (balanced), 50+ (high-quality)
    num_steps: int = 10
    # The official large-data recipe samples continuous times from
    # Beta(1.5, 1). For small-data fitting, ``integration_grid`` instead
    # trains exactly the Euler times used by sample_actions:
    # {0, 1/num_steps, ..., (num_steps-1)/num_steps}.
    flow_time_sampling: str = "beta"
    # Optional small-data emphasis on the first Euler evaluation. At t=0 the
    # noisy action contains no ground-truth trajectory information, so this is
    # the only grid point that must infer the whole chunk from observations.
    # Zero preserves uniform integration-grid sampling.
    flow_time_zero_probability: float = 0.0

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    # Keep proprioception out of the learned prefix by default. The World-Ego
    # branch uses current EEF pose only as an analytic coordinate carrier.
    encode_robot_state: bool = False

    # Frozen-VLM image/point adapter mode. This keeps the official SmolVLM
    # vision/language backbone unchanged and frozen, while the existing point
    # encoders, point/action fusion and Action Expert remain trainable.
    vla_adapter_enable: bool = False
    vla_adapter_freeze_vlm: bool = True

    # Song point-cloud foreground/background conditioning.
    pointseg_enable: bool = False
    pointseg_checkpoint_path: str | None = None
    pointseg_backbone_type: str = "litept"
    pointseg_grid_size: float = 0.01
    pointseg_feature_dim: int = 64
    pointseg_foreground_ratio: float = 0.08
    pointseg_background_ratio: float = 0.25
    pointseg_min_foreground_points: int = 512
    pointseg_min_background_points: int = 512
    pointseg_aux_loss_weight: float = 0.20
    pointseg_use_temporal_priors_as_input: bool = False
    pointseg_use_pseudo_selection: bool = True
    point_action_fusion_enable: bool = True
    point_action_fusion_heads: int = 4
    point_action_fusion_dropout: float = 0.0

    # Relative importance of the physical pose9 + gripper action groups in the
    # standard flow-matching objective. Values are normalized to preserve the
    # overall loss scale. Defaults reproduce the original per-dimension MSE.
    action_loss_translation_weight: float = 1.0
    action_loss_rotation_weight: float = 1.0
    action_loss_gripper_weight: float = 1.0
    # Standard SmolVLA assumes normalized Euclidean action channels. Song's
    # identity-normalized pose9 instead contains a 6D rotation representation,
    # so zero-centred Gaussian vectors are not valid rotation states. This
    # option samples a valid SE(3) pose9 in both training and inference while
    # retaining the standard Euclidean flow-matching objective.
    pose9_action_noise_enable: bool = False
    pose9_action_noise_trans_scale: float = 0.15
    pose9_action_noise_rot_scale: float = 0.35
    pose9_action_noise_gripper_scale: float = 0.05

    # Joint World–Ego trajectory branch. PointSeg selects foreground XYZRGB in
    # the Ego/body frame, then the current EEF pose analytically maps those
    # exact points into World. World owns an independent LitePT + PointAction
    # front-end, but both streams share the official Action Expert. One
    # attention-pooled global scene token per stream forms the causal-prefix
    # scene block; all World/Ego action tokens form one bidirectional block.
    # The branch is active in training and inference.
    worldflow_enable: bool = False
    worldflow_feature_dim: int = 64
    worldflow_grid_size: float = 0.01
    worldflow_loss_weight: float = 0.05
    worldflow_geo_loss_weight: float = 0.05
    worldflow_bridge_loss_weight: float = 0.05
    worldflow_trans_weight: float = 1.0
    worldflow_rot_weight: float = 1.0
    worldflow_se3_head_enable: bool = False
    worldflow_equiv_loss_weight: float = 0.02
    # 0 keeps the complete predicted foreground. A positive value is an
    # optional memory cap applied after PointSeg foreground selection.
    worldflow_max_points: int = 0
    # Command-target sidecars keep the World target exactly aligned with the
    # Ego action chunk. False preserves legacy datasets that only contain
    # achieved poses; production WorldFlow recipes should enable this guard.
    worldflow_require_action_target_sidecar: bool = False
    # Legacy v0.5 CLI fields. v0.5.1+ shares the Ego Action Expert, so these
    # values are parsed for old commands but do not instantiate another expert.
    worldflow_action_expert_layers: int = -1
    worldflow_action_expert_dropout: float = 0.0
    # WorldFlow denoises an SE(3) spatial transform. Unit Gaussian twists
    # correspond to metre-scale translations and roughly 90 degree rotations,
    # which is far outside a tabletop robot's action distribution. Keep the
    # noise on the same physical scale as the reachable trajectory.
    worldflow_noise_trans_scale: float = 0.15
    worldflow_noise_rot_scale: float = 0.20
    # Noise relationship between the two action streams.
    #
    # ``independent`` reproduces the original experimental implementation:
    # Ego body motion and World spatial motion start from unrelated random
    # transforms. ``conjugate_ego`` samples one physically valid Ego SE(3)
    # prior B_0 and derives the World prior exactly as G_0 = C B_0 C^{-1},
    # where C is the current EEF-to-World pose.  The latter is the recommended
    # stochastic double-flow contract.  It requires either the valid-pose9
    # prior or the complete manifold SE(3) flow because legacy Euclidean
    # Gaussian rotation-6D vectors are not elements of SE(3) and therefore
    # cannot be conjugated safely.
    worldflow_noise_coupling: str = "independent"
    # Checkpoint-adaptation compatibility switch used by later WorldFlow
    # branches.  It is deliberately false for this point-only continuation;
    # with worldflow_enable=False no WorldFlow branch is instantiated.
    worldflow_bootstrap_from_ego: bool = False
    worldflow_augmentation_trans_scale: float = 0.20
    worldflow_augmentation_rot_scale: float = 0.75
    # Legacy Dense-ObjectFlow options retained only so old command lines and
    # configs remain parseable. The independent branch does not consume role
    # scores, predict point flow, or run Kabsch.
    worldflow_min_transport_points: int = 3
    worldflow_transport_score_threshold: float = 0.05

    # ET-SEED-style SE(3) action generation.
    se3_enable: bool = False
    # ``direct_twist`` uses dedicated randomly initialized 7D Ego / 6D World
    # heads. ``projected_pose9`` reuses the pretrained pose9 velocity heads and
    # analytically projects their instantaneous rigid-body derivative onto a
    # spatial SE(3) twist. ``pose9_endpoint`` instead preserves the old flow
    # head's endpoint semantics: x_t + (1-t) v_pose9 is projected to SE(3), then
    # converted to the exact left-trivialized geodesic velocity from x_t. All
    # modes use the same manifold-valued prior, geodesic training path and
    # group integration; only the output parameterization differs.
    se3_twist_head_mode: str = "direct_twist"
    se3_noise_trans_scale: float = 0.15
    se3_noise_rot_scale: float = 0.75
    se3_noise_gripper_scale: float = 0.05
    se3_pose_loss_weight: float = 1.0
    se3_gripper_loss_weight: float = 1.0
    se3_endpoint_loss_weight: float = 0.25
    se3_final_correction_enable: bool = False
    se3_final_correction_loss_weight: float = 0.20
    se3_equivariance_loss_weight: float = 0.02

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 100 #1_000
    scheduler_decay_steps: int = 60_000 #30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    # Keep the existing SmolVLA implementation as the default. ``molmo2_text``
    # is the historical point-cloud-only 3B control (18/36 text layers, no
    # vision). ``molmo2_full`` is the Frozen Full-Molmo2-ER WEP-VLA contract:
    # native RGB vision plus all 36 text layers, paired with a 36-layer expert.
    vlm_backend: str = "smolvlm"
    # Optional raw SmolVLM directory/repository or SmolVLA policy checkpoint
    # used only as the source of the VLM weights. `vlm_model_name` remains the
    # source of the architecture and processor when a policy checkpoint is used.
    vlm_weights_path: str | None = None
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights
    # One-shot initialization used only when make_policy creates a fresh
    # `--policy.type=smolvla` model. Full checkpoints loaded through
    # `--policy.path` already contain these tensors and are never overwritten.
    load_action_expert_weights: bool = False
    # Defaults to vlm_weights_path when omitted. The source must be a complete
    # SmolVLA policy checkpoint, not a raw SmolVLM repository.
    action_expert_weights_path: str | None = None
    # The official 32 action channels need not share semantics with Song's
    # physical pose9 + gripper channels. Keep the task-specific projections
    # random by default while reusing the transformer and timestep MLP.
    load_action_expert_projection_weights: bool = False

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.vlm_backend not in {"smolvlm", "molmo2_text", "molmo2_full"}:
            raise ValueError(
                "vlm_backend must be one of 'smolvlm', 'molmo2_text', or 'molmo2_full'; "
                f"got {self.vlm_backend!r}."
            )
        if self.vlm_backend == "molmo2_text":
            if len(self.selected_camera_views) != 1:
                raise ValueError(
                    "The registered Molmo2-ER control uses one point-cloud view and therefore exactly "
                    "one foreground token plus one background token."
                )
            if self.vla_adapter_enable:
                raise ValueError(
                    "vlm_backend='molmo2_text' is the point-cloud-only control experiment and "
                    "therefore requires vla_adapter_enable=False."
                )
            if self.add_image_special_tokens:
                raise ValueError(
                    "vlm_backend='molmo2_text' does not accept image tokens; "
                    "set add_image_special_tokens=False."
                )
            if not self.load_vlm_weights:
                raise ValueError(
                    "vlm_backend='molmo2_text' requires load_vlm_weights=True so the frozen "
                    "Molmo2-ER prefix is actually pretrained."
                )
            if not self.train_expert_only:
                raise ValueError(
                    "vlm_backend='molmo2_text' requires train_expert_only=True; the Molmo text "
                    "prefix is frozen exactly like the v0.4.3 control VLM."
                )
            if self.num_vlm_layers != 18:
                raise ValueError(
                    "The registered Molmo2-ER 3B recipe keeps the continuous first half of the "
                    f"36-layer decoder, so num_vlm_layers must be 18, got {self.num_vlm_layers}."
                )
            if self.num_expert_layers not in {-1, 18}:
                raise ValueError(
                    "The Molmo2-ER control keeps Action Expert depth aligned with the retained "
                    f"VLM depth (18); got num_expert_layers={self.num_expert_layers}."
                )
            if self.self_attn_every_n_layers != 2:
                raise ValueError(
                    "The Molmo2-ER control preserves the SmolVLA 1:1 self/cross interleave, so "
                    "self_attn_every_n_layers must be 2."
                )
            if abs(float(self.expert_width_multiplier) - 0.75) > 1e-12:
                raise ValueError(
                    "The Molmo2-ER control preserves SmolVLA's 0.75 Action Expert width multiplier."
                )
            registered_scheduler_decay_lrs = (2.5e-6, 3e-5)
            if not any(
                abs(float(self.scheduler_decay_lr) - value) <= 1e-12
                for value in registered_scheduler_decay_lrs
            ):
                raise ValueError(
                    "Molmo2-ER point-only scheduler_decay_lr must be either "
                    f"2.5e-6 (cold start) or 3e-5 (checkpoint continuation), got "
                    f"{self.scheduler_decay_lr}."
                )
            # This backend is a registered one-variable experiment, not a
            # general Molmo policy.  Fail at config parsing if any v0.4.3 WEP
            # choice drifts: only the VLM/Expert depth and width may change.
            control_contract = {
                "n_obs_steps": 1,
                "chunk_size": 32,
                "n_action_steps": 16,
                "action_chunk_start_offset": 0,
                "max_state_dim": 10,
                "max_action_dim": 10,
                "camera_views": "agentview",
                "rgb_camera_views": None,
                "empty_cameras": 0,
                "tokenizer_max_length": 48,
                "num_steps": 10,
                "flow_time_sampling": "beta",
                "flow_time_zero_probability": 0.0,
                "use_cache": True,
                "freeze_vision_encoder": True,
                "train_state_proj": True,
                "encode_robot_state": False,
                "vla_adapter_freeze_vlm": True,
                "pointseg_enable": True,
                "pointseg_checkpoint_path": None,
                "pointseg_backbone_type": "litept",
                "pointseg_grid_size": 0.01,
                "pointseg_feature_dim": 64,
                "pointseg_foreground_ratio": 0.025,
                "pointseg_background_ratio": 0.025,
                "pointseg_min_foreground_points": 2500,
                "pointseg_min_background_points": 0,
                "pointseg_aux_loss_weight": 0.0005,
                "pointseg_use_temporal_priors_as_input": False,
                "pointseg_use_pseudo_selection": False,
                "point_action_fusion_enable": True,
                "point_action_fusion_heads": 4,
                "point_action_fusion_dropout": 0.0,
                "action_loss_translation_weight": 1.0,
                "action_loss_rotation_weight": 1.0,
                "action_loss_gripper_weight": 1.0,
                "pose9_action_noise_enable": False,
                "worldflow_enable": False,
                "worldflow_se3_head_enable": False,
                "worldflow_bootstrap_from_ego": False,
                "se3_enable": False,
                "se3_final_correction_enable": False,
                "optimizer_lr": 1e-4,
                "optimizer_betas": (0.9, 0.95),
                "optimizer_eps": 1e-8,
                "optimizer_weight_decay": 1e-10,
                "optimizer_grad_clip_norm": 10,
                "scheduler_warmup_steps": 100,
                "scheduler_decay_steps": 30_000,
                "load_action_expert_weights": False,
                "load_action_expert_projection_weights": False,
                "attention_mode": "cross_attn",
                "prefix_length": -1,
                "pad_language_to": "longest",
                "compile_model": False,
                "use_amp": False,
                "use_peft": False,
            }

            def control_value_matches(expected, actual):
                if isinstance(expected, tuple):
                    return tuple(actual) == expected
                return actual == expected

            drift = {
                name: (expected, getattr(self, name))
                for name, expected in control_contract.items()
                if not control_value_matches(expected, getattr(self, name))
            }
            if drift:
                details = ", ".join(
                    f"{name}: expected={expected!r}, actual={actual!r}"
                    for name, (expected, actual) in drift.items()
                )
                raise ValueError(
                    "Molmo2-ER point-only 3B is a locked v0.4.3 control; only VLM/Expert "
                    f"depth and width may differ ({details})."
                )
        elif self.vlm_backend == "molmo2_full":
            if self.selected_camera_views != ("agentview",):
                raise ValueError(
                    "vlm_backend='molmo2_full' requires exactly camera_views='agentview'."
                )
            if self.selected_rgb_camera_views != ("agentview",):
                raise ValueError(
                    "vlm_backend='molmo2_full' requires rgb_camera_views='agentview' so native RGB "
                    "cannot be silently omitted."
                )
            if self.vla_adapter_enable:
                raise ValueError(
                    "vlm_backend='molmo2_full' uses Molmo's native multimodal path and requires "
                    "vla_adapter_enable=False."
                )
            if self.add_image_special_tokens:
                raise ValueError(
                    "Molmo native processing owns the complete image token template; "
                    "add_image_special_tokens must be False."
                )
            if not self.load_vlm_weights:
                raise ValueError("vlm_backend='molmo2_full' requires pretrained Molmo2-ER weights.")
            if not self.train_expert_only:
                raise ValueError("vlm_backend='molmo2_full' requires a completely frozen Molmo backbone.")
            if not self.freeze_vision_encoder:
                raise ValueError("vlm_backend='molmo2_full' requires freeze_vision_encoder=True.")
            if self.num_vlm_layers != 36:
                raise ValueError(
                    "The Full-Molmo2-ER contract uses all 36 VLM layers; "
                    f"got num_vlm_layers={self.num_vlm_layers}."
                )
            if self.num_expert_layers not in {-1, 36}:
                raise ValueError(
                    "The Full-Molmo2-ER Action Expert must align 1:1 with all 36 VLM layers; "
                    f"got num_expert_layers={self.num_expert_layers}."
                )
            if self.self_attn_every_n_layers != 2:
                raise ValueError("Full-Molmo2-ER requires alternating even-SA/odd-CA Expert layers.")
            if abs(float(self.expert_width_multiplier) - 0.75) > 1e-12:
                raise ValueError("Full-Molmo2-ER requires expert_width_multiplier=0.75.")
            if not any(
                abs(float(self.scheduler_decay_lr) - value) <= 1e-12
                for value in (2.5e-6, 3e-5)
            ):
                raise ValueError(
                    "Full-Molmo2-ER scheduler_decay_lr must be 2.5e-6 for the fresh 36k stage "
                    "or 3e-5 for a 30k warm-start fine-tuning stage."
                )

            full_contract = {
                "n_obs_steps": 1,
                "chunk_size": 32,
                "n_action_steps": 16,
                "action_chunk_start_offset": 0,
                "max_state_dim": 10,
                "max_action_dim": 10,
                "camera_views": "agentview",
                "rgb_camera_views": "agentview",
                "empty_cameras": 0,
                "tokenizer_max_length": 48,
                "num_steps": 10,
                "flow_time_sampling": "beta",
                "flow_time_zero_probability": 0.0,
                "use_cache": True,
                "freeze_vision_encoder": True,
                "train_state_proj": True,
                "encode_robot_state": False,
                "vla_adapter_freeze_vlm": True,
                "pointseg_enable": True,
                "pointseg_checkpoint_path": None,
                "pointseg_backbone_type": "litept",
                "pointseg_grid_size": 0.01,
                "pointseg_feature_dim": 64,
                "pointseg_foreground_ratio": 0.025,
                "pointseg_background_ratio": 0.025,
                "pointseg_min_foreground_points": 2500,
                "pointseg_min_background_points": 0,
                "pointseg_aux_loss_weight": 0.0005,
                "pointseg_use_temporal_priors_as_input": False,
                "pointseg_use_pseudo_selection": False,
                "point_action_fusion_enable": True,
                "point_action_fusion_heads": 4,
                "point_action_fusion_dropout": 0.0,
                "action_loss_translation_weight": 1.0,
                "action_loss_rotation_weight": 1.0,
                "action_loss_gripper_weight": 1.0,
                "pose9_action_noise_enable": False,
                "worldflow_enable": False,
                "worldflow_se3_head_enable": False,
                "worldflow_bootstrap_from_ego": False,
                "se3_enable": False,
                "se3_final_correction_enable": False,
                "optimizer_lr": 1e-4,
                "optimizer_betas": (0.9, 0.95),
                "optimizer_eps": 1e-8,
                "optimizer_weight_decay": 1e-10,
                "optimizer_grad_clip_norm": 10,
                "scheduler_warmup_steps": 100,
                "scheduler_decay_steps": 30_000,
                "load_action_expert_weights": False,
                "load_action_expert_projection_weights": False,
                "attention_mode": "cross_attn",
                "prefix_length": -1,
                "pad_language_to": "longest",
                "compile_model": False,
                "use_amp": False,
                "use_peft": False,
            }
            drift = {
                name: (expected, getattr(self, name))
                for name, expected in full_contract.items()
                if not (
                    tuple(getattr(self, name)) == expected
                    if isinstance(expected, tuple)
                    else getattr(self, name) == expected
                )
            }
            if drift:
                details = ", ".join(
                    f"{name}: expected={expected!r}, actual={actual!r}"
                    for name, (expected, actual) in drift.items()
                )
                raise ValueError(f"Full-Molmo2-ER WEP-VLA contract drifted ({details}).")
        if int(self.action_chunk_start_offset) < 0:
            raise ValueError("action_chunk_start_offset must be non-negative.")
        if int(self.action_chunk_start_offset) > 0:
            warnings.warn(
                "ACTION TEMPORAL CONTRACT: action_chunk_start_offset="
                f"{int(self.action_chunk_start_offset)} means predicted token 0 is trained from "
                f"dataset action[i+{int(self.action_chunk_start_offset)}]. Online inference must "
                "execute predicted token 0 (action_index=0); do not apply the offset a second time.",
                stacklevel=2,
            )
        if self.flow_time_sampling not in {"beta", "uniform", "integration_grid"}:
            raise ValueError(
                "flow_time_sampling must be one of beta, uniform, integration_grid; "
                f"got {self.flow_time_sampling!r}."
            )
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if not 0.0 <= float(self.flow_time_zero_probability) < 1.0:
            raise ValueError("flow_time_zero_probability must be in [0, 1).")
        if self.flow_time_zero_probability > 0 and self.flow_time_sampling != "integration_grid":
            raise ValueError(
                "flow_time_zero_probability is only supported with "
                "flow_time_sampling='integration_grid'."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.se3_enable:
            action_norm = self.normalization_mapping.get("ACTION")
            if action_norm is not NormalizationMode.IDENTITY:
                raise ValueError("se3_enable=True requires ACTION normalization to be IDENTITY.")
            if self.max_action_dim < 10:
                raise ValueError("se3_enable=True requires max_action_dim >= 10.")
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("se3_enable=True is not supported with RTC enabled in v1.")
            if self.se3_twist_head_mode not in {
                "direct_twist",
                "projected_pose9",
                "pose9_endpoint",
            }:
                raise ValueError(
                    "se3_twist_head_mode must be 'direct_twist', 'projected_pose9', "
                    "or 'pose9_endpoint'; "
                    f"got {self.se3_twist_head_mode!r}."
                )
            for name in (
                "se3_noise_trans_scale",
                "se3_noise_rot_scale",
                "se3_noise_gripper_scale",
            ):
                if float(getattr(self, name)) < 0:
                    raise ValueError(f"{name} must be non-negative.")
        if self.se3_enable and self.pose9_action_noise_enable:
            raise ValueError(
                "se3_enable and pose9_action_noise_enable are mutually exclusive: "
                "se3_enable already supplies the complete manifold-valued SE(3) prior and flow."
            )
        for name in (
            "action_loss_translation_weight",
            "action_loss_rotation_weight",
            "action_loss_gripper_weight",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.pose9_action_noise_enable:
            action_norm = self.normalization_mapping.get("ACTION")
            if action_norm is not NormalizationMode.IDENTITY:
                raise ValueError(
                    "pose9_action_noise_enable=True requires ACTION normalization to be IDENTITY."
                )
            if self.max_action_dim < 10:
                raise ValueError("pose9_action_noise_enable=True requires max_action_dim >= 10.")
            for name in (
                "pose9_action_noise_trans_scale",
                "pose9_action_noise_rot_scale",
                "pose9_action_noise_gripper_scale",
            ):
                if float(getattr(self, name)) < 0:
                    raise ValueError(f"{name} must be non-negative.")
        if self.worldflow_se3_head_enable:
            warnings.warn(
                "worldflow_se3_head_enable is kept only for CLI compatibility and is ignored. "
                "WorldFlow directly flow-matches an SE(3) spatial transform through its "
                "LitePT/PointAction front-end and the shared World–Ego Action Expert.",
                stacklevel=2,
            )
        if self.worldflow_enable:
            action_norm = self.normalization_mapping.get("ACTION")
            if action_norm is not NormalizationMode.IDENTITY:
                raise ValueError(
                    "worldflow_enable=True requires ACTION normalization to be IDENTITY because "
                    "the World-Ego bridge interprets Ego pose9 predictions as physical SE(3) transforms."
                )
            if self.worldflow_feature_dim <= 0:
                raise ValueError("worldflow_feature_dim must be positive.")
            if not self.point_action_fusion_enable:
                raise ValueError(
                    "worldflow_enable=True requires point_action_fusion_enable=True so both "
                    "coordinate streams provide point-fused action tokens to the shared expert."
                )
            if self.worldflow_max_points < 0:
                raise ValueError("worldflow_max_points must be non-negative.")
            if self.worldflow_noise_trans_scale < 0:
                raise ValueError("worldflow_noise_trans_scale must be non-negative.")
            if self.worldflow_noise_rot_scale < 0:
                raise ValueError("worldflow_noise_rot_scale must be non-negative.")
            if self.worldflow_noise_coupling not in {"independent", "conjugate_ego"}:
                raise ValueError(
                    "worldflow_noise_coupling must be 'independent' or 'conjugate_ego'; "
                    f"got {self.worldflow_noise_coupling!r}."
                )
            if self.worldflow_noise_coupling == "conjugate_ego" and not (
                self.pose9_action_noise_enable or self.se3_enable
            ):
                raise ValueError(
                    "worldflow_noise_coupling='conjugate_ego' requires "
                    "pose9_action_noise_enable=True or se3_enable=True so the Ego prior is a valid "
                    "SE(3) transform."
                )
            if self.worldflow_noise_coupling == "conjugate_ego" and (
                self.worldflow_noise_trans_scale != 0.15 or self.worldflow_noise_rot_scale != 0.20
            ):
                warnings.warn(
                    "worldflow_noise_trans_scale/rot_scale are ignored when "
                    "worldflow_noise_coupling='conjugate_ego'; the World prior is derived from "
                    "the Ego prior and current pose by exact conjugation.",
                    stacklevel=2,
                )
            ego_random_prior = (
                any(
                    float(getattr(self, name)) > 0.0
                    for name in (
                        "se3_noise_trans_scale",
                        "se3_noise_rot_scale",
                        "se3_noise_gripper_scale",
                    )
                )
                if self.se3_enable
                else self.pose9_action_noise_enable
                and any(
                    float(getattr(self, name)) > 0.0
                    for name in (
                        "pose9_action_noise_trans_scale",
                        "pose9_action_noise_rot_scale",
                        "pose9_action_noise_gripper_scale",
                    )
                )
            )
            if self.worldflow_noise_coupling == "independent":
                detail = (
                    " Both priors are valid poses, but their random origins still describe different "
                    "physical trajectories."
                    if ego_random_prior
                    else " The legacy Ego prior is also an unconstrained rotation-6D vector rather than "
                    "an SE(3) pose."
                )
                warnings.warn(
                    "WorldFlow and Ego use independent random priors. This preserves legacy behavior "
                    "but does not satisfy G_0=C B_0 C^{-1}." + detail + " Use "
                    "pose9_action_noise_enable=True (or se3_enable=True) together with "
                    "worldflow_noise_coupling='conjugate_ego' for geometrically coupled double-flow training.",
                    stacklevel=2,
                )
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("worldflow_enable=True is not compatible with RTC.")
            if self.worldflow_action_expert_layers != -1 or self.worldflow_action_expert_dropout != 0.0:
                warnings.warn(
                    "worldflow_action_expert_layers/dropout are legacy v0.5 options and are ignored "
                    "because World and Ego now share one Action Expert.",
                    stacklevel=2,
                )

        if self.se3_final_correction_enable:
            warnings.warn(
                "se3_final_correction_enable is kept only for CLI compatibility and is ignored. "
                "The PCA-style final correction branch has been removed from the active policy path.",
                stacklevel=2,
            )
        if self.vla_adapter_enable and not self.load_vlm_weights:
            warnings.warn(
                "vla_adapter_enable=True requires pretrained VLM weights; "
                "overriding load_vlm_weights=True.",
                stacklevel=2,
            )
            self.load_vlm_weights = True
        if self.load_action_expert_weights:
            action_expert_source = self.action_expert_weights_path or self.vlm_weights_path
            if action_expert_source is None or str(action_expert_source).strip().lower() in {
                "",
                "0",
                "false",
                "none",
                "off",
            }:
                raise ValueError(
                    "load_action_expert_weights=True requires action_expert_weights_path "
                    "or vlm_weights_path pointing to a complete SmolVLA policy checkpoint."
                )
        if self.vla_adapter_enable and self.vla_adapter_freeze_vlm:
            if not self.train_expert_only:
                warnings.warn(
                    "vla_adapter_enable=True with vla_adapter_freeze_vlm=True freezes the VLM; "
                    "overriding train_expert_only=True.",
                    stacklevel=2,
                )
                self.train_expert_only = True
            self.freeze_vision_encoder = True

    def flow_contract_summary(self) -> str:
        """Describe the final resolved temporal and flow-origin contract."""

        if self.se3_enable:
            scales = (
                float(self.se3_noise_trans_scale),
                float(self.se3_noise_rot_scale),
                float(self.se3_noise_gripper_scale),
            )
            origin = "se3_identity_deterministic" if scales == (0.0, 0.0, 0.0) else "se3_manifold_random"
            origin = f"{origin}(trans_m={scales[0]},rot_rad={scales[1]},gripper_m={scales[2]})"
            flow = f"se3_geodesic_left_trivialized(head={self.se3_twist_head_mode})"
        elif self.pose9_action_noise_enable:
            scales = (
                float(self.pose9_action_noise_trans_scale),
                float(self.pose9_action_noise_rot_scale),
                float(self.pose9_action_noise_gripper_scale),
            )
            origin = "pose9_identity_deterministic" if scales == (0.0, 0.0, 0.0) else "pose9_valid_se3_random"
            origin = f"{origin}(trans_m={scales[0]},rot_rad={scales[1]},gripper_m={scales[2]})"
            flow = "pose9_euclidean"
        else:
            origin = "v0.4.2_raw_channel_gaussian(std=0.1)"
            flow = "channel_euclidean"
        if self.worldflow_enable:
            target_contract = (
                "commanded_required"
                if self.worldflow_require_action_target_sidecar
                else "legacy_fallback_allowed"
            )
            world = (
                f",worldflow={self.worldflow_noise_coupling},"
                f"worldflow_targets={target_contract}"
            )
        else:
            world = ",worldflow=disabled"
        return (
            f"action_chunk_start_offset={int(self.action_chunk_start_offset)},"
            f"online_action_index=0,ego_origin={origin},ego_flow={flow}{world},"
            f"num_steps={int(self.num_steps)}"
        )

    def validate_features(self) -> None:
        selected = set(self.selected_rgb_camera_views)
        for key in list(self.input_features):
            if not key.startswith(f"{OBS_IMAGES}."):
                continue
            camera = key[len(OBS_IMAGES) + 1 :]
            if camera in {"agentview", "robot0_eye_in_hand"} and camera not in selected:
                del self.input_features[key]

        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    @staticmethod
    def _parse_camera_views(value, *, field_name: str) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            parts = [str(part).strip() for part in value]
        else:
            text = str(value).strip().strip("[]")
            parts = [part.strip().strip("\"'") for part in text.split(",")]
        views = tuple(part for part in parts if part) or ("agentview",)
        supported = {"agentview", "robot0_eye_in_hand"}
        unknown = [view for view in views if view not in supported]
        if unknown:
            raise ValueError(
                f"Unsupported {field_name} view(s) {unknown}; supported views are {sorted(supported)}."
            )
        if len(set(views)) != len(views):
            raise ValueError(f"{field_name} contains duplicates: {views}.")
        return views

    @property
    def selected_camera_views(self) -> tuple[str, ...]:
        return self._parse_camera_views(self.camera_views, field_name="camera_views")

    @property
    def selected_rgb_camera_views(self) -> tuple[str, ...]:
        if self.vlm_backend == "molmo2_text":
            return ()
        value = self.camera_views if self.rgb_camera_views is None else self.rgb_camera_views
        return self._parse_camera_views(value, field_name="rgb_camera_views")

    @property
    def requires_rgb(self) -> bool:
        """Whether every policy call must include the configured RGB observation."""

        return bool(self.vla_adapter_enable or self.vlm_backend == "molmo2_full")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        start = int(self.action_chunk_start_offset)
        return list(range(start, start + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
