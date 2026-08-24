#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.molmo2_processing import (
    MOLMO2_EXPECTED_IMAGE_POSITIONS,
    OBS_MOLMO_IMAGE_GRIDS,
    OBS_MOLMO_IMAGE_NUM_CROPS,
    OBS_MOLMO_IMAGE_TOKEN_POOLING,
    OBS_MOLMO_PIXEL_VALUES,
    OBS_MOLMO_TOKEN_TYPE_IDS,
    load_local_molmo2_processor,
    prepare_molmo2_multimodal_batch,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UMIProcessor,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def tokenize_molmo2_text(
    tokenizer: Any,
    text: str | list[str],
    *,
    max_length: int,
    padding: str,
    padding_side: str = "right",
    truncation: bool = True,
) -> dict[str, torch.Tensor]:
    """Insert Molmo's native BOS exactly once within the existing length budget."""

    bos_token = tokenizer.bos_token or tokenizer.eos_token
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token is None or bos_token_id is None:
        raise ValueError("Molmo2-ER tokenizer must define a BOS or EOS token.")

    texts = [text] if isinstance(text, str) else list(text)
    # Molmo2Processor inserts BOS after tokenization.  Prefixing the
    # registered special-token string is equivalent for text-only prompts,
    # includes BOS in max_length, and keeps right padding at the end.
    texts = [value if value.startswith(bos_token) else f"{bos_token}{value}" for value in texts]
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        padding_side=padding_side,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"].to(dtype=torch.bool)
    if input_ids.shape[1] > max_length:
        raise RuntimeError(
            f"Molmo token sequence exceeded tokenizer_max_length={max_length}: "
            f"shape={tuple(input_ids.shape)}."
        )
    first_valid = attention_mask.to(dtype=torch.int64).argmax(dim=1)
    first_tokens = input_ids.gather(1, first_valid[:, None]).squeeze(1)
    if not torch.all(first_tokens == int(bos_token_id)):
        raise RuntimeError("Molmo tokenizer failed to insert exactly one leading BOS token.")
    tokenized["attention_mask"] = attention_mask
    return tokenized


@dataclass
@ProcessorStepRegistry.register(name="molmo2_bos_tokenizer_processor")
class Molmo2BOSTokenizerProcessorStep(TokenizerProcessorStep):
    """Tokenize text with Molmo's native leading BOS while keeping the 48-token cap."""

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        return tokenize_molmo2_text(
            self.input_tokenizer,
            text,
            max_length=self.max_length,
            padding=self.padding,
            padding_side=self.padding_side,
            truncation=self.truncation,
        )


@dataclass
@ProcessorStepRegistry.register(name="molmo2_full_multimodal_processor")
class Molmo2FullMultimodalProcessorStep(ObservationProcessorStep):
    """Run Molmo2-ER's native one-image chat processor on CPU.

    The processor itself is deliberately excluded from serialized config. A
    checkpoint stores only its local source directory and reconstructs the
    trusted slow ``AutoProcessor`` when the pipeline is loaded.
    """

    processor_name: str | None = None
    processor: Any | None = None
    image_key: str = "observation.images.agentview"
    task_key: str = "task"
    max_text_length: int = 48
    use_fast_image_path: bool = True

    input_processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < int(self.max_text_length) <= 48:
            raise ValueError(
                "Full Molmo2-ER limits only natural-language content to at most 48 tokens; "
                f"got max_text_length={self.max_text_length}."
            )
        self.max_text_length = int(self.max_text_length)

        if self.processor is not None:
            self.input_processor = self.processor
            if getattr(self.input_processor, "tokenizer", None) is None:
                raise ValueError("The supplied Molmo processor does not expose a tokenizer.")
            self.input_processor.tokenizer.padding_side = "right"
        elif self.processor_name is not None:
            self.processor_name = str(self.processor_name)
            self.input_processor = load_local_molmo2_processor(self.processor_name)
        else:
            raise ValueError(
                "Either processor_name or a pre-initialized processor must be provided for Molmo2-ER."
            )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.image_key not in observation:
            raise KeyError(
                f"Full Molmo2-ER requires RGB observation {self.image_key!r}; "
                f"available keys are {sorted(observation)}."
            )
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or self.task_key not in complementary_data:
            raise KeyError(f"Full Molmo2-ER requires complementary task key {self.task_key!r}.")

        native = prepare_molmo2_multimodal_batch(
            self.input_processor,
            complementary_data[self.task_key],
            observation[self.image_key],
            max_text_length=self.max_text_length,
            use_fast_image_path=self.use_fast_image_path,
        )
        new_observation = dict(observation)
        new_observation[OBS_LANGUAGE_TOKENS] = native["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = native["attention_mask"]
        new_observation[OBS_MOLMO_TOKEN_TYPE_IDS] = native["token_type_ids"]
        new_observation[OBS_MOLMO_PIXEL_VALUES] = native["pixel_values"]
        new_observation[OBS_MOLMO_IMAGE_TOKEN_POOLING] = native["image_token_pooling"]
        new_observation[OBS_MOLMO_IMAGE_GRIDS] = native["image_grids"]
        new_observation[OBS_MOLMO_IMAGE_NUM_CROPS] = native["image_num_crops"]
        return new_observation

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "image_key": self.image_key,
            "task_key": self.task_key,
            "max_text_length": self.max_text_length,
            "use_fast_image_path": self.use_fast_image_path,
        }
        if self.processor_name is not None and self.processor is None:
            config["processor_name"] = self.processor_name
        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation_features = features.setdefault(PipelineFeatureType.OBSERVATION, {})
        # 410 native image positions + at most 48 task tokens + the fixed
        # 9-token BOS/user/end/assistant template overhead.
        sequence_length = MOLMO2_EXPECTED_IMAGE_POSITIONS + self.max_text_length + 9
        observation_features[OBS_LANGUAGE_TOKENS] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(sequence_length,)
        )
        observation_features[OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(sequence_length,)
        )
        observation_features[OBS_MOLMO_TOKEN_TYPE_IDS] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(sequence_length,)
        )
        observation_features[OBS_MOLMO_PIXEL_VALUES] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(2, 729, 588)
        )
        observation_features[OBS_MOLMO_IMAGE_TOKEN_POOLING] = PolicyFeature(
            type=FeatureType.ENV, shape=(392, 4)
        )
        observation_features[OBS_MOLMO_IMAGE_GRIDS] = PolicyFeature(type=FeatureType.ENV, shape=(4,))
        observation_features[OBS_MOLMO_IMAGE_NUM_CROPS] = PolicyFeature(type=FeatureType.ENV, shape=())
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Converting world coordinates to egocentric coordinates relative to end-effector pose.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Normalizing actions to [-1, 1] range for flow matching (required for training stability).
    7.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Unnormalizing the output actions to their original scale (inverse of normalization).
    2.  Moving data to the CPU.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
    ]
    if config.vlm_backend == "molmo2_full":
        image_key = f"observation.images.{config.selected_rgb_camera_views[0]}"
        input_steps.extend(
            [
                UMIProcessor(),
                Molmo2FullMultimodalProcessorStep(
                    processor_name=config.vlm_model_name,
                    image_key=image_key,
                    max_text_length=config.tokenizer_max_length,
                    use_fast_image_path=getattr(config, "molmo_image_fast_path", True),
                ),
            ]
        )
    else:
        tokenizer_step_cls = (
            Molmo2BOSTokenizerProcessorStep if config.vlm_backend == "molmo2_text" else TokenizerProcessorStep
        )
        input_steps.extend(
            [
                SmolVLANewLineProcessor(),
                tokenizer_step_cls(
                    tokenizer_name=config.vlm_model_name,
                    padding=config.pad_language_to,
                    padding_side="right",
                    max_length=config.tokenizer_max_length,
                ),
                UMIProcessor(),
            ]
        )
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
                device=config.device,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def validate_smolvla_worldflow_preprocessor(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> None:
    """Reject processor pipelines that would violate the World--Ego frame contract.

    Checkpoints load their saved processor topology instead of rebuilding the
    current SmolVLA defaults. WorldFlow requires exactly one UMI transform, and
    that transform must run before action normalization so the model receives
    current-EEF-relative physical pose9 actions.
    """

    umi_indices = [index for index, step in enumerate(preprocessor.steps) if isinstance(step, UMIProcessor)]
    if len(umi_indices) != 1:
        raise ValueError(
            "worldflow_enable=True requires exactly one UMIProcessor in the policy "
            f"preprocessor, found {len(umi_indices)}. Rebuild the processor from the current "
            "SmolVLA configuration instead of reusing an incompatible checkpoint pipeline."
        )

    normalizer_indices = [
        index for index, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep)
    ]
    if normalizer_indices and umi_indices[0] > min(normalizer_indices):
        raise ValueError(
            "WorldFlow requires UMIProcessor to run before NormalizerProcessorStep so the "
            "Ego action is converted to the current EEF frame in physical pose9 coordinates."
        )


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
