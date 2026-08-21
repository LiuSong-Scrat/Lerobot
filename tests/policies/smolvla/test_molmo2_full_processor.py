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

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.policies.smolvla import processor_smolvla
from lerobot.policies.smolvla.molmo2_processing import (
    OBS_MOLMO_IMAGE_GRIDS,
    OBS_MOLMO_IMAGE_NUM_CROPS,
    OBS_MOLMO_IMAGE_TOKEN_POOLING,
    OBS_MOLMO_PIXEL_VALUES,
    OBS_MOLMO_TOKEN_TYPE_IDS,
    prepare_molmo2_multimodal_batch,
)
from lerobot.policies.smolvla.processor_smolvla import (
    Molmo2FullMultimodalProcessorStep,
    make_smolvla_pre_post_processors,
)
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class _FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 1
    padding_side = "right"
    patch_token_id = 99

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        words = text.split()
        ids = list(range(10, 10 + len(words)))
        if kwargs.get("truncation"):
            ids = ids[: int(kwargs["max_length"])]
        self.calls.append({"text": text, **kwargs, "length": len(ids)})
        return {"input_ids": ids}

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return " ".join("word" for _ in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<im_patch>"
        return self.patch_token_id


class _FakeMolmoProcessor:
    def __init__(self, *, crops: int = 2) -> None:
        self.tokenizer = _FakeTokenizer()
        self.crops = crops
        self.messages: list[list[dict[str, Any]]] = []
        self.native_call: dict[str, Any] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0] == {"type": "image"}
        self.messages.append(messages)
        return f"<image>|{messages[0]['content'][1]['text']}|<assistant>"

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.native_call = kwargs
        assert self.tokenizer.padding_side == "left"
        prompts = kwargs["text"]
        batch_size = len(prompts)
        task_lengths = [
            len(prompt.removeprefix("<image>|").removesuffix("|<assistant>").split()) for prompt in prompts
        ]
        valid_lengths = [419 + task_length for task_length in task_lengths]
        sequence_length = max(valid_lengths)

        input_ids = torch.full((batch_size, sequence_length), self.tokenizer.pad_token_id)
        attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
        token_type_ids = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
        for batch_index, valid_length in enumerate(valid_lengths):
            offset = sequence_length - valid_length
            sequence = torch.full((valid_length,), 7, dtype=torch.long)
            sequence[0] = self.tokenizer.bos_token_id
            sequence[1:393] = self.tokenizer.patch_token_id
            input_ids[batch_index, offset:] = sequence
            attention_mask[batch_index, offset:] = True
            token_type_ids[batch_index, offset + 1 : offset + 411] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "pixel_values": torch.zeros(batch_size * self.crops, 729, 588),
            "image_token_pooling": torch.zeros(batch_size * 392, 4, dtype=torch.long),
            "image_grids": torch.full((batch_size, 4), 14, dtype=torch.long),
            "image_num_crops": torch.full((batch_size,), self.crops, dtype=torch.long),
        }


def test_prepare_molmo2_batch_caps_only_text_and_preserves_native_image_sequence() -> None:
    processor = _FakeMolmoProcessor()
    chw = torch.linspace(0, 1, 3 * 256 * 256).reshape(3, 256, 256)
    hwc = chw.permute(1, 2, 0).numpy()
    tasks = ["short task", " ".join(f"word-{index}" for index in range(80))]

    output = prepare_molmo2_multimodal_batch(
        processor,
        tasks,
        [chw, hwc],
        max_text_length=48,
    )

    assert len(processor.messages) == 2
    assert len(processor.messages[1][0]["content"][1]["text"].split()) == 48
    assert processor.native_call is not None
    assert processor.native_call["padding"] == "longest"
    assert processor.native_call["truncation"] is False
    assert "max_length" not in processor.native_call
    assert all(image.shape == (256, 256, 3) for image in processor.native_call["images"])
    np.testing.assert_allclose(processor.native_call["images"][0], hwc)

    assert output["input_ids"].shape[1] > 410
    assert output["attention_mask"].dtype == torch.bool
    assert output["token_type_ids"].dtype == torch.bool
    assert output["attention_mask"].sum(dim=1).tolist() == [421, 467]
    assert not output["attention_mask"][0, 421:].any()
    assert (output["input_ids"] == processor.tokenizer.patch_token_id).sum(dim=1).tolist() == [392, 392]
    assert output["token_type_ids"].sum(dim=1).tolist() == [410, 410]
    assert output["pixel_values"].shape == (4, 729, 588)
    assert output["image_token_pooling"].shape == (784, 4)
    assert output["image_num_crops"].tolist() == [2, 2]
    assert processor.tokenizer.padding_side == "right"


def test_prepare_molmo2_batch_rejects_contract_drift() -> None:
    images = torch.zeros(1, 3, 256, 256)
    with pytest.raises(RuntimeError, match="exactly two crops"):
        prepare_molmo2_multimodal_batch(
            _FakeMolmoProcessor(crops=1),
            ["task"],
            images,
            max_text_length=48,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_molmo2_multimodal_batch(
            _FakeMolmoProcessor(),
            ["task"],
            images + 2,
            max_text_length=48,
        )


def test_registered_step_writes_native_fields_and_roundtrips_config(tmp_path, monkeypatch) -> None:
    processor = _FakeMolmoProcessor()
    monkeypatch.setattr(processor_smolvla, "load_local_molmo2_processor", lambda _: processor)
    step = Molmo2FullMultimodalProcessorStep(
        processor_name="/local/Molmo2-ER",
        image_key="observation.images.agentview",
        max_text_length=48,
    )
    transition = create_transition(
        observation={"observation.images.agentview": torch.zeros(1, 3, 256, 256)},
        complementary_data={"task": ["pick object"]},
    )
    result = step(transition)[TransitionKey.OBSERVATION]
    assert result is not None
    assert {
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_MOLMO_TOKEN_TYPE_IDS,
        OBS_MOLMO_PIXEL_VALUES,
        OBS_MOLMO_IMAGE_TOKEN_POOLING,
        OBS_MOLMO_IMAGE_GRIDS,
        OBS_MOLMO_IMAGE_NUM_CROPS,
    }.issubset(result)
    assert ProcessorStepRegistry.get("molmo2_full_multimodal_processor") is (
        Molmo2FullMultimodalProcessorStep
    )

    pipeline = PolicyProcessorPipeline(steps=[step], name="molmo_test")
    pipeline.save_pretrained(tmp_path, config_filename="molmo_test.json")
    restored = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="molmo_test.json",
        local_files_only=True,
    )
    assert isinstance(restored.steps[0], Molmo2FullMultimodalProcessorStep)
    assert restored.steps[0].get_config() == step.get_config()


def test_full_backend_replaces_tokenizer_immediately_before_device(monkeypatch) -> None:
    processor = _FakeMolmoProcessor()
    monkeypatch.setattr(processor_smolvla, "load_local_molmo2_processor", lambda _: processor)
    config = SimpleNamespace(
        vlm_backend="molmo2_full",
        selected_rgb_camera_views=("agentview",),
        vlm_model_name="/local/Molmo2-ER",
        tokenizer_max_length=48,
        input_features={},
        output_features={},
        normalization_mapping={},
        device="cpu",
    )
    preprocessor, _ = make_smolvla_pre_post_processors(config)
    full_index = next(
        index
        for index, step in enumerate(preprocessor.steps)
        if isinstance(step, Molmo2FullMultimodalProcessorStep)
    )
    device_index = next(
        index for index, step in enumerate(preprocessor.steps) if isinstance(step, DeviceProcessorStep)
    )
    assert full_index + 1 == device_index
    assert not any(isinstance(step, TokenizerProcessorStep) for step in preprocessor.steps)


def test_native_processor_accepts_only_singleton_training_time_axis() -> None:
    processor = _FakeMolmoProcessor()
    temporal = torch.zeros(2, 1, 3, 256, 256)
    result = prepare_molmo2_multimodal_batch(
        processor,
        ["task one", "task two"],
        temporal,
        max_text_length=48,
    )
    assert result["input_ids"].shape[0] == 2

    with pytest.raises(ValueError, match="n_obs_steps=1"):
        prepare_molmo2_multimodal_batch(
            processor,
            ["task one", "task two"],
            temporal.expand(-1, 2, -1, -1, -1),
            max_text_length=48,
        )
