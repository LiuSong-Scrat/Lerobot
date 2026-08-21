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

"""Native multimodal preprocessing for the frozen full Molmo2-ER backend.

The model-facing names live here so the policy processor and model share one
contract. Language IDs and their validity mask deliberately keep LeRobot's
existing ``observation.language.*`` names; the five fields below are the
additional native Molmo visual inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


OBS_MOLMO_TOKEN_TYPE_IDS = "observation.molmo.token_type_ids"
OBS_MOLMO_PIXEL_VALUES = "observation.molmo.pixel_values"
OBS_MOLMO_IMAGE_TOKEN_POOLING = "observation.molmo.image_token_pooling"
OBS_MOLMO_IMAGE_GRIDS = "observation.molmo.image_grids"
OBS_MOLMO_IMAGE_NUM_CROPS = "observation.molmo.image_num_crops"

MOLMO2_NATIVE_OUTPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
)

MOLMO2_EXPECTED_IMAGE_CROPS = 2
MOLMO2_EXPECTED_PATCH_POSITIONS = 392
MOLMO2_EXPECTED_IMAGE_POSITIONS = 410
MOLMO2_EXPECTED_IMAGE_SIZE = 256
MOLMO2_DEFAULT_MAX_TEXT_TOKENS = 48


def load_local_molmo2_processor(processor_name: str | Path) -> Any:
    """Load the trusted, slow Molmo2 processor from a local model directory."""

    if not _transformers_available or AutoProcessor is None:
        raise ImportError("The 'transformers' library is required for native Molmo2-ER preprocessing.")

    processor_path = Path(processor_name).expanduser()
    if not processor_path.is_dir():
        raise ValueError(
            f"Molmo2-ER native preprocessing requires a local model directory; got {str(processor_name)!r}."
        )

    processor = AutoProcessor.from_pretrained(
        str(processor_path.resolve()),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    if getattr(processor, "tokenizer", None) is None:
        raise ValueError("The local Molmo2-ER processor does not expose a tokenizer.")
    processor.tokenizer.padding_side = "right"
    return processor


def _normalize_tasks(tasks: str | Sequence[str], batch_size: int) -> list[str]:
    if isinstance(tasks, str):
        normalized = [tasks]
    elif isinstance(tasks, Sequence) and all(isinstance(task, str) for task in tasks):
        normalized = list(tasks)
    else:
        raise TypeError("Molmo2 tasks must be a string or a sequence of strings.")

    if len(normalized) != batch_size:
        raise ValueError(f"Molmo2 received {len(normalized)} tasks for an RGB batch of size {batch_size}.")
    return normalized


def _truncate_task_content(tokenizer: Any, task: str, max_text_tokens: int) -> tuple[str, int]:
    """Truncate only user-authored text, before adding image and role tokens."""

    encoded = tokenizer(
        task,
        add_special_tokens=False,
        truncation=True,
        max_length=max_text_tokens,
        padding=False,
    )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError("A single Molmo task unexpectedly tokenized as a batch.")
        encoded = encoded[0]

    truncated = tokenizer.decode(
        encoded,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    roundtrip_ids = tokenizer(truncated, add_special_tokens=False, padding=False)["input_ids"]
    if isinstance(roundtrip_ids, torch.Tensor):
        roundtrip_ids = roundtrip_ids.detach().cpu().tolist()
    if roundtrip_ids and isinstance(roundtrip_ids[0], list):
        roundtrip_ids = roundtrip_ids[0]
    if len(roundtrip_ids) > max_text_tokens:
        raise RuntimeError(
            "Molmo tokenizer decode/encode round trip exceeded the independent text budget: "
            f"{len(roundtrip_ids)} > {max_text_tokens}."
        )
    return truncated, len(roundtrip_ids)


def _normalize_images(images: torch.Tensor | Sequence[Any]) -> torch.Tensor:
    """Normalize a batched tensor or per-sample image list to BCHW."""

    if isinstance(images, torch.Tensor):
        if images.ndim == 5:
            if images.shape[1] != 1:
                raise ValueError(
                    "Full-Molmo2-ER is locked to n_obs_steps=1; expected a singleton temporal "
                    f"RGB axis in BTCHW input, got shape={tuple(images.shape)}."
                )
            images = images[:, 0]
        return images
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or len(images) == 0:
        raise TypeError("Molmo2 images must be a BCHW tensor or a non-empty image sequence.")

    normalized: list[torch.Tensor] = []
    for index, image in enumerate(images):
        if not isinstance(image, torch.Tensor):
            try:
                image = torch.as_tensor(image)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Molmo2 image {index} is not tensor-like.") from exc
        if image.ndim != 3:
            raise ValueError(f"Molmo2 image {index} must be rank 3, got shape={tuple(image.shape)}.")
        if tuple(image.shape) == (
            3,
            MOLMO2_EXPECTED_IMAGE_SIZE,
            MOLMO2_EXPECTED_IMAGE_SIZE,
        ):
            normalized.append(image)
        elif tuple(image.shape) == (
            MOLMO2_EXPECTED_IMAGE_SIZE,
            MOLMO2_EXPECTED_IMAGE_SIZE,
            3,
        ):
            normalized.append(image.permute(2, 0, 1))
        else:
            raise ValueError(
                f"Molmo2 image {index} must be CHW or HWC 256x256 RGB, got {tuple(image.shape)}."
            )
    try:
        return torch.stack(normalized, dim=0)
    except RuntimeError as exc:
        raise ValueError("Molmo2 image list entries must share dtype and device.") from exc


def _validate_raw_rgb(images: torch.Tensor) -> None:
    expected = (
        "Molmo2 RGB must be a float tensor shaped "
        f"(B, 3, {MOLMO2_EXPECTED_IMAGE_SIZE}, {MOLMO2_EXPECTED_IMAGE_SIZE}) in [0, 1]"
    )
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{expected}; got {type(images).__name__}.")
    if images.ndim != 4 or tuple(images.shape[1:]) != (
        3,
        MOLMO2_EXPECTED_IMAGE_SIZE,
        MOLMO2_EXPECTED_IMAGE_SIZE,
    ):
        raise ValueError(f"{expected}; got shape={tuple(images.shape)}.")
    if not images.is_floating_point():
        raise TypeError(f"{expected}; got dtype={images.dtype}.")
    if not torch.isfinite(images).all():
        raise ValueError("Molmo2 RGB contains NaN or infinity.")
    minimum = float(images.detach().amin().cpu())
    maximum = float(images.detach().amax().cpu())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"Molmo2 RGB must remain in [0, 1], got range [{minimum}, {maximum}].")


def _as_tensor(value: Any, *, key: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Molmo processor output {key!r} is not tensor-like.") from exc


def _repad_text_fields_to_the_right(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move each valid native sequence to index zero without changing token order."""

    repadded_ids = torch.full_like(input_ids, pad_token_id)
    repadded_attention = torch.zeros_like(attention_mask, dtype=torch.bool)
    repadded_types = torch.zeros_like(token_type_ids, dtype=torch.bool)
    for batch_index in range(input_ids.shape[0]):
        valid = attention_mask[batch_index].to(dtype=torch.bool)
        valid_count = int(valid.sum().item())
        repadded_ids[batch_index, :valid_count] = input_ids[batch_index, valid]
        repadded_attention[batch_index, :valid_count] = True
        repadded_types[batch_index, :valid_count] = token_type_ids[batch_index, valid].to(dtype=torch.bool)
    return repadded_ids, repadded_attention, repadded_types


def prepare_molmo2_multimodal_batch(
    processor: Any,
    tasks: str | Sequence[str],
    images: torch.Tensor | Sequence[Any],
    *,
    max_text_length: int,
) -> dict[str, torch.Tensor]:
    """Create a full, untruncated native Molmo2 batch from LeRobot RGB and tasks.

    ``max_text_length`` applies only to each task's natural-language content.
    The native BOS, 410-position image block, chat role tokens, and assistant
    generation prompt are added afterwards and are never subject to a sequence
    ``max_length``.
    """

    if max_text_length <= 0:
        raise ValueError(f"max_text_length must be positive, got {max_text_length}.")
    if getattr(processor, "tokenizer", None) is None:
        raise ValueError("Molmo processor must expose its native tokenizer.")

    images_tensor = _normalize_images(images)
    _validate_raw_rgb(images_tensor)
    batch_size = int(images_tensor.shape[0])
    normalized_tasks = _normalize_tasks(tasks, batch_size)
    processor.tokenizer.padding_side = "right"

    prompts: list[str] = []
    text_lengths: list[int] = []
    for task in normalized_tasks:
        truncated_task, text_length = _truncate_task_content(processor.tokenizer, task, max_text_length)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": truncated_task},
                ],
            }
        ]
        prompts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        text_lengths.append(text_length)

    # Molmo's image processor accepts HWC float arrays in [0, 1].  Keeping a
    # list here preserves the one-image-per-example association for a batch.
    images_hwc = images_tensor.detach().to(device="cpu", dtype=torch.float32).permute(0, 2, 3, 1).contiguous()
    # Molmo2-ER's remote ``insert_bos`` implementation expects left-padded
    # tokenizer output. Calling it on right-padded rows turns trailing PADs
    # into valid tokens. Use its native side internally, then immediately
    # canonicalize all three sequence fields to right padding below.
    processor.tokenizer.padding_side = "left"
    try:
        native = processor(
            text=prompts,
            images=[image.numpy() for image in images_hwc],
            padding="longest",
            truncation=False,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
    finally:
        processor.tokenizer.padding_side = "right"

    missing = [key for key in MOLMO2_NATIVE_OUTPUT_KEYS if key not in native]
    if missing:
        raise RuntimeError(f"Molmo processor omitted required native outputs: {missing}.")
    output = {key: _as_tensor(native[key], key=key) for key in MOLMO2_NATIVE_OUTPUT_KEYS}
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("Molmo tokenizer must define a PAD token ID for right padding.")
    (
        output["input_ids"],
        output["attention_mask"],
        output["token_type_ids"],
    ) = _repad_text_fields_to_the_right(
        output["input_ids"],
        output["attention_mask"],
        output["token_type_ids"],
        pad_token_id=int(pad_token_id),
    )

    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    token_type_ids = output["token_type_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
        raise RuntimeError(
            f"Molmo input_ids must have shape (B, S), got {tuple(input_ids.shape)} for B={batch_size}."
        )
    if attention_mask.shape != input_ids.shape or token_type_ids.shape != input_ids.shape:
        raise RuntimeError(
            "Molmo attention_mask and token_type_ids must exactly match input_ids; "
            f"got ids={tuple(input_ids.shape)}, attention={tuple(attention_mask.shape)}, "
            f"types={tuple(token_type_ids.shape)}."
        )

    # A right-padded mask is monotonically non-increasing along the sequence.
    if attention_mask.shape[1] > 1 and torch.any(attention_mask[:, 1:] & ~attention_mask[:, :-1]):
        raise RuntimeError("Molmo native batch is not right padded.")
    if torch.any(token_type_ids & ~attention_mask):
        raise RuntimeError("Molmo marked a padded position as an IMAGE token.")

    image_num_crops = output["image_num_crops"]
    if image_num_crops.shape != (batch_size,) or not torch.all(
        image_num_crops == MOLMO2_EXPECTED_IMAGE_CROPS
    ):
        raise RuntimeError(
            "Each 256x256 Molmo image must produce exactly two crops; "
            f"got shape={tuple(image_num_crops.shape)}, values={image_num_crops.tolist()}."
        )

    image_patch_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")
    if image_patch_id is None or int(image_patch_id) < 0:
        raise RuntimeError("Molmo tokenizer does not define the native <im_patch> token.")
    patch_counts = ((input_ids == int(image_patch_id)) & attention_mask).sum(dim=1)
    if not torch.all(patch_counts == MOLMO2_EXPECTED_PATCH_POSITIONS):
        raise RuntimeError(
            f"Each Molmo sample must contain exactly 392 <im_patch> positions; got {patch_counts.tolist()}."
        )

    image_counts = (token_type_ids & attention_mask).sum(dim=1)
    if not torch.all(image_counts == MOLMO2_EXPECTED_IMAGE_POSITIONS):
        raise RuntimeError(
            f"Each Molmo sample must contain exactly 410 IMAGE positions; got {image_counts.tolist()}."
        )

    pixel_values = output["pixel_values"]
    image_token_pooling = output["image_token_pooling"]
    image_grids = output["image_grids"]
    if pixel_values.ndim < 1 or pixel_values.shape[0] != batch_size * MOLMO2_EXPECTED_IMAGE_CROPS:
        raise RuntimeError(
            f"Molmo pixel_values must contain two crops per sample; got shape={tuple(pixel_values.shape)}."
        )
    if image_token_pooling.ndim < 1 or image_token_pooling.shape[0] != (
        batch_size * MOLMO2_EXPECTED_PATCH_POSITIONS
    ):
        raise RuntimeError(
            "Molmo image_token_pooling must contain 392 rows per sample; "
            f"got shape={tuple(image_token_pooling.shape)}."
        )
    if image_grids.ndim != 2 or image_grids.shape[0] != batch_size:
        raise RuntimeError(
            f"Molmo image_grids must contain one row per sample, got {tuple(image_grids.shape)}."
        )

    if any(length > max_text_length for length in text_lengths):
        raise RuntimeError(f"Molmo task text exceeded its independent {max_text_length}-token budget.")
    return output


__all__ = [
    "MOLMO2_DEFAULT_MAX_TEXT_TOKENS",
    "MOLMO2_EXPECTED_IMAGE_CROPS",
    "MOLMO2_EXPECTED_IMAGE_POSITIONS",
    "MOLMO2_EXPECTED_PATCH_POSITIONS",
    "MOLMO2_NATIVE_OUTPUT_KEYS",
    "OBS_MOLMO_IMAGE_GRIDS",
    "OBS_MOLMO_IMAGE_NUM_CROPS",
    "OBS_MOLMO_IMAGE_TOKEN_POOLING",
    "OBS_MOLMO_PIXEL_VALUES",
    "OBS_MOLMO_TOKEN_TYPE_IDS",
    "load_local_molmo2_processor",
    "prepare_molmo2_multimodal_batch",
]
