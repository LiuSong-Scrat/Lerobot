#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.datasets.utils import write_json
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int | None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def build_phase_cosine_decay_scheduler(
    optimizer: Optimizer,
    *,
    start_lr: float,
    end_lr: float,
    num_decay_steps: int,
    phase_step: int = 0,
) -> LambdaLR:
    """Build a phase-relative cosine decay for an optimizer-state-only resume.

    ``phase_step`` is independent from the global training step. At phase step
    zero the first optimizer update uses ``start_lr``; after exactly
    ``num_decay_steps`` scheduler updates it uses ``end_lr``. Additional steps
    remain clamped at ``end_lr``.

    Multiple optimizer groups retain their pre-existing LR ratios. The explicit
    start/end values refer to group zero, which is the base policy group for the
    Song training presets.
    """

    start_lr = float(start_lr)
    end_lr = float(end_lr)
    if not math.isfinite(start_lr) or start_lr <= 0.0:
        raise ValueError(f"start_lr must be finite and positive, got {start_lr!r}.")
    if not math.isfinite(end_lr) or end_lr < 0.0 or end_lr > start_lr:
        raise ValueError(f"end_lr must be finite and in [0, start_lr], got {end_lr!r}.")
    if isinstance(num_decay_steps, bool) or not isinstance(num_decay_steps, int) or num_decay_steps < 1:
        raise ValueError(f"num_decay_steps must be a positive integer, got {num_decay_steps!r}.")
    if isinstance(phase_step, bool) or not isinstance(phase_step, int) or phase_step < 0:
        raise ValueError(f"phase_step must be a non-negative integer, got {phase_step!r}.")
    if not optimizer.param_groups:
        raise ValueError("Cannot build a resume scheduler for an optimizer with no parameter groups.")

    reference_lrs = [
        float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups
    ]
    base_reference_lr = reference_lrs[0]
    if not math.isfinite(base_reference_lr) or base_reference_lr <= 0.0:
        raise ValueError(
            "Optimizer group zero must have a finite positive reference LR before "
            f"scheduler restart, got {base_reference_lr!r}."
        )

    group_start_lrs: list[float] = []
    for group_index, (group, reference_lr) in enumerate(
        zip(optimizer.param_groups, reference_lrs, strict=True)
    ):
        if not math.isfinite(reference_lr) or reference_lr <= 0.0:
            raise ValueError(
                f"Optimizer group {group_index} has invalid reference LR {reference_lr!r}."
            )
        group_start_lr = start_lr * (reference_lr / base_reference_lr)
        group["lr"] = group_start_lr
        # LambdaLR reads this field as its immutable multiplicative base when
        # last_epoch is non-negative, including reconstruction mid-phase.
        group["initial_lr"] = group_start_lr
        group_start_lrs.append(group_start_lr)

    if "lr" in optimizer.defaults:
        optimizer.defaults["lr"] = start_lr

    end_ratio = end_lr / start_lr

    def lr_lambda(current_phase_step: int) -> float:
        bounded_step = min(max(int(current_phase_step), 0), num_decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * bounded_step / num_decay_steps))
        return end_ratio + (1.0 - end_ratio) * cosine

    # LRScheduler performs one initial step in its constructor. Using
    # phase_step - 1 therefore reconstructs exactly the requested current phase
    # without replaying optimizer updates or loading the old scheduler state.
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=phase_step - 1)
    expected_lrs = [base_lr * lr_lambda(phase_step) for base_lr in group_start_lrs]
    actual_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0) for actual, expected in zip(actual_lrs, expected_lrs, strict=True)):
        raise RuntimeError(
            "Phase-relative scheduler initialization produced unexpected optimizer LRs: "
            f"expected={expected_lrs}, actual={actual_lrs}."
        )
    return scheduler


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
