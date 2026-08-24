#!/usr/bin/env python

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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    load_training_state_for_resume,
    load_training_step,
    save_checkpoint,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


@patch("lerobot.utils.train_utils.save_training_state")
def test_save_checkpoint(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


@patch("lerobot.utils.train_utils.save_training_state")
def test_save_checkpoint_peft(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    policy.config = Mock()
    policy.config.save_pretrained = Mock()
    cfg = Mock()
    cfg.use_peft = True
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    policy.config.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


def test_save_load_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler


def test_resume_restart_scheduler_restores_adam_state_but_not_old_lr_state(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    saved_optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.95))
    saved_scheduler = torch.optim.lr_scheduler.LambdaLR(saved_optimizer, lambda _: 0.3)
    parameter.grad = torch.tensor([0.5, -0.25])
    saved_optimizer.step()
    saved_scheduler.step()
    saved_optimizer.zero_grad(set_to_none=True)
    saved_adam_state = {
        name: value.detach().clone()
        for name, value in saved_optimizer.state[parameter].items()
    }
    save_training_state(tmp_path, 4_500, saved_optimizer, saved_scheduler)

    resumed_optimizer = torch.optim.AdamW([parameter], lr=9e-4, betas=(0.8, 0.9))
    scheduler_to_discard = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 0.5)
    cfg = SimpleNamespace(
        checkpoint_path=tmp_path,
        resume_restart_scheduler=True,
        resume_scheduler_start_lr=7e-5,
        resume_scheduler_end_lr=2e-5,
        resume_scheduler_decay_steps=30_000,
        resume_scheduler_phase_start_step=None,
        steps=34_500,
    )

    step, resumed_optimizer, resumed_scheduler = load_training_state_for_resume(
        cfg,
        resumed_optimizer,
        scheduler_to_discard,
    )

    assert step == 4_500
    assert cfg.resume_scheduler_phase_start_step == 4_500
    assert resumed_scheduler is not scheduler_to_discard
    assert resumed_scheduler.last_epoch == 0
    assert resumed_optimizer.param_groups[0]["initial_lr"] == 7e-5
    assert resumed_optimizer.param_groups[0]["lr"] == 7e-5
    assert resumed_optimizer.param_groups[0]["betas"] == (0.8, 0.9)
    for state_name, expected_value in saved_adam_state.items():
        torch.testing.assert_close(resumed_optimizer.state[parameter][state_name], expected_value)


def test_resume_helper_default_preserves_checkpoint_optimizer_and_scheduler(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    saved_optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    saved_scheduler = torch.optim.lr_scheduler.LambdaLR(saved_optimizer, lambda _: 0.3)
    parameter.grad = torch.tensor(0.5)
    saved_optimizer.step()
    saved_scheduler.step()
    save_training_state(tmp_path, 12, saved_optimizer, saved_scheduler)

    resumed_optimizer = torch.optim.AdamW([parameter], lr=7e-5)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 0.8)
    cfg = SimpleNamespace(
        checkpoint_path=tmp_path,
        resume_restart_scheduler=False,
    )

    step, resumed_optimizer, resumed_scheduler = load_training_state_for_resume(
        cfg,
        resumed_optimizer,
        resumed_scheduler,
    )

    assert step == 12
    assert resumed_optimizer.param_groups[0]["lr"] == saved_optimizer.param_groups[0]["lr"]
    assert resumed_scheduler.state_dict() == saved_scheduler.state_dict()
