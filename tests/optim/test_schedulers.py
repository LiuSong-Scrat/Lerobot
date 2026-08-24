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
import torch
from packaging.version import Version
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig,
    VQBeTSchedulerConfig,
    build_phase_cosine_decay_scheduler,
    load_scheduler_state,
    save_scheduler_state,
)
from lerobot.utils.constants import SCHEDULER_STATE


def test_diffuser_scheduler(optimizer):
    config = DiffuserSchedulerConfig(name="cosine", num_warmup_steps=5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()  # so that we don't get torch warning
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0002],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_vqbet_scheduler(optimizer):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=0.5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.001],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_cosine_decay_with_warmup_scheduler(optimizer):
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=90, peak_lr=0.01, decay_lr=0.001
    )
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0001818181818181819],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    loaded_scheduler = load_scheduler_state(scheduler, tmp_path)

    assert scheduler.state_dict() == loaded_scheduler.state_dict()


def test_phase_cosine_resume_scheduler_exact_start_and_end():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    second_parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter], "lr": 1e-4},
            {"params": [second_parameter], "lr": 2e-4},
        ]
    )
    scheduler = build_phase_cosine_decay_scheduler(
        optimizer,
        start_lr=7e-5,
        end_lr=2e-5,
        num_decay_steps=30_000,
    )

    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["initial_lr"] == 7e-5
    assert optimizer.param_groups[0]["lr"] == 7e-5
    assert optimizer.param_groups[1]["initial_lr"] == 1.4e-4
    assert optimizer.param_groups[1]["lr"] == 1.4e-4

    for _ in range(30_000):
        optimizer.step()
        scheduler.step()

    assert scheduler.last_epoch == 30_000
    assert optimizer.param_groups[0]["lr"] == 2e-5
    assert optimizer.param_groups[1]["lr"] == 4e-5

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2e-5
    assert optimizer.param_groups[1]["lr"] == 4e-5


def test_phase_cosine_resume_scheduler_reconstructs_mid_phase():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = build_phase_cosine_decay_scheduler(
        optimizer,
        start_lr=7e-5,
        end_lr=2e-5,
        num_decay_steps=30_000,
        phase_step=12_000,
    )

    expected_factor = 2 / 7 + (1 - 2 / 7) * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * 0.4)))
    assert scheduler.last_epoch == 12_000
    torch.testing.assert_close(
        torch.tensor(optimizer.param_groups[0]["lr"]),
        torch.tensor(7e-5) * expected_factor,
    )
