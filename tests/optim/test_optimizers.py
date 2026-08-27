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
import pytest
import torch

from lerobot.optim.optimizers import (
    AdamConfig,
    AdamWConfig,
    FP32MasterAdamW,
    MultiAdamConfig,
    SGDConfig,
    load_optimizer_state,
    optimizer_model_parameters,
    save_optimizer_state,
)
from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)


@pytest.mark.parametrize(
    "config_cls, expected_class",
    [
        (AdamConfig, torch.optim.Adam),
        (AdamWConfig, torch.optim.AdamW),
        (SGDConfig, torch.optim.SGD),
        (MultiAdamConfig, dict),
    ],
)
def test_optimizer_build(config_cls, expected_class, model_params):
    config = config_cls()
    if config_cls == MultiAdamConfig:
        params_dict = {"default": model_params}
        optimizer = config.build(params_dict)
        assert isinstance(optimizer, expected_class)
        assert isinstance(optimizer["default"], torch.optim.Adam)
        assert optimizer["default"].defaults["lr"] == config.lr
    else:
        optimizer = config.build(model_params)
        assert isinstance(optimizer, expected_class)
        assert optimizer.defaults["lr"] == config.lr


def test_save_optimizer_state(optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    assert (tmp_path / OPTIMIZER_STATE).is_file()
    assert (tmp_path / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_and_load_optimizer_state(model_params, optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    loaded_optimizer = AdamConfig().build(model_params)
    loaded_optimizer = load_optimizer_state(loaded_optimizer, tmp_path)

    torch.testing.assert_close(optimizer.state_dict(), loaded_optimizer.state_dict())


def test_load_optimizer_state_can_keep_current_param_group_hyperparameters(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    saved_optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.95))
    parameter.grad = torch.tensor([0.5, -0.25])
    saved_optimizer.step()
    save_optimizer_state(saved_optimizer, tmp_path)

    current_optimizer = torch.optim.AdamW([parameter], lr=7e-5, betas=(0.8, 0.9))
    loaded_optimizer = load_optimizer_state(
        current_optimizer,
        tmp_path,
        restore_param_group_hyperparameters=False,
    )

    assert loaded_optimizer.param_groups[0]["lr"] == 7e-5
    assert loaded_optimizer.param_groups[0]["betas"] == (0.8, 0.9)
    for state_name, expected_value in saved_optimizer.state[parameter].items():
        torch.testing.assert_close(loaded_optimizer.state[parameter][state_name], expected_value)


def test_adamw_fp32_master_is_opt_in_and_accumulates_sub_ulp_updates():
    ordinary_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    master_parameter = torch.nn.Parameter(ordinary_parameter.detach().clone())
    common = {
        "lr": 1e-4,
        "betas": (0.0, 0.0),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    ordinary_optimizer = AdamWConfig(**common).build([ordinary_parameter])
    master_optimizer = AdamWConfig(**common, fp32_master_weights=True).build([master_parameter])

    assert type(ordinary_optimizer) is torch.optim.AdamW
    assert isinstance(master_optimizer, FP32MasterAdamW)

    for _ in range(100):
        ordinary_parameter.grad = torch.ones_like(ordinary_parameter)
        master_parameter.grad = torch.ones_like(master_parameter)
        ordinary_optimizer.step()
        master_optimizer.step()
        ordinary_optimizer.zero_grad()
        master_optimizer.zero_grad()

    # A 1e-4 update is below the BF16 ULP around 1.0, so ordinary AdamW
    # repeatedly loses it. The FP32 master accumulates all 100 updates.
    torch.testing.assert_close(ordinary_parameter.detach(), torch.tensor([1.0], dtype=torch.bfloat16))
    assert master_parameter.item() < 1.0
    torch.testing.assert_close(
        master_optimizer.optimizer_parameter_for(master_parameter).detach(),
        torch.tensor([0.99]),
        atol=2e-6,
        rtol=0.0,
    )


def test_fp32_master_preserves_param_groups_scheduler_and_model_identity():
    low_precision = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    full_precision = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = AdamWConfig(
        lr=1e-3,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        fp32_master_weights=True,
    ).build(
        [
            {"params": [low_precision], "lr": 1e-4, "group_name": "low"},
            {"params": [full_precision], "lr": 2e-4, "group_name": "full"},
        ]
    )

    low_master = optimizer.optimizer_parameter_for(low_precision)
    assert low_master is not low_precision
    assert low_master.dtype == torch.float32
    master_parameters = list(optimizer.master_parameters())
    assert len(master_parameters) == 1
    assert master_parameters[0] is low_master
    assert optimizer.optimizer_parameter_for(full_precision) is full_precision
    assert [group["group_name"] for group in optimizer.param_groups] == ["low", "full"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 2e-4]
    represented_parameters = list(optimizer_model_parameters(optimizer))
    assert represented_parameters[0] is low_precision
    assert represented_parameters[1] is full_precision

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 0.5)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5, 1e-4]
    low_precision.grad = torch.ones_like(low_precision)
    full_precision.grad = torch.ones_like(full_precision)
    optimizer.step()
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5, 1e-4]


def test_fp32_master_zero_grad_clears_model_and_master_gradients():
    low_precision = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    full_precision = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = AdamWConfig(fp32_master_weights=True).build([low_precision, full_precision])
    low_master = optimizer.optimizer_parameter_for(low_precision)

    # Use a view to mirror DDP's gradient_as_bucket_view=True behavior.
    low_precision.grad = torch.ones(2, dtype=torch.bfloat16)[:1]
    full_precision.grad = torch.ones_like(full_precision)
    optimizer.step()
    assert low_master.grad is not None

    optimizer.zero_grad(set_to_none=False)
    torch.testing.assert_close(low_precision.grad, torch.zeros_like(low_precision))
    torch.testing.assert_close(low_master.grad, torch.zeros_like(low_master))
    torch.testing.assert_close(full_precision.grad, torch.zeros_like(full_precision))

    optimizer.zero_grad(set_to_none=True)
    assert low_precision.grad is None
    assert low_master.grad is None
    assert full_precision.grad is None


def test_fp32_master_synchronization_preserves_unrounded_master(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = AdamWConfig(fp32_master_weights=True).build([parameter])
    master = optimizer.optimizer_parameter_for(parameter)
    with torch.no_grad():
        master.copy_(torch.tensor([0.998]))

    broadcasts = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_broadcast(tensor, src):
        broadcasts.append((tensor, src))

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    optimizer.synchronize_master_parameters(src=0)

    assert broadcasts == [(master, 0)]
    torch.testing.assert_close(master, torch.tensor([0.998]), atol=0.0, rtol=0.0)
    torch.testing.assert_close(parameter, master.to(dtype=torch.bfloat16), atol=0.0, rtol=0.0)


def test_fp32_master_state_roundtrip_preserves_unrounded_master(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = AdamWConfig(
        lr=1e-4,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        fp32_master_weights=True,
    ).build([parameter])
    for _ in range(17):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    expected_master = optimizer.optimizer_parameter_for(parameter).detach().clone()
    save_optimizer_state(optimizer, tmp_path)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = AdamWConfig(
        lr=1e-4,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        fp32_master_weights=True,
    ).build([resumed_parameter])
    load_optimizer_state(resumed_optimizer, tmp_path)
    resumed_master = resumed_optimizer.optimizer_parameter_for(resumed_parameter)

    torch.testing.assert_close(resumed_master, expected_master, atol=0.0, rtol=0.0)
    assert resumed_optimizer.state[resumed_master]["exp_avg"].dtype == torch.float32
    assert resumed_optimizer.state[resumed_master]["exp_avg_sq"].dtype == torch.float32

    parameter.grad = torch.ones_like(parameter)
    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    optimizer.step()
    resumed_optimizer.step()
    torch.testing.assert_close(resumed_master, optimizer.optimizer_parameter_for(parameter))
    torch.testing.assert_close(resumed_parameter, parameter)


def test_fp32_master_state_roundtrip_before_first_adam_step(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = AdamWConfig(fp32_master_weights=True).build([parameter])
    master = optimizer.optimizer_parameter_for(parameter)
    with torch.no_grad():
        master.add_(-1e-4)
    expected_master = master.detach().clone()
    assert not optimizer.state
    save_optimizer_state(optimizer, tmp_path)
    # state_dict() must not make the serialized master a persistent live-state
    # allocation, and load must not make AdamW mistake it for initialized m/v.
    assert not optimizer.state

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = AdamWConfig(fp32_master_weights=True).build([resumed_parameter])
    load_optimizer_state(resumed_optimizer, tmp_path)
    resumed_master = resumed_optimizer.optimizer_parameter_for(resumed_parameter)

    torch.testing.assert_close(resumed_master, expected_master, atol=0.0, rtol=0.0)
    assert not resumed_optimizer.state


def test_fp32_master_loads_and_upgrades_legacy_bf16_adamw_state(tmp_path):
    legacy_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    legacy_optimizer = torch.optim.AdamW([legacy_parameter], lr=1e-3)
    legacy_parameter.grad = torch.ones_like(legacy_parameter)
    legacy_optimizer.step()
    legacy_state_dict = legacy_optimizer.state_dict()
    assert legacy_state_dict["state"][0]["exp_avg"].dtype == torch.bfloat16
    save_optimizer_state(legacy_optimizer, tmp_path)

    resumed_parameter = torch.nn.Parameter(legacy_parameter.detach().clone())
    resumed_optimizer = AdamWConfig(lr=1e-3, fp32_master_weights=True).build([resumed_parameter])
    load_optimizer_state(resumed_optimizer, tmp_path)
    resumed_master = resumed_optimizer.optimizer_parameter_for(resumed_parameter)
    resumed_state = resumed_optimizer.state[resumed_master]

    assert resumed_state["exp_avg"].dtype == torch.float32
    assert resumed_state["exp_avg_sq"].dtype == torch.float32
    torch.testing.assert_close(resumed_master, resumed_parameter.float())


def test_fp32_master_checkpoint_can_load_without_master_mode(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = AdamWConfig(lr=1e-3, fp32_master_weights=True).build([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    save_optimizer_state(optimizer, tmp_path)

    ordinary_parameter = torch.nn.Parameter(parameter.detach().clone())
    ordinary_optimizer = AdamWConfig(lr=1e-3).build([ordinary_parameter])
    load_optimizer_state(ordinary_optimizer, tmp_path)

    assert "fp32_master_param" not in ordinary_optimizer.state[ordinary_parameter]
    assert ordinary_optimizer.state[ordinary_parameter]["exp_avg"].dtype == torch.bfloat16


@pytest.fixture
def base_params_dict():
    return {
        "actor": [torch.nn.Parameter(torch.randn(10, 10))],
        "critic": [torch.nn.Parameter(torch.randn(5, 5))],
        "temperature": [torch.nn.Parameter(torch.randn(3, 3))],
    }


@pytest.mark.parametrize(
    "config_params, expected_values",
    [
        # Test 1: Basic configuration with different learning rates
        (
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_groups": {
                    "actor": {"lr": 1e-4},
                    "critic": {"lr": 5e-4},
                    "temperature": {"lr": 2e-3},
                },
            },
            {
                "actor": {"lr": 1e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
                "critic": {"lr": 5e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
                "temperature": {"lr": 2e-3, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
            },
        ),
        # Test 2: Different weight decays and beta values
        (
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_groups": {
                    "actor": {"lr": 1e-4, "weight_decay": 1e-5},
                    "critic": {"lr": 5e-4, "weight_decay": 1e-6},
                    "temperature": {"lr": 2e-3, "betas": (0.95, 0.999)},
                },
            },
            {
                "actor": {"lr": 1e-4, "weight_decay": 1e-5, "betas": (0.9, 0.999)},
                "critic": {"lr": 5e-4, "weight_decay": 1e-6, "betas": (0.9, 0.999)},
                "temperature": {"lr": 2e-3, "weight_decay": 1e-4, "betas": (0.95, 0.999)},
            },
        ),
        # Test 3: Epsilon parameter customization
        (
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_groups": {
                    "actor": {"lr": 1e-4, "eps": 1e-6},
                    "critic": {"lr": 5e-4, "eps": 1e-7},
                    "temperature": {"lr": 2e-3, "eps": 1e-8},
                },
            },
            {
                "actor": {"lr": 1e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-6},
                "critic": {"lr": 5e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-7},
                "temperature": {"lr": 2e-3, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8},
            },
        ),
    ],
)
def test_multi_adam_configuration(base_params_dict, config_params, expected_values):
    # Create config with the given parameters
    config = MultiAdamConfig(**config_params)
    optimizers = config.build(base_params_dict)

    # Verify optimizer count and keys
    assert len(optimizers) == len(expected_values)
    assert set(optimizers.keys()) == set(expected_values.keys())

    # Check that all optimizers are Adam instances
    for opt in optimizers.values():
        assert isinstance(opt, torch.optim.Adam)

    # Verify hyperparameters for each optimizer
    for name, expected in expected_values.items():
        optimizer = optimizers[name]
        for param, value in expected.items():
            assert optimizer.defaults[param] == value


@pytest.fixture
def multi_optimizers(base_params_dict):
    config = MultiAdamConfig(
        lr=1e-3,
        optimizer_groups={
            "actor": {"lr": 1e-4},
            "critic": {"lr": 5e-4},
            "temperature": {"lr": 2e-3},
        },
    )
    return config.build(base_params_dict)


def test_save_multi_optimizer_state(multi_optimizers, tmp_path):
    # Save optimizer states
    save_optimizer_state(multi_optimizers, tmp_path)

    # Verify that directories were created for each optimizer
    for name in multi_optimizers:
        assert (tmp_path / name).is_dir()
        assert (tmp_path / name / OPTIMIZER_STATE).is_file()
        assert (tmp_path / name / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_and_load_multi_optimizer_state(base_params_dict, multi_optimizers, tmp_path):
    # Option 1: Add a minimal backward pass to populate optimizer states
    for name, params in base_params_dict.items():
        if name in multi_optimizers:
            # Create a dummy loss and do backward
            dummy_loss = params[0].sum()
            dummy_loss.backward()
            # Perform an optimization step
            multi_optimizers[name].step()
            # Zero gradients for next steps
            multi_optimizers[name].zero_grad()

    # Save optimizer states
    save_optimizer_state(multi_optimizers, tmp_path)

    # Create new optimizers with the same config
    config = MultiAdamConfig(
        lr=1e-3,
        optimizer_groups={
            "actor": {"lr": 1e-4},
            "critic": {"lr": 5e-4},
            "temperature": {"lr": 2e-3},
        },
    )
    new_optimizers = config.build(base_params_dict)

    # Load optimizer states
    loaded_optimizers = load_optimizer_state(new_optimizers, tmp_path)

    # Verify state dictionaries match
    for name in multi_optimizers:
        torch.testing.assert_close(multi_optimizers[name].state_dict(), loaded_optimizers[name].state_dict())


def test_save_and_load_empty_multi_optimizer_state(base_params_dict, tmp_path):
    """Test saving and loading optimizer states even when the state is empty (no backward pass)."""
    # Create config and build optimizers
    config = MultiAdamConfig(
        lr=1e-3,
        optimizer_groups={
            "actor": {"lr": 1e-4},
            "critic": {"lr": 5e-4},
            "temperature": {"lr": 2e-3},
        },
    )
    optimizers = config.build(base_params_dict)

    # Save optimizer states without any backward pass (empty state)
    save_optimizer_state(optimizers, tmp_path)

    # Create new optimizers with the same config
    new_optimizers = config.build(base_params_dict)

    # Load optimizer states
    loaded_optimizers = load_optimizer_state(new_optimizers, tmp_path)

    # Verify hyperparameters match even with empty state
    for name, optimizer in optimizers.items():
        assert optimizer.defaults["lr"] == loaded_optimizers[name].defaults["lr"]
        assert optimizer.defaults["weight_decay"] == loaded_optimizers[name].defaults["weight_decay"]
        assert optimizer.defaults["betas"] == loaded_optimizers[name].defaults["betas"]

        # Verify state dictionaries match (they will be empty)
        torch.testing.assert_close(
            optimizer.state_dict()["param_groups"], loaded_optimizers[name].state_dict()["param_groups"]
        )
