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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.utils import flatten_dict, unflatten_dict, write_json
from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.utils.io_utils import deserialize_json_into_object

# Type alias for parameters accepted by optimizer build() methods.
# This matches PyTorch's optimizer signature while also supporting:
# - dict[str, Parameter]: Named parameters for differential LR by name (e.g., XVLA)
# - dict[str, Iterable]: Multiple parameter groups for multi-optimizer configs (e.g., SAC)
OptimizerParams = (
    Iterable[torch.nn.Parameter]  # From model.parameters()
    | Iterable[dict[str, Any]]  # List of param groups with lr/weight_decay overrides
    | dict[str, torch.nn.Parameter]  # From dict(model.named_parameters()) for name-based LR
    | dict[str, Any]  # For multi-optimizer configs (SAC) with multiple param groups
)


_FP32_MASTER_STATE_KEY = "fp32_master_param"
_LOW_PRECISION_DTYPES = (torch.bfloat16, torch.float16)


class FP32MasterAdamW(torch.optim.AdamW):
    """AdamW with persistent FP32 master weights for low-precision parameters.

    The model keeps its original BF16/FP16 parameters for forward and backward.
    This optimizer substitutes an FP32 leaf tensor in the corresponding AdamW
    parameter group, copies the already-clipped model gradient to that tensor at
    ``step()``, and casts the updated master value back to the model parameter.
    FP32 model parameters stay in the optimizer unchanged.

    The master value is included in ``state_dict()`` because the rounded model
    checkpoint alone cannot preserve sub-ULP updates accumulated by the master.
    It is not retained as an extra live optimizer-state tensor, avoiding a
    second persistent FP32 copy. Checkpoints produced by ordinary AdamW are
    also accepted; their BF16/FP16 moment tensors are upgraded to FP32 by
    loading them against the FP32 master parameter.
    """

    def __init__(self, params: OptimizerParams, **kwargs: Any) -> None:
        materialized = list(params)
        if materialized and isinstance(materialized[0], dict):
            source_groups = []
            for source_group in materialized:
                group = dict(source_group)
                group["params"] = list(group["params"])
                source_groups.append(group)
        else:
            source_groups = [{"params": materialized}]

        optimizer_groups: list[dict[str, Any]] = []
        self._optimizer_to_model_parameter: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self._model_to_optimizer_parameter: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self._low_precision_optimizer_parameters: set[torch.nn.Parameter] = set()

        seen_model_parameter_ids: set[int] = set()
        for source_group in source_groups:
            optimizer_group = {key: value for key, value in source_group.items() if key != "params"}
            optimizer_parameters = []
            for model_parameter in source_group["params"]:
                if not isinstance(model_parameter, torch.Tensor):
                    raise TypeError(
                        "FP32MasterAdamW parameters must be torch tensors, "
                        f"got {type(model_parameter).__name__}."
                    )
                if id(model_parameter) in seen_model_parameter_ids:
                    raise ValueError("A model parameter appears in more than one FP32MasterAdamW group.")
                seen_model_parameter_ids.add(id(model_parameter))
                if model_parameter.dtype in _LOW_PRECISION_DTYPES:
                    optimizer_parameter = torch.nn.Parameter(
                        model_parameter.detach().to(dtype=torch.float32),
                        requires_grad=True,
                    )
                    self._low_precision_optimizer_parameters.add(optimizer_parameter)
                else:
                    optimizer_parameter = model_parameter
                optimizer_parameters.append(optimizer_parameter)
                self._optimizer_to_model_parameter[optimizer_parameter] = model_parameter
                self._model_to_optimizer_parameter[model_parameter] = optimizer_parameter
            optimizer_group["params"] = optimizer_parameters
            optimizer_groups.append(optimizer_group)

        super().__init__(optimizer_groups, **kwargs)

    def model_parameter_for(self, optimizer_parameter: torch.nn.Parameter) -> torch.nn.Parameter:
        """Return the model parameter represented by an optimizer parameter."""

        return self._optimizer_to_model_parameter[optimizer_parameter]

    def optimizer_parameter_for(self, model_parameter: torch.nn.Parameter) -> torch.nn.Parameter:
        """Return the optimizer/master parameter representing a model parameter."""

        return self._model_to_optimizer_parameter[model_parameter]

    def model_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Iterate model parameters in optimizer group/order without duplicates."""

        seen: set[int] = set()
        for group in self.param_groups:
            for optimizer_parameter in group["params"]:
                model_parameter = self.model_parameter_for(optimizer_parameter)
                if id(model_parameter) not in seen:
                    seen.add(id(model_parameter))
                    yield model_parameter

    def master_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Iterate FP32 masters in optimizer group/order."""

        for group in self.param_groups:
            for optimizer_parameter in group["params"]:
                if optimizer_parameter in self._low_precision_optimizer_parameters:
                    yield optimizer_parameter

    @torch.no_grad()
    def synchronize_master_parameters(self, src: int = 0) -> None:
        """Broadcast persistent masters without rebuilding them from rounded model weights.

        DDP broadcasts module parameters when it wraps the model, but optimizer-only
        FP32 masters are not module parameters. Broadcasting the masters themselves
        keeps fresh multi-rank initialization deterministic and, unlike copying the
        BF16/FP16 model values back into the masters, preserves sub-ULP state restored
        from an FP32-master checkpoint.
        """

        self._move_master_parameters_to_model_devices()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for master_parameter in self.master_parameters():
                torch.distributed.broadcast(master_parameter, src=src)
        # Make every rank's low-precision forward weights the exact rounded view
        # of the synchronized persistent masters.
        self._copy_master_parameters_to_model()

    @torch.no_grad()
    def _move_master_parameters_to_model_devices(self) -> None:
        # Unlike ordinary optimizer parameters, masters are not registered on
        # the module. Accelerate may move the model after optimizer creation,
        # so lazily follow the represented model tensor before the first step.
        for group in self.param_groups:
            capturable_or_fused = bool(group.get("capturable", False) or group.get("fused", False))
            for optimizer_parameter in group["params"]:
                if optimizer_parameter not in self._low_precision_optimizer_parameters:
                    continue
                model_parameter = self.model_parameter_for(optimizer_parameter)
                if optimizer_parameter.device == model_parameter.device:
                    continue
                optimizer_parameter.data = optimizer_parameter.data.to(device=model_parameter.device)
                parameter_state = self.state.get(optimizer_parameter, {})
                for state_name, value in tuple(parameter_state.items()):
                    if not torch.is_tensor(value):
                        continue
                    # Non-capturable AdamW intentionally keeps its scalar step
                    # counter on CPU; moment tensors follow the parameter.
                    if state_name == "step" and not capturable_or_fused:
                        continue
                    parameter_state[state_name] = value.to(device=model_parameter.device)

    @torch.no_grad()
    def _copy_model_gradients_to_master(self) -> None:
        self._move_master_parameters_to_model_devices()
        for optimizer_parameter in self._low_precision_optimizer_parameters:
            model_gradient = self.model_parameter_for(optimizer_parameter).grad
            if model_gradient is None:
                optimizer_parameter.grad = None
                continue
            if model_gradient.is_sparse:
                raise RuntimeError("FP32MasterAdamW does not support sparse gradients.")
            if optimizer_parameter.grad is None:
                optimizer_parameter.grad = model_gradient.detach().to(dtype=torch.float32)
            else:
                optimizer_parameter.grad.copy_(model_gradient.detach())

    @torch.no_grad()
    def _copy_master_parameters_to_model(self) -> None:
        for optimizer_parameter in self._low_precision_optimizer_parameters:
            model_parameter = self.model_parameter_for(optimizer_parameter)
            model_parameter.copy_(optimizer_parameter.to(dtype=model_parameter.dtype))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._copy_model_gradients_to_master()
        super().step()
        self._copy_master_parameters_to_model()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Clears FP32 parameters (including original FP32 model parameters) and
        # all FP32-master gradients.
        super().zero_grad(set_to_none=set_to_none)
        # Low-precision model parameters are not present in ``param_groups``.
        for optimizer_parameter in self._low_precision_optimizer_parameters:
            model_parameter = self.model_parameter_for(optimizer_parameter)
            if model_parameter.grad is None:
                continue
            if set_to_none:
                model_parameter.grad = None
            else:
                if model_parameter.grad.grad_fn is not None:
                    model_parameter.grad.detach_()
                else:
                    # DDP gradient_as_bucket_view gradients cannot be detached
                    # in-place but can be marked non-differentiable and zeroed.
                    model_parameter.grad.requires_grad_(False)
                model_parameter.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        # Match live parameter objects to the stable integer ids assigned by
        # Optimizer.state_dict(). A detached view is sufficient for immediate
        # serialization and avoids allocating another full FP32 copy.
        for group, serialized_group in zip(self.param_groups, state_dict["param_groups"], strict=True):
            for optimizer_parameter, parameter_id in zip(
                group["params"], serialized_group["params"], strict=True
            ):
                if optimizer_parameter in self._low_precision_optimizer_parameters:
                    parameter_state = dict(state_dict["state"].get(parameter_id, {}))
                    parameter_state[_FP32_MASTER_STATE_KEY] = optimizer_parameter.detach()
                    state_dict["state"][parameter_id] = parameter_state
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # AdamW assumes any non-empty per-parameter state already contains its
        # scalar ``step`` and moment tensors. A checkpoint made before the
        # first optimizer step can contain only our master value, so extract
        # that extension before delegating to AdamW. This also keeps the master
        # out of live state after loading.
        serialized_to_optimizer_parameter: dict[int, torch.nn.Parameter] = {}
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) == len(self.param_groups):
            for saved_group, current_group in zip(saved_groups, self.param_groups, strict=True):
                saved_parameter_ids = saved_group.get("params", [])
                current_parameters = current_group["params"]
                if len(saved_parameter_ids) != len(current_parameters):
                    break
                serialized_to_optimizer_parameter.update(
                    zip(saved_parameter_ids, current_parameters, strict=True)
                )

        saved_masters: dict[int, torch.Tensor] = {}
        cleaned_state = {}
        for parameter_id, parameter_state in state_dict.get("state", {}).items():
            parameter_state = dict(parameter_state)
            saved_master = parameter_state.pop(_FP32_MASTER_STATE_KEY, None)
            if saved_master is not None:
                saved_masters[parameter_id] = saved_master
            if parameter_state:
                cleaned_state[parameter_id] = parameter_state
        cleaned_state_dict = dict(state_dict)
        cleaned_state_dict["state"] = cleaned_state

        super().load_state_dict(cleaned_state_dict)
        with torch.no_grad():
            for parameter_id, saved_master in saved_masters.items():
                optimizer_parameter = serialized_to_optimizer_parameter.get(parameter_id)
                if optimizer_parameter in self._low_precision_optimizer_parameters:
                    optimizer_parameter.copy_(
                        saved_master.to(device=optimizer_parameter.device, dtype=torch.float32)
                    )
            for optimizer_parameter in self._low_precision_optimizer_parameters:
                parameter_state = self.state.get(optimizer_parameter, {})
                # Explicitly upgrade legacy low-precision Adam state. PyTorch
                # normally casts it to the target FP32 master automatically;
                # this guard makes that checkpoint compatibility invariant.
                for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = parameter_state.get(state_name)
                    if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                        parameter_state[state_name] = value.float()
        self._copy_master_parameters_to_model()


def optimizer_model_parameters(optimizer: torch.optim.Optimizer) -> Iterable[torch.nn.Parameter]:
    """Iterate the model parameters represented by an optimizer.

    This keeps parameter-identity audits correct when the optimizer contains
    FP32 master tensors rather than the low-precision model tensors. It also
    accepts Accelerate's optimizer wrapper.
    """

    unwrapped = getattr(optimizer, "optimizer", optimizer)
    if isinstance(unwrapped, FP32MasterAdamW):
        yield from unwrapped.model_parameters()
        return
    seen: set[int] = set()
    for group in unwrapped.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self, params: OptimizerParams) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """
        Build the optimizer. It can be a single optimizer or a dictionary of optimizers.

        NOTE: Multiple optimizers are useful when you have different models to optimize.
        For example, you can have one optimizer for the policy and another one for the value function
        in reinforcement learning settings.

        Args:
            params: Parameters to optimize. Accepts multiple formats depending on the optimizer:
                - Iterable[Parameter]: From model.parameters() - standard PyTorch usage
                - Iterable[dict]: List of param groups with 'params' key and optional
                  'lr', 'weight_decay' overrides (e.g., ACT, VQBeT policies)
                - dict[str, Parameter]: From dict(model.named_parameters()) for optimizers
                  that apply differential learning rates by parameter name (e.g., XVLA)
                - dict[str, Iterable]: For multi-optimizer configs where each key maps to
                  a separate optimizer's parameters (e.g., SAC with actor/critic/temperature)

        Returns:
            The optimizer or a dictionary of optimizers.
        """
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0
    fp32_master_weights: bool = False

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        fp32_master_weights = kwargs.pop("fp32_master_weights")
        optimizer_cls = FP32MasterAdamW if fp32_master_weights else torch.optim.AdamW
        return optimizer_cls(params, **kwargs)


@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)


@OptimizerConfig.register_subclass("xvla-adamw")
@dataclass
class XVLAAdamWConfig(OptimizerConfig):
    """Custom AdamW optimizer for XVLA with differential learning rates.

    The Vision-Language Model (VLM) is trained with 1/10 of the base learning rate
    for stable optimization, while all other components use the full LR.

    This LR ratio is crucial for achieving strong and stable finetuning performance.

    Soft-prompts can optionally use a separate learning rate with warm-up support.
    Set `soft_prompt_lr_scale` to a value < 1.0 (e.g., 0.1) to start soft-prompts
    at a lower LR. Combine with a warmup scheduler for optimal results.

    Note:
        Completely matching official reported performance may require an additional
        warm-up LR schedule for soft-prompts, which can bring minor improvements.
        When `soft_prompt_warmup_lr_scale` is set, soft-prompts start at
        `lr * soft_prompt_warmup_lr_scale` and should be warmed up via the scheduler.

    Parameter Groups:
        - Group 0 (vlm): VLM parameters at lr * 0.1, weight_decay * 0.1
        - Group 1 (soft_prompts): Soft-prompt parameters at lr * soft_prompt_lr_scale
        - Group 2 (other): All other parameters at full lr
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    # Soft-prompt specific settings
    soft_prompt_lr_scale: float = 1.0  # Scale factor for soft-prompt LR (1.0 = same as base LR)
    soft_prompt_warmup_lr_scale: float | None = None  # If set, start soft-prompts at this scale (e.g., 0.01)

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with differential learning rates.

        Args:
            params: Must be a dict[str, Parameter] from dict(model.named_parameters())
                or equivalent.

        Returns:
            AdamW optimizer with parameter groups for VLM, soft-prompts, and other components

        Raises:
            AssertionError: If params is not a dict (e.g., from model.parameters())
        """
        assert isinstance(params, dict), "Custom LR optimizer requires `named_parameters()` as inputs."

        vlm_group, soft_prompt_group, other_group = [], [], []
        for name, p in params.items():
            if not p.requires_grad:
                continue
            if "vlm" in name.lower():
                vlm_group.append(p)
            elif "soft_prompt" in name.lower():
                soft_prompt_group.append(p)
            else:
                other_group.append(p)

        # Determine soft-prompt LR
        soft_prompt_lr = self.lr * self.soft_prompt_lr_scale
        if self.soft_prompt_warmup_lr_scale is not None:
            # Start at warmup scale, scheduler will warm up to soft_prompt_lr
            soft_prompt_lr = self.lr * self.soft_prompt_warmup_lr_scale

        param_groups: list[dict[str, Any]] = [
            {
                "params": vlm_group,
                "lr": self.lr * 0.1,
                "weight_decay": self.weight_decay * 0.1,
                "name": "vlm",
            },
            {
                "params": soft_prompt_group,
                "lr": soft_prompt_lr,
                "weight_decay": self.weight_decay,
                "name": "soft_prompts",
            },
            {
                "params": other_group,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "name": "other",
            },
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
        )


@OptimizerConfig.register_subclass("multi_adam")
@dataclass
class MultiAdamConfig(OptimizerConfig):
    """Configuration for multiple Adam optimizers with different parameter groups.

    This creates a dictionary of Adam optimizers, each with its own hyperparameters.

    Args:
        lr: Default learning rate (used if not specified for a group)
        weight_decay: Default weight decay (used if not specified for a group)
        optimizer_groups: Dictionary mapping parameter group names to their hyperparameters
        grad_clip_norm: Gradient clipping norm
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    optimizer_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build(self, params: OptimizerParams) -> dict[str, torch.optim.Optimizer]:
        """Build multiple Adam optimizers.

        Args:
            params: Must be a dict[str, Iterable[Parameter]] mapping parameter group names
                to iterables of parameters. The keys should match the keys in optimizer_groups.
                Typically from policies that need separate optimizers (e.g., SAC with
                actor/critic/temperature).

        Returns:
            Dictionary mapping parameter group names to their optimizers

        Raises:
            AssertionError: If params is not a dict
        """
        assert isinstance(params, dict), "MultiAdamConfig requires a dict of parameter groups as inputs."
        optimizers = {}

        for name, group_params in params.items():
            # Get group-specific hyperparameters or use defaults
            group_config = self.optimizer_groups.get(name, {})

            # Create optimizer with merged parameters (defaults + group-specific)
            optimizer_kwargs = {
                "lr": group_config.get("lr", self.lr),
                "betas": group_config.get("betas", (0.9, 0.999)),
                "eps": group_config.get("eps", 1e-5),
                "weight_decay": group_config.get("weight_decay", self.weight_decay),
            }

            optimizers[name] = torch.optim.Adam(group_params, **optimizer_kwargs)

        return optimizers


def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    """Save optimizer state to disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to save the optimizer state.
    """
    if isinstance(optimizer, dict):
        # Handle dictionary of optimizers
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        # Handle single optimizer
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    """Save a single optimizer's state to disk."""
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    save_dir: Path,
    *,
    restore_param_group_hyperparameters: bool = True,
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """Load optimizer state from disk.

    Args:
        optimizer: Either a single optimizer or a dictionary of optimizers.
        save_dir: Directory to load the optimizer state from.
        restore_param_group_hyperparameters: Whether to restore the checkpoint's
            param-group options in addition to per-parameter optimizer tensors.

    Returns:
        The updated optimizer(s) with loaded state.
    """
    if isinstance(optimizer, dict):
        # Handle dictionary of optimizers
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(
                    opt,
                    optimizer_dir,
                    restore_param_group_hyperparameters=restore_param_group_hyperparameters,
                )
            else:
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        # Handle single optimizer
        return _load_single_optimizer_state(
            optimizer,
            save_dir,
            restore_param_group_hyperparameters=restore_param_group_hyperparameters,
        )


def _load_single_optimizer_state(
    optimizer: torch.optim.Optimizer,
    save_dir: Path,
    *,
    restore_param_group_hyperparameters: bool = True,
) -> torch.optim.Optimizer:
    """Load a single optimizer's state from disk."""
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    # A run may deliberately turn FP32 masters back off. Do not retain the
    # serialized master as an unknown Adam state tensor (or let a pre-first-step
    # master-only state make ordinary AdamW expect a missing ``step`` key).
    unwrapped_optimizer = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(unwrapped_optimizer, FP32MasterAdamW) and "state" in state:
        for parameter_id, parameter_state in tuple(state["state"].items()):
            parameter_state.pop(_FP32_MASTER_STATE_KEY, None)
            if not parameter_state:
                del state["state"][parameter_id]

    # Handle case where 'state' key might not exist (for newly created optimizers)
    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        if restore_param_group_hyperparameters:
            param_groups = deserialize_json_into_object(
                save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
            )
        else:
            # Keep the freshly constructed optimizer's CLI/config values while
            # restoring Adam's per-parameter step, exp_avg and exp_avg_sq.
            # Parameter ids and group membership still come from the current
            # optimizer, so load_state_dict performs its normal shape/count
            # compatibility checks.
            param_groups = current_state_dict["param_groups"]
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
