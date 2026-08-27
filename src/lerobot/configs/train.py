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
import builtins
import datetime as dt
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"

# W&B creates this directory as soon as a run is initialized. In distributed
# launches it may therefore be visible to one process while another process is
# still validating the training configuration. It is logger-owned and does not
# indicate that model outputs or checkpoints would be overwritten.
_NON_TRAINING_OUTPUT_ENTRIES = frozenset({"wandb"})


def _output_dir_has_training_artifacts(output_dir: Path) -> bool:
    """Return whether an output directory contains anything besides logger files."""
    try:
        return any(entry.name not in _NON_TRAINING_OUTPUT_ENTRIES for entry in output_dir.iterdir())
    except FileNotFoundError:
        # The directory may be removed concurrently between is_dir() and
        # iterdir(), especially on a shared filesystem during cluster startup.
        return False


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # Optional optimizer-state-only resume mode. Policy weights, RNG, global
    # step and Adam's per-parameter tensors (step/exp_avg/exp_avg_sq) are
    # restored. Param-group hyperparameters remain those constructed from the
    # current config, the old scheduler is ignored, and a phase-relative cosine
    # schedule sets lr/initial_lr. Defaults preserve historical full resume.
    resume_restart_scheduler: bool = False
    resume_scheduler_start_lr: float | None = None
    resume_scheduler_end_lr: float | None = None
    resume_scheduler_decay_steps: int | None = None
    # Persisted into checkpoints created by the restarted phase. It is filled
    # from the first restored global step when omitted, allowing an interrupted
    # restarted phase to reconstruct the same scheduler position later.
    resume_scheduler_phase_start_step: int | None = None
    # One-shot Adam-state ablation that still restores the checkpoint policy,
    # global step and RNG state.  The first resumed checkpoint records the
    # restored step below; later resumes from that phase do not reset the
    # newly accumulated moments a second time.
    resume_reset_optimizer_moments: bool = False
    resume_optimizer_moments_reset_step: int | None = None
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    # Opt-in capacity diagnostic: cache the first complete batch and train on
    # that exact batch with a restored RNG state before every forward pass.
    # Defaults leave the production data and stochastic-flow paths unchanged.
    diagnostic_repeat_first_batch: bool = False
    diagnostic_fixed_batch_seed: int = 20260827
    diagnostic_fixed_forward_seed: int = 20260828
    # When false, keep the exact cached batch but let flow time/noise and point
    # sampling advance exactly as in production training. The default retains
    # the original single-target capacity diagnostic.
    diagnostic_repeat_forward_rng: bool = True
    batch_size: int = 8
    # Match an equal-weight task benchmark by drawing the same number of valid
    # frames from each task per dataloader epoch. Disabled by default to retain
    # the historical global-uniform-over-frames training distribution.
    task_balanced_sampling: bool = False
    # Optional exact number of samples contributing gradients to each optimizer
    # update across all distributed ranks. The training entrypoint validates
    # that this can be represented by its configured accumulation schedule.
    global_batch_size: int | None = None
    # Number of micro-batches accumulated before one optimizer update.  ``steps``
    # continues to count optimizer updates, so enabling accumulation does not
    # silently change checkpoint or scheduler semantics.
    gradient_accumulation_steps: int = 1
    steps: int = 100_000
    pointseg_sample_cache_dir: str = ""
    eval_freq: int = 20_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # RA-BC (Reward-Aligned Behavior Cloning) parameters
    use_rabc: bool = False  # Enable reward-weighted training
    rabc_progress_path: str | None = None  # Path to precomputed SARM progress parquet file
    rabc_kappa: float = 0.01  # Hard threshold for high-quality samples
    rabc_epsilon: float = 1e-6  # Small constant for numerical stability
    rabc_head_mode: str | None = "sparse"  # For dual-head models: "sparse" or "dense"

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    def validate(self) -> None:
        if self.resume and parser.get_path_arg("policy"):
            raise ValueError(
                "A config_path resume cannot also specify --policy.path; the policy must be "
                "restored from the same checkpoint as its optimizer and training state."
            )
        if self.resume_restart_scheduler:
            if not self.resume:
                raise ValueError("resume_restart_scheduler=true requires resume=true.")
            restart_values = {
                "resume_scheduler_start_lr": self.resume_scheduler_start_lr,
                "resume_scheduler_end_lr": self.resume_scheduler_end_lr,
                "resume_scheduler_decay_steps": self.resume_scheduler_decay_steps,
            }
            missing = [name for name, value in restart_values.items() if value is None]
            if missing:
                raise ValueError(
                    "Restarting the resume scheduler requires explicit values for "
                    f"{missing}."
                )
            start_lr = float(self.resume_scheduler_start_lr)
            end_lr = float(self.resume_scheduler_end_lr)
            decay_steps = self.resume_scheduler_decay_steps
            if not math.isfinite(start_lr) or start_lr <= 0.0:
                raise ValueError(
                    "resume_scheduler_start_lr must be finite and positive, "
                    f"got {self.resume_scheduler_start_lr!r}."
                )
            if not math.isfinite(end_lr) or end_lr < 0.0 or end_lr > start_lr:
                raise ValueError(
                    "resume_scheduler_end_lr must be finite and in [0, start_lr], "
                    f"got {self.resume_scheduler_end_lr!r}."
                )
            if (
                isinstance(decay_steps, bool)
                or not isinstance(decay_steps, int)
                or decay_steps < 1
            ):
                raise ValueError(
                    "resume_scheduler_decay_steps must be a positive integer, "
                    f"got {decay_steps!r}."
                )
            phase_start = self.resume_scheduler_phase_start_step
            if phase_start is not None and (
                isinstance(phase_start, bool)
                or not isinstance(phase_start, int)
                or phase_start < 0
            ):
                raise ValueError(
                    "resume_scheduler_phase_start_step must be a non-negative integer or None, "
                    f"got {phase_start!r}."
                )
        if self.resume_reset_optimizer_moments and not self.resume:
            raise ValueError("resume_reset_optimizer_moments=true requires resume=true.")
        reset_step = self.resume_optimizer_moments_reset_step
        if reset_step is not None and (
            isinstance(reset_step, bool) or not isinstance(reset_step, int) or reset_step < 0
        ):
            raise ValueError(
                "resume_optimizer_moments_reset_step must be a non-negative integer or None, "
                f"got {reset_step!r}."
            )
        if int(self.gradient_accumulation_steps) < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be at least 1, got {self.gradient_accumulation_steps}."
            )
        if self.diagnostic_repeat_first_batch:
            for name in ("diagnostic_fixed_batch_seed", "diagnostic_fixed_forward_seed"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
            if int(self.gradient_accumulation_steps) != 1:
                raise ValueError(
                    "diagnostic_repeat_first_batch requires gradient_accumulation_steps=1."
                )
        if self.global_batch_size is not None and (
            isinstance(self.global_batch_size, bool)
            or not isinstance(self.global_batch_size, int)
            or self.global_batch_size < 1
        ):
            raise ValueError(
                f"global_batch_size must be a positive integer or None, got {self.global_batch_size!r}."
            )
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            self.checkpoint_path = policy_dir.parent

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with `--policy.path`."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if (
            not self.resume
            and isinstance(self.output_dir, Path)
            and self.output_dir.is_dir()
            and _output_dir_has_training_artifacts(self.output_dir)
        ):
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten. "
                "An empty directory or a directory containing only W&B logs is safe and is accepted."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

        if self.use_rabc and not self.rabc_progress_path:
            # Auto-detect from dataset path
            repo_id = self.dataset.repo_id
            if self.dataset.root:
                self.rabc_progress_path = str(Path(self.dataset.root) / "sarm_progress.parquet")
            else:
                self.rabc_progress_path = f"hf://datasets/{repo_id}/sarm_progress.parquet"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        # Draccus builds the CLI parser from the annotated base policy type
        # before it decodes the concrete policy choice stored in
        # train_config.json. Consequently, nested ``--policy.*`` fields from a
        # resume command are otherwise rejected as unknown. Parse the training
        # config without those arguments, then apply the de-nested overrides to
        # the concrete policy config next to the checkpoint.
        policy_cli_overrides = parser.get_cli_overrides("policy", cli_args)
        requested_policy_type = parser.get_type_arg("policy", cli_args)
        has_policy_cli = bool(policy_cli_overrides) or requested_policy_type is not None
        train_cli_args = (
            [arg for arg in cli_args if not arg.startswith("--policy.")]
            if has_policy_cli
            else cli_args
        )
        with draccus.config_type("json"):
            config = draccus.parse(cls, config_file, args=train_cli_args)

        if requested_policy_type is not None and requested_policy_type != config.policy.type:
            raise ValueError(
                "A config_path resume cannot change the checkpoint policy type: "
                f"checkpoint={config.policy.type!r}, requested={requested_policy_type!r}."
            )
        if policy_cli_overrides:
            policy_dir = Path(config_file).parent
            config.policy = PreTrainedConfig.from_pretrained(
                policy_dir,
                cli_overrides=policy_cli_overrides,
            )
            config.policy.pretrained_path = policy_dir
        return config


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional
