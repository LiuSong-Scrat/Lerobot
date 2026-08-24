#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import sys
from pathlib import Path

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig, _output_dir_has_training_artifacts
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def test_empty_output_dir_has_no_training_artifacts(tmp_path: Path):
    assert not _output_dir_has_training_artifacts(tmp_path)


def test_wandb_only_output_dir_has_no_training_artifacts(tmp_path: Path):
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    (wandb_dir / "latest-run").touch()

    assert not _output_dir_has_training_artifacts(tmp_path)


def test_checkpoint_output_dir_has_training_artifacts(tmp_path: Path):
    (tmp_path / "wandb").mkdir()
    (tmp_path / "checkpoints").mkdir()

    assert _output_dir_has_training_artifacts(tmp_path)


def test_train_config_output_file_counts_as_training_artifact(tmp_path: Path):
    (tmp_path / "train_config.json").touch()

    assert _output_dir_has_training_artifacts(tmp_path)


def _write_resume_config(tmp_path: Path) -> Path:
    pretrained_dir = tmp_path / "checkpoints" / "000100" / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    policy = SmolVLAConfig(device="cpu", num_steps=10, push_to_hub=False)
    policy.save_pretrained(pretrained_dir)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/dataset"),
        policy=policy,
        output_dir=tmp_path / "run",
    )
    config.save_pretrained(pretrained_dir)
    return pretrained_dir / "train_config.json"


def test_resume_config_path_applies_nested_concrete_policy_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_resume_config(tmp_path)
    cli_args = [
        "--resume=true",
        "--policy.type=smolvla",
        "--policy.num_steps=17",
        "--steps=30100",
    ]
    monkeypatch.setattr(sys, "argv", ["train", f"--config_path={config_path}", *cli_args])

    loaded = TrainPipelineConfig.from_pretrained(config_path, cli_args=cli_args)
    loaded.validate()

    assert loaded.resume
    assert loaded.steps == 30_100
    assert isinstance(loaded.policy, SmolVLAConfig)
    assert loaded.policy.num_steps == 17
    assert loaded.policy.chunk_size == 32
    assert loaded.policy.pretrained_path == config_path.parent
    assert loaded.checkpoint_path == config_path.parent.parent


def test_resume_config_path_without_policy_override_preserves_checkpoint_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_resume_config(tmp_path)
    cli_args = ["--resume=true", "--steps=30100"]
    monkeypatch.setattr(sys, "argv", ["train", f"--config_path={config_path}", *cli_args])

    loaded = TrainPipelineConfig.from_pretrained(config_path, cli_args=cli_args)
    loaded.validate()

    assert isinstance(loaded.policy, SmolVLAConfig)
    assert loaded.policy.num_steps == 10


def test_resume_config_path_rejects_policy_type_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_resume_config(tmp_path)
    cli_args = ["--resume=true", "--policy.type=act"]
    monkeypatch.setattr(sys, "argv", ["train", f"--config_path={config_path}", *cli_args])

    with pytest.raises(ValueError, match="cannot change the checkpoint policy type"):
        TrainPipelineConfig.from_pretrained(config_path, cli_args=cli_args)


def test_resume_config_path_rejects_separate_policy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_resume_config(tmp_path)
    cli_args = ["--resume=true"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            f"--config_path={config_path}",
            "--resume=true",
            f"--policy.path={config_path.parent}",
        ],
    )
    loaded = TrainPipelineConfig.from_pretrained(config_path, cli_args=cli_args)

    with pytest.raises(ValueError, match="cannot also specify --policy.path"):
        loaded.validate()
