"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tps_surrogate.config import experiment_from_dict, load_config
from tps_surrogate.paths import project_root


def test_load_small_demo_yaml() -> None:
    path = project_root() / "configs" / "small_demo.yaml"
    config = load_config(path)
    assert config.seed == 7
    assert config.domain.shape == (16, 16, 16)
    assert config.dataset.n_cv_folds >= 2
    assert abs(config.dataset.train_fraction + config.dataset.val_fraction + config.dataset.test_fraction - 1.0) < 1e-9


def test_load_default_yaml() -> None:
    config = load_config(project_root() / "configs" / "default.yaml")
    assert config.name == "default"
    assert config.microstructure.n_samples >= 8


def test_invalid_split_rejected() -> None:
    raw = load_config(project_root() / "configs" / "small_demo.yaml").to_dict()
    raw["dataset"]["train_fraction"] = 0.9
    raw["dataset"]["val_fraction"] = 0.9
    raw["dataset"]["test_fraction"] = 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        experiment_from_dict(raw)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("/definitely/not/a/config.yaml"))
