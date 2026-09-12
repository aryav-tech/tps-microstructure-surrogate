"""Dataset sampling helpers and skip-on-failure behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tps_surrogate.config import load_config
from tps_surrogate.dataset import (
    generate_dataset,
    generate_one_sample,
    latin_hypercube,
    sample_parameters,
)
from tps_surrogate.microstructure import DegenerateVolumeError
from tps_surrogate.paths import project_root
from tps_surrogate.seed import seed_everything


def test_latin_hypercube_stratification() -> None:
    rng = np.random.default_rng(0)
    sample = latin_hypercube(8, 3, rng)
    assert sample.shape == (8, 3)
    assert np.all(sample >= 0.0) and np.all(sample <= 1.0)
    # one sample per stratum along each axis
    for dim in range(3):
        bins = np.floor(sample[:, dim] * 8).astype(int)
        bins = np.clip(bins, 0, 7)
        assert len(set(bins.tolist())) == 8


def test_sample_parameters_count() -> None:
    config = load_config(project_root() / "configs" / "small_demo.yaml")
    rng = seed_everything(config.seed)
    rows = sample_parameters(config, rng)
    assert len(rows) == config.microstructure.n_samples
    assert rows[0]["target_porosity"] >= config.microstructure.porosity_range[0]


def test_generate_one_sample_writes_npz(tmp_path: Path) -> None:
    config = load_config(project_root() / "configs" / "small_demo.yaml")
    rng = seed_everything(1)
    params = sample_parameters(config, rng)[0]
    row = generate_one_sample(config, params, 0, rng, tmp_path)
    assert row["status"] == "ok"
    assert Path(row["volume_path"]).is_file()
    assert row["k_eff_z"] > 0.0


def test_dataset_skips_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(project_root() / "configs" / "small_demo.yaml")
    config.microstructure.n_samples = 3
    config.output_dir = str(tmp_path)
    config.overwrite = True

    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DegenerateVolumeError("injected failure")
        return generate_one_sample(*_args, **_kwargs)

    # Patch after first-failure injection by wrapping generate_one_sample
    import tps_surrogate.dataset as ds

    real = ds.generate_one_sample

    def wrapped(*args, **kwargs):
        if args[2] == 0:  # sample_id
            raise DegenerateVolumeError("injected failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(ds, "generate_one_sample", wrapped)
    manifest = generate_dataset(config, output_dir=tmp_path)
    assert manifest["n_failed"] == 1
    assert manifest["n_success"] == 2
    assert (tmp_path / "processed" / "failures.csv").is_file()
    frame = pd.read_csv(tmp_path / "processed" / "dataset.csv")
    assert len(frame) == 2
