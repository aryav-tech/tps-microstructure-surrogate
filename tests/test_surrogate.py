"""Surrogate training, CV, and physical-consistency flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tps_surrogate.config import load_config
from tps_surrogate.evaluation import break_even_n, physical_consistency_checks
from tps_surrogate.paths import project_root
from tps_surrogate.surrogate import DEFAULT_TARGETS, predict_with_surrogate, train_surrogates


def _toy_frame(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    porosity = rng.uniform(0.7, 0.9, n)
    radius = rng.uniform(1.0, 2.0, n)
    k_rxn = rng.uniform(5.0, 40.0, n)
    deff = rng.uniform(2e-8, 8e-8, n)
    # simple monotone synthetic physics
    depth = 4.0 + 20.0 * (1.0 - porosity) + 0.05 * k_rxn + rng.normal(0.0, 0.2, n)
    k_z = 0.2 + 6.0 * (1.0 - porosity) + rng.normal(0.0, 0.05, n)
    cum = 0.1 * k_rxn * (1.0 - porosity) + rng.normal(0.0, 0.02, n)
    tort = 1.1 + 0.8 * (1.0 - porosity)
    return pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "target_porosity": porosity,
            "fiber_radius_voxels": radius,
            "in_plane_spread_deg": rng.uniform(10, 40, n),
            "out_of_plane_spread_deg": rng.uniform(5, 20, n),
            "diffusivity_m2_s": deff,
            "reaction_rate_1_s": k_rxn,
            "sim_time_s": np.full(n, 0.003),
            "realized_porosity": porosity,
            "solid_fraction": 1.0 - porosity,
            "interfacial_area_per_volume_1_m": rng.uniform(1000, 5000, n),
            "pore_connected_fraction": rng.uniform(0.7, 1.0, n),
            "continuity_x": rng.uniform(0.2, 0.6, n),
            "continuity_y": rng.uniform(0.2, 0.6, n),
            "continuity_z": rng.uniform(0.1, 0.4, n),
            "mean_pore_radius_m": rng.uniform(1e-5, 8e-5, n),
            "interfacial_face_count": rng.uniform(50, 200, n),
            "oxidation_penetration_depth_voxels": depth,
            "cumulative_reaction_normalized": np.clip(cum, 0, None),
            "k_eff_z": np.clip(k_z, 0.05, None),
            "tortuosity_proxy": tort,
        }
    )


def test_train_predict_shapes(tmp_path: Path) -> None:
    config = load_config(project_root() / "configs" / "small_demo.yaml")
    config.dataset.n_cv_folds = 3
    config.dataset.n_cv_repeats = 2
    frame = _toy_frame()
    feature_cols = [
        "target_porosity",
        "fiber_radius_voxels",
        "in_plane_spread_deg",
        "out_of_plane_spread_deg",
        "diffusivity_m2_s",
        "reaction_rate_1_s",
        "sim_time_s",
        "realized_porosity",
        "solid_fraction",
        "interfacial_area_per_volume_1_m",
        "pore_connected_fraction",
        "continuity_x",
        "continuity_y",
        "continuity_z",
        "mean_pore_radius_m",
        "interfacial_face_count",
    ]
    snapshot = train_surrogates(config, frame, feature_cols, output_dir=tmp_path)
    assert snapshot["n_samples"] == len(frame)
    assert (tmp_path / "surrogate_random_forest.joblib").is_file()
    from tps_surrogate.surrogate import load_surrogate

    bundle = load_surrogate(tmp_path / "surrogate_random_forest.joblib")
    pred = predict_with_surrogate(bundle, frame[feature_cols])
    assert pred.shape == (len(frame), len(DEFAULT_TARGETS))
    assert "random_forest" in snapshot["cv"]
    assert snapshot["cv"]["random_forest"]["n_folds_run"] > 0


def test_physical_consistency_flags_negatives() -> None:
    pred = pd.DataFrame(
        {
            "k_eff_z": [-0.1, 0.5],
            "oxidation_penetration_depth_voxels": [3.0, 99.0],
            "cumulative_reaction_normalized": [0.2, -1.0],
        }
    )
    report = physical_consistency_checks(pred, domain_depth_voxels=16, k_bounds=(0.0, 10.0))
    assert not report["passed"]
    names = {f["check"] for f in report["flags"]}
    assert "nonnegative_conductivity" in names
    assert "oxidation_depth_range" in names
    assert "nonnegative_cumulative_reaction" in names


def test_break_even_guard() -> None:
    assert np.isnan(break_even_n(10.0, 1.0, 0.1, 0.2))
    assert break_even_n(10.0, 2.0, 1.0, 0.1) == pytest.approx(12.0 / 0.9)
