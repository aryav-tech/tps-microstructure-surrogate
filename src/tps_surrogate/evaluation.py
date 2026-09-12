"""Metrics, physical-consistency checks, and runtime break-even analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tps_surrogate.constants import K_EFF_MAX_W_M_K, K_EFF_MIN_W_M_K
from tps_surrogate.schemas import ExperimentConfig
from tps_surrogate.units import UnitSystem, conductivity_unit

LOGGER = logging.getLogger(__name__)


def physical_consistency_checks(
    predictions: pd.DataFrame,
    *,
    domain_depth_voxels: int,
    k_bounds: tuple[float, float] | None = None,
    reaction_rate: np.ndarray | None = None,
    comparable_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Flag physically implausible surrogate outputs. Never silently rewrite them.

    Returns a report with violation counts. A monotonicity check (higher
    reaction rate should not systematically yield lower oxidation damage for
    comparable microstructures) is included when rates are provided.
    """
    lo, hi = k_bounds or (K_EFF_MIN_W_M_K, K_EFF_MAX_W_M_K)
    flags: list[dict[str, Any]] = []

    if "k_eff_z" in predictions.columns:
        k = predictions["k_eff_z"].to_numpy(dtype=float)
        n_neg = int(np.sum(k < 0.0))
        n_oob = int(np.sum((k < lo) | (k > hi)))
        if n_neg:
            flags.append({"check": "nonnegative_conductivity", "n_violations": n_neg})
        if n_oob:
            flags.append(
                {
                    "check": "conductivity_bounds",
                    "n_violations": n_oob,
                    "bounds": [lo, hi],
                    "unit": conductivity_unit(),
                }
            )

    if "oxidation_penetration_depth_voxels" in predictions.columns:
        depth = predictions["oxidation_penetration_depth_voxels"].to_numpy(dtype=float)
        n_bad = int(np.sum((depth < 0.0) | (depth > float(domain_depth_voxels))))
        if n_bad:
            flags.append(
                {
                    "check": "oxidation_depth_range",
                    "n_violations": n_bad,
                    "allowed": [0, domain_depth_voxels],
                }
            )

    if "cumulative_reaction_normalized" in predictions.columns:
        cr = predictions["cumulative_reaction_normalized"].to_numpy(dtype=float)
        n_neg_r = int(np.sum(cr < 0.0))
        if n_neg_r:
            flags.append({"check": "nonnegative_cumulative_reaction", "n_violations": n_neg_r})

    monotonic = None
    if (
        reaction_rate is not None
        and "cumulative_reaction_normalized" in predictions.columns
        and len(reaction_rate) == len(predictions)
    ):
        y = predictions["cumulative_reaction_normalized"].to_numpy(dtype=float)
        x = np.asarray(reaction_rate, dtype=float)
        if comparable_mask is not None:
            x = x[comparable_mask]
            y = y[comparable_mask]
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(finite)) >= 4:
            corr, pval = spearmanr(x[finite], y[finite])
            monotonic = {"spearman_rho": float(corr), "p_value": float(pval)}
            if corr < 0.0:
                flags.append(
                    {
                        "check": "monotonic_reaction_rate_vs_damage",
                        "n_violations": 1,
                        "detail": (
                            "Spearman correlation between reaction rate and "
                            f"cumulative reaction is {corr:.3f} (expected >= 0 "
                            "for comparable microstructures)."
                        ),
                    }
                )

    report = {
        "n_flags": len(flags),
        "flags": flags,
        "monotonicity": monotonic,
        "passed": len(flags) == 0,
        "units": {
            "k_eff_z": conductivity_unit(),
            "oxidation_penetration_depth_voxels": "voxels",
        },
    }
    if flags:
        LOGGER.warning("Physical consistency flags: %s", flags)
    return report


def break_even_n(
    dataset_generation_time_s: float,
    training_time_s: float,
    baseline_time_per_case_s: float,
    inference_time_per_case_s: float,
) -> float:
    """N such that surrogate-upfront cost is recovered vs repeating the baseline.

    Returns NaN if the denominator is non-positive (inference not faster).
    """
    denom = baseline_time_per_case_s - inference_time_per_case_s
    if denom <= 0.0:
        LOGGER.warning(
            "Break-even undefined: baseline_time_per_case (%.4g s) <= inference (%.4g s)",
            baseline_time_per_case_s,
            inference_time_per_case_s,
        )
        return float("nan")
    return (dataset_generation_time_s + training_time_s) / denom


def format_cv_metric(mean: float, std: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ± {std:.{digits}g}"


def depth_in_meters(
    depth_voxels: float | np.ndarray,
    voxel_length_m: float,
) -> float | np.ndarray:
    units = UnitSystem(voxel_length_m=voxel_length_m)
    if np.isscalar(depth_voxels):
        return units.voxels_to_meters(float(depth_voxels))
    return np.asarray(depth_voxels, dtype=float) * units.dx_m


def summarize_holdout_and_cv(
    snapshot: dict[str, Any],
    target_names: list[str],
    model_name: str = "random_forest",
) -> pd.DataFrame:
    """Flatten CV mean±std and holdout point metrics into a table."""
    rows = []
    cv = snapshot.get("cv", {}).get(model_name, {})
    hold = snapshot.get("holdout", {}).get(model_name, {})
    cv_mean = cv.get("mean", {})
    cv_std = cv.get("std", {})
    for i, target in enumerate(target_names):
        row = {"model": model_name, "target": target}
        for metric in ("mae", "rmse", "r2", "mape", "max_ae"):
            means = cv_mean.get(metric, [])
            stds = cv_std.get(metric, [])
            row[f"cv_{metric}_mean"] = _index(means, i)
            row[f"cv_{metric}_std"] = _index(stds, i)
            row[f"holdout_{metric}"] = _index(hold.get(metric, []), i)
        rows.append(row)
    return pd.DataFrame(rows)


def _index(seq: Any, i: int) -> float:
    try:
        return float(seq[i])
    except (IndexError, TypeError, KeyError, ValueError):
        return float("nan")


def evaluate_predictions_frame(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run consistency checks on a predicted target table."""
    return physical_consistency_checks(
        y_pred,
        domain_depth_voxels=config.domain.shape[2],
        k_bounds=config.conductivity.k_bounds_w_m_k,
        reaction_rate=y_true["reaction_rate_1_s"].to_numpy()
        if "reaction_rate_1_s" in y_true.columns
        else None,
    )
