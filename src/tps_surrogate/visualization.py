"""Matplotlib / seaborn figures for evaluation and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from tps_surrogate.microstructure import central_slices

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    xlabel: str,
    path: Path,
    yerr: np.ndarray | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    if yerr is not None:
        ax.errorbar(y_true, y_pred, yerr=yerr, fmt="o", alpha=0.8, ecolor="0.5")
    else:
        ax.scatter(y_true, y_pred, alpha=0.8, edgecolor="k", linewidth=0.3)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    return _save(fig, path)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    path: Path,
) -> Path:
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(y_pred, resid, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title(title)
    return _save(fig, path)


def plot_runtime_bars(
    baseline_s: float,
    inference_s: float,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    labels = ["Baseline\n(per case)", "Surrogate\ninference (per case)"]
    ax.bar(labels, [baseline_s, inference_s], color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Runtime comparison (illustrative, this machine)")
    return _save(fig, path)


def plot_feature_importance(importance: dict[str, float], path: Path, top_n: int = 12) -> Path:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(names, vals, color="#4c72b0")
    ax.set_xlabel("Relative importance")
    ax.set_title("Random-forest feature importance")
    return _save(fig, path)


def plot_pareto(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    path: Path,
    pareto_col: str = "pareto_optimal",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if pareto_col in frame.columns:
        other = frame.loc[~frame[pareto_col]]
        front = frame.loc[frame[pareto_col]]
        ax.scatter(other[x_col], other[y_col], alpha=0.6, label="dominated")
        ax.scatter(front[x_col], front[y_col], c="C3", s=70, label="Pareto front")
    else:
        ax.scatter(frame[x_col], frame[y_col], alpha=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Parameter-space Pareto scatter")
    ax.legend(fontsize=10)
    return _save(fig, path)


def plot_volume_slices(volume: npt.NDArray[np.integer], path: Path, title: str = "Central slices") -> Path:
    slices = central_slices(volume)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    keys = ("xy_at_zmid", "xz_at_ymid", "yz_at_xmid")
    labels = ("xy (z mid)", "xz (y mid)", "yz (x mid)")
    for ax, key, lab in zip(axes, keys, labels, strict=True):
        ax.imshow(slices[key].T, origin="lower", cmap="gray_r", interpolation="nearest")
        ax.set_title(lab, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)


def plot_field_slice(
    field: npt.NDArray[np.floating],
    path: Path,
    title: str,
    cmap: str = "viridis",
) -> Path:
    mid = field.shape[1] // 2
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(field[:, mid, :].T, origin="lower", cmap=cmap, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    return _save(fig, path)


def write_standard_figures(
    *,
    figures_dir: Path,
    dataset: pd.DataFrame,
    snapshot: dict[str, Any],
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    target_names: list[str],
    volume: npt.NDArray[np.integer] | None,
    concentration: npt.NDArray[np.floating] | None,
    damage: npt.NDArray[np.floating] | None,
    opt_frame: pd.DataFrame | None,
    baseline_s: float,
    inference_s: float,
) -> list[Path]:
    """Create the evaluation figure set. Missing inputs are skipped."""
    figures_dir = Path(figures_dir)
    written: list[Path] = []
    if y_true is not None and y_pred is not None:
        for i, name in enumerate(target_names):
            written.append(
                plot_parity(
                    y_true[:, i],
                    y_pred[:, i],
                    title=f"Parity: {name}",
                    xlabel=f"Baseline {name}",
                    path=figures_dir / f"parity_{name}.png",
                )
            )
            written.append(
                plot_residuals(
                    y_true[:, i],
                    y_pred[:, i],
                    title=f"Residuals: {name}",
                    path=figures_dir / f"residual_{name}.png",
                )
            )
    written.append(plot_runtime_bars(baseline_s, inference_s, figures_dir / "runtime_comparison.png"))
    importance = snapshot.get("feature_importance", {}).get("random_forest")
    if importance:
        written.append(plot_feature_importance(importance, figures_dir / "feature_importance.png"))
    if volume is not None:
        written.append(plot_volume_slices(volume, figures_dir / "example_slices.png"))
    if concentration is not None:
        written.append(
            plot_field_slice(concentration, figures_dir / "concentration_slice.png", "Oxygen concentration")
        )
    if damage is not None:
        written.append(plot_field_slice(damage, figures_dir / "damage_slice.png", "Oxidation damage", cmap="magma"))
    if opt_frame is not None and not opt_frame.empty:
        xcol = "pred_k_eff_z" if "pred_k_eff_z" in opt_frame.columns else "k_eff_z"
        ycol = (
            "pred_oxidation_penetration_depth_voxels"
            if "pred_oxidation_penetration_depth_voxels" in opt_frame.columns
            else "oxidation_penetration_depth_voxels"
        )
        if xcol in opt_frame.columns and ycol in opt_frame.columns:
            written.append(
                plot_pareto(opt_frame, x_col=xcol, y_col=ycol, path=figures_dir / "pareto_scatter.png")
            )
    _ = dataset  # reserved for future faceted plots
    return written
