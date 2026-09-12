"""Markdown experiment report writer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tps_surrogate import __version__
from tps_surrogate.evaluation import break_even_n, format_cv_metric
from tps_surrogate.paths import git_commit_hash
from tps_surrogate.schemas import ExperimentConfig
from tps_surrogate.units import (
    conductivity_unit,
    diffusivity_unit,
    length_unit,
    reaction_rate_unit,
)

DISCLAIMER = """\
**Scope and non-claims.** This is an educational research prototype. It does
**not** model a complete spacecraft reentry, full hypersonic CFD, or
flight-certified heat-shield performance. Material properties and boundary
conditions are illustrative or configurable unless a cited source appears in
`docs/references.md`. The surrogate is validated **only against this project's
own numerical baseline**. Agreement with the baseline demonstrates that the
surrogate learned the simplified numerical model, **not** that the simplified
model matches physical reality or NASA flight material.
"""


def _metric_table_md(metrics: pd.DataFrame, target_names: list[str]) -> str:
    if metrics.empty:
        return "_No metrics available._\n"
    lines = [
        "| model | target | CV MAE | CV RMSE | CV R² | CV MAPE (%) | CV max AE | holdout R² |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "| {model} | {target} | {mae} | {rmse} | {r2} | {mape} | {maxae} | {hr2:.4g} |".format(
                model=row["model"],
                target=row["target"],
                mae=format_cv_metric(row.get("cv_mae_mean", float("nan")), row.get("cv_mae_std", float("nan"))),
                rmse=format_cv_metric(row.get("cv_rmse_mean", float("nan")), row.get("cv_rmse_std", float("nan"))),
                r2=format_cv_metric(row.get("cv_r2_mean", float("nan")), row.get("cv_r2_std", float("nan"))),
                mape=format_cv_metric(row.get("cv_mape_mean", float("nan")), row.get("cv_mape_std", float("nan"))),
                maxae=format_cv_metric(row.get("cv_max_ae_mean", float("nan")), row.get("cv_max_ae_std", float("nan"))),
                hr2=float(row.get("holdout_r2", float("nan"))),
            )
        )
    _ = target_names
    return "\n".join(lines) + "\n"


def write_summary_report(
    *,
    path: Path,
    config: ExperimentConfig,
    manifest: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    metrics: pd.DataFrame,
    consistency: dict[str, Any] | None,
    n_failed: int,
    baseline_time_s: float,
    inference_time_s: float,
    generation_time_s: float,
    training_time_s: float,
    figure_paths: list[Path] | None = None,
    verification_notes: str | None = None,
) -> Path:
    """Write ``outputs/reports/summary.md``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = int(manifest.get("n_success", 0)) if manifest else 0
    n_req = int(manifest.get("n_requested", config.microstructure.n_samples)) if manifest else config.microstructure.n_samples
    be = break_even_n(generation_time_s, training_time_s, baseline_time_s, inference_time_s)
    be_s = "undefined (inference not faster than baseline)" if pd.isna(be) else f"{be:.2f}"

    target_names = list((snapshot or {}).get("target_names", []))
    fig_list = "\n".join(f"- `{p}`" for p in (figure_paths or [])) or "- _(none)_"

    ver = verification_notes or (
        "See `docs/verification.md` for analytical cross-checks of the diffusion "
        "and conductivity solvers and for the grid / time-step refinement study. "
        "Those checks verify the **numerical baseline**, not physical reality."
    )

    md = f"""# Experiment summary: {config.name}

Generated: {datetime.now(UTC).isoformat()}
Code version: `{__version__}` (git `{git_commit_hash()}`)

## 1. Scope and non-claims

{DISCLAIMER}

## 2. Dataset size and split / CV scheme

- Requested samples: {n_req}
- Successful samples: {n_ok}
- Failed / skipped samples: {n_failed}
- Held-out test fraction: {config.dataset.test_fraction}
- Repeated k-fold CV: k = {config.dataset.n_cv_folds}, repeats = {config.dataset.n_cv_repeats}
- Headline accuracy is **CV mean ± standard deviation**. The single held-out
  split is reported for completeness but is **not** treated as the uncertainty
  estimate on a small dataset.

## 3. Configuration

- Global seed: `{config.seed}`
- Domain: `{config.domain.shape}` voxels, voxel length = {config.domain.voxel_length_m:.3g} {length_unit()}
- Porosity range: {config.microstructure.porosity_range}
- Fiber radius (voxels): {config.microstructure.fiber_radius_voxels_range}
- Diffusion steps: {config.diffusion.n_time_steps}, dt = {config.diffusion.dt_s:.3g} s
- Diffusivity range: {config.diffusion.diffusivity_m2_s_range} {diffusivity_unit()}
- Reaction-rate range: {config.diffusion.reaction_rate_1_s_range} {reaction_rate_unit()}
- k_solid = {config.conductivity.k_solid_w_m_k} {conductivity_unit()} (illustrative)
- k_pore = {config.conductivity.k_pore_w_m_k} {conductivity_unit()} (illustrative)
- z_max BC: `{config.diffusion.z_max_bc}`

## 4. Numerical-model assumptions

- Two-phase voxel geometry (carbon solid + pore), synthetically generated.
- Oxygen lives in pores only; first-order sink at the solid–pore interface.
- Solid geometry is frozen unless exploratory degradation is enabled
  (currently `{config.diffusion.enable_degradation}`).
- Effective conductivity from a steady finite-difference solve, not from
  radiation, contact resistance, or anisotropic fiber conductivity tensors.
- Full assumption list: `docs/model_assumptions.md`.

## 5. Analytical verification

{ver}

## 6. Accuracy metrics (CV mean ± std)

{_metric_table_md(metrics, target_names)}

Holdout metrics are single-split point values and should be read with the CV
spread, not in isolation.

## 7. Runtime metrics

| quantity | value |
|---|---|
| dataset generation wall time (s) | {generation_time_s:.4g} |
| training wall time (s) | {training_time_s:.4g} |
| baseline time per successful case (s) | {baseline_time_s:.4g} |
| surrogate inference time per case (s) | {inference_time_s:.4g} |
| break-even number of inference cases | {be_s} |

`N_break_even = (dataset_generation_time + training_time) / (baseline_time_per_case − inference_time_per_case)`.
The denominator is guarded: if inference is not faster, the value is undefined.

## 8. Failed-sample count

{n_failed} sample(s) were skipped (degenerate volumes, solver failures, or
invalid parameters) and recorded in the dataset manifest. They are **not**
silently dropped from the audit trail.

## 9. Physical-consistency results

{_consistency_md(consistency)}

Violations are flagged, not rewritten to force a pass.

## 10. Figures

{fig_list}

## 11. Limitations

- No experimental FiberForm / PICA validation in this repository.
- No PuMA comparison is required for the core pipeline.
- Small-N tabular models can overfit; CV spread is the honest uncertainty.
- Inverse design is random search in a four-parameter space, not topology
  optimization.
- Constants are illustrative unless cited.

## 12. Next steps

- Optional PuMA/pumapy comparison on shared descriptors (`docs/future_puma_integration.md`).
- Later interfaces for a 3-D CNN, U-Net, FNO, or PINO — not implemented here.
- Tighter analytical manufactured-solution tests and larger refinement studies.
- Only after those: any discussion of laboratory comparison, clearly labeled
  as future work.

---
Units used in this report are SI unless noted: length in {length_unit()},
conductivity in {conductivity_unit()}, diffusivity in {diffusivity_unit()},
reaction rate in {reaction_rate_unit()}. Conversions go through
`tps_surrogate.units`.
"""
    path.write_text(md, encoding="utf-8")
    return path


def _consistency_md(consistency: dict[str, Any] | None) -> str:
    if not consistency:
        return "_Consistency checks were not run._\n"
    if consistency.get("passed"):
        return "All configured physical-consistency checks passed on the predicted table.\n"
    lines = [f"Flagged checks: **{consistency.get('n_flags', 0)}**", ""]
    for flag in consistency.get("flags", []):
        lines.append(f"- `{flag.get('check')}`: {flag}")
    return "\n".join(lines) + "\n"
