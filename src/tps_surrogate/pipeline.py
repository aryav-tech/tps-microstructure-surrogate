"""End-to-end orchestration used by ``run_demo`` and ``run_all``.

This helper is thin glue over dataset / surrogate / evaluation / reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tps_surrogate.dataset import generate_dataset
from tps_surrogate.evaluation import (
    physical_consistency_checks,
    summarize_holdout_and_cv,
)
from tps_surrogate.optimization import optimize_microstructure
from tps_surrogate.paths import standard_output_layout
from tps_surrogate.reporting import write_summary_report
from tps_surrogate.schemas import ExperimentConfig
from tps_surrogate.surrogate import load_surrogate, predict_with_surrogate, train_surrogates
from tps_surrogate.visualization import write_standard_figures

LOGGER = logging.getLogger(__name__)


def _require_dataset(processed: Path, overwrite: bool) -> None:
    dataset_path = processed / "dataset.csv"
    if dataset_path.is_file() and not overwrite:
        raise FileExistsError(
            f"{dataset_path} exists; pass --overwrite or set overwrite: true in the YAML"
        )


def run_generate(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    layout = standard_output_layout(output_dir)
    _require_dataset(layout["processed"], config.overwrite)
    return generate_dataset(config, output_dir=output_dir)


def run_train(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    layout = standard_output_layout(output_dir)
    dataset_path = layout["processed"] / "dataset.csv"
    manifest_path = layout["processed"] / "manifest.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing {dataset_path}; run generate_dataset.py first")
    frame = pd.read_csv(dataset_path)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_cols = list(manifest.get("feature_columns", []))
    else:
        feature_cols = [c for c in frame.columns if c not in {"sample_id", "status", "error"}]
    model_path = layout["models"] / "surrogate_random_forest.joblib"
    if model_path.is_file() and not config.overwrite:
        raise FileExistsError(f"{model_path} exists; pass --overwrite")
    return train_surrogates(config, frame, feature_cols, output_dir=layout["models"])


def run_evaluate(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    layout = standard_output_layout(output_dir)
    dataset = pd.read_csv(layout["processed"] / "dataset.csv")
    manifest = json.loads((layout["processed"] / "manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((layout["models"] / "surrogate_snapshot.json").read_text(encoding="utf-8"))
    bundle = load_surrogate(layout["models"] / "surrogate_random_forest.joblib")
    target_cols = list(snapshot["target_names"])
    test_ids = set(snapshot.get("test_ids", []))
    if "sample_id" in dataset.columns and test_ids:
        test_df = dataset[dataset["sample_id"].isin(test_ids)].copy()
    else:
        test_df = dataset.copy()
    pred = predict_with_surrogate(bundle, test_df)
    pred_df = pd.DataFrame(pred, columns=target_cols)
    y_true = test_df[target_cols].to_numpy(dtype=float)
    metrics = summarize_holdout_and_cv(snapshot, target_cols, "random_forest")
    dummy = summarize_holdout_and_cv(snapshot, target_cols, "dummy")
    hgb = summarize_holdout_and_cv(snapshot, target_cols, "hist_gbm")
    metrics = pd.concat([metrics, dummy, hgb], ignore_index=True)
    metrics.to_csv(layout["tables"] / "metrics.csv", index=False)

    consistency = physical_consistency_checks(
        pred_df,
        domain_depth_voxels=config.domain.shape[2],
        k_bounds=config.conductivity.k_bounds_w_m_k,
        reaction_rate=test_df["reaction_rate_1_s"].to_numpy()
        if "reaction_rate_1_s" in test_df.columns
        else None,
    )
    (layout["tables"] / "physical_consistency.json").write_text(
        json.dumps(consistency, indent=2), encoding="utf-8"
    )

    volume = None
    concentration = None
    damage = None
    vol_dir = layout["volumes"]
    npz_files = sorted(vol_dir.glob("sample_*.npz"))
    if npz_files:
        with np.load(npz_files[0]) as data:
            volume = data["volume"]
            concentration = data.get("concentration")
            damage = data.get("damage")

    opt_path = layout["tables"] / "optimization_candidates.csv"
    opt_frame = pd.read_csv(opt_path) if opt_path.is_file() else None

    baseline_s = float(manifest.get("mean_baseline_s") or dataset.get("baseline_total_s", pd.Series([0.0])).mean())
    infer_s = float(snapshot.get("inference_time_per_case_s") or 0.0)
    figures = write_standard_figures(
        figures_dir=layout["figures"],
        dataset=dataset,
        snapshot=snapshot,
        y_true=y_true,
        y_pred=pred,
        target_names=target_cols,
        volume=volume,
        concentration=concentration,
        damage=damage,
        opt_frame=opt_frame,
        baseline_s=baseline_s,
        inference_s=max(infer_s, 1e-9),
    )
    report = write_summary_report(
        path=layout["reports"] / "summary.md",
        config=config,
        manifest=manifest,
        snapshot=snapshot,
        metrics=metrics,
        consistency=consistency,
        n_failed=int(manifest.get("n_failed", 0)),
        baseline_time_s=baseline_s,
        inference_time_s=infer_s,
        generation_time_s=float(manifest.get("generation_time_s") or 0.0),
        training_time_s=float(sum(snapshot.get("train_time_s", {}).values())),
        figure_paths=figures,
    )
    LOGGER.info("Wrote report %s", report)
    return {"report": str(report), "consistency": consistency, "metrics_path": str(layout["tables"] / "metrics.csv")}


def run_optimize(config: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    layout = standard_output_layout(output_dir)
    dataset = pd.read_csv(layout["processed"] / "dataset.csv")
    return optimize_microstructure(
        config,
        dataset=dataset,
        model_path=layout["models"] / "surrogate_random_forest.joblib",
        output_csv=layout["tables"] / "optimization_candidates.csv",
    )


def run_pipeline(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    """Generate, train, evaluate, then optimize (demo / run_all)."""
    config.overwrite = True
    manifest = run_generate(config, output_dir)
    snapshot = run_train(config, output_dir)
    eval_info = run_evaluate(config, output_dir)
    opt = run_optimize(config, output_dir)
    # re-evaluate so the Pareto figure is included
    eval_info = run_evaluate(config, output_dir)
    return {"manifest": manifest, "snapshot": snapshot, "eval": eval_info, "n_opt": int(len(opt))}
