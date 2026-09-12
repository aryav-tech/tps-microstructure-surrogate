"""Tabular physics-guided surrogates (not a PINO or neural operator).

The label "physics-guided" means inputs, targets, and post-hoc consistency
checks come from the numerical physical model. The estimators themselves are
standard scikit-learn regressors.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from tps_surrogate.schemas import ExperimentConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGETS = (
    "oxidation_penetration_depth_voxels",
    "cumulative_reaction_normalized",
    "k_eff_z",
    "tortuosity_proxy",
)


def _impute_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-impute NaNs so tree models always see a finite matrix."""
    return frame.replace([np.inf, -np.inf], np.nan).fillna(frame.median(numeric_only=True))


def build_estimators(config: ExperimentConfig, seed: int) -> dict[str, Any]:
    rf_kw = {
        "n_estimators": 80,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "random_state": seed,
        "n_jobs": 1,
    }
    rf_kw.update(config.model.random_forest)
    hgb_kw = {
        "max_depth": 5,
        "max_iter": 80,
        "learning_rate": 0.08,
        "random_state": seed,
    }
    hgb_kw.update(config.model.hist_gbm)
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(**rf_kw),
        "hist_gbm": MultiOutputRegressor(HistGradientBoostingRegressor(**hgb_kw)),
    }


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    denom = np.maximum(np.abs(y_true), eps)
    mape = np.mean(np.abs((y_true - y_pred) / denom), axis=0) * 100.0
    max_ae = np.max(np.abs(y_true - y_pred), axis=0)
    return {
        "mae": mae,
        "rmse": np.sqrt(mse),
        "r2": r2,
        "mape": mape,
        "max_ae": max_ae,
    }


def _empty_like(n_targets: int) -> dict[str, np.ndarray]:
    nan = np.full(n_targets, np.nan)
    return {"mae": nan, "rmse": nan, "r2": nan, "mape": nan, "max_ae": nan}


def cross_validate_model(
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    eps: float,
) -> dict[str, Any]:
    """Repeated k-fold CV. Returns mean and std of each metric per target."""
    n_targets = y.shape[1]
    n_usable = int(len(x))
    splits = min(n_splits, n_usable)
    if splits < 2 or n_usable < 4:
        LOGGER.warning("Too few samples (%d) for CV; reporting NaN CV metrics.", n_usable)
        return {
            "mean": _empty_like(n_targets),
            "std": _empty_like(n_targets),
            "n_folds_run": 0,
        }

    cv = RepeatedKFold(n_splits=splits, n_repeats=n_repeats, random_state=seed)
    bags: dict[str, list[np.ndarray]] = {k: [] for k in ("mae", "rmse", "r2", "mape", "max_ae")}
    n_run = 0
    for train_idx, test_idx in cv.split(x):
        if len(test_idx) < 1 or len(train_idx) < 2:
            continue
        model = _clone_fit(estimator, x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        block = _metric_block(y[test_idx], pred, eps)
        for key, value in block.items():
            bags[key].append(np.asarray(value, dtype=np.float64))
        n_run += 1
    if not bags["mae"]:
        return {"mean": _empty_like(n_targets), "std": _empty_like(n_targets), "n_folds_run": 0}
    mean = {k: np.mean(np.stack(v), axis=0) for k, v in bags.items()}
    std = {k: np.std(np.stack(v), axis=0, ddof=1) if len(v) > 1 else np.zeros(n_targets) for k, v in bags.items()}
    return {"mean": mean, "std": std, "n_folds_run": n_run}


def _clone_fit(estimator: Any, x: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.base import clone

    model = clone(estimator)
    model.fit(x, y)
    return model


def rf_tree_uncertainty(model: RandomForestRegressor, x: np.ndarray) -> np.ndarray:
    """Per-sample standard deviation across trees (multi-output if applicable)."""
    tree_preds = np.stack([tree.predict(x) for tree in model.estimators_], axis=0)
    return np.std(tree_preds, axis=0)


def feature_importance(model: Any, feature_names: list[str]) -> dict[str, float] | None:
    raw = getattr(model, "feature_importances_", None)
    if raw is None and hasattr(model, "estimators_"):
        # MultiOutputRegressor of trees
        imps = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imps.append(np.asarray(est.feature_importances_, dtype=np.float64))
        if imps:
            raw = np.mean(np.stack(imps), axis=0)
    if raw is None:
        return None
    return {name: float(val) for name, val in zip(feature_names, raw, strict=False)}


def prepare_xy(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in feature_cols + target_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"dataset missing columns: {missing}")
    features = _impute_features(frame[feature_cols].apply(pd.to_numeric, errors="coerce"))
    targets = frame[target_cols].apply(pd.to_numeric, errors="coerce")
    mask = targets.notna().all(axis=1) & features.notna().all(axis=1)
    return features.loc[mask], targets.loc[mask]


def train_surrogates(
    config: ExperimentConfig,
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str] | None = None,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit Dummy / RF / HGB models, run repeated CV, persist the RF as primary."""
    target_cols = list(target_cols or DEFAULT_TARGETS)
    features, targets = prepare_xy(dataset, feature_cols, target_cols)
    if len(features) < 4:
        raise ValueError(f"need at least 4 complete samples to train, got {len(features)}")

    x = features.to_numpy(dtype=np.float64)
    y = targets.to_numpy(dtype=np.float64)
    ids = (
        dataset.loc[features.index, "sample_id"].to_numpy()
        if "sample_id" in dataset.columns
        else np.arange(len(features))
    )

    test_frac = config.dataset.test_fraction
    if len(features) * test_frac < 1:
        test_frac = max(1 / len(features), 0.2)
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x,
        y,
        ids,
        test_size=test_frac,
        random_state=config.seed,
    )

    estimators = build_estimators(config, config.seed)
    eps = config.evaluation.mape_eps
    cv_report: dict[str, Any] = {}
    holdout: dict[str, Any] = {}
    trained: dict[str, Any] = {}
    importances: dict[str, Any] = {}
    train_time: dict[str, float] = {}

    for name, est in estimators.items():
        t0 = time.perf_counter()
        cv_report[name] = cross_validate_model(
            est,
            x,
            y,
            n_splits=config.dataset.n_cv_folds,
            n_repeats=config.dataset.n_cv_repeats,
            seed=config.seed,
            eps=eps,
        )
        model = _clone_fit(est, x_train, y_train)
        trained[name] = model
        train_time[name] = time.perf_counter() - t0
        pred = model.predict(x_test)
        holdout[name] = _metric_block(y_test, pred, eps)
        imp = feature_importance(model, feature_cols)
        if imp is None and name == "random_forest":
            imp = feature_importance(model, feature_cols)
        if imp:
            importances[name] = imp
        LOGGER.info("Trained %s in %.2f s", name, train_time[name])

    t_inf0 = time.perf_counter()
    _ = trained["random_forest"].predict(x_test)
    infer_s = (time.perf_counter() - t_inf0) / max(len(x_test), 1)

    uncertainty = None
    rf_model = trained["random_forest"]
    if isinstance(rf_model, RandomForestRegressor):
        uncertainty = rf_tree_uncertainty(rf_model, x_test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "surrogate_random_forest.joblib"
    joblib.dump(
        {
            "model": rf_model,
            "feature_names": feature_cols,
            "target_names": target_cols,
            "all_models": trained,
        },
        model_path,
    )

    def _serialize_metric_dict(block: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in block.items():
            if isinstance(value, dict):
                out[key] = {
                    metric: list(np.asarray(arr, dtype=float)) for metric, arr in value.items()
                }
            else:
                out[key] = value
        return out

    snapshot = {
        "config": config.to_dict(),
        "feature_names": feature_cols,
        "target_names": target_cols,
        "n_samples": int(len(features)),
        "train_ids": [int(v) for v in id_train],
        "test_ids": [int(v) for v in id_test],
        "cv": {name: _serialize_metric_dict(val) for name, val in cv_report.items()},
        "holdout": {
            name: {k: list(np.asarray(v, dtype=float)) for k, v in block.items()}
            for name, block in holdout.items()
        },
        "feature_importance": importances,
        "train_time_s": train_time,
        "inference_time_per_case_s": infer_s,
        "model_path": str(model_path),
        "rf_test_uncertainty_std": None
        if uncertainty is None
        else np.asarray(uncertainty, dtype=float).tolist(),
    }
    snap_path = output_dir / "surrogate_snapshot.json"
    snap_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def load_surrogate(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if "model" not in payload:
        raise KeyError(f"{path} is not a surrogate bundle")
    return payload


def predict_with_surrogate(
    bundle: dict[str, Any],
    features: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    model = bundle["model"]
    names = bundle["feature_names"]
    if isinstance(features, pd.DataFrame):
        matrix = _impute_features(features[names]).to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(features, dtype=np.float64)
    return np.asarray(model.predict(matrix), dtype=np.float64)
