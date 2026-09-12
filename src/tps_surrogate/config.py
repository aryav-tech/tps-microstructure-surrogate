"""YAML experiment configuration loading and CLI helpers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from tps_surrogate.paths import project_root
from tps_surrogate.schemas import (
    ConductivityConfig,
    DatasetSplitConfig,
    DiffusionConfig,
    DomainConfig,
    EvaluationConfig,
    ExperimentConfig,
    MicrostructureConfig,
    ModelHyperParams,
    OptimizationConfig,
)

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent console logging for scripts."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    root.setLevel(level)


def _as_tuple2(value: Any) -> tuple[float, float]:
    if isinstance(value, list | tuple) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"expected a length-2 list, got {value!r}")


def _as_shape(value: Any) -> tuple[int, int, int]:
    if isinstance(value, list | tuple) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"domain.shape must be [nx, ny, nz], got {value!r}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def experiment_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build a validated ``ExperimentConfig`` from a nested mapping."""
    domain = DomainConfig(
        shape=_as_shape(data["domain"]["shape"]),
        voxel_length_m=float(data["domain"]["voxel_length_m"]),
    )
    micro = data["microstructure"]
    microstructure = MicrostructureConfig(
        n_samples=int(micro["n_samples"]),
        porosity_range=_as_tuple2(micro["porosity_range"]),
        fiber_radius_voxels_range=_as_tuple2(micro["fiber_radius_voxels_range"]),
        in_plane_spread_deg_range=_as_tuple2(micro["in_plane_spread_deg_range"]),
        out_of_plane_spread_deg_range=_as_tuple2(micro["out_of_plane_spread_deg_range"]),
        porosity_tolerance=float(micro.get("porosity_tolerance", 0.10)),
        max_generation_attempts=int(micro.get("max_generation_attempts", 8)),
        n_fibers_max=int(micro.get("n_fibers_max", 120)),
        fiber_length_voxels=float(micro.get("fiber_length_voxels", 24.0)),
    )
    diff = data["diffusion"]
    diffusion = DiffusionConfig(
        n_time_steps=int(diff["n_time_steps"]),
        dt_s=float(diff["dt_s"]),
        diffusivity_m2_s_range=_as_tuple2(diff["diffusivity_m2_s_range"]),
        reaction_rate_1_s_range=_as_tuple2(diff["reaction_rate_1_s_range"]),
        z_max_bc=str(diff.get("z_max_bc", "zero_flux")),
        oxidation_threshold=float(diff.get("oxidation_threshold", 0.05)),
        enable_degradation=bool(diff.get("enable_degradation", False)),
        degradation_damage_threshold=float(diff.get("degradation_damage_threshold", 0.8)),
        degradation_max_fraction=float(diff.get("degradation_max_fraction", 0.02)),
        snapshot_times=[float(x) for x in diff.get("snapshot_times", [1.0])],
    )
    cond = data["conductivity"]
    conductivity = ConductivityConfig(
        k_solid_w_m_k=float(cond["k_solid_w_m_k"]),
        k_pore_w_m_k=float(cond["k_pore_w_m_k"]),
        solver=str(cond.get("solver", "cg")),
        tol=float(cond.get("tol", 1e-6)),
        maxiter=int(cond.get("maxiter", 400)),
        k_bounds_w_m_k=_as_tuple2(cond.get("k_bounds_w_m_k", [1e-4, 50.0])),
    )
    dset = data.get("dataset", {})
    dataset = DatasetSplitConfig(
        train_fraction=float(dset.get("train_fraction", 0.6)),
        val_fraction=float(dset.get("val_fraction", 0.2)),
        test_fraction=float(dset.get("test_fraction", 0.2)),
        n_cv_folds=int(dset.get("n_cv_folds", 5)),
        n_cv_repeats=int(dset.get("n_cv_repeats", 3)),
    )
    model_raw = data.get("model", {})
    model = ModelHyperParams(
        random_forest=dict(model_raw.get("random_forest", {})),
        hist_gbm=dict(model_raw.get("hist_gbm", {})),
    )
    opt_raw = data.get("optimization", {})
    optimization = OptimizationConfig(
        n_candidates=int(opt_raw.get("n_candidates", 32)),
        n_confirm=int(opt_raw.get("n_confirm", 5)),
        seed=int(opt_raw.get("seed", 0)),
    )
    ev_raw = data.get("evaluation", {})
    evaluation = EvaluationConfig(mape_eps=float(ev_raw.get("mape_eps", 1e-8)))
    return ExperimentConfig(
        seed=int(data["seed"]),
        name=str(data.get("name", "experiment")),
        output_dir=str(data.get("output_dir", "outputs/run")),
        domain=domain,
        microstructure=microstructure,
        diffusion=diffusion,
        conductivity=conductivity,
        dataset=dataset,
        model=model,
        optimization=optimization,
        evaluation=evaluation,
        overwrite=bool(data.get("overwrite", False)),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    data = load_yaml(path)
    config = experiment_from_dict(data)
    LOGGER.info("Loaded experiment config %s from %s", config.name, path)
    return config


def apply_overrides(
    config: ExperimentConfig,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
) -> ExperimentConfig:
    if seed is not None:
        config.seed = int(seed)
    if output_dir is not None:
        config.output_dir = str(output_dir)
    if overwrite is not None:
        config.overwrite = bool(overwrite)
    return config


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment config")
    parser.add_argument("--seed", type=int, default=None, help="override global seed")
    parser.add_argument("--output-dir", type=Path, default=None, help="override output directory")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    return parser


def config_from_cli(args: argparse.Namespace) -> ExperimentConfig:
    path = Path(args.config)
    if not path.is_absolute():
        candidate = project_root() / path
        path = candidate if candidate.is_file() else path
    config = load_config(path)
    return apply_overrides(
        config,
        seed=args.seed,
        output_dir=args.output_dir,
        overwrite=True if args.overwrite else None,
    )
