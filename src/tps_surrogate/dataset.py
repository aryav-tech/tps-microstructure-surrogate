"""Dataset generation: LHS sampling, baselines, CSV/NPZ/manifest writers."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from tps_surrogate import __version__
from tps_surrogate.conductivity import run_from_config as run_conductivity
from tps_surrogate.descriptors import compute_descriptors, validate_descriptor_dict
from tps_surrogate.diffusion_reaction import run_from_config as run_diffusion
from tps_surrogate.microstructure import (
    DegenerateVolumeError,
    compute_porosity,
    generate_fiber_volume,
    save_volume_npz,
)
from tps_surrogate.paths import git_commit_hash, standard_output_layout
from tps_surrogate.schemas import ExperimentConfig
from tps_surrogate.seed import seed_everything, spawn_seed

LOGGER = logging.getLogger(__name__)

FEATURE_PARAM_KEYS = (
    "target_porosity",
    "fiber_radius_voxels",
    "in_plane_spread_deg",
    "out_of_plane_spread_deg",
    "diffusivity_m2_s",
    "reaction_rate_1_s",
    "sim_time_s",
)

TARGET_KEYS = (
    "oxidation_penetration_depth_voxels",
    "oxidation_penetration_depth_m",
    "cumulative_reaction_normalized",
    "k_eff_z",
    "k_eff_x",
    "k_eff_y",
    "tortuosity_proxy",
    "porosity_change_proxy",
)


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin-hypercube sample in the unit hypercube, shape (n_samples, n_dims)."""
    if n_samples < 1 or n_dims < 1:
        raise ValueError("n_samples and n_dims must be >= 1")
    result = np.empty((n_samples, n_dims), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_samples + 1)
    for dim in range(n_dims):
        lows, highs = edges[:-1], edges[1:]
        result[:, dim] = rng.uniform(lows, highs)
        rng.shuffle(result[:, dim])
    return result


def _lerp(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


def sample_parameters(config: ExperimentConfig, rng: np.random.Generator) -> list[dict[str, float]]:
    """Draw LHS parameter dictionaries for one experiment."""
    n = config.microstructure.n_samples
    unit = latin_hypercube(n, 7, rng)
    rows: list[dict[str, float]] = []
    micro = config.microstructure
    diff = config.diffusion
    sim_time_s = diff.n_time_steps * diff.dt_s
    # sim_time is nearly fixed; allow a small stratified scale in [0.7, 1.0]
    for i in range(n):
        rows.append(
            {
                "target_porosity": _lerp(unit[i, 0], *micro.porosity_range),
                "fiber_radius_voxels": _lerp(unit[i, 1], *micro.fiber_radius_voxels_range),
                "in_plane_spread_deg": _lerp(unit[i, 2], *micro.in_plane_spread_deg_range),
                "out_of_plane_spread_deg": _lerp(unit[i, 3], *micro.out_of_plane_spread_deg_range),
                "diffusivity_m2_s": _lerp(unit[i, 4], *diff.diffusivity_m2_s_range),
                "reaction_rate_1_s": _lerp(unit[i, 5], *diff.reaction_rate_1_s_range),
                "sim_time_s": sim_time_s * (0.7 + 0.3 * unit[i, 6]),
            }
        )
    return rows


def _sample_time_steps(config: ExperimentConfig, sim_time_s: float) -> int:
    steps = int(round(sim_time_s / config.diffusion.dt_s))
    return max(steps, 1)


def generate_one_sample(
    config: ExperimentConfig,
    params: dict[str, float],
    sample_id: int,
    rng: np.random.Generator,
    volume_dir: Path,
) -> dict[str, Any]:
    """Generate one volume, descriptors, and baseline targets."""
    seed = spawn_seed(rng)
    volume = generate_fiber_volume(
        shape=config.domain.shape,
        target_porosity=params["target_porosity"],
        fiber_radius_voxels=params["fiber_radius_voxels"],
        in_plane_spread_deg=params["in_plane_spread_deg"],
        out_of_plane_spread_deg=params["out_of_plane_spread_deg"],
        seed=seed,
        n_fibers_max=config.microstructure.n_fibers_max,
        fiber_length_voxels=config.microstructure.fiber_length_voxels,
        porosity_tolerance=config.microstructure.porosity_tolerance,
        max_attempts=config.microstructure.max_generation_attempts,
    )
    descriptors = validate_descriptor_dict(
        compute_descriptors(volume, config.domain.voxel_length_m)
    )
    n_steps = _sample_time_steps(config, params["sim_time_s"])
    t0 = time.perf_counter()
    if n_steps == config.diffusion.n_time_steps:
        sim = run_diffusion(
            volume,
            config.diffusion,
            config.domain.voxel_length_m,
            params["diffusivity_m2_s"],
            params["reaction_rate_1_s"],
            rng=rng,
        )
    else:
        from tps_surrogate.diffusion_reaction import run_diffusion_reaction

        sim = run_diffusion_reaction(
            volume,
            voxel_length_m=config.domain.voxel_length_m,
            n_time_steps=n_steps,
            dt_s=config.diffusion.dt_s,
            diffusivity_m2_s=params["diffusivity_m2_s"],
            reaction_rate_1_s=params["reaction_rate_1_s"],
            z_max_bc=config.diffusion.z_max_bc,
            oxidation_threshold=config.diffusion.oxidation_threshold,
            enable_degradation=config.diffusion.enable_degradation,
            snapshot_times=config.diffusion.snapshot_times,
            rng=rng,
        )
    t_diff = time.perf_counter() - t0
    t1 = time.perf_counter()
    cond = run_conductivity(volume, config.conductivity, config.domain.voxel_length_m)
    t_cond = time.perf_counter() - t1

    volume_path = volume_dir / f"sample_{sample_id:04d}.npz"
    save_volume_npz(
        volume_path,
        volume,
        concentration=sim.concentration,
        cumulative_reaction=sim.cumulative_reaction,
        damage=sim.damage,
        oxidation_profile=sim.oxidation_profile,
    )
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "seed": seed,
        "status": "ok",
        "error": "",
        "volume_path": str(volume_path),
        "realized_porosity": compute_porosity(volume),
        "baseline_diffusion_s": t_diff,
        "baseline_conductivity_s": t_cond,
        "baseline_total_s": t_diff + t_cond,
        "n_time_steps_used": n_steps,
        **params,
        **descriptors,
        "oxidation_penetration_depth_voxels": sim.oxidation_penetration_depth_voxels,
        "oxidation_penetration_depth_m": sim.oxidation_penetration_depth_m,
        "cumulative_reaction_normalized": sim.cumulative_reaction_normalized,
        "k_eff_z": cond.k_eff_z,
        "k_eff_x": cond.k_eff_x,
        "k_eff_y": cond.k_eff_y,
        "anisotropy_xz": cond.anisotropy_xz,
        "porosity_change_proxy": sim.porosity_change_proxy,
        "conductivity_fallback": int(cond.used_fallback),
    }
    return row


def generate_dataset(
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full sampling + baseline pipeline and write artifacts."""
    rng = seed_everything(config.seed)
    out = Path(output_dir) if output_dir is not None else None
    if out is None:
        from tps_surrogate.paths import resolve_output_dir

        out = resolve_output_dir(config.output_dir)
    layout = standard_output_layout(out)
    processed = layout["processed"]
    volume_dir = layout["volumes"]

    params_list = sample_parameters(config, rng)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Generating dataset", total=len(params_list))
        for i, params in enumerate(params_list):
            try:
                row = generate_one_sample(config, params, i, rng, volume_dir)
                rows.append(row)
                LOGGER.info("sample %d ok porosity=%.3f k_z=%.4g", i, row["realized_porosity"], row["k_eff_z"])
            except (DegenerateVolumeError, ValueError, RuntimeError) as exc:
                failure = {
                    "sample_id": i,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    **params,
                }
                failures.append(failure)
                LOGGER.warning("sample %d failed: %s", i, exc)
            progress.advance(task)

    elapsed = time.perf_counter() - t_start
    combined = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failures) if failures else pd.DataFrame()

    feature_cols = [c for c in FEATURE_PARAM_KEYS if c in combined.columns]
    # keep realized porosity and geometric descriptors as features
    extra_features = [
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
    feature_cols = feature_cols + [c for c in extra_features if c in combined.columns]
    target_cols = [c for c in TARGET_KEYS if c in combined.columns]

    features_path = processed / "features.csv"
    targets_path = processed / "targets.csv"
    combined_path = processed / "dataset.csv"
    failures_path = processed / "failures.csv"
    if not combined.empty:
        combined[feature_cols].to_csv(features_path, index=False)
        combined[target_cols].to_csv(targets_path, index=False)
        combined.to_csv(combined_path, index=False)
    else:
        pd.DataFrame().to_csv(features_path, index=False)
        pd.DataFrame().to_csv(targets_path, index=False)
        pd.DataFrame().to_csv(combined_path, index=False)
    if not failed_df.empty:
        failed_df.to_csv(failures_path, index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "git_commit": git_commit_hash(),
        "seed": config.seed,
        "config": config.to_dict(),
        "n_requested": config.microstructure.n_samples,
        "n_success": int(len(rows)),
        "n_failed": int(len(failures)),
        "generation_time_s": elapsed,
        "mean_baseline_s": float(combined["baseline_total_s"].mean()) if not combined.empty else None,
        "paths": {
            "features": str(features_path),
            "targets": str(targets_path),
            "dataset": str(combined_path),
            "failures": str(failures_path) if not failed_df.empty else None,
            "volumes": str(volume_dir),
        },
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "failed_samples": failures,
    }
    manifest_path = processed / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Dataset complete: %d ok, %d failed, %.2f s",
        len(rows),
        len(failures),
        elapsed,
    )
    return manifest
