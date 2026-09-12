"""Parameter-space inverse design (not voxel-level topology optimization).

A random search over (porosity, fiber radius, in-plane spread, out-of-plane
spread) scores candidates with the trained surrogate, then re-runs the
numerical baseline on the top few designs for confirmation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tps_surrogate.conductivity import run_from_config as run_conductivity
from tps_surrogate.descriptors import compute_descriptors, validate_descriptor_dict
from tps_surrogate.diffusion_reaction import run_from_config as run_diffusion
from tps_surrogate.microstructure import DegenerateVolumeError, generate_fiber_volume
from tps_surrogate.schemas import ExperimentConfig
from tps_surrogate.seed import seed_everything, spawn_seed
from tps_surrogate.surrogate import load_surrogate, predict_with_surrogate

LOGGER = logging.getLogger(__name__)


def pareto_mask(k_eff_z: np.ndarray, oxidation_depth: np.ndarray) -> np.ndarray:
    """True for points that are non-dominated for simultaneous minimization."""
    n = len(k_eff_z)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = (
            (k_eff_z <= k_eff_z[i] + 1e-15)
            & (oxidation_depth <= oxidation_depth[i] + 1e-15)
            & ((k_eff_z < k_eff_z[i] - 1e-15) | (oxidation_depth < oxidation_depth[i] - 1e-15))
        )
        if np.any(dominated):
            mask[i] = False
    return mask


def _median_process_params(dataset: pd.DataFrame) -> dict[str, float]:
    keys = ["diffusivity_m2_s", "reaction_rate_1_s", "sim_time_s"]
    out: dict[str, float] = {}
    for key in keys:
        if key in dataset.columns:
            out[key] = float(dataset[key].median())
    return out


def _candidate_feature_row(
    config: ExperimentConfig,
    params: dict[str, float],
    descriptors: dict[str, float],
    process: dict[str, float],
    feature_names: list[str],
) -> dict[str, float]:
    row = {
        "target_porosity": params["target_porosity"],
        "fiber_radius_voxels": params["fiber_radius_voxels"],
        "in_plane_spread_deg": params["in_plane_spread_deg"],
        "out_of_plane_spread_deg": params["out_of_plane_spread_deg"],
        "diffusivity_m2_s": process.get("diffusivity_m2_s", float(np.mean(config.diffusion.diffusivity_m2_s_range))),
        "reaction_rate_1_s": process.get("reaction_rate_1_s", float(np.mean(config.diffusion.reaction_rate_1_s_range))),
        "sim_time_s": process.get("sim_time_s", config.diffusion.n_time_steps * config.diffusion.dt_s),
        "realized_porosity": descriptors["porosity"],
        **descriptors,
    }
    return {name: float(row.get(name, 0.0)) for name in feature_names}


def random_search_candidates(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n: int,
) -> list[dict[str, float]]:
    micro = config.microstructure
    rows = []
    for _ in range(n):
        rows.append(
            {
                "target_porosity": float(rng.uniform(*micro.porosity_range)),
                "fiber_radius_voxels": float(rng.uniform(*micro.fiber_radius_voxels_range)),
                "in_plane_spread_deg": float(rng.uniform(*micro.in_plane_spread_deg_range)),
                "out_of_plane_spread_deg": float(rng.uniform(*micro.out_of_plane_spread_deg_range)),
            }
        )
    return rows


def optimize_microstructure(
    config: ExperimentConfig,
    *,
    dataset: pd.DataFrame,
    model_path: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Surrogate-scored random search with baseline confirmation of the top designs.

    This searches a **parameterized design space**, not arbitrary voxel topology.
    """
    rng = seed_everything(config.optimization.seed or config.seed)
    bundle = load_surrogate(model_path)
    feature_names = list(bundle["feature_names"])
    target_names = list(bundle["target_names"])
    process = _median_process_params(dataset)
    candidates = random_search_candidates(config, rng, config.optimization.n_candidates)

    records: list[dict[str, Any]] = []
    for i, params in enumerate(candidates):
        seed = spawn_seed(rng)
        try:
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
        except (DegenerateVolumeError, ValueError) as exc:
            LOGGER.warning("optimization candidate %d skipped: %s", i, exc)
            continue
        feat = _candidate_feature_row(config, params, descriptors, process, feature_names)
        pred = predict_with_surrogate(bundle, pd.DataFrame([feat]))[0]
        rec = {
            "candidate_id": i,
            "seed": seed,
            **params,
            **{f"pred_{name}": float(pred[j]) for j, name in enumerate(target_names)},
        }
        records.append(rec)

    if not records:
        raise RuntimeError("all optimization candidates failed volume generation")

    frame = pd.DataFrame(records)
    depth_col = "pred_oxidation_penetration_depth_voxels"
    k_col = "pred_k_eff_z"
    if depth_col not in frame.columns or k_col not in frame.columns:
        raise KeyError("surrogate targets must include oxidation depth and k_eff_z")

    # rank by sum of min-max normalized objectives
    k_n = (frame[k_col] - frame[k_col].min()) / (frame[k_col].max() - frame[k_col].min() + 1e-15)
    d_n = (frame[depth_col] - frame[depth_col].min()) / (
        frame[depth_col].max() - frame[depth_col].min() + 1e-15
    )
    frame["scalarized_score"] = k_n + d_n
    frame = frame.sort_values("scalarized_score").reset_index(drop=True)

    n_confirm = min(config.optimization.n_confirm, len(frame))
    confirm_idx = list(range(n_confirm))
    for col in (
        "base_oxidation_penetration_depth_voxels",
        "base_k_eff_z",
        "base_cumulative_reaction_normalized",
        "abs_err_depth",
        "abs_err_k_eff_z",
        "confirm_time_s",
    ):
        frame[col] = np.nan

    for rank in confirm_idx:
        row = frame.iloc[rank]
        t0 = time.perf_counter()
        try:
            volume = generate_fiber_volume(
                shape=config.domain.shape,
                target_porosity=float(row["target_porosity"]),
                fiber_radius_voxels=float(row["fiber_radius_voxels"]),
                in_plane_spread_deg=float(row["in_plane_spread_deg"]),
                out_of_plane_spread_deg=float(row["out_of_plane_spread_deg"]),
                seed=int(row["seed"]),
                n_fibers_max=config.microstructure.n_fibers_max,
                fiber_length_voxels=config.microstructure.fiber_length_voxels,
                porosity_tolerance=config.microstructure.porosity_tolerance,
                max_attempts=config.microstructure.max_generation_attempts,
            )
            sim = run_diffusion(
                volume,
                config.diffusion,
                config.domain.voxel_length_m,
                process.get("diffusivity_m2_s", float(np.mean(config.diffusion.diffusivity_m2_s_range))),
                process.get("reaction_rate_1_s", float(np.mean(config.diffusion.reaction_rate_1_s_range))),
            )
            cond = run_conductivity(volume, config.conductivity, config.domain.voxel_length_m)
            frame.at[rank, "base_oxidation_penetration_depth_voxels"] = sim.oxidation_penetration_depth_voxels
            frame.at[rank, "base_k_eff_z"] = cond.k_eff_z
            frame.at[rank, "base_cumulative_reaction_normalized"] = sim.cumulative_reaction_normalized
            frame.at[rank, "abs_err_depth"] = abs(
                sim.oxidation_penetration_depth_voxels - float(row[depth_col])
            )
            frame.at[rank, "abs_err_k_eff_z"] = abs(cond.k_eff_z - float(row[k_col]))
        except (DegenerateVolumeError, ValueError, RuntimeError) as exc:
            LOGGER.warning("confirmation of candidate rank %d failed: %s", rank, exc)
        frame.at[rank, "confirm_time_s"] = time.perf_counter() - t0

    front = pareto_mask(frame[k_col].to_numpy(), frame[depth_col].to_numpy())
    frame["pareto_optimal"] = front
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    LOGGER.info("Wrote optimization table to %s (%d candidates)", output_csv, len(frame))
    return frame
