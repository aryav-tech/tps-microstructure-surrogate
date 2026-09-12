"""Explicit finite-difference oxygen diffusion–reaction baseline.

Model (pore voxels only)::

    dC/dt = Deff * Laplacian(C) - k_reaction * C * interface_mask

Boundary conditions:
    * C = 1.0 on pore voxels at z = 0
    * zero-flux on x and y faces
    * configurable condition at z = max (zero_flux | fixed_zero | fixed_one)

Reaction occurs only in pore voxels that share a face with carbon solid.
The first-version geometry is frozen unless ``enable_degradation`` is set;
that optional mode is documented as exploratory.

All properties are illustrative unless cited in ``docs/references.md``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

from tps_surrogate.constants import (
    EXPLICIT_DIFFUSION_FO_LIMIT_3D,
    FO_SAFETY_FACTOR,
    PORE_LABEL,
    SOLID_LABEL,
)
from tps_surrogate.schemas import DiffusionConfig, SimulationResult
from tps_surrogate.units import UnitSystem

LOGGER = logging.getLogger(__name__)

ZMaxBC = Literal["zero_flux", "fixed_zero", "fixed_one"]


class DiffusionStabilityError(ValueError):
    """Raised when the explicit scheme violates the 3-D Fourier-number limit."""


def max_stable_dt(diffusivity_m2_s: float, voxel_length_m: float) -> float:
    """Largest dt that satisfies Fo < 1/6 with a safety factor."""
    if diffusivity_m2_s <= 0.0:
        raise ValueError("diffusivity must be positive")
    dx2 = voxel_length_m**2
    return FO_SAFETY_FACTOR * EXPLICIT_DIFFUSION_FO_LIMIT_3D * dx2 / diffusivity_m2_s


def check_explicit_stability(dt_s: float, diffusivity_m2_s: float, voxel_length_m: float) -> float:
    """Return the Fourier number, or raise if the step is too large."""
    fo = diffusivity_m2_s * dt_s / (voxel_length_m**2)
    limit = EXPLICIT_DIFFUSION_FO_LIMIT_3D * FO_SAFETY_FACTOR
    if fo > limit + 1e-15:
        raise DiffusionStabilityError(
            f"Explicit diffusion is unstable: Fourier number Fo=Deff*dt/dx^2 = {fo:.4g} "
            f"exceeds the 3-D limit {limit:.4g} (dx={voxel_length_m:.3g} m, "
            f"Deff={diffusivity_m2_s:.3g} m^2/s, dt={dt_s:.3g} s). "
            f"Reduce dt below {max_stable_dt(diffusivity_m2_s, voxel_length_m):.3g} s "
            f"or reduce Deff."
        )
    return fo


def interface_mask(volume: npt.NDArray[np.integer]) -> npt.NDArray[np.bool_]:
    """Pore voxels that have at least one 6-neighbor solid voxel."""
    pore = volume == PORE_LABEL
    solid = volume == SOLID_LABEL
    adj = np.zeros_like(pore)
    adj[1:, :, :] |= solid[:-1, :, :]
    adj[:-1, :, :] |= solid[1:, :, :]
    adj[:, 1:, :] |= solid[:, :-1, :]
    adj[:, :-1, :] |= solid[:, 1:, :]
    adj[:, :, 1:] |= solid[:, :, :-1]
    adj[:, :, :-1] |= solid[:, :, 1:]
    return pore & adj


def _apply_boundaries(
    concentration: npt.NDArray[np.floating],
    pore: npt.NDArray[np.bool_],
    z_max_bc: str,
) -> None:
    concentration[:, :, 0] = np.where(pore[:, :, 0], 1.0, 0.0)
    if z_max_bc == "fixed_zero":
        concentration[:, :, -1] = 0.0
    elif z_max_bc == "fixed_one":
        concentration[:, :, -1] = np.where(pore[:, :, -1], 1.0, 0.0)
    concentration[~pore] = 0.0


def _laplacian_neumann(
    field: npt.NDArray[np.floating],
    pore: npt.NDArray[np.bool_],
    dx: float,
) -> npt.NDArray[np.floating]:
    """Second-order Laplacian with zero-flux (copy) ghosts on all faces.

    Dirichlet faces are overwritten after the update by :func:`_apply_boundaries`.
    Solid voxels do not contribute as neighbors (treated as no-flux walls).
    """
    padded = np.pad(field, 1, mode="edge")
    pore_p = np.pad(pore, 1, mode="constant", constant_values=False)
    # If a neighbor is not pore, reuse the center value (zero flux into solid / outside).
    c = padded[1:-1, 1:-1, 1:-1]
    xp = np.where(pore_p[2:, 1:-1, 1:-1], padded[2:, 1:-1, 1:-1], c)
    xm = np.where(pore_p[:-2, 1:-1, 1:-1], padded[:-2, 1:-1, 1:-1], c)
    yp = np.where(pore_p[1:-1, 2:, 1:-1], padded[1:-1, 2:, 1:-1], c)
    ym = np.where(pore_p[1:-1, :-2, 1:-1], padded[1:-1, :-2, 1:-1], c)
    zp = np.where(pore_p[1:-1, 1:-1, 2:], padded[1:-1, 1:-1, 2:], c)
    zm = np.where(pore_p[1:-1, 1:-1, :-2], padded[1:-1, 1:-1, :-2], c)
    lap = (xp + xm + yp + ym + zp + zm - 6.0 * c) / (dx * dx)
    lap[~pore] = 0.0
    return lap


def oxidation_penetration_depth(
    cumulative_reaction: npt.NDArray[np.floating],
    threshold: float,
) -> float:
    """Largest z-index (voxels) whose mean cumulative reaction exceeds ``threshold``.

    Returns 0 if the threshold is never met.
    """
    if cumulative_reaction.ndim != 3:
        raise ValueError("cumulative_reaction must be 3-D")
    profile = np.mean(cumulative_reaction, axis=(0, 1))
    if float(np.max(profile)) <= 0.0:
        return 0.0
    hits = np.where(profile >= threshold)[0]
    if hits.size == 0:
        return 0.0
    return float(hits.max())


def _optional_degradation(
    volume: npt.NDArray[np.integer],
    damage: npt.NDArray[np.floating],
    threshold: float,
    max_fraction: float,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.uint8]:
    """Exploratory: convert a limited set of highly damaged interface solids to pore.

    This is **not** a validated recession model. It exists so later studies can
    explore one-way coupling. Geometry is otherwise frozen.
    """
    solid = volume == SOLID_LABEL
    # damage lives on pores; look at solid voxels adjacent to high-damage pores
    high = damage >= threshold
    adj_high = np.zeros_like(solid)
    adj_high[1:, :, :] |= high[:-1, :, :]
    adj_high[:-1, :, :] |= high[1:, :, :]
    adj_high[:, 1:, :] |= high[:, :-1, :]
    adj_high[:, :-1, :] |= high[:, 1:, :]
    adj_high[:, :, 1:] |= high[:, :, :-1]
    adj_high[:, :, :-1] |= high[:, :, 1:]
    candidates = np.argwhere(solid & adj_high)
    degraded = np.array(volume, dtype=np.uint8, copy=True)
    if candidates.size == 0:
        return degraded
    n_allow = max(1, int(max_fraction * int(np.sum(solid))))
    n_allow = min(n_allow, len(candidates))
    if rng is None:
        rng = np.random.default_rng(0)
    pick = rng.choice(len(candidates), size=n_allow, replace=False)
    for idx in pick:
        x, y, z = (int(candidates[idx, 0]), int(candidates[idx, 1]), int(candidates[idx, 2]))
        degraded[x, y, z] = PORE_LABEL
    return degraded


def run_diffusion_reaction(
    volume: npt.NDArray[np.integer],
    *,
    voxel_length_m: float,
    n_time_steps: int,
    dt_s: float,
    diffusivity_m2_s: float,
    reaction_rate_1_s: float,
    z_max_bc: str = "zero_flux",
    oxidation_threshold: float = 0.05,
    enable_degradation: bool = False,
    degradation_damage_threshold: float = 0.8,
    degradation_max_fraction: float = 0.02,
    snapshot_times: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Integrate the reduced oxygen model and return a :class:`SimulationResult`."""
    if volume.ndim != 3:
        raise ValueError("volume must be 3-D")
    if n_time_steps < 1:
        raise ValueError("n_time_steps must be >= 1")
    units = UnitSystem(voxel_length_m=voxel_length_m)
    fo = check_explicit_stability(dt_s, diffusivity_m2_s, voxel_length_m)
    LOGGER.debug("Diffusion Fo=%.4g (stable)", fo)

    pore = volume == PORE_LABEL
    if not np.any(pore):
        raise ValueError("volume has no pore voxels; cannot transport oxygen")
    iface = interface_mask(volume)
    concentration = np.zeros(volume.shape, dtype=np.float64)
    _apply_boundaries(concentration, pore, z_max_bc)
    cumulative = np.zeros_like(concentration)
    snapshots: dict[str, npt.NDArray[np.floating]] = {}
    fractions = snapshot_times or [1.0]
    snap_steps = {max(1, int(round(f * n_time_steps))) for f in fractions}

    dx = units.dx_m
    for step in range(1, n_time_steps + 1):
        lap = _laplacian_neumann(concentration, pore, dx)
        reaction = reaction_rate_1_s * concentration * iface.astype(np.float64)
        concentration = concentration + dt_s * (diffusivity_m2_s * lap - reaction)
        concentration = np.clip(concentration, 0.0, 1.0)
        _apply_boundaries(concentration, pore, z_max_bc)
        cumulative += reaction * dt_s
        if step in snap_steps:
            snapshots[f"step_{step}"] = concentration.copy()

    cmax = float(np.max(cumulative))
    damage = cumulative / cmax if cmax > 0.0 else np.zeros_like(cumulative)
    profile = np.mean(cumulative, axis=(0, 1))
    depth_vox = oxidation_penetration_depth(cumulative, oxidation_threshold)
    n_iface = max(int(np.sum(iface)), 1)
    cum_norm = float(np.sum(cumulative) / n_iface)
    porosity_proxy = float(np.mean(damage[iface])) if np.any(iface) else 0.0

    degraded = None
    if enable_degradation:
        degraded = _optional_degradation(
            volume,
            damage,
            degradation_damage_threshold,
            degradation_max_fraction,
            rng=rng,
        )

    return SimulationResult(
        concentration=concentration,
        cumulative_reaction=cumulative,
        oxidation_profile=profile,
        oxidation_penetration_depth_voxels=depth_vox,
        oxidation_penetration_depth_m=units.voxels_to_meters(depth_vox),
        damage=damage,
        porosity_change_proxy=porosity_proxy,
        cumulative_reaction_normalized=cum_norm,
        degraded_volume=degraded,
        n_time_steps=n_time_steps,
        dt_s=dt_s,
        diffusivity_m2_s=diffusivity_m2_s,
        reaction_rate_1_s=reaction_rate_1_s,
        snapshots=snapshots,
        used_degradation=enable_degradation,
    )


def run_from_config(
    volume: npt.NDArray[np.integer],
    config: DiffusionConfig,
    voxel_length_m: float,
    diffusivity_m2_s: float,
    reaction_rate_1_s: float,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Convenience wrapper using a :class:`DiffusionConfig`."""
    return run_diffusion_reaction(
        volume,
        voxel_length_m=voxel_length_m,
        n_time_steps=config.n_time_steps,
        dt_s=config.dt_s,
        diffusivity_m2_s=diffusivity_m2_s,
        reaction_rate_1_s=reaction_rate_1_s,
        z_max_bc=config.z_max_bc,
        oxidation_threshold=config.oxidation_threshold,
        enable_degradation=config.enable_degradation,
        degradation_damage_threshold=config.degradation_damage_threshold,
        degradation_max_fraction=config.degradation_max_fraction,
        snapshot_times=config.snapshot_times,
        rng=rng,
    )


def analytical_slab_dirichlet(
    z_m: npt.NDArray[np.floating],
    t_s: float,
    length_m: float,
    diffusivity_m2_s: float,
    n_terms: int = 80,
) -> npt.NDArray[np.floating]:
    """Closed-form 1-D transient diffusion, C(0)=1, C(L)=0, C(z,0)=0.

    C(z,t) = 1 - z/L - (2/π) Σ_{n=1}^N (1/n) sin(nπz/L) exp(-n²π² Deff t / L²)
    """
    if t_s < 0.0:
        raise ValueError("time must be non-negative")
    z = np.asarray(z_m, dtype=np.float64)
    L = float(length_m)
    series = np.zeros_like(z)
    for n in range(1, n_terms + 1):
        series += (1.0 / n) * np.sin(n * np.pi * z / L) * np.exp(
            -((n * np.pi) ** 2) * diffusivity_m2_s * t_s / (L * L)
        )
    return (1.0 - z / L) - (2.0 / np.pi) * series
