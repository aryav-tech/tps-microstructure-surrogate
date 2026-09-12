"""Analytical cross-check: 1-D transient diffusion (reaction off)."""

from __future__ import annotations

import numpy as np
import pytest

from tps_surrogate.diffusion_reaction import analytical_slab_dirichlet, run_diffusion_reaction
from tps_surrogate.units import UnitSystem

# Documented tolerance for this verification test (see docs/verification.md).
RMSE_TOLERANCE = 0.05
MAX_ABS_TOLERANCE = 0.10


def test_explicit_diffusion_matches_fourier_slab() -> None:
    """All-pore slab, C(0)=1, C(L)=0, IC=0, k_reaction=0.

    Compared to the standard Fourier-series solution. Grid and dt are chosen
    so the explicit scheme is stable (Fo < 1/6).
    """
    nz = 21
    volume = np.zeros((1, 1, nz), dtype=np.uint8)
    dx = 1.0  # meters; test-only length scale, not a TPS claim
    deff = 1.0
    dt = 0.04
    n_steps = 80
    t = n_steps * dt
    result = run_diffusion_reaction(
        volume,
        voxel_length_m=dx,
        n_time_steps=n_steps,
        dt_s=dt,
        diffusivity_m2_s=deff,
        reaction_rate_1_s=0.0,
        z_max_bc="fixed_zero",
    )
    units = UnitSystem(voxel_length_m=dx)
    z = units.voxels_to_meters(np.arange(nz, dtype=np.float64))
    length = units.voxels_to_meters(nz - 1)
    exact = analytical_slab_dirichlet(z, t, length, deff, n_terms=120)
    numeric = result.concentration[0, 0, :]
    # exclude Dirichlet nodes from RMSE (they are constrained)
    err = numeric[1:-1] - exact[1:-1]
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    assert rmse < RMSE_TOLERANCE, f"RMSE={rmse:.4g} exceeds {RMSE_TOLERANCE}"
    assert max_abs < MAX_ABS_TOLERANCE, f"max abs={max_abs:.4g} exceeds {MAX_ABS_TOLERANCE}"


def test_analytical_steady_state_limit() -> None:
    z = np.linspace(0.0, 1.0, 11)
    late = analytical_slab_dirichlet(z, t_s=50.0, length_m=1.0, diffusivity_m2_s=1.0)
    assert late == pytest.approx(1.0 - z, abs=1e-6)
