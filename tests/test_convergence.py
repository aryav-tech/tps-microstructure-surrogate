"""Grid / time-step refinement study (marked slow; skipped in default CI)."""

from __future__ import annotations

import numpy as np
import pytest

from tps_surrogate.conductivity import effective_conductivity
from tps_surrogate.constants import SOLID_LABEL
from tps_surrogate.diffusion_reaction import analytical_slab_dirichlet, run_diffusion_reaction
from tps_surrogate.units import UnitSystem

pytestmark = pytest.mark.slow


def _diffusion_rmse(n_steps: int, dt: float, nz: int = 21) -> float:
    volume = np.zeros((1, 1, nz), dtype=np.uint8)
    dx = 1.0
    deff = 1.0
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
    exact = analytical_slab_dirichlet(z, n_steps * dt, units.voxels_to_meters(nz - 1), deff)
    err = result.concentration[0, 0, 1:-1] - exact[1:-1]
    return float(np.sqrt(np.mean(err**2)))


def test_diffusion_error_decreases_with_smaller_dt() -> None:
    # same physical time t=3.2 s
    rmse_coarse = _diffusion_rmse(n_steps=40, dt=0.08)
    rmse_fine = _diffusion_rmse(n_steps=80, dt=0.04)
    assert rmse_fine <= rmse_coarse * 1.05 + 1e-12


def test_homogeneous_conductivity_stable_across_grids() -> None:
    k = 3.0
    errors = []
    for n in (6, 8, 10):
        volume = np.ones((n, n, n), dtype=np.uint8) * SOLID_LABEL
        result = effective_conductivity(volume, k_solid_w_m_k=k, k_pore_w_m_k=0.1, maxiter=300)
        errors.append(abs(result.k_eff_z - k) / k)
    # homogeneous k_eff should stay accurate (not grow) as n increases
    assert errors[-1] <= errors[0] + 0.02
    assert errors[-1] < 0.03
