"""Analytical series / parallel resistor checks for layered slabs."""

from __future__ import annotations

import numpy as np
import pytest

from tps_surrogate.conductivity import (
    effective_conductivity,
    layered_volume,
    parallel_conductivity,
    series_conductivity,
)

K_SOLID = 10.0
K_PORE = 1.0
REL_TOL = 0.08


def test_series_layers_perpendicular_to_z() -> None:
    """Layers stacked along z; heat flows in z → harmonic mean."""
    volume = layered_volume((8, 8, 8), axis="z", period=2)
    frac_solid = float(np.mean(volume == 1))
    expected = series_conductivity(K_SOLID, K_PORE, frac_solid)
    result = effective_conductivity(
        volume, k_solid_w_m_k=K_SOLID, k_pore_w_m_k=K_PORE, maxiter=400
    )
    assert result.k_eff_z == pytest.approx(expected, rel=REL_TOL)


def test_parallel_layers_along_z() -> None:
    """Layers stacked along x; heat flows in z → arithmetic mean."""
    volume = layered_volume((8, 8, 8), axis="x", period=2)
    frac_solid = float(np.mean(volume == 1))
    expected = parallel_conductivity(K_SOLID, K_PORE, frac_solid)
    result = effective_conductivity(
        volume, k_solid_w_m_k=K_SOLID, k_pore_w_m_k=K_PORE, maxiter=400
    )
    assert result.k_eff_z == pytest.approx(expected, rel=REL_TOL)


def test_mixture_rules_ordering() -> None:
    k_par = parallel_conductivity(K_SOLID, K_PORE, 0.5)
    k_ser = series_conductivity(K_SOLID, K_PORE, 0.5)
    assert k_par > k_ser
