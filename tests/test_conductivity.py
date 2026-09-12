"""Homogeneous-volume conductivity checks."""

from __future__ import annotations

import numpy as np
import pytest

from tps_surrogate.conductivity import effective_conductivity
from tps_surrogate.constants import SOLID_LABEL


def test_homogeneous_solid() -> None:
    k = 4.0
    volume = np.ones((8, 8, 8), dtype=np.uint8) * SOLID_LABEL
    result = effective_conductivity(volume, k_solid_w_m_k=k, k_pore_w_m_k=0.05, maxiter=200)
    assert result.k_eff_x == pytest.approx(k, rel=0.03)
    assert result.k_eff_y == pytest.approx(k, rel=0.03)
    assert result.k_eff_z == pytest.approx(k, rel=0.03)
    assert not result.used_fallback


def test_homogeneous_pore() -> None:
    k = 0.25
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    result = effective_conductivity(volume, k_solid_w_m_k=8.0, k_pore_w_m_k=k, maxiter=200)
    assert result.k_eff_z == pytest.approx(k, rel=0.03)


def test_rejects_2d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        effective_conductivity(np.zeros((4, 4), dtype=np.uint8), k_solid_w_m_k=1.0, k_pore_w_m_k=0.1)
