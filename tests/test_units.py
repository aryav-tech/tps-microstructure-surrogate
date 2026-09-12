"""Voxel ↔ meter conversions must be exact round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from tps_surrogate.units import (
    UnitSystem,
    conductivity_unit,
    diffusivity_unit,
    reaction_rate_unit,
)


def test_roundtrip_scalar() -> None:
    units = UnitSystem(voxel_length_m=4.0e-5)
    for n in (0.0, 1.0, 2.5, 16.0, 100.0):
        meters = units.voxels_to_meters(n)
        back = units.meters_to_voxels(meters)
        assert back == pytest.approx(n, rel=0, abs=0.0)


def test_roundtrip_array() -> None:
    units = UnitSystem(voxel_length_m=5.0e-5)
    voxels = np.array([0.0, 1.0, 3.0, 8.25], dtype=np.float64)
    meters = units.voxels_to_meters(voxels)
    recovered = units.meters_to_voxels(meters)
    assert np.allclose(recovered, voxels)
    assert float(np.asarray(meters)[0]) == units.voxels_to_meters(float(voxels[0]))


def test_derived_geometry() -> None:
    units = UnitSystem(voxel_length_m=2.0e-6)
    assert units.voxel_area_m2 == pytest.approx(4.0e-12)
    assert units.voxel_volume_m3 == pytest.approx(8.0e-18)
    sav = units.interfacial_area_per_volume(10, 100)
    assert sav == pytest.approx(10 * units.voxel_area_m2 / (100 * units.voxel_volume_m3))


def test_rejects_nonpositive_voxel() -> None:
    with pytest.raises(ValueError):
        UnitSystem(voxel_length_m=0.0)


def test_unit_strings() -> None:
    assert "W" in conductivity_unit()
    assert "m^2" in diffusivity_unit()
    assert "1/s" in reaction_rate_unit()
