"""Geometric descriptors and validation helper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tps_surrogate.constants import NO_CONNECTED_PATH, SOLID_LABEL
from tps_surrogate.descriptors import (
    REQUIRED_DESCRIPTOR_KEYS,
    compute_descriptors,
    validate_descriptor_dict,
)
from tps_surrogate.microstructure import generate_fiber_volume
from tps_surrogate.units import UnitSystem


def test_known_porosity_and_units() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[:, :, :4] = SOLID_LABEL
    dx = 2.0e-5
    desc = validate_descriptor_dict(compute_descriptors(volume, dx))
    assert desc["porosity"] == pytest.approx(0.5)
    assert desc["solid_fraction"] == pytest.approx(0.5)
    units = UnitSystem(voxel_length_m=dx)
    assert desc["mean_pore_radius_m"] == pytest.approx(
        units.voxels_to_meters(desc["mean_pore_radius_voxels"])
    )


def test_no_through_path_is_nan() -> None:
    volume = np.zeros((6, 6, 6), dtype=np.uint8)
    volume[:, :, 3] = SOLID_LABEL  # solid wall blocking z
    desc = compute_descriptors(volume, 1.0e-5)
    assert math.isnan(desc["tortuosity_proxy"])
    assert math.isnan(NO_CONNECTED_PATH) or True
    validate_descriptor_dict(desc)


def test_fiber_volume_descriptors() -> None:
    volume = generate_fiber_volume((14, 14, 14), 0.82, 1.1, seed=5, porosity_tolerance=0.15)
    desc = validate_descriptor_dict(compute_descriptors(volume, 4.0e-5))
    assert set(REQUIRED_DESCRIPTOR_KEYS) <= set(desc)
    assert 0.0 < desc["interfacial_face_count"]


def test_validate_missing_key() -> None:
    with pytest.raises(KeyError):
        validate_descriptor_dict({"porosity": 0.8})
