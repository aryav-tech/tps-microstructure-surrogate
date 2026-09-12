"""Synthetic fiber-volume generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tps_surrogate.constants import PORE_LABEL, SOLID_LABEL
from tps_surrogate.microstructure import (
    DegenerateVolumeError,
    central_slices,
    compute_porosity,
    generate_fiber_volume,
    load_volume_npz,
    save_volume_npz,
)


def test_porosity_and_binary_labels() -> None:
    volume = generate_fiber_volume(
        (16, 16, 16),
        target_porosity=0.80,
        fiber_radius_voxels=1.2,
        seed=3,
        porosity_tolerance=0.12,
    )
    assert volume.dtype == np.uint8
    assert set(np.unique(volume)).issubset({PORE_LABEL, SOLID_LABEL})
    porosity = compute_porosity(volume)
    assert 0.68 <= porosity <= 0.92
    assert 0.0 < porosity < 1.0


def test_deterministic_seed() -> None:
    a = generate_fiber_volume((14, 14, 14), 0.82, 1.1, seed=99, porosity_tolerance=0.15)
    b = generate_fiber_volume((14, 14, 14), 0.82, 1.1, seed=99, porosity_tolerance=0.15)
    assert np.array_equal(a, b)


def test_degenerate_fully_pore() -> None:
    with pytest.raises(DegenerateVolumeError):
        generate_fiber_volume(
            (12, 12, 12),
            target_porosity=0.80,
            fiber_radius_voxels=1.0,
            n_fibers_max=0,
            max_attempts=2,
            seed=1,
        )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    volume = generate_fiber_volume((12, 12, 12), 0.85, 1.0, seed=2, porosity_tolerance=0.15)
    path = tmp_path / "vol.npz"
    save_volume_npz(path, volume, porosity=compute_porosity(volume))
    loaded, meta = load_volume_npz(path)
    assert np.array_equal(loaded, volume)
    assert "porosity" in meta


def test_central_slices_shapes() -> None:
    volume = np.zeros((10, 12, 14), dtype=np.uint8)
    slices = central_slices(volume)
    assert slices["yz_at_xmid"].shape == (12, 14)
    assert slices["xz_at_ymid"].shape == (10, 14)
    assert slices["xy_at_zmid"].shape == (10, 12)
