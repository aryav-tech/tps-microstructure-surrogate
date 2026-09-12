"""Diffusion–reaction baseline: stability, BCs, pore-only transport."""

from __future__ import annotations

import numpy as np
import pytest

from tps_surrogate.constants import SOLID_LABEL
from tps_surrogate.diffusion_reaction import (
    DiffusionStabilityError,
    check_explicit_stability,
    interface_mask,
    run_diffusion_reaction,
)
from tps_surrogate.microstructure import generate_fiber_volume


def test_stability_guard() -> None:
    with pytest.raises(DiffusionStabilityError, match="Fourier"):
        check_explicit_stability(dt_s=1.0, diffusivity_m2_s=1.0e-6, voxel_length_m=1.0e-5)


def test_stable_fourier_number() -> None:
    fo = check_explicit_stability(dt_s=1.0e-4, diffusivity_m2_s=1.0e-8, voxel_length_m=5.0e-5)
    assert fo < 1.0 / 6.0


def test_concentration_only_in_pores() -> None:
    volume = generate_fiber_volume((12, 12, 12), 0.82, 1.1, seed=4, porosity_tolerance=0.15)
    result = run_diffusion_reaction(
        volume,
        voxel_length_m=5.0e-5,
        n_time_steps=8,
        dt_s=2.0e-4,
        diffusivity_m2_s=4.0e-8,
        reaction_rate_1_s=10.0,
    )
    solid = volume == SOLID_LABEL
    assert np.all(result.concentration[solid] == 0.0)
    assert result.concentration[:, :, 0][volume[:, :, 0] == 0].max() == pytest.approx(1.0)
    assert result.oxidation_penetration_depth_voxels >= 0.0
    assert result.cumulative_reaction_normalized >= 0.0


def test_interface_mask_is_pore_adjacent_to_solid() -> None:
    volume = np.zeros((5, 5, 5), dtype=np.uint8)
    volume[2, 2, 2] = SOLID_LABEL
    mask = interface_mask(volume)
    assert mask[2, 2, 2] is np.False_ or mask[2, 2, 2] == False  # noqa: E712
    assert bool(mask[1, 2, 2])
    assert not bool(mask[0, 0, 0])


def test_no_pore_raises() -> None:
    volume = np.ones((6, 6, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="no pore"):
        run_diffusion_reaction(
            volume,
            voxel_length_m=1.0e-5,
            n_time_steps=2,
            dt_s=1.0e-8,
            diffusivity_m2_s=1.0e-8,
            reaction_rate_1_s=1.0,
        )
