"""Central voxel ↔ physical-unit conversions.

Every module that reports a physical quantity must call through this module
so tables and reports carry consistent units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class UnitSystem:
    """Length scale and derived geometric conversions for one voxel grid."""

    voxel_length_m: float

    def __post_init__(self) -> None:
        if self.voxel_length_m <= 0.0:
            raise ValueError(
                f"voxel_length_m must be positive, got {self.voxel_length_m}"
            )

    @property
    def dx_m(self) -> float:
        return self.voxel_length_m

    @property
    def voxel_area_m2(self) -> float:
        return self.voxel_length_m**2

    @property
    def voxel_volume_m3(self) -> float:
        return self.voxel_length_m**3

    def voxels_to_meters(
        self, n_voxels: float | npt.NDArray[np.floating]
    ) -> float | npt.NDArray[np.floating]:
        """Convert a length measured in voxels to meters (scalar or array)."""
        arr = np.asarray(n_voxels, dtype=np.float64) * self.voxel_length_m
        if arr.ndim == 0:
            return float(arr)
        return arr

    def meters_to_voxels(
        self, length_m: float | npt.NDArray[np.floating]
    ) -> float | npt.NDArray[np.floating]:
        """Convert a length in meters back to voxels (scalar or array)."""
        arr = np.asarray(length_m, dtype=np.float64) / self.voxel_length_m
        if arr.ndim == 0:
            return float(arr)
        return arr

    def voxel_count_to_volume_m3(self, n_voxels: float) -> float:
        return float(n_voxels) * self.voxel_volume_m3

    def voxel_face_count_to_area_m2(self, n_faces: float) -> float:
        return float(n_faces) * self.voxel_area_m2

    def interfacial_area_per_volume(self, n_faces: float, n_voxels: float) -> float:
        """Specific interfacial area [1/m] from shared solid–pore faces."""
        if n_voxels <= 0:
            return float("nan")
        area_m2 = self.voxel_face_count_to_area_m2(n_faces)
        volume_m3 = self.voxel_count_to_volume_m3(n_voxels)
        return area_m2 / volume_m3


def conductivity_unit() -> str:
    """SI unit string for thermal conductivity."""
    return "W/(m·K)"


def diffusivity_unit() -> str:
    """SI unit string for diffusivity."""
    return "m^2/s"


def reaction_rate_unit() -> str:
    """SI unit string for the first-order reaction coefficient."""
    return "1/s"


def length_unit() -> str:
    return "m"


def temperature_unit() -> str:
    """Normalized temperature is dimensionless in the conduction solve."""
    return "1 (normalized)"


def concentration_unit() -> str:
    return "1 (normalized)"


def require_positive(name: str, value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)
