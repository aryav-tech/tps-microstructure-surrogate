"""Scalar geometric descriptors of a two-phase voxel microstructure.

Returned dictionaries are flat (CSV / tabular-ML ready). Physical-unit
fields are converted through :mod:`tps_surrogate.units`. Missing connected
pathways are reported as NaN (``constants.NO_CONNECTED_PATH``).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from tps_surrogate.constants import NO_CONNECTED_PATH, PORE_LABEL, SOLID_LABEL
from tps_surrogate.microstructure import compute_porosity
from tps_surrogate.units import UnitSystem

REQUIRED_DESCRIPTOR_KEYS = (
    "porosity",
    "solid_fraction",
    "n_voxels",
    "n_solid_voxels",
    "n_pore_voxels",
    "interfacial_face_count",
    "interfacial_area_m2",
    "interfacial_area_per_volume_1_m",
    "pore_connected_fraction",
    "tortuosity_proxy",
    "geodesic_path_voxels",
    "continuity_x",
    "continuity_y",
    "continuity_z",
    "mean_pore_radius_voxels",
    "mean_pore_radius_m",
    "max_pore_radius_voxels",
    "max_pore_radius_m",
)

NAN_ALLOWED_KEYS = {
    "tortuosity_proxy",
    "geodesic_path_voxels",
    "mean_pore_radius_voxels",
    "mean_pore_radius_m",
    "max_pore_radius_voxels",
    "max_pore_radius_m",
    "pore_connected_fraction",
}


def _interface_face_count(volume: npt.NDArray[np.integer]) -> int:
    """Count 6-connected solid–pore shared faces."""
    solid = volume == SOLID_LABEL
    count = 0
    count += int(np.sum(solid[1:, :, :] != solid[:-1, :, :]))
    count += int(np.sum(solid[:, 1:, :] != solid[:, :-1, :]))
    count += int(np.sum(solid[:, :, 1:] != solid[:, :, :-1]))
    return count


def _directional_continuity(volume: npt.NDArray[np.integer], axis: int) -> float:
    """Fraction of solid voxels that also have a solid neighbor along ``axis``."""
    solid = volume == SOLID_LABEL
    n_solid = int(np.sum(solid))
    if n_solid == 0:
        return 0.0
    shifted = np.roll(solid, shift=-1, axis=axis)
    # do not wrap: zero the last plane
    slicer = [slice(None)] * 3
    slicer[axis] = slice(-1, None)
    shifted[tuple(slicer)] = False
    both = solid & shifted
    return float(np.sum(both) / n_solid)


def _pore_spanning_fraction(pore: npt.NDArray[np.bool_]) -> float:
    """Fraction of pore voxels belonging to a component that touches z=0 and z=max."""
    if not np.any(pore):
        return 0.0
    labeled, n_labels = ndimage.label(pore, structure=ndimage.generate_binary_structure(3, 1))
    if n_labels == 0:
        return 0.0
    touch_low = set(int(v) for v in np.unique(labeled[:, :, 0]) if v != 0)
    touch_high = set(int(v) for v in np.unique(labeled[:, :, -1]) if v != 0)
    spanning = touch_low & touch_high
    if not spanning:
        return 0.0
    mask = np.isin(labeled, list(spanning))
    return float(np.sum(mask) / np.sum(pore))


def _geodesic_z_path_length(pore: npt.NDArray[np.bool_]) -> float:
    """Shortest 6-connected pore path length (voxels) from z=0 to z=max.

    Returns NaN if no pathway exists.
    """
    if not np.any(pore[:, :, 0]) or not np.any(pore[:, :, -1]):
        return NO_CONNECTED_PATH
    nx, ny, nz = pore.shape
    dist = np.full(pore.shape, -1, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()
    xs, ys = np.nonzero(pore[:, :, 0])
    for x, y in zip(xs.tolist(), ys.tolist(), strict=False):
        queue.append((x, y, 0))
        dist[x, y, 0] = 0
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    best = -1
    while queue:
        x, y, z = queue.popleft()
        if z == nz - 1:
            best = int(dist[x, y, z])
            break
        dcur = int(dist[x, y, z])
        for dx, dy, dz in neighbors:
            xn, yn, zn = x + dx, y + dy, z + dz
            if xn < 0 or yn < 0 or zn < 0 or xn >= nx or yn >= ny or zn >= nz:
                continue
            if not pore[xn, yn, zn] or dist[xn, yn, zn] >= 0:
                continue
            dist[xn, yn, zn] = dcur + 1
            queue.append((xn, yn, zn))
    if best < 0:
        return NO_CONNECTED_PATH
    return float(best)


def compute_descriptors(
    volume: npt.NDArray[np.integer],
    voxel_length_m: float,
) -> dict[str, float]:
    """Return a flat descriptor dictionary for one binary volume."""
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    units = UnitSystem(voxel_length_m=voxel_length_m)
    porosity = compute_porosity(volume)
    solid_fraction = 1.0 - porosity
    n_voxels = int(volume.size)
    n_pore = int(np.sum(volume == PORE_LABEL))
    n_solid = n_voxels - n_pore
    n_faces = _interface_face_count(volume)
    area_m2 = units.voxel_face_count_to_area_m2(n_faces)
    sav = units.interfacial_area_per_volume(n_faces, n_voxels)

    pore = volume == PORE_LABEL
    connected_frac = _pore_spanning_fraction(pore)
    path_vox = _geodesic_z_path_length(pore)
    if math.isnan(path_vox):
        tortuosity = NO_CONNECTED_PATH
    else:
        depth = max(volume.shape[2] - 1, 1)
        tortuosity = float(path_vox / depth)

    if np.any(pore):
        edt = ndimage.distance_transform_edt(pore)
        pore_r = edt[pore]
        mean_r_vox = float(np.mean(pore_r))
        max_r_vox = float(np.max(pore_r))
    else:
        mean_r_vox = NO_CONNECTED_PATH
        max_r_vox = NO_CONNECTED_PATH

    mean_r_m = units.voxels_to_meters(mean_r_vox) if not math.isnan(mean_r_vox) else NO_CONNECTED_PATH
    max_r_m = units.voxels_to_meters(max_r_vox) if not math.isnan(max_r_vox) else NO_CONNECTED_PATH

    return {
        "porosity": porosity,
        "solid_fraction": solid_fraction,
        "n_voxels": float(n_voxels),
        "n_solid_voxels": float(n_solid),
        "n_pore_voxels": float(n_pore),
        "interfacial_face_count": float(n_faces),
        "interfacial_area_m2": float(area_m2),
        "interfacial_area_per_volume_1_m": float(sav),
        "pore_connected_fraction": float(connected_frac),
        "tortuosity_proxy": float(tortuosity),
        "geodesic_path_voxels": float(path_vox),
        "continuity_x": _directional_continuity(volume, 0),
        "continuity_y": _directional_continuity(volume, 1),
        "continuity_z": _directional_continuity(volume, 2),
        "mean_pore_radius_voxels": mean_r_vox,
        "mean_pore_radius_m": mean_r_m,
        "max_pore_radius_voxels": max_r_vox,
        "max_pore_radius_m": max_r_m,
    }


def validate_descriptor_dict(data: dict[str, Any]) -> dict[str, float]:
    """Check required keys and finite-ness (NaN allowed for documented keys)."""
    missing = [key for key in REQUIRED_DESCRIPTOR_KEYS if key not in data]
    if missing:
        raise KeyError(f"descriptor dict missing keys: {missing}")
    cleaned: dict[str, float] = {}
    for key in REQUIRED_DESCRIPTOR_KEYS:
        value = float(data[key])
        if math.isnan(value) or math.isinf(value):
            if key not in NAN_ALLOWED_KEYS:
                raise ValueError(f"descriptor {key} must be finite, got {value}")
        cleaned[key] = value
    return cleaned
