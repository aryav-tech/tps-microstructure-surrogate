"""Synthetic FiberForm-like fibrous voxel microstructures.

The generator places random cylindrical fibers into a two-phase grid
(1 = carbon solid, 0 = pore). Geometry is approximate and is **not** a
reconstruction of proprietary NASA flight material.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tps_surrogate.constants import PORE_LABEL, SOLID_LABEL

LOGGER = logging.getLogger(__name__)

VolumeArray = npt.NDArray[np.uint8]


class DegenerateVolumeError(ValueError):
    """Raised when a generated volume is fully solid, fully pore, or off-target.

    Downstream descriptor and solver pipelines must not silently consume
    degenerate volumes. Callers (dataset generation) should skip the sample.
    """


def compute_porosity(volume: npt.NDArray[np.integer]) -> float:
    """Pore-volume fraction of a binary voxel grid."""
    if volume.size == 0:
        raise ValueError("volume is empty")
    return float(np.mean(volume == PORE_LABEL))


def _distance_to_segment_grid(
    ii: npt.NDArray[np.integer],
    jj: npt.NDArray[np.integer],
    kk: npt.NDArray[np.integer],
    p0: npt.NDArray[np.floating],
    p1: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Vectorized Euclidean distance from voxel centers to a line segment."""
    p = np.stack([ii.astype(np.float64), jj.astype(np.float64), kk.astype(np.float64)], axis=-1)
    v = p1 - p0
    denom = float(np.dot(v, v))
    if denom < 1e-16:
        return np.linalg.norm(p - p0, axis=-1)
    t = np.clip(np.sum((p - p0) * v, axis=-1) / denom, 0.0, 1.0)
    closest = p0 + t[..., None] * v
    return np.linalg.norm(p - closest, axis=-1)


def _draw_cylinder(
    volume: VolumeArray,
    p0: npt.NDArray[np.floating],
    p1: npt.NDArray[np.floating],
    radius: float,
) -> None:
    """Rasterize a finite cylinder into ``volume`` (in-place)."""
    nx, ny, nz = volume.shape
    r_pad = int(np.ceil(radius)) + 1
    mins = np.floor(np.minimum(p0, p1)).astype(int) - r_pad
    maxs = np.ceil(np.maximum(p0, p1)).astype(int) + r_pad
    i0, j0, k0 = (int(np.clip(mins[0], 0, nx - 1)), int(np.clip(mins[1], 0, ny - 1)), int(np.clip(mins[2], 0, nz - 1)))
    i1, j1, k1 = (int(np.clip(maxs[0], 0, nx - 1)), int(np.clip(maxs[1], 0, ny - 1)), int(np.clip(maxs[2], 0, nz - 1)))
    if i1 < i0 or j1 < j0 or k1 < k0:
        return
    ii, jj, kk = np.meshgrid(
        np.arange(i0, i1 + 1),
        np.arange(j0, j1 + 1),
        np.arange(k0, k1 + 1),
        indexing="ij",
    )
    dist = _distance_to_segment_grid(ii, jj, kk, p0, p1)
    mask = dist <= radius
    volume[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1][mask] = SOLID_LABEL


def _random_direction(
    rng: np.random.Generator,
    in_plane_spread_deg: float,
    out_of_plane_spread_deg: float,
) -> npt.NDArray[np.floating]:
    """FiberForm-inspired direction: mostly in-plane (xy) with controllable tilt.

    ``in_plane_spread_deg`` is the half-width of a uniform azimuth draw about a
    random mean in-plane heading. ``out_of_plane_spread_deg`` is the half-width
    of elevation away from the xy plane.
    """
    mean_az = rng.uniform(0.0, 2.0 * np.pi)
    az = mean_az + np.deg2rad(rng.uniform(-in_plane_spread_deg, in_plane_spread_deg))
    el = np.deg2rad(rng.uniform(-out_of_plane_spread_deg, out_of_plane_spread_deg))
    dx = np.cos(el) * np.cos(az)
    dy = np.cos(el) * np.sin(az)
    dz = np.sin(el)
    vec = np.array([dx, dy, dz], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-16:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _add_fiber(
    volume: VolumeArray,
    rng: np.random.Generator,
    radius: float,
    length: float,
    in_plane_spread_deg: float,
    out_of_plane_spread_deg: float,
) -> None:
    nx, ny, nz = volume.shape
    center = rng.uniform(low=[0.0, 0.0, 0.0], high=[nx - 1.0, ny - 1.0, nz - 1.0])
    direction = _random_direction(rng, in_plane_spread_deg, out_of_plane_spread_deg)
    half = 0.5 * length
    p0 = center - half * direction
    p1 = center + half * direction
    _draw_cylinder(volume, p0, p1, radius)


def _fast_fallback_volume(
    shape: tuple[int, int, int],
    target_porosity: float,
    radius: float,
    rng: np.random.Generator,
) -> VolumeArray:
    """Cheap low-resolution fiber field: short axis-aligned plus tilted sticks."""
    volume = np.zeros(shape, dtype=np.uint8)
    nx, ny, nz = shape
    n_guess = max(4, int((1.0 - target_porosity) * nx * 3))
    length = float(max(nx, ny, nz))
    for _ in range(n_guess):
        if rng.random() < 0.7:
            # primarily in-plane
            y = rng.integers(0, ny)
            z = rng.integers(0, nz)
            for x in range(nx):
                for dy in range(-int(radius), int(radius) + 1):
                    for dz in range(-int(radius), int(radius) + 1):
                        if dy * dy + dz * dz <= radius * radius:
                            yy, zz = int(y + dy), int(z + dz)
                            if 0 <= yy < ny and 0 <= zz < nz:
                                volume[x, yy, zz] = SOLID_LABEL
        else:
            x0 = rng.uniform(0, nx - 1)
            y0 = rng.uniform(0, ny - 1)
            z0 = rng.uniform(0, nz - 1)
            direction = np.array(
                [rng.normal(1.0, 0.3), rng.normal(0.0, 0.4), rng.normal(0.0, 0.3)],
                dtype=np.float64,
            )
            direction /= max(np.linalg.norm(direction), 1e-12)
            _draw_cylinder(volume, np.array([x0, y0, z0]) - 0.5 * length * direction,
                           np.array([x0, y0, z0]) + 0.5 * length * direction, radius)
        if compute_porosity(volume) <= target_porosity:
            break
    return volume


def generate_fiber_volume(
    shape: tuple[int, int, int],
    target_porosity: float,
    fiber_radius_voxels: float,
    *,
    in_plane_spread_deg: float = 30.0,
    out_of_plane_spread_deg: float = 15.0,
    seed: int | None = None,
    n_fibers_max: int = 120,
    fiber_length_voxels: float | None = None,
    porosity_tolerance: float = 0.10,
    max_attempts: int = 8,
    rng: np.random.Generator | None = None,
) -> VolumeArray:
    """Generate a binary fibrous volume near ``target_porosity``.

    Raises:
        DegenerateVolumeError: if after ``max_attempts`` the volume is fully
            solid, fully pore, or ``|porosity - target| > porosity_tolerance``.
        ValueError: invalid parameters.
    """
    if len(shape) != 3 or any(int(v) < 4 for v in shape):
        raise ValueError(f"shape must be 3-D with each dim >= 4, got {shape}")
    if not 0.05 < target_porosity < 0.99:
        raise ValueError(f"target_porosity must be in (0.05, 0.99), got {target_porosity}")
    if fiber_radius_voxels < 0.5:
        raise ValueError("fiber_radius_voxels must be >= 0.5")

    shape_i = (int(shape[0]), int(shape[1]), int(shape[2]))
    if rng is None:
        rng = np.random.default_rng(seed)
    length = float(fiber_length_voxels or max(shape_i))
    use_fallback = min(shape_i) < 12

    last_error = ""
    for attempt in range(max_attempts):
        if use_fallback:
            volume = _fast_fallback_volume(shape_i, target_porosity, fiber_radius_voxels, rng)
        else:
            volume = np.zeros(shape_i, dtype=np.uint8)
            for _ in range(n_fibers_max):
                _add_fiber(
                    volume,
                    rng,
                    fiber_radius_voxels,
                    length,
                    in_plane_spread_deg,
                    out_of_plane_spread_deg,
                )
                if compute_porosity(volume) <= target_porosity:
                    break
        porosity = compute_porosity(volume)
        if porosity <= 0.0:
            last_error = "volume is fully solid"
            continue
        if porosity >= 1.0:
            last_error = "volume is fully pore (no fibers deposited)"
            continue
        if abs(porosity - target_porosity) > porosity_tolerance:
            last_error = (
                f"porosity {porosity:.3f} is outside tolerance "
                f"{porosity_tolerance:.3f} of target {target_porosity:.3f}"
            )
            continue
        LOGGER.debug(
            "Generated volume attempt %d porosity=%.3f target=%.3f",
            attempt,
            porosity,
            target_porosity,
        )
        return volume

    raise DegenerateVolumeError(
        f"Failed to generate a non-degenerate volume after {max_attempts} attempts: {last_error}"
    )


def save_volume_npz(path: Path, volume: npt.NDArray[np.integer], **metadata: object) -> None:
    """Save a voxel grid and optional scalar metadata to ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"volume": np.asarray(volume, dtype=np.uint8)}
    for key, value in metadata.items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def load_volume_npz(path: Path) -> tuple[VolumeArray, dict[str, np.ndarray]]:
    """Load a volume previously written by :func:`save_volume_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "volume" not in data:
            raise KeyError(f"{path} does not contain a 'volume' array")
        volume = np.asarray(data["volume"], dtype=np.uint8)
        meta = {key: data[key] for key in data.files if key != "volume"}
    return volume, meta


def central_slices(volume: npt.NDArray[np.integer]) -> dict[str, npt.NDArray[np.integer]]:
    """Return mid-plane slices in x, y, and z."""
    nx, ny, nz = volume.shape
    return {
        "yz_at_xmid": np.asarray(volume[nx // 2, :, :]),
        "xz_at_ymid": np.asarray(volume[:, ny // 2, :]),
        "xy_at_zmid": np.asarray(volume[:, :, nz // 2]),
    }
