"""Approximate directional effective thermal conductivity.

Solves the steady conduction problem ``div(k grad T) = 0`` on the voxel grid
with T=1 / T=0 on opposite faces and insulating conditions on the others.

``k`` is piecewise constant: ``k_solid`` on carbon voxels and ``k_pore`` on
pore voxels. Face conductivities use the harmonic mean. Dirichlet nodes are
eliminated so the reduced system is symmetric positive definite and CG applies.

If the sparse solve fails, a documented volume-weighted arithmetic-mean
fallback is returned. That fallback is a crude parallel bound, not a
substitute for the PDE solve.

All conductivities are illustrative unless cited in ``docs/references.md``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import bicgstab, cg

from tps_surrogate.constants import SOLID_LABEL
from tps_surrogate.schemas import ConductivityConfig, ConductivityResult
from tps_surrogate.units import UnitSystem, conductivity_unit

LOGGER = logging.getLogger(__name__)

AxisName = Literal["x", "y", "z"]
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("conductivities must be positive")
    return 2.0 * a * b / (a + b)


def assign_conductivity(
    volume: npt.NDArray[np.integer],
    k_solid_w_m_k: float,
    k_pore_w_m_k: float,
) -> npt.NDArray[np.floating]:
    k = np.where(volume == SOLID_LABEL, k_solid_w_m_k, k_pore_w_m_k)
    return np.asarray(k, dtype=np.float64)


def _solve_direction(
    k_field: npt.NDArray[np.floating],
    axis: AxisName,
    *,
    solver: str,
    tol: float,
    maxiter: int,
) -> tuple[float, float, str]:
    """Return (k_eff, residual_norm, solver_tag).

    Dirichlet faces are eliminated from the unknown vector so the reduced
    operator is the standard symmetric 7-point stencil.
    """
    nx, ny, nz = k_field.shape
    axis_i = _AXIS_INDEX[axis]
    n_axis = k_field.shape[axis_i]
    dx = 1.0

    def axis_coord(i: int, j: int, k: int) -> int:
        return (i, j, k)[axis_i]

    def is_dirichlet(i: int, j: int, k: int) -> bool:
        ac = axis_coord(i, j, k)
        return ac == 0 or ac == n_axis - 1

    def bc_value(i: int, j: int, k: int) -> float:
        return 1.0 if axis_coord(i, j, k) == 0 else 0.0

    index_of: dict[tuple[int, int, int], int] = {}
    interior: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not is_dirichlet(i, j, k):
                    index_of[(i, j, k)] = len(interior)
                    interior.append((i, j, k))

    n_unk = len(interior)
    if n_unk == 0:
        # 2-voxel-thick domain: flux from the two Dirichlet faces only
        t_field = np.zeros(k_field.shape, dtype=np.float64)
        low = [slice(None)] * 3
        low[axis_i] = 0
        t_field[tuple(low)] = 1.0
        return _flux_keff(k_field, t_field, axis), 0.0, "dirichlet-only"

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_unk, dtype=np.float64)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    for p, (i, j, k) in enumerate(interior):
        kp = float(k_field[i, j, k])
        diag = 0.0
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz:
                continue
            kn = float(k_field[ni, nj, nk])
            coef = harmonic_mean(kp, kn) / (dx * dx)
            diag += coef
            if is_dirichlet(ni, nj, nk):
                rhs[p] += coef * bc_value(ni, nj, nk)
            else:
                q = index_of[(ni, nj, nk)]
                rows.append(p)
                cols.append(q)
                vals.append(-coef)
        rows.append(p)
        cols.append(p)
        vals.append(diag)

    a_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_unk, n_unk))

    def _iterative(solver_fn, tag: str):
        try:
            vec, info = solver_fn(a_mat, rhs, rtol=tol, maxiter=maxiter)
        except TypeError:
            vec, info = solver_fn(a_mat, rhs, tol=tol, maxiter=maxiter)
        return vec, info, tag

    if solver == "bicgstab":
        t_int, info, tag = _iterative(bicgstab, "bicgstab")
    else:
        t_int, info, tag = _iterative(cg, "cg")
    if info != 0:
        raise RuntimeError(f"{tag} failed with info={info}")

    residual = float(np.linalg.norm(a_mat @ t_int - rhs))
    t_field = np.zeros(k_field.shape, dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if is_dirichlet(i, j, k):
                    t_field[i, j, k] = bc_value(i, j, k)
                else:
                    t_field[i, j, k] = t_int[index_of[(i, j, k)]]

    k_eff = _flux_keff(k_field, t_field, axis)
    return k_eff, residual, tag


def _flux_keff(
    k_field: npt.NDArray[np.floating],
    temperature: npt.NDArray[np.floating],
    axis: AxisName,
) -> float:
    """k_eff = q_avg * L / ΔT with ΔT = 1 and L = (n-1)*dx, dx=1 voxel."""
    axis_i = _AXIS_INDEX[axis]
    n_axis = k_field.shape[axis_i]
    if n_axis < 2:
        raise ValueError("need at least 2 voxels along the gradient axis")
    slicer0 = [slice(None), slice(None), slice(None)]
    slicer1 = [slice(None), slice(None), slice(None)]
    slicer0[axis_i] = 0
    slicer1[axis_i] = 1
    k0 = k_field[tuple(slicer0)]
    k1 = k_field[tuple(slicer1)]
    kf = 2.0 * k0 * k1 / (k0 + k1)
    dt = temperature[tuple(slicer0)] - temperature[tuple(slicer1)]
    q = kf * dt
    q_avg = float(np.mean(q))
    length = float(n_axis - 1)
    return q_avg * length


def volume_weighted_k(
    volume: npt.NDArray[np.integer],
    k_solid_w_m_k: float,
    k_pore_w_m_k: float,
) -> float:
    """Arithmetic (parallel) mixture rule used only as a solver fallback."""
    solid_frac = float(np.mean(volume == SOLID_LABEL))
    return solid_frac * k_solid_w_m_k + (1.0 - solid_frac) * k_pore_w_m_k


def effective_conductivity(
    volume: npt.NDArray[np.integer],
    *,
    k_solid_w_m_k: float,
    k_pore_w_m_k: float,
    solver: str = "cg",
    tol: float = 1e-6,
    maxiter: int = 2000,
    axes: tuple[AxisName, ...] = ("x", "y", "z"),
) -> ConductivityResult:
    """Solve for ``k_eff`` in the requested directions [W/(m·K)]."""
    if volume.ndim != 3:
        raise ValueError("volume must be 3-D")
    k_field = assign_conductivity(volume, k_solid_w_m_k, k_pore_w_m_k)
    values: dict[str, float] = {}
    residual_z = None
    used_fallback = False
    solver_used = solver
    for axis in axes:
        try:
            keff, residual, tag = _solve_direction(
                k_field, axis, solver=solver, tol=tol, maxiter=maxiter
            )
            values[axis] = keff
            solver_used = tag
            if axis == "z":
                residual_z = residual
        except Exception as exc:  # noqa: BLE001 — documented fallback
            LOGGER.warning(
                "Sparse conductivity solve failed for axis %s (%s); "
                "using volume-weighted arithmetic-mean fallback. "
                "This is a crude parallel bound, not a PDE solution.",
                axis,
                exc,
            )
            values[axis] = volume_weighted_k(volume, k_solid_w_m_k, k_pore_w_m_k)
            used_fallback = True
            solver_used = "volume_weighted_fallback"

    kx = values.get("x", float("nan"))
    ky = values.get("y", float("nan"))
    kz = values.get("z", float("nan"))

    def _ratio(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-30:
            return float("nan")
        return float(a / b)

    LOGGER.debug(
        "k_eff = (%.4g, %.4g, %.4g) %s fallback=%s",
        kx,
        ky,
        kz,
        conductivity_unit(),
        used_fallback,
    )
    return ConductivityResult(
        k_eff_x=float(kx),
        k_eff_y=float(ky),
        k_eff_z=float(kz),
        anisotropy_xy=_ratio(kx, ky),
        anisotropy_xz=_ratio(kx, kz),
        anisotropy_yz=_ratio(ky, kz),
        solver_used=solver_used,
        used_fallback=used_fallback,
        residual_z=residual_z,
    )


def run_from_config(
    volume: npt.NDArray[np.integer],
    config: ConductivityConfig,
    voxel_length_m: float | None = None,
) -> ConductivityResult:
    """``voxel_length_m`` is accepted for API symmetry; k_eff is already SI."""
    if voxel_length_m is not None:
        UnitSystem(voxel_length_m=voxel_length_m)
    return effective_conductivity(
        volume,
        k_solid_w_m_k=config.k_solid_w_m_k,
        k_pore_w_m_k=config.k_pore_w_m_k,
        solver=config.solver,
        tol=config.tol,
        maxiter=config.maxiter,
    )


def layered_volume(
    shape: tuple[int, int, int],
    *,
    axis: AxisName,
    period: int = 2,
) -> npt.NDArray[np.uint8]:
    """Alternating solid/pore slabs stacked along ``axis`` (for analytical tests)."""
    volume = np.zeros(shape, dtype=np.uint8)
    coords = np.indices(shape)[_AXIS_INDEX[axis]]
    volume[coords % period < period // 2] = SOLID_LABEL
    return volume


def series_conductivity(k_a: float, k_b: float, frac_a: float) -> float:
    """Harmonic mean for layers perpendicular to the heat flux."""
    frac_a = float(frac_a)
    frac_b = 1.0 - frac_a
    return 1.0 / (frac_a / k_a + frac_b / k_b)


def parallel_conductivity(k_a: float, k_b: float, frac_a: float) -> float:
    """Arithmetic mean for layers parallel to the heat flux."""
    frac_a = float(frac_a)
    return frac_a * k_a + (1.0 - frac_a) * k_b
