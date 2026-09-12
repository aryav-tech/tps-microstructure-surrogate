# Methods

## 1. Synthetic microstructure

`tps_surrogate.microstructure.generate_fiber_volume` deposits random cylindrical fibers into a 3-D `uint8` grid until the pore fraction is within `porosity_tolerance` of the requested target, or until `n_fibers_max` is reached.

- Fiber axes are mostly in the *xy* plane. `in_plane_spread_deg` and `out_of_plane_spread_deg` control azimuthal and elevation half-widths.
- Cylinders are rasterized by thresholding Euclidean distance to a line segment (NumPy only).
- Volumes with `min(shape) < 12` use a faster stick-like fallback intended for demo grids.
- Fully solid, fully pore, or off-target volumes raise `DegenerateVolumeError` after a bounded number of retries.

Porosity is `mean(volume == 0)`. Utilities: `save_volume_npz`, `load_volume_npz`, `central_slices`.

## 2. Descriptors

`tps_surrogate.descriptors.compute_descriptors` returns a flat dictionary:

| field | meaning | units |
|---|---|---|
| porosity, solid_fraction | phase fractions | 1 |
| interfacial_face_count | 6-connected solid–pore faces | voxels |
| interfacial_area_per_volume_1_m | faces converted via `units.py` | 1/m |
| pore_connected_fraction | pore voxels in components spanning z=0 to z=max | 1 |
| tortuosity_proxy | geodesic 6-path length / (nz−1); **NaN if no path** | 1 |
| continuity_{x,y,z} | solid–solid neighbor fraction along each axis | 1 |
| mean/max pore radius | `scipy.ndimage.distance_transform_edt` on pores | voxels and m |

`validate_descriptor_dict` enforces required keys. NaN is allowed only for documented path/radius fields.

## 3. Diffusion–reaction baseline

Explicit 3-D finite difference on pore voxels:

```
C^{n+1} = C^n + dt [ Deff ∇²C^n − k_reaction C^n 1_interface ]
```

- Interface mask: pore voxels with a 6-neighbor solid.
- Solid voxels hold C = 0 and act as no-flux walls.
- z = 0 pores: C = 1. x/y faces: zero flux. z = max: `zero_flux`, `fixed_zero`, or `fixed_one`.
- Stability: Fourier number `Fo = Deff dt / dx²` must be `< 0.95/6`. Otherwise `DiffusionStabilityError`.
- Oxidation penetration depth: largest z whose mean cumulative reaction ≥ `oxidation_threshold`.
- Damage is cumulative reaction normalized by its maximum. Geometry is frozen unless `enable_degradation` is true.

Physical conversions (depth in meters, etc.) go through `UnitSystem`.

### Analytical check (details in `verification.md`)

With reaction off, an all-pore slab, C(0)=1, C(L)=0, C(z,0)=0, the solver is compared to

```
C(z,t) = 1 − z/L − (2/π) Σ_n (1/n) sin(nπz/L) exp(−n²π² Deff t / L²)
```

Default test tolerance: RMSE < 0.05 on interior nodes.

## 4. Effective conductivity baseline

Steady `div(k ∇T) = 0` with T = 1 and T = 0 on opposite faces and insulated remaining faces. Face `k` is the harmonic mean. The system is a SciPy CSR matrix solved by CG or BiCGSTAB.

`k_eff = q_avg L / ΔT` with ΔT = 1 and L = (n−1) dx. Because dx is uniform it cancels; reported `k_eff` has units W/(m·K) inherited from the assigned voxel conductivities.

If the sparse solve fails, a **volume-weighted arithmetic mean** is returned and flagged. That fallback is a crude parallel bound, not a PDE solution.

### Analytical checks

- Homogeneous solid or pore: `k_eff ≈ k_assigned` (relative error ≲ 3% on 8³).
- Layered slab perpendicular to the flux: series / harmonic mean.
- Layered slab parallel to the flux: parallel / arithmetic mean.

Tolerances and numbers: `verification.md`.

## 5. Dataset

Latin-hypercube sampling over porosity, fiber radius, both angular spreads, `Deff`, `k_reaction`, and a modest simulation-time scale. Each draw:

1. generate volume (skip + record `DegenerateVolumeError`);
2. descriptors;
3. diffusion–reaction;
4. conductivity in x, y, z;
5. write CSV tables, per-sample NPZ, and a manifest (seed, git hash, timestamps, paths).

Failed samples never abort the experiment.

## 6. Surrogate

Targets (default):

- `oxidation_penetration_depth_voxels`
- `cumulative_reaction_normalized`
- `k_eff_z`
- `tortuosity_proxy`

Estimators: `DummyRegressor(mean)`, `RandomForestRegressor`, `HistGradientBoostingRegressor` (multi-output). This is **not** a neural operator.

Validation:

- Repeated k-fold CV (`n_cv_folds` × `n_cv_repeats`) → mean **and** standard deviation of MAE, RMSE, R², MAPE, max AE.
- One held-out test split for a final point estimate. The CV spread is the honest uncertainty on small N.

Random-forest tree-to-tree standard deviation is stored as an uncertainty proxy. Feature importances are written when the estimator exposes them.

After prediction, `evaluation.physical_consistency_checks` flags negative k, k outside configured bounds, depth outside `[0, nz]`, negative cumulative reaction, and a Spearman test that higher reaction rate should not systematically produce lower damage.

## 7. Inverse design

Random search over the **same four geometric parameters**, not voxel-level topology optimization. The surrogate scores candidates; the numerical baseline is re-run on the top `n_confirm` designs. A Pareto flag marks non-dominated `(k_eff_z, oxidation_depth)` pairs (both minimized).

## 8. Grid / time-step refinement

`tests/test_convergence.py` (pytest mark `slow`) repeats the diffusion analytical problem at two time steps and the homogeneous conductivity problem on 6³ / 8³ / 10³ grids. Expected behavior: diffusion RMSE does not increase when `dt` is halved at fixed final time; homogeneous `k_eff` stays within a few percent of the assigned k. Summary of local runs belongs in `verification.md`.
