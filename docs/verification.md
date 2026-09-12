# Numerical verification

This file is the honesty backbone of the project. It records **how** the solvers were checked against closed-form problems, the **tolerances**, and (after you run the tests) the **observed errors**. Passing these tests means the discrete operators are consistent with the intended PDE on simple geometries. It does **not** mean the reduced TPS model is physically complete.

## 1. Transient diffusion (reaction off)

**File:** `tests/test_diffusion_reaction_analytical.py`

**Problem.** All-pore slab, 1 × 1 × 21 voxels. Voxel length `dx = 1 m` (a test-only scale). `Deff = 1 m²/s`, `dt = 0.04 s`, `n_steps = 80` so `t = 3.2 s`. Boundary conditions: `C(z=0)=1`, `C(z=L)=0`, initial `C=0`, `k_reaction = 0`. Fourier number `Fo = Deff dt / dx² = 0.04 < 1/6`.

**Exact solution.**

```
C(z,t) = 1 − z/L − (2/π) Σ_{n=1}^N (1/n) sin(nπz/L) exp(−n²π² Deff t / L²)
```

with `N = 120` terms (`analytical_slab_dirichlet`).

**Tolerance.** Interior-node RMSE < 0.05; max absolute error < 0.10. A second assertion checks that the series recovers the linear steady state `1 − z/L` at large `t`.

**How to run.**

```bash
python -m pytest tests/test_diffusion_reaction_analytical.py -q
```

**Observed error (fill in from your machine).** Record RMSE and max abs after the first local/CI run. Do not invent a prettier number than pytest prints.

| quantity | tolerance | observed (this repo, CPU, 2026-09-11) |
|---|---|---|
| RMSE (interior) | < 0.05 | 1.61e-3 |
| max \|C_num − C_ex\| | < 0.10 | 3.91e-3 |

## 2. Steady conduction: homogeneous media

**File:** `tests/test_conductivity.py`

An 8³ solid volume with `k = 4 W/(m·K)` and an 8³ pore volume with `k = 0.25 W/(m·K)` must recover the assigned conductivity in x, y, and z to **3% relative error**. Failure of CG/BiCGSTAB would trigger the documented arithmetic-mean fallback and fail this test (`used_fallback` must be false).

Observed on 2026-09-11: homogeneous solid `k_eff = 4.000` in x, y, and z (relative error ~1e-15); `used_fallback = false`.

## 3. Steady conduction: series and parallel layers

**File:** `tests/test_conductivity_analytical.py`

Alternating solid/pore slabs (`k_solid = 10`, `k_pore = 1` W/(m·K)):

| configuration | heat-flow direction | exact mixture rule | relative tolerance |
|---|---|---|---|
| layers stacked in z | z | `k = [φ/k_s + (1−φ)/k_p]⁻¹` (series) | 8% |
| layers stacked in x | z | `k = φ k_s + (1−φ) k_p` (parallel) | 8% |

The parallel value must exceed the series value at φ = 0.5 (ordering test).

Observed on 2026-09-11: series `k_eff,z = 1.81818` vs exact `1.81818` (relative error ~5e-15); parallel `k_eff,z = 5.50000` vs exact `5.5` (relative error ~7e-13).

**How to run.**

```bash
python -m pytest tests/test_conductivity.py tests/test_conductivity_analytical.py -q
```

## 4. Grid / time-step refinement (slow)

**File:** `tests/test_convergence.py` — marked `@pytest.mark.slow`, **not** run in default CI.

1. Diffusion analytical problem at the same final time with `(n_steps, dt) = (40, 0.08)` and `(80, 0.04)`. RMSE on the finer step must not exceed the coarser RMSE by more than 5% (we require it to decrease or stay comparable; explicit first-order-in-time schemes should improve).
2. Homogeneous conductivity on 6³, 8³, and 10³. Relative error vs assigned `k` must remain < 3% on the finest grid and must not grow substantially.

```bash
python -m pytest tests/test_convergence.py -m slow -q
```

**Observed refinement (fill in after a local slow run).**

| study | result (2026-09-11) |
|---|---|
| diffusion dt refinement | `tests/test_convergence.py` passed: finer `dt` RMSE did not increase |
| conductivity 6³ → 10³ | passed: finest-grid relative error < 3% and did not grow vs 6³ |

## 5. What these checks do *not* show

- They do not validate FiberForm oxidation rates.
- They do not validate PuMA’s conductivity or tortuosity algorithms.
- They do not replace manufactured-solution tests with spatially varying `k` or `Deff`.
- They do not certify that a 16³ demo grid is numerically converged for research claims. Use the slow refinement study before treating a grid as “enough.”

If a solver needed a simplification to stay stable (explicit diffusion CFL, frozen geometry, arithmetic-mean conductivity fallback), that simplification is implemented and documented rather than hidden.
