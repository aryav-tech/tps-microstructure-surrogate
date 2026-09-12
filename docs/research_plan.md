# Physics-Guided Surrogate Modeling of Oxidation and Thermal Transport in Porous Carbon-Fiber Thermal-Protection Materials

Working title (ISEF / science-fair board). This is a **pre-registration-style** plan. It does not contain fabricated numeric results.

## Objective

Given a synthetic porous carbon-fiber microstructure and controlled environmental parameters, predict (i) oxidation penetration depth and (ii) through-thickness effective thermal conductivity much faster than a numerical diffusion–reaction + conduction baseline, with errors characterized by repeated cross-validation.

## Primary research question

Can a physics-guided machine-learning surrogate predict oxidation penetration depth and through-thickness effective thermal conductivity of FiberForm-like porous carbon-fiber microstructures with low error and substantially lower inference time than a numerical diffusion–reaction baseline?

## Hypothesis

A physics-guided ML surrogate trained on numerical simulations of controlled synthetic fiber microstructures can predict oxidation penetration depth and through-thickness `k_eff,z` within a pre-registered error threshold while evaluating unseen material designs much faster than repeatedly solving the numerical model.

## Independent / dependent / controlled variables

**Independent (design / process, sampled by Latin hypercube)**

- Target porosity
- Fiber radius (voxels)
- In-plane angular spread
- Out-of-plane angular spread
- Effective diffusivity `Deff`
- First-order reaction coefficient `k_reaction`
- Simulation time (narrow scale around `n_steps * dt`)

**Dependent (baseline-computed targets)**

- Oxidation penetration depth (voxels and meters)
- Normalized cumulative interface reaction
- Effective conductivities `k_eff,x`, `k_eff,y`, `k_eff,z`
- Tortuosity proxy and porosity-change proxy

**Controlled**

- Domain shape and voxel length
- Boundary-condition family (C=1 at z=0; insulating sides)
- Assigned `k_solid`, `k_pore` (illustrative)
- Global random seed
- Solver tolerances and explicit-scheme CFL limit

## Baseline vs. surrogate

| item | baseline | surrogate |
|---|---|---|
| geometry | synthetic fiber voxels | descriptors + process parameters |
| oxidation | explicit diffusion–reaction PDE | tabular regression |
| conductivity | sparse steady conduction PDE | tabular regression |
| role | training labels + confirmation | fast evaluation / screening |

The surrogate is called “physics-guided” because its features, labels, and post-hoc consistency checks come from the numerical physical model — **not** because it is a neural operator.

## Statistical and validation plan

Do **not** treat a single train/validation/test split as the headline result on a dataset of tens to low hundreds of samples.

1. **Repeated k-fold CV** with `n_cv_folds` (default 5; demo 4) and `n_cv_repeats` (default 3; demo 2), different fold seeds. Report **mean and standard deviation** of MAE, RMSE, R², MAPE (where defined), and maximum absolute error for each target.
2. **Held-out test split** (`test_fraction`) for a final point estimate used in `summary.md`. Interpret it only together with the CV spread.
3. Compare every estimator to a `DummyRegressor(mean)` baseline.
4. Physical-consistency flags (non-negative k, bounds, depth in `[0, nz]`, non-negative cumulative reaction, Spearman monotonicity of reaction rate vs. damage). Flags are reported, never silently clipped away.
5. Runtime: mean baseline wall time per successful case vs. surrogate inference time; break-even count

   `N = (T_generate + T_train) / (t_baseline − t_infer)`

   with an explicit undefined state if the denominator is not positive.

6. Inverse-design confirmation: re-solve the numerical model on the top 5–10 surrogate-ranked parameter vectors and report absolute errors.

## Pre-registered success criteria (no fabricated numbers)

Declare these thresholds **before** looking at a full `default.yaml` campaign. Suggested starting criteria (edit if your mentor requires different values):

- CV mean R² for `k_eff,z` **greater than Dummy R²** and CV mean RMSE below 15% of the observed target standard deviation.
- CV mean RMSE for oxidation depth below 20% of the observed target standard deviation.
- Surrogate inference at least **10×** faster than the mean baseline wall time per case on the same CPU.
- Zero silent degenerate volumes in the manifest (failures recorded, not dropped).
- Analytical diffusion RMSE < 0.05 and layered-conductivity relative error < 8% (see `verification.md`).

If a criterion is missed, report the miss. Do not retune until it “passes” without saying so.

## Threats to validity

- **Internal:** small N, descriptor–target leakage (tortuosity is both a descriptor-like geometric quantity and a target), unstable explicit dt, conductivity fallback silently degrading labels (mitigated by a `conductivity_fallback` column).
- **Construct:** FiberForm-*like* cylinders are not tomographic FiberForm; first-order k is not laboratory oxidation kinetics.
- **External:** no claim of generalization to flight TPS, different fiber materials, or PuMA-computed quantities.
- **Statistical:** a lucky single split can look excellent; that is why CV repeats are mandatory.

## Limitations

See `model_assumptions.md`. In short: two-phase, no pyrolysis, no reentry CFD, no experimental validation, surrogate-vs-baseline only.

## Proposed ISEF board figures / tables

1. Pipeline diagram (README Mermaid).
2. Example xy / xz / yz slices of a synthetic volume.
3. Oxygen concentration and damage mid-slices.
4. Parity plots with CV error bars / spread stated in the caption.
5. Residual plots.
6. Runtime bar chart and break-even calculation.
7. Feature-importance bar chart.
8. Pareto scatter of `k_eff,z` vs. oxidation depth (parameter-space search).
9. Table: CV mean ± std vs. Dummy for each target.
10. Table: analytical verification tolerances and observed errors.
11. Table: failed-sample count and consistency flags.
12. Prominent disclaimer plaque: *validated against our numerical baseline, not experiment or NASA flight data.*

## Board language to avoid

- “We simulated reentry.”
- “This is PICA / NASA FiberForm performance.”
- “The model is validated.” (say **verified numerically** and **validated against the baseline**.)
- “PINO” or “neural operator” for the Random Forest.
