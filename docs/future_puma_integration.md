# Optional future PuMA / pumapy integration

The **core pipeline does not import PuMA**. Install it only if you want optional comparisons later. Windows users often find `pumapy` difficult to build; that is expected and is why this project is standalone.

## Install (optional extra)

```bash
pip install -e ".[puma]"
# or
pip install pumapy
```

PuMA also ships as a desktop application and as a conda/spack stack; follow the version of the docs that matches **your** install:

- Source: [https://github.com/nasa/puma](https://github.com/nasa/puma)
- Docs: [https://puma-nasa.readthedocs.io/](https://puma-nasa.readthedocs.io/)

**API names change between releases.** The snippets below are illustrative. Verify function names against the documentation for the version you installed before relying on them.

## Loading an example FiberForm TIFF

PuMA’s test data historically included FiberForm micro-CT TIFFs. After you locate a local example file (do not commit large TIFFs):

```python
# Illustrative only — confirm the import name for your pumapy version.
import pumapy as puma

ws = puma.import_3Dtiff("path/to/FiberForm_example.tif")
# ws.matrix is typically a 3-D ndarray of material IDs
```

Map PuMA labels onto this project’s convention (`1 = solid carbon`, `0 = pore`) **explicitly**. Do not assume ID 0/1 match.

A thin adapter (future work) should live in a new module such as `src/tps_surrogate/puma_bridge.py` and must be imported only inside a `try/except ImportError` so the core extra-less install stays green.

## Potential future comparisons (not implemented)

Compare, on the **same** voxel array:

| quantity | this repo | PuMA (typical capability; verify API) |
|---|---|---|
| porosity | `compute_porosity` | workspace porosity helper |
| through-thickness `k_eff` | `conductivity.effective_conductivity` | PuMA thermal conductivity solvers |
| tortuosity proxy | geodesic pore path / (nz−1) | PuMA tortuosity / random-walk tools |
| orientation / continuity | neighbor-continuity proxies | PuMA orientation / fiber-direction stats |

Any comparison table must state:

1. whether the geometry was synthetic (this repo) or micro-CT (PuMA example);
2. that disagreement can come from different discretizations, not from a “wrong” NASA code;
3. that this project still does **not** become experimentally validated by matching PuMA.

## What not to do

- Do not make `pumapy` a required dependency.
- Do not copy proprietary NASA flight reconstructions into `data/`.
- Do not claim PuMA results are this project’s results, or vice versa.
- Do not hard-code PuMA function names in CI.

## Suggested integration steps for a later PR

1. Add `puma_bridge.py` with `is_puma_available() -> bool`.
2. A script `scripts/compare_puma.py` that no-ops with a clear message when PuMA is missing.
3. One optional test marked `@pytest.mark.puma` skipped unless the extra is installed.
4. Update this document with the **exact** `pumapy` version and the function names you actually called.
