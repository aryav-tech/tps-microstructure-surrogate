# Notebooks

Jupyter notebooks are **optional exploration only**. All reusable project logic lives in `src/tps_surrogate/` and is launched from `scripts/`.

Do not add notebooks that re-implement the generator, solvers, or surrogate. If you create a notebook, import the installed package:

```python
from tps_surrogate.config import load_config
from tps_surrogate.microstructure import generate_fiber_volume
```

Notebook outputs and checkpoints are git-ignored.
