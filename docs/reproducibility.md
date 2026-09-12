# Reproducibility

## Lockfile

Exact versions used to produce any reported numbers should come from `requirements-lock.txt` (generated with `pip freeze` after a clean install). The GitHub Actions workflow installs that lockfile before `pytest`.

If you change dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip freeze | grep -v -i tps-microstructure > requirements-lock.txt
```

Commit the updated lockfile in the same PR as the dependency change.

## Seeds

`ExperimentConfig.seed` is applied through `tps_surrogate.seed.seed_everything` (stdlib `random`, NumPy, `PYTHONHASHSEED`). Dataset sample seeds are drawn from that generator and stored per row. Surrogate `random_state` uses the same global seed. Optimization uses `optimization.seed`.

## What to archive with a result

1. The YAML config (`configs/*.yaml` plus any CLI overrides).
2. `outputs/*/processed/manifest.json` (git hash, timestamps, failed samples).
3. `outputs/*/models/surrogate_snapshot.json` (feature/target names, split ids, CV metrics).
4. `requirements-lock.txt` and the git commit.
5. `outputs/*/reports/summary.md`.

Do **not** commit large NPZ volumes or trained `joblib` files; they are git-ignored. Re-generate them with the lockfile and the recorded seed.

## Platforms

The core pipeline is CPU-only and does not require PuMA or PyTorch. CI runs on `ubuntu-latest` / Python 3.11. Local Python 3.11 or 3.12 should match if you install from the lockfile. Windows users who cannot build optional `pumapy` can still run every core command.

## Commands that must work from the repository root

```bash
python scripts/run_demo.py --config configs/small_demo.yaml --overwrite
python scripts/generate_dataset.py --config configs/default.yaml
python scripts/train_surrogate.py --config configs/default.yaml
python scripts/evaluate_model.py --config configs/default.yaml
python scripts/optimize_microstructure.py --config configs/default.yaml
python scripts/run_all.py --config configs/default.yaml --overwrite
python -m pytest -m "not slow"
ruff check src scripts tests
```

## Collaboration

- Work on a feature branch; open a pull request; do not push unreviewed experimental dumps to `main`.
- Issues should cite the config name and the git hash from the manifest.
- Never commit `.env` files, tokens, or credentials. None are required by this project.
