#!/usr/bin/env python3
"""Train Dummy / RandomForest / HistGBM surrogates on a generated dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tps_surrogate.config import add_common_args, config_from_cli, configure_logging  # noqa: E402
from tps_surrogate.paths import resolve_output_dir  # noqa: E402
from tps_surrogate.pipeline import run_train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train tabular physics-guided surrogates.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging()
    config = config_from_cli(args)
    out = resolve_output_dir(config.output_dir)
    snapshot = run_train(config, out)
    print(f"Trained models on {snapshot['n_samples']} samples. Snapshot: {out / 'models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
