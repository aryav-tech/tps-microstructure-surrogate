#!/usr/bin/env python3
"""Generate the tabular + NPZ dataset from a YAML experiment config."""

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
from tps_surrogate.pipeline import run_generate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the numerical-baseline dataset.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging()
    config = config_from_cli(args)
    out = resolve_output_dir(config.output_dir)
    manifest = run_generate(config, out)
    print(f"Wrote dataset with {manifest['n_success']} samples ({manifest['n_failed']} failed) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
