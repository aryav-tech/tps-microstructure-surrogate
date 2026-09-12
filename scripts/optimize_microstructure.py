#!/usr/bin/env python3
"""Parameter-space inverse design with surrogate scoring + baseline confirm."""

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
from tps_surrogate.pipeline import run_optimize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Random-search parameter-space optimization (not topology optimization)."
    )
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging()
    config = config_from_cli(args)
    out = resolve_output_dir(config.output_dir)
    frame = run_optimize(config, out)
    print(f"Scored {len(frame)} candidates. Table: {out / 'tables' / 'optimization_candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
