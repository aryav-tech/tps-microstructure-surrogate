"""Repository and output-path helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def project_root() -> Path:
    """Return the repository root (directory that contains ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate pyproject.toml from tps_surrogate.paths")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_dir(output_dir: str | Path, *, root: Path | None = None) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = (root or project_root()) / path
    return path


def git_commit_hash(*, fallback: str = "unknown") -> str:
    """Best-effort short git hash for dataset manifests. Never raises."""
    try:
        root = project_root()
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or fallback
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return fallback


def standard_output_layout(output_dir: Path) -> dict[str, Path]:
    """Create the conventional figures/models/reports/tables layout."""
    layout = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "reports": output_dir / "reports",
        "tables": output_dir / "tables",
        "volumes": output_dir / "volumes",
        "processed": output_dir / "processed",
    }
    for path in layout.values():
        ensure_dir(path)
    return layout
