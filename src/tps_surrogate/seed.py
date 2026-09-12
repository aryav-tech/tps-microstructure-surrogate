"""Deterministic seeding for numpy, the stdlib RNG, and scikit-learn."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed process-wide RNGs and return a numpy Generator for local use."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def spawn_seed(rng: np.random.Generator) -> int:
    """Draw a 32-bit seed from an existing generator."""
    return int(rng.integers(0, 2**31 - 1))
