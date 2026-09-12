"""Illustrative physical constants and documented sentinels.

Values here are **configurable defaults**, not NASA-approved material properties,
unless a source is cited in ``docs/references.md``.
"""

from __future__ import annotations

# Dimensionless / numerical sentinels
NO_CONNECTED_PATH: float = float("nan")
SOLID_LABEL: int = 1
PORE_LABEL: int = 0

# Illustrative thermal conductivities [W/(m·K)]
# Amorphous / rayon-derived carbon fibers in low-density felts are far more
# resistive than crystalline graphite. These numbers are placeholders for
# comparative numerical studies only.
ILLUSTRATIVE_K_SOLID_W_M_K: float = 8.0
ILLUSTRATIVE_K_PORE_W_M_K: float = 0.03

# Illustrative effective oxygen diffusivity in a pore network [m^2/s].
# Free-air O2 diffusivity is ~2e-5 m^2/s; porous-media Deff is lower.
ILLUSTRATIVE_DEFF_M2_S: float = 5.0e-8

# Illustrative first-order reaction coefficient [1/s] in the reduced model
# dC/dt = Deff ∇²C − k C, applied only at the solid–pore interface.
ILLUSTRATIVE_K_REACTION_1_S: float = 20.0

# 3-D explicit diffusion CFL-like limit: Fo = Deff * dt / dx^2 < 1/6
EXPLICIT_DIFFUSION_FO_LIMIT_3D: float = 1.0 / 6.0
FO_SAFETY_FACTOR: float = 0.95

# Broad physical-consistency bounds for predicted k_eff [W/(m·K)]
K_EFF_MIN_W_M_K: float = 1.0e-4
K_EFF_MAX_W_M_K: float = 50.0
