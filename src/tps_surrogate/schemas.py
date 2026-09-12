"""Dataclasses for configuration snapshots, descriptors, and solver results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

VolumeArray = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.floating]


@dataclass
class DomainConfig:
    shape: tuple[int, int, int]
    voxel_length_m: float

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError(f"domain.shape must have 3 ints, got {self.shape}")
        if any(int(v) < 4 for v in self.shape):
            raise ValueError(f"each domain dimension must be >= 4, got {self.shape}")
        self.shape = (int(self.shape[0]), int(self.shape[1]), int(self.shape[2]))
        if self.voxel_length_m <= 0.0:
            raise ValueError("voxel_length_m must be positive")


@dataclass
class MicrostructureConfig:
    n_samples: int
    porosity_range: tuple[float, float]
    fiber_radius_voxels_range: tuple[float, float]
    in_plane_spread_deg_range: tuple[float, float]
    out_of_plane_spread_deg_range: tuple[float, float]
    porosity_tolerance: float = 0.10
    max_generation_attempts: int = 8
    n_fibers_max: int = 120
    fiber_length_voxels: float = 24.0

    def __post_init__(self) -> None:
        _check_range("porosity_range", self.porosity_range, 0.05, 0.99)
        _check_range("fiber_radius_voxels_range", self.fiber_radius_voxels_range, 0.5, 20.0)
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.porosity_tolerance <= 0.0:
            raise ValueError("porosity_tolerance must be positive")
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be >= 1")


@dataclass
class DiffusionConfig:
    n_time_steps: int
    dt_s: float
    diffusivity_m2_s_range: tuple[float, float]
    reaction_rate_1_s_range: tuple[float, float]
    z_max_bc: str = "zero_flux"
    oxidation_threshold: float = 0.05
    enable_degradation: bool = False
    degradation_damage_threshold: float = 0.8
    degradation_max_fraction: float = 0.02
    snapshot_times: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])

    def __post_init__(self) -> None:
        if self.n_time_steps < 1:
            raise ValueError("n_time_steps must be >= 1")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        _check_range("diffusivity_m2_s_range", self.diffusivity_m2_s_range, 1e-14, 1e-3)
        _check_range("reaction_rate_1_s_range", self.reaction_rate_1_s_range, 0.0, 1e6)
        allowed = {"zero_flux", "fixed_zero", "fixed_one"}
        if self.z_max_bc not in allowed:
            raise ValueError(f"z_max_bc must be one of {sorted(allowed)}")


@dataclass
class ConductivityConfig:
    k_solid_w_m_k: float
    k_pore_w_m_k: float
    solver: str = "cg"
    tol: float = 1e-6
    maxiter: int = 2000
    k_bounds_w_m_k: tuple[float, float] = (1e-4, 50.0)

    def __post_init__(self) -> None:
        if self.k_solid_w_m_k <= 0.0 or self.k_pore_w_m_k <= 0.0:
            raise ValueError("conductivities must be positive")
        if self.solver not in {"cg", "bicgstab"}:
            raise ValueError("solver must be 'cg' or 'bicgstab'")


@dataclass
class DatasetSplitConfig:
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    n_cv_folds: int = 5
    n_cv_repeats: int = 3

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test fractions must sum to 1, got {total}")
        if self.n_cv_folds < 2:
            raise ValueError("n_cv_folds must be >= 2")
        if self.n_cv_repeats < 1:
            raise ValueError("n_cv_repeats must be >= 1")


@dataclass
class ModelHyperParams:
    random_forest: dict[str, Any] = field(default_factory=dict)
    hist_gbm: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    n_candidates: int = 32
    n_confirm: int = 5
    seed: int = 0


@dataclass
class EvaluationConfig:
    mape_eps: float = 1e-8


@dataclass
class ExperimentConfig:
    seed: int
    name: str
    output_dir: str
    domain: DomainConfig
    microstructure: MicrostructureConfig
    diffusion: DiffusionConfig
    conductivity: ConductivityConfig
    dataset: DatasetSplitConfig
    model: ModelHyperParams
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    concentration: FloatArray
    cumulative_reaction: FloatArray
    oxidation_profile: FloatArray
    oxidation_penetration_depth_voxels: float
    oxidation_penetration_depth_m: float
    damage: FloatArray
    porosity_change_proxy: float
    cumulative_reaction_normalized: float
    degraded_volume: VolumeArray | None
    n_time_steps: int
    dt_s: float
    diffusivity_m2_s: float
    reaction_rate_1_s: float
    snapshots: dict[str, FloatArray] = field(default_factory=dict)
    used_degradation: bool = False


@dataclass
class ConductivityResult:
    k_eff_x: float
    k_eff_y: float
    k_eff_z: float
    anisotropy_xy: float
    anisotropy_xz: float
    anisotropy_yz: float
    solver_used: str
    used_fallback: bool
    residual_z: float | None = None


def _check_range(name: str, pair: tuple[float, float], lo: float, hi: float) -> None:
    if len(pair) != 2:
        raise ValueError(f"{name} must be a (min, max) pair")
    a, b = float(pair[0]), float(pair[1])
    if a > b:
        raise ValueError(f"{name} min {a} is greater than max {b}")
    if a < lo - 1e-15 or b > hi + 1e-15:
        raise ValueError(f"{name}={pair} is outside allowed [{lo}, {hi}]")
