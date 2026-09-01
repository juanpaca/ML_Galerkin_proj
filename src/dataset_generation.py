"""Production dataset generation for RFB bubble training.

Orchestrates parameter sampling, reference FD solves, preprocessing,
and train/val/test splitting for the two bubble modes
(b̂ = L⁻¹(1) and b̃ = L⁻¹(ξ)).  Supports constant and variable ε.
The two-mode system spans the full RFB space for any P1 source
(see the GD-abstract theory).

Typical usage::

    from src.dataset_generation import generate_dataset

    dataset = generate_dataset(
        n_samples=5000,
        h=1/16,
        eps_range=(1e-6, 1.0),
        beta_range=(1.0, 1.0),
        sigma_range=(0.0, 10.0),
        strategy="lhs",
        val_split=0.15,
        test_split=0.15,
        seed=42,
    )
    # dataset["train"], dataset["val"], dataset["test"]
    # Each is a dict with keys per mode: "constant", "xi"
    # Plus dataset["metadata"] and dataset["scaler"]
"""

from __future__ import annotations

import json
import os
import warnings
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from scipy.stats import qmc

from src.rfb_bubble import KANBubble1D, MultiKANBubble1D
from src.rfb_local import (
    solve_reference_rfb, local_parameters, interpolate_target,
    evaluate_diffusion_profile, trapezoidal_weights,
)
from src.rfb_training import (
    generate_rfb_training_data_cs,
    save_training_data,
    load_training_data,
    DATASET_SUBDIR,
    DTYPE,
)

# Do not alter the warning policy of applications importing this module.

DATASET_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Parameter sampling strategies
# ---------------------------------------------------------------------------


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class ParameterSpace:
    """Defines the parameter ranges and sampling strategy.

    Attributes
    ----------
    eps_range : tuple[float, float]
        (min, max) for diffusion coefficient.
    beta_range : tuple[float, float]
        (min, max) for advection coefficient.
    sigma_range : tuple[float, float]
        (min, max) for reaction coefficient.
    h : float
        Element length (fixed for a given mesh).
    eps_is_log : bool
        If True, sample ε log-uniformly.
    """
    eps_range: tuple[float, float] = (1e-6, 1.0)
    beta_range: tuple[float, float] = (1.0, 1.0)
    sigma_range: tuple[float, float] = (0.0, 0.0)
    h: float = 1 / 16
    eps_is_log: bool = True

    @property
    def n_dims(self) -> int:
        return 3  # eps, beta, sigma


def sample_parameters_lhs(
    space: ParameterSpace,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube Sampling of (eps, beta, sigma).

    Returns array of shape (n_samples, 3).  ε is sampled in log-space
    if ``space.eps_is_log`` is True.
    """
    sampler = qmc.LatinHypercube(d=space.n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    eps_min, eps_max = space.eps_range
    beta_min, beta_max = space.beta_range
    sigma_min, sigma_max = space.sigma_range
    if space.eps_is_log:
        log_eps_low = np.log10(max(eps_min, 1e-16))
        log_eps_high = np.log10(eps_max)
        if abs(log_eps_high - log_eps_low) < 1e-15:
            eps = np.full(n_samples, eps_min)
        else:
            eps = 10.0 ** qmc.scale(
                samples[:, [0]], log_eps_low, log_eps_high,
            ).ravel()
    else:
        eps = qmc.scale(samples[:, [0]], eps_min, eps_max).ravel()
    if abs(beta_max - beta_min) < 1e-15:
        beta = np.full(n_samples, beta_min)
    else:
        beta = qmc.scale(samples[:, [1]], beta_min, beta_max).ravel()
    if abs(sigma_max - sigma_min) < 1e-15:
        sigma = np.full(n_samples, sigma_min)
    else:
        sigma = qmc.scale(samples[:, [2]], sigma_min, sigma_max).ravel()
    return np.column_stack([eps, beta, sigma])


def sample_parameters_stratified(
    space: ParameterSpace,
    n_samples: int,
    seed: int | None = None,
    eps_decade_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Stratified sampling with configurable per-decade allocation for ε.

    Parameters
    ----------
    space : ParameterSpace
        Parameter ranges.
    n_samples : int
        Total number of samples.
    seed : int or None
        Random seed.
    eps_decade_weights : ndarray or None
        Weights for each decade in the ε range.  Length must equal
        the number of decades spanned by ``space.eps_range``.
        If None, uniform allocation per decade is used.

    Example
    -------
    >>> space = ParameterSpace(eps_range=(1e-6, 1.0))
    >>> # Heavier weight on advection-dominated regimes
    >>> weights = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    >>> params = sample_parameters_stratified(space, 1000, eps_decade_weights=weights)
    """
    rng = _rng(seed)
    eps_min, eps_max = space.eps_range
    log_min = np.log10(max(eps_min, 1e-16))
    log_max = np.log10(eps_max)
    n_decades = max(1, int(np.ceil(log_max - log_min)))
    log_edges = np.linspace(log_min, log_max, n_decades + 1)
    if eps_decade_weights is None:
        eps_decade_weights = np.ones(n_decades) / n_decades
    else:
        eps_decade_weights = np.asarray(eps_decade_weights, dtype=float)
        eps_decade_weights /= eps_decade_weights.sum()
    if len(eps_decade_weights) != n_decades:
        raise ValueError(
            f"eps_decade_weights length {len(eps_decade_weights)} != "
            f"n_decades {n_decades}"
        )
    counts = np.maximum(
        1, (eps_decade_weights * n_samples).astype(int)
    )
    counts[-1] += n_samples - counts.sum()
    all_params = []
    for i, c in enumerate(counts):
        eps_strat = 10.0 ** rng.uniform(log_edges[i], log_edges[i + 1], c)
        beta_strat = rng.uniform(*space.beta_range, c)
        sigma_strat = rng.uniform(*space.sigma_range, c)
        all_params.append(np.column_stack([eps_strat, beta_strat, sigma_strat]))
    result = np.vstack(all_params)
    rng.shuffle(result)
    return result[:n_samples]


def sample_parameters_grid(
    space: ParameterSpace,
    n_per_dim: int | tuple[int, int, int] = 20,
) -> np.ndarray:
    """Full factorial grid of (eps, beta, sigma).  ε is log-spaced.

    Use this for systematic coverage rather than random sampling.

    Returns
    -------
    ndarray of shape (n_eps * n_beta * n_sigma, 3).
    """
    if isinstance(n_per_dim, int):
        n_per_dim = (n_per_dim, n_per_dim, n_per_dim)
    eps_min, eps_max = space.eps_range
    if space.eps_is_log:
        eps = np.logspace(np.log10(max(eps_min, 1e-16)), np.log10(eps_max), n_per_dim[0])
    else:
        eps = np.linspace(eps_min, eps_max, n_per_dim[0])
    beta = np.linspace(*space.beta_range, n_per_dim[1])
    sigma = np.linspace(*space.sigma_range, n_per_dim[2])
    EE, BB, SS = np.meshgrid(eps, beta, sigma, indexing="ij")
    return np.column_stack([EE.ravel(), BB.ravel(), SS.ravel()])


def sample_parameters_log_pe_rho(
    pe_range: tuple[float, float] = (0.3, 312.0),
    rho_range: tuple[float, float] = (0.01, 380.0),
    n_samples: int = 5000,
    h: float = 1 / 16,
    beta: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample uniformly in log(eps) x log(sigma) space, derive (Pe, rho).

    Given fixed h and beta, the mapping is:
        Pe = beta * h / (2 * eps)
        rho = sigma * eps / h**2

    Parameters
    ----------
    pe_range : (Pe_min, Pe_max)
        Péclet number range (both > 0).  Used for reference only.
    rho_range : (rho_min, rho_max)
        Reaction number range (both > 0).  Used for reference only.
    n_samples : int
        Number of samples.
    h : float
        Element length.
    beta : float
        Advection coefficient.
    seed : int or None
        Random seed.

    Returns
    -------
    ndarray of shape (n_samples, 3) : (eps, beta, sigma) columns,
        compatible with the existing FD solve pipeline.
    """
    rng = _rng(seed)

    log_pe = rng.uniform(np.log10(pe_range[0]), np.log10(pe_range[1]), n_samples)
    log_rho = rng.uniform(np.log10(rho_range[0]), np.log10(rho_range[1]), n_samples)

    pe = 10.0 ** log_pe
    rho = 10.0 ** log_rho

    eps = beta * h / (2.0 * pe)
    sigma = rho * eps / (h ** 2)

    return np.column_stack([eps, np.full(n_samples, beta), sigma])


# ---------------------------------------------------------------------------
# Variable ε profile families
# ---------------------------------------------------------------------------


def eps_profile_constant(
    xi: np.ndarray, eps_base: float
) -> np.ndarray:
    """Constant diffusion profile."""
    return np.full_like(xi, eps_base, dtype=float)


def eps_profile_sinusoidal(
    xi: np.ndarray,
    eps_mean: float,
    amplitude_ratio: float = 0.5,
    n_periods: int = 2,
) -> np.ndarray:
    """Sinusoidal variation around eps_mean."""
    return eps_mean * (1.0 + amplitude_ratio * np.sin(n_periods * np.pi * xi))


def eps_profile_layered(
    xi: np.ndarray,
    eps_mean: float,
    n_layers: int = 4,
    contrast: float = 10.0,
    seed: int | None = None,
) -> np.ndarray:
    """Piecewise-constant layered profile."""
    rng = _rng(seed)
    edges = np.sort(rng.uniform(0.0, 1.0, n_layers - 1))
    edges = np.concatenate([[0.0], edges, [1.0]])
    factors = contrast ** rng.uniform(-1.0, 1.0, n_layers)
    profile = np.zeros_like(xi)
    for i in range(n_layers):
        mask = (xi >= edges[i]) & (xi < edges[i + 1])
        profile[mask] = eps_mean * factors[i]
    profile[xi >= edges[-1]] = eps_mean * factors[-1]
    return profile


def eps_profile_smooth_random(
    xi: np.ndarray,
    eps_mean: float,
    amplitude_ratio: float = 0.3,
    length_scale: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """Smooth random perturbation via Gaussian process (squared-exponential kernel)."""
    rng = _rng(seed)
    n = len(xi)
    dist = np.abs(xi[:, None] - xi[None, :])
    K = np.exp(-0.5 * (dist / length_scale) ** 2)
    K += 1e-8 * np.eye(n)
    perturbation = rng.multivariate_normal(np.zeros(n), K)
    perturbation = perturbation / np.std(perturbation)
    return eps_mean * (1.0 + amplitude_ratio * perturbation)


EPS_PROFILE_FAMILIES: dict[str, Callable] = {
    "constant": eps_profile_constant,
    "sinusoidal": eps_profile_sinusoidal,
    "layered": eps_profile_layered,
    "smooth_random": eps_profile_smooth_random,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _build_parameter_sweep(
    params: np.ndarray,
    h: float,
    residual_mode: str,
    n_fd_points: int = 400,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Generate reference FD solutions for a batch of (eps, beta, sigma)."""
    n = len(params)
    samples = []
    for i in range(n):
        eps_i, beta_i, sigma_i = params[i]
        target = solve_reference_rfb(
            float(eps_i), float(beta_i), float(sigma_i), h,
            residual_mode=residual_mode,
            n_points=n_fd_points,
        )
        pe, rho = local_parameters(float(eps_i), float(beta_i), float(sigma_i), h)
        target["pe"] = pe
        target["rho"] = rho
        target["idx"] = i
        samples.append(target)
        if progress_callback is not None:
            progress_callback(i + 1, n)
    return samples


def _build_variable_eps_sweep(
    params: np.ndarray,
    h: float,
    eps_profile_fn: Callable,
    n_eps: int,
    residual_mode: str,
    n_fd_points: int = 400,
    eps_scale_per_sample: bool = True,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Generate reference FD solutions for variable ε profiles."""
    rng = _rng(seed)
    xi_fd = np.linspace(0.0, 1.0, n_fd_points)
    xi_eps = _gauss_legendre_01(n_eps)
    n = len(params)
    samples = []
    for i in range(n):
        eps_mean_i, beta_i, sigma_i = params[i]
        if eps_scale_per_sample:
            scale = 10.0 ** rng.uniform(-2.0, 0.0)
        else:
            scale = 1.0
        profile_seed = int(rng.integers(0, 2**32 - 1))
        try:
            eps_on_xi = np.asarray(
                eps_profile_fn(xi_fd, eps_mean_i, seed=profile_seed), dtype=float
            )
        except TypeError:
            eps_on_xi = np.asarray(eps_profile_fn(xi_fd, eps_mean_i), dtype=float)
        eps_on_xi = evaluate_diffusion_profile(eps_on_xi * scale, xi_fd)
        eps_avg = float(np.sum(
            trapezoidal_weights(xi_fd) * eps_on_xi
        ))
        target = solve_reference_rfb(
            eps_on_xi, float(beta_i), float(sigma_i), h,
            residual_mode=residual_mode,
            n_points=n_fd_points,
        )
        eps_at_sample = np.interp(xi_eps, xi_fd, eps_on_xi)
        target["eps_ratios"] = np.asarray(eps_at_sample / eps_avg, dtype=float)
        target["pe"] = (
            float(abs(beta_i) * h / (2.0 * eps_avg)) if eps_avg > 0 else np.inf
        )
        target["rho"] = (
            float(sigma_i * h * h / eps_avg) if eps_avg > 0 else np.inf
        )
        target["idx"] = i
        samples.append(target)
        if progress_callback is not None:
            progress_callback(i + 1, n)
    return samples


def _gauss_legendre_01(n: int) -> np.ndarray:
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0)


# ---------------------------------------------------------------------------
# Preprocessing (standardization)
# ---------------------------------------------------------------------------


@dataclass
class DataScaler:
    """Minimal standard scaler for RFB inputs (Pe, ρ) and optionally ε_ratios.

    Attributes
    ----------
    mean_ : ndarray
        Per-feature mean.
    scale_ : ndarray
        Per-feature standard deviation (clamped to avoid division by zero).
    features_ : list[str]
        Feature names.
    """
    mean_: np.ndarray = field(default_factory=lambda: np.array([]))
    scale_: np.ndarray = field(default_factory=lambda: np.array([]))
    features_: list[str] = field(default_factory=list)

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> DataScaler:
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std < 1e-15, 1.0, std)
        if feature_names is not None:
            self.features_ = feature_names
        else:
            self.features_ = [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "features": self.features_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataScaler:
        obj = cls()
        obj.mean_ = np.array(d["mean"])
        obj.scale_ = np.array(d["scale"])
        obj.features_ = d["features"]
        return obj


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------


def _pe_regime(pe: float) -> str:
    if np.isinf(pe) or pe > 1e4:
        return "inf"
    if pe > 100.0:
        return "high"
    if pe > 1.0:
        return "mid"
    if pe > 0.01:
        return "low"
    return "diffusion"


def stratified_split(
    pe_values: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split by Péclet regime.

    Preserves the proportion of each regime across all three splits.

    Returns
    -------
    train_idx, val_idx, test_idx : ndarray
        Indices into the original dataset.
    """
    rng = _rng(seed)
    regimes = np.array([_pe_regime(pe) for pe in pe_values])
    unique_regimes = list(set(regimes))
    train_idx, val_idx, test_idx = [], [], []
    for regime in unique_regimes:
        mask = regimes == regime
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(test_split * n)))
        n_val = max(1, int(round(val_split * n)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = 1
            n_val = max(0, (n - n_train) // 2)
            n_test = n - n_train - n_val
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _pe_rho_cell(
    pe: float | np.ndarray,
    rho: float | np.ndarray,
    pe_bins: tuple[float, ...] = (0, 1, 10, 100, 1000, np.inf),
    rho_bins: tuple[float, ...] = (-np.inf, 0, 1, 100, np.inf),
) -> tuple[int, ...] | tuple[np.ndarray, np.ndarray]:
    """Assign each sample to a (Pe decade, ρ range) cell.

    Parameters
    ----------
    pe : float or ndarray
        Péclet number(s).
    rho : float or ndarray
        Reaction number(s).
    pe_bins : tuple
        Bin edges for Pe.  ``digitize(pe, pe_bins) - 1`` gives the index.
    rho_bins : tuple
        Bin edges for ρ.

    Returns
    -------
    tuple[int, int] or tuple[ndarray, ndarray]
        Cell coordinates (pe_idx, rho_idx) for each sample.
    """
    pe_a = np.atleast_1d(np.asarray(pe, dtype=float))
    rho_a = np.atleast_1d(np.asarray(rho, dtype=float))
    pe_a = np.where(np.isinf(pe_a), 1e15, pe_a)
    rho_a = np.where(np.isinf(rho_a), 1e15, rho_a)
    pe_idx = np.digitize(pe_a, pe_bins, right=False) - 1
    pe_idx = np.clip(pe_idx, 0, len(pe_bins) - 1)
    rho_idx = np.digitize(rho_a, rho_bins, right=False) - 1
    rho_idx = np.clip(rho_idx, 0, len(rho_bins) - 1)
    if len(pe_a) == 1:
        return (int(pe_idx[0]), int(rho_idx[0]))
    return (pe_idx, rho_idx)


def cell_based_split(
    pe_values: np.ndarray,
    rho_values: np.ndarray,
    n_val_cells: int = 3,
    n_test_cells: int = 3,
    pe_bins: tuple[float, ...] = (0, 1, 10, 100, 1000, np.inf),
    rho_bins: tuple[float, ...] = (-np.inf, 0, 1, 100, np.inf),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train/val/test split by (Pe, ρ) cell — entire cells are held out.

    Each sample is assigned to a cell.  Cells are then randomly split
    into train / val / test groups, so every sample from a held-out
    cell is completely unseen by the training set.

    Parameters
    ----------
    pe_values, rho_values : ndarray
        Parameter values for each sample.
    n_val_cells, n_test_cells : int
        Number of cells assigned to validation and test.
    pe_bins, rho_bins : tuple
        Bin edges for cell definition.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    train_idx, val_idx, test_idx : ndarray
        Indices into the original dataset.
    cell_map : dict
        Maps cell coordinate → (split, indices).
    """
    rng = _rng(seed)
    pe_idx, rho_idx = _pe_rho_cell(pe_values, rho_values, pe_bins, rho_bins)

    cells: dict[tuple[int, int], list[int]] = {}
    for i in range(len(pe_values)):
        c = (int(pe_idx[i]), int(rho_idx[i]))
        cells.setdefault(c, []).append(i)

    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    n_cells = len(cell_keys)
    n_val = min(n_val_cells, max(1, n_cells // 5))
    n_test = min(n_test_cells, max(1, n_cells // 5))
    n_train = n_cells - n_val - n_test
    if n_train <= 0:
        n_train = 1
        n_val = n_val_cells if n_val_cells < n_cells else max(0, n_cells - n_train)
        n_test = n_cells - n_train - n_val

    train_cells = set(cell_keys[:n_train])
    val_cells = set(cell_keys[n_train:n_train + n_val])
    test_cells = set(cell_keys[n_train + n_val:])

    train_idx = [i for c in train_cells for i in cells[c]]
    val_idx = [i for c in val_cells for i in cells[c]]
    test_idx = [i for c in test_cells for i in cells[c]]

    cell_map = {}
    for c in train_cells:
        cell_map[c] = ("train", np.array(cells[c]))
    for c in val_cells:
        cell_map[c] = ("val", np.array(cells[c]))
    for c in test_cells:
        cell_map[c] = ("test", np.array(cells[c]))

    return np.array(train_idx), np.array(val_idx), np.array(test_idx), cell_map


def regime_holdout_split(
    pe_values: np.ndarray,
    holdout_regime: str = "high",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out an entire Péclet regime for generalization testing.

    Parameters
    ----------
    pe_values : ndarray
        Péclet numbers for each sample.
    holdout_regime : str
        Regime to hold out ("high", "mid", "low", "diffusion", "inf").
    seed : int or None
        Random seed for train splitting.

    Returns
    -------
    train_idx, test_idx : ndarray
    """
    rng = _rng(seed)
    regimes = np.array([_pe_regime(pe) for pe in pe_values])
    test_mask = regimes == holdout_regime
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    rng.shuffle(train_idx)
    return train_idx, test_idx


def frame_split(
    pe_values: np.ndarray,
    rho_values: np.ndarray,
    d_prime_fraction: float = 0.7,
    val_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Geometric frame split in log(Pe) x log(rho) space.

    Defines a bounding box D in log-space covering all samples.
    D' is a centered subrectangle of width ``d_prime_fraction`` of D
    along each axis.  Samples inside D' are split  (1 - val_fraction) /
    val_fraction  into train / val.  Samples outside D' (the frame
    T = D \\ D') become the test set.

    This ensures the test set covers 4 corner regimes where the model
    has never seen extreme Pe or extreme rho during training, providing
    a rigorous extrapolation test.

    Parameters
    ----------
    pe_values, rho_values : ndarray
        Parameter values for each sample (must be > 0).
    d_prime_fraction : float
        Width of D' as a fraction of D along each axis (0, 1].
        0.7 means D' covers 70% of each axis, frame is 15% margin.
    val_fraction : float
        Fraction of D' samples used for validation.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    train_idx, val_idx, test_idx : ndarray
        Indices into the original dataset.
    frame_metadata : dict
        Contains D' bounds, margin fractions, and per-sample classification.
    """
    rng = _rng(seed)

    log_pe = np.log10(np.clip(pe_values, 1e-15, None))
    log_rho = np.log10(np.clip(rho_values, 1e-15, None))

    log_pe_min, log_pe_max = float(log_pe.min()), float(log_pe.max())
    log_rho_min, log_rho_max = float(log_rho.min()), float(log_rho.max())

    margin = (1.0 - d_prime_fraction) / 2.0

    d_prime_pe_lo = log_pe_min + margin * (log_pe_max - log_pe_min)
    d_prime_pe_hi = log_pe_max - margin * (log_pe_max - log_pe_min)
    d_prime_rho_lo = log_rho_min + margin * (log_rho_max - log_rho_min)
    d_prime_rho_hi = log_rho_max - margin * (log_rho_max - log_rho_min)

    inside = (
        (log_pe >= d_prime_pe_lo) & (log_pe <= d_prime_pe_hi) &
        (log_rho >= d_prime_rho_lo) & (log_rho <= d_prime_rho_hi)
    )
    inside_idx = np.where(inside)[0]
    frame_idx = np.where(~inside)[0]

    rng.shuffle(inside_idx)
    n_val = max(1, int(round(val_fraction * len(inside_idx))))
    n_val = min(n_val, len(inside_idx) - 1)
    val_idx = inside_idx[:n_val]
    train_idx = inside_idx[n_val:]

    frame_metadata = {
        "d_prime_fraction": d_prime_fraction,
        "val_fraction": val_fraction,
        "log_pe_range": [log_pe_min, log_pe_max],
        "log_rho_range": [log_rho_min, log_rho_max],
        "d_prime_pe_bounds": [d_prime_pe_lo, d_prime_pe_hi],
        "d_prime_rho_bounds": [d_prime_rho_lo, d_prime_rho_hi],
        "n_inside": int(len(inside_idx)),
        "n_frame": int(len(frame_idx)),
    }

    return np.array(train_idx), np.array(val_idx), np.array(frame_idx), frame_metadata


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for a full dataset generation run.

    Attributes
    ----------
    n_samples : int
        Number of parameter samples.
    h : float
        Element length.
    eps_range : tuple[float, float]
        Diffusion range.
    beta_range : tuple[float, float]
        Advection range.
    sigma_range : tuple[float, float]
        Reaction range.
    strategy : str
        Sampling strategy: ``"lhs"``, ``"stratified"``, or ``"grid"``.
    n_stratified_decade_weights : list[float] or None
        Per-decade weights for stratified sampling (see
        ``sample_parameters_stratified``).
    variable_eps_fraction : float
        Fraction of samples with variable ε profiles (0 = all constant).
        The rest use constant ε with eps_ratios = 1.
    variable_eps_profile : str
        Profile family name (``"sinusoidal"``, ``"layered"``,
        ``"smooth_random"``, ``"constant"``).
    variable_eps_n_quad : int
        Number of quadrature points for ε sampling.  All samples
        (including constant ε) get this many ε_ratio values.
    n_fd_points : int
        FD grid resolution for reference solves.
    val_split : float
        Fraction for validation.
    test_split : float
        Fraction for testing.
    split_strategy : str
        ``"stratified"`` (stratified by Pe regime),
        ``"cell"`` (hold out entire (Pe, ρ) cells),
        or ``"random"``.
    n_val_cells : int
        Number of cells to hold out for validation (only when
        ``split_strategy="cell"``).
    n_test_cells : int
        Number of cells to hold out for testing (only when
        ``split_strategy="cell"``).
    pe_bins : tuple[float, ...] or None
        Pe bin edges for cell-based split.  If None, uses defaults.
    rho_bins : tuple[float, ...] or None
        ρ bin edges for cell-based split.  If None, uses defaults.
    holdout_regime : str or None
        If set, holds out this Pe regime for testing (overrides splits).
    standardize : bool
        If True, fit a DataScaler and store it.
    seed : int
        Master random seed.
    name : str or None
        Dataset name for persistence.  If None, a name is generated.
    """
    n_samples: int = 5000
    h: float = 1 / 16
    eps_range: tuple[float, float] = (1e-6, 1.0)
    beta_range: tuple[float, float] = (1.0, 1.0)
    sigma_range: tuple[float, float] = (0.0, 0.0)
    strategy: Literal["lhs", "stratified", "grid", "log_pe_rho"] = "lhs"
    n_stratified_decade_weights: list[float] | None = None
    pe_range: tuple[float, float] = (0.3, 312.0)
    rho_range: tuple[float, float] = (0.01, 380.0)
    variable_eps_fraction: float = 0.0
    variable_eps_profile: str = "sinusoidal"
    variable_eps_n_quad: int = 5
    n_fd_points: int = 400
    val_split: float = 0.15
    test_split: float = 0.15
    split_strategy: Literal["stratified", "cell", "random", "frame"] = "cell"
    n_val_cells: int = 3
    n_test_cells: int = 3
    pe_bins: tuple[float, ...] | None = None
    rho_bins: tuple[float, ...] | None = None
    frame_d_prime_fraction: float = 0.7
    frame_val_fraction: float = 0.2
    holdout_regime: str | None = None
    standardize: bool = True
    seed: int = 42
    name: str | None = None

    def _generate_name(self) -> str:
        tag = f"varfrac{self.variable_eps_fraction:.1f}"
        strategy_tag = self.strategy
        return f"rfb_dataset_{tag}_{strategy_tag}_n{self.n_samples}_s{self.seed}"


def _extract_arrays(samples: list[dict]) -> dict[str, np.ndarray]:
    """Extract flat arrays from a list of sample dicts."""
    arrays = {}
    for key in ("pe", "rho", "idx"):
        arrays[key] = np.array([s[key] for s in samples], dtype=DTYPE)
    arrays["b"] = np.array([s["b"] for s in samples], dtype=DTYPE)
    arrays["db"] = np.array([s["db"] for s in samples], dtype=DTYPE)
    arrays["xi"] = samples[0]["xi"].astype(DTYPE)
    if "eps_ratios" in samples[0]:
        arrays["eps_ratios"] = np.array(
            [s["eps_ratios"] for s in samples], dtype=DTYPE
        )
    return arrays


def _save_metadata(metadata: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _load_metadata(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_dataset(
    config: DatasetConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    **overrides,
) -> dict[str, Any]:
    """Generate a complete train/val/test dataset for RFB bubble training.

    Parameters
    ----------
    config : DatasetConfig or None
        Full configuration.  If None, defaults are used.
    progress_callback : callable or None
        Called as ``callback(current, total)`` during FD solves.
    **overrides
        Override individual config fields (e.g. ``n_samples=10000``).

    Returns
    -------
    dict with keys:
        "train" : dict of arrays (training split)
        "val" : dict of arrays (validation split)
        "test" : dict of arrays (test split)
        "metadata" : dict (full config + split indices)
        "scaler" : DataScaler or None
    """
    if config is None:
        config = DatasetConfig()
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            raise ValueError(f"Unknown config field: {k}")

    rng = _rng(config.seed)

    # ---- 1. Sample parameters ----
    space = ParameterSpace(
        eps_range=config.eps_range,
        beta_range=config.beta_range,
        sigma_range=config.sigma_range,
        h=config.h,
    )
    if config.strategy == "lhs":
        all_params = sample_parameters_lhs(space, config.n_samples, seed=config.seed)
    elif config.strategy == "stratified":
        all_params = sample_parameters_stratified(
            space, config.n_samples, seed=config.seed,
            eps_decade_weights=(
                np.array(config.n_stratified_decade_weights)
                if config.n_stratified_decade_weights is not None
                else None
            ),
        )
    elif config.strategy == "grid":
        all_params = sample_parameters_grid(space)
    elif config.strategy == "log_pe_rho":
        all_params = sample_parameters_log_pe_rho(
            pe_range=config.pe_range,
            rho_range=config.rho_range,
            n_samples=config.n_samples,
            h=config.h,
            beta=config.beta_range[0],
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    # ---- 2. Generate FD solves for each mode ----
    modes = ("constant", "xi")
    mode_names = ("constant", "xi")
    n_eps = config.variable_eps_n_quad if config.variable_eps_fraction > 0.0 else 0

    n_var = int(round(config.variable_eps_fraction * config.n_samples))
    n_const = config.n_samples - n_var

    rng_split = _rng(config.seed + 100)
    perm = rng_split.permutation(config.n_samples)
    const_idx = perm[:n_const] if n_const > 0 else np.array([], dtype=int)
    var_idx = perm[n_const:] if n_var > 0 else np.array([], dtype=int)

    raw_by_mode: dict[str, list[dict]] = {mname: [] for mname in mode_names}

    # Constant ε samples (always, possibly zero)
    if n_const > 0:
        const_params = all_params[const_idx]
        xi_eps = _gauss_legendre_01(n_eps) if n_eps > 0 else np.array([])
        for mode, mname in zip(modes, mode_names):
            cb = (lambda c, t, m=mname: progress_callback(c, t, prefix=f"{m}_const")) \
                if progress_callback is not None else None
            samples = _build_parameter_sweep(
                const_params, config.h, mode,
                n_fd_points=config.n_fd_points,
                progress_callback=cb,
            )
            if n_eps > 0:
                for s in samples:
                    s["eps_ratios"] = np.ones(n_eps, dtype=float)
            raw_by_mode[mname].extend(samples)

    # Variable ε samples
    if n_var > 0:
        var_params = all_params[var_idx]
        profile_fn = EPS_PROFILE_FAMILIES.get(config.variable_eps_profile)
        if profile_fn is None:
            raise ValueError(
                f"Unknown variable_eps_profile: {config.variable_eps_profile}. "
                f"Options: {list(EPS_PROFILE_FAMILIES)}"
            )
        for mode, mname in zip(modes, mode_names):
            cb = (lambda c, t, m=mname: progress_callback(c, t, prefix=f"{m}_var")) \
                if progress_callback is not None else None
            samples = _build_variable_eps_sweep(
                var_params, config.h,
                eps_profile_fn=profile_fn,
                n_eps=n_eps,
                residual_mode=mode,
                n_fd_points=config.n_fd_points,
                seed=config.seed + 1,
                progress_callback=cb,
            )
            raw_by_mode[mname].extend(samples)

    # ---- 3. Extract arrays and build feature matrix for splitting ----
    arrays_by_mode = {
        mname: _extract_arrays(samples)
        for mname, samples in raw_by_mode.items()
    }
    pe_all = arrays_by_mode["constant"]["pe"]
    n_total = len(pe_all)

    # ---- 4. Split ----
    cell_map = None
    frame_meta = None
    if config.holdout_regime is not None:
        train_idx, test_idx = regime_holdout_split(
            pe_all, holdout_regime=config.holdout_regime, seed=config.seed + 2
        )
        val_idx = np.array([], dtype=int)
    elif config.split_strategy == "frame":
        rho_all = arrays_by_mode["constant"]["rho"]
        train_idx, val_idx, test_idx, frame_meta = frame_split(
            pe_all, rho_all,
            d_prime_fraction=config.frame_d_prime_fraction,
            val_fraction=config.frame_val_fraction,
            seed=config.seed + 2,
        )
    elif config.split_strategy == "cell":
        rho_all = arrays_by_mode["constant"]["rho"]
        train_idx, val_idx, test_idx, cell_map = cell_based_split(
            pe_all, rho_all,
            n_val_cells=config.n_val_cells,
            n_test_cells=config.n_test_cells,
            pe_bins=config.pe_bins or (0, 1, 10, 100, 1000, np.inf),
            rho_bins=config.rho_bins or (-np.inf, 0, 1, 100, np.inf),
            seed=config.seed + 2,
        )
    elif config.split_strategy == "stratified":
        train_idx, val_idx, test_idx = stratified_split(
            pe_all,
            val_split=config.val_split,
            test_split=config.test_split,
            seed=config.seed + 2,
        )
    else:
        idx = np.arange(n_total)
        rng.shuffle(idx)
        n_test = max(1, int(round(config.test_split * n_total)))
        n_val = max(1, int(round(config.val_split * n_total)))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

    # ---- 5. Preprocessing (standardization) ----
    scaler = None
    if config.standardize:
        input_features = ["pe_log", "rho_log"]
        X_raw = np.column_stack([
            np.log(np.where(pe_all < 1e-15, 1e-15, pe_all)),
            np.log(np.where(np.abs(arrays_by_mode["constant"]["rho"]) < 1e-15, 1e-15, np.abs(arrays_by_mode["constant"]["rho"]))),
        ])
        scaler = DataScaler().fit(X_raw, feature_names=input_features)

    # ---- 6. Build split dataset dicts ----
    def _split_data(arrays: dict, split_idx: np.ndarray) -> dict:
        out = {}
        for k, v in arrays.items():
            if k == "xi":
                out[k] = v
            else:
                out[k] = v[split_idx]
        if scaler is not None and "pe" in out:
            pe_log = np.log(np.where(out["pe"] < 1e-15, 1e-15, out["pe"]))
            rho_log = np.log(np.where(np.abs(out["rho"]) < 1e-15, 1e-15, np.abs(out["rho"])))
            out["input_scaled"] = scaler.transform(
                np.column_stack([pe_log, rho_log])
            )
        return out

    split_names = {"train": train_idx, "val": val_idx, "test": test_idx}
    dataset = {}
    for sname, sidx in split_names.items():
        if len(sidx) == 0:
            continue
        dataset[sname] = {
            mname: _split_data(arrays_by_mode[mname], sidx)
            for mname in mode_names
        }

    # ---- 7. Metadata ----
    metadata = asdict(config)
    metadata["mode_names"] = list(mode_names)
    metadata["n_eps"] = n_eps
    metadata["n_const"] = n_const
    metadata["n_var"] = n_var
    metadata["n_total"] = n_total
    metadata["n_train"] = len(train_idx)
    metadata["n_val"] = len(val_idx)
    metadata["n_test"] = len(test_idx)
    metadata["split_indices"] = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    if cell_map is not None:
        metadata["cell_map"] = {
            f"({pe},{rh})": (split, idxs.tolist())
            for (pe, rh), (split, idxs) in cell_map.items()
        }
    if frame_meta is not None:
        metadata["frame_meta"] = frame_meta
    if config.holdout_regime is not None:
        metadata["split_strategy"] = f"holdout_{config.holdout_regime}"
    dataset["metadata"] = metadata
    dataset["scaler"] = scaler
    dataset["mode_names"] = mode_names

    return dataset


def save_dataset(
    dataset: dict[str, Any],
    name: str | None = None,
    subdir: str | None = None,
) -> str:
    """Persist a generated dataset to disk.

    Saves NPZ files per mode per split and a JSON metadata file.

    Parameters
    ----------
    dataset : dict
        Output of ``generate_dataset``.
    name : str or None
        Dataset name.  If None, uses metadata name.
    subdir : str or None
        Subdirectory under ``DATASET_SUBDIR``.  If None, saves directly
        in the dataset directory.

    Returns
    -------
    str : path to the metadata JSON file.
    """
    metadata = dict(dataset["metadata"])
    if name is None:
        name = metadata.get("name") or "rfb_dataset"
    base = Path(DATASET_SUBDIR)
    if subdir is not None:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)

    mode_names = dataset.get("mode_names", metadata.get("mode_names", []))
    split_names = [k for k in ("train", "val", "test") if k in dataset]

    file_hashes = {}
    for sname in split_names:
        for mname in mode_names:
            data = dataset[sname].get(mname)
            if data is None:
                continue
            path = base / f"{name}_{sname}_{mname}.npz"
            arrays = {
                k: v for k, v in data.items() if isinstance(v, np.ndarray)
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            np.savez_compressed(str(tmp_path), **arrays)
            tmp_npz = Path(str(tmp_path) + ".npz")
            tmp_npz.replace(path)
            file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    metadata["schema_version"] = DATASET_SCHEMA_VERSION
    metadata["file_hashes"] = file_hashes
    metadata_path = base / f"{name}_metadata.json"
    _save_metadata(metadata, metadata_path)

    if dataset.get("scaler") is not None:
        scaler_path = base / f"{name}_scaler.json"
        _save_metadata(dataset["scaler"].to_dict(), scaler_path)

    return str(metadata_path)


def load_dataset(
    name: str,
    subdir: str | None = None,
) -> dict[str, Any]:
    """Load a previously saved dataset.

    Parameters
    ----------
    name : str
        Dataset name (matching the prefix used in ``save_dataset``).
    subdir : str or None
        Subdirectory under ``DATASET_SUBDIR``.

    Returns
    -------
    dict : dataset with the same structure as ``generate_dataset`` output.
    """
    base = Path(DATASET_SUBDIR)
    if subdir is not None:
        base = base / subdir

    metadata_path = base / f"{name}_metadata.json"
    metadata = _load_metadata(metadata_path)
    if metadata.get("schema_version", 0) > DATASET_SCHEMA_VERSION:
        raise ValueError("dataset schema is newer than this library")

    scaler = None
    scaler_path = base / f"{name}_scaler.json"
    if scaler_path.exists():
        scaler = DataScaler.from_dict(_load_metadata(scaler_path))

    mode_names = metadata.get("mode_names", ["constant", "xi"])
    split_names = ["train", "val", "test"]

    dataset = {"metadata": metadata, "scaler": scaler, "mode_names": mode_names}

    for sname in split_names:
        dataset[sname] = {}
        for mname in mode_names:
            path = base / f"{name}_{sname}_{mname}.npz"
            if not path.exists():
                continue
            data = dict(np.load(str(path)))
            expected_hash = metadata.get("file_hashes", {}).get(path.name)
            if expected_hash is not None:
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(f"dataset file checksum mismatch: {path.name}")
            lengths = {len(v) for k, v in data.items() if k not in ("xi", "input_scaled") and np.ndim(v) > 0}
            if lengths and len(lengths) != 1:
                raise ValueError(f"inconsistent array lengths in {path.name}")
            dataset[sname][mname] = data

    for sname, split in dataset.items():
        if sname not in split_names:
            continue
        if not split:
            continue
        mode_lengths = {m: len(v["pe"]) for m, v in split.items()}
        if len(set(mode_lengths.values())) != 1:
            raise ValueError(f"mode lengths differ in split {sname}: {mode_lengths}")

    return dataset


# ---------------------------------------------------------------------------
# Dataset analysis utilities
# ---------------------------------------------------------------------------


def dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    """Print and return a summary of dataset statistics."""
    metadata = dataset["metadata"]
    mode_names = dataset.get("mode_names", metadata.get("mode_names", []))

    summary = {
        "name": metadata.get("name", "unknown"),
        "n_total": metadata["n_total"],
        "n_train": metadata["n_train"],
        "n_val": metadata["n_val"],
        "n_test": metadata["n_test"],
        "strategy": metadata.get("strategy"),
        "variable_eps_fraction": metadata.get("variable_eps_fraction", 0.0),
        "n_eps": metadata.get("n_eps", 0),
        "eps_range": metadata.get("eps_range"),
        "beta_range": metadata.get("beta_range"),
        "sigma_range": metadata.get("sigma_range"),
        "mode_names": mode_names,
        "split_strategy": metadata.get("split_strategy"),
        "pe_stats": {},
    }

    for sname in ("train", "val", "test"):
        if sname not in dataset:
            continue
        mode = dataset["mode_names"][0]
        pe_vals = dataset[sname][mode]["pe"]
        summary["pe_stats"][sname] = {
            "min": float(pe_vals.min()),
            "max": float(pe_vals.max()),
            "mean": float(pe_vals.mean()),
            "median": float(np.median(pe_vals)),
        }

    print("=" * 60)
    print(f"Dataset: {summary['name']}")
    print(f"  Samples: {summary['n_total']} total "
          f"({summary['n_train']} train / {summary['n_val']} val / {summary['n_test']} test)")
    vfrac = summary['variable_eps_fraction']
    print(f"  Strategy: {summary['strategy']}, var_eps_frac: {vfrac:.2f} (n_eps={summary['n_eps']})")
    print(f"  ε ∈ [{summary['eps_range'][0]:.1e}, {summary['eps_range'][1]:.1e}]")
    print(f"  β ∈ [{summary['beta_range'][0]}, {summary['beta_range'][1]}]")
    print(f"  σ ∈ [{summary['sigma_range'][0]}, {summary['sigma_range'][1]}]")
    print(f"  Modes: {', '.join(mode_names)}")
    print(f"  Split: {summary['split_strategy']}")
    if metadata.get("cell_map"):
        n_train_cells = sum(1 for v in metadata["cell_map"].values() if v[0] == "train")
        n_val_cells = sum(1 for v in metadata["cell_map"].values() if v[0] == "val")
        n_test_cells = sum(1 for v in metadata["cell_map"].values() if v[0] == "test")
        print(f"  Cells: {n_train_cells} train / {n_val_cells} val / {n_test_cells} test")
    for sname in ("train", "val", "test"):
        if sname in summary["pe_stats"]:
            ps = summary["pe_stats"][sname]
            print(f"    {sname}: Pe ∈ [{ps['min']:.2e}, {ps['max']:.2e}] "
                  f"(median {ps['median']:.2e})")
    print("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Bubble similarity / redundancy analysis
# ---------------------------------------------------------------------------


_DEF_LAMBDA_DERIV = 0.2


def _quadrature_weights(xi: np.ndarray) -> np.ndarray:
    """Trapezoidal rule weights on the FD grid (endpoints get half weight)."""
    weights = np.empty(xi.size, dtype=float)
    weights[0] = 0.5 * (xi[1] - xi[0])
    weights[-1] = 0.5 * (xi[-1] - xi[-2])
    weights[1:-1] = 0.5 * (xi[2:] - xi[:-2])
    return weights


def _bubble_h1_features(
    b_array: np.ndarray,
    xi: np.ndarray,
    lambda_deriv: float = _DEF_LAMBDA_DERIV,
) -> np.ndarray:
    """Augmented feature matrix whose Euclidean product is the H1 inner product.

    F(b) = (√w · b, λ · √w · db/dξ), so that for two bubbles
        F(b) · F(b') = ∫₀¹ b b' dξ + λ² ∫₀¹ (db/dξ)(db'/dξ) dξ
    where ∫ is the trapezoidal quadrature.  With ``lambda_deriv = 0`` this
    reduces exactly to the plain L2 feature matrix √w · b.
    """
    w = _quadrature_weights(xi)
    sw = np.sqrt(w)
    if lambda_deriv is None or lambda_deriv <= 0.0:
        return b_array * sw
    db = np.gradient(b_array, xi, axis=1)
    return np.concatenate([b_array * sw, lambda_deriv * db * sw], axis=1)


def bubble_gram_matrix(
    b_array: np.ndarray,
    xi: np.ndarray,
    lambda_deriv: float = _DEF_LAMBDA_DERIV,
) -> np.ndarray:
    """H1 Gram matrix of bubbles: G[i, j] = ⟨b_i, b_j⟩_H1
    = ∫₀¹ b_i b_j dξ + λ² ∫₀¹ b_i' b_j' dξ.

    The integrals are approximated with the trapezoidal rule on the FD grid
    and a centered-difference derivative.  Since every bubble is normalized
    (b(0.5) = 1), G[i, i] is the squared H1 norm of bubble i.  With
    ``lambda_deriv = 0`` this is the plain L2 Gram matrix.

    Parameters
    ----------
    b_array : ndarray, shape (N, n_fd)
        Bubble values on the FD grid, one row per sample.
    xi : ndarray, shape (n_fd,)
        FD grid points in [0, 1].
    lambda_deriv : float
        Length-scale weight of the derivative term (default 0.2, the H1
        inner product).  The derivative-aware version distinguishes shapes
        that share mass but differ in their *changes* (boundary-layer
        sharpness, ripples) which plain L2 would call near-identical.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric positive semidefinite Gram matrix.
    """
    b = np.asarray(b_array, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if b.ndim != 2 or xi.ndim != 1 or b.shape[1] != xi.size:
        raise ValueError("b_array must have shape (N, len(xi))")
    if xi.size < 2 or not np.all(np.isfinite(b)) or not np.all(np.isfinite(xi)):
        raise ValueError("bubble data must be finite and contain at least two grid points")
    if np.any(np.diff(xi) <= 0):
        raise ValueError("xi must be strictly increasing")
    V = _bubble_h1_features(b, xi, lambda_deriv)
    return V @ V.T


def bubble_cosine_similarity(
    b_array: np.ndarray,
    xi: np.ndarray,
    lambda_deriv: float = _DEF_LAMBDA_DERIV,
) -> np.ndarray:
    """Normalized (cosine) similarity matrix C[i, j] ∈ [-1, 1].

    C[i, j] = ⟨b_i, b_j⟩ / (‖b_i‖ ‖b_j‖) with the H1 inner product
    (derivative-aware; ``lambda_deriv=0`` recovers the L2 version).
    C[i, j] ≈ 1 means the two bubbles are nearly identical in *shape and
    change* (equal up to a positive scalar multiple), regardless of their
    norms.  Unlike the plain L2 cosine, this metric flags shapes that differ
    in their derivatives (boundary-layer sharpness, oscillation) as distinct.

    Parameters
    ----------
    b_array : ndarray, shape (N, n_fd)
        Bubble values on the FD grid, one row per sample.
    xi : ndarray, shape (n_fd,)
        FD grid points in [0, 1].
    lambda_deriv : float
        Derivative-term weight (default 0.2).

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric cosine similarity matrix.
    """
    G = bubble_gram_matrix(b_array, xi, lambda_deriv)
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = G / np.outer(d, d)
    return np.clip(C, -1.0, 1.0)


def max_cross_similarity(
    other: np.ndarray,
    reference: np.ndarray,
    xi: np.ndarray,
    lambda_deriv: float = _DEF_LAMBDA_DERIV,
    block_size: int = 1024,
) -> np.ndarray:
    """Return each ``other`` row's maximum cosine similarity to ``reference``.

    Both arrays are compared with the derivative-aware H1 inner product
    (``lambda_deriv=0`` recovers the L2 version).  The calculation is
    blockwise and therefore avoids materializing an
    ``len(other) x len(reference)`` matrix.  This is the production primitive
    for leakage audits and large no-twin splits.
    """
    other = np.asarray(other, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    xi = np.asarray(xi, dtype=float)
    if xi.ndim != 1 or xi.size < 2 or np.any(np.diff(xi) <= 0):
        raise ValueError("xi must be a strictly increasing one-dimensional grid")
    if other.ndim != 2 or reference.ndim != 2 or other.shape[1] != reference.shape[1]:
        raise ValueError("bubble arrays must be two-dimensional with matching grids")
    if other.shape[1] != xi.size or not np.all(np.isfinite(other)) or not np.all(np.isfinite(reference)):
        raise ValueError("bubble arrays must be finite and match xi")
    V_other = _bubble_h1_features(other, xi, lambda_deriv)
    V_ref = _bubble_h1_features(reference, xi, lambda_deriv)
    other_n = np.sqrt(np.maximum(np.sum(V_other * V_other, axis=1), 1e-30))
    ref_n = np.sqrt(np.maximum(np.sum(V_ref * V_ref, axis=1), 1e-30))
    out = np.empty(other.shape[0], dtype=float)
    V_refT = V_ref.T
    for start in range(0, other.shape[0], block_size):
        stop = min(start + block_size, other.shape[0])
        sims = (V_other[start:stop] @ V_refT)
        sims /= other_n[start:stop, None] * ref_n[None, :]
        out[start:stop] = np.clip(sims, -1.0, 1.0).max(axis=1)
    return out


def _off_diagonal_stats(C: np.ndarray) -> dict[str, float]:
    """Summary statistics of the strictly off-diagonal entries of C."""
    n = C.shape[0]
    if n < 2:
        return {
            "mean": np.nan, "median": np.nan, "std": np.nan,
            "min": np.nan, "max": np.nan,
            "frac_gt_0.90": np.nan, "frac_gt_0.95": np.nan,
            "frac_gt_0.99": np.nan, "n_pairs": 0,
        }
    vals = C[np.triu_indices(n, k=1)]
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "frac_gt_0.90": float(np.mean(vals > 0.90)),
        "frac_gt_0.95": float(np.mean(vals > 0.95)),
        "frac_gt_0.99": float(np.mean(vals > 0.99)),
        "n_pairs": int(vals.size),
    }


def _off_diagonal_stats_blockwise(
    b: np.ndarray, xi: np.ndarray, lambda_deriv: float = _DEF_LAMBDA_DERIV,
    block_size: int = 1024,
) -> dict[str, float]:
    """Compute off-diagonal cosine statistics without an NxN matrix."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    b = np.asarray(b, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if b.ndim != 2 or xi.ndim != 1 or b.shape[1] != xi.size or xi.size < 2:
        raise ValueError("bubble data must have shape (N, len(xi))")
    n = b.shape[0]
    if n < 2:
        return _off_diagonal_stats(np.empty((1, 1)))
    V = _bubble_h1_features(b, xi, lambda_deriv)
    norms = np.sqrt(np.maximum(np.sum(V * V, axis=1), 1e-30))
    values = []
    weighted = V
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = (V[start:stop] @ weighted.T) / norms[start:stop, None] / norms[None, :]
        rows = np.arange(start, stop)
        block[rows - start, rows] = np.nan
        values.append(block[~np.isnan(block)])
    vals = np.concatenate(values)
    return {
        "mean": float(vals.mean()), "median": float(np.median(vals)),
        "std": float(vals.std()), "min": float(vals.min()),
        "max": float(vals.max()), "frac_gt_0.90": float(np.mean(vals > 0.90)),
        "frac_gt_0.95": float(np.mean(vals > 0.95)),
        "frac_gt_0.99": float(np.mean(vals > 0.99)),
        "n_pairs": int(vals.size),
    }


def _effective_rank(C: np.ndarray) -> dict[str, float]:
    """Eigenvalue decay and effective rank of a similarity matrix.

    Eigenvalues λ₁ ≥ λ₂ ≥ ... are normalized to a probability vector
    p_k = λ_k / Σλ. The effective rank is the exponential of the spectral
    entropy, exp(−Σ p_k log p_k): a rank-1 matrix (all bubbles identical)
    gives 1, while a full-rank matrix with uniform spectrum gives N.
    """
    evals = np.linalg.eigvalsh((C + C.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    n = len(evals)
    if total < 1e-30:
        return {"effective_rank": 1.0, "rank": 0, "n_for_95pct_energy": 1}
    p = evals / total
    entropy = -np.sum(p * np.log(p + 1e-30))
    cum_asc = np.cumsum(evals[::-1]) / total  # energy of top-k eigenvalues, k ascending
    n_95 = int(np.searchsorted(cum_asc, 0.95)) + 1
    n_95 = max(1, min(n_95, n))
    return {
        "effective_rank": float(np.exp(entropy)),
        "rank": int(np.sum(evals > total * 1e-12)),
        "n_for_95pct_energy": n_95,
        "top1_energy": float(evals[-1] / total),
        "top5_energy": float(evals[-5:].sum() / total),
        "top10_energy": float(evals[-10:].sum() / total),
    }


def bubble_similarity_analysis(
    dataset: dict[str, Any],
    mode: str = "constant",
    splits: tuple[str, ...] = ("train", "val", "test"),
    lambda_deriv: float = _DEF_LAMBDA_DERIV,
    n_eig_samples: int = 500,
    seed: int = 0,
    return_matrices: bool = False,
    block_size: int = 1024,
    verbose: bool = True,
) -> dict[str, Any]:
    """Analyze bubble redundancy and train/test leakage via similarities.

    Motivation (professor's concern): if all bubbles in the dataset are
    nearly identical in shape, the KAN can memorize a single function and
    still report low test error because every test bubble is already
    (almost) present in the training set. The effective number of distinct
    functions may be much smaller than the number of samples, inflating
    generalization numbers.

    Similarity is measured with the *derivative-aware H1 cosine*: two
    bubbles are ``twin``-like only when they agree in value *and* in
    derivative (boundary-layer sharpness, oscillation).  The plain L2
    cosine reports any two bubbles with overlapping mass as near-identical
    even when their shapes differ measurably; with ``lambda_deriv=0`` the
    analysis reverts to that L2 metric.

    For each requested split the function computes:
      * the H1 cosine-similarity matrix
        C[i, j] = ⟨b_i, b_j⟩_H1 / (‖b_i‖_H1‖b_j‖_H1),
      * off-diagonal statistics (how often distinct bubbles are near-duplicates),
      * the eigenvalue decay / effective rank on a random subsample
        (how many genuinely distinct bubble shapes the split spans).

    Cross-split (train vs val and train vs test) it reports, for every
    non-train bubble, the *maximum* similarity to any training bubble.  If
    most test bubbles have a near-duplicate twin in the training set
    (similarity > 0.99), the split is effectively leaking: the test set is
    not a genuine out-of-distribution benchmark.

    Parameters
    ----------
    dataset : dict
        Output of ``load_dataset`` or ``generate_dataset``.
    mode : str
        Bubble mode to analyze (``"constant"`` or ``"xi"``).
    splits : tuple[str, ...]
        Splits to analyze. The first one is treated as the reference
        (training) split for cross-split leakage.
    lambda_deriv : float
        Derivative term weight of the H1 inner product (default 0.2).
    n_eig_samples : int
        Number of (subsampled) bubbles used for the eigenvalue analysis.
    seed : int
        Seed for the eigenvalue subsample.
    return_matrices : bool
        If True, include the full cosine-similarity matrices in the result
        (can be large: N×N float64 per split).
    verbose : bool
        Print a human-readable summary.

    Returns
    -------
    dict with keys ``"mode"``, ``"within"`` (per-split stats and, if
    requested, matrices) and ``"cross"`` (train-vs-split max similarities).
    """
    if mode not in ("constant", "xi"):
        raise ValueError(f"Unknown mode: {mode} (expected 'constant' or 'xi')")

    rng = np.random.default_rng(seed)
    ref_split = splits[0]
    result: dict[str, Any] = {"mode": mode, "within": {}, "cross": {}}
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sname in splits:
        if sname not in dataset:
            if verbose:
                print(f"  [skip] split '{sname}' not in dataset")
            continue
        b = np.asarray(dataset[sname][mode]["b"], dtype=float)
        xi = np.asarray(dataset[sname][mode]["xi"], dtype=float)
        split_arrays[sname] = (b, xi)

        within = {"n_samples": b.shape[0]}
        within["off_diagonal"] = _off_diagonal_stats_blockwise(
            b, xi, lambda_deriv, block_size)

        n = b.shape[0]
        n_sub = min(n, n_eig_samples)
        idx = rng.choice(n, size=n_sub, replace=False)
        C_sub = bubble_cosine_similarity(b[idx], xi, lambda_deriv)
        within["effective_rank"] = _effective_rank(C_sub)

        if return_matrices:
            within["C"] = bubble_cosine_similarity(b, xi, lambda_deriv)
        result["within"][sname] = within

    if ref_split in split_arrays:
        B_ref, xi_ref = split_arrays[ref_split]
        for sname, (b, xi) in split_arrays.items():
            if sname == ref_split:
                continue
            # Cross Gram: rows = non-train samples, cols = reference samples.
            # The normal production path computes only nearest similarities.
            # Full cross matrices remain available explicitly for diagnostics.
            max_sim = max_cross_similarity(b, B_ref, xi_ref, lambda_deriv, block_size)
            stats = {
                "n_other": b.shape[0],
                "max_sim_mean": float(max_sim.mean()),
                "max_sim_median": float(np.median(max_sim)),
                "max_sim_min": float(max_sim.min()),
                "max_sim_max": float(max_sim.max()),
                "frac_gt_0.90": float(np.mean(max_sim > 0.90)),
                "frac_gt_0.95": float(np.mean(max_sim > 0.95)),
                "frac_gt_0.99": float(np.mean(max_sim > 0.99)),
            }
            entry: dict[str, Any] = {"max_similarity": max_sim, "stats": stats}
            if return_matrices:
                V_ref = _bubble_h1_features(B_ref, xi_ref, lambda_deriv)
                V_oth = _bubble_h1_features(b, xi, lambda_deriv)
                d_ref = np.sqrt(np.maximum(np.sum(V_ref * V_ref, axis=1), 1e-30))
                d_oth = np.sqrt(np.maximum(np.sum(V_oth * V_oth, axis=1), 1e-30))
                C_cross = (V_oth @ V_ref.T) / np.outer(d_oth, d_ref)
                entry["C"] = np.clip(C_cross, -1.0, 1.0)
            result["cross"][f"{ref_split}_vs_{sname}"] = entry

    if verbose:
        _print_similarity_summary(result)
    return result


def _print_similarity_summary(result: dict[str, Any]) -> None:
    """Pretty-print the output of bubble_similarity_analysis."""
    mode = result["mode"]
    print("\n" + "=" * 72)
    print(f"  BUBBLE SIMILARITY ANALYSIS  (mode: {mode})")
    print("=" * 72)
    print("  Cosine similarity C[i,j] = <b_i,b_j> / (||b_i|| ||b_j||) in H1 (derivative-aware).")
    print("  C near 1 => two bubbles are near-duplicates in shape AND change.")
    print()

    for sname, within in result["within"].items():
        od = within["off_diagonal"]
        er = within["effective_rank"]
        print(f"  Split '{sname}'  (N={within['n_samples']})")
        print(f"    off-diagonal C: mean={od['mean']:.4f}  median={od['median']:.4f}"
              f"  max={od['max']:.4f}")
        print(f"    pairs with C>0.90: {100*od['frac_gt_0.90']:.1f}%  "
              f"C>0.95: {100*od['frac_gt_0.95']:.1f}%  C>0.99: {100*od['frac_gt_0.99']:.1f}%")
        print(f"    effective rank ({'subsample' if within['n_samples'] > 500 else 'full'}) = "
              f"{er['effective_rank']:.1f}  |  "
              f"{er['n_for_95pct_energy']} eigenvalues hold 95% of the energy  |  "
              f"top-1 eigenvalue = {100*er['top1_energy']:.1f}% of energy")
        print()

    if result["cross"]:
        print("  CROSS-SPLIT LEAKAGE (each other-split bubble vs nearest train bubble)")
        for key, entry in result["cross"].items():
            st = entry["stats"]
            print(f"    {key}:  max-sim mean={st['max_sim_mean']:.4f}  "
                  f"median={st['max_sim_median']:.4f}  max={st['max_sim_max']:.4f}")
            print(f"      bubbles with a train twin: C>0.90: {100*st['frac_gt_0.90']:.1f}%  "
                  f"C>0.95: {100*st['frac_gt_0.95']:.1f}%  C>0.99: {100*st['frac_gt_0.99']:.1f}%")
        print()
        print("  Reading: high within-split similarity => the dataset is redundant.")
        print("  High fraction of test bubbles with a train twin (C>0.99) => the")
        print("  test set is NOT a genuine out-of-distribution benchmark.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Batch training from array-format dataset
# ---------------------------------------------------------------------------


def _to_tensor(x, dtype=torch.float32, device=None):
    t = torch.as_tensor(np.asarray(x, dtype=DTYPE), dtype=dtype)
    return t.to(device) if device is not None else t


def train_bubble_on_dataset(
    model: KANBubble1D | MultiKANBubble1D,
    mode_data: dict[str, np.ndarray],
    n_epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    grad_weight: float = 1e-3,
    n_quad: int = 80,
    verbose: bool = True,
    device: torch.device | None = None,
    lr_scheduler: str | None = None,
    energy_weight: float = 0.0,
    val_split: dict[str, np.ndarray] | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    restore_best: bool = True,
    weight_decay: float = 0.0,
    es_warmup: int = 20,
    es_ema_alpha: float = 0.95,
) -> list[float] | dict[str, list[float]]:
    """Train a bubble model from array-format dataset (batch mode).

    Parameters
    ----------
    model : KANBubble1D or MultiKANBubble1D
        The bubble model to train.
    mode_data : dict
        Mode data from a dataset split, e.g. ``dataset['train']['constant']``.
        Must contain keys ``"pe"``, ``"rho"``, ``"b"``, ``"db"``, ``"xi"``.
    n_epochs, batch_size, lr : int, int, float
        Training hyperparameters.
    grad_weight : float
        Weight for the gradient-matching loss term (autograd-based;
        historically unstable, keep 0).
    n_quad : int
        Number of quadrature points (the target arrays are interpolated
        to this grid for each sample).
    verbose : bool
        Print progress.
    device : torch.device or None
        Device for training.
    lr_scheduler : str or None
        Optional per-epoch learning-rate schedule. ``"cosine"`` uses
        CosineAnnealingLR from ``lr`` down to ~0 over ``n_epochs``.
    energy_weight : float
        Weight for the Darcy energy-norm error term
        ``∫ eps(x) |d(b_pred - b_target)/dx|^2 dx`` added to the value MSE.
        Derivatives are input-space central differences of prediction and
        target values (no second-order autograd graph), evaluated on the
        interior quadrature nodes. Requires ``eps_profile`` in mode_data.
    val_split : dict or None
        Validation split (same layout as ``mode_data``). If given, the
        per-epoch validation loss is computed (no-grad) and returned
        alongside the training loss.
    patience : int or None
        Early-stopping patience in **epochs** (e.g. 50). Requires
        ``val_split``. Training stops when the early-stopping signal (an
        EMA-smoothed validation loss, see ``es_ema_alpha``) has not improved
        by at least ``min_delta`` for ``patience`` consecutive epochs; with
        ``restore_best`` the model is reverted to the state at the best
        (post-warm-up) validation epoch. The returned histories are truncated
        to the epochs actually run (best epoch = last index of the
        histories). The cosine scheduler keeps to ``n_epochs`` regardless.
    min_delta : float
        Minimum improvement in the EMA-smoothed validation loss required to
        reset the patience counter.
    restore_best : bool
        If True (default), reload the model weights from the best
        validation epoch after early stopping (or when the loop finishes).
    weight_decay : float
        L2 weight-regularization strength λ. Adds ``λ·Σ_w ‖w‖²`` (over all
        model parameters) to the training loss only — the reported training
        loss and the validation loss remain the pure data loss, so the train/
        val curves stay comparable (the decay is a regularizer for the
        optimizer, not a monitored objective).
    es_warmup : int
        Number of leading epochs during which the early-stopping signal is
        *not* tracked (training is never stopped and the "best" state is not
        anchored). Early-run validation spikes from the random init are
        discarded this way.
    es_ema_alpha : float
        EM-factor (0 < alpha < 1) for smoothing the validation loss before
        the patience decision: ``ema = alpha*ema + (1-alpha)*val`` each epoch.
        Deciding on the EMA (instead of the raw single-epoch value) makes the
        criterion immune to 1%-scale oscillation. The *reported* ``"val"``
        history stays the raw per-epoch loss for plotting.

    Returns
    -------
    list[float] or dict[str, list[float]] : if ``val_split`` is None returns
        the training-loss history ``list[float]``; otherwise returns
        ``{"train": [...], "val": [...]}``. With early stopping the lists are
        truncated to the number of epochs actually run, so the best epoch for
        each mode is ``len(history) - 1``.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    if patience is not None:
        if patience <= 0:
            raise ValueError("patience must be > 0 (epochs)")
        if val_split is None:
            raise ValueError("early stopping (patience) requires val_split")
    N = len(mode_data["pe"])
    xi_base = torch.linspace(0.0, 1.0, n_quad, dtype=torch.float32)
    xi_base[0] = 1e-6
    xi_base[-1] = 1.0 - 1e-6
    if device is not None:
        xi_base = xi_base.to(device)

    pe_all = _to_tensor(mode_data["pe"], device=device)        # (N,)
    rho_all = _to_tensor(mode_data["rho"], device=device)       # (N,)

    b_target_all = np.zeros((N, n_quad), dtype=DTYPE)
    db_target_all = np.zeros((N, n_quad), dtype=DTYPE)
    xi_np = xi_base.detach().cpu().numpy()
    for i in range(N):
        target = {"xi": mode_data["xi"], "b": mode_data["b"][i], "db": mode_data["db"][i]}
        b_target_all[i], db_target_all[i] = interpolate_target(target, xi_np)
    b_target_all = _to_tensor(b_target_all, device=device)     # (N, Q)
    db_target_all = _to_tensor(db_target_all, device=device)    # (N, Q)

    eps_ratios_all = None
    if "eps_ratios" in mode_data:
        eps_ratios_all = _to_tensor(mode_data["eps_ratios"], device=device)  # (N, n_eps)

    eps_grid_all = None
    trap_w_int = None
    if energy_weight > 0.0:
        if "eps_profile" not in mode_data:
            raise ValueError("energy_weight > 0 requires 'eps_profile' in mode_data")
        xi_src = np.asarray(mode_data["xi"], dtype=np.float64)
        eps_np = np.empty((N, n_quad), dtype=DTYPE)
        for i in range(N):
            eps_np[i] = np.interp(xi_np, xi_src, mode_data["eps_profile"][i])
        eps_grid_all = _to_tensor(eps_np, device=device)          # (N, Q)
        h = float(xi_base[1] - xi_base[0])
        w = np.full(n_quad, h, dtype=DTYPE)
        w[0] = w[-1] = 0.5 * h
        trap_w_int = _to_tensor(w[1:-1], device=device)           # (Q-2,)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif lr_scheduler is not None:
        raise ValueError(f"unknown lr_scheduler: {lr_scheduler}")
    losses = []

    # ---- optional validation split: precompute targets on the same grid ----
    val_pe = val_rho = val_eps = val_b = None
    if val_split is not None:
        Nv = len(val_split["pe"])
        val_pe = _to_tensor(val_split["pe"], device=device)          # (Nv,)
        val_rho = _to_tensor(val_split["rho"], device=device)        # (Nv,)
        val_b = np.zeros((Nv, n_quad), dtype=DTYPE)
        for i in range(Nv):
            target = {"xi": val_split["xi"], "b": val_split["b"][i],
                      "db": val_split["db"][i]}
            val_b[i], _ = interpolate_target(target, xi_np)
        val_b = _to_tensor(val_b, device=device)                     # (Nv, Q)
        if "eps_ratios" in val_split:
            val_eps = _to_tensor(val_split["eps_ratios"], device=device)
        val_losses = []
        best_val = None
        best_epoch = 0
        best_state = None
        ema_val = None

    n_batches = max(1, (N + batch_size - 1) // batch_size)
    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_processed = 0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            bs = len(idx)
            optimizer.zero_grad()

            Q = n_quad
            pe_b = pe_all[idx]           # (B,)
            rho_b = rho_all[idx]         # (B,)
            xi_flat = xi_base.unsqueeze(0).expand(bs, -1).reshape(-1).requires_grad_(True)   # (B*Q,)
            pe_flat = pe_b.unsqueeze(1).expand(bs, Q).reshape(-1)                           # (B*Q,)
            rho_flat = rho_b.unsqueeze(1).expand(bs, Q).reshape(-1)                         # (B*Q,)
            eps_flat = (
                eps_ratios_all[idx].unsqueeze(1).expand(bs, Q, -1).reshape(bs * Q, -1)
                if eps_ratios_all is not None else None
            )
            eps_per = eps_ratios_all[idx] if eps_ratios_all is not None else None

            nf = model.norm_at_mid(pe_b, rho_b, eps_ratios=eps_per)  # (B,) or (n_bubbles, B)
            pred_flat = model(xi_flat, pe_flat, rho_flat, eps_ratios=eps_flat,
                              norm_factor=nf)  # (B*Q,)
            pred = pred_flat.reshape(bs, Q)

            if grad_weight > 0.0:
                dpred_flat = torch.autograd.grad(
                    pred_flat, xi_flat, torch.ones_like(pred_flat), create_graph=True
                )[0]
                dpred = dpred_flat.reshape(bs, Q)

            b_t = b_target_all[idx]
            db_t = db_target_all[idx]

            loss = torch.mean((pred - b_t) ** 2)
            if grad_weight > 0.0:
                loss = loss + grad_weight * torch.mean((dpred - db_t) ** 2)
            if energy_weight > 0.0:
                # Central input-space differences of prediction and target
                # values: same discrete operator on both sides, no
                # create_graph (the divergent legacy path is grad_weight).
                h_q = float(xi_base[1] - xi_base[0])
                dpred_fd = (pred[:, 2:] - pred[:, :-2]) / (2.0 * h_q)
                dtarg_fd = (b_t[:, 2:] - b_t[:, :-2]) / (2.0 * h_q)
                eps_b = eps_grid_all[idx][:, 1:-1]
                energy = torch.sum(
                    trap_w_int.unsqueeze(0) * eps_b * (dpred_fd - dtarg_fd) ** 2,
                    dim=1,
                )
                loss = loss + energy_weight * torch.mean(energy)
            data_loss = loss
            if weight_decay > 0.0:
                loss = loss + weight_decay * sum(
                    p.pow(2).sum() for p in model.parameters()
                )
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + float(data_loss.detach()) * bs
            n_processed += bs

        avg_loss = epoch_loss / n_processed
        losses.append(avg_loss)
        if scheduler is not None:
            scheduler.step()

        val_msg = ""
        if val_split is not None:
            model.eval()
            vQ = n_quad
            vepoch_loss = 0.0
            vn = 0
            with torch.no_grad():
                for vs in range(0, len(val_pe), batch_size):
                    vidx = torch.arange(vs, min(vs + batch_size, len(val_pe)),
                                        device=device)
                    vbs = len(vidx)
                    vpe = val_pe[vidx]
                    vrho = val_rho[vidx]
                    vxi = xi_base.unsqueeze(0).expand(vbs, -1).reshape(-1)
                    vpe_f = vpe.unsqueeze(1).expand(vbs, vQ).reshape(-1)
                    vrho_f = vrho.unsqueeze(1).expand(vbs, vQ).reshape(-1)
                    veps_f = (val_eps[vidx].unsqueeze(1).expand(vbs, vQ, -1)
                              .reshape(vbs * vQ, -1) if val_eps is not None else None)
                    veps_per = val_eps[vidx] if val_eps is not None else None
                    vnf = model.norm_at_mid(vpe, vrho, eps_ratios=veps_per)
                    vpred = model(vxi, vpe_f, vrho_f, eps_ratios=veps_f,
                                  norm_factor=vnf).reshape(vbs, vQ)
                    vloss = torch.mean((vpred - val_b[vidx]) ** 2)
                    vepoch_loss = vepoch_loss + float(vloss) * vbs
                    vn += vbs
            model.train()
            avg_val = vepoch_loss / vn
            val_losses.append(avg_val)
            val_msg = f", val={avg_val:.6e}"

            if patience is not None:
                # EMA-smooth the validation signal,decision on the EMA so a
                # single lucky/unlucky epoch (1%-scale oscillation) cannot
                # move the early-stop criterion.
                ema_val = (avg_val if ema_val is None
                           else es_ema_alpha * ema_val + (1 - es_ema_alpha) * avg_val)
                if epoch >= es_warmup:
                    if best_val is None or ema_val < best_val - min_delta:
                        best_val, best_epoch = ema_val, epoch
                        best_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}
                    if epoch - best_epoch >= patience:
                        if verbose:
                            print(f"  early stop at epoch {epoch + 1}/{n_epochs} "
                                  f"(best EMA val={best_val:.6e} @ epoch "
                                  f"{best_epoch + 1}, warm-up {es_warmup})")
                        break

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6e}{val_msg}")

    if patience is not None and best_state is not None and restore_best:
        if verbose and val_losses:
            print(f"  restoring best-val weights (epoch {best_epoch + 1})")
        model.load_state_dict(best_state)

    if val_split is not None:
        return {"train": losses, "val": val_losses}
    return losses


def train_multi_bubble_on_dataset(
    model: MultiKANBubble1D,
    dataset_split: dict[str, dict[str, np.ndarray]],
    mode_names: tuple[str, ...] = ("constant", "xi"),
    n_epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    grad_weight: float = 1e-3,
    n_quad: int = 80,
    verbose: bool = True,
    device: torch.device | None = None,
    lr_scheduler: str | None = None,
    energy_weight: float = 0.0,
    val_split: dict[str, dict[str, np.ndarray]] | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    restore_best: bool = True,
    weight_decay: float = 0.0,
    es_warmup: int = 20,
    es_ema_alpha: float = 0.95,
) -> dict[str, list[float]] | dict[str, dict[str, list[float]]]:
    """Train a multi-bubble model on all modes from a dataset split.

    Trains each bubble independently (one mode per bubble).

    Parameters
    ----------
    model : MultiKANBubble1D
        Multi-bubble model with ``n_bubbles = len(mode_names)``.
    dataset_split : dict
        E.g. ``dataset['train']`` with keys ``"constant"``, ``"xi"``.
    mode_names : tuple[str, ...]
        Mode names in the dataset, matching the model's bubble order.
    val_split : dict or None
        E.g. ``dataset['val']``. If given, per-epoch validation loss is
        computed for each mode and returned as ``{"train": [...], "val": [...]}``
        per mode (see ``train_bubble_on_dataset``).
    patience, min_delta, restore_best
        Forwarded to ``train_bubble_on_dataset`` for early stopping per mode
        (each mode early-stops independently on its own validation loss).
    weight_decay : float
        Forwarded to ``train_bubble_on_dataset``: L2 regularization strength
        applied to each mode bubble's parameters (added to that mode's
        training loss only).
    es_warmup, es_ema_alpha
        Forwarded to ``train_bubble_on_dataset``: early-stopping warm-up
        (epochs) and EMA smoothing factor for the validation-loss signal.

    Returns
    -------
    dict : per-mode loss history. Without ``val_split`` each value is a plain
        ``list[float]`` (train loss); with ``val_split`` each value is
        ``{"train": [...], "val": [...]}``. Histories are truncated to the
        epochs actually run when early stopping triggers.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    histories = {}
    for i, mname in enumerate(mode_names):
        if verbose:
            print(f"Training mode '{mname}' ({i + 1}/{len(mode_names)})")
        vdata = None
        if val_split is not None and mname in val_split:
            vdata = val_split[mname]
        history = train_bubble_on_dataset(
            model.bubbles[i],
            dataset_split[mname],
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            grad_weight=grad_weight,
            n_quad=n_quad,
            verbose=verbose,
            device=device,
            lr_scheduler=lr_scheduler,
            energy_weight=energy_weight,
            val_split=vdata,
            patience=patience,
            min_delta=min_delta,
            restore_best=restore_best,
            weight_decay=weight_decay,
            es_warmup=es_warmup,
            es_ema_alpha=es_ema_alpha,
        )
        histories[mname] = history
    return histories
