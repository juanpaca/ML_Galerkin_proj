"""One-dimensional Darcy data with piecewise variable diffusion.

The reference problem is

    -(epsilon(x) u'(x))' = f(x),   x in (0, L),
    u(0) = u(L) = 0.

Solutions are normalized by their midpoint for shape-based learning. The
profile is represented to a model by fixed samples of epsilon divided by its
trapezoidal mean; the full profile and piece boundaries are retained in the
pool for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.rfb_local import _solve_tridiagonal


@dataclass(frozen=True)
class PiecewiseDiffusion:
    """Positive piecewise-constant diffusion on the normalized interval."""

    edges: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        edges = np.asarray(self.edges, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if edges.ndim != 1 or values.ndim != 1 or edges.size != values.size + 1:
            raise ValueError("edges must have one more entry than values")
        if edges[0] != 0.0 or edges[-1] != 1.0 or np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing from 0 to 1")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("diffusion values must be positive and finite")

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        if np.any(xi < 0.0) or np.any(xi > 1.0):
            raise ValueError("profile coordinates must lie in [0, 1]")
        indices = np.minimum(np.searchsorted(self.edges[1:-1], xi, side="right"),
                             self.values.size - 1)
        return self.values[indices]


def random_piecewise_diffusion(
    rng: np.random.Generator,
    n_pieces: int | None = None,
    eps_range: tuple[float, float] = (0.1, 100.0),
    n_pieces_range: tuple[int, int] = (2, 8),
) -> PiecewiseDiffusion:
    """Generate a random positive piecewise profile with log-uniform values."""
    lo, hi = map(float, eps_range)
    if lo <= 0.0 or hi <= lo:
        raise ValueError("eps_range must be positive and increasing")
    if n_pieces is None:
        p_lo, p_hi = n_pieces_range
        if p_lo < 1 or p_hi < p_lo:
            raise ValueError("invalid n_pieces_range")
        n_pieces = int(rng.integers(p_lo, p_hi + 1))
    if n_pieces < 1:
        raise ValueError("n_pieces must be positive")
    interior = np.sort(rng.uniform(0.0, 1.0, n_pieces - 1))
    edges = np.concatenate(([0.0], interior, [1.0]))
    values = 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n_pieces)
    return PiecewiseDiffusion(edges, values)


def solve_darcy_1d(
    diffusion: PiecewiseDiffusion | Callable[[np.ndarray], np.ndarray] | np.ndarray | float,
    length: float = 1.0,
    source: Callable[[np.ndarray], np.ndarray] | float = 1.0,
    n_points: int = 801,
) -> dict[str, np.ndarray | float]:
    """Solve the conservative 1D Darcy problem on a uniform grid.

    Harmonic face averages make the discretization conservative across
    diffusion jumps. The returned ``u_norm`` is the shape target for training.
    """
    if length <= 0.0 or n_points < 5:
        raise ValueError("length must be positive and n_points must be >= 5")
    xi = np.linspace(0.0, 1.0, n_points)
    x = length * xi
    if isinstance(diffusion, PiecewiseDiffusion):
        eps = diffusion.evaluate(xi)
    elif callable(diffusion):
        eps = np.asarray(diffusion(xi), dtype=float)
    else:
        eps = np.asarray(diffusion, dtype=float)
    if eps.ndim == 0:
        eps = np.full(n_points, float(eps))
    if eps.shape != xi.shape or not np.all(np.isfinite(eps)) or np.any(eps <= 0.0):
        raise ValueError("diffusion must be positive finite data on the grid")
    rhs = np.full(n_points - 2, float(source)) if np.isscalar(source) else np.asarray(source(x[1:-1]), dtype=float)
    if rhs.shape != (n_points - 2,) or not np.all(np.isfinite(rhs)):
        raise ValueError("source must return finite interior values")

    dx = x[1] - x[0]
    faces = 2.0 * eps[:-1] * eps[1:] / (eps[:-1] + eps[1:])
    lower = faces[:-1] / dx**2
    upper = faces[1:] / dx**2
    diagonal = lower + upper
    u_inner = _solve_tridiagonal(-lower[1:], diagonal, -upper[:-1], rhs)
    u = np.zeros(n_points, dtype=float)
    u[1:-1] = u_inner
    center = float(np.interp(0.5, xi, u))
    if abs(center) < 1e-14:
        raise np.linalg.LinAlgError("Darcy solution midpoint is too small to normalize")
    u_norm = u / center
    return {
        "x": x, "xi": xi, "u": u, "u_norm": u_norm,
        "du_norm": np.gradient(u_norm, xi), "center": center,
        "eps": eps,
    }


def profile_features(profile: PiecewiseDiffusion, n_features: int) -> np.ndarray:
    """Sample a profile at fixed Gauss points and normalize by its mean."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n_features)
    xi = 0.5 * (nodes + 1.0)
    values = profile.evaluate(xi)
    # Gauss weights integrate the profile on [0, 1].
    mean = float(np.sum(0.5 * weights * values))
    return values / mean


def generate_darcy_pool(
    n_samples: int = 5000,
    n_fd_points: int = 801,
    n_profile_features: int = 8,
    length_range: tuple[float, float] = (0.5, 2.0),
    eps_range: tuple[float, float] = (0.1, 100.0),
    n_pieces_range: tuple[int, int] = (2, 8),
    seed: int = 42,
) -> dict:
    """Generate a deterministic pool of piecewise-diffusion Darcy shapes."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(*length_range, n_samples)
    xi = np.linspace(0.0, 1.0, n_fd_points)
    b, db, ratios, profiles, edges, values = [], [], [], [], [], []
    for i in range(n_samples):
        profile = random_piecewise_diffusion(rng, eps_range=eps_range,
                                              n_pieces_range=n_pieces_range)
        sol = solve_darcy_1d(profile, length=float(lengths[i]), n_points=n_fd_points)
        b.append(sol["u_norm"])
        db.append(sol["du_norm"])
        ratios.append(profile_features(profile, n_profile_features))
        profiles.append(sol["eps"])
        edges.append(profile.edges)
        values.append(profile.values)

    # Variable piece counts cannot form a rectangular edge/value array; retain
    # them as metadata-friendly object arrays while the fixed profile samples
    # are the model features.
    mode = {
        "pe": np.zeros(n_samples, dtype=np.float32),
        "rho": np.zeros(n_samples, dtype=np.float32),
        "length": lengths.astype(np.float32),
        "b": np.asarray(b, dtype=np.float32),
        "db": np.asarray(db, dtype=np.float32),
        "eps_ratios": np.asarray(ratios, dtype=np.float32),
        "eps_profile": np.asarray(profiles, dtype=np.float32),
        "xi": xi.astype(np.float32),
        "idx": np.arange(n_samples, dtype=np.int64),
    }
    return {
        "constant": mode,
        "mode_names": ("constant",),
        "piece_edges": edges,
        "piece_values": values,
        "metadata": {
            "problem": "-(epsilon(x) u\')\' = f",
            "coordinate": "xi=x/L",
            "n_samples": n_samples,
            "n_fd_points": n_fd_points,
            "n_profile_features": n_profile_features,
            "length_range": list(length_range),
            "eps_range": list(eps_range),
            "n_pieces_range": list(n_pieces_range),
            "seed": seed,
            "normalization": "u/u(0.5L)",
        },
    }
