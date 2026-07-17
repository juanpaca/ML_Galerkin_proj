"""Manufactured solutions for 1D advection-diffusion with random parameters.

We generate synthetic solutions u_k(x) that exhibit boundary-layer behavior
controlled by random parameters, serving as the training manifold.
"""

import numpy as np
from typing import Callable, Tuple
import sympy as sp


def boundary_layer_solution(
    x: np.ndarray,
    eps: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    layer_amp: float = 1.0,
) -> np.ndarray:
    """Smooth manufactured solution with boundary layers.

    u(x) = x*(1-x) * (alpha + layer_amp * (exp(-x/eps) + exp(-(1-x)/eps)))
         + beta * sin(pi*x)

    When eps is small, u has sharp boundary layers at x=0 and x=1.
    """
    x = np.asarray(x, dtype=float)
    smooth = alpha + layer_amp * (np.exp(-x / eps) + np.exp(-(1.0 - x) / eps))
    return x * (1.0 - x) * smooth + beta * np.sin(np.pi * x)


def boundary_layer_solution_grad(
    x: np.ndarray,
    eps: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    layer_amp: float = 1.0,
) -> np.ndarray:
    """Derivative of boundary_layer_solution w.r.t. x."""
    x = np.asarray(x, dtype=float)
    exp0 = np.exp(-x / eps)
    exp1 = np.exp(-(1.0 - x) / eps)
    smooth = alpha + layer_amp * (exp0 + exp1)
    dsmooth = layer_amp * (-exp0 / eps + exp1 / eps)

    dudx = (1.0 - 2.0 * x) * smooth + x * (1.0 - x) * dsmooth
    dudx += beta * np.pi * np.cos(np.pi * x)
    return dudx


def boundary_layer_solution_laplacian(
    x: np.ndarray,
    eps: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    layer_amp: float = 1.0,
) -> np.ndarray:
    """Second derivative of boundary_layer_solution w.r.t. x."""
    x = np.asarray(x, dtype=float)
    exp0 = np.exp(-x / eps)
    exp1 = np.exp(-(1.0 - x) / eps)
    smooth = alpha + layer_amp * (exp0 + exp1)
    dsmooth = layer_amp * (-exp0 / eps + exp1 / eps)
    d2smooth = layer_amp * (exp0 / eps**2 + exp1 / eps**2)

    d2udx2 = (
        -2.0 * smooth
        + 2.0 * (1.0 - 2.0 * x) * dsmooth
        + x * (1.0 - x) * d2smooth
    )
    d2udx2 -= beta * np.pi**2 * np.sin(np.pi * x)
    return d2udx2


def compute_source_from_solution(
    u_grad: Callable,
    u_laplacian: Callable,
    eps_pde: float,
    beta_pde: float,
) -> Callable:
    """Compute the PDE source term f(x) = -eps * u'' + beta * u'.

    Given derivatives of the analytical solution, returns f such
    that u satisfies -eps u'' + beta u' = f.
    """
    def source(x):
        x = np.asarray(x, dtype=float)
        return -eps_pde * u_laplacian(x) + beta_pde * u_grad(x)
    return source


def random_manufactured_solution(
    rng: np.random.Generator | None = None,
    eps_range: tuple[float, float] = (0.01, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    beta_range: tuple[float, float] = (0.0, 2.0),
    layer_amp_range: tuple[float, float] = (0.0, 2.0),
) -> dict:
    """Create a random manufactured solution with its source term.

    Returns a dict with keys:
        'u': callable u(x)
        'u_grad': callable u'(x)
        'u_laplacian': callable u''(x)
        'params': dict of parameters used
    """
    if rng is None:
        rng = np.random.default_rng()

    eps = 10.0 ** rng.uniform(np.log10(eps_range[0]), np.log10(eps_range[1]))
    alpha = rng.uniform(*alpha_range)
    beta = rng.uniform(*beta_range)
    layer_amp = rng.uniform(*layer_amp_range)

    params = {"eps": eps, "alpha": alpha, "beta": beta, "layer_amp": layer_amp}

    return {
        "u": lambda x, p=params: boundary_layer_solution(x, **p),
        "u_grad": lambda x, p=params: boundary_layer_solution_grad(x, **p),
        "u_laplacian": lambda x, p=params: boundary_layer_solution_laplacian(x, **p),
        "params": params,
    }


def generate_training_manifold(
    n_snapshots: int = 10,
    seed: int = 42,
    **kwargs,
) -> list[dict]:
    """Generate a training manifold of random manufactured solutions."""
    rng = np.random.default_rng(seed)
    return [random_manufactured_solution(rng, **kwargs) for _ in range(n_snapshots)]


def _sinh_ratio_stable(z: np.ndarray, m: float) -> np.ndarray:
    """Compute sinh(m z) / sinh(m) stably for z in [-1, 1]."""
    z = np.asarray(z, dtype=float)
    if m < 50.0:
        return np.sinh(m * z) / np.sinh(m)

    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = np.exp(m * (z[pos] - 1.0)) * (
        1.0 - np.exp(-2.0 * m * z[pos])
    ) / (1.0 - np.exp(-2.0 * m))
    zp = -z[~pos]
    out[~pos] = -np.exp(m * (zp - 1.0)) * (
        1.0 - np.exp(-2.0 * m * zp)
    ) / (1.0 - np.exp(-2.0 * m))
    return out


def advection_diffusion_layer_solution(
    y: np.ndarray,
    eps: float = 1e-3,
    a: float = 1.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """Exact 1D profile for the unit-square layer benchmark.

    The 2D benchmark solution is independent of x, so the same formula is a
    1D solution in y of

        -eps u'' + a u' + sigma u = 1,   u(0)=u(1)=0.

    Parameters follow the notation in the benchmark: advective field
    alpha=(0,a)^T, diffusion eps, and reaction sigma.
    """
    y = np.asarray(y, dtype=float)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if a < 0.0 or sigma < 0.0:
        raise ValueError("a and sigma must be non-negative")

    if sigma > 0.0:
        m = np.sqrt(a * a + 4.0 * eps * sigma) / (2.0 * eps)
        r = a / (2.0 * eps)
        term_left = _sinh_ratio_stable(y - 1.0, m) * np.exp(r * y)
        term_right = _sinh_ratio_stable(y, m) * np.exp(r * (y - 1.0))
        return (term_left - term_right + 1.0) / sigma

    if a <= 0.0:
        return y * (1.0 - y) / (2.0 * eps)

    k = a / (2.0 * eps)
    if k < 50.0:
        layer_term = np.sinh(k * y) / np.sinh(k) * np.exp(k * (y - 1.0))
    else:
        layer_term = np.exp(2.0 * k * (y - 1.0)) * (
            1.0 - np.exp(-2.0 * k * y)
        ) / (1.0 - np.exp(-2.0 * k))
    return (y - layer_term) / a


def _finite_difference_derivative(fn: Callable, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yp = np.minimum(1.0, y + h)
    ym = np.maximum(0.0, y - h)
    denom = yp - ym
    return (fn(yp) - fn(ym)) / denom


def advection_diffusion_layer_solution_grad(
    y: np.ndarray,
    eps: float = 1e-3,
    a: float = 1.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """Numerical derivative of advection_diffusion_layer_solution."""
    fn = lambda z: advection_diffusion_layer_solution(z, eps=eps, a=a, sigma=sigma)
    h = min(1e-6, max(1e-10, eps * 1e-2))
    return _finite_difference_derivative(fn, y, h=h)


def advection_diffusion_layer_solution_laplacian(
    y: np.ndarray,
    eps: float = 1e-3,
    a: float = 1.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """Compute u'' from the PDE identity -eps u'' + a u' + sigma u = 1."""
    y = np.asarray(y, dtype=float)
    u = advection_diffusion_layer_solution(y, eps=eps, a=a, sigma=sigma)
    ug = advection_diffusion_layer_solution_grad(y, eps=eps, a=a, sigma=sigma)
    return (a * ug + sigma * u - 1.0) / eps


def random_advection_diffusion_layer_solution(
    rng: np.random.Generator | None = None,
    eps_range: tuple[float, float] = (1e-3, 1e-1),
    a_range: tuple[float, float] = (1.0, 1.0),
    sigma_range: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Random snapshot from the exact advection-diffusion layer benchmark."""
    if rng is None:
        rng = np.random.default_rng()

    eps = 10.0 ** rng.uniform(np.log10(eps_range[0]), np.log10(eps_range[1]))
    a = rng.uniform(*a_range)
    sigma = rng.uniform(*sigma_range)
    params = {"eps": eps, "a": a, "sigma": sigma}

    return {
        "u": lambda y, p=params: advection_diffusion_layer_solution(y, **p),
        "u_grad": lambda y, p=params: advection_diffusion_layer_solution_grad(y, **p),
        "u_laplacian": lambda y, p=params: advection_diffusion_layer_solution_laplacian(y, **p),
        "source": lambda y: np.ones_like(np.asarray(y, dtype=float)),
        "params": params,
    }


def generate_advection_diffusion_layer_manifold(
    n_snapshots: int = 10,
    seed: int = 42,
    **kwargs,
) -> list[dict]:
    """Generate the benchmark layer-solution training manifold."""
    rng = np.random.default_rng(seed)
    return [
        random_advection_diffusion_layer_solution(rng, **kwargs)
        for _ in range(n_snapshots)
    ]
