import numpy as np
from typing import Callable, Protocol
from src.mesh import Mesh1D


class RFBSolutionProtocol(Protocol):
    mesh: Mesh1D
    def __call__(self, x: np.ndarray) -> np.ndarray: ...


def compute_l2_error(
    solution: RFBSolutionProtocol,
    u_exact: Callable,
    n_points: int = 1000,
) -> float:
    """Compute the L2 error ||u_h - u||_{L²(Ω)}."""
    x = np.linspace(solution.mesh.x_min, solution.mesh.x_max, n_points)
    u_h = solution(x)
    u_e = u_exact(x)

    # Trapezoidal quadrature (fine grid)
    dx = x[1] - x[0]
    error_sq = np.sum((u_h - u_e) ** 2) * dx
    norm_sq = np.sum(u_e ** 2) * dx

    return np.sqrt(error_sq), np.sqrt(norm_sq)


def compute_h1_error(
    solution: RFBSolutionProtocol,
    u_exact: Callable,
    u_exact_grad: Callable,
    n_points: int = 1000,
) -> float:
    """Compute the H1 error ||u_h - u||_{H¹(Ω)}."""
    x = np.linspace(solution.mesh.x_min, solution.mesh.x_max, n_points)
    dx = x[1] - x[0]

    u_h = solution(x)
    u_e = u_exact(x)

    # Approximate u_h' via finite differences
    u_h_grad = np.gradient(u_h, dx)
    u_e_grad = u_exact_grad(x)

    error_sq = np.sum((u_h - u_e) ** 2) * dx + np.sum((u_h_grad - u_e_grad) ** 2) * dx
    norm_sq = np.sum(u_e ** 2) * dx + np.sum(u_e_grad ** 2) * dx

    return np.sqrt(error_sq), np.sqrt(norm_sq)


def compute_energy_error(
    solution: RFBSolutionProtocol,
    u_exact: Callable,
    u_exact_grad: Callable,
    eps: float,
    beta: float,
    n_points: int = 1000,
) -> float:
    """Compute the energy error sqrt(a(u_h - u, u_h - u))."""
    x = np.linspace(solution.mesh.x_min, solution.mesh.x_max, n_points)
    dx = x[1] - x[0]

    u_h = solution(x)
    u_e = u_exact(x)
    u_h_grad = np.gradient(u_h, dx)
    u_e_grad = u_exact_grad(x)

    diff = u_h - u_e
    diff_grad = u_h_grad - u_e_grad

    # Energy norm: a(v, v) = ∫(ε v'² + β v' v) dx
    integrand = eps * diff_grad ** 2 + beta * diff_grad * diff
    error_sq = np.sum(integrand) * dx

    return np.sqrt(max(error_sq, 0.0))


def relative_error_percentage(
    absolute_error: float, norm: float
) -> float:
    """Convert absolute error to relative percentage."""
    if norm < 1e-15:
        return 0.0
    return 100.0 * absolute_error / norm
