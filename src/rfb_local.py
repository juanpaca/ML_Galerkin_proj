import numpy as np
from typing import Callable


def evaluate_diffusion_profile(
    eps: float | np.ndarray | Callable[[np.ndarray], np.ndarray], xi: np.ndarray
) -> np.ndarray:
    """Evaluate and validate a positive diffusion profile on ``xi``.

    ``eps`` may be a positive scalar, an array sampled on ``xi``, or a
    callable accepting the complete vector of coordinates. This is the common
    profile boundary used by the FD reference solver and FEM assembly.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.ndim != 1 or xi.size < 2 or not np.all(np.isfinite(xi)):
        raise ValueError("xi must be a finite one-dimensional grid")
    if np.any(np.diff(xi) <= 0):
        raise ValueError("xi must be strictly increasing")
    if callable(eps):
        values = np.asarray(eps(xi), dtype=float)
    else:
        values = np.asarray(eps, dtype=float)
    if values.ndim == 0:
        values = np.full(xi.shape, float(values))
    if values.shape != xi.shape:
        raise ValueError("diffusion profile must match the xi grid")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("diffusion profile must contain positive finite values")
    return values


def trapezoidal_weights(xi: np.ndarray) -> np.ndarray:
    """Return integration weights for a strictly increasing 1D grid."""
    xi = np.asarray(xi, dtype=float)
    if xi.ndim != 1 or xi.size < 2 or np.any(np.diff(xi) <= 0):
        raise ValueError("xi must be a strictly increasing grid")
    weights = np.empty(xi.size, dtype=float)
    weights[0] = 0.5 * (xi[1] - xi[0])
    weights[-1] = 0.5 * (xi[-1] - xi[-2])
    weights[1:-1] = 0.5 * (xi[2:] - xi[:-2])
    return weights


def _solve_tridiagonal(
    sub: np.ndarray, main: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system Ax = rhs.

    sub[i] = A[i+1, i]  for i = 0 .. n-2   (sub-diagonal, length n-1)
    main[i] = A[i, i]    for i = 0 .. n-1   (main diagonal, length n)
    sup[i] = A[i, i+1]   for i = 0 .. n-2   (super-diagonal, length n-1)

    Returns x of length n.
    """
    sub = np.asarray(sub, dtype=float)
    main = np.asarray(main, dtype=float)
    sup = np.asarray(sup, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    n = len(main)
    if n < 1 or sub.size != n - 1 or sup.size != n - 1 or rhs.size != n:
        raise ValueError("invalid tridiagonal system dimensions")
    if not all(np.all(np.isfinite(a)) for a in (sub, main, sup, rhs)):
        raise ValueError("tridiagonal system contains non-finite values")
    scale = max(1.0, float(np.max(np.abs(main))))
    pivot_tol = 100.0 * np.finfo(float).eps * scale
    if abs(main[0]) <= pivot_tol:
        raise np.linalg.LinAlgError("zero or unstable tridiagonal pivot")
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = sup[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n):
        denom = main[i] - sub[i - 1] * cp[i - 1]
        if abs(denom) <= pivot_tol:
            raise np.linalg.LinAlgError("zero or unstable tridiagonal pivot")
        if i < n - 1:
            cp[i] = sup[i] / denom
        dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def reference_p1_basis(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P1 basis and reference derivatives on [0, 1]."""
    xi = np.asarray(xi, dtype=float)
    phi = np.vstack([1.0 - xi, xi])
    dphi_dxi = np.vstack([-np.ones_like(xi), np.ones_like(xi)])
    return phi, dphi_dxi


def local_parameters(eps: float, beta: float, sigma: float, h: float) -> tuple[float, float]:
    if not np.all(np.isfinite([eps, beta, sigma, h])) or eps <= 0.0 or h <= 0.0:
        raise ValueError("eps and h must be positive finite values")
    pe = abs(beta) * h / (2.0 * eps)
    rho = sigma * h * h / eps
    return pe, rho


def solve_reference_rfb(
    eps: float | np.ndarray,
    beta: float,
    sigma: float,
    h: float,
    residual_mode: str = "constant",
    n_points: int = 400,
) -> dict:
    """Solve the local residual-free bubble problem on [0, 1].

    Reference equation:
        -(eps(xi)/h^2) b_xixi + (beta/h) b_xi + sigma b = r(xi),
        b(0)=b(1)=0.

    Parameters
    ----------
    eps : float, ndarray, or callable
        Diffusion coefficient. A scalar, a profile sampled on the FD grid, or
        a callable ``eps(xi)`` returning the profile can be supplied.
    beta, sigma : float
        Advection and reaction (constant within the element).
    h : float
        Element length.
    residual_mode : str
        Right-hand side function. One of:
        ``"constant"`` (RHS = 1),
        ``"xi"`` (RHS = xi),
        ``"one_minus_xi"`` (RHS = 1 - xi),
        ``"companion_1"`` (RHS = beta/h - sigma*(1-xi)),
        ``"companion_2"`` (RHS = -beta/h - sigma*xi).
    n_points : int
        Number of FD grid points (must be >= 5).

    Returns the normalized bubble b/b(0.5), its derivative, and raw values.
    """
    if n_points < 5:
        raise ValueError("n_points must be at least 5")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a positive finite value")
    if not np.isfinite(beta) or not np.isfinite(sigma):
        raise ValueError("beta and sigma must be finite")
    xi = np.linspace(0.0, 1.0, n_points)
    dxi = xi[1] - xi[0]
    interior = xi[1:-1]
    n = len(interior)

    if residual_mode == "constant":
        rhs = np.ones(n)
    elif residual_mode == "xi":
        rhs = interior.copy()
    elif residual_mode == "one_minus_xi":
        rhs = 1.0 - interior
    elif residual_mode == "companion_1":
        rhs = beta / h - sigma * (1.0 - interior)
    elif residual_mode == "companion_2":
        rhs = -beta / h - sigma * interior
    else:
        raise ValueError(f"unknown residual_mode: {residual_mode}")

    eps_grid = evaluate_diffusion_profile(eps, xi)
    # Face diffusion values give a conservative discretization of
    # -(eps(xi) b_xi)_xi. Harmonic averaging is robust across jumps.
    eps_faces = 2.0 * eps_grid[:-1] * eps_grid[1:] / (eps_grid[:-1] + eps_grid[1:])
    diff_scale = 1.0 / (h * h * dxi**2)
    diff_left = eps_faces[:-1] * diff_scale
    diff_right = eps_faces[1:] * diff_scale
    adv = beta / h
    adv_coef = adv / dxi
    if beta >= 0.0:
        lower = -diff_left - adv_coef
        diag = diff_left + diff_right + sigma + adv_coef
        upper = -diff_right
    else:
        lower = -diff_left
        diag = diff_left + diff_right + sigma - adv_coef
        upper = -diff_right + adv_coef

    b_int = _solve_tridiagonal(lower[1:], diag, upper[:-1], rhs)

    b = np.zeros(n_points, dtype=float)
    b[1:-1] = b_int
    db = np.gradient(b, dxi)

    center = np.interp(0.5, xi, b)
    if abs(center) < 1e-14:
        raise np.linalg.LinAlgError("bubble midpoint is too small to normalize")
    else:
        b_norm = b / center
        db_norm = db / center

    return {
        "xi": xi,
        "b": b_norm,
        "db": db_norm,
        "b_raw": b,
        "db_raw": db,
        "center": center,
        "params": {"eps": eps, "beta": beta, "sigma": sigma, "h": h},
    }


def interpolate_target(target: dict, xi_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi_eval = np.asarray(xi_eval, dtype=float)
    b = np.interp(xi_eval, target["xi"], target["b"])
    db = np.interp(xi_eval, target["xi"], target["db"])
    return b, db
