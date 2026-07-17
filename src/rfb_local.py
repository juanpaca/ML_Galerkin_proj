import numpy as np


def _solve_tridiagonal(
    sub: np.ndarray, main: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system Ax = rhs.

    sub[i] = A[i+1, i]  for i = 0 .. n-2   (sub-diagonal, length n-1)
    main[i] = A[i, i]    for i = 0 .. n-1   (main diagonal, length n)
    sup[i] = A[i, i+1]   for i = 0 .. n-2   (super-diagonal, length n-1)

    Returns x of length n.
    """
    n = len(main)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = sup[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n):
        denom = main[i] - sub[i - 1] * cp[i - 1]
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
    pe = abs(beta) * h / (2.0 * eps) if eps > 0.0 else np.inf
    rho = sigma * h * h / eps if eps > 0.0 else np.inf
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
    eps : float or ndarray of shape (n_points,)
        Diffusion coefficient.  If a float, constant across the element.
        If an array, per-grid-point values are used (supports variable
        diffusion within the element).  The array must have length n_points.
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

    eps_arr = np.asarray(eps, dtype=float)
    if eps_arr.ndim == 0:
        eps_interior = np.full(n, eps_arr)
    else:
        if len(eps_arr) != n_points:
            raise ValueError(
                f"eps array length {len(eps_arr)} != n_points {n_points}"
            )
        eps_interior = np.interp(interior, xi, eps_arr)

    diff = eps_interior / (h * h)
    diff_coef = diff / dxi**2
    adv = beta / h
    adv_coef = adv / dxi
    if beta >= 0.0:
        lower = -diff_coef - adv_coef
        diag = 2.0 * diff_coef + sigma + adv_coef
        upper = -diff_coef
    else:
        lower = -diff_coef
        diag = 2.0 * diff_coef + sigma - adv_coef
        upper = -diff_coef + adv_coef

    b_int = _solve_tridiagonal(lower[1:], diag, upper[:-1], rhs)

    b = np.zeros(n_points, dtype=float)
    b[1:-1] = b_int
    db = np.gradient(b, dxi)

    center = np.interp(0.5, xi, b)
    if abs(center) < 1e-14:
        b_norm = b
        db_norm = db
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
