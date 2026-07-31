"""Analytic residual-free bubble for constant coefficients.

For constant (eps, beta, sigma) on a single element the local bubble problem

    -eps/h^2 b'' + beta/h b' + sigma b = r(xi),   b(0) = b(1) = 0

is solvable in closed form.  Multiplying by h^2/eps gives

    -b'' + 2 Pe b' + rho b = (h^2/eps) r(xi),

so the *normalized* bubble b/b(0.5) depends only on the dimensionless
numbers Pe and rho (the h^2/eps prefactor cancels).  This module provides
that exact solution as the ground-truth reference for validating the finite
difference solver used to generate the training data.

Solution form: b(xi) = b_p(xi) + A e^{lam1 xi} + B e^{lam2 xi}, with
lam1 = Pe + sqrt(Pe^2 + rho) > 0 and lam2 = Pe - sqrt(Pe^2 + rho) < 0,
A, B fixed by the homogeneous boundary conditions.  All quantities are
computed in forms that stay finite as Pe grows (lam1 ~ 2 Pe).
"""

import numpy as np


def _roots(pe: float, rho: float) -> tuple[float, float]:
    lam1 = pe + np.sqrt(pe * pe + rho)
    lam2 = pe - np.sqrt(pe * pe + rho)
    return lam1, lam2


def _particular(pe: float, rho: float, mode: str):
    """Return callable b_p(xi) and b_p'(xi) for the residual mode."""
    pe = float(pe)
    rho = float(rho)
    if mode == "constant":
        if rho > 0.0:
            c = 1.0 / rho
            return (lambda xi: np.full_like(np.asarray(xi, dtype=float), c),
                    lambda xi: np.zeros_like(np.asarray(xi, dtype=float)))
        else:
            a = 1.0 / (2.0 * pe)
            return (lambda xi: a * np.asarray(xi, dtype=float),
                    lambda xi: np.full_like(np.asarray(xi, dtype=float), a))
    elif mode == "xi":
        if rho > 0.0:
            a = 1.0 / rho
            b0 = -2.0 * pe / (rho * rho)
            return (lambda xi: a * np.asarray(xi, dtype=float) + b0,
                    lambda xi: np.full_like(np.asarray(xi, dtype=float), a))
        else:
            a = 1.0 / (4.0 * pe)
            b0 = 1.0 / (4.0 * pe * pe)
            return (lambda xi: a * np.asarray(xi, dtype=float) ** 2
                    + b0 * np.asarray(xi, dtype=float),
                    lambda xi: 2.0 * a * np.asarray(xi, dtype=float) + b0)
    else:
        raise ValueError(f"unknown residual_mode: {mode}")


def exact_rfb(pe: float, rho: float, mode: str = "constant",
              xi: np.ndarray | None = None) -> dict:
    """Exact (unnormalized) bubble for constant coefficients.

    Parameters
    ----------
    pe, rho : float
        Dimensionless numbers (Pe = beta h / 2 eps, rho = sigma h^2 / eps).
    mode : str
        "constant" (RHS = 1) or "xi" (RHS = xi).
    xi : ndarray, optional
        Evaluation points. Defaults to a fine grid.

    Returns
    -------
    dict with ``xi``, ``b`` (unnormalized), ``center`` = b(0.5),
    ``b_norm`` = b / b(0.5), ``db_norm`` (analytic derivative).
    """
    xi = np.linspace(0.0, 1.0, 8001) if xi is None else np.asarray(xi, dtype=float)
    lam1, lam2 = _roots(pe, rho)
    bp_fn, dbp_fn = _particular(pe, rho, mode)
    bp = bp_fn(xi)
    bp0, bp1 = float(bp_fn(0.0)), float(bp_fn(1.0))

    # A = (bp0 e^{lam2} - bp1) / (e^{lam1} - e^{lam2})   (stable form)
    # A e^{lam1} = (bp0 e^{lam2} - bp1) / (1 - e^{lam2-lam1})
    e2m1 = np.exp(lam2 - lam1)
    A_eL1 = (bp0 * np.exp(lam2) - bp1) / (1.0 - e2m1)   # = A * e^{lam1}
    B = -bp0 - A_eL1 * np.exp(-lam1)                    # B = -bp0 - A

    b = bp + A_eL1 * np.exp(lam1 * (xi - 1.0)) + B * np.exp(lam2 * xi)
    db = (dbp_fn(xi) + A_eL1 * lam1 * np.exp(lam1 * (xi - 1.0))
          + B * lam2 * np.exp(lam2 * xi))

    center = float(np.interp(0.5, xi, b))
    if abs(center) > 1e-300:
        b_norm = b / center
        db_norm = db / center
    else:
        b_norm, db_norm = b, db

    return {"xi": xi, "b": b, "center": center, "b_norm": b_norm,
            "db_norm": db_norm, "roots": (lam1, lam2)}


def fd_rfb(pe: float, rho: float, mode: str = "constant",
           n_points: int = 400, xi_ref: np.ndarray | None = None) -> dict:
    """FD bubble at representative coefficients (eps = 1, h = 1)."""
    from src.rfb_local import solve_reference_rfb

    target = solve_reference_rfb(1.0, 2.0 * pe, rho, 1.0,
                                 residual_mode=mode, n_points=n_points)
    b_norm = target["b"]
    db_norm = target["db"]
    if xi_ref is not None:
        b_norm = np.interp(xi_ref, target["xi"], b_norm)
        db_norm = np.interp(xi_ref, target["xi"], db_norm)
    return {"xi": xi_ref if xi_ref is not None else target["xi"],
            "b_norm": b_norm, "db_norm": db_norm}


def fd_error_metrics(fd: dict, exact: dict, interior: bool = True) -> dict:
    """Error metrics of the FD bubble vs the analytic solution.

    ``b_norm`` can span e^{-Pe}..e^{+Pe} at high Pe, so a single relative
    L2 number is misleading.  Reported metrics:

    * ``l2_rel``          — relative L2 over ``interior`` region only
    * ``l2_rel_full``     — relative L2 over the full domain (incl. layer)
    * ``sup_rel_full``    — max |fd - ex| / max |ex| over the full domain
    * ``l2_rel_layer``    — relative L2 over the boundary layer xi in [0.9, 1]
    * ``peak_rel``        — |max(fd) - max(ex)| / max(ex)
    * ``osc_amp``, ``n_neg`` — negative-value (oscillation) diagnostics
    """
    xi = exact["xi"]
    if interior:
        mask = (xi >= 0.05) & (xi <= 0.95)
    else:
        mask = np.ones_like(xi, dtype=bool)
    b_fd = fd["b_norm"]
    b_ex = exact["b_norm"]
    denom = np.sqrt(np.sum(b_ex[mask] ** 2))
    l2_rel = float(np.sqrt(np.sum((b_fd[mask] - b_ex[mask]) ** 2)) / denom)
    denom_full = np.sqrt(np.sum(b_ex ** 2))
    l2_rel_full = float(np.sqrt(np.sum((b_fd - b_ex) ** 2)) / denom_full)
    sup_rel_full = float(np.max(np.abs(b_fd - b_ex)) / np.max(np.abs(b_ex)))
    layer = (xi >= 0.9) & (xi <= 1.0)
    denom_layer = np.sqrt(np.sum(b_ex[layer] ** 2))
    l2_rel_layer = float(np.sqrt(np.sum((b_fd[layer] - b_ex[layer]) ** 2))
                         / denom_layer)
    peak_rel = float(abs(np.max(b_fd) - np.max(b_ex)) / abs(np.max(b_ex)))
    osc_amp = float(np.min(b_fd)) if np.any(b_fd < 0.0) else 0.0
    n_neg = int(np.sum(b_fd < -1e-14))
    d_fd = fd["db_norm"]
    d_ex = exact["db_norm"]
    denom_d = np.sqrt(np.sum(d_ex[mask] ** 2))
    l2_rel_db = float(np.sqrt(np.sum((d_fd[mask] - d_ex[mask]) ** 2)) / denom_d)
    return {"l2_rel": l2_rel, "l2_rel_full": l2_rel_full,
            "sup_rel_full": sup_rel_full, "l2_rel_layer": l2_rel_layer,
            "peak_rel": peak_rel, "l2_rel_db": l2_rel_db,
            "osc_amp": osc_amp, "n_neg": n_neg}
