"""Static-condensation assembly for enriched P1 with residual-free bubbles.

Problem:  -(epsilon(x) u')' = f,   u(0) = u(L) = 0,
with piecewise-constant epsilon and affine sources f in span{1, x}.

The bubbles are the exact residual functions b_hat = L^-1(1) and
b_tilde = L^-1(xi) on the full domain (computed with the same conservative
FD solver `solve_darcy_1d`).  Enriching a coarse P1 space with these two
bubbles and statically condensing the bubble DOFs yields the residual-free
solution u_hb which, by the RFB theory, reproduces the reference almost
exactly on every element ("if the bubble is well computed, the enriched
solution is near-perfect").  This module also provides the H1-norm gate used
to audit dataset bubbles before training.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.darcy_variable import PiecewiseDiffusion, solve_darcy_1d


def element_stiffness(xa: float, xb: float, profile: PiecewiseDiffusion) -> np.ndarray:
    """Exact local stiffness for piecewise-constant epsilon on [xa, xb]."""
    h = xb - xa
    total = 0.0
    for e0, e1, v in zip(profile.edges[:-1], profile.edges[1:], profile.values):
        lo, hi = max(xa, e0), min(xb, e1)
        if hi > lo:
            total += v * (hi - lo)
    return total / h**2 * np.array([[1.0, -1.0], [-1.0, 1.0]])


def element_load(xa: float, xb: float, source) -> np.ndarray:
    """Local load vector via 3-point Gauss (exact for linear f times P1)."""
    nodes, weights = np.polynomial.legendre.leggauss(3)
    xm, hr = 0.5 * (xa + xb), 0.5 * (xb - xa)
    xg = xm + hr * nodes
    h = xb - xa
    phi1 = (xb - xg) / h
    phi2 = (xg - xa) / h
    fg = np.asarray(source(xg), dtype=float)
    return np.array([
        np.sum(weights * phi1 * fg),
        np.sum(weights * phi2 * fg),
    ]) * hr


def assemble_p1(mesh: np.ndarray, profile: PiecewiseDiffusion, source):
    """Classical P1 Galerkin system with Dirichlet rows/cols removed."""
    n = mesh.size
    A = np.zeros((n, n))
    F = np.zeros(n)
    for e in range(n - 1):
        ke = element_stiffness(mesh[e], mesh[e + 1], profile)
        fe = element_load(mesh[e], mesh[e + 1], source)
        A[e:e + 2, e:e + 2] += ke
        F[e:e + 2] += fe
    free = np.arange(1, n - 1)  # u(0) = u(L) = 0
    return A[np.ix_(free, free)], F[free], free


def assemble_enriched(mesh, profile, source, bubbles, xi_ref, dx):
    """Enriched Galerkin system condensed onto the bubble DOFs.

    Returns the full nodal solution on ``mesh`` plus the bubble coefficients.
    Bubble couplings use composite trapezoid on the fine reference grid;
    P1 blocks stay analytic.
    """
    n = mesh.size
    nb = bubbles.shape[0]
    A_LL, F_L, free = assemble_p1(mesh, profile, source)

    eps_ref = profile.evaluate(xi_ref)
    w = np.full_like(xi_ref, dx)
    w[0] = w[-1] = 0.5 * dx

    def deriv(b):
        return np.gradient(b, xi_ref)

    dbs = [deriv(b) for b in bubbles]

    # Coupling <phi'_m, eps b'_j>: on element e, phi'_{e+1} = +1/h_e,
    # phi'_e = -1/h_e (trapezoid weights halved per element).
    n_grid = xi_ref.size
    step = (n_grid - 1) // (n - 1)
    if step * (n - 1) != n_grid - 1:
        raise ValueError("reference grid must align with the mesh")
    A_Lb_full = np.zeros((n, nb))
    for e in range(n - 1):
        i0, i1 = e * step, (e + 1) * step
        w_loc = np.full(i1 - i0 + 1, dx)
        w_loc[0] *= 0.5
        w_loc[-1] *= 0.5
        h_e = mesh[e + 1] - mesh[e]
        for j, db in enumerate(dbs):
            g_e = np.sum(w_loc * eps_ref[i0:i1 + 1] * db[i0:i1 + 1]) / h_e
            A_Lb_full[e, j] -= g_e
            A_Lb_full[e + 1, j] += g_e
    A_Lb = A_Lb_full[free]

    A_bb = np.empty((nb, nb))
    for i in range(nb):
        for j in range(nb):
            A_bb[i, j] = np.sum(w * eps_ref * dbs[i] * dbs[j])
    F_b = np.array([np.sum(w * np.asarray(source(xi_ref)) * b) for b in bubbles])

    # Static condensation: eliminate nodal DOFs, solve the bubble system.
    A_LL_inv_F = np.linalg.solve(A_LL, F_L)
    A_LL_inv_ALb = np.linalg.solve(A_LL, A_Lb)
    S = A_bb - A_Lb.T @ A_LL_inv_ALb
    g = F_b - A_Lb.T @ A_LL_inv_F
    coeffs = np.linalg.solve(S, g)
    U_free = A_LL_inv_F - A_LL_inv_ALb @ coeffs

    U = np.zeros(n)
    U[free] = U_free
    return U, coeffs


def eval_enriched(u_nodes: np.ndarray, coeffs: np.ndarray | None,
                  bubbles: np.ndarray | None, mesh: np.ndarray,
                  x_query: np.ndarray) -> np.ndarray:
    """Full enriched field: P1 interpolant plus the bubble correction."""
    u = np.interp(x_query, mesh, u_nodes)
    if coeffs is not None:
        for c, b in zip(coeffs, bubbles):
            u += c * b
    return u


def rel_l2(u_h, u_ref, x):
    return np.linalg.norm(u_h - u_ref) / np.maximum(np.linalg.norm(u_ref), 1e-14)


def local_element_bubbles(profile: PiecewiseDiffusion, xa: float, xb: float,
                          n_fd: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FD-computed local residual bubbles for one element [xa, xb].

    Solves -(eps u')' = r on (xa, xb), u(xa)=u(xb)=0 for r in {1, xi_local}
    with the same harmonic-conservative solver as the reference, normalized
    by the local midpoint. Returns (xi_local, bubbles[2, n_fd]).
    """
    h = xb - xa
    diffusion = lambda xi_loc: profile.evaluate(xa + h * xi_loc)
    b0 = solve_darcy_1d(diffusion, length=h, source=1.0, n_points=n_fd)["u_norm"]
    b1 = solve_darcy_1d(diffusion, length=h,
                        source=lambda x: (x - xa) / h, n_points=n_fd)["u_norm"]
    xi_local = np.linspace(0.0, 1.0, n_fd)
    return xi_local, np.stack([b0, b1])


def local_enrichment_h1_error(
    profile: PiecewiseDiffusion,
    source: Callable[[np.ndarray], np.ndarray] | float = 1.0,
    n_fd_bubble: int = 801,
    n_fd_ref: int = 16001,
    n_el: int = 8,
) -> dict:
    """True RFB enrichment audit: per-element local bubbles + condensation.

    Reference u_ref (fine FD) vs the P1+local-bubble static-condensation
    solution u_hb. For affine sources f in span{1, x} and accurately
    computed local bubbles this is near machine precision; the residual is
    dominated by the FD quality of the bubble data (the dataset label).
    """
    xi_ref = np.linspace(0.0, 1.0, n_fd_ref)
    dx_ref = xi_ref[1] - xi_ref[0]
    ref = solve_darcy_1d(profile, length=1.0, source=source, n_points=n_fd_ref)
    u_ref, du_ref = ref["u"], np.gradient(ref["u"], dx_ref)

    mesh = np.linspace(0.0, 1.0, n_el + 1)
    n_nodes = mesh.size
    free = np.arange(1, n_nodes - 1)

    # ---- per-element local blocks ----
    blocks = []  # (xa, xb, A_LL, A_Lb, A_bb, F_L, F_b, xi_loc, bub, x_loc)
    for e in range(n_el):
        xa, xb = mesh[e], mesh[e + 1]
        h = xb - xa
        xi_loc, B = local_element_bubbles(profile, xa, xb, n_fd_bubble)
        dxi = xi_loc[1] - xi_loc[0]
        w = np.full(n_fd_bubble, dxi)
        w[0] = w[-1] = 0.5 * dxi
        x_loc = xa + h * xi_loc
        eps_loc = profile.evaluate(x_loc)
        db = np.gradient(B, xi_loc, axis=1) / h  # dB/dx

        A_bb = np.array([[np.sum(w * eps_loc * dbi * dbj)
                          for dbj in db] for dbi in db])
        F_b = np.array([np.sum(w * np.asarray(source(x_loc)) * b) for b in B])
        I_b = np.array([np.sum(w * eps_loc * dbb) for dbb in db])  # int eps b'
        A_Lb = np.vstack([-I_b / h, +I_b / h])   # phi'_0=-1/h, phi'_1=+1/h
        A_LL = element_stiffness(xa, xb, profile)
        F_L = element_load(xa, xb, source)
        blocks.append((xa, xb, A_LL, A_Lb, A_bb, F_L, F_b, xi_loc, B, x_loc))

    # ---- assemble condensed nodal system + recover bubble coeffs ----
    A_cond = np.zeros((n_nodes, n_nodes))
    F_cond = np.zeros(n_nodes)
    bubble_coeffs = np.zeros((n_el, 2))
    for e, (xa, xb, A_LL, A_Lb, A_bb, F_L, F_b, xi_loc, B, x_loc) in enumerate(blocks):
        Abb_inv = np.linalg.solve(A_bb, np.eye(2))
        A_cond[e:e + 2, e:e + 2] += A_LL - A_Lb @ Abb_inv @ A_Lb.T
        F_cond[e:e + 2] += F_L - A_Lb @ Abb_inv @ F_b

    U = np.zeros(n_nodes)
    U[free] = np.linalg.solve(A_cond[np.ix_(free, free)], F_cond[free])

    for e, (xa, xb, A_LL, A_Lb, A_bb, F_L, F_b, xi_loc, B, x_loc) in enumerate(blocks):
        U_e = U[e:e + 2]
        bubble_coeffs[e] = np.linalg.solve(A_bb, F_b - A_Lb.T @ U_e)

    # ---- evaluate u_hb on the reference grid ----
    u_hb = np.interp(xi_ref, mesh, U)
    for e, (xa, xb, A_LL, A_Lb, A_bb, F_L, F_b, xi_loc, B, x_loc) in enumerate(blocks):
        mask = (xi_ref >= xa) & (xi_ref <= xb)
        if e < n_el - 1:
            mask = (xi_ref >= xa) & (xi_ref < xb)
        xi_e = (xi_ref[mask] - xa) / (xb - xa)
        b_e = np.vstack([np.interp(xi_e, xi_loc, B[k]) for k in range(B.shape[0])])
        u_hb[mask] += bubble_coeffs[e] @ b_e
    du_hb = np.gradient(u_hb, dx_ref)

    def rel_h1(u_h, du_h, u_e, du_e):
        err_sq = np.sum((u_h - u_e) ** 2) * dx_ref + np.sum((du_h - du_e) ** 2) * dx_ref
        norm_sq = np.sum(u_e ** 2) * dx_ref + np.sum(du_e ** 2) * dx_ref
        return float(np.sqrt(err_sq) / np.sqrt(norm_sq))

    return {
        "rel_h1_enriched": rel_h1(u_hb, du_hb, u_ref, du_ref),
        "rel_l2_enriched": float(rel_l2(u_hb, u_ref, xi_ref)),
    }


def enrichment_h1_error(
    profile: PiecewiseDiffusion,
    source: Callable[[np.ndarray], np.ndarray] | float = 1.0,
    n_fd: int = 801,
    n_el: int = 8,
) -> dict:
    """Enrichment-quality audit for one profile: u_ref vs P1 vs enriched.

    Reference and bubbles share the same conservative FD solver and grid, so
    the enriched-solution H1-relative error isolates (FD + quadrature) error.
    A well-computed bubble pair gives a value near the FD truncation level.
    """
    xi_ref = np.linspace(0.0, 1.0, n_fd)
    dx = xi_ref[1] - xi_ref[0]

    ref = solve_darcy_1d(profile, length=1.0, source=source, n_points=n_fd)
    u_ref = ref["u"]
    du_ref = np.gradient(u_ref, dx)

    bubbles = np.stack([
        solve_darcy_1d(profile, length=1.0, source=s, n_points=n_fd)["u_norm"]
        for s in (1.0, lambda x: x)
    ])

    mesh = np.linspace(0.0, 1.0, n_el + 1)
    A_LL, F_L, free = assemble_p1(mesh, profile, source)
    u_gal_nodes = np.zeros(n_el + 1)
    u_gal_nodes[free] = np.linalg.solve(A_LL, F_L)
    u_gal = np.interp(xi_ref, mesh, u_gal_nodes)
    du_gal = np.gradient(u_gal, dx)

    u_en_nodes, coeffs = assemble_enriched(mesh, profile, source, bubbles,
                                           xi_ref, dx)
    u_en = eval_enriched(u_en_nodes, coeffs, bubbles, mesh, xi_ref)
    du_en = np.gradient(u_en, dx)

    def rel_h1_eval(u_h, du_h, u_e, du_e):
        err_sq = np.sum((u_h - u_e) ** 2) * dx + np.sum((du_h - du_e) ** 2) * dx
        norm_sq = np.sum(u_e ** 2) * dx + np.sum(du_e ** 2) * dx
        return float(np.sqrt(err_sq) / np.sqrt(norm_sq))

    return {
        "rel_h1_galerkin": rel_h1_eval(u_gal, du_gal, u_ref, du_ref),
        "rel_h1_enriched": rel_h1_eval(u_en, du_en, u_ref, du_ref),
        "rel_l2_enriched": float(rel_l2(u_en, u_ref, xi_ref)),
        "coeffs": coeffs,
    }


def enrichment_l2_gate(
    profile: PiecewiseDiffusion,
    bubbles: np.ndarray,
    source_specs: dict[str, Callable],
    n_ref: int = 32001,
    n_el: int = 8,
) -> dict[str, float]:
    """Per-source enriched rel-L2 error vs an independent fine reference.

    ``bubbles`` is the (n_modes, n_fd) stack of the full-domain residual
    bubbles (b_hat = L^-1(1), b_tilde = L^-1(xi)) on the normalized grid.
    For every ``source_specs`` mode the P1+nodal solution on a uniform
    ``n_el``+1 grid is enriched with the bubbles via global static
    condensation (the deployment assembly in ``assemble_enriched``), then
    compared to a fresh reference solve at ``n_ref`` points.  This decouples
    label quality from the reference-grid truncation: a well-computed bubble
    gives rel-L2 of order ~1e-3 (typical) / ~1e-2 (worst-profile) at
    n_fd=3201, which is well below the KAN approximation error.  Samples
    above ``gate-threshold`` are the FD-under-resolved outliers dropped by
    ``--gate-drop``.
    """
    xi = np.linspace(0.0, 1.0, bubbles.shape[1])
    dx = xi[1] - xi[0]
    mesh = np.linspace(0.0, 1.0, n_el + 1)
    if (xi.size - 1) % n_el != 0:
        raise ValueError(
            f"bubble grid ({xi.size}) must align with the P1 mesh ({n_el} elements)")
    out = {}
    for mode, src in source_specs.items():
        ref = solve_darcy_1d(profile, length=1.0, source=src, n_points=n_ref)
        u_nodes, coeffs = assemble_enriched(mesh, profile, src, bubbles, xi, dx)
        u_en = eval_enriched(u_nodes, coeffs, bubbles, mesh, xi)
        out[mode] = float(rel_l2(np.interp(ref["x"], xi, u_en), ref["u"], ref["x"]))
    return out