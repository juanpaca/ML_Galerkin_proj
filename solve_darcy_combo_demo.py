"""Solve one variable-diffusion Darcy problem three ways.

Problem:  -(epsilon(x) u')' = 1 - x,   u(0) = u(1) = 0,
with a piecewise-constant coefficient taken from the *test* split of the
trained Darcy pool. Three approximations are compared:

1. Reference      -- conservative FD solve on a fine grid
                     (`src.darcy_variable.solve_darcy_1d`).
2. Galerkin       -- classical P1 FEM on a coarse uniform mesh.
3. RFB-KAN        -- the P1 space enriched with the two KAN-learned
                     residual-free bubbles b_hat = L^-1(1) and
                     b_tilde = L^-1(xi), assembled via static
                     condensation (Schur complement onto the bubble DOFs).

``--check`` replaces the KAN shapes by the exact FD bubbles; since
f = 1 - xi lies in span{1, xi}, the enriched solution then reproduces the
reference almost exactly, validating the assembler independently of the
network.

Usage:
    venv/bin/python solve_darcy_combo_demo.py [--index N] [--elements M]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset_generation import load_dataset
from src.darcy_variable import PiecewiseDiffusion, make_profile_features, solve_darcy_1d
from src.rfb_bubble import MultiKANBubble1D

DATASET_NAME = "darcy_piecewise_combo20k"
SUBDIR = "data_darcy_variable"
MODEL_PATH = Path(f"models/{DATASET_NAME}_kan.pt")
N_FEATURES = 24


def element_stiffness(xa: float, xb: float, profile: PiecewiseDiffusion) -> np.ndarray:
    """Exact local stiffness for piecewise-constant epsilon on [xa, xb]."""
    h = xb - xa
    # int_a^b epsilon dx by intersection with the piece intervals.
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
    free = np.arange(1, n - 1)  # u(0) = u(1) = 0
    return A[np.ix_(free, free)], F[free], free


def eval_bubbles(model: MultiKANBubble1D, profile: PiecewiseDiffusion,
                 xi: np.ndarray) -> np.ndarray:
    """KAN bubble shapes on xi, forced to vanish exactly at both ends."""
    feats = make_profile_features(profile, N_FEATURES, "scaled_combo")
    eps = torch.tensor(feats, dtype=torch.float32)
    xi_t = torch.tensor(xi, dtype=torch.float32)
    zeros = torch.zeros_like(xi_t)
    ef = eps.unsqueeze(0).expand(len(xi), -1)
    with torch.no_grad():
        shapes = torch.stack([
            b(xi_t, zeros, zeros, eps_ratios=ef) for b in model.bubbles
        ]).numpy()
    for b in shapes:  # remove tiny endpoint leakage -> homogeneous BCs
        b -= b[0] * (1.0 - xi) + b[-1] * xi
    return shapes


def assemble_enriched(mesh, profile, source, bubbles, xi_ref, dx):
    """Enriched Galerkin system condensed onto the bubble DOFs.

    Returns the full solution on ``mesh`` plus the bubble coefficients.
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

    # Coupling <phi'_m, eps b'_j>: assemble element-wise with proper local
    # trapezoid weights (shared endpoints halved per element). On element e,
    # phi'_{e+1} = +1/h_e and phi'_e = -1/h_e.
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
    dbs = [deriv(b) for b in bubbles]
    for i in range(nb):
        for j in range(nb):
            A_bb[i, j] = np.sum(w * eps_ref * dbs[i] * dbs[j])
    F_b = np.array([np.sum(w * np.asarray(source(xi_ref)) * b) for b in bubbles])

    # Static condensation: eliminate nodal DOFs, solve 2x2 bubble system.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=int, default=4143,
                        help="test-split sample index (4143 = thin resistive layer)")
    parser.add_argument("--elements", type=int, default=32)
    parser.add_argument("--fd-points", type=int, default=4001)
    parser.add_argument("--check", action="store_true",
                        help="also run the assembler with exact FD bubbles")
    parser.add_argument("--figure", type=Path, default=Path("darcy_combo_solve_demo.png"))
    args = parser.parse_args()

    ds = load_dataset(DATASET_NAME, subdir=SUBDIR)
    test_idx = ds["metadata"]["split_indices"]["test"]
    pool_idx = int(test_idx[args.index])
    edges = np.asarray(ds["metadata"]["piece_edges"][pool_idx])
    values = np.asarray(ds["metadata"]["piece_values"][pool_idx])
    profile = PiecewiseDiffusion(edges, values)
    print(f"test[{args.index}] (pool {pool_idx}): {values.size} pieces, "
          f"eps in [{values.min():.3g}, {values.max():.3g}] "
          f"(contrast {values.max() / values.min():.0f}x)")

    source = lambda x: 1.0 - x
    ref = solve_darcy_1d(profile, length=1.0, source=source,
                         n_points=args.fd_points)
    x_ref, u_ref = ref["x"], ref["u"]

    mesh = np.linspace(0.0, 1.0, args.elements + 1)
    A, F, _ = assemble_p1(mesh, profile, source)
    u_gal_nodes = np.zeros(args.elements + 1)
    u_gal_nodes[1:-1] = np.linalg.solve(A, F)
    u_gal = np.interp(x_ref, mesh, u_gal_nodes)

    model = MultiKANBubble1D(
        n_bubbles=2, n_hidden=32, n_grid=12, n_eps=2 * N_FEATURES,
        eps_transform="none",
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                     weights_only=True))
    model.eval()
    bubbles = eval_bubbles(model, profile, x_ref)
    u_kan_nodes, coeffs = assemble_enriched(mesh, profile, source, bubbles,
                                            x_ref, x_ref[1] - x_ref[0])
    u_kan = eval_enriched(u_kan_nodes, coeffs, bubbles, mesh, x_ref)
    print(f"bubble coefficients (KAN): c_hat={coeffs[0]:+.4f} "
          f"c_tilde={coeffs[1]:+.4f}")

    err_gal = rel_l2(u_gal, u_ref, x_ref)
    err_kan = rel_l2(u_kan, u_ref, x_ref)
    print(f"Galerkin P1 ({args.elements} el): rel L2 = {err_gal:.3e}")
    print(f"RFB-KAN enriched        : rel L2 = {err_kan:.3e}")

    exact_bubbles = None
    if args.check:
        exact_bubbles = np.stack([
            solve_darcy_1d(profile, length=1.0, source=s, n_points=args.fd_points)["u_norm"]
            for s in (1.0, lambda x: x)
        ])
        u_ex_nodes, ex_coeffs = assemble_enriched(mesh, profile, source,
                                                  exact_bubbles.copy(), x_ref,
                                                  x_ref[1] - x_ref[0])
        u_ex = eval_enriched(u_ex_nodes, ex_coeffs, exact_bubbles, mesh, x_ref)
        print(f"[check] exact-bubble enriched : rel L2 = "
              f"{rel_l2(u_ex, u_ref, x_ref):.3e}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   height_ratios=[2.4, 1.0])
    ax0.step(edges, np.concatenate([values, values[-1:]]), where="post",
             color="0.6", lw=1.2, label=r"$\varepsilon(x)$", zorder=1)
    ax0.set_ylabel(r"$\varepsilon$", color="0.4")
    ax0.tick_params(axis="y", labelcolor="0.4")
    ax2 = ax0.twinx()
    ax2.plot(x_ref, u_ref, "k-", lw=1.6, label="reference (fine FD)")
    ax2.plot(mesh, u_gal_nodes, "C0--", lw=1.4, marker="o", ms=2.5,
             label=f"P1 Galerkin ({args.elements} el)")
    ax2.plot(x_ref, u_kan, "C3-.", lw=1.4,
             label=f"RFB-KAN (P1 + 2 KAN bubbles, L2 {err_kan:.3f})")
    if exact_bubbles is not None:
        ax2.plot(x_ref, u_ex, "C2:", lw=2.5, alpha=0.8,
                 label="enriched w/ exact bubbles (check)")
    ax2.set_ylabel("u(x)")
    lines, labels = [], []
    for ax in (ax0, ax2):
        ln, lb = ax.get_legend_handles_labels()
        lines += ln
        labels += lb
    ax2.legend(lines, labels, loc="upper right", fontsize=9)
    ax0.set_title(f"Darcy $-(\\varepsilon u')'=1-x$, "
                  f"test[{args.index}] "
                  f"({values.size} pieces, {values.max() / values.min():.0f}x contrast)")

    ax1.semilogy(x_ref, np.abs(u_gal - u_ref) + 1e-16, "C0-", lw=1.2,
                 label="Galerkin")
    ax1.semilogy(x_ref, np.abs(u_kan - u_ref) + 1e-16, "C3-", lw=1.2,
                 label="RFB-KAN")
    if exact_bubbles is not None:
        ax1.semilogy(x_ref, np.abs(u_ex - u_ref) + 1e-16, "C2--", lw=1.0,
                     label="exact bubbles")
    ax1.set_xlabel("x")
    ax1.set_ylabel("|error|")
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    print(f"figure saved to {args.figure}")


if __name__ == "__main__":
    main()
