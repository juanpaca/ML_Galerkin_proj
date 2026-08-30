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

from src.darcy_assembly import (
    assemble_enriched,
    assemble_p1,
    element_load,
    element_stiffness,
    eval_enriched,
    rel_l2,
)
from src.dataset_generation import load_dataset
from src.darcy_variable import PiecewiseDiffusion, make_profile_features, solve_darcy_1d
from src.rfb_bubble import MultiKANBubble1D

DATASET_NAME = "darcy_piecewise_combo20k"
SUBDIR = "data_darcy_variable"
MODEL_PATH = Path(f"models/{DATASET_NAME}_kan.pt")
N_FEATURES = 24


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
