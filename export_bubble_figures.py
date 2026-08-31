#!/usr/bin/env python3
"""Export bubble + enriched-solution figures from the 5-piece dataset.

Figure 1: diffusion profile epsilon(x) and the two residual-free bubbles
b_hat = L^-1(1), b_tilde = L^-1(x) computed with that (discontinuous)
coefficient.
Figure 2: solution of -(epsilon u')' = 1 + x by the Galerkin+bubble method
(statically condensed exact FD bubbles on a uniform P1 mesh) vs the
reference finite-difference solution, with the relative H1 error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.darcy_assembly import assemble_enriched, eval_enriched
from src.darcy_variable import PiecewiseDiffusion, solve_darcy_1d

ROOT = Path(__file__).resolve().parent
DS = ROOT / "datasets" / "data_darcy_variable"
OUT = ROOT / "figures"


def load_sample(split: str, which: int, target_contrast: float | None = None):
    """Load a (profile, b_hat, b_tilde) triple from the dataset.

    ``eps_profile`` (piecewise-constant e samples on the grid) reconstructs
    the diffusion profile exactly (pieces are >=100 cells wide), so we do
    not depend on the pool-index bookkeeping across the gate-filter. If
    ``target_contrast`` is given, picks the sample whose realized contrast
    eps_max/eps_min is closest to it.
    """
    mode0 = np.load(DS / f"darcy_piecewise_5pc_{split}_constant.npz")
    xi = mode0["xi"]
    n = len(mode0["pe"])
    eps_all = np.asarray(mode0["eps_profile"], dtype=float)
    contrasts = np.array([
        eps_all[i].max() / eps_all[i].min() for i in range(n)
    ])

    def build_profile(k):
        eps = eps_all[k]
        jumps = np.flatnonzero(np.diff(eps) != 0) + 1
        edges = np.concatenate(([0.0], xi[jumps], [1.0]))
        values = np.concatenate(([eps[0]], eps[jumps]))
        return PiecewiseDiffusion(edges, values)

    if target_contrast is not None:
        k = int(np.argmin(np.abs(contrasts - target_contrast)))
    else:
        k = which
    profile = build_profile(int(k))
    b_hat = mode0["b"][k]
    b_tilde = np.load(DS / f"darcy_piecewise_5pc_{split}_xi.npz")["b"][k]
    return profile, xi, b_hat, b_tilde, int(k)


def rel_h1(u_h, du_h, u_ref, du_ref, x):
    dx = x[1] - x[0]
    err_sq = np.sum((u_h - u_ref) ** 2 + (du_h - du_ref) ** 2) * dx
    norm_sq = np.sum(u_ref ** 2 + du_ref ** 2) * dx
    return float(np.sqrt(err_sq) / np.sqrt(norm_sq))


def enrichment_for_source(profile, xi, bubbles, source, n_el=8):
    """Deployed assembly: uniform n_el P1 mesh + one global bubble pair."""
    mesh = np.linspace(0.0, 1.0, n_el + 1)
    dx = xi[1] - xi[0]
    u_nodes, coeffs = assemble_enriched(mesh, profile, source, bubbles, xi, dx)
    return eval_enriched(u_nodes, coeffs, bubbles, mesh, xi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--target-contrast", type=float, default=30.0,
                    help="pick a representative profile with this contrast")
    ap.add_argument("--n-ref", type=int, default=60001)
    ap.add_argument("--n-el", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    profile, xi, b_hat, b_tilde, idx = load_sample(args.split, 0, args.target_contrast)
    bubbles = np.stack([b_hat, b_tilde])

    # ---- Figure 1: epsilon + bubbles ----
    fig1, (ax_eps, ax_b) = plt.subplots(
        2, 1, figsize=(7.0, 6.2), sharex=True, height_ratios=(1, 1.4))
    xd = np.linspace(0.0, 1.0, 4001)
    ax_eps.step(xd, profile.evaluate(xd), where="post", color="C1",
                linewidth=2.0)
    for e in profile.edges[1:-1]:
        ax_eps.axvline(e, color="C1", alpha=0.25, linewidth=0.8)
    ax_eps.set_ylabel(r"$\varepsilon(x)$", fontsize=13)
    ax_eps.set_yscale("log")
    ax_eps.set_ylim(0.8 * profile.values.min(), 1.25 * profile.values.max())
    ax_eps.set_title(
        "Residual-free bubbles for a discontinuous profile "
        f"(contrast {profile.values.max() / profile.values.min():.1f}x)", fontsize=12)

    ax_b.axhline(0.0, color="k", linewidth=0.6)
    ax_b.plot(xi, b_hat, color="C0", linewidth=2.0,
              label=r"$\hat b = L^{-1}(1)$  (residual mode $1$)")
    ax_b.plot(xi, b_tilde, color="C3", linewidth=2.0,
              label=r"$\tilde b = L^{-1}(x)$  (residual mode $x$)")
    for e in profile.edges[1:-1]:
        ax_b.axvline(e, color="C1", alpha=0.25, linewidth=0.8)
    ax_b.set_xlabel(r"$x$", fontsize=13)
    ax_b.set_ylabel(r"$b(x)$", fontsize=13)
    ax_b.legend(loc="upper center", fontsize=11, frameon=False)
    ax_b.set_xlim(0.0, 1.0)
    fig1.tight_layout()
    f1 = OUT / "bubbles_5pc.png"
    fig1.savefig(f1, dpi=200)
    print(f"saved {f1}")

    # ---- Figure 2: enriched vs reference solution, rel H1 error ----
    src = lambda x: 1.0 + np.asarray(x, dtype=float)
    ref = solve_darcy_1d(profile, length=1.0, source=src, n_points=args.n_ref)
    x_ref = ref["x"]
    u_ref = ref["u"]

    u_en = enrichment_for_source(profile, xi, bubbles, src, n_el=args.n_el)
    u_en_ref = np.interp(x_ref, xi, u_en)
    dx_ref = x_ref[1] - x_ref[0]
    du_ref = np.gradient(u_ref, dx_ref)
    du_en = np.gradient(u_en_ref, dx_ref)
    e_h1 = rel_h1(u_en_ref, du_en, u_ref, du_ref, x_ref)
    e_l2 = float(np.linalg.norm(u_en_ref - u_ref) / np.linalg.norm(u_ref))

    fig2, (ax_sol, ax_err) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": (1.15, 1.0)})
    ax_sol.plot(x_ref, u_ref, color="k", linewidth=2.0, label="reference (FD)")
    ax_sol.plot(x_ref, u_en_ref, color="C2", linewidth=1.6,
                linestyle="--", label="Galerkin + bubbles")
    for e in profile.edges[1:-1]:
        ax_sol.axvline(e, color="C1", alpha=0.3, linewidth=0.8)
    ax_sol.set_xlabel(r"$x$", fontsize=13)
    ax_sol.set_ylabel(r"$u(x)$", fontsize=13)
    ax_sol.set_title(r"$-\left(\varepsilon u'\right)' = 1 + x$", fontsize=12)
    ax_sol.legend(loc="best", fontsize=11, frameon=False)
    ax_sol.text(
        0.03, 0.97,
        f"$\\|u - u_{{hb}}\\|_{{L_2}} / \\|u\\|_{{L_2}}$ = {e_l2:.2e}\n"
        f"$\\|u - u_{{hb}}\\|_{{H_1}} / \\|u\\|_{{H_1}}$ = {e_h1:.2e}",
        transform=ax_sol.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax_err.semilogy(x_ref, np.abs(u_ref - u_en_ref), color="C2",
                    linewidth=1.8, label=r"$|u - u_{hb}|$")
    ax_err.semilogy(x_ref, np.abs(du_ref - du_en), color="C6",
                    linewidth=1.8, linestyle="--", label=r"$|u' - u_{hb}'|$")
    for e in profile.edges[1:-1]:
        ax_err.axvline(e, color="C1", alpha=0.3, linewidth=0.8)
    ax_err.set_xlabel(r"$x$", fontsize=13)
    ax_err.set_ylabel("pointwise error", fontsize=13)
    ax_err.set_title(f"relative $H^1$ error = {e_h1:.2e}", fontsize=12)
    ax_err.legend(loc="best", fontsize=11, frameon=False)
    for a in (ax_sol, ax_err):
        a.set_xlim(0.0, 1.0)
    fig2.tight_layout()
    f2 = OUT / "solution_f1px_5pc.png"
    fig2.savefig(f2, dpi=200)
    print(f"saved {f2}")
    print(f"profile idx={idx}  eps in "
          f"[{profile.values.min():.3f}, {profile.values.max():.3f}]  "
          f"n_el={args.n_el}  rel L2={e_l2:.3e}  rel H1={e_h1:.3e}")


if __name__ == "__main__":
    main()