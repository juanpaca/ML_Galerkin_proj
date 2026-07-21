"""Mesh convergence study for Classical P1 vs Exact RFB vs KAN-RFB.

Provides two main functions:
  - solve_single_pde(): solve one PDE instance with all three methods
  - convergence_study(): sweep mesh sizes, return results for plotting

Usage in Colab:
    from src.convergence import convergence_study, print_table, plot_convergence
    results = convergence_study(eps=1e-3, beta=1.0, sigma=0.0, kan_model=multi)
    print_table(results)
    plot_convergence(results)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.basis import LagrangeBasis1D
from src.rfb_exact import ExactRFBubbleSet1D
from src.rfb_assembly import (
    assemble_classical_system, assemble_rfb_condensed_system,
    recover_bubble_coefficients, RFBSolution1D,
)
from src.errors import compute_l2_error, compute_h1_error, relative_error_percentage


# ── Fine FD reference solve ────────────────────────────────────────────

def fd_exact(eps, beta, sigma, n=8000):
    """Solve -(εu')' + βu' + σu = 1, u(0)=u(1)=0 via central FD."""
    dx = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    A = np.zeros((n, n))
    rhs = np.ones(n)
    for i in range(1, n - 1):
        A[i, i - 1] = -eps / dx**2 - beta / (2 * dx)
        A[i, i]     =  2 * eps / dx**2 + sigma
        A[i, i + 1] = -eps / dx**2 + beta / (2 * dx)
    A[0, 0] = A[-1, -1] = 1.0
    rhs[0] = rhs[-1] = 0.0
    u = np.linalg.solve(A, rhs)
    return x, u


# ── P1 interpolation wrapper ──────────────────────────────────────────

class _P1Solution:
    """Piecewise-linear interpolation from nodal values."""
    def __init__(self, mesh, u):
        self.mesh = mesh
        self.u = u
    def __call__(self, x):
        basis = LagrangeBasis1D(self.mesh)
        out = np.zeros_like(x)
        for i in range(self.mesh.n_nodes):
            out += self.u[i] * basis.eval(x, i)
        return out


# ── Single-iteration solve ─────────────────────────────────────────────

def solve_single_pde(eps, beta, sigma, n_el, kan_model=None,
                     quad_n=16, n_fd=8000):
    """Solve one PDE instance with Classical P1, Exact RFB, and KAN-RFB.

    Parameters
    ----------
    eps, beta, sigma : float
        PDE coefficients.
    n_el : int
        Number of elements.
    kan_model : MultiKANBubble1D or None
        If None, only Classical and Exact RFB are computed.
    quad_n : int
        Quadrature points per element.
    n_fd : int
        FD grid points for reference solution.

    Returns
    -------
    dict with keys:
        h, pe, rho,
        l2_classical, l2_exact_rfb, l2_kan_rfb (float or np.nan),
        h1_classical, h1_exact_rfb, h1_kan_rfb,
        norm (float)
    """
    h = 1.0 / n_el
    mesh = Mesh1D(0.0, 1.0, n_el)
    quad = GaussLegendre(quad_n)
    pde = AdvectionDiffusion1D(eps, beta, sigma)
    pde.set_source_from_function(lambda x: np.ones_like(x))

    pe = beta * h / (2 * eps)
    rho = sigma * h**2 / eps

    # Reference solution
    x_ref, u_ref = fd_exact(eps, beta, sigma, n=n_fd)
    x_grad_ref = x_ref.copy()
    u_grad_ref = np.gradient(u_ref, x_ref[1] - x_ref[0])
    exact_u = lambda x: np.interp(x, x_ref, u_ref)
    exact_u_grad = lambda x: np.interp(x, x_grad_ref, u_grad_ref)

    # 1. Classical P1
    A_cl, F_cl = assemble_classical_system(mesh, quad, pde)
    u_cl = np.linalg.solve(A_cl, F_cl)
    sol_cl = _P1Solution(mesh, u_cl)
    l2_cl, norm = compute_l2_error(sol_cl, exact_u)
    h1_cl, _ = compute_h1_error(sol_cl, exact_u, exact_u_grad)

    # 2. Exact RFB
    bubble_ex = ExactRFBubbleSet1D(eps, beta, sigma, h,
                                    residual_modes=("constant", "xi"),
                                    n_points=8000)
    A_ex, F_ex, local_ex = assemble_rfb_condensed_system(mesh, quad, pde, bubble_ex)
    u_ex = np.linalg.solve(A_ex, F_ex)
    ub_ex = recover_bubble_coefficients(u_ex, mesh, local_ex)
    sol_ex = RFBSolution1D(u_ex, ub_ex, mesh, bubble_ex, pde)
    l2_ex, _ = compute_l2_error(sol_ex, exact_u)
    h1_ex, _ = compute_h1_error(sol_ex, exact_u, exact_u_grad)

    # 3. KAN-RFB
    l2_kan = h1_kan = np.nan
    if kan_model is not None:
        A_kan, F_kan, local_kan = assemble_rfb_condensed_system(mesh, quad, pde, kan_model)
        u_kan = np.linalg.solve(A_kan, F_kan)
        ub_kan = recover_bubble_coefficients(u_kan, mesh, local_kan)
        sol_kan = RFBSolution1D(u_kan, ub_kan, mesh, kan_model, pde)
        l2_kan, _ = compute_l2_error(sol_kan, exact_u)
        h1_kan, _ = compute_h1_error(sol_kan, exact_u, exact_u_grad)

    return {
        "h": h, "n_el": n_el, "pe": pe, "rho": rho,
        "l2_classical": l2_cl, "l2_exact_rfb": l2_ex, "l2_kan_rfb": l2_kan,
        "h1_classical": h1_cl, "h1_exact_rfb": h1_ex, "h1_kan_rfb": h1_kan,
        "norm": norm,
    }


# ── Convergence sweep ──────────────────────────────────────────────────

def convergence_study(eps, beta, sigma, mesh_sizes=None, kan_model=None,
                     quad_n=16, n_fd=8000):
    """Sweep mesh sizes and collect errors for all three methods.

    Parameters
    ----------
    eps, beta, sigma : float
        PDE coefficients.
    mesh_sizes : list[int] or None
        Element counts to test. Default: [4, 8, 16, 32, 64].
    kan_model : MultiKANBubble1D or None
        Trained KAN model. If None, KAN-RFB column is NaN.
    quad_n : int
        Quadrature points per element.
    n_fd : int
        FD grid points for reference.

    Returns
    -------
    list[dict] — one dict per mesh size (output of solve_single_pde).
    """
    if mesh_sizes is None:
        mesh_sizes = [4, 8, 16, 32, 64]

    results = []
    for n_el in mesh_sizes:
        r = solve_single_pde(eps, beta, sigma, n_el, kan_model, quad_n, n_fd)
        results.append(r)

    return results


# ── Pretty print ───────────────────────────────────────────────────────

def print_table(results, title=None):
    """Print convergence rates table.

    Parameters
    ----------
    results : list[dict]
        Output of convergence_study().
    title : str or None
        Optional title.
    """
    if title:
        print(f"\n{'='*72}")
        print(f"  {title}")
        print(f"{'='*72}")

    has_kan = any(not np.isnan(r["l2_kan_rfb"]) for r in results)
    header = f"  {'h':>8s}  {'N':>4s}  {'Pe':>8s}  {'L2 P1':>10s}  {'L2 RFB':>10s}"
    if has_kan:
        header += f"  {'L2 KAN':>10s}"
    print(header)
    print(f"  {'-'*8}  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*10}", end="")
    if has_kan:
        print(f"  {'-'*10}", end="")
    print()

    prev_l2_cl = prev_l2_ex = prev_l2_kan = None
    for r in results:
        l2_cl = r["l2_classical"]
        l2_ex = r["l2_exact_rfb"]
        l2_kan = r["l2_kan_rfb"]

        rate_cl = f"{np.log10(l2_cl / prev_l2_cl) / np.log10(r['h'] / prev_h):.2f}" if prev_l2_cl is not None else "—"
        rate_ex = f"{np.log10(l2_ex / prev_l2_ex) / np.log10(r['h'] / prev_h):.2f}" if prev_l2_ex is not None else "—"

        line = f"  {r['h']:>8.4f}  {r['n_el']:>4d}  {r['pe']:>8.1f}  {l2_cl:>10.3e}  {l2_ex:>10.3e}"
        if has_kan:
            if not np.isnan(l2_kan):
                rate_kan = f"{np.log10(l2_kan / prev_l2_kan) / np.log10(r['h'] / prev_h):.2f}" if prev_l2_kan is not None else "—"
                line += f"  {l2_kan:>10.3e}"
            else:
                rate_kan = "—"
                line += f"  {'nan':>10s}"
        print(line)

        # Rate sub-line
        rate_line = f"  {'':>8s}  {'':>4s}  {'':>8s}  {'rate '+rate_cl:>10s}  {'rate '+rate_ex:>10s}"
        if has_kan:
            rate_line += f"  {'rate '+rate_kan:>10s}"
        print(rate_line)

        prev_l2_cl, prev_l2_ex, prev_h = l2_cl, l2_ex, r["h"]
        prev_l2_kan = l2_kan if not np.isnan(l2_kan) else prev_l2_kan

    print()


# ── Plot ───────────────────────────────────────────────────────────────

def plot_convergence(results_list, labels=None, save_path=None):
    """Log-log convergence plot.

    Parameters
    ----------
    results_list : list[list[dict]]
        Each element is the output of convergence_study() for one test case.
    labels : list[str] or None
        Label for each test case.
    save_path : str or None
        If given, save plot to this path. Otherwise plt.show().
    """
    if isinstance(results_list[0], dict):
        results_list = [results_list]
    if labels is None:
        labels = [f"Case {i+1}" for i in range(len(results_list))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    styles = ["-", "--", ":"]
    colors = {"classical": "C0", "exact_rfb": "C1", "kan_rfb": "C2"}
    markers = {"classical": "s", "exact_rfb": "o", "kan_rfb": "^"}

    for case_idx, (res, label) in enumerate(zip(results_list, labels)):
        hs = np.array([r["h"] for r in res])
        ls = styles[case_idx % 3]

        for method, name in [("classical", "Classical P1"),
                              ("exact_rfb", "Exact RFB"),
                              ("kan_rfb", "KAN-RFB")]:
            for err_key, ax_idx in [("l2", 0), ("h1", 1)]:
                vals = np.array([r[f"{err_key}_{method}"] for r in res])
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                lbl = f"{name} — {label}" if method == "classical" else name
                axes[ax_idx].loglog(hs[valid], vals[valid],
                                    marker=markers[method], color=colors[method],
                                    linestyle=ls, label=lbl, linewidth=2, markersize=6)

    titles = ["L2 error", "H1 error"]
    for ax, title in zip(axes, titles):
        ax.set_xlabel("h (element size)")
        ax.set_ylabel("||u_h - u||")
        ax.set_title(f"Convergence — {title}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
