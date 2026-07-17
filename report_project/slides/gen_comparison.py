import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.rfb_exact import ExactRFBubble1D
from src.rfb_assembly import (
    assemble_classical_system,
    assemble_rfb_condensed_system,
    recover_bubble_coefficients,
    RFBSolution1D,
)
from src.manufactured_solutions import advection_diffusion_layer_solution

# --- Parameters ---
eps = 1e-3
beta = 1.0
sigma = 0.0
mesh = Mesh1D(0.0, 1.0, 16)
quad = GaussLegendre(16)

pde = AdvectionDiffusion1D(eps=eps, beta=beta, sigma=sigma)
pde.set_source_from_function(lambda x: np.ones_like(np.asarray(x, dtype=float)))
u_exact_fn = lambda x: advection_diffusion_layer_solution(x, eps=eps, a=beta, sigma=sigma)

# --- Classical Galerkin ---
A_c, f_c = assemble_classical_system(mesh, quad, pde)
coeffs_c = np.linalg.solve(A_c, f_c)
sol_c = RFBSolution1D(coeffs_c, None, mesh)

# --- Exact RFB Galerkin (single constant mode — sufficient for σ=0, constant f) ---
exact_bubble = ExactRFBubble1D(
    eps=eps, beta=beta, sigma=sigma, h=mesh.h,
    residual_mode="constant", n_points=8000,
)
A_r, f_r, local_data = assemble_rfb_condensed_system(mesh, quad, pde, exact_bubble)
coeffs_r = np.linalg.solve(A_r, f_r)
bubble_coeffs = recover_bubble_coefficients(coeffs_r, mesh, local_data)
sol_r = RFBSolution1D(coeffs_r, bubble_coeffs, mesh, exact_bubble, pde)

# --- Plot ---
x = np.linspace(0, 1, 4000)
u_exact = u_exact_fn(x)
u_cl = sol_c(x)
u_rfb = sol_r(x)

fig, ax = plt.subplots(figsize=(7, 3.5))

ax.plot(x, u_exact, color="#2C3E50", linewidth=1.8, linestyle="--", label="Exact solution")
ax.plot(x, u_cl, color="#E74C3C", linewidth=1.0, label="Classical P1 Galerkin")
ax.plot(x, u_rfb, color="#1F4E79", linewidth=1.5, linestyle="-.", label="Galerkin + RFB")

ax.set_xlabel(r"$x$", fontsize=11)
ax.set_ylabel(r"$u(x)$", fontsize=11)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9)

l2_c = np.sqrt(np.sum((u_cl - u_exact)**2) * (x[1]-x[0])) / np.sqrt(np.sum(u_exact**2) * (x[1]-x[0]))
l2_r = np.sqrt(np.sum((u_rfb - u_exact)**2) * (x[1]-x[0])) / np.sqrt(np.sum(u_exact**2) * (x[1]-x[0]))
ax.text(0.65, 0.95,
    f"Classical: rel. L² error {l2_c*100:.1f}%\nRFB:       rel. L² error {l2_r*100:.4f}%",
    transform=ax.transAxes, fontsize=9, ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#D4E6F1", edgecolor="#1F4E79", alpha=0.8))

fig.tight_layout()

figure_dir = os.path.join(os.path.dirname(__file__), "figures")
fig.savefig(f"{figure_dir}/comparison_galerkin_rfb.pdf", bbox_inches="tight")
fig.savefig(f"{figure_dir}/comparison_galerkin_rfb.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"L2 rel. error: classical={l2_c*100:.2f}%, RFB={l2_r*100:.4f}%")
