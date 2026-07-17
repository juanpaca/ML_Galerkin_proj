#!/usr/bin/env python3
"""Test the full RFB assembly pipeline with an untrained KAN bubble model.

Verifies that the infrastructure works end-to-end before training.
Compares: classical Galerkin, KAN-RFB (untrained), and exact RFB.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.basis import LagrangeBasis1D
from src.rfb_bubble import MultiKANBubble1D
from src.rfb_exact import ExactRFBubbleSet1D
from src.rfb_assembly import (
    assemble_classical_system, assemble_rfb_condensed_system,
    recover_bubble_coefficients, RFBSolution1D,
)
from src.errors import compute_l2_error, relative_error_percentage

# --- Setup ---
EPS = 1e-3
BETA = 1.0
SIGMA = 0.0
N_EL = 16
QUAD_N = 16
N_FD_REF = 8000

mesh = Mesh1D(0.0, 1.0, N_EL)
quad = GaussLegendre(QUAD_N)
pde = AdvectionDiffusion1D(EPS, BETA, SIGMA)
pde.set_source_from_function(lambda x: np.ones_like(x))

# "Ground truth" — fine FD solve
def u_exact_fd(eps, beta, sigma, n=2000):
    dx = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    A = np.zeros((n, n))
    rhs = np.ones(n)
    for i in range(1, n - 1):
        A[i, i - 1] = -eps / dx**2 - beta / (2 * dx)
        A[i, i] = 2 * eps / dx**2 + sigma
        A[i, i + 1] = -eps / dx**2 + beta / (2 * dx)
    A[0, 0] = A[-1, -1] = 1.0
    rhs[0] = rhs[-1] = 0.0
    u = np.linalg.solve(A, rhs)
    return x, u

x_ref, u_ref = u_exact_fd(EPS, BETA, SIGMA, n=N_FD_REF)

def exact_u(x):
    return np.interp(x, x_ref, u_ref)

# --- 1. Classical Galerkin ---
A_cl, F_cl = assemble_classical_system(mesh, quad, pde)
u_cl = np.linalg.solve(A_cl, F_cl)

class ClassicalSolution:
    def __init__(self, mesh, u):
        self.mesh = mesh
        self.u = u
    def __call__(self, x):
        basis = LagrangeBasis1D(self.mesh)
        out = np.zeros_like(x)
        for i in range(self.mesh.n_nodes):
            out += self.u[i] * basis.eval(x, i)
        return out

l2_cl, norm = compute_l2_error(ClassicalSolution(mesh, u_cl), exact_u)
print(f"Classical P1:  L2 = {l2_cl:.4e}  ({relative_error_percentage(l2_cl, norm):.1f}%)")

# --- 2. Exact RFB (ground truth enrichment) ---
bubble_exact = ExactRFBubbleSet1D(EPS, BETA, SIGMA, mesh.h, residual_modes=("constant", "xi"), n_points=8000)
A_ex, F_ex, local_ex = assemble_rfb_condensed_system(mesh, quad, pde, bubble_exact)
u_ex = np.linalg.solve(A_ex, F_ex)
ub_ex = recover_bubble_coefficients(u_ex, mesh, local_ex)
sol_ex = RFBSolution1D(u_ex, ub_ex, mesh, bubble_exact, pde)
l2_ex, _ = compute_l2_error(sol_ex, exact_u)
print(f"Exact RFB:    L2 = {l2_ex:.4e}  ({relative_error_percentage(l2_ex, norm):.1f}%)  [{-100*(1-l2_ex/l2_cl):+.1f}% vs P1]")

# --- 3. KAN-RFB (untrained) ---
torch.manual_seed(42)
bubble_kan = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
n_params = sum(p.numel() for p in bubble_kan.parameters())
print(f"\nKAN-RFB:  {n_params} params (untrained, seed=42)")

# Evaluate bubble shape for this test case
pe_val = BETA * mesh.h / (2 * EPS)  # ≈ 31.25
rho_val = SIGMA * mesh.h**2 / EPS   # = 0
xi_test = np.linspace(0, 1, 101)
b, db = bubble_kan.value_grad_numpy(xi_test, pe_val, rho_val)
print(f"  pe={pe_val:.2f}, rho={rho_val:.0f}")
print(f"  b̂(0.5) = {b[0, 50]:.4f}  (exact: ~1)")
print(f"  b̃(0.5) = {b[1, 50]:.4f}  (exact: ~0.5)")

A_kan, F_kan, local_kan = assemble_rfb_condensed_system(mesh, quad, pde, bubble_kan)
u_kan = np.linalg.solve(A_kan, F_kan)
ub_kan = recover_bubble_coefficients(u_kan, mesh, local_kan)
sol_kan = RFBSolution1D(u_kan, ub_kan, mesh, bubble_kan, pde)
l2_kan, _ = compute_l2_error(sol_kan, exact_u)
ratio_vs_p1 = l2_kan / l2_cl
ratio_vs_ex = l2_kan / l2_ex
print(f"KAN-RFB:      L2 = {l2_kan:.4e}  ({relative_error_percentage(l2_kan, norm):.1f}%)")
print(f"  vs Classical: x{ratio_vs_p1:.2f}  vs Exact RFB: x{ratio_vs_ex:.2f}")

# --- Summary ---
print("\n" + "=" * 50)
print(f"  {'Method':<20} {'L2 error':<12} {'Rel %':<8}")
print(f"  {'-'*20} {'-'*12} {'-'*8}")
for name, err in [("Classical P1", l2_cl), ("Exact RFB", l2_ex), ("KAN-RFB (untrained)", l2_kan)]:
    print(f"  {name:<20} {err:<12.4e} {relative_error_percentage(err, norm):<8.1f}")
print(f"\n✅ Assembly pipeline OK with untrained KAN")
