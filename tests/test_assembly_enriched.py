"""Static condensation with residual-free bubbles vs classical P1.

Uses the exact constant-coefficient RFB (ExactRFBubbleSet1D) so the
enrichment recovers the analytic advection-diffusion-layer solution almost
exactly while classical P1 still needs refinement.  Also covers bubble
coefficient recovery, RFBSolution1D evaluation, and a KAN bubble in the
assembly pipeline.
"""

import numpy as np
import pytest

from src.errors import compute_energy_error, compute_h1_error, compute_l2_error
from src.manufactured_solutions import (
    advection_diffusion_layer_solution,
    advection_diffusion_layer_solution_grad,
)
from src.mesh import Mesh1D
from src.pde import AdvectionDiffusion1D
from src.quadrature import GaussLegendre
from src.rfb_assembly import (
    assemble_classical_system,
    assemble_rfb_condensed_system,
    local_enriched_matrices,
    recover_bubble_coefficients,
    RFBSolution1D,
)
from src.rfb_bubble import KANBubble1D
from src.rfb_exact import ExactRFBubbleSet1D


EPS, BETA, SIGMA = 0.2, 1.0, 0.0
QUAD = GaussLegendre(12)
N_BUBBLE = 4000


@pytest.fixture()
def problem():
    pde = AdvectionDiffusion1D(eps=EPS, beta=BETA, sigma=SIGMA)
    pde.set_source_from_function(lambda x: np.ones_like(np.asarray(x, dtype=float)))
    return pde


@pytest.fixture()
def exact_u():
    return (lambda x: advection_diffusion_layer_solution(
                np.asarray(x, dtype=float), eps=EPS, a=BETA, sigma=SIGMA),
            lambda x: advection_diffusion_layer_solution_grad(
                np.asarray(x, dtype=float), eps=EPS, a=BETA, sigma=SIGMA))


def _classical(pde, n_el):
    mesh = Mesh1D(0.0, 1.0, n_el)
    A, f = assemble_classical_system(mesh, QUAD, pde)
    nodal = np.linalg.solve(A, f)
    return RFBSolution1D(nodal, None, mesh)


def _enriched(pde, n_el):
    mesh = Mesh1D(0.0, 1.0, n_el)
    bubble = ExactRFBubbleSet1D(eps=EPS, beta=BETA, sigma=SIGMA, h=mesh.h,
                                residual_modes=("constant",), n_points=N_BUBBLE)
    A, f, local_data = assemble_rfb_condensed_system(mesh, QUAD, pde, bubble)
    nodal = np.linalg.solve(A, f)
    ub = recover_bubble_coefficients(nodal, mesh, local_data)
    return RFBSolution1D(nodal, ub, mesh, bubble, pde), nodal, ub, local_data, bubble


def _rel_l2_h1(sol, exact_u, exact_ug):
    e_l2, n_l2 = compute_l2_error(sol, exact_u, n_points=4000)
    e_h1, n_h1 = compute_h1_error(sol, exact_u, exact_ug, n_points=4000)
    return e_l2 / n_l2, e_h1 / n_h1


# --------------------------------------------------------------------------
# local enriched matrices (shape invariants)
# --------------------------------------------------------------------------

def test_local_enriched_matrices_shapes(problem):
    bubble = ExactRFBubbleSet1D(eps=EPS, beta=BETA, sigma=SIGMA, h=0.25,
                                residual_modes=("constant",), n_points=N_BUBBLE)
    A, F = local_enriched_matrices(0.0, 0.25, QUAD, problem, bubble)
    assert A.shape == (3, 3) and F.shape == (3,)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(F))


# --------------------------------------------------------------------------
# enriched vs classical convergence
# --------------------------------------------------------------------------

def test_enriched_converges_rapidly_in_mesh(problem, exact_u):
    u_exact, u_grad = exact_u
    errs = []
    for n_el in (2, 4, 8, 16):
        sol, *_ = _enriched(problem, n_el)
        rel_l2, _ = _rel_l2_h1(sol, u_exact, u_grad)
        errs.append(rel_l2)
        assert rel_l2 < 1e-3
    # With exact RFB data the enriched error decays fast as the mesh refines
    # (the bubbles capture the boundary layer; only a small nodal residual
    # remains, which shrinks with h).
    assert errs[-1] < errs[0] / 100.0


def test_enriched_beats_classical_and_classical_converges(problem, exact_u):
    u_exact, u_grad = exact_u
    classical = []
    for n_el in (16, 32, 64, 128):
        sol = _classical(problem, n_el)
        rel_l2, rel_h1 = _rel_l2_h1(sol, u_exact, u_grad)
        classical.append((rel_l2, rel_h1))

    for (l2, h1), (l2_prev, _) in zip(classical[1:], classical):
        assert l2 < l2_prev * 0.95           # classical keeps improving
        assert l2 > 1e-6                      # still far from exact

    s8, *_ = _enriched(problem, 8)
    e_l2, e_h1 = _rel_l2_h1(s8, u_exact, u_grad)
    assert e_l2 < classical[-1][0] / 10       # enriched wins by an order of magnitude
    assert e_h1 < classical[-1][1] / 5


def test_enriched_energy_error_improves_classical(problem, exact_u):
    u_exact, u_grad = exact_u
    n_el = 32
    mesh = Mesh1D(0.0, 1.0, n_el)
    bubble = ExactRFBubbleSet1D(eps=EPS, beta=BETA, sigma=SIGMA, h=mesh.h,
                                residual_modes=("constant",), n_points=N_BUBBLE)

    A, f = assemble_classical_system(mesh, QUAD, problem)
    c_sol = RFBSolution1D(np.linalg.solve(A, f), None, mesh)
    A2, f2, ld = assemble_rfb_condensed_system(mesh, QUAD, problem, bubble)
    e_sol = RFBSolution1D(np.linalg.solve(A2, f2),
                          recover_bubble_coefficients(np.linalg.solve(A2, f2),
                                                      mesh, ld),
                          mesh, bubble, problem)

    e_c = compute_energy_error(c_sol, u_exact, u_grad, EPS, BETA, n_points=4000)
    e_e = compute_energy_error(e_sol, u_exact, u_grad, EPS, BETA, n_points=4000)
    assert e_e <= 1.05 * e_c
    assert e_e > 0.0


def test_reaction_case_enriched_never_worse(problem, exact_u):
    pde_reac = AdvectionDiffusion1D(eps=0.1, beta=1.0, sigma=1.0)
    pde_reac.set_source_from_function(lambda x: np.ones_like(np.asarray(x, dtype=float)))
    u_exact = lambda y: advection_diffusion_layer_solution(
        np.asarray(y, dtype=float), eps=0.1, a=1.0, sigma=1.0)
    u_grad = lambda y: advection_diffusion_layer_solution_grad(
        np.asarray(y, dtype=float), eps=0.1, a=1.0, sigma=1.0)

    for n_el in (8, 16, 32):
        mesh = Mesh1D(0.0, 1.0, n_el)
        bubble = ExactRFBubbleSet1D(eps=0.1, beta=1.0, sigma=1.0, h=mesh.h,
                                    residual_modes=("constant",), n_points=N_BUBBLE)
        A, f = assemble_classical_system(mesh, QUAD, pde_reac)
        c_sol = RFBSolution1D(np.linalg.solve(A, f), None, mesh)
        A2, f2, ld = assemble_rfb_condensed_system(mesh, QUAD, pde_reac, bubble)
        e_sol = RFBSolution1D(np.linalg.solve(A2, f2),
                              recover_bubble_coefficients(np.linalg.solve(A2, f2),
                                                          mesh, ld),
                              mesh, bubble, pde_reac)
        e_c, n_c = compute_l2_error(c_sol, u_exact, n_points=4000)
        e_e, n_e = compute_l2_error(e_sol, u_exact, n_points=4000)
        assert e_e / n_e <= 1.05 * (e_c / n_c)
        assert e_e / n_e < 0.1


# --------------------------------------------------------------------------
# coefficient recovery / evaluation consistency
# --------------------------------------------------------------------------

def test_bubble_coefficient_recovery_consistent(problem, exact_u):
    u_exact, u_grad = exact_u
    sol, nodal, ub, local_data, bubble = _enriched(problem, 8)

    # Recovered coefficients reproduce the assembled condensed solution.
    xq = np.linspace(1e-6, 1.0 - 1e-6, 500)
    sol2 = RFBSolution1D(nodal, ub, sol.mesh, bubble, problem)
    assert np.allclose(sol2(xq), sol(xq), atol=1e-10)

    rel_l2, rel_h1 = _rel_l2_h1(sol2, u_exact, u_grad)
    assert rel_l2 < 1e-3
    assert rel_h1 < 1e-2

    # Without the bubble the enriched object degenerates to the P1 part.
    sol_lin = RFBSolution1D(nodal, None, sol.mesh)
    np_target = np.interp(xq, sol.mesh.nodes, nodal)
    assert np.allclose(sol_lin(xq), np_target, atol=1e-12)


# --------------------------------------------------------------------------
# KAN bubble through the assembly pipeline (untrained)
# --------------------------------------------------------------------------

def test_untrained_kan_bubble_pipeline_runs():
    pde = AdvectionDiffusion1D(eps=0.1, beta=0.5, sigma=0.0)
    pde.set_source_from_function(lambda x: np.ones_like(np.asarray(x, dtype=float)))
    mesh = Mesh1D(0.0, 1.0, 4)
    bubble = KANBubble1D(n_hidden=3, n_grid=6, spline_order=3)
    A, f, local_data = assemble_rfb_condensed_system(mesh, GaussLegendre(8),
                                                     pde, bubble)
    nodal = np.linalg.solve(A, f)
    assert np.all(np.isfinite(nodal))
    ub = recover_bubble_coefficients(nodal, mesh, local_data)
    assert ub.shape == (mesh.n_elements,)
    sol = RFBSolution1D(nodal, ub, mesh, bubble, pde)
    xs = np.linspace(0, 1, 257)
    vals = sol(xs)
    assert np.all(np.isfinite(vals))
    # Dirichlet homogeneous BCs respected at the endpoints.
    assert abs(vals[0]) < 1e-12 and abs(vals[-1]) < 1e-12