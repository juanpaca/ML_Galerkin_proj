"""P1 FEM infrastructure: Mesh1D, Lagrange basis, Gauss-Legendre quadrature,
the advection-diffusion PDE coefficient model, error estimators, and Galerkin
assembly **without** bubbles (classical P1), including numerical convergence
to a manufactured solution.
"""

import numpy as np
import pytest

from src.basis import LagrangeBasis1D
from src.errors import compute_h1_error, compute_l2_error, relative_error_percentage
from src.mesh import Mesh1D
from src.pde import AdvectionDiffusion1D
from src.quadrature import GaussLegendre
from src.rfb_assembly import assemble_classical_system, RFBSolution1D


# --------------------------------------------------------------------------
# Mesh1D
# --------------------------------------------------------------------------

def test_mesh_basics():
    mesh = Mesh1D(0.0, 1.0, 4)
    assert mesh.n_nodes == 5
    assert mesh.n_elements == 4
    assert np.allclose(mesh.nodes, np.linspace(0, 1, 5))
    assert np.allclose(mesh.h, 0.25)
    assert mesh.element_vertices(0) == (0.0, 0.25)
    assert mesh.element_vertices(3) == (0.75, 1.0)
    assert mesh.element_dofs(2) == [2, 3]


def test_mesh_validation():
    with pytest.raises(ValueError):
        Mesh1D(0.0, 1.0, 0)
    with pytest.raises(ValueError):
        Mesh1D(0.0, 1.0, -3)
    mesh = Mesh1D(0.0, 1.0, 2)
    with pytest.raises(NotImplementedError):
        mesh.element_dofs(0, degree=2)


# --------------------------------------------------------------------------
# LagrangeBasis1D
# --------------------------------------------------------------------------

def test_lagrange_node_interpolation():
    mesh = Mesh1D(0.0, 1.0, 6)
    basis = LagrangeBasis1D(mesh)
    assert basis.num_dofs() == mesh.n_nodes
    x = mesh.nodes
    for i in range(mesh.n_nodes):
        assert np.allclose(basis.eval(x, i),
                           np.eye(mesh.n_nodes)[i], atol=1e-14)


def test_lagrange_partition_of_unity():
    mesh = Mesh1D(0.0, 1.0, 6)
    basis = LagrangeBasis1D(mesh)
    xs = np.linspace(mesh.nodes[2], mesh.nodes[3], 17)
    total = sum(basis.eval(xs, i) for i in range(mesh.n_nodes))
    assert np.allclose(total, 1.0, atol=1e-12)


def test_lagrange_grad_piecewise():
    mesh = Mesh1D(0.0, 1.0, 4)
    basis = LagrangeBasis1D(mesh)
    xa, xb = mesh.element_vertices(1)
    xm = 0.5 * (xa + xb)
    # Inside element [x1, x2]: left node slopes down, right node up.
    assert np.allclose(basis.grad(np.array([xm]), 1), -1.0 / mesh.h)
    assert np.allclose(basis.grad(np.array([xm]), 2), 1.0 / mesh.h)
    # Outside support -> zero.
    assert np.allclose(basis.grad(np.array([0.1]), 3), 0.0)
    assert np.allclose(basis.eval(np.array([0.1]), 3), 0.0)


# --------------------------------------------------------------------------
# GaussLegendre
# --------------------------------------------------------------------------

def test_gauss_legendre_mapping():
    quad = GaussLegendre(5)
    for a, b in [(0.0, 1.0), (-1.0, 1.0), (0.3, 2.7)]:
        x, w = quad.map_to_interval(a, b)
        assert abs(w.sum() - (b - a)) < 1e-14
        assert np.all(x >= a) and np.all(x <= b)
    # 5-point Gauss integrates degree-9 polynomials exactly.
    xq, wq = quad.map_to_interval(0.0, 1.0)
    for p in range(10):
        ref = 1.0 / (p + 1)
        assert abs(np.sum(wq * xq ** p) - ref) < 1e-12


# --------------------------------------------------------------------------
# AdvectionDiffusion1D
# --------------------------------------------------------------------------

def test_pde_constants_and_function_precedence():
    pde = AdvectionDiffusion1D(eps=2.0, beta=3.0, sigma=1.0)
    x = np.linspace(0, 1, 9)
    assert np.allclose(pde.diffusion(x), 2.0)
    assert np.allclose(pde.advection(x), 3.0)
    assert np.allclose(pde.reaction(x), 1.0)
    assert np.allclose(pde.source(x), 0.0)

    pde.set_diffusion_from_function(lambda t: t + 1.0)
    pde.set_advection_from_function(lambda t: 2 * t)
    pde.set_source_from_function(lambda t: t ** 2)
    assert np.allclose(pde.diffusion(x), x + 1.0)
    assert np.allclose(pde.advection(x), 2 * x)
    assert np.allclose(pde.source(x), x ** 2)
    assert np.allclose(pde.reaction(x), 1.0)   # constant remains
    # Functions take precedence over the constant.
    assert not np.allclose(pde.diffusion(x), 2.0)

    # Scalar evaluation paths.
    assert pde.diffusion(0.5) == 1.5


# --------------------------------------------------------------------------
# Error helpers
# --------------------------------------------------------------------------

def test_relative_error_percentage_edge_cases():
    assert relative_error_percentage(1e-3, 1.0) == pytest.approx(0.1)
    assert relative_error_percentage(0.5, 0.0) == 0.0
    assert relative_error_percentage(1.0, 1e-18) == 0.0


# --------------------------------------------------------------------------
# Classical P1 Galerkin assembly
# --------------------------------------------------------------------------

def test_classical_assembly_single_element():
    mesh = Mesh1D(0.0, 1.0, 1)
    quad = GaussLegendre(5)
    pde = AdvectionDiffusion1D(eps=1.0, beta=0.0, sigma=0.0)
    pde.set_source_from_function(lambda x: np.ones_like(x))
    A, f = assemble_classical_system(mesh, quad, pde, apply_bc=False)
    assert np.allclose(A, [[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(f, [0.5, 0.5])
    assert np.allclose(A, A.T)               # symmetric (beta = 0)


def test_classical_assembly_dirichlet_rows():
    mesh = Mesh1D(0.0, 1.0, 3)
    quad = GaussLegendre(5)
    pde = AdvectionDiffusion1D(eps=1.0, beta=0.0, sigma=0.0)
    pde.set_source_from_function(lambda x: np.ones_like(x))
    A, f = assemble_classical_system(mesh, quad, pde, apply_bc=True)
    assert np.allclose(A[0], np.eye(A.shape[0])[0])
    assert np.allclose(A[:, 0], np.eye(A.shape[0])[:, 0])
    assert f[0] == 0.0
    assert f[-1] == 0.0


def test_classical_p1_convergence_rates():
    """-eps u'' = 2 eps on [0,1], u = x(1-x).  P1 gives L2 ~ h^2, H1 ~ h."""
    eps = 1.0
    u_exact = lambda x: np.asarray(x, dtype=float) * (1.0 - np.asarray(x, dtype=float))
    u_exact_grad = lambda x: 1.0 - 2.0 * np.asarray(x, dtype=float)

    pde = AdvectionDiffusion1D(eps=eps, beta=0.0, sigma=0.0)
    pde.set_source_from_function(lambda x: np.full_like(np.asarray(x, dtype=float), 2.0 * eps))

    l2_hist, h1_hist = [], []
    for n_el in (8, 16, 32, 64):
        mesh = Mesh1D(0.0, 1.0, n_el)
        quad = GaussLegendre(5)
        A, f = assemble_classical_system(mesh, quad, pde)
        nodal = np.linalg.solve(A, f)
        sol = RFBSolution1D(nodal, None, mesh)
        e_l2, n_l2 = compute_l2_error(sol, u_exact, n_points=4000)
        e_h1, n_h1 = compute_h1_error(sol, u_exact, u_exact_grad, n_points=4000)
        l2_hist.append(e_l2 / n_l2)
        h1_hist.append(e_h1 / n_h1)

    # P1: L2 ~ h^2 and H1 ~ h, so each mesh halving shrinks the error by ~4
    # (L2) and ~2 (H1): successive-error ratios are ~0.25 and ~0.5.
    l2_ratios = [l2_hist[i] / l2_hist[i - 1] for i in range(1, len(l2_hist))]
    h1_ratios = [h1_hist[i] / h1_hist[i - 1] for i in range(1, len(h1_hist))]
    assert sum(l2_ratios) / len(l2_ratios) == pytest.approx(0.25, abs=0.05)
    assert sum(h1_ratios) / len(h1_ratios) == pytest.approx(0.5, abs=0.05)
    assert l2_hist[-1] < 1e-3 and h1_hist[-1] < 5e-2