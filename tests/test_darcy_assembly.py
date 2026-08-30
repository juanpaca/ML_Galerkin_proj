"""Darcy variable-diffusion assembly (src/darcy_assembly.py): exact
piecewise-constant element stiffness, P1 Galerkin, static condensation
with the full-domain residual bubbles, per-element local bubbles, and the
enrichment quality gates.
"""

import numpy as np
import pytest

from src.darcy_assembly import (
    assemble_enriched,
    assemble_p1,
    element_load,
    element_stiffness,
    enrichment_h1_error,
    enrichment_l2_gate,
    eval_enriched,
    local_element_bubbles,
    local_enrichment_h1_error,
    rel_l2,
)
from src.darcy_variable import PiecewiseDiffusion, solve_darcy_1d


def _profile(edges=(0.0, 0.3, 0.7, 1.0), values=(1.0, 5.0, 0.4)):
    return PiecewiseDiffusion(np.asarray(edges, dtype=float),
                              np.asarray(values, dtype=float))


ONE = lambda x: np.ones_like(np.asarray(x, dtype=float))
XI = lambda x: np.asarray(x, dtype=float)


# --------------------------------------------------------------------------
# element blocks
# --------------------------------------------------------------------------

def test_element_stiffness_exact_for_piecewise_constant():
    profile = _profile()
    xa, xb = 0.2, 0.9
    h = xb - xa
    # Manual: 1/h^2 * int_eps [[1,-1],[-1,1]].
    pairs = list(zip(zip(profile.edges[:-1], profile.edges[1:]),
                     profile.values))
    int_eps = sum(v * (min(xb, e1) - max(xa, e0))
                  for (e0, e1), v in pairs)
    expected = int_eps / h ** 2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(element_stiffness(xa, xb, profile), expected)

    # Constant element -> eps * [1 -1; -1 1] / h.
    eps0 = 2.0
    flat = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([eps0]))
    ke = element_stiffness(0.25, 0.75, flat)
    assert np.allclose(ke, eps0 / 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]]))


def test_element_load_matches_high_order_quadrature():
    xa, xb = 0.3, 0.7
    h = xb - xa
    nodes, weights = np.polynomial.legendre.leggauss(20)
    xm, hr = 0.5 * (xa + xb), 0.5 * h
    xq = xm + hr * nodes
    for src in (ONE, XI):
        fg = src(xq)
        phi1 = (xb - xq) / h
        phi2 = (xq - xa) / h
        ref = np.array([np.sum(weights * fg * phi1), np.sum(weights * fg * phi2)]) * hr
        F = element_load(xa, xb, src)
        assert np.allclose(F, ref, atol=1e-12)
    # Constant unit source on a full element gives (h/2, h/2).
    assert np.allclose(element_load(0.0, 0.5, ONE), [0.25, 0.25])


# --------------------------------------------------------------------------
# P1 Galerkin on a constant profile
# --------------------------------------------------------------------------

def test_assemble_p1_reproduces_nodal_exactness():
    kappa = 2.0
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([kappa]))
    mesh = np.linspace(0.0, 1.0, 9)
    A, F, free = assemble_p1(mesh, profile, ONE)
    u = np.zeros(mesh.size)
    u[free] = np.linalg.solve(A, F)
    # u = x(1-x)/(2 kappa) is reproduced at nodes by P1 with constant load.
    ref = 0.5 * mesh * (1.0 - mesh) / kappa
    assert np.allclose(u, ref, atol=1e-10)
    assert np.allclose(A, A.T)


# --------------------------------------------------------------------------
# enriched assembly on constant / smooth profiles
# --------------------------------------------------------------------------

def _bubbles_for(profile, n_fd):
    return np.stack([
        solve_darcy_1d(profile, length=1.0, source=1.0, n_points=n_fd)["u_norm"],
        solve_darcy_1d(profile, length=1.0, source=XI, n_points=n_fd)["u_norm"],
    ])


def test_assemble_enriched_constant_profile_exact():
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([1.0]))
    n_fd = 801
    bubbles = _bubbles_for(profile, n_fd)
    xi = np.linspace(0.0, 1.0, n_fd)
    dx = xi[1] - xi[0]
    for src in (ONE, XI):
        u_nodes, coeffs = assemble_enriched(
            np.linspace(0.0, 1.0, 5), profile, src, bubbles, xi, dx)
        assert coeffs.shape == (2,)
        u_en = eval_enriched(u_nodes, coeffs, bubbles, np.linspace(0.0, 1.0, 5), xi)
        ref = solve_darcy_1d(profile, length=1.0, source=src, n_points=1201)
        assert rel_l2(np.interp(ref["x"], xi, u_en), ref["u"], ref["x"]) < 5e-3
        assert np.all(np.isfinite(u_en))


def test_eval_enriched_degenerates_to_interp():
    mesh = np.linspace(0.0, 1.0, 5)
    u_nodes = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    xq = np.linspace(0.0, 1.0, 99)
    assert np.allclose(eval_enriched(u_nodes, None, None, mesh, xq),
                       np.interp(xq, mesh, u_nodes))


# --------------------------------------------------------------------------
# per-element local bubbles
# --------------------------------------------------------------------------

def test_local_element_bubbles_invariants():
    profile = _profile()
    xi_loc, B = local_element_bubbles(profile, 0.2, 0.8, n_fd=401)
    assert xi_loc.shape == (401,) and B.shape == (2, 401)
    mid = 200
    assert abs(B[0, mid] - 1.0) < 1e-3 and abs(B[1, mid] - 1.0) < 1e-3
    assert abs(B[0, 0]) < 1e-3 and abs(B[0, -1]) < 1e-3
    assert (B[0] >= 0).all() and (B[1] >= 0).all()


def test_local_enrichment_h1_error_reproduces():
    # On a constant profile every uniform element is diffusion-aligned, so
    # A_Lb == 0 and the local-bubble condensation is span-only and exact.
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([1.0]))
    res = local_enrichment_h1_error(profile, source=ONE,
                                    n_fd_bubble=801, n_fd_ref=4001, n_el=8)
    assert set(("rel_h1_enriched", "rel_l2_enriched")) <= set(res)
    assert res["rel_l2_enriched"] < 1e-3
    assert res["rel_h1_enriched"] < 1e-2


# --------------------------------------------------------------------------
# enrichment gates
# --------------------------------------------------------------------------

def test_enrichment_l2_gate_constant_profile_passes():
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([1.0]))
    bubbles = _bubbles_for(profile, 801)
    out = enrichment_l2_gate(profile, bubbles,
                             {"constant": ONE, "xi": XI}, n_ref=8001, n_el=8)
    assert {"constant", "xi"} == set(out)
    assert out["constant"] < 5e-3 and out["xi"] < 5e-3


def test_enrichment_l2_gate_requires_aligned_grid():
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([1.0]))
    bubbles = _bubbles_for(profile, 1000)     # 999 not divisible by 8
    with pytest.raises(ValueError):
        enrichment_l2_gate(profile, bubbles, {"constant": ONE}, n_ref=801)


def test_enrichment_l2_gate_detects_bad_bubbles():
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([1.0]))
    rng = np.random.default_rng(3)
    bad = rng.standard_normal((2, 801))       # not residual bubbles at all
    out = enrichment_l2_gate(profile, bad, {"constant": ONE, "xi": XI},
                             n_ref=801, n_el=8)
    # Garbage bubbles fail the data-quality bar (~1e-2), i.e. they would be
    # dropped by a gate at the 1e-2 threshold.
    assert max(out.values()) > 1e-2


def test_enrichment_h1_error_summary():
    profile = _profile()
    res = enrichment_h1_error(profile, source=ONE, n_fd=801, n_el=8)
    assert set(("rel_h1_galerkin", "rel_h1_enriched", "rel_l2_enriched",
                "coeffs")) <= set(res)
    assert res["rel_h1_enriched"] < res["rel_h1_galerkin"]
    assert res["rel_l2_enriched"] < 2e-2
    assert res["coeffs"].shape == (2,)