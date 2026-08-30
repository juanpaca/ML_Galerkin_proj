"""KAN implementation: B-spline basis, KAN1D model, KANLayer, boundary cases.

Validated against ``scipy.interpolate.BSpline.design_matrix`` (including
endpoint folds), polynomial-reproduction degree pinning (k=3 -> quadratic),
and finite-difference gradients.
"""

import numpy as np
import pytest
import torch

from scipy.interpolate import BSpline

from src.kan import KAN1D, _eval_bspline_basis, _extend_knots, silu
from src.rfb_bubble import KANLayer


# --------------------------------------------------------------------------
# B-spline basis evaluation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("grid,k", [(4, 2), (8, 3), (12, 3), (6, 4)])
def test_bspline_basis_matches_scipy(grid, k):
    knots = _extend_knots(-1.0, 1.0, grid, k).double()
    xd = torch.linspace(-1.0, 1.0, 601, dtype=torch.float64)
    B = _eval_bspline_basis(xd, knots, k)
    ref = BSpline.design_matrix(xd.numpy(), knots.numpy(), k - 1).toarray()
    assert np.abs(B.numpy() - ref).max() < 1e-10


@pytest.mark.parametrize("grid,k", [(4, 2), (8, 3), (12, 3), (6, 4)])
def test_bspline_partition_of_unity_closed_domain(grid, k):
    knots = _extend_knots(-1.0, 1.0, grid, k).double()
    xd = torch.linspace(-1.0, 1.0, 251, dtype=torch.float64)
    B = _eval_bspline_basis(xd, knots, k)
    rowsum = B.sum(dim=1)
    ends = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    Be = _eval_bspline_basis(ends, knots, k)
    assert abs(float(rowsum.max()) - 1.0) < 1e-10
    assert abs(float(rowsum.min()) - 1.0) < 1e-10
    assert abs(float(Be[0].sum()) - 1.0) < 1e-10
    assert abs(float(Be[1].sum()) - 1.0) < 1e-10
    # Non-negative and locally supported (<= k active basis per row).
    assert B.min() >= -1e-12
    assert (B.abs() > 1e-14).sum(dim=1).max() <= k


def test_bspline_zero_outside_domain():
    knots = _extend_knots(-1.0, 1.0, 8, 3).double()
    out = _eval_bspline_basis(
        torch.tensor([-5.0, -1.5, 1.5, 9.0], dtype=torch.float64), knots, 3)
    assert out.abs().max() == 0.0


def test_polynomial_reproduction_pins_true_degree():
    knots = _extend_knots(-1.0, 1.0, 10, 3).double()
    xd = torch.linspace(-1, 1, 400, dtype=torch.float64)
    A = _eval_bspline_basis(xd, knots, 3).numpy()
    for p, expect_exact in [(1, True), (2, True), (3, False)]:
        coef, *_ = np.linalg.lstsq(A, xd.numpy() ** p, rcond=None)
        err = float(np.abs(A @ coef - xd.numpy() ** p).max())
        if expect_exact:
            assert err < 1e-9
        else:
            assert err > 1e-4


def test_bspline_gradient_autograd_matches_fd():
    knots = _extend_knots(-1.0, 1.0, 8, 3).double()
    xg = torch.tensor([-0.73, -0.31, 0.17, 0.55], dtype=torch.float64,
                      requires_grad=True)
    w = torch.randn(knots.numel() - 3, dtype=torch.float64)

    def f_basis(xx):
        return _eval_bspline_basis(xx, knots, 3) @ w

    d_aut = torch.autograd.grad(f_basis(xg).sum(), xg)[0]
    h = 1e-6
    with torch.no_grad():
        d_fd = (f_basis(xg.detach() + h) - f_basis(xg.detach() - h)) / (2 * h)
    assert (d_aut - d_fd).abs().max() < 1e-4


# --------------------------------------------------------------------------
# KAN1D model
# --------------------------------------------------------------------------

def test_kan1d_model_api_and_dtypes():
    torch.manual_seed(42)
    m = KAN1D(n_grid=8, k=3)
    assert m.n_basis == 10
    assert m.w_s.item() == 1.0
    assert abs(m.c.std().item() - 0.1) < 0.05

    xc = torch.rand(64) * 2 - 1
    y_direct = m(xc)
    y_precomp = m(xc, precomputed_basis=_eval_bspline_basis(xc, m.knots, m.k))
    assert torch.allclose(y_direct, y_precomp, atol=1e-7)

    g_, dg_ = m.forward_with_deriv(xc[:8])
    assert torch.allclose(g_, y_direct[:8], atol=1e-7)

    co = m.get_coefficients()
    assert set(co) == {"w_b", "w_s", "c"}
    assert co["c"].shape == (m.n_basis,)

    # float64 follows the input dtype.
    xf = torch.tensor([-0.5, 0.25], dtype=torch.float64)
    assert m.double()(xf).dtype == torch.float64

    # Column-vector and arbitrary batch shapes.
    x2 = torch.rand(8, 1) * 2 - 1
    assert torch.allclose(m(x2), m(x2[:, 0]), atol=1e-7)
    xbox = torch.rand(4, 6) * 2 - 1
    assert m(xbox).shape == (4 * 6,)


def test_kan1d_axis_gradient_matches_fd():
    m = KAN1D(n_grid=8, k=3)
    xd = torch.linspace(-0.95, 0.95, 33, dtype=torch.float64,
                        requires_grad=True)
    d_model = torch.autograd.grad(m.double()(xd).sum(), xd)[0]
    h = 1e-6
    with torch.no_grad():
        d_fd = (m.double()((xd + h).detach()) - m.double()((xd - h).detach())) / (2 * h)
    assert (d_model - d_fd).abs().max() < 1e-4


def test_kan1d_outside_domain_spline_zero_but_finite():
    """Outside [x_min, x_max] the spline term must vanish, leaving the
    bounded SiLU base term (exactly, since torch operations are linear)."""
    m = KAN1D(n_grid=8, k=3, x_min=-1.0, x_max=1.0)
    xo = torch.tensor([-2.0, 1.5, 5.0])
    expected = m.w_b * silu(xo)
    assert torch.allclose(m(xo), expected, atol=1e-6)
    assert torch.isfinite(m(xo)).all()


# --------------------------------------------------------------------------
# KANLayer (used by all trained bubble models)
# --------------------------------------------------------------------------

def test_kanlayer_parameter_count_and_knots():
    torch.manual_seed(0)
    layer = KANLayer(4, 3, n_grid=12, k=3)
    n_expected = layer.n_out * layer.n_in * (layer.n_basis + 2)
    n_actual = sum(p.numel() for p in layer.parameters())
    assert n_actual == n_expected
    assert layer.n_basis == 12 + 3 - 1


def test_kanlayer_b_splines_match_kan_evaluator():
    torch.manual_seed(0)
    layer = KANLayer(4, 3, n_grid=12, k=3)
    xb = torch.rand(128, 4, dtype=torch.float64) * 2 - 1
    B_lyr = layer.double().b_splines(xb)[:, 2, :]
    B_ref = _eval_bspline_basis(xb[:, 2], layer.knots.double(), layer.k)
    assert torch.allclose(B_lyr, B_ref, atol=1e-10)


def test_kanlayer_endpoint_fold_partition_of_unity():
    layer = KANLayer(4, 3, n_grid=12, k=3)
    xe = torch.ones(5, 4)
    Be = layer.double().b_splines(xe)
    assert (Be.sum(-1) - 1).abs().max() < 1e-10
    # All mass lands on the last active basis at the right edge.
    assert (Be[..., layer.n_basis - 1] - 1).abs().max() < 1e-10

    x0 = torch.zeros(5, 4)
    B0 = layer.double().b_splines(x0)
    assert (B0.sum(-1) - 1).abs().max() < 1e-10


def test_kanlayer_forward_and_backward_finite():
    torch.manual_seed(0)
    layer32 = KANLayer(4, 3, n_grid=12, k=3)
    xin = torch.rand(64, 4)
    out = layer32(xin)
    assert out.shape == (64, 3)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for p in layer32.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------
# End-to-end approximation smoke test (small budget)
# --------------------------------------------------------------------------

def test_kan1d_fits_smooth_function():
    torch.manual_seed(7)
    model = KAN1D(n_grid=24, k=4)
    xt = torch.linspace(-1, 1, 512)
    target = torch.sin(3 * xt) + 0.5 * torch.cos(7 * xt)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    for _ in range(2000):
        opt.zero_grad()
        ((model(xt) - target) ** 2).mean().backward()
        opt.step()
        sched.step()
    rel_l2 = float((((model(xt) - target) ** 2).sum() / (target ** 2).sum()).sqrt())
    assert rel_l2 < 1e-3