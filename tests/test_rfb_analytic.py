"""Analytic constant-coefficient residual-free bubbles (src/rfb_analytic.py):
exact solution of the local bubble ODE, boundary/normalization invariants,
ODE residual, Pe=0 symmetry, and the FD-vs-analytic cross-validation that
justifies the dataset's grid resolution choices.
"""

import numpy as np
import pytest

from src.rfb_analytic import exact_rfb, fd_error_metrics, fd_rfb


@pytest.mark.parametrize("mode", ["constant", "xi"])
def test_exact_bubble_bc_norm_positivity(mode):
    ex = exact_rfb(1.0, 1.0, mode, xi=np.linspace(0.0, 1.0, 401))
    assert abs(ex["b"][0]) < 1e-12
    assert abs(ex["b"][-1]) < 1e-12
    assert abs(float(np.interp(0.5, ex["xi"], ex["b_norm"])) - 1.0) < 1e-9
    assert (ex["b"] >= 0).all()
    assert ex["center"] > 0


def test_exact_bubble_ode_residual():
    ex = exact_rfb(1.0, 1.0, "constant", xi=np.linspace(0.0, 1.0, 2001))
    dxi = ex["xi"][1] - ex["xi"][0]
    # -b'' + 2 Pe b' + rho b = r/center, with Pe=rho=1 and r = 1.
    d2 = np.gradient(np.gradient(ex["b_norm"], dxi), dxi)
    res = -d2 + 2.0 * 1.0 * np.gradient(ex["b_norm"], dxi) \
        + 1.0 * ex["b_norm"] - 1.0 / ex["center"]
    assert np.max(np.abs(res[50:-50])) < 1e-3


def test_exact_bubble_pe_zero_symmetric():
    ex = exact_rfb(0.0, 1.0, "constant", xi=np.linspace(0.0, 1.0, 401))
    err = np.max(np.abs(ex["b_norm"] - ex["b_norm"][::-1]))
    assert err < 1e-10


def test_exact_bubble_reaction_case():
    ex = exact_rfb(1.0, 10.0, "constant", xi=np.linspace(0.0, 1.0, 401))
    assert np.isfinite(ex["b"]).all() and (ex["b"] >= 0).all()
    assert abs(ex["b"][0]) < 1e-12 and abs(ex["b"][-1]) < 1e-12
    assert abs(float(np.interp(0.5, ex["xi"], ex["b_norm"])) - 1.0) < 1e-9


def test_exact_bubble_high_pe_stable():
    for pe in (100.0, 1e4):
        ex = exact_rfb(pe, 1.0, "xi", xi=np.linspace(0.0, 1.0, 2001))
        assert np.isfinite(ex["b"]).all()
        assert ex["center"] > 1e-300
    # Asymmetry grows with Pe (advection layer moved to the right edge).
    ex_low = exact_rfb(1.0, 1.0, "xi", xi=np.linspace(0.0, 1.0, 401))
    ex_high = exact_rfb(50.0, 1.0, "xi", xi=np.linspace(0.0, 1.0, 401))
    asym = lambda e: np.max(np.abs(e["b_norm"] - e["b_norm"][::-1]))
    assert asym(ex_high) > asym(ex_low)


def test_exact_bubble_unknown_mode_raises():
    with pytest.raises(ValueError):
        exact_rfb(1.0, 1.0, "bogus")


# --------------------------------------------------------------------------
# FD vs analytic cross-validation (justifies n_fd resolution)
# --------------------------------------------------------------------------

def test_fd_3200_resolves_pe100_layer_but_400_does_not():
    ex = exact_rfb(100.0, 1.0, "xi", xi=np.linspace(0.0, 1.0, 4001))
    fd_400 = fd_rfb(100.0, 1.0, "xi", n_points=400, xi_ref=ex["xi"])
    fd_3200 = fd_rfb(100.0, 1.0, "xi", n_points=3200, xi_ref=ex["xi"])
    e400 = fd_error_metrics(fd_400, ex)["l2_rel_full"]
    e3200 = fd_error_metrics(fd_3200, ex)["l2_rel_full"]
    assert e400 > 1e-2          # under-resolved boundary layer
    assert e3200 < 5e-3         # resolved


@pytest.mark.parametrize("mode", ["constant", "xi"])
def test_fd_matches_analytic_at_pe1(mode):
    ex = exact_rfb(1.0, 1.0, mode, xi=np.linspace(0.0, 1.0, 401))
    fd = fd_rfb(1.0, 1.0, mode, n_points=400, xi_ref=ex["xi"])
    err = fd_error_metrics(fd, ex)["l2_rel_full"]
    assert err < 1e-2


def test_fd_error_metrics_contents():
    ex = exact_rfb(1.0, 1.0, "constant", xi=np.linspace(0.0, 1.0, 401))
    fd = fd_rfb(1.0, 1.0, "constant", n_points=400, xi_ref=ex["xi"])
    m = fd_error_metrics(fd, ex)
    assert set(("l2_rel", "l2_rel_full", "sup_rel_full", "l2_rel_layer",
                "peak_rel", "l2_rel_db", "osc_amp", "n_neg")) <= set(m)
    assert m["n_neg"] == 0              # upwind scheme is monotone
    assert m["osc_amp"] == 0.0