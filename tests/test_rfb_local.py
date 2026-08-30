"""FD reference solver (src.rfb_local): profile validation, trapezoidal
weights, tridiagonal (Thomas) solver, local dimensionless parameters, and
the residual-free bubble solver for every residual mode.
"""

import numpy as np
import pytest

from src.rfb_local import (
    _solve_tridiagonal,
    evaluate_diffusion_profile,
    interpolate_target,
    local_parameters,
    reference_p1_basis,
    solve_reference_rfb,
    trapezoidal_weights,
)


# --------------------------------------------------------------------------
# local_parameters
# --------------------------------------------------------------------------

def test_local_parameters_formulas():
    pe, rho = local_parameters(eps=0.01, beta=1.0, sigma=0.0, h=1 / 16)
    assert abs(pe - 1.0 / (32 * 0.01)) < 1e-3
    assert abs(rho - 0.0) < 1e-10

    pe2, rho2 = local_parameters(eps=1e-4, beta=1.0, sigma=1.0, h=1 / 16)
    assert abs(pe2 - 312.5) < 0.1
    assert abs(rho2 - 1.0 / 256 / 1e-4) < 1e-3


@pytest.mark.parametrize("eps,beta,sigma,h", [
    (0.0, 1.0, 0.0, 0.1),
    (-1.0, 1.0, 0.0, 0.1),
    (1.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0, -0.1),
    (np.nan, 1.0, 0.0, 0.1),
])
def test_local_parameters_rejects_invalid(eps, beta, sigma, h):
    with pytest.raises(ValueError):
        local_parameters(eps, beta, sigma, h)


# --------------------------------------------------------------------------
# evaluate_diffusion_profile
# --------------------------------------------------------------------------

def test_profile_scalar_callable_array():
    xi = np.linspace(0, 1, 11)
    assert np.allclose(evaluate_diffusion_profile(0.5, xi), 0.5)
    assert np.allclose(evaluate_diffusion_profile(lambda x: 2 * x + 1, xi),
                       2 * xi + 1)
    arr = np.linspace(1, 2, xi.size)
    assert np.allclose(evaluate_diffusion_profile(arr, xi), arr)


@pytest.mark.parametrize("eps,xi,expect", [
    (np.array([1.0, 1.0]), np.linspace(0, 1, 3), "shape"),       # shape mismatch
    (np.array([1.0, -1.0, 1.0]), np.linspace(0, 1, 3), "positive"),
    (np.array([1.0, np.nan, 1.0]), np.linspace(0, 1, 3), "finite"),
    (lambda x: np.zeros_like(x), np.linspace(0, 1, 6), "positive"),
    (lambda x: np.ones(len(x) + 1), np.linspace(0, 1, 6), "shape"),
    (lambda x: x[::-1], np.linspace(0, 1, 6), "shape"),          # reversed shape
])
def test_profile_rejects_invalid(eps, xi, expect):
    with pytest.raises(ValueError):
        evaluate_diffusion_profile(eps, xi)


@pytest.mark.parametrize("xi", [
    np.array([0.0]),                       # too short
    np.array([0.0, np.nan, 1.0]),          # non-finite
    np.array([0.0, 0.5, 0.25]),            # non-increasing
    np.zeros((3, 3)),                      # not 1-D
])
def test_profile_rejects_bad_grid(xi):
    with pytest.raises(ValueError):
        evaluate_diffusion_profile(1.0, xi)


# --------------------------------------------------------------------------
# trapezoidal weights
# --------------------------------------------------------------------------

def test_trapezoidal_weights_integrate_exactly():
    xi = np.linspace(0, 1, 101)
    w = trapezoidal_weights(xi)
    # Sum of weights == interval length; integrates constant === length.
    assert abs(w.sum() - 1.0) < 1e-14
    assert abs(np.sum(w * 3.7) - 3.7) < 1e-14
    # Linear functions are integrated exactly by the trapezoid rule.
    f = 2.0 * xi + 1.0
    assert abs(np.sum(w * f) - 2.0) < 1e-12

    with pytest.raises(ValueError):
        trapezoidal_weights(np.array([0.0, 1.0, 0.5]))


# --------------------------------------------------------------------------
# tridiagonal solve
# --------------------------------------------------------------------------

def test_tridiagonal_matches_dense_solve():
    rng = np.random.default_rng(0)
    for n in (2, 8, 64):
        sub = rng.normal(size=max(0, n - 1))
        main = np.abs(rng.normal(size=n)) + 1.0  # diagonal dominance
        sup = rng.normal(size=max(0, n - 1))
        rhs = rng.normal(size=n)
        A = np.diag(main)
        if n > 1:
            A += np.diag(sub, -1) + np.diag(sup, 1)
        x = _solve_tridiagonal(sub, main, sup, rhs)
        assert np.allclose(A @ x, rhs, atol=1e-12)


def test_tridiagonal_validation():
    with pytest.raises(ValueError):
        _solve_tridiagonal(np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        _solve_tridiagonal(np.zeros(2), np.ones(3), np.ones(2),
                           np.array([np.nan] * 3))
    # Zero pivot -> LinAlgError.
    with pytest.raises(np.linalg.LinAlgError):
        _solve_tridiagonal(np.ones(2), np.array([0.0, 1.0, 1.0]),
                           np.ones(2), np.ones(3))


# --------------------------------------------------------------------------
# reference P1 basis
# --------------------------------------------------------------------------

def test_reference_p1_basis_sum_is_one():
    xi = np.linspace(0, 1, 13)
    phi, dphi = reference_p1_basis(xi)
    assert np.allclose(phi.sum(0), 1.0)
    assert np.allclose(dphi[0], -1.0)
    assert np.allclose(dphi[1], 1.0)


# --------------------------------------------------------------------------
# solve_reference_rfb
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["constant", "xi", "one_minus_xi",
                                  "companion_1", "companion_2"])
def test_fd_bubble_modes_exist_and_normalized(mode):
    sol = solve_reference_rfb(eps=1e-4, beta=1.0, sigma=1.0, h=1 / 16,
                              residual_mode=mode, n_points=400)
    assert all(k in sol for k in ("xi", "b", "db", "b_raw", "center", "params"))
    assert abs(sol["xi"][0]) < 1e-14 and abs(sol["xi"][-1] - 1.0) < 1e-14
    assert abs(sol["b"][0]) < 1e-14 and abs(sol["b"][-1]) < 1e-14
    mid = int(np.argmin(np.abs(sol["xi"] - 0.5)))
    assert abs(sol["b"][mid] - 1.0) < 1e-2
    assert np.isfinite(sol["b"]).all()
    # Companion residuals may have either sign at the midpoint; what matters
    # is the normalization is nonzero so b(0.5) == 1 exactly.
    assert abs(sol["center"]) > 1e-12 and np.sign(sol["center"]) == np.sign(sol["b_raw"][mid])


@pytest.mark.parametrize("mode", ["constant", "xi", "one_minus_xi"])
def test_fd_bubble_physical_modes_positive(mode):
    sol = solve_reference_rfb(eps=1e-4, beta=1.0, sigma=0.0, h=1 / 16,
                              residual_mode=mode, n_points=400)
    assert (sol["b"] >= 0).all()
    assert sol["center"] > 0


def test_fd_bubble_reaction_case_finite():
    sol = solve_reference_rfb(eps=1e-2, beta=1.0, sigma=5.0, h=1 / 16,
                              residual_mode="constant", n_points=400)
    assert np.isfinite(sol["b"]).all()
    mid = int(np.argmin(np.abs(sol["xi"] - 0.5)))
    assert abs(sol["b"][mid] - 1.0) < 1e-2


def test_fd_bubble_with_profile_callable():
    eps_fn = lambda xi: 1.0 + np.sin(np.pi * xi)          # smooth, > 0
    sol = solve_reference_rfb(eps=eps_fn, beta=0.5, sigma=0.0, h=0.25,
                              residual_mode="constant", n_points=400)
    assert np.isfinite(sol["b"]).all()
    assert abs(sol["b"][np.argmin(np.abs(sol["xi"] - 0.5))] - 1.0) < 1e-2


def test_fd_bubble_rejects_bad_inputs():
    with pytest.raises(ValueError):
        solve_reference_rfb(1.0, 1.0, 0.0, 1.0, n_points=4)
    with pytest.raises(ValueError):
        solve_reference_rfb(1.0, 1.0, 0.0, -1.0)
    with pytest.raises(ValueError):
        solve_reference_rfb(1.0, 1.0, 0.0, 1.0, residual_mode="bogus")
    with pytest.raises(ValueError):
        solve_reference_rfb(1.0, np.nan, 0.0, 1.0)


def test_interpolate_target_matches():
    sol = solve_reference_rfb(eps=0.01, beta=1.0, sigma=0.0, h=1 / 16,
                              residual_mode="constant", n_points=400)
    xi_q = np.linspace(0, 1, 37)
    b, db = interpolate_target(sol, xi_q)
    b_ref = np.interp(xi_q, sol["xi"], sol["b"])
    db_ref = np.interp(xi_q, sol["xi"], sol["db"])
    assert np.allclose(b, b_ref)
    assert np.allclose(db, db_ref)