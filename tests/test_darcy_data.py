"""Darcy variable-diffusion data generation (src/darcy_variable.py):
PiecewiseDiffusion validation/semantics, the conservative FD solver,
profile features (gauss ratios, resistivity CDF, scaled combos), and pool
generation invariants.
"""

import numpy as np
import pytest

from src.darcy_variable import (
    PiecewiseDiffusion,
    cumulative_resistivity_features,
    generate_darcy_pool,
    make_profile_features,
    profile_features,
    random_piecewise_diffusion,
    scaled_combo_features,
    solve_darcy_1d,
)


# --------------------------------------------------------------------------
# PiecewiseDiffusion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("edges,values", [
    (np.array([0.0, 1.0]), np.array([1.0])),                  # minimal ok
    (np.array([0.0, 0.4, 1.0]), np.array([2.0, 7.0])),        # ok
    (np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0])),        # not from 0
    (np.array([0.0, 0.5, 0.5, 1.0]), np.array([1.0, 1.0, 1.0])),  # non-strict
    (np.array([0.0, 0.5, 1.0]), np.array([1.0])),             # wrong count
    (np.array([0.0, 0.5, 1.0]), np.array([1.0, -1.0])),       # non-positive
    (np.array([0.0, 0.5, 1.0]), np.array([1.0, np.nan])),     # non-finite
])
def test_piecewise_diffusion_validation(edges, values):
    if (np.diff(edges) > 0).all() and edges[0] == 0.0 and edges[-1] == 1.0 \
            and values.size + 1 == edges.size and (values > 0).all() \
            and np.isfinite(values).all():
        PiecewiseDiffusion(edges, values)   # must not raise
    else:
        with pytest.raises(ValueError):
            PiecewiseDiffusion(edges, values)


def test_piecewise_diffusion_evaluate():
    profile = PiecewiseDiffusion(np.array([0.0, 0.4, 1.0]),
                                 np.array([2.0, 7.0]))
    xi = np.array([0.0, 0.1, 0.4, 0.3999, 0.4001, 0.9, 1.0])
    assert np.allclose(profile.evaluate(xi), [2.0, 2.0, 7.0, 2.0, 7.0, 7.0, 7.0])
    with pytest.raises(ValueError):
        profile.evaluate(np.array([-0.1]))
    with pytest.raises(ValueError):
        profile.evaluate(np.array([1.1]))


# --------------------------------------------------------------------------
# random_piecewise_diffusion
# --------------------------------------------------------------------------

def test_random_profile_determinism_and_spec():
    rng = np.random.default_rng(5)
    p1 = random_piecewise_diffusion(rng, n_pieces=4, eps_range=(0.1, 10.0))
    rng2 = np.random.default_rng(5)
    p2 = random_piecewise_diffusion(rng2, n_pieces=4, eps_range=(0.1, 10.0))
    assert np.allclose(p1.edges, p2.edges)
    assert np.allclose(p1.values, p2.values)

    assert p1.edges.size == 5
    assert p1.values.size == 4
    assert p1.edges[0] == 0.0 and p1.edges[-1] == 1.0
    assert np.all(np.diff(p1.edges) > 0)
    assert p1.values.min() >= 0.1 and p1.values.max() <= 10.0


def test_random_profile_min_width():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = random_piecewise_diffusion(rng, n_pieces=3, min_width=0.1)
        assert np.diff(p.edges).min() >= 0.1 - 1e-12


def test_random_profile_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_piecewise_diffusion(rng, eps_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        random_piecewise_diffusion(rng, eps_range=(-1.0, 1.0))
    with pytest.raises(ValueError):
        random_piecewise_diffusion(rng, min_width=-1.0)
    with pytest.raises(ValueError):
        random_piecewise_diffusion(rng, n_pieces=0)
    with pytest.raises(ValueError):
        random_piecewise_diffusion(rng, n_pieces=3, min_width=0.4)  # 3*0.4>=1


# --------------------------------------------------------------------------
# solve_darcy_1d
# --------------------------------------------------------------------------

def test_darcy_constant_diffusion_analytic_shape():
    xi = np.linspace(0.0, 1.0, 801)
    for length in (0.7, 1.0, 1.9):
        sol = solve_darcy_1d(1.0, length=length, source=1.0, n_points=801)
        assert abs(sol["u"][0]) < 1e-12 and abs(sol["u"][-1]) < 1e-12
        # u = x(L-x)/(2 eps); normalized shape is x/L (1 - x/L).
        assert abs(float(np.interp(0.25, sol["xi"], sol["u_norm"])) - 0.75) < 1e-3
        assert abs(float(np.interp(0.5, sol["xi"], sol["u_norm"])) - 1.0) < 1e-9
        assert (sol["u"][1:-1] > 0).all()


def test_darcy_xi_source_analytic_shape():
    # -(eps u')' = x/L => u = L^2 (xi - xi^3) / (6 eps), u_norm(0.25)=5/8.
    sol = solve_darcy_1d(2.0, length=1.3, source=lambda x: x / 1.3, n_points=801)
    assert abs(float(np.interp(0.25, sol["xi"], sol["u_norm"])) - 0.625) < 1e-3
    assert abs(float(np.interp(0.5, sol["xi"], sol["u_norm"])) - 1.0) < 1e-9


def test_darcy_profile_inputs_scalar_callable_array():
    xi = np.linspace(0.0, 1.0, 401)
    eps_fn = lambda x: 1.0 + np.sin(np.pi * x)
    s1 = solve_darcy_1d(eps_fn, source=1.0, n_points=401)
    s2 = solve_darcy_1d(1.0 + np.sin(np.pi * xi), source=1.0, n_points=401)
    s3 = solve_darcy_1d(2.0, source=1.0, n_points=401)
    assert np.allclose(s1["u"], s2["u"])
    assert (s1["u"][1:-1] > 0).all()
    assert (s3["u"][1:-1] > 0).all()


def test_darcy_scale_invariance_of_normalized_shape():
    rng = np.random.default_rng(1)
    base = random_piecewise_diffusion(rng, n_pieces=3, eps_range=(0.1, 10.0))
    scaled = PiecewiseDiffusion(base.edges, 3.0 * base.values)
    u1 = solve_darcy_1d(base, source=1.0, n_points=601)["u_norm"]
    u2 = solve_darcy_1d(scaled, source=1.0, n_points=601)["u_norm"]
    assert np.abs(u1 - u2).max() < 1e-6   # rescaling eps leaves shapes unchanged


def test_darcy_validation():
    with pytest.raises(ValueError):
        solve_darcy_1d(1.0, length=0.0)
    with pytest.raises(ValueError):
        solve_darcy_1d(1.0, n_points=4)
    with pytest.raises(ValueError):
        solve_darcy_1d(-1.0, n_points=100)
    with pytest.raises(ValueError):
        solve_darcy_1d(1.0, source=lambda x: np.ones(len(x) + 2), n_points=100)


# --------------------------------------------------------------------------
# profile features
# --------------------------------------------------------------------------

def test_gauss_ratio_features_constant_profile():
    profile = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([3.7]))
    f = profile_features(profile, 8)
    assert np.allclose(f, 1.0)
    assert f.shape == (8,)


def test_gauss_ratio_feature_mean():
    rng = np.random.default_rng(2)
    profile = random_piecewise_diffusion(rng, n_pieces=3, eps_range=(0.1, 10.0))
    f = profile_features(profile, 16)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    weighted_mean = float(np.sum(0.5 * weights * f))
    assert abs(weighted_mean - 1.0) < 1e-12


def test_cumulative_resistivity_monotone_and_jump():
    rng = np.random.default_rng(3)
    profile = random_piecewise_diffusion(rng, n_pieces=3, eps_range=(0.1, 10.0))
    f = cumulative_resistivity_features(profile, 8)
    assert (np.diff(f) >= -1e-12).all()
    assert f.min() >= 1e-12 - 1e-12 and f.max() <= 1.0 + 1e-12

    const = PiecewiseDiffusion(np.array([0.0, 1.0]), np.array([2.0]))
    c = cumulative_resistivity_features(const, 8)
    legs, _ = np.polynomial.legendre.leggauss(8)
    xi_node = 0.5 * (legs + 1.0)
    assert np.allclose(c, xi_node, atol=1e-3)   # R(x) = x for constant eps


def test_scaled_combo_bounds_and_dimension():
    rng = np.random.default_rng(4)
    profile = random_piecewise_diffusion(rng, n_pieces=3, eps_range=(0.1, 10.0))
    g = scaled_combo_features(profile, 8)
    assert g.shape == (16,)
    assert g.min() >= -1.0 - 1e-12 and g.max() <= 1.0 + 1e-12
    g2 = scaled_combo_features(profile, 8, log_scale=4.0)
    assert g2.shape == (16,)


def test_make_profile_features_dispatch():
    rng = np.random.default_rng(0)
    profile = random_piecewise_diffusion(rng, n_pieces=3, eps_range=(0.1, 10.0))
    for kind, n in [("gauss_ratio", 8), ("resistivity_cdf", 8),
                    ("scaled_combo", 16), ("scaled_combo_v2", 16)]:
        f = make_profile_features(profile, 8, kind)
        assert f.shape == (n,)
        assert np.isfinite(f).all()
    with pytest.raises(ValueError):
        make_profile_features(profile, 8, "bogus")
    with pytest.raises(ValueError):
        profile_features(profile, 0)


# --------------------------------------------------------------------------
# generate_darcy_pool
# --------------------------------------------------------------------------

def test_pool_deterministic_and_shape_consistent():
    p1 = generate_darcy_pool(n_samples=6, n_fd_points=161, n_profile_features=8,
                             seed=11)
    p2 = generate_darcy_pool(n_samples=6, n_fd_points=161, n_profile_features=8,
                             seed=11)
    assert np.array_equal(p1["constant"]["b"], p2["constant"]["b"])
    assert np.array_equal(p1["constant"]["eps_profile"], p2["constant"]["eps_profile"])

    modes = p1["mode_names"]
    assert modes == ("constant", "xi")
    for mode in modes:
        d = p1[mode]
        assert d["b"].shape == (6, 161)
        assert d["db"].shape == (6, 161)
        assert d["length"].shape == (6,)
        assert d["pe"].shape == (6,) and d["rho"].shape == (6,)
        assert d["eps_ratios"].shape == (6, 8)
        assert d["idx"].tolist() == list(range(6))
        assert abs(d["b"][:, 80] - 1.0).max() < 1e-3     # u_norm(0.5) = 1
    assert np.all(p1["constant"]["eps_profile"] > 0)
    assert np.all(p1["constant"]["length"] >= 0.5)
    assert np.all(p1["constant"]["length"] <= 2.0)

    # Modes differ (source 1 vs xi).
    assert not np.allclose(p1["constant"]["b"], p1["xi"]["b"])


def test_pool_feature_kinds_and_grid_consistency():
    pool = generate_darcy_pool(n_samples=4, n_fd_points=161, n_profile_features=8,
                               seed=3, feature_kind="scaled_combo_v2")
    assert pool["constant"]["eps_ratios"].shape == (4, 16)
    assert pool["constant"]["eps_ratios"].min() >= -1.0 - 1e-12
    assert pool["constant"]["eps_ratios"].max() <= 1.0 + 1e-12
    xi = pool["constant"]["xi"]
    db = np.gradient(pool["constant"]["b"][0], xi)
    assert np.abs(pool["constant"]["db"][0] - db).max() < 1e-4


def test_pool_validation():
    with pytest.raises(ValueError):
        generate_darcy_pool(n_samples=0)
    with pytest.raises(ValueError):
        generate_darcy_pool(n_samples=4, feature_kind="bogus")