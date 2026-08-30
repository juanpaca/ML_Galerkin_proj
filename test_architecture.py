"""Focused regression tests for architectural correctness and scalability hooks."""

import json
import numpy as np
import pytest

from data_generation import (
    shape_no_leak_split, shape_no_leak_split_from_pool, generate_pool, merge_pool,
)
from src.dataset_generation import (
    bubble_cosine_similarity,
    bubble_gram_matrix,
    max_cross_similarity,
    bubble_similarity_analysis,
    save_dataset,
    load_dataset,
)
from src.rfb_local import (
    local_parameters, solve_reference_rfb, _solve_tridiagonal,
    evaluate_diffusion_profile,
)
from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.rfb_assembly import assemble_classical_system
from src.rfb_assembly import assemble_rfb_condensed_system
from src.rfb_bubble import MultiKANBubble1D
from src.rfb_exact import ExactRFBubble1D
from src.darcy_variable import (
    PiecewiseDiffusion, random_piecewise_diffusion, solve_darcy_1d,
    profile_features, generate_darcy_pool,
)
from data_generation_darcy_variable import build_split_dataset
from src.training import train_multi_bubble_on_dataset
from src.dataset_generation import train_multi_bubble_on_dataset as implementation_trainer


def test_similarity_uses_h1_inner_product():
    xi = np.array([0.0, 0.25, 1.0])
    b = np.array([[1.0, 1.0, 1.0]])
    # A constant bubble has zero derivative: H1 gram == L2 gram == 1.
    assert np.isclose(bubble_gram_matrix(b, xi, lambda_deriv=0.2)[0, 0], 1.0)
    # A non-constant bubble gains derivative content: H1 gram > L2 gram.
    b_lin = np.array([[0.0, 1.0, 2.0]])
    G_l2 = bubble_gram_matrix(b_lin, xi, lambda_deriv=0.0)[0, 0]
    G_h1 = bubble_gram_matrix(b_lin, xi, lambda_deriv=0.2)[0, 0]
    assert G_h1 > G_l2


def test_blockwise_cross_similarity_matches_dense():
    rng = np.random.default_rng(4)
    xi = np.linspace(0.0, 1.0, 17)
    other = rng.normal(size=(7, xi.size))
    reference = rng.normal(size=(11, xi.size))
    dense = bubble_cosine_similarity(
        np.vstack([other, reference]), xi
    )[: len(other), len(other):].max(axis=1)
    assert np.allclose(max_cross_similarity(other, reference, xi, block_size=3), dense)


def test_shape_split_honors_training_cardinality_and_no_twins():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(12, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    C = vectors @ vectors.T
    split = shape_no_leak_split(C, n_train=4, n_val=2, n_test=2, theta=0.999)
    assert len(split["train"]) == 4
    assert len(split["val"]) == 2
    assert len(split["test"]) == 2
    assert not set(split["train"]) & set(split["val"] + split["test"])
    ood = split["val"] + split["test"]
    assert np.max(C[np.ix_(split["train"], ood)]) <= 0.999
    all_safe = shape_no_leak_split(C, n_train=None, n_val=2, n_test=2, theta=0.999)
    assert len(all_safe["train"]) >= len(split["train"])


def test_shape_split_fails_when_target_cannot_survive():
    C = np.ones((6, 6))
    np.fill_diagonal(C, 1.0)
    with pytest.raises(RuntimeError, match="survive"):
        shape_no_leak_split(C, n_train=4, n_val=1, n_test=1, theta=0.99)


def test_blockwise_pool_split_matches_dense_split():
    rng = np.random.default_rng(8)
    xi = np.linspace(0.0, 1.0, 13)
    b0 = rng.normal(size=(14, xi.size))
    b1 = rng.normal(size=(14, xi.size))
    pool = {"constant": {"b": b0, "xi": xi}, "xi": {"b": b1, "xi": xi}}
    C = np.maximum(bubble_cosine_similarity(b0, xi), bubble_cosine_similarity(b1, xi))
    dense = shape_no_leak_split(C, 5, 3, 3, theta=0.99)
    block = shape_no_leak_split_from_pool(pool, ("constant", "xi"), 5, 3, 3,
                                          theta=0.99, block_size=4)
    assert dense["train"] == block["train"]
    assert dense["val"] == block["val"]
    assert dense["test"] == block["test"]


def test_local_api_rejects_invalid_coefficients():
    with pytest.raises(ValueError):
        local_parameters(0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        solve_reference_rfb(-1.0, 1.0, 0.0, 1.0)
    with pytest.raises(np.linalg.LinAlgError):
        _solve_tridiagonal(np.array([0.0]), np.array([0.0, 1.0]),
                           np.array([0.0]), np.ones(2))


def test_constant_diffusion_scalar_array_and_callable_are_equivalent():
    xi = np.linspace(0.0, 1.0, 101)
    scalar = solve_reference_rfb(0.2, 1.0, 0.3, 1.0, n_points=101)
    array = solve_reference_rfb(np.full(xi.size, 0.2), 1.0, 0.3, 1.0,
                                n_points=101)
    function = solve_reference_rfb(lambda x: np.full(x.shape, 0.2), 1.0,
                                   0.3, 1.0, n_points=101)
    assert np.allclose(scalar["b"], array["b"])
    assert np.allclose(scalar["b"], function["b"])


def test_piecewise_diffusion_profile_is_supported_and_validated():
    xi = np.linspace(0.0, 1.0, 401)
    profile = lambda x: np.where(x < 0.5, 0.1, 1.0)
    values = evaluate_diffusion_profile(profile, xi)
    assert values.shape == xi.shape
    assert np.all(values > 0.0)
    result = solve_reference_rfb(profile, 0.5, 0.2, 1.0, n_points=401)
    assert np.all(np.isfinite(result["b"]))
    assert np.isclose(result["b"][0], 0.0)
    assert np.isclose(result["b"][-1], 0.0)
    with pytest.raises(ValueError, match="positive"):
        evaluate_diffusion_profile(lambda x: np.where(x < 0.5, 0.0, 1.0), xi)
    exact = ExactRFBubble1D(profile, 0.5, 0.2, 1.0,
                            residual_mode="constant", n_points=401)
    values, gradients = exact.value_grad_numpy(xi)
    assert values.shape == xi.shape
    assert gradients.shape == xi.shape


def test_dataset_round_trip_and_checksum(tmp_path):
    xi = np.linspace(0.0, 1.0, 9)
    mode = {
        "pe": np.array([1.0, 2.0]), "rho": np.array([0.1, 0.2]),
        "b": np.ones((2, xi.size)), "db": np.zeros((2, xi.size)),
        "xi": xi,
    }
    dataset = {
        "train": {"constant": mode},
        "metadata": {"name": "roundtrip", "mode_names": ["constant"]},
        "mode_names": ["constant"],
    }
    save_dataset(dataset, name="roundtrip", subdir=str(tmp_path))
    loaded = load_dataset("roundtrip", subdir=str(tmp_path))
    assert np.array_equal(loaded["train"]["constant"]["b"], mode["b"])
    path = tmp_path / "roundtrip_train_constant.npz"
    with path.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset("roundtrip", subdir=str(tmp_path))


def test_sparse_classical_assembly_matches_dense():
    mesh = Mesh1D(0.0, 1.0, 8)
    quad = GaussLegendre(4)
    pde = AdvectionDiffusion1D(0.01, 1.0, 0.2)
    dense_a, dense_f = assemble_classical_system(mesh, quad, pde)
    sparse_a, sparse_f = assemble_classical_system(
        mesh, quad, pde, sparse_output=True
    )
    assert sparse_a.format == "csr"
    assert np.allclose(sparse_a.toarray(), dense_a)
    assert np.allclose(sparse_f, dense_f)


def test_piecewise_diffusion_assembly_is_finite():
    mesh = Mesh1D(0.0, 1.0, 8)
    quad = GaussLegendre(8)
    pde = AdvectionDiffusion1D(0.2, 1.0, 0.1)
    pde.set_diffusion_from_function(lambda x: np.where(x < 0.5, 0.1, 1.0))
    A, f = assemble_classical_system(mesh, quad, pde, sparse_output=True)
    assert np.all(np.isfinite(A.data))
    assert np.all(np.isfinite(f))
    bubble = MultiKANBubble1D(n_bubbles=2, n_hidden=3, n_grid=4, n_eps=4)
    A_rfb, f_rfb, _ = assemble_rfb_condensed_system(
        mesh, quad, pde, bubble, sparse_output=True
    )
    assert np.all(np.isfinite(A_rfb.data))
    assert np.all(np.isfinite(f_rfb))


def test_canonical_training_facade():
    assert train_multi_bubble_on_dataset is implementation_trainer


def test_data_generation_switch_supports_piecewise_profiles_reproducibly():
    ds_a, cfg_a = generate_pool(
        8, (0.5, 2.0), (0.1, 1.0), 17, n_fd_points=41,
        diffusion_profile="layered", variable_eps_fraction=1.0,
        variable_eps_n_quad=3,
    )
    ds_b, cfg_b = generate_pool(
        8, (0.5, 2.0), (0.1, 1.0), 17, n_fd_points=41,
        diffusion_profile="layered", variable_eps_fraction=1.0,
        variable_eps_n_quad=3,
    )
    assert cfg_a.variable_eps_profile == "layered"
    assert cfg_a.variable_eps_fraction == 1.0
    pool_a, _, _ = merge_pool(ds_a)
    pool_b, _, _ = merge_pool(ds_b)
    assert "eps_ratios" in pool_a["constant"]
    assert pool_a["constant"]["eps_ratios"].shape == (8, 3)
    assert np.allclose(pool_a["constant"]["eps_ratios"], pool_b["constant"]["eps_ratios"])


def test_darcy_constant_profile_matches_normalized_exact_shape():
    result = solve_darcy_1d(2.0, length=3.0, n_points=101)
    xi = result["xi"]
    assert np.allclose(result["u_norm"], 4.0 * xi * (1.0 - xi), atol=2e-4)


def test_darcy_piecewise_profile_and_features():
    profile = PiecewiseDiffusion(np.array([0.0, 0.4, 1.0]), np.array([0.1, 100.0]))
    xi = np.array([0.1, 0.4, 0.9])
    assert np.array_equal(profile.evaluate(xi), np.array([0.1, 100.0, 100.0]))
    features = profile_features(profile, 6)
    assert features.shape == (6,)
    assert np.all(np.isfinite(features)) and np.all(features > 0.0)
    result = solve_darcy_1d(profile, length=2.0, n_points=201)
    assert np.all(np.isfinite(result["u_norm"]))
    assert np.isclose(result["u_norm"][0], 0.0)
    assert np.isclose(result["u_norm"][-1], 0.0)


def test_darcy_pool_is_reproducible_and_has_fixed_features():
    pool_a = generate_darcy_pool(n_samples=12, n_fd_points=41,
                                 n_profile_features=5, seed=13)
    pool_b = generate_darcy_pool(n_samples=12, n_fd_points=41,
                                 n_profile_features=5, seed=13)
    a = pool_a["constant"]
    b = pool_b["constant"]
    assert pool_a["mode_names"] == ("constant", "xi")
    assert pool_a["xi"]["b"].shape == (12, 41)
    assert a["b"].shape == (12, 41)
    assert a["eps_ratios"].shape == (12, 5)
    assert np.allclose(a["b"], b["b"])
    assert np.allclose(a["eps_ratios"], b["eps_ratios"])


def test_darcy_tutorial_split_is_leak_free():
    pool = generate_darcy_pool(n_samples=100, n_fd_points=41,
                               n_profile_features=4, seed=2)
    dataset = build_split_dataset(pool, theta=0.99,
                                  train_frac=0.35, val_frac=0.15,
                                  test_frac=0.25)
    report = bubble_similarity_analysis(dataset, mode="constant", verbose=False)
    stats = report["cross"]["train_vs_test"]["stats"]
    assert stats["frac_gt_0.99"] == 0.0
    assert stats["max_sim_max"] <= 0.99
