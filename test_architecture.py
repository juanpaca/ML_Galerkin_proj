"""Focused regression tests for architectural correctness and scalability hooks."""

import json
import numpy as np
import pytest

from data_generation import shape_no_leak_split, shape_no_leak_split_from_pool
from src.dataset_generation import (
    bubble_cosine_similarity,
    bubble_gram_matrix,
    max_cross_similarity,
    save_dataset,
    load_dataset,
)
from src.rfb_local import local_parameters, solve_reference_rfb, _solve_tridiagonal
from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.rfb_assembly import assemble_classical_system
from src.training import train_multi_bubble_on_dataset
from src.dataset_generation import train_multi_bubble_on_dataset as implementation_trainer


def test_similarity_uses_trapezoidal_weights():
    xi = np.array([0.0, 0.25, 1.0])
    b = np.array([[1.0, 1.0, 1.0]])
    # Integral of the constant function on [0, 1] is exactly one.
    assert np.isclose(bubble_gram_matrix(b, xi)[0, 0], 1.0)


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


def test_canonical_training_facade():
    assert train_multi_bubble_on_dataset is implementation_trainer
