"""Dataset splits: shape-based no-twin split (data_generation.py), the
(Pe, rho) cell-based split, and leak-free verification.
"""

import numpy as np
import pytest

from data_generation import shape_no_leak_split, shape_no_leak_split_from_pool
from src.dataset_generation import (
    _pe_rho_cell,
    bubble_cosine_similarity,
    cell_based_split,
    max_cross_similarity,
)


def _synth_pool(n, n_pts=101, rng=None):
    """Near-orthogonal sine bubble family (distinct frequencies) so the pool
    is genuinely diverse: no accidental near-duplicates, the splitter decides
    purely by shape centrality."""
    rng = rng or np.random.default_rng(0)
    xi = np.linspace(0, 1, n_pts)
    ks = np.arange(1, n + 1)
    b_const = np.sin(ks[:, None] * np.pi * xi)
    b_const /= b_const[:, np.argmin(np.abs(xi - 0.5))][:, None]
    b_xi = np.sin(ks[:, None] * np.pi * xi) + 0.3 * np.sin((ks + 1)[:, None] * np.pi * xi)
    b_xi /= b_xi[:, np.argmin(np.abs(xi - 0.5))][:, None]
    pool = {
        "constant": {"b": b_const, "xi": xi, "pe": rng.uniform(1, 100, n),
                     "rho": rng.uniform(0, 30, n)},
        "xi": {"b": b_xi, "xi": xi, "pe": rng.uniform(1, 100, n),
               "rho": rng.uniform(0, 30, n)},
    }
    return pool


def _trapezoid_weights(xi):
    w = np.empty(xi.size)
    w[0] = 0.5 * (xi[1] - xi[0])
    w[-1] = 0.5 * (xi[-1] - xi[-2])
    w[1:-1] = 0.5 * (xi[2:] - xi[:-2])
    return w


def _max_train_ood_sim(split, pool, theta):
    """Recompute the exact no-twin metric used inside the splitter."""
    n_pts = pool["constant"]["xi"].size
    weights = _trapezoid_weights(pool["constant"]["xi"])
    train = np.asarray(split["train"], dtype=int)
    ood = np.concatenate([np.asarray(split["val"], dtype=int),
                          np.asarray(split["test"], dtype=int)])
    worst = 0.0
    for mode in ("constant", "xi"):
        b = pool[mode]["b"]
        norm = np.sqrt(np.maximum(np.sum(b ** 2 * weights, axis=1), 1e-30))
        V = (b * np.sqrt(weights)) / norm[:, None]
        worst = max(worst, float((V[train] @ V[ood].T).max()))
    return worst


# --------------------------------------------------------------------------
# shape_no_leak_split_from_pool
# --------------------------------------------------------------------------

def test_no_twin_split_disjoint_counts_and_leakage_free():
    pool = _synth_pool(80, rng=np.random.default_rng(1))
    theta = 0.99
    split = shape_no_leak_split_from_pool(pool, ("constant", "xi"),
                                          n_train=40, n_val=12, n_test=18,
                                          theta=theta)
    train, val, test = map(set, (split["train"], split["val"], split["test"]))
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert len(train) == 40 and len(val) == 12 and len(test) == 18
    assert all(0 <= i < 80 for i in train | val | test)
    # The exact twin guard holds: no train sample is a >theta twin of val/test.
    worst = _max_train_ood_sim(split, pool, theta)
    assert worst <= theta + 1e-12
    assert split["stats"]["n_pool"] == 80


def test_max_cross_similarity_consistent_with_splitter_metric():
    pool = _synth_pool(50, rng=np.random.default_rng(2))
    split = shape_no_leak_split_from_pool(pool, ("constant", "xi"),
                                          n_train=25, n_val=8, n_test=10,
                                          theta=0.99)
    train = np.asarray(split["train"], dtype=int)
    ood = np.concatenate([np.asarray(split["val"], dtype=int),
                          np.asarray(split["test"], dtype=int)])
    sim = max_cross_similarity(pool["constant"]["b"][ood],
                               pool["constant"]["b"][train],
                               pool["constant"]["xi"])
    assert np.all(sim <= 0.99 + 1e-9)


def test_no_twin_split_identical_bubbles_raise_when_train_requested():
    xi = np.linspace(0, 1, 51)
    b = 4.0 * xi * (1.0 - xi)
    pool = {"constant": {"b": np.tile(b, (10, 1)), "xi": xi},
            "xi": {"b": np.tile(2.0 * b, (10, 1)), "xi": xi}}
    with pytest.raises(RuntimeError):
        shape_no_leak_split_from_pool(pool, ("constant", "xi"), n_train=5,
                                      n_val=2, n_test=3, theta=0.99)


def test_no_twin_split_validation():
    pool = _synth_pool(10)
    with pytest.raises(ValueError):
        shape_no_leak_split_from_pool(pool, ("constant", "xi"), 3, 2, 2,
                                      block_size=0)
    with pytest.raises(ValueError):
        shape_no_leak_split_from_pool(pool, (), 3, 2, 2)


# --------------------------------------------------------------------------
# shape_no_leak_split (similarity-matrix version)
# --------------------------------------------------------------------------

def test_shape_no_leak_split_matrix_version():
    # Near-orthogonal bubble shapes -> every sample is a valid train
    # candidate at theta=0.99, exercising the centrality/argsort path.
    xi = np.linspace(0.0, 1.0, 201)
    modes = np.vstack([np.sin((i + 1) * np.pi * xi) for i in range(1, 31)])
    C = bubble_cosine_similarity(modes, xi)
    split = shape_no_leak_split(C, n_train=18, n_val=5, n_test=5)
    assert len(split["train"]) == 18 and len(split["val"]) == 5 and len(split["test"]) == 5
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    stats = split["stats"]
    assert stats["theta"] == 0.99 and stats["n_pool"] == 30


# --------------------------------------------------------------------------
# (Pe, rho) cell-based split
# --------------------------------------------------------------------------

def test_pe_rho_cell_binning():
    pe = np.array([0.5, 5.0, 50.0, 500.0, 5000.0])
    rho = np.array([0.0, 0.5, 5.0, 50.0, 500.0])
    pe_idx, rho_idx = _pe_rho_cell(pe, rho)
    assert np.array_equal(pe_idx, [0, 1, 2, 3, 4])
    assert np.array_equal(rho_idx, [1, 1, 2, 2, 3])


def test_cell_based_split_no_leak():
    np.random.seed(42)
    n = 500
    pe_data = 10.0 ** np.random.uniform(-0.5, 4.5, n)
    rho_data = np.abs(np.random.randn(n) * 10)
    rho_bins = (-np.inf, 0, 0.1, 1, 10, 100, np.inf)

    train_idx, val_idx, test_idx, cell_map = cell_based_split(
        pe_data, rho_data, n_val_cells=3, n_test_cells=3,
        rho_bins=rho_bins, seed=123)

    train_cells = {c for c, (s, _) in cell_map.items() if s == "train"}
    val_cells = {c for c, (s, _) in cell_map.items() if s == "val"}
    test_cells = {c for c, (s, _) in cell_map.items() if s == "test"}
    assert train_cells.isdisjoint(val_cells) and train_cells.isdisjoint(test_cells)
    assert len(val_cells) >= 1 and len(test_cells) >= 1
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == n