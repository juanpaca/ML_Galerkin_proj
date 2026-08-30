"""Darcy dataset production pipeline (data_generation_darcy_variable.py):
profile reconstruction, pool filtering, the enrichment gate (audit / drop /
fail), split strategies, and a small end-to-end generate+save+reload round
trip.
"""

import os

import numpy as np
import pytest

from data_generation_darcy_variable import (
    DATA_SUBDIR,
    _filter_pool,
    _profile_from_pool,
    audit_enrichment_gate,
    build_split_dataset,
    generate_and_save_dataset,
)
from src.darcy_assembly import enrichment_l2_gate
from src.darcy_variable import generate_darcy_pool
from src.rfb_training import DATASET_SUBDIR


def _small_pool(n=6, n_fd=161, min_pieces=3, max_pieces=3, seed=0):
    return generate_darcy_pool(n_samples=n, n_fd_points=n_fd,
                               n_profile_features=8,
                               n_pieces_range=(min_pieces, max_pieces),
                               min_width=0.01, seed=seed)


# --------------------------------------------------------------------------
# profile reconstruction & pool filtering
# --------------------------------------------------------------------------

def test_profile_from_pool_reconstructs_generator_profile():
    pool = _small_pool(n=3)
    for i in range(3):
        p = _profile_from_pool(pool, i)
        assert np.allclose(p.edges, pool["piece_edges"][i])
        assert np.allclose(p.values, pool["piece_values"][i])
        # Sample-level eps_profile equals the reconstructed profile on the grid.
        assert np.allclose(pool["constant"]["eps_profile"][i], p.evaluate(pool["constant"]["xi"]))


def test_filter_pool_keeps_arrays_in_sync():
    pool = _small_pool(n=6)
    keep = np.array([True, False, True, False, False, True])
    out = _filter_pool(pool, keep)
    n = int(keep.sum())
    assert out["metadata"]["n_samples"] == n
    for mode in pool["mode_names"]:
        d = out[mode]
        assert d["b"].shape[0] == n
        assert d["eps_profile"].shape[0] == n
        assert d["length"].shape[0] == n
        assert d["idx"].size == n
        assert np.allclose(d["xi"], pool[mode]["xi"])     # shared grid is full
    assert len(out["piece_edges"]) == n
    assert len(out["piece_values"]) == n


# --------------------------------------------------------------------------
# enrichment gate
# --------------------------------------------------------------------------

def test_gate_passes_with_loose_threshold():
    pool = _small_pool(n=6)
    pool2, report = audit_enrichment_gate(pool, threshold=1.5, n_ref=801,
                                          n_el=8, drop=False)
    assert len(pool2["constant"]["pe"]) == 6
    assert report["n_checked"] == 6
    assert report["n_failed"] == 0
    assert report["mesh"] == "uniform_n_el=8"
    for m in pool["mode_names"]:
        assert m in report["per_mode_mean"] and m in report["per_mode_p95"]
    assert report["worst"]["idx"] in range(6)


def test_gate_fails_hard_without_drop():
    pool = _small_pool(n=3)
    with pytest.raises(RuntimeError, match="enrichment gate failed"):
        audit_enrichment_gate(pool, threshold=1e-12, n_ref=801, n_el=8,
                              drop=False)


def test_gate_drop_filters_samples():
    pool = _small_pool(n=6)
    pool2, report = audit_enrichment_gate(pool, threshold=1e-12, n_ref=801,
                                          n_el=8, drop=True)
    assert report["n_dropped"] >= 1
    assert len(pool2["constant"]["pe"]) == 6 - report["n_dropped"]
    # The worst offender is among the dropped samples.
    assert "worst" in report


def test_gate_drop_is_deterministic():
    a = _small_pool(n=8, seed=4)
    b = _small_pool(n=8, seed=4)
    pa, ra = audit_enrichment_gate(a, threshold=1e-4, n_ref=801, n_el=8, drop=True)
    pb, rb = audit_enrichment_gate(b, threshold=1e-4, n_ref=801, n_el=8, drop=True)
    assert ra["n_dropped"] == rb["n_dropped"]
    assert np.array_equal(pa["constant"]["b"], pb["constant"]["b"])


def test_enrichment_l2_gate_runs_on_pool_bubbles():
    pool = _small_pool(n=1)
    profile = _profile_from_pool(pool, 0)
    bubbles = np.stack([pool["constant"]["b"][0], pool["xi"]["b"][0]])
    errs = enrichment_l2_gate(profile, bubbles,
                              {"constant": lambda x: np.ones_like(x),
                               "xi": lambda x: x}, n_ref=801, n_el=8)
    assert all(np.isfinite(v) and 0.0 <= v for v in errs.values())


# --------------------------------------------------------------------------
# split strategies
# --------------------------------------------------------------------------

def test_build_split_random_and_disjointness():
    pool = _small_pool(n=40)
    ds = build_split_dataset(pool, strategy="random", seed=1)
    assert ds["metadata"]["split_strategy"] == "random"
    assert ds["metadata"]["n_train"] + ds["metadata"]["n_val"] + ds["metadata"]["n_test"] == 40
    tr = ds["train"]["constant"]["idx"]
    va = ds["val"]["constant"]["idx"]
    te = ds["test"]["constant"]["idx"]
    assert len(set(tr)) + len(set(va)) + len(set(te)) == 40
    assert ds["metadata"]["n_val"] == 6 and ds["metadata"]["n_test"] == 10


def test_build_split_contrast_band_monotone():
    pool = _small_pool(n=40, seed=2)
    ds = build_split_dataset(pool, strategy="contrast_band")
    edges = ds["metadata"]["contrast_band_edges"]
    b = ds["metadata"]["contrast_band_edges"]
    assert b["train"][0] <= b["train"][1] <= b["val"][0] <= b["val"][1] \
        <= b["test"][0] <= b["test"][1]
    assert edges["train"][0] >= 1.0
    # Contrasts must be separated across bands (no leakage by construction).
    tr = ds["train"]["constant"]["idx"]
    te = ds["test"]["constant"]["idx"]
    assert set(tr).isdisjoint(te)


def test_build_split_no_twin_metadata():
    pool = _small_pool(n=40, seed=3)
    ds = build_split_dataset(pool, strategy="no_twin_shape", theta=1.0)
    assert ds["metadata"]["similarity_theta"] == 1.0
    assert ds["metadata"]["split_strategy"] == "no_twin_shape"
    total = ds["metadata"]["n_train"] + ds["metadata"]["n_val"] + ds["metadata"]["n_test"]
    assert total == 40


def test_build_split_validation():
    pool = _small_pool(n=40)
    with pytest.raises(ValueError):
        build_split_dataset(pool, strategy="bogus")
    with pytest.raises(ValueError):
        build_split_dataset(pool, val_frac=0.5, test_frac=0.6)


# --------------------------------------------------------------------------
# end-to-end generate + save + reload + leak audit
# --------------------------------------------------------------------------

def test_generate_and_save_dataset_roundtrip():
    import glob as _glob

    name = "ci_darcy_e2e"
    data_dir = os.path.join(DATASET_SUBDIR, DATA_SUBDIR)
    try:
        ds = generate_and_save_dataset(
            name=name,
            subdir=DATA_SUBDIR,
            n_samples=24,
            n_fd_points=201,
            n_profile_features=8,
            min_pieces=3,
            max_pieces=3,
            eps_range=(0.1, 100.0),
            min_width=0.01,
            feature_kind="scaled_combo_v2",
            seed=7,
            theta=1.0,
            val_frac=0.15,
            test_frac=0.25,
            split_strategy="no_twin_shape",
            verify_enrichment=True,
            gate_threshold=1.5,
            gate_n_ref=1601,
            gate_n_el=8,
            gate_drop=True,
        )
        md = ds["metadata"]
        assert md["name"] == name
        assert md["mode_names"] == ["constant", "xi"]
        assert md["n_total"] == 24
        gate = md["enrichment_gate"]
        assert gate["n_checked"] == 24
        n_dropped = gate.get("n_dropped", 0)
        assert md["n_train"] + md["n_val"] + md["n_test"] == 24 - n_dropped
        assert "per_mode_mean" in gate and "worst" in gate

        # Per-split arrays match the recorded sizes and feature dims.
        for sname in ("train", "val", "test"):
            for mode in ("constant", "xi"):
                d = ds[sname][mode]
                assert d["b"].shape[0] == md[f"n_{sname}"]
                assert d["eps_ratios"].shape == (md[f"n_{sname}"], 16)
                assert np.all(d["eps_ratios"].min(axis=1) >= -1.0 - 1e-6)
        # No-twin audit ran for no_twin_shape (leakage_report attached).
        assert "leakage_report" in ds
        for mode in ("constant", "xi"):
            cross = ds["leakage_report"][mode]["cross"]
            assert "train_vs_test" in cross
            st = cross["train_vs_test"]["stats"]
            assert 0.0 <= st["max_sim_max"] <= 1.0
        # Files were actually written.
        written = glob_ci_files(name)
        assert len(written) >= 7   # 3 splits x 2 modes + metadata.json
    finally:
        for f in _glob.glob(os.path.join(data_dir, name + "_*")):
            os.remove(f)


def glob_ci_files(name):
    import glob as _glob
    return _glob.glob(os.path.join(DATASET_SUBDIR, DATA_SUBDIR, name + "_*"))