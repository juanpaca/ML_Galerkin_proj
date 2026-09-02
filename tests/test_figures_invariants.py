"""Numeric invariants behind export_bubble_figures.py.

Recomputes bubble positivity/vanishing/normalization plus the enriched
P1 solution quality (rel L2, rel H1) for a fixed 5-piece profile from the
saved darcy_piecewise_5pc dataset, and confirms the enriched solution beats
plain Galerkin. Skipped when the dataset files are absent (CI without data).
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import tutorial_darcy_variable as tutorial
from src.darcy_assembly import (
    assemble_enriched,
    assemble_p1,
    eval_enriched,
    rel_l2,
)
from src.darcy_variable import PiecewiseDiffusion, solve_darcy_1d

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.normpath(os.path.join(HERE, "..", "datasets", "data_darcy_variable"))
FILES = [
    os.path.join(DS, "darcy_piecewise_5pc_test_constant.npz"),
    os.path.join(DS, "darcy_piecewise_5pc_test_xi.npz"),
]
skip = pytest.mark.skipif(
    not all(os.path.exists(f) for f in FILES),
    reason="darcy_piecewise_5pc dataset not present",
)


def _load(target_contrast=30.0):
    dc = dict(np.load(FILES[0]))
    dx = dict(np.load(FILES[1]))
    contrasts = dc["eps_profile"].max(axis=1) / np.maximum(
        dc["eps_profile"].min(axis=1), 1e-30)
    k = int(np.argmin(np.abs(contrasts - target_contrast)))
    return k, dc, dx


def _rel_h1(u_h, u_ref, xi, dx):
    du_h = np.gradient(u_h, dx)
    du_ref = np.gradient(u_ref, dx)
    err = np.sum((u_h - u_ref) ** 2) * dx + np.sum((du_h - du_ref) ** 2) * dx
    norm = np.sum(u_ref ** 2) * dx + np.sum(du_ref ** 2) * dx
    return float(np.sqrt(err) / np.sqrt(norm))


def _profile_from_samples(k, dc):
    """Rebuild the piecewise profile from the sampled eps_profile (jump edges).

    This mirrors export_bubble_figures.py, which reconstructs the profile
    from the stored samples rather than from pool bookkeeping.
    """
    xi = np.asarray(dc["xi"], dtype=float)
    eps = np.asarray(dc["eps_profile"][k], dtype=float)
    step = xi[1] - xi[0]
    edge_ref = (eps[:-1] + eps[1:]) / 2.0
    jumps = np.abs(np.diff(eps)) > 1e-4 * edge_ref + 1e-12
    interior = xi[1:][jumps]
    edges = np.unique(np.concatenate(([0.0], interior, [1.0])))
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.array([np.median(eps[(xi >= a) & (xi <= b)])
                       for a, b in zip(edges[:-1], edges[1:])])
    profile = PiecewiseDiffusion(edges, values)
    assert np.allclose(profile.evaluate(xi), eps, rtol=1e-3, atol=1e-4)
    return profile


@skip
def test_bubble_invariants_from_dataset():
    k, dc, _ = _load()
    b_hat = dc["b"][k]
    xi = dc["xi"]
    assert (b_hat >= 0.0).all()
    assert b_hat[0] < 1e-3 and b_hat[-1] < 1e-3
    mid = int(np.argmin(np.abs(xi - 0.5)))
    assert abs(b_hat[mid] - 1.0) < 1e-3
    assert (dc["eps_profile"][k] > 0).all()


@skip
def test_enriched_beats_galerkin_and_matches_expected_quality():
    k, dc, dx_ = _load()
    profile = _profile_from_samples(k, dc)
    xi = dc["xi"]
    dx = float(xi[1] - xi[0])
    source = lambda x: 1.0 + np.asarray(x, dtype=float)
    n_el = 8
    mesh = np.linspace(0.0, 1.0, n_el + 1)

    n_ref = 4001
    ref = solve_darcy_1d(profile, length=1.0, source=source, n_points=n_ref)
    u_ref = ref["u"]

    bubbles = np.stack([dc["b"][k], dx_["b"][k]]).astype(float)

    u_nodes, coeffs = assemble_enriched(mesh, profile, source, bubbles, xi, dx)
    u_en = eval_enriched(u_nodes, coeffs, bubbles, mesh, xi)
    u_en_ref = np.interp(ref["x"], xi, u_en)

    A_LL, F_L, free = assemble_p1(mesh, profile, source)
    u_gal = np.zeros(n_el + 1)
    u_gal[free] = np.linalg.solve(A_LL, F_L)
    u_gal_ref = np.interp(ref["x"], mesh, u_gal)

    l2_en = float(rel_l2(u_en_ref, u_ref, ref["x"]))
    l2_gal = float(rel_l2(u_gal_ref, u_ref, ref["x"]))
    h1_en = _rel_h1(u_en_ref, u_ref, ref["x"], ref["x"][1] - ref["x"][0])

    # Expected quality band from the figure export (rel L2 ~1e-3, H1 ~2e-2).
    assert l2_en < 5e-3
    assert h1_en < 5e-2
    # Enrichment genuinely recovers the boundary-layer solution.
    assert l2_gal > 5.0 * l2_en
    assert coeffs.shape == (2,)


# --------------------------------------------------------------------------
# CUDA/NVML flake: bubble forward fallback to CPU
# --------------------------------------------------------------------------

class _FakeParam:
    """Mimics a CUDA parameter for the device checks in _bubble_call."""
    is_cuda = True
    dtype = torch.float32
    device = torch.device("cpu")


class _FlakyBubble(nn.Module):
    """Raises the allocator's NVML assert until moved to CPU, then succeeds."""

    def __init__(self):
        super().__init__()
        self._moved = False

    def parameters(self, recurse=True):
        return iter([_FakeParam()])

    def to(self, device):
        self._moved = True
        return self

    def forward(self, xi, pe, rho, eps_ratios=None):
        if not self._moved:
            raise RuntimeError(
                "NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() "
                "INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:963")
        return xi


class _BadBubble(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("unrelated failure")


@pytest.fixture()
def restore_gpu_eval_flag():
    yield
    tutorial._USE_GPU_EVAL = True


def test_bubble_call_nvml_falls_back_to_cpu(restore_gpu_eval_flag):
    tutorial._USE_GPU_EVAL = True
    out = tutorial._bubble_call(
        _FlakyBubble(), torch.arange(5.0), torch.zeros(1), torch.zeros(1))
    assert out.numel() == 5
    assert tutorial._USE_GPU_EVAL is False


def test_bubble_call_unrelated_error_not_swallowed(restore_gpu_eval_flag):
    tutorial._USE_GPU_EVAL = True
    with pytest.raises(RuntimeError, match="unrelated failure"):
        tutorial._bubble_call(
            _BadBubble(), torch.zeros(2), torch.zeros(1), torch.zeros(1))


def test_evaluate_split_deep_model_normal_path(restore_gpu_eval_flag):
    from src.rfb_bubble import MultiKANBubble1D
    tutorial._USE_GPU_EVAL = True
    model = MultiKANBubble1D(
        n_bubbles=2, hidden_sizes=[32, 64, 128, 64, 32],
        n_eps=16, eps_transform="none")
    fake = {
        "xi": np.linspace(0, 1, 129),
        "eps_ratios": np.random.RandomState(0).rand(6, 16).astype(np.float32),
        "b": np.random.RandomState(1).rand(6, 129).astype(np.float32),
    }
    rmse, rel = tutorial.evaluate_split(model, fake, 0, chunk_size=2)
    assert rmse.shape == (6,) and rel.shape == (6,)
    assert np.isfinite(rel).all()