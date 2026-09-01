"""KANBubble1D / MultiKANBubble1D: envelope normalization, parameter
scaling, eps-profile transforms, batched evaluation, and the loss numerics
used by training (value-only, value+gradient, gradient flow).
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.rfb_bubble import (
    KANBubble1D,
    MultiKANBubble1D,
    _scale_pe_rho,
)


XI = torch.linspace(0, 1, 101)


def _mid(shape=(101,)):
    return shape[0] // 2


# --------------------------------------------------------------------------
# Construction & input scaling
# --------------------------------------------------------------------------

def test_bubble_constructor_parameter_count():
    bubble = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
    n_params = sum(p.numel() for p in bubble.parameters())
    assert n_params == 240


def test_bubble_eps_transform_validation():
    for bad in ("linear2", "", "tanh", 3):
        with pytest.raises((ValueError, TypeError)):
            KANBubble1D(n_hidden=5, eps_transform=bad)
    for ok in ("log", "linear", "none"):
        KANBubble1D(n_hidden=5, eps_transform=ok)  # must not raise


def test_scale_pe_rho_basics():
    pe_s, rho_s = _scale_pe_rho(
        torch.tensor([0.0, 1.0, 100.0, 1e6]),
        torch.tensor([0.0, 0.1, 10.0, 1e6]),
    )
    assert pe_s.shape == (4,)
    assert float(pe_s[0]) == 0.0
    assert torch.isfinite(pe_s).all() and torch.isfinite(rho_s).all()
    # Monotone (after the clamp) — log1p(clamp u)/6 per the implementation.
    assert torch.all(pe_s[1:] >= pe_s[:-1]) and torch.all(rho_s[1:] >= rho_s[:-1])
    assert abs(float(pe_s[1]) - math.log1p(1.0) / 6.0) < 1e-6
    assert abs(float(pe_s[2]) - math.log1p(100.0) / 6.0) < 1e-6
    assert abs(float(pe_s[3]) - math.log1p(1e6) / 6.0) < 1e-6
    assert abs(float(rho_s[3]) - math.log1p(1e6) / 6.0) < 1e-6


# --------------------------------------------------------------------------
# Forward / envelope / normalization invariants
# --------------------------------------------------------------------------

@pytest.fixture()
def bubble():
    torch.manual_seed(0)
    return KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)


def test_forward_shapes_bounds_and_normalization(bubble):
    pe = torch.tensor(100.0)
    rho = torch.tensor(0.0)
    b = bubble(XI, pe, rho)
    assert b.shape == (101,)
    assert (b >= 0).all()
    assert abs(float(b[0])) < 1e-3          # envelope vanishes at x=0
    assert abs(float(b[-1])) < 1e-3         # envelope vanishes at x=1
    assert abs(float(b[50]) - 1.0) < 1e-4   # normalized to 1 at the midpoint


def test_forward_derivative_finite(bubble):
    xi_g = XI.clone().requires_grad_(True)
    b_g = bubble(xi_g, torch.tensor(100.0), torch.tensor(0.0))
    db_g = torch.autograd.grad(b_g.sum(), xi_g, create_graph=True)[0]
    assert db_g.shape == (101,)
    assert torch.isfinite(db_g).all()
    assert torch.isfinite(db_g[[0, -1]]).all()


def test_value_grad_numpy_matches_autograd(bubble):
    xi_np = np.linspace(0, 1, 101)
    b_np, db_np = bubble.value_grad_numpy(xi_np, 100.0, 0.0)
    assert b_np.shape == (101,) and db_np.shape == (101,)
    xi_g = torch.tensor(xi_np, dtype=torch.float32, requires_grad=True)
    b_g = bubble(xi_g, torch.tensor(100.0), torch.tensor(0.0))
    db_g = torch.autograd.grad(b_g.sum(), xi_g)[0]
    assert np.abs(b_np - b_g.detach().numpy()).mean() < 1e-5
    assert np.abs(db_np - db_g.detach().numpy()).mean() < 1e-4


@pytest.mark.parametrize(
    "pe_val,rho_val",
    [(0.5, 0.0), (10.0, 5.0), (1e4, 1e3), (1.0, 10.0), (0.0, 0.0), (1e-7, 1e6)],
)
def test_normalization_across_regimes(bubble, pe_val, rho_val):
    b = bubble(XI, torch.tensor(float(pe_val)), torch.tensor(float(rho_val)))
    assert torch.isfinite(b).all()
    assert abs(float(b[50]) - 1.0) < 1e-4
    assert (b >= 0).all()


def test_batched_forward_matches_single(bubble):
    bs = 8
    pe_exp = torch.tensor([1.0, 10.0, 100.0, 1000.0, 0.5, 5.0, 50.0, 500.0])
    rho_exp = torch.zeros(bs)
    xi_exp = XI.unsqueeze(0).expand(bs, -1).reshape(-1)
    pe_exp_b = pe_exp.unsqueeze(1).expand(-1, 101).reshape(-1)
    rho_exp_b = rho_exp.unsqueeze(1).expand(-1, 101).reshape(-1)
    b_batch = bubble(xi_exp, pe_exp_b, rho_exp_b).reshape(bs, 101)
    assert b_batch.shape == (bs, 101)
    for i in range(bs):
        b_single = bubble(XI, pe_exp[i], rho_exp[i])
        assert (b_batch[i] - b_single).abs().max().item() < 1e-5


# --------------------------------------------------------------------------
# eps-profile (variable diffusion) inputs
# --------------------------------------------------------------------------

def test_bubble_with_eps_ratios_all_transforms(bubble):
    n_pts = 101
    for transform, ratios_ok in (("log", np.exp(np.random.RandomState(0).normal(size=(n_pts, 3)))),
                                 ("linear", np.random.RandomState(0).rand(n_pts, 3)),
                                 ("none", np.random.RandomState(0).rand(n_pts, 3))):
        b = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3, n_eps=3,
                        eps_transform=transform)
        out = b(XI, torch.tensor(10.0), torch.tensor(1.0),
                eps_ratios=torch.tensor(ratios_ok, dtype=torch.float32))
        assert out.shape == (n_pts,)
        assert torch.isfinite(out).all()
        assert abs(float(out[50]) - 1.0) < 1e-4


def test_eps_ratio_transforms_reject_bad_input():
    n = 21
    bad_ratios = {
        "log": np.full((n, 2), -0.5),       # non-positive
        "linear": np.full((n, 2), 1.5),     # > 1
    }
    for transform, ratios in bad_ratios.items():
        b = KANBubble1D(n_hidden=3, n_eps=2, eps_transform=transform)
        with pytest.raises(ValueError):
            b(XI[:n], torch.tensor(10.0), torch.tensor(0.0),
              eps_ratios=torch.tensor(ratios, dtype=torch.float32))


def test_eps_ratio_batch_mismatch_raises():
    b = KANBubble1D(n_hidden=3, n_eps=2)
    with pytest.raises(ValueError):
        b(XI, torch.tensor(10.0), torch.tensor(0.0),
          eps_ratios=torch.tensor(np.ones((5, 2)), dtype=torch.float32))
    with pytest.raises(ValueError):
        b(XI, torch.tensor(10.0), torch.tensor(0.0),
          eps_ratios=torch.tensor([[np.nan, 1.0]] * 101, dtype=torch.float32))


# --------------------------------------------------------------------------
# MultiKANBubble1D
# --------------------------------------------------------------------------

def test_multi_bubble_shapes():
    multi = MultiKANBubble1D(n_bubbles=2)
    b_multi = multi(XI, torch.tensor(100.0), torch.tensor(0.0))
    assert b_multi.shape == (2, 101)
    assert abs(float(b_multi[0, 50]) - 1.0) < 1e-4

    xi_np = np.linspace(0, 1, 101)
    b_np, db_np = multi.value_grad_numpy(xi_np, 100.0, 0.0)
    assert b_np.shape == (2, 101) and db_np.shape == (2, 101)


# --------------------------------------------------------------------------
# Depth / width generalization (hidden_sizes / n_layers)
# --------------------------------------------------------------------------

def _kan_arch(bubble):
    """[(n_in, n_out)] of every KANLayer, in order."""
    return [(m.n_in, m.n_out) for m in bubble.kan]


def _kan_param_count(bubble):
    """Reference count: per layer n_out*n_in*(n_basis + 2) (base + spline +
    scaler weights)."""
    n_basis = bubble.kan[0].n_basis
    return sum(m.n_out * m.n_in * (n_basis + 2) for m in bubble.kan)


def test_hidden_sizes_list_builds_per_layer_widths():
    b = KANBubble1D(n_hidden=999, hidden_sizes=[4, 8, 3])
    assert _kan_arch(b) == [(3, 4), (4, 8), (8, 3), (3, 1)]
    assert b.n_layers == 4
    assert b.hidden_sizes == (4, 8, 3)


def test_hidden_sizes_int_with_n_layers_builds_uniform():
    b = KANBubble1D(hidden_sizes=8, n_layers=4)
    assert _kan_arch(b) == [(3, 8), (8, 8), (8, 8), (8, 1)]
    assert b.n_layers == 4


def test_hidden_sizes_single_matches_legacy_architecture():
    a = KANBubble1D(n_hidden=6, n_grid=8)
    b = KANBubble1D(hidden_sizes=[6], n_grid=8)
    assert _kan_arch(a) == _kan_arch(b) == [(3, 6), (6, 1)]
    # identical init order -> identical weights -> identical outputs
    x = torch.linspace(0, 1, 33)
    torch.manual_seed(7)
    ya = KANBubble1D(n_hidden=6)(x, torch.tensor(10.0), torch.tensor(1.0))
    torch.manual_seed(7)
    yb = KANBubble1D(hidden_sizes=[6])(x, torch.tensor(10.0), torch.tensor(1.0))
    assert torch.equal(ya, yb)


def test_uniform_depth_equals_repeated_hidden_sizes():
    a = KANBubble1D(n_hidden=7, n_layers=4)
    b = KANBubble1D(hidden_sizes=[7, 7, 7])
    assert _kan_arch(a) == _kan_arch(b) == [(3, 7), (7, 7), (7, 7), (7, 1)]
    assert set(a.state_dict()) == set(b.state_dict())


def test_hidden_sizes_takes_precedence_over_uniform():
    b = KANBubble1D(n_hidden=4, n_layers=5, hidden_sizes=[32, 16])
    assert _kan_arch(b) == [(3, 32), (32, 16), (16, 1)]
    assert b.n_layers == 3


def test_hidden_sizes_empty_raises():
    with pytest.raises(ValueError):
        KANBubble1D(hidden_sizes=[])


def test_hidden_sizes_bad_width_raises():
    for bad in ([0], [-2], [4, 0], [-1, 8]):
        with pytest.raises(ValueError):
            KANBubble1D(hidden_sizes=bad)


def test_n_layers_validation():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            KANBubble1D(n_layers=bad)


def test_n_layers_1_single_layer():
    b = KANBubble1D(n_hidden=6, n_layers=1)
    assert _kan_arch(b) == [(3, 1)]
    assert b.n_layers == 1
    assert b.hidden_sizes == ()


def test_hidden_sizes_parameter_count():
    b = KANBubble1D(n_hidden=5, hidden_sizes=[4, 5], n_grid=8)
    assert _kan_param_count(b) == 144 + 240 + 60
    assert _kan_param_count(b) == sum(p.numel() for p in b.parameters())


def test_deep_bubble_forward_normalization_and_grad():
    torch.manual_seed(0)
    b = KANBubble1D(n_eps=2, hidden_sizes=[8, 16], spline_order=3)
    eps = torch.rand(101, 2) * 0.5 + 0.5  # positive, valid for log transform
    out = b(XI, torch.tensor(100.0), torch.tensor(5.0), eps_ratios=eps)
    assert out.shape == (101,)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert abs(float(out[0])) < 1e-3          # envelope vanishes at x=0
    assert abs(float(out[-1])) < 1e-3         # envelope vanishes at x=1
    assert abs(float(out[50]) - 1.0) < 1e-4   # normalized to 1 at the midpoint

    xi_g = XI.clone().requires_grad_(True)
    out_g = b(xi_g, torch.tensor(100.0), torch.tensor(5.0), eps_ratios=eps)
    out_g.sum().backward()
    assert torch.isfinite(xi_g.grad).all()
    assert float(xi_g.grad.abs().sum()) > 0


def test_multi_bubble_hidden_sizes():
    multi = MultiKANBubble1D(n_bubbles=3, hidden_sizes=[4, 6])
    assert len(multi.bubbles) == 3
    assert all(_kan_arch(b) == [(3, 4), (4, 6), (6, 1)] for b in multi.bubbles)
    out = multi(XI, torch.tensor(10.0), torch.tensor(0.0))
    assert out.shape == (3, 101)
    for i in range(3):
        assert abs(float(out[i, 50]) - 1.0) < 1e-4


# --------------------------------------------------------------------------
# Training loss numerics (value-only; value + gradient; gradient flow)
# --------------------------------------------------------------------------

def test_value_loss_numerics(bubble):
    target = bubble(XI, torch.tensor(100.0), torch.tensor(0.0))
    model = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
    pred = model(XI, torch.tensor(100.0), torch.tensor(0.0))
    loss = F.mse_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_value_loss_batched_matches_single(bubble):
    model = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
    target = bubble(XI, torch.tensor(100.0), torch.tensor(0.0))
    pred = model(XI, torch.tensor(100.0), torch.tensor(0.0))
    loss_val = F.mse_loss(pred, target)

    bs = 16
    xi_exp = XI.unsqueeze(0).expand(bs, -1).reshape(-1)
    pe_exp = torch.full((bs, 101), 100.0).reshape(-1)
    rho_exp = torch.zeros(bs * 101)
    targets = target.unsqueeze(0).expand(bs, -1)
    preds = model(xi_exp, pe_exp, rho_exp).reshape(bs, 101)
    loss_batch = F.mse_loss(preds, targets)
    assert abs(float(loss_batch) - float(loss_val)) < 1e-5


def test_value_plus_gradient_loss_and_gradient_flow():
    model = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
    target = model(XI, torch.tensor(100.0), torch.tensor(0.0))

    xi_g = XI.clone().detach().requires_grad_(True)
    pred = model(xi_g, torch.full_like(xi_g, 100.0), torch.zeros_like(xi_g))
    dpred = torch.autograd.grad(pred, xi_g, torch.ones_like(pred),
                                create_graph=True)[0]
    dtarget = torch.tensor(np.gradient(target.detach().numpy(),
                                       float(1.0 / (XI.numel() - 1))),
                           dtype=torch.float32)
    loss = F.mse_loss(pred, target) + 1e-3 * F.mse_loss(dpred, dtarget)
    assert torch.isfinite(loss) and loss.item() > 0

    loss.backward()
    total_norm = sum(p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None)
    assert total_norm > 0