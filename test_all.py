#!/usr/bin/env python3
"""Comprehensive unit tests for all RFB modules."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

np.set_printoptions(precision=4, suppress=True, linewidth=120)
torch.set_printoptions(precision=4, sci_mode=False)

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}: {detail}")

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol

# =========================================================================
# 1. KAN1D — single learnable edge function
# =========================================================================
print("\n" + "=" * 60)
print("1. KAN1D (src/kan.py)")
print("=" * 60)

from src.kan import KAN1D, _eval_bspline_basis as _eval_bspline_internal, _extend_knots

# --- 1.1 Initialization ---
k = KAN1D(n_grid=8, k=3, x_min=-1.0, x_max=1.0)
check("init: n_basis = 8+3-1 = 10", k.n_basis == 10)
check("init: knots len = n_grid+1+2*(k-1) = 8+1+4=13", k.knots.shape[0] == 13)
check("init: knots[0]=-1", approx(k.knots[0].item(), -1.0))
check("init: knots[-1]=1", approx(k.knots[-1].item(), 1.0))
check("init: w_b Xavier", k.w_b.shape == (1,))
check("init: w_s=1.0", approx(k.w_s.item(), 1.0))
check("init: c std≈0.1", abs(k.c.std().item() - 0.1) < 0.05)

# --- 1.2 Forward on grid ---
x = torch.linspace(-1.0, 1.0, 21)
y = k(x)
check("forward: output shape matches input", y.shape == (21,))
check("forward: finite values", torch.isfinite(y).all())
check("forward: SiLU base + spline", True)

# --- 1.3 Forward with derivative ---
x2 = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
y2 = k(x2)
dydx = torch.autograd.grad(y2.sum(), x2, create_graph=True)[0]
check("forward/grad: gradient shape", dydx.shape == (3,))
check("forward/grad: d(silu)/dx = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))", True)

# --- 1.4 Finite difference check at x=0 ---
x_fd = torch.tensor([0.0])
k.eval()
with torch.no_grad():
    y0 = k(torch.tensor([0.0])).item()
    y_eps = k(torch.tensor([1e-5])).item()
    fd_grad = (y_eps - y0) / 1e-5
x_fd_grad = torch.tensor([0.0], requires_grad=True)
y_fd = k(x_fd_grad)
auto_grad = torch.autograd.grad(y_fd, x_fd_grad, torch.ones_like(y_fd), create_graph=False)[0].item()
check("forward/grad: FD ≈ autograd", approx(auto_grad, fd_grad, tol=5e-3),
      f"auto={auto_grad:.4f}, fd={fd_grad:.4f}")

# --- 1.5 forward_with_deriv ---
g, dg = k.forward_with_deriv(torch.tensor([0.3, 0.7]))
check("forward_with_deriv: shapes", g.shape == (2,) and dg.shape == (2,))

# --- 1.6 Batched input ---
x_batch = torch.rand(16, 1)
y_batch = k(x_batch)
check("batched 2D input shape", y_batch.shape == (16,))

# =========================================================================
# 2. KANLayer — KAN layer aggregation
# =========================================================================
print("\n" + "=" * 60)
print("2. KANLayer (src/rfb_bubble.py)")
print("=" * 60)

from src.rfb_bubble import KANLayer

# --- 2.1 3→5 layer ---
layer_35 = KANLayer(n_in=3, n_out=5, n_grid=8, k=3)
x_in = torch.rand(20, 3)
y_out = layer_35(x_in)
check("3→5: output shape", y_out.shape == (20, 5))
check("3→5: finite", torch.isfinite(y_out).all())

# --- 2.2 5→1 layer ---
layer_51 = KANLayer(n_in=5, n_out=1, n_grid=8, k=3)
x_in2 = torch.rand(20, 5)
y_out2 = layer_51(x_in2)
check("5→1: output shape", y_out2.shape == (20, 1))
check("5→1: single entry = sum of 5 edge functions", True)

# --- 2.3 Sequential: same as KANBubble1D's kan ---
layer1 = KANLayer(3, 5)
layer2 = KANLayer(5, 1)
x = torch.rand(10, 3)
y = layer2(layer1(x))
check("sequential 3→5→1: shape", y.shape == (10, 1))

# --- 2.4 Gradient w.r.t. input ---
x_g = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
y_g = layer2(layer1(x_g))
dy = torch.autograd.grad(y_g.sum(), x_g, create_graph=True)[0]
check("3→5→1: gradients exist", dy.shape == (1, 3) and torch.isfinite(dy).all())

# =========================================================================
# 3. KANBubble1D — complete bubble model
# =========================================================================
print("\n" + "=" * 60)
print("3. KANBubble1D (src/rfb_bubble.py)")
print("=" * 60)

from src.rfb_bubble import KANBubble1D, _scale_pe_rho

# --- 3.1 Constructor ---
bubble = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
n_params = sum(p.numel() for p in bubble.parameters())
check("bubble: 240 params", n_params == 240, f"got {n_params}")

# --- 3.2 _scale_pe_rho ---
pe_s, rho_s = _scale_pe_rho(
    torch.tensor([0.0, 1.0, 100.0, 1e6]),
    torch.tensor([0.0, 0.1, 10.0, 1e6])
)
check("scale: shape", pe_s.shape == (4,))
check("scale: pe=0 → log1p(0)/6 = 0", approx(pe_s[0].item(), 0.0))
check("scale: finite", torch.isfinite(pe_s).all())

# --- 3.3 Single sample forward ---
xi = torch.linspace(0, 1, 101)
pe = torch.tensor(100.0)
rho = torch.tensor(0.0)
b = bubble(xi, pe, rho)
check("forward: shape (101,)", b.shape == (101,))
check("forward: b ≥ 0 (softplus+delta)", (b >= 0).all())
check("forward: b(0) ≈ 0 (envelope)", approx(b[0].item(), 0.0, tol=1e-3))
check("forward: b(1) ≈ 0 (envelope)", approx(b[-1].item(), 0.0, tol=1e-3))
check("forward: b(0.5) ≈ 1 (normalization)", approx(b[50].item(), 1.0, tol=1e-1),
      f"got {b[50].item():.4f}")

# --- 3.4 derivative ---
xi_g = torch.linspace(0, 1, 101, requires_grad=True)
b_g = bubble(xi_g, pe, rho)
db_g = torch.autograd.grad(b_g.sum(), xi_g, create_graph=True)[0]
check("derivative: shape", db_g.shape == (101,))
check("derivative: finite", torch.isfinite(db_g).all())
# Note: b' at boundaries = 4*ratio (not 0), check it's finite
check("derivative: finite at boundaries", torch.isfinite(db_g[[0, -1]]).all(),
      f"db[0]={db_g[0].item():.4f}, db[-1]={db_g[-1].item():.4f}")

# --- 3.5 value_grad_numpy ---
xi_np = np.linspace(0, 1, 101)
b_np, db_np = bubble.value_grad_numpy(xi_np, 100.0, 0.0)
check("value_grad_numpy: shapes", b_np.shape == (101,) and db_np.shape == (101,))
check("value_grad_numpy: b ≈ b_g", approx(np.mean(np.abs(b_np - b_g.detach().numpy())), 0.0, tol=1e-5))

# --- 3.6 Multiple parameter regimes ---
for pe_val, rho_val in [(0.5, 0.0), (10.0, 5.0), (1e4, 1e3), (1.0, 10.0)]:
    b_m = bubble(xi, torch.tensor(float(pe_val)), torch.tensor(float(rho_val)))
    check(f"forward (pe={pe_val:.1e}, rho={rho_val:.1e}): b(0.5)≈1",
          approx(b_m[50].item(), 1.0, tol=1e-1),
          f"got {b_m[50].item():.4f}")

# --- 3.7 Batched forward ---
bs = 8
xi_exp = xi.unsqueeze(0).expand(bs, -1).reshape(-1)
pe_exp = torch.tensor([1.0, 10.0, 100.0, 1000.0, 0.5, 5.0, 50.0, 500.0], dtype=torch.float32)
rho_exp = torch.zeros(bs, dtype=torch.float32)
pe_exp_b = pe_exp.unsqueeze(1).expand(-1, 101).reshape(-1)
rho_exp_b = rho_exp.unsqueeze(1).expand(-1, 101).reshape(-1)
b_batch = bubble(xi_exp, pe_exp_b, rho_exp_b).reshape(bs, 101)
check("batched: shape", b_batch.shape == (8, 101))
for i in range(bs):
    b_single = bubble(xi, pe_exp[i], rho_exp[i])
    diff = (b_batch[i] - b_single).abs().max().item()
    check(f"batched: sample {i} matches single", diff < 1e-5, f"max diff={diff:.2e}")

# =========================================================================
# 3b. MultiKANBubble1D
# =========================================================================
print("\n" + "=" * 60)
print("3b. MultiKANBubble1D")
print("=" * 60)

from src.rfb_bubble import MultiKANBubble1D

multi = MultiKANBubble1D(n_bubbles=2)
b_multi = multi(xi, pe, rho)
check("multi: shape (n_bubbles, n_pts)", b_multi.shape == (2, 101))
check("multi: b[0] ≈ 1 at mid", approx(b_multi[0, 50].item(), 1.0, tol=1e-1))

b_np_m, db_np_m = multi.value_grad_numpy(xi_np, 100.0, 0.0)
check("multi value_grad_numpy: shapes", b_np_m.shape == (2, 101) and db_np_m.shape == (2, 101))

# =========================================================================
# 4. FD solver & local parameters
# =========================================================================
print("\n" + "=" * 60)
print("4. FD solver (src/rfb_local.py)")
print("=" * 60)

from src.rfb_local import solve_reference_rfb, local_parameters

# --- 4.1 local_parameters ---
pe, rho = local_parameters(eps=0.01, beta=1.0, sigma=0.0, h=1/16)
check("local_parameters: pe = βh/(2ε) = 1/(32*0.01) ≈ 3.125",
      approx(pe, 3.125, tol=1e-3), f"got {pe:.4f}")
check("local_parameters: ρ = σh²/ε = 0",
      approx(rho, 0.0, tol=1e-10))

pe2, rho2 = local_parameters(eps=1e-4, beta=1.0, sigma=1.0, h=1/16)
check("local_parameters: pe = 1/(32*1e-4) ≈ 312.5",
      approx(pe2, 312.5, tol=1e-1), f"got {pe2:.1f}")

# --- 4.2 FD solve: advection-dominated (ε=1e-4, β=1, σ=0) ---
sol = solve_reference_rfb(eps=1e-4, beta=1.0, sigma=0.0, h=1/16,
                           residual_mode="constant", n_points=200)
check("FD: constant mode keys", all(k in sol for k in ["xi", "b", "db", "params"]))
check("FD: xi ∈ [0,1]", approx(sol["xi"][0], 0.0) and approx(sol["xi"][-1], 1.0))
check("FD: b ≥ 0", (sol["b"] >= 0).all())
check("FD: b(0) ≈ 0", approx(sol["b"][0], 0.0, tol=1e-3))
check("FD: b(1) ≈ 0", approx(sol["b"][-1], 0.0, tol=1e-3))
mid_idx = int(np.argmin(np.abs(sol["xi"] - 0.5)))
check("FD: normalized b(0.5) ≈ 1", approx(sol["b"][mid_idx], 1.0, tol=1e-2))

# --- 4.3 FD solve: xi mode ---
sol_xi = solve_reference_rfb(eps=1e-4, beta=1.0, sigma=0.0, h=1/16,
                              residual_mode="xi", n_points=200)
check("FD xi: shape matches", sol_xi["b"].shape == sol["b"].shape)
mid_xi = int(np.argmin(np.abs(sol_xi["xi"] - 0.5)))
check("FD xi: normalized", approx(sol_xi["b"][mid_xi], 1.0, tol=1e-2))

# --- 4.4 Reaction case (σ > 0) ---
sol_r = solve_reference_rfb(eps=1e-2, beta=1.0, sigma=5.0, h=1/16,
                             residual_mode="constant", n_points=200)
check("FD reaction: finite", np.isfinite(sol_r["b"]).all())
mid_r = int(np.argmin(np.abs(sol_r["xi"] - 0.5)))
check("FD reaction: normalised", approx(sol_r["b"][mid_r], 1.0, tol=1e-2))

# --- 4.5 σ=0 (pure advection-diffusion) ---
sol_ad = solve_reference_rfb(eps=1e-3, beta=1.0, sigma=0.0, h=1/16,
                              residual_mode="constant", n_points=200)
check("FD advection: finite", np.isfinite(sol_ad["b"]).all())

# =========================================================================
# 5. Dataset: cell-based split
# =========================================================================
print("\n" + "=" * 60)
print("5. Dataset: cell-based split")
print("=" * 60)

from src.dataset_generation import (
    _pe_rho_cell, cell_based_split, load_dataset,
)

# --- 5.1 _pe_rho_cell ---
pe_test = np.array([0.5, 5.0, 50.0, 500.0, 5000.0])
rho_test = np.array([0.0, 0.5, 5.0, 50.0, 500.0])
pe_idx, rho_idx = _pe_rho_cell(pe_test, rho_test)
check("_pe_rho_cell: shapes", pe_idx.shape == (5,) and rho_idx.shape == (5,))
# Pe bins: (0,1,10,100,1000,inf) → pe=0.5→0, pe=5→1, pe=50→2, pe=500→3, pe=5000→4
# Rho bins: (-inf,0,1,100,inf) → ρ=0→1, ρ=0.5→1, ρ=5→2, ρ=50→2, ρ=500→3
expected_pe = [0, 1, 2, 3, 4]
expected_rho = [1, 1, 2, 2, 3]
for i in range(5):
    check(f"  sample {i}: pe_idx={pe_idx[i]}, rho_idx={rho_idx[i]}",
          pe_idx[i] == expected_pe[i] and rho_idx[i] == expected_rho[i])

# --- 5.2 cell_based_split with custom bins ---
np.random.seed(42)
n = 500
pe_data = 10.0 ** np.random.uniform(-0.5, 4.5, n)
rho_data = np.abs(np.random.randn(n) * 10)
rho_bins = (-np.inf, 0, 0.1, 1, 10, 100, np.inf)

train_idx, val_idx, test_idx, cell_map = cell_based_split(
    pe_data, rho_data, n_val_cells=3, n_test_cells=3,
    rho_bins=rho_bins, seed=123
)

train_cells = {c for c, (s, _) in cell_map.items() if s == "train"}
val_cells = {c for c, (s, _) in cell_map.items() if s == "val"}
test_cells = {c for c, (s, _) in cell_map.items() if s == "test"}

check("cell_based_split: no cell overlap",
      train_cells.isdisjoint(val_cells) and train_cells.isdisjoint(test_cells))
check("cell_based_split: val has cells", len(val_cells) >= 2)
check("cell_based_split: test has cells", len(test_cells) >= 2)
check("cell_based_split: no sample overlap",
      set(train_idx).isdisjoint(set(val_idx)) and
      set(train_idx).isdisjoint(set(test_idx)))
check("cell_based_split: total samples match",
      len(train_idx) + len(val_idx) + len(test_idx) == n,
      f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, total={len(train_idx)+len(val_idx)+len(test_idx)}")

# --- 5.3 Load existing dataset ---
ds = load_dataset("rfb_5k_frame")
check("load_dataset: has train/val/test",
      "train" in ds and "val" in ds and "test" in ds)
check("load_dataset: constant mode", "constant" in ds["train"])
check("load_dataset: xi mode", "xi" in ds["train"])
check("load_dataset: train data shapes",
      ds["train"]["constant"]["pe"].ndim == 1 and
      ds["train"]["constant"]["b"].ndim == 2)
check("load_dataset: n_total > 0",
      ds["metadata"]["n_total"] > 0,
      f"got {ds['metadata']['n_total']}")

# --- 5.4 Split-index integrity ---
split_indices = ds["metadata"].get("split_indices", {})
check("dataset: split_indices exists", len(split_indices) == 3,
      f"got {list(split_indices)}")
all_indices = set()
for sname in ("train", "val", "test"):
    if sname in split_indices:
        all_indices.update(int(i) for i in split_indices[sname])
n_tot = ds["metadata"]["n_total"]
check("dataset: all indices 0..n_tot-1",
      all_indices == set(range(n_tot)),
      f"got min={min(all_indices)}, max={max(all_indices)}, len={len(all_indices)}")

# =========================================================================
# 6. Training — loss computations
# =========================================================================
print("\n" + "=" * 60)
print("6. Training — loss sanity checks")
print("=" * 60)

# --- 6.1 Value-only loss ---
model_t = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
xi_t = torch.linspace(0, 1, 101)
pe_t = torch.tensor(100.0)
rho_t = torch.tensor(0.0)
# Generate target using the model's own prediction (just to test loss numerics)
target = bubble(xi_t, pe_t, rho_t)
pred = model_t(xi_t, pe_t, rho_t)
loss_val = F.mse_loss(pred, target)
check("training: value loss finite", torch.isfinite(loss_val).all())
check("training: initial value loss > 0", loss_val.item() > 0,
      f"loss={loss_val.item():.4e}")

# --- 6.2 Batched value-only loss ---
bs = 16
xi_exp = xi_t.unsqueeze(0).expand(bs, -1).reshape(-1)
pe_exp = torch.full((bs, 101), 100.0, dtype=torch.float32).reshape(-1)
rho_exp = torch.zeros(bs * 101)
targets = target.unsqueeze(0).expand(bs, -1)
preds = model_t(xi_exp, pe_exp, rho_exp).reshape(bs, 101)
loss_batch = F.mse_loss(preds, targets)
check("training: batched value loss ≈ single",
      approx(loss_batch.item(), loss_val.item(), tol=1e-5),
      f"batch={loss_batch.item():.4e}, single={loss_val.item():.4e}")

# --- 6.3 Value + gradient loss (with create_graph=True) ---
xi_g = xi_t.clone().detach().requires_grad_(True)
pe_g = torch.full_like(xi_g, 100.0)
rho_g = torch.zeros_like(xi_g)
pred_g = model_t(xi_g, pe_g, rho_g)
dpred_g = torch.autograd.grad(pred_g, xi_g, torch.ones_like(pred_g), create_graph=True)[0]

# Compute target gradient via FD
target_np = target.detach().numpy()
dx = 1.0 / (101 - 1)
dtarget_np = np.gradient(target_np, dx)

loss_val_grad = F.mse_loss(pred_g, target) + 1e-3 * F.mse_loss(dpred_g, torch.tensor(dtarget_np, dtype=torch.float32))
check("training: value+grad loss finite", torch.isfinite(loss_val_grad))
check("training: value+grad > 0", loss_val_grad.item() > 0)

# --- 6.4 Loss scale check for real data ---
print("\n--- 6.4 Real data loss scale ---")
ds_train = ds["train"]["constant"]
xi_full = ds_train["xi"]
xi_s = torch.tensor(np.linspace(0, 1, 100), dtype=torch.float32)
b_target_s = torch.tensor(np.interp(xi_s.numpy(), xi_full, ds_train["b"][0]), dtype=torch.float32)
db_target_s = torch.tensor(np.interp(xi_s.numpy(), xi_full, ds_train["db"][0]), dtype=torch.float32)

pe_r = torch.tensor(ds_train["pe"][0], dtype=torch.float32)
rho_r = torch.tensor(ds_train["rho"][0], dtype=torch.float32)

# Evaluate untrained model
model0 = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
xi_eval = xi_s.clone().requires_grad_(True)
pe_eval = torch.full_like(xi_eval, pe_r)
rho_eval = torch.full_like(xi_eval, rho_r)
pred0 = model0(xi_eval, pe_eval, rho_eval)
dpred0 = torch.autograd.grad(pred0, xi_eval, torch.ones_like(pred0), create_graph=True)[0]

v0 = F.mse_loss(pred0, b_target_s).item()
g0 = F.mse_loss(dpred0, db_target_s).item()
total0 = v0 + 1e-3 * g0
check("loss scale: value MSE", 1e-4 < v0 < 10.0, f"v0={v0:.4e}")
check("loss scale: grad MSE ≈ 0.01-1e5", 0.01 < g0 < 1e5, f"g0={g0:.4e}")
check("loss scale: total", 1e-4 < total0 < 10.0, f"total={total0:.4e}")
print(f"  Initial losses: value={v0:.4e}, grad={g0:.4e}, total={total0:.4e}")

# --- 6.5 Batched loss for a subset of real data ---
print("\n--- 6.5 Batched real data loss ---")
n_sample = 32
n_train_actual = len(ds_train["pe"])
batch_idx = np.random.choice(min(n_train_actual, 3278), n_sample, replace=False)
pe_b = torch.tensor(ds_train["pe"][batch_idx], dtype=torch.float32)
rho_b = torch.tensor(ds_train["rho"][batch_idx], dtype=torch.float32)
b_b = torch.tensor(np.array([np.interp(xi_s.numpy(), xi_full, ds_train["b"][i]) for i in batch_idx]), dtype=torch.float32)
db_b = torch.tensor(np.array([np.interp(xi_s.numpy(), xi_full, ds_train["db"][i]) for i in batch_idx]), dtype=torch.float32)

n_pts = 100
xi_exp_b = xi_s.unsqueeze(0).expand(n_sample, -1).reshape(-1).requires_grad_(True)
pe_exp_b = pe_b.unsqueeze(1).expand(-1, n_pts).reshape(-1)
rho_exp_b = rho_b.unsqueeze(1).expand(-1, n_pts).reshape(-1)

pred_b = model0(xi_exp_b, pe_exp_b, rho_exp_b)
dpred_b = torch.autograd.grad(pred_b, xi_exp_b, torch.ones_like(pred_b), create_graph=True)[0]
pred_b = pred_b.reshape(n_sample, n_pts)
dpred_b = dpred_b.reshape(n_sample, n_pts)

loss_batch_real = F.mse_loss(pred_b, b_b) + 1e-3 * F.mse_loss(dpred_b, db_b)
check("training: batched real loss finite", torch.isfinite(loss_batch_real))
check("training: batched real loss > 0", loss_batch_real.item() > 0)
print(f"  Batched real loss: {loss_batch_real.item():.4e}")

# --- 6.6 Gradients flow check — one step of SGD ---
loss_batch_real.backward()
total_grad_norm = sum(p.grad.norm().item() for p in model0.parameters() if p.grad is not None)
check("training: gradients flow", total_grad_norm > 0,
      f"total_grad_norm={total_grad_norm:.4e}")

# =========================================================================
# 7. Exact RFB bubble (ground truth comparison)
# =========================================================================
print("\n" + "=" * 60)
print("7. ExactRFBubble1D (src/rfb_exact.py)")
print("=" * 60)

from src.rfb_exact import ExactRFBubble1D

# --- 7.1 Constructor ---
exact = ExactRFBubble1D(eps=0.01, beta=1.0, sigma=0.0, h=1/16)
xi_e = np.linspace(0, 1, 101)
b_e, db_e = exact.value_grad_numpy(xi_e)
check("exact: b ≥ 0", (b_e >= 0).all())
check("exact: b(0) ≈ 0", approx(b_e[0], 0.0, tol=1e-3))
check("exact: b(1) ≈ 0", approx(b_e[-1], 0.0, tol=1e-3))
mid_ex = int(np.argmin(np.abs(xi_e - 0.5)))
check("exact: b(0.5) ≈ 1", approx(b_e[mid_ex], 1.0, tol=1e-2))

# =========================================================================
# 8. Bubble similarity / redundancy analysis
# =========================================================================
print("\n" + "=" * 60)
print("8. Bubble similarity / redundancy analysis")
print("=" * 60)

from src.dataset_generation import (
    bubble_gram_matrix, bubble_cosine_similarity,
    _off_diagonal_stats, _effective_rank, bubble_similarity_analysis,
)

xi_sim = np.linspace(0, 1, 101)

# --- 8.1 Gram matrix matches analytic H1 inner product (lambda=0.2) ---
b_parab = 4.0 * xi_sim * (1.0 - xi_sim)
lam = 0.2
G_parab = bubble_gram_matrix(np.vstack([b_parab]), xi_sim, lambda_deriv=lam)
expected = 16.0 / 30.0 + lam**2 * (16.0 / 3.0)  # int b^2 + lam^2 * int (b')^2
check("similarity: Gram diag (H1) for 4x(1-x)", abs(G_parab[0, 0] - expected) / expected < 1e-3,
      f"got {G_parab[0,0]:.6f}, expected {expected:.6f}")
G_l2 = bubble_gram_matrix(np.vstack([b_parab]), xi_sim, lambda_deriv=0.0)
check("similarity: Gram diag (L2) for 4x(1-x)", abs(G_l2[0, 0] - 16.0 / 30.0) / (16.0 / 30.0) < 1e-3,
      f"got {G_l2[0,0]:.6f}, expected {16.0/30.0:.6f}")

# --- 8.2 Cosine similarity: identical, scaled, orthogonal ---
B_id = np.vstack([b_parab, b_parab])
C_id = bubble_cosine_similarity(B_id, xi_sim)
check("similarity: identical bubbles → C=1", abs(C_id[0, 1] - 1.0) < 1e-12)
B_scale = np.vstack([b_parab, 3.0 * b_parab])
C_scale = bubble_cosine_similarity(B_scale, xi_sim)
check("similarity: scalar multiple → C=1", abs(C_scale[0, 1] - 1.0) < 1e-12)
b_odd = np.sin(2 * np.pi * xi_sim)  # odd about 0.5 → orthogonal to even parabola
B_orth = np.vstack([b_odd, b_parab])
C_orth = bubble_cosine_similarity(B_orth, xi_sim)
check("similarity: orthogonal bubbles → C≈0", abs(C_orth[0, 1]) < 1e-2,
      f"got {C_orth[0,1]:.4e}")

# --- 8.3 Off-diagonal stats ---
stats = _off_diagonal_stats(C_id)
check("similarity: off-diag stats n_pairs",
      stats["n_pairs"] == 1 and stats["mean"] > 0.999)
check("similarity: off-diag stats keys",
      set(("mean", "median", "std", "min", "max", "frac_gt_0.90",
           "frac_gt_0.95", "frac_gt_0.99", "n_pairs")) <= set(stats))

# --- 8.4 Effective rank ---
B_rank1 = np.vstack([b_parab * k for k in range(1, 8)])
C_rank1 = bubble_cosine_similarity(B_rank1, xi_sim)
er1 = _effective_rank(C_rank1)
check("similarity: rank-1 set → eff_rank≈1", er1["effective_rank"] < 1.5,
      f"got {er1['effective_rank']:.3f}")
check("similarity: rank-1 set → n_95=1", er1["n_for_95pct_energy"] == 1,
      f"got {er1['n_for_95pct_energy']}")
B_sines = np.vstack([np.sin((k + 1) * np.pi * xi_sim) for k in range(6)])
C_sines = bubble_cosine_similarity(B_sines, xi_sim)
er6 = _effective_rank(C_sines)
check("similarity: 6 distinct shapes → eff_rank>3", er6["effective_rank"] > 3.0,
      f"got {er6['effective_rank']:.3f}")

# --- 8.5 Full analysis on synthetic dataset (structure + speed) ---
rng = np.random.default_rng(0)
xi_data = np.linspace(0, 1, 100)
def _synth_bubble(n, spread=0.3):
    out = []
    for _ in range(n):
        pe_r = 10.0 ** rng.uniform(-0.5, 2.0)
        rho_r = 10.0 ** rng.uniform(-0.5, 2.0)
        sig = rng.uniform(spread, 2.0)
        cx = rng.uniform(0.5 - 0.4 * sig, 0.5 + 0.4 * sig)
        b = np.exp(-((xi_data - cx) ** 2) / (2 * sig ** 2))
        b *= np.sin(np.pi * xi_data)  # vanish at boundaries
        b /= b[np.argmin(abs(xi_data - 0.5))]
        out.append(b)
    return np.array(out)
synth = {
    "train": {"constant": {"pe": rng.uniform(1, 100, 60), "rho": rng.uniform(0, 100, 60),
                           "b": _synth_bubble(60), "xi": xi_data}},
    "val":   {"constant": {"pe": rng.uniform(1, 100, 30), "rho": rng.uniform(0, 100, 30),
                           "b": _synth_bubble(30), "xi": xi_data}},
    "test":  {"constant": {"pe": rng.uniform(1, 100, 30), "rho": rng.uniform(0, 100, 30),
                           "b": _synth_bubble(30), "xi": xi_data}},
}
res_syn = bubble_similarity_analysis(synth, mode="constant", splits=("train", "val", "test"),
                                     verbose=False, return_matrices=True)
check("similarity: analysis has within/cross",
      set(("within", "cross")) <= set(res_syn))
check("similarity: within-split stats present",
      set(("off_diagonal", "effective_rank")) <= set(res_syn["within"]["train"]))
check("similarity: cross-split entries present",
      "train_vs_val" in res_syn["cross"] and "train_vs_test" in res_syn["cross"])
check("similarity: cross-split max-sim in [0,1]",
      0.0 <= res_syn["cross"]["train_vs_val"]["stats"]["max_sim_mean"] <= 1.0)
check("similarity: C matrices returned",
      res_syn["within"]["train"]["C"].shape == (60, 60))

# --- 8.6 Real dataset redundancy snapshot ---
res_real = bubble_similarity_analysis(ds, mode="constant", splits=("train",),
                                      n_eig_samples=300, verbose=False)
er_real = res_real["within"]["train"]["effective_rank"]
od_real = res_real["within"]["train"]["off_diagonal"]
check("similarity: real train eff_rank small (redundant)",
      er_real["effective_rank"] < 5.0, f"got {er_real['effective_rank']:.3f}")
check("similarity: real train off-diag mean in [0,1]",
      0.0 <= od_real["mean"] <= 1.0, f"got {od_real['mean']:.4f}")

# =========================================================================
# 9. Analytic RFB bubble (src/rfb_analytic.py)
# =========================================================================
print("\n" + "=" * 60)
print("9. Analytic RFB bubble (src/rfb_analytic.py)")
print("=" * 60)

from src.rfb_analytic import exact_rfb, fd_rfb, fd_error_metrics

# --- 9.1 Normalization, BCs, positivity at low Pe ---
for mode_a in ("constant", "xi"):
    ex = exact_rfb(1.0, 1.0, mode_a, xi=np.linspace(0.0, 1.0, 401))
    check(f"analytic[{mode_a}]: b(0) ≈ 0", abs(ex["b"][0]) < 1e-12)
    check(f"analytic[{mode_a}]: b(1) ≈ 0", abs(ex["b"][-1]) < 1e-12)
    check(f"analytic[{mode_a}]: normalized b(0.5) = 1",
          abs(float(np.interp(0.5, ex["xi"], ex["b_norm"])) - 1.0) < 1e-9)
    check(f"analytic[{mode_a}]: bubble ≥ 0", (ex["b"] >= 0).all())

# --- 9.2 Residual of the ODE (low Pe, normalized bubble) ---
ex = exact_rfb(1.0, 1.0, "constant", xi=np.linspace(0.0, 1.0, 2001))
d2 = np.gradient(np.gradient(ex["b_norm"], ex["xi"][1] - ex["xi"][0]),
                 ex["xi"][1] - ex["xi"][0])
res = -d2 + 2.0 * 1.0 * np.gradient(ex["b_norm"], ex["xi"][1] - ex["xi"][0]) \
      + 1.0 * ex["b_norm"] - 1.0 / ex["center"]
check("analytic: normalized bubble satisfies ODE (residual ~ 0)",
      np.max(np.abs(res[50:-50])) < 1e-3,
      f"max|res| = {np.max(np.abs(res[50:-50])):.2e}")

# --- 9.3 Symmetry for pure diffusion (Pe = 0) ---
ex = exact_rfb(0.0, 1.0, "constant", xi=np.linspace(0.0, 1.0, 401))
sym_err = np.max(np.abs(ex["b_norm"] - ex["b_norm"][::-1]))
check("analytic: Pe=0 bubble is symmetric", sym_err < 1e-10,
      f"max asym = {sym_err:.2e}")

# --- 9.4 FD-400 under-resolves the Pe=100 layer, n=3200 fixes it ---
ex = exact_rfb(100.0, 1.0, "xi", xi=np.linspace(0.0, 1.0, 4001))
fd_400 = fd_rfb(100.0, 1.0, "xi", n_points=400, xi_ref=ex["xi"])
fd_3200 = fd_rfb(100.0, 1.0, "xi", n_points=3200, xi_ref=ex["xi"])
e400 = fd_error_metrics(fd_400, ex)["l2_rel_full"]
e3200 = fd_error_metrics(fd_3200, ex)["l2_rel_full"]
check("analytic: FD-400 Pe=100 error > 1% (under-resolved)", e400 > 1e-2,
      f"got {e400:.2e}")
check("analytic: FD-3200 Pe=100 error < 0.5% (resolved)", e3200 < 5e-3,
      f"got {e3200:.2e}")

# --- 9.5 FD-400 vs analytic agree at low Pe (both modes) ---
for mode_a in ("constant", "xi"):
    ex = exact_rfb(1.0, 1.0, mode_a, xi=np.linspace(0.0, 1.0, 401))
    fd = fd_rfb(1.0, 1.0, mode_a, n_points=400, xi_ref=ex["xi"])
    err = fd_error_metrics(fd, ex)["l2_rel_full"]
    check(f"analytic: FD-400 matches analytic at Pe=1 [{mode_a}]", err < 1e-2,
          f"L2 rel = {err:.2e}")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
