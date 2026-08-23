"""Unit tests for the KAN architecture (src/kan.py, KANLayer in src/rfb_bubble.py).

Validates against B-spline theory and the scipy reference implementation:

  * knot extension structure (clamped, multiplicity k-1)
  * basis values == scipy BSpline.design_matrix (degree k-1), closed domain,
    partition of unity at BOTH endpoints, exact zeros outside
  * polynomial reproduction up to degree k-1 (and NOT above)
  * d(basis)/dx via autograd vs central finite differences
  * KAN1D: init statistics, precomputed-basis path, dtype handling,
    forward_with_deriv, get_coefficients, CPU/GPU parity
  * KANLayer.b_splines: agreement with the kan.py evaluator, endpoint fold
    (x == x_max keeps its spline contribution), parameter counts
  * approximation smoke test: a trained edge fits a smooth target

Spline convention used throughout this repo: ``k`` is the number of active
basis functions (spline ORDER); reproduced polynomial degree is ``k - 1``.
This differs from pykan / Blealtan efficient-kan where ``spline_order``
denotes the DEGREE (their cubic default == k=4 here).

Run:  venv/bin/python test_kan.py
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline

from src.kan import KAN1D, _eval_bspline_basis, _extend_knots, silu
from src.rfb_bubble import KANLayer

N_fail = 0


def check(name, ok, detail=""):
    global N_fail
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        N_fail += 1


# =========================================================================
print("\n=== 1. building blocks ===")

xs_ = torch.linspace(-3, 3, 101, dtype=torch.float64)
check("silu matches torch.nn.functional.silu",
      torch.allclose(silu(xs_), F.silu(xs_)))

G, k = 8, 3
kn64 = _extend_knots(-1.0, 1.0, G, k).double()
check("knot length = G+1+2(k-1)", kn64.numel() == G + 1 + 2 * (k - 1))
check("left ghosts repeat x_min", torch.all(kn64[:k - 1] == -1.0))
check("right ghosts repeat x_max", torch.all(kn64[-(k - 1):] == 1.0))
check("interior uniform", torch.allclose(kn64[k - 1:-(k - 1)],
                                         torch.linspace(-1, 1, G + 1, dtype=torch.float64)))

# =========================================================================
print("\n=== 2. B-spline basis vs scipy (degree k-1) ===")

for G_, k_ in [(4, 2), (8, 3), (12, 3), (6, 4)]:
    kn = _extend_knots(-1.0, 1.0, G_, k_).double()
    xd = torch.linspace(-1.0, 1.0, 601, dtype=torch.float64)
    B = _eval_bspline_basis(xd, kn, k_)
    ref = BSpline.design_matrix(xd.numpy(), kn.numpy(), k_ - 1).toarray()
    check(f"(G={G_},k={k_}) == scipy deg {k_-1} incl endpoints",
          np.abs(B.numpy() - ref).max() < 1e-10,
          f"max diff {np.abs(B.numpy()-ref).max():.1e}")

    rowsum = B.sum(dim=1)
    ends = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    Be = _eval_bspline_basis(ends, kn, k_)
    check(f"(G={G_},k={k_}) partition of unity (closed domain)",
          abs(rowsum.max() - 1) < 1e-10 and abs(rowsum.min() - 1) < 1e-10
          and abs(Be[0].sum() - 1) < 1e-10 and abs(Be[1].sum() - 1) < 1e-10)
    check(f"(G={G_},k={k_}) non-negative & local support",
          B.min() >= -1e-12 and (B.abs() > 1e-14).sum(dim=1).max() <= k_)

    out = _eval_bspline_basis(torch.tensor([-5.0, -1.5, 1.5, 9.0], dtype=torch.float64), kn, k_)
    check(f"(G={G_},k={k_}) zero outside [x_min,x_max]", out.abs().max() == 0.0)

# =========================================================================
print("\n=== 3. polynomial reproduction pins the true degree ===")

kn = _extend_knots(-1.0, 1.0, 10, 3).double()
xd = torch.linspace(-1, 1, 400, dtype=torch.float64)
A = _eval_bspline_basis(xd, kn, 3).numpy()
for p, expect_exact in [(1, True), (2, True), (3, False)]:
    coef, *_ = np.linalg.lstsq(A, xd.numpy() ** p, rcond=None)
    err = np.abs(A @ coef - xd.numpy() ** p).max()
    ok = err < 1e-9 if expect_exact else err > 1e-4
    check(f"x^{p} reproduced {'exactly' if expect_exact else 'NOT exactly'} "
          f"(k=3 -> quadratic)", ok, f"fit err {err:.1e}")

# =========================================================================
print("\n=== 4. gradients through the basis ===")

xg = torch.tensor([-0.73, -0.31, 0.17, 0.55], dtype=torch.float64, requires_grad=True)
w = torch.randn(kn.numel() - 3, dtype=torch.float64)


def f_basis(xx):
    return _eval_bspline_basis(xx, kn, 3) @ w


d_aut = torch.autograd.grad(f_basis(xg).sum(), xg)[0]
h = 1e-6
with torch.no_grad():
    d_fd = (f_basis(xg.detach() + h) - f_basis(xg.detach() - h)) / (2 * h)
check("dB/dx autograd == central FD", (d_aut - d_fd).abs().max() < 1e-4,
      f"max diff {(d_aut-d_fd).abs().max():.1e}")

# =========================================================================
print("\n=== 5. KAN1D model ===")

torch.manual_seed(42)
m = KAN1D(n_grid=8, k=3)
check("init: n_basis = n_grid+k-1", m.n_basis == 10)
check("init: w_s == 1", m.w_s.item() == 1.0)
check("init: c std ~ 0.1", abs(m.c.std().item() - 0.1) < 0.05)

xc = torch.rand(64) * 2 - 1
y_direct = m(xc)
y_precomp = m(xc, precomputed_basis=_eval_bspline_basis(xc, m.knots, m.k))
check("precomputed_basis path == direct eval",
     torch.allclose(y_direct, y_precomp, atol=1e-7))
g_, dg_ = m.forward_with_deriv(xc[:8])
check("forward_with_deriv value consistent",
     torch.allclose(g_, y_direct[:8], atol=1e-7))

xf = torch.tensor([-0.5, 0.25], dtype=torch.float64)
check("float64 input supported (dtype follows input)",
     m.double()(xf).dtype == torch.float64)

co = m.get_coefficients()
check("get_coefficients keys/values", set(co) == {"w_b", "w_s", "c"}
      and co["c"].shape == (m.n_basis,))

xd2 = torch.linspace(-0.95, 0.95, 33, dtype=torch.float64, requires_grad=True)
out = m.double()(xd2)
d_model = torch.autograd.grad(out.sum(), xd2)[0]
with torch.no_grad():
    fd2 = (m.double()((xd2 + h).detach()) - m.double()((xd2 - h).detach())) / (2 * h)
check("KAN1D d(output)/dx == FD (interior pts)", (d_model - fd2).abs().max() < 1e-4,
      f"max diff {(d_model-fd2).abs().max():.1e}")

if torch.cuda.is_available():
    mg = KAN1D(n_grid=8, k=3).cuda()
    mg.load_state_dict({p: t.cpu() for p, t in m.state_dict().items()})
    check("CUDA parity", torch.allclose(y_direct, mg(xc.cuda()).cpu(), atol=1e-6))
    gg, ddgg = mg.forward_with_deriv(torch.linspace(-1, 1, 32, device="cuda"))
    (gg * ddgg).sum().backward()
    check("GPU second-order grad flow finite",
          all(p.grad is None or torch.isfinite(p.grad).all() for p in mg.parameters()))
else:
    print("  [SKIP] CUDA not available")

# =========================================================================
print("\n=== 6. KANLayer (used by all trained bubble models) ===")

torch.manual_seed(0)
layer = KANLayer(4, 3, n_grid=12, k=3)
n_expected = layer.n_out * layer.n_in * (layer.n_basis + 2)   # base_w + scaler + spline_w
n_actual = sum(p.numel() for p in layer.parameters())
check("parameter count = n_out*n_in*(n_basis+2)", n_actual == n_expected,
      f"{n_actual} params")

xb = torch.rand(128, 4, dtype=torch.float64) * 2 - 1
B_layer = layer.double().b_splines(xb)[:, 2, :]
B_ref = _eval_bspline_basis(xb[:, 2], layer.knots.double(), layer.k)
check("b_splines == kan.py evaluator", torch.allclose(B_layer, B_ref, atol=1e-10),
      f"max diff {(B_layer-B_ref).abs().max():.1e}")

xe = torch.ones(5, 4)
Be = layer.double().b_splines(xe)
check("partition of unity at x == x_max (endpoint fold)",
     (Be.sum(-1) - 1).abs().max() < 1e-10)
check("endpoint mass on last basis",
     (Be[..., layer.n_basis - 1] - 1).abs().max() < 1e-10)

layer32 = KANLayer(4, 3, n_grid=12, k=3)
xin = torch.rand(64, 4)
out = layer32(xin)
check("forward shape + finite", out.shape == (64, 3) and torch.isfinite(out).all())
loss = out.pow(2).mean()
loss.backward()
grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer32.parameters())
check("backward gives finite grads on every parameter", grads_finite)

# =========================================================================
print("\n=== 7. approximation smoke test (CPU, small budget) ===")

torch.manual_seed(7)
model = KAN1D(n_grid=24, k=4)
xt = torch.linspace(-1, 1, 512)
target = torch.sin(3 * xt) + 0.5 * torch.cos(7 * xt)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
for _ in range(2000):
    opt.zero_grad()
    ((model(xt) - target) ** 2).mean().backward()
    opt.step()
    sched.step()
rel_l2 = (((model(xt) - target) ** 2).sum() / (target ** 2).sum()).sqrt().item()
check("trained edge fits sin+cos target", rel_l2 < 1e-3, f"rel L2 {rel_l2:.1e}")

# =========================================================================
print()
if N_fail:
    print(f"{N_fail} FAILURE(S)")
    sys.exit(1)
print("ALL TESTS PASSED")
