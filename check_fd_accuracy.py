"""Validate the FD bubble solver (n_points=400) against the analytic
constant-coefficient solution.

The production dataset stores normalized bubbles b/b(0.5) computed with
``solve_reference_rfb(..., n_points=400)`` (uniform grid, upwind advection).
For constant coefficients the normalized bubble depends only on (Pe, rho)
and has a closed-form solution (src/rfb_analytic.py).  This script:

  1. sweeps Pe x rho and reports the FD-400 error vs analytic (per mode),
  2. shows how the error decays as n_points is refined (is "solve finer"
     a cure, and at what cost?),
  3. audits every sample in a saved dataset (default rfb_5k_noleak):
     error distribution per split, fraction of inaccurate samples, and
     any oscillating (negative) bubbles.

Run:
    venv/bin/python check_fd_accuracy.py [--name rfb_5k_noleak]
"""

import argparse

import numpy as np

from src.dataset_generation import load_dataset
from src.rfb_analytic import exact_rfb, fd_rfb, fd_error_metrics

MODES = ("constant", "xi")
N_POINTS = 400
THRESH_OK = 1e-2      # L2 rel error (full domain) below this = "accurate"
THRESH_BAD = 1e-1     # above this = "clearly wrong"


def sweep_table(pe_vals, rho_vals):
    print("=" * 84)
    print(f"STEP 1 - FD solver (n_points={N_POINTS}) error vs ANALYTIC solution")
    print("=" * 84)
    print(f"{'Pe':>6} {'rho':>6} | "
          f"{'const L2full':>12} {'layer':>8} {'peak':>8} | "
          f"{'xi L2full':>12} {'layer':>8} {'peak':>8} | osc")
    print("-" * 84)
    for pe in pe_vals:
        for rho in rho_vals:
            cells, osc = [], "-"
            for mode in MODES:
                ex = exact_rfb(pe, rho, mode)
                fd = fd_rfb(pe, rho, mode, n_points=N_POINTS, xi_ref=ex["xi"])
                m = fd_error_metrics(fd, ex)
                flag = " " if m["l2_rel_full"] < THRESH_OK else (
                    " *" if m["l2_rel_full"] < THRESH_BAD else " **")
                cells.append(f"{m['l2_rel_full']:9.1e}{flag} "
                             f"{m['l2_rel_layer']:8.1e} {m['peak_rel']:8.1e}")
                if m["n_neg"]:
                    osc = f"{m['osc_amp']:.0e}"
            print(f"{pe:6.1f} {rho:6.1f} | {cells[0]} | {cells[1]} | {osc}")
    print("-" * 84)
    print("columns: L2full = rel. L2 over full domain, layer = rel. L2 on")
    print("xi in [0.9,1], peak = rel. peak-height error; flags * = 1..10%,")
    print("** = >10%; osc = most-negative bubble value if it dips below 0")
    print()


def resolution_study(cases):
    print("=" * 84)
    print("STEP 2 - Convergence: does refining n_points fix the high-Pe case?")
    print("=" * 84)
    print(f"{'case':>16} | " + " | ".join(
        [f"{n:>7}" for n in (400, 800, 1600, 3200, 6400)])
        + "   (xi-mode L2 rel err, full domain)")
    print("-" * 84)
    for pe, rho in cases:
        cells = []
        for n in (400, 800, 1600, 3200, 6400):
            ex = exact_rfb(pe, rho, "xi")
            fd = fd_rfb(pe, rho, "xi", n_points=n, xi_ref=ex["xi"])
            cells.append(f"{fd_error_metrics(fd, ex)['l2_rel_full']:7.1e}")
        print(f"Pe={pe:5.0f}, rho={rho:4.0f} | " + " | ".join(cells))
    print()
    print("Pe=100 => boundary layer width ~ 1/(2 Pe) = 0.005 in xi,")
    print("so n_points = 400 (dx ~ 2.5e-3) puts ~2 points inside the layer.")
    print()


def audit_dataset(name):
    print("=" * 84)
    ds = load_dataset(name)
    meta = ds["metadata"]
    n_fd = ds["train"]["constant"]["b"].shape[1]
    print(f"STEP 3 - Audit of saved dataset '{name}' (stored {n_fd}-pt FD vs analytic)")
    print("=" * 84)
    pr = meta.get("pe_range", "?"); rr = meta.get("rho_range", "?")
    print(f"  pool range: Pe in {pr}, rho in {rr}")
    print(f"  flagging samples with full-domain L2 rel err >= {THRESH_OK}")
    for sp in ("train", "val", "test"):
        if sp not in ds:
            continue
        for mode in MODES:
            d = ds[sp][mode]
            pe = np.asarray(d["pe"], dtype=float)
            rho = np.asarray(d["rho"], dtype=float)
            b = np.asarray(d["b"], dtype=float)          # normalized bubble
            n = len(pe)
            errs = np.empty(n)
            layer_errs = np.empty(n)
            peak_errs = np.empty(n)
            osc = np.zeros(n, dtype=bool)
            osc_amp = np.zeros(n)
            worst_pe, worst_rho, worst_e = None, None, 0.0
            xi_grid = np.linspace(0.0, 1.0, b.shape[1])
            for i in range(n):
                ex = exact_rfb(float(pe[i]), float(rho[i]), mode)
                xi_eval = ex["xi"]
                b_ex = ex["b_norm"]
                b_fd = np.interp(xi_eval, xi_grid, b[i])
                m = fd_error_metrics(
                    {"b_norm": b_fd, "db_norm": b_fd}, ex)
                errs[i] = m["l2_rel_full"]
                layer_errs[i] = m["l2_rel_layer"]
                peak_errs[i] = m["peak_rel"]
                osc[i] = m["n_neg"] > 0
                osc_amp[i] = m["osc_amp"]
                if errs[i] > worst_e:
                    worst_e, worst_pe, worst_rho = errs[i], pe[i], rho[i]
            n_bad = int(np.sum(errs >= THRESH_OK))
            n_osc = int(np.sum(osc))
            print(f"\n  [{mode}] {sp} (N={n})")
            print(f"    L2 rel err (full): median={np.median(errs):.2e}  "
                  f"p99={np.percentile(errs, 99):.2e}  "
                  f"max={errs.max():.2e} (at Pe={worst_pe:.1f}, rho={worst_rho:.1f})")
            print(f"    layer L2: median={np.median(layer_errs):.2e}  "
                  f"max={layer_errs.max():.2e} | "
                  f"peak err: median={np.median(peak_errs):.2e}  "
                  f"max={peak_errs.max():.2e}")
            print(f"    flagged (>{THRESH_OK}): {100*n_bad/n:.1f}%   "
                  f"oscillating (negative): {n_osc}  "
                  f"(max amplitude {osc_amp.max():.2e})")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="rfb_5k_noleak")
    ap.add_argument("--pe-max", type=float, default=300.0)
    args = ap.parse_args()

    pe_vals = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, args.pe_max]
    rho_vals = [0.2, 1.0, 10.0, 100.0]
    sweep_table(pe_vals, rho_vals)
    resolution_study([(100.0, 1.0), (100.0, 100.0), (300.0, 1.0)])
    audit_dataset(args.name)


if __name__ == "__main__":
    main()
