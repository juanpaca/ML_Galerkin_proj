#!/usr/bin/env python
"""Darcy variable-diffusion framework: prepare, generate, train, evaluate,
audit the worst H1-similarity pair, then apply the learned bubbles to the
real problem  -(eps u')' = 1 + x  and compare Reference / Galerkin /
Gal+bubble(exact) / Gal+bubble(KAN).

Solves  -(epsilon(x) u'(x))' = f(x)  on (0, 1),  u(0) = u(1) = 0,
where epsilon(x) is a 5-piece constant profile (thinnest piece ~ l/10).

Run with CUDA for the fastest configuration:
    source venv/bin/activate && venv/bin/python tutorial_darcy_variable.py
"""

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation_darcy_variable import generate_and_save_dataset
from src.darcy_assembly import assemble_p1, assemble_enriched, eval_enriched
from src.darcy_variable import PiecewiseDiffusion, solve_darcy_1d
from src.dataset_generation import load_dataset, _bubble_h1_features
from src.rfb_bubble import MultiKANBubble1D
from src.training import train_multi_bubble_on_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "darcy_piecewise_5pc_cband"
DATASET_SUBDIR = "data_darcy_variable"
MODEL_PATH = Path(f"models/{DATASET_NAME}_kan.pt")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 5-piece spec: exactly 5 pieces of constant eps, thinnest piece ~ l/10,
# eps in [0.1, 10] (contrast c = eps_max/eps_min in [1, 100]).
N_SAMPLES = 5000
N_FD_POINTS = 3201
N_PROFILE_FEATURES = 8
MIN_PIECES = 5
MAX_PIECES = 5
MIN_PIECE_WIDTH = 0.1
EPS_RANGE = (0.1, 10.0)
FEATURE_KIND = "scaled_combo_v2"
EPS_TRANSFORM = "none"
SPLIT_STRATEGY = "contrast_band"
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

N_EPOCHS = 1400
BATCH_SIZE = 256
LR = 1e-3
N_HIDDEN = 32
N_GRID = 12
N_QUAD = 160

TRAIN = True          # False -> load an existing checkpoint from MODEL_PATH
N_APPLY_EL = 8        # P1 elements used in the f=1+x application mesh
N_APPLY_REF = 32001   # fine reference grid for the application comparison


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def evaluate_split(model, data, mode_index, chunk_size=64):
    """Predict bubbles and compute per-sample relative L2."""
    model.eval()
    n, q = len(data["b"]), len(data["xi"])
    xi_t = torch.tensor(data["xi"], dtype=torch.float32, device=DEVICE)
    eps = torch.tensor(data["eps_ratios"], dtype=torch.float32, device=DEVICE)
    pred = np.empty((n, q), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            m = e - s
            xi_f = xi_t.unsqueeze(0).expand(m, -1).reshape(-1)
            pe_f = torch.zeros(m * q, device=DEVICE)
            rho_f = torch.zeros(m * q, device=DEVICE)
            eps_f = eps[s:e].unsqueeze(1).expand(-1, q, -1).reshape(m * q, -1)
            pred[s:e] = model.bubbles[mode_index](
                xi_f, pe_f, rho_f, eps_ratios=eps_f,
            ).reshape(m, q).cpu().numpy()
    target = data["b"]
    err = pred - target
    rmse = np.sqrt(np.mean(err ** 2, axis=1))
    rel_l2 = np.linalg.norm(err, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1), 1e-12)
    return rmse, rel_l2


def predict_bubble(model, mode_index, xi_grid, eps_ratios):
    """Predict one bubble on ``xi_grid`` for the profile feature vector."""
    model.eval()
    xi = torch.tensor(xi_grid, dtype=torch.float32, device=DEVICE)
    eps = torch.tensor(eps_ratios, dtype=torch.float32, device=DEVICE)
    m = xi.shape[0]
    xi_f = xi.unsqueeze(0).expand(1, -1).reshape(-1)
    pe_f = torch.zeros(m, device=DEVICE)
    rho_f = torch.zeros(m, device=DEVICE)
    eps_f = eps.unsqueeze(0).expand(m, -1)
    with torch.no_grad():
        return model.bubbles[mode_index](
            xi_f, pe_f, rho_f, eps_ratios=eps_f,
        ).reshape(-1).cpu().numpy()


def predict_bubbles_batch(model, xi_grid, eps_ratios_all, mode_index,
                          chunk_size=256):
    """Predict bubbles for many profiles (``eps_ratios_all``: N x n_eps).

    Returns an (N, n_fd) array. Vectorized over the batch so it runs fast on
    CUDA.
    """
    model.eval()
    q = len(xi_grid)
    n = len(eps_ratios_all)
    xi = torch.tensor(xi_grid, dtype=torch.float32, device=DEVICE)
    eps = torch.tensor(np.asarray(eps_ratios_all, dtype=np.float32),
                       device=DEVICE)
    out = np.empty((n, q), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            m = e - s
            xi_f = xi.unsqueeze(0).expand(m, -1).reshape(-1)
            pe_f = torch.zeros(m * q, device=DEVICE)
            rho_f = torch.zeros(m * q, device=DEVICE)
            eps_f = eps[s:e].unsqueeze(1).expand(-1, q, -1).reshape(m * q, -1)
            out[s:e] = model.bubbles[mode_index](
                xi_f, pe_f, rho_f, eps_ratios=eps_f,
            ).reshape(m, q).cpu().numpy()
    return out


def worst_h1_pair(ds, lambda_deriv=0.2):
    """Global worst (lowest) H1 cosine pair across (train+val) vs test.

    Returns (worst_simil, ref_b, test_b, ref_eps_ratios, test_eps_ratios,
             ref_pool_idx, test_pool_idx, xi).
    """
    mode = "constant"
    xi = ds["train"][mode]["xi"]
    b_ref = np.concatenate([ds["train"][mode]["b"], ds["val"][mode]["b"]])
    b_test = np.asarray(ds["test"][mode]["b"])
    eps_ref = np.concatenate([ds["train"][mode]["eps_ratios"],
                              ds["val"][mode]["eps_ratios"]])
    eps_test = np.asarray(ds["test"][mode]["eps_ratios"])

    # full cross H1 cosine matrix (rows = test, cols = train+val)
    F_ref = _bubble_h1_features(b_ref, xi, lambda_deriv)
    F_test = _bubble_h1_features(b_test, xi, lambda_deriv)
    d_ref = np.sqrt(np.maximum(np.sum(F_ref * F_ref, axis=1), 1e-30))
    d_test = np.sqrt(np.maximum(np.sum(F_test * F_test, axis=1), 1e-30))
    C = (F_test @ F_ref.T) / np.outer(d_test, d_ref)
    C = np.clip(C, -1.0, 1.0)

    # worst similarity = global min (furthest pair)
    tt, rr = np.unravel_index(np.argmin(C), C.shape)
    worst_simil = float(C[tt, rr])
    return (worst_simil, b_ref[rr], b_test[tt], eps_ref[rr], eps_test[tt],
            rr, tt, xi)


def rel_h1(u_h, u_ref, x, dx):
    """Relative H1 error between two fields on a shared grid."""
    du_h = np.gradient(u_h, x)
    du_r = np.gradient(u_ref, x)
    err_sq = np.sum((u_h - u_ref) ** 2) * dx + np.sum((du_h - du_r) ** 2) * dx
    norm_sq = np.sum(u_ref ** 2) * dx + np.sum(du_r ** 2) * dx
    return float(np.sqrt(err_sq) / np.sqrt(norm_sq))


# ---------------------------------------------------------------------------
# Application: solve - (eps u')' = 1 + x with each method
# ---------------------------------------------------------------------------
def solve_f1px(model, profile, eps_ratios):
    """Return (x_ref, u_ref) and per-method solutions on the bubble grid for
    the problem -(eps u')' = f, f = 1 + x, u(0) = u(1) = 0.

    Methods: Reference (fine FD), Galerkin (P1), Gal+bubble(exact),
    Gal+bubble(KAN). Errors (rel L2 and rel H1 vs reference) are returned.
    """
    source = lambda x: np.asarray(x, dtype=float) + 1.0  # f = 1 + x

    # ---- reference (fine FD, independent grid) ----
    ref = solve_darcy_1d(profile, length=1.0, source=source,
                         n_points=N_APPLY_REF)
    x_ref, u_ref = ref["x"], ref["u"]

    # ---- bubble grid (aligned with the P1 mesh for assemble_enriched) ----
    n = N_APPLY_EL + 1
    mesh = np.linspace(0.0, 1.0, n)
    xi_bub = np.linspace(0.0, 1.0, N_FD_POINTS)
    if (N_FD_POINTS - 1) % N_APPLY_EL != 0:
        raise ValueError("bubble grid must align with the P1 mesh")
    dx_bub = xi_bub[1] - xi_bub[0]
    x_bub = xi_bub  # length = 1

    # ---- Galerkin (classical P1, no bubbles) ----
    A_LL, F_L, free = assemble_p1(mesh, profile, source)
    u_gal_nodes = np.zeros(n)
    u_gal_nodes[free] = np.linalg.solve(A_LL, F_L)
    u_gal = np.interp(x_bub, mesh, u_gal_nodes)

    # ---- exact bubbles: L^-1(1), L^-1(x) (full-domain, normalized) ----
    b_exact = np.stack([
        solve_darcy_1d(profile, length=1.0, source=1.0,
                       n_points=N_FD_POINTS)["u_norm"],
        solve_darcy_1d(profile, length=1.0, source=lambda y: np.asarray(y),
                       n_points=N_FD_POINTS)["u_norm"],
    ])
    u_ex_nodes, c_ex = assemble_enriched(mesh, profile, source, b_exact,
                                         xi_bub, dx_bub)
    u_ex = eval_enriched(u_ex_nodes, c_ex, b_exact, mesh, x_bub)

    # ---- KAN bubbles predicted from the profile ----
    b_kan = np.stack([
        predict_bubble(model, 0, xi_bub, eps_ratios),
        predict_bubble(model, 1, xi_bub, eps_ratios),
    ])
    u_kan_nodes, c_kan = assemble_enriched(mesh, profile, source, b_kan,
                                           xi_bub, dx_bub)
    u_kan = eval_enriched(u_kan_nodes, c_kan, b_kan, mesh, x_bub)

    # all solutions live on the bubble grid; interpolate to the reference grid
    sols = {
        "Galerkin": u_gal,
        "Gal+bubble(exact)": u_ex,
        "Gal+bubble(KAN)": u_kan,
    }
    errors = {}
    for name, u_h in sols.items():
        u_h_ref = np.interp(x_ref, x_bub, u_h)
        errors[name] = {
            "rel_l2": float(np.linalg.norm(u_h_ref - u_ref)
                            / np.linalg.norm(u_ref)),
            "rel_h1": rel_h1(u_h_ref, u_ref, x_ref, x_ref[1] - x_ref[0]),
        }
    # return all solutions on the reference grid for plotting
    sols_ref = {name: np.interp(x_ref, x_bub, u_h) for name, u_h in sols.items()}
    return x_ref, u_ref, sols_ref, errors


def step_eps(profile, xs):
    """Evaluate the piecewise-constant eps(x) on ``xs`` for plotting."""
    return np.asarray([profile.evaluate(np.array([x]))[0] for x in xs])


def plot_profile_step(ax, profile):
    """Draw the piecewise-constant eps(x) as a step plot on [0, 1]."""
    edges = profile.edges
    # sample each piece interior for a flat step; use plot-line style
    for e0, e1, v in zip(edges[:-1], edges[1:], profile.values):
        ax.hlines(v, e0, e1, color="C2", lw=1.6)
    ax.set_ylim(0.9 * min(profile.values), 1.1 * max(profile.values))


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    # ---- 1. prepare / generate ----
    meta_path = Path(DATASET_SUBDIR) / f"{DATASET_NAME}_metadata.json"
    if not meta_path.exists():
        print("Generating dataset...")
        ds = generate_and_save_dataset(
            name=DATASET_NAME, subdir=DATASET_SUBDIR,
            n_samples=N_SAMPLES, n_fd_points=N_FD_POINTS,
            n_profile_features=N_PROFILE_FEATURES,
            min_pieces=MIN_PIECES, max_pieces=MAX_PIECES,
            min_width=MIN_PIECE_WIDTH, eps_range=EPS_RANGE,
            val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED,
            feature_kind=FEATURE_KIND, split_strategy=SPLIT_STRATEGY,
        )
    else:
        ds = load_dataset(DATASET_NAME, subdir=DATASET_SUBDIR)

    n_eps = ds["train"]["constant"]["eps_ratios"].shape[1]
    xi = ds["train"]["constant"]["xi"]
    print(f"Dataset: {DATASET_NAME}")
    print(f"  Train / val / test: {len(ds['train']['constant']['b'])} / "
          f"{len(ds['val']['constant']['b'])} / {len(ds['test']['constant']['b'])}")
    print(f"  FD grid: {len(xi)} pts, eps_ratios: {n_eps} features")

    # ---- 2. build model ----
    model = MultiKANBubble1D(
        n_bubbles=2, n_hidden=N_HIDDEN, n_grid=N_GRID,
        n_eps=n_eps, eps_transform=EPS_TRANSFORM,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KAN: {n_params} parameters ({n_params // 2} per mode)")

    # ---- 3. train (or load) ----
    if TRAIN:
        t0 = time.time()
        history = train_multi_bubble_on_dataset(
            model, ds["train"],
            mode_names=("constant", "xi"),
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            grad_weight=0.0, n_quad=N_QUAD,
            verbose=True, device=DEVICE, lr_scheduler="cosine",
        )
        sync()
        print(f"Training: {time.time() - t0:.1f}s, final loss: "
              f"{min(min(v) for v in history.values()):.4e}")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved: {MODEL_PATH}")
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded: {MODEL_PATH}")

    # ---- 4. plot losses ----
    if TRAIN:
        plt.figure(figsize=(6, 4))
        for mode_name, losses in history.items():
            plt.semilogy(losses, label=mode_name)
        plt.xlabel("epoch"); plt.ylabel("value MSE")
        plt.title("KAN training loss"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "losses.png", dpi=150)
        plt.close()
        print(f"Saved: {FIG_DIR / 'losses.png'}")
        del losses

    # ---- 5. per-split error table ----
    print()
    print(f"{'split':>5} {'mode':>8} {'mean RMSE':>12} {'mean rel L2':>12} {'worst rel L2':>12}")
    print("-" * 55)
    for split in ("train", "val", "test"):
        for mi, mode in enumerate(("constant", "xi")):
            rmse, rel_l2 = evaluate_split(model, ds[split][mode], mi)
            print(f"{split:>5} {mode:>8} {rmse.mean():12.4e} {rel_l2.mean():12.4e} "
                  f"{rel_l2.max():12.4e}")

    # ---- 6. worst H1-cosine pair (train/val vs test), plot pair + eps ----
    (worst_simil, b_ref_pair, b_test_pair, _er, _et, rr, tt, _xi) = worst_h1_pair(ds)

    def pool_index_of_combined(r):
        n_tr = len(ds["train"]["constant"]["pe"])
        if r < n_tr:
            return int(ds["metadata"]["split_indices"]["train"][r])
        return int(ds["metadata"]["split_indices"]["val"][r - n_tr])

    pool_pairs = (pool_index_of_combined(rr),
                  int(ds["metadata"]["split_indices"]["test"][tt]))
    print()
    print("=" * 66)
    print(f"WORST PAIR H1 similarity: worst_simil = {worst_simil:.6f}")
    print("=" * 66)
    print(f"  train/val pool idx {pool_pairs[0]}  <->  test pool idx {pool_pairs[1]}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for col, (tag, b, pool_idx) in enumerate(
            [("train/val", b_ref_pair, pool_pairs[0]),
             ("test", b_test_pair, pool_pairs[1])]):
        ax = axes[0, col]
        ax.plot(_xi, b)
        ax.set_title(f"{tag} bubble (pool {pool_idx})")
        ax.set_xlabel("xi"); ax.grid(True, alpha=0.3)
        prof = PiecewiseDiffusion(
            np.asarray(ds["metadata"]["piece_edges"][pool_idx]),
            np.asarray(ds["metadata"]["piece_values"][pool_idx]))
        ax = axes[1, col]
        plot_profile_step(ax, prof)
        ax.set_title(f"{tag} eps(x)")
        ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
    # overlay both eps on the third column for direct comparison
    ax = axes[0, 2]
    prof_ref = PiecewiseDiffusion(
        np.asarray(ds["metadata"]["piece_edges"][pool_pairs[0]]),
        np.asarray(ds["metadata"]["piece_values"][pool_pairs[0]]))
    # both bubbles overlaid
    ax.plot(_xi, b_ref_pair, label="train/val")
    ax.plot(_xi, b_test_pair, label="test")
    ax.set_title(f"Both bubbles (sim={worst_simil:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1, 2]
    xs = np.linspace(0, 1, 2000)
    for tag, pool_idx, c in [("train/val", pool_pairs[0], "C0"),
                             ("test", pool_pairs[1], "C1")]:
        p = PiecewiseDiffusion(np.asarray(ds["metadata"]["piece_edges"][pool_idx]),
                               np.asarray(ds["metadata"]["piece_values"][pool_idx]))
        ax.plot(xs, step_eps(p, xs), c, lw=1.6, label=tag)
    ax.set_title("Both eps(x)"); ax.legend(); ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Worst H1-cosine pair: worst_simil = {worst_simil:.6f}")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "worst_pair.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'worst_pair.png'}")

    # ---- 7. plot learned bubbles vs real on train samples (+ eps panel) ----
    n_rep = 4
    reps = np.linspace(0, len(ds["train"]["constant"]["b"]) - 1, n_rep).astype(int)
    preds = {mi: predict_bubbles_batch(model, xi, ds["train"]["constant"]["eps_ratios"][reps], mi)
             for mi in (0, 1)}
    profs_train = [PiecewiseDiffusion(
        np.asarray(ds["metadata"]["piece_edges"][ds["metadata"]["split_indices"]["train"][i]]),
        np.asarray(ds["metadata"]["piece_values"][ds["metadata"]["split_indices"]["train"][i]]))
        for i in reps]
    xs = np.linspace(0, 1, 1000)
    n_rows = 3
    plt.figure(figsize=(13, n_rows * n_rep))
    for mi, mode in enumerate(("constant", "xi")):
        for j, i in enumerate(reps):
            ax = plt.subplot(n_rows, n_rep, mi * n_rep + j + 1)
            b_target = ds["train"][mode]["b"][i]
            ax.plot(xi, preds[mi][j], "C3", label="KAN")
            ax.plot(xi, b_target, "k--", label="target FD")
            rel = np.linalg.norm(preds[mi][j] - b_target) / np.linalg.norm(b_target)
            ax.set_title(f"{mode}[{i}] rel L2={rel:.2e}", fontsize=9)
            ax.grid(True, alpha=0.3)
    for j, i in enumerate(reps):
        ax = plt.subplot(n_rows, n_rep, 2 * n_rep + j + 1)
        ax.plot(xs, step_eps(profs_train[j], xs), "C2", lw=1.4)
        ax.set_title(f"eps(x) train[{i}]", fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Learned vs reference bubbles (train set) + eps profiles")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bubbles_train.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'bubbles_train.png'}")

    # ---- 8. apply to the real problem f = 1 + x ----
    i_app = int(reps[1])  # one representative train profile
    pool_idx = ds["metadata"]["split_indices"]["train"][i_app]
    piece_edges = ds["metadata"]["piece_edges"][pool_idx]
    piece_values = ds["metadata"]["piece_values"][pool_idx]
    profile = PiecewiseDiffusion(np.asarray(piece_edges),
                                 np.asarray(piece_values))
    eps_ratios = ds["train"]["constant"]["eps_ratios"][i_app]

    x_ref, u_ref, sols, errors = solve_f1px(model, profile, eps_ratios)

    print()
    print("=" * 66)
    print(f"APPLICATION: -(eps u')' = 1 + x   (P1 mesh: {N_APPLY_EL} elements)")
    print("=" * 66)
    print(f"  epsilon profile (pool idx {pool_idx}):")
    print(f"    pieces: {list(zip(piece_edges, piece_values))}")
    print(f"    contrast c = {np.max(piece_values)/np.min(piece_values):.3f}")
    print(f"{'method':<22}{'rel L2':>12}{'rel H1':>12}")
    print("-" * 46)
    print(f"{'Reference':<22}{1.0:>12.4e}{1.0:>12.4e}")
    for name, e in errors.items():
        print(f"{name:<22}{e['rel_l2']:>12.4e}{e['rel_h1']:>12.4e}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(x_ref, u_ref, "k-", lw=1.8, label="Reference")
    for name in sols:
        ax.plot(x_ref, sols[name], label=name)
    ax.set_ylabel("u(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title("Solution of -(eps u')' = 1 + x by each method")
    ax = axes[1]
    ax.plot(x_ref, step_eps(profile, x_ref), "C2", lw=1.6)
    ax.set_ylabel("eps(x)"); ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "solution_f1px.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'solution_f1px.png'}")


if __name__ == "__main__":
    main()
