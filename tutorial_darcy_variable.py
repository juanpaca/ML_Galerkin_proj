#!/usr/bin/env python
"""Darcy variable-diffusion framework: prepare, generate, train, evaluate,
then apply the learned bubbles to the real problem  -(eps u')' = 1 + x
and compare Reference / Galerkin / Gal+bubble(exact) / Gal+bubble(KAN).

Solves  -(epsilon(x) u'(x))' = f(x)  on (0, 1),  u(0) = u(1) = 0,
where epsilon(x) is a random piecewise-constant profile.
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
from src.dataset_generation import load_dataset
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

    # ---- 6. plot learned bubbles vs real on train samples ----
    n_rep = 4
    reps = np.linspace(0, len(ds["train"]["constant"]["b"]) - 1, n_rep).astype(int)
    plt.figure(figsize=(12, 4))
    for mi, mode in enumerate(("constant", "xi")):
        for j, i in enumerate(reps):
            ax = plt.subplot(2, n_rep, mi * n_rep + j + 1)
            b_target = ds["train"][mode]["b"][i]
            b_pred = predict_bubble(model, mi, xi,
                                    ds["train"][mode]["eps_ratios"][i])
            ax.plot(xi, b_pred, "C3", label="KAN")
            ax.plot(xi, b_target, "k--", label="target FD")
            rel = np.linalg.norm(b_pred - b_target) / np.linalg.norm(b_target)
            ax.set_title(f"{mode}[{i}] rel L2={rel:.2e}", fontsize=9)
    plt.suptitle("Learned vs reference bubbles (train set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bubbles_train.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'bubbles_train.png'}")

    # ---- 7. apply to the real problem f = 1 + x ----
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
    print(f"{'method':<22}{'rel L2':>12}{'rel H1':>12}")
    print("-" * 46)
    print(f"{'Reference':<22}{1.0:>12.4e}{1.0:>12.4e}")
    for name, e in errors.items():
        print(f"{name:<22}{e['rel_l2']:>12.4e}{e['rel_h1']:>12.4e}")

    plt.figure(figsize=(9, 6))
    plt.plot(x_ref, u_ref, "k-", lw=1.8, label="Reference")
    for name in sols:
        plt.plot(x_ref, sols[name], label=name)
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.title("Solution of -(eps u')' = 1 + x by each method")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "solution_f1px.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'solution_f1px.png'}")


if __name__ == "__main__":
    main()
