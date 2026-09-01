#!/usr/bin/env python
"""Darcy variable-diffusion framework: prepare, generate, train, evaluate,
audit the closest (most-similar train/val-test) H1 pair for leakage, then apply the learned bubbles to the
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
from src.darcy_variable import (PiecewiseDiffusion, solve_darcy_1d,
                                random_piecewise_diffusion,
                                make_profile_features)
from src.dataset_generation import load_dataset, _bubble_h1_features
from src.rfb_bubble import MultiKANBubble1D
from src.training import train_multi_bubble_on_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "darcy_piecewise_5pc_cband_15k"
DATASET_SUBDIR = "data_darcy_variable"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 5-piece spec: exactly 5 pieces of constant eps, thinnest piece ~ l/10,
# eps in [0.1, 10] (contrast c = eps_max/eps_min in [1, 100]).
N_SAMPLES = 15000
# Higher bubble resolution: the enrichment gate is disabled (see below), so the
# budget it would have spent re-solving a fine reference per sample goes into
# generating the reference bubbles themselves at a finer grid instead.
N_FD_POINTS = 6401
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

N_EPOCHS = 400
BATCH_SIZE = 256
LR = 1e-3
# Uniform-KAN depth scheme (legacy): N_LAYERS = total KANLayer objects
# (2 = one hidden layer of width N_HIDDEN).
N_HIDDEN = 1
N_LAYERS = 4
# Explicit per-hidden-layer widths. When non-empty this TAKES PRECEDENCE over
# N_HIDDEN/N_LAYERS, e.g. [32, 64, 32] -> [n_in -> 32 -> 64 -> 32 -> 1].
HIDDEN_SIZES = []


def _net_tag():
    """Short architecture tag for the model filename. Mirrors the builder's
    precedence: HIDDEN_SIZES (explicit widths) wins over n_hidden/n_layers."""
    if HIDDEN_SIZES:
        return "h" + "_".join(str(w) for w in HIDDEN_SIZES)
    return f"l{N_LAYERS}h{N_HIDDEN}"


# The checkpoint name self-describes the model: DATASET_NAME already encodes
# the pool size (e.g. ..._15k), and _net_tag() the exact depth/width used.
MODEL_PATH = Path(f"models/{DATASET_NAME}_{_net_tag()}_kan.pt")


N_GRID = 12
N_QUAD = 80
# Early stopping on validation loss: stop when val loss has not improved for
# this many epochs; the model is reverted to the best-val epoch's weights.
EARLY_STOP_PATIENCE = 50
# Early stopping robustness: ignore the first ES_WARMUP epochs (init spikes)
# and decide on an EMA-smoothed validation loss with factor ES_EMA_ALPHA so
# single-epoch oscillation cannot trigger premature stopping.
ES_WARMUP = 20
ES_EMA_ALPHA = 0.95
# L2 weight regularization strength lambda (added as lambda * sum(w^2) to the
# training loss only; the reported train/val losses stay the pure data loss).
WEIGHT_DECAY = 1e-3

# If True, retrain even when a checkpoint already exists. If False, a present
# checkpoint is loaded (train/save skipped) so you can re-run just the analysis
# and figures without re-training.
FORCE_RETRAIN = False
N_APPLY_EL = 8        # P1 elements used in the f=1+x application mesh
N_APPLY_REF = 32001   # fine reference grid for the application comparison
N_APPLY_SAMPLES = 4   # number of test-set profiles to apply f = 1 + x to


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


def closest_h1_pair(ds, lambda_deriv=0.2):
    """Closest (most similar) train/val vs test pair by H1 cosine.

    A high value here is the leakage diagnostic: if a test bubble is nearly a
    twin of a train bubble (similarity ~ 1), the OOD test is compromised.

    Returns (best_simil, ref_b, test_b, ref_eps_ratios, test_eps_ratios,
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

    # closest pair = global max (the most-similar train/test twins)
    tt, rr = np.unravel_index(np.argmax(C), C.shape)
    best_simil = float(C[tt, rr])
    return (best_simil, b_ref[rr], b_test[tt], eps_ref[rr], eps_test[tt],
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
    source = lambda x: np.asarray(0*x, dtype=float) + 1.0  # f = 1 + x
    #source = lambda x:  1.0  # f = 1 + x

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


def random_profile_in_contrast_band(rng, band, n_pieces=5,
                                    eps_range=(0.1, 10.0), min_width=0.1,
                                    max_tries=2000):
    """Generate a random piecewise profile whose realized contrast
    c = eps_max/eps_min lies inside ``band = [c_lo, c_hi]``.

    Rejection-samples ``random_piecewise_diffusion`` (log-uniform values in
    ``eps_range``) until the contrast falls in the band. Used to build
    interpolation test profiles: same generation spec as the training pool,
    but with contrast interior to the train band so the model is evaluated
    inside (not at/outside) the range of contrasts it has seen.
    """
    for _ in range(max_tries):
        p = random_piecewise_diffusion(rng, n_pieces=n_pieces,
                                       eps_range=eps_range, min_width=min_width)
        c = float(p.values.max() / p.values.min())
        if band[0] <= c <= band[1]:
            return p, c
    raise RuntimeError(f"no profile found in contrast band {band} "
                       f"after {max_tries} draws")


def is_new_profile(pool_edges, pool_values, profile) -> bool:
    """True if ``profile``'s (edges, values) tuple is not in the dataset pool."""
    edges = np.round(np.asarray(profile.edges), 12)
    values = np.round(np.asarray(profile.values), 12)
    for e, v in zip(pool_edges, pool_values):
        if (np.array_equal(np.round(np.asarray(e), 12), edges)
                and np.array_equal(np.round(np.asarray(v), 12), values)):
            return False
    return True


def max_h1_simil_to_pool(profile, ds, mode="constant", lambda_deriv=0.2):
    """H1-cosine similarity of the profile's reference bubble to the full
    train+val pool. Near 1 ⇒ the new profile is effectively a twin of an
    existing shape (would defeat the "new profile" test).
    """
    b_train = np.concatenate([ds["train"][mode]["b"], ds["val"][mode]["b"]])
    xi = ds["train"][mode]["xi"]
    b_new = solve_darcy_1d(profile, length=1.0, source=1.0,
                           n_points=len(xi))["u_norm"]
    F_pool = _bubble_h1_features(b_train, xi, lambda_deriv)
    d_pool = np.sqrt(np.maximum(np.sum(F_pool * F_pool, axis=1), 1e-30))
    F_new = _bubble_h1_features(np.asarray([b_new]), xi, lambda_deriv)[0]
    d_new = np.sqrt(max(float(np.sum(F_new * F_new)), 1e-30))
    c = (F_new @ F_pool.T) / (d_new * d_pool)
    return float(np.clip(c, -1.0, 1.0).max())


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    # ---- 1. prepare / generate ----
    # Resolve the on-disk location exactly as load_dataset/save_dataset do
    # (base "datasets" + subdir), so the guard does not regenerate data that
    # already exists.
    data_dir = Path("datasets") / DATASET_SUBDIR
    meta_path = data_dir / f"{DATASET_NAME}_metadata.json"
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
            verify_enrichment=False,
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
        n_bubbles=2, n_grid=N_GRID,
        n_eps=n_eps, eps_transform=EPS_TRANSFORM,
        hidden_sizes=HIDDEN_SIZES or None,
        n_hidden=N_HIDDEN, n_layers=N_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KAN: {n_params} parameters ({n_params // 2} per mode)")

    # ---- 3. train (or load) ----
    history_path = MODEL_PATH.with_suffix(".history.json")
    if MODEL_PATH.exists() and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded: {MODEL_PATH}")
        history = None
        if history_path.exists():
            import json as _json
            history = _json.loads(history_path.read_text())
            print(f"Loaded loss history: {history_path}")
    else:
        t0 = time.time()
        history = train_multi_bubble_on_dataset(
            model, ds["train"],
            mode_names=("constant", "xi"),
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            grad_weight=0.000, energy_weight=0.000, n_quad=N_QUAD,
            verbose=True, device=DEVICE, lr_scheduler="cosine",
            val_split=ds["val"],
            patience=EARLY_STOP_PATIENCE,
            weight_decay=WEIGHT_DECAY,
            es_warmup=ES_WARMUP,
            es_ema_alpha=ES_EMA_ALPHA,
        )
        sync()
        tr_min = min(min(v["train"]) for v in history.values())
        vl_min = min(min(v["val"]) for v in history.values())
        n_run = {m: len(hist["val"]) for m, hist in history.items()}
        es_msg = ", ".join(f"{m}: {ne}/{N_EPOCHS} epochs"
                           for m, ne in n_run.items())
        print(f"Training: {time.time() - t0:.1f}s, final train loss: "
              f"{tr_min:.4e}, final val loss: {vl_min:.4e}")
        print(f"Early stopping (patience {EARLY_STOP_PATIENCE}): epochs run "
              f"per mode -> {es_msg} (best-val weights restored per mode)")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved: {MODEL_PATH}")
        import json as _json
        history_path.write_text(_json.dumps(history))
        print(f"Saved loss history: {history_path}")

    # ---- 4. plot losses (train + val), when history is available ----
    if history is not None:
        plt.figure(figsize=(7, 4.5))
        for mode_name, modes in history.items():
            train_l = np.asarray(modes["train"]) if isinstance(modes, dict) else np.asarray(modes)
            plt.semilogy(train_l, label=f"{mode_name} (train)")
            if isinstance(modes, dict) and "val" in modes:
                plt.semilogy(modes["val"], "--", label=f"{mode_name} (val)")
        plt.xlabel("epoch"); plt.ylabel("value MSE")
        plt.title("KAN training & validation loss"); plt.legend()
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "losses.png", dpi=150)
        plt.close()
        print(f"Saved: {FIG_DIR / 'losses.png'}")
        del train_l

    # ---- 5. per-split error table ----
    print()
    print(f"{'split':>5} {'mode':>8} {'mean RMSE':>12} {'mean rel L2':>12} {'worst rel L2':>12}")
    print("-" * 55)
    for split in ("train", "val", "test"):
        for mi, mode in enumerate(("constant", "xi")):
            rmse, rel_l2 = evaluate_split(model, ds[split][mode], mi)
            print(f"{split:>5} {mode:>8} {rmse.mean():12.4e} {rel_l2.mean():12.4e} "
                  f"{rel_l2.max():12.4e}")

    # ---- 6. closest H1-cosine pair (train/val vs test): leakage check ----
    (best_simil, b_ref_pair, b_test_pair, _er, _et, rr, tt, _xi) = closest_h1_pair(ds)

    def pool_index_of_combined(r):
        n_tr = len(ds["train"]["constant"]["pe"])
        if r < n_tr:
            return int(ds["metadata"]["split_indices"]["train"][r])
        return int(ds["metadata"]["split_indices"]["val"][r - n_tr])

    pool_pairs = (pool_index_of_combined(rr),
                  int(ds["metadata"]["split_indices"]["test"][tt]))
    print()
    print("=" * 66)
    print(f"CLOSEST PAIR H1 similarity (max): best_simil = {best_simil:.6f}")
    print("  -> the most-similar train/val-test bubbles; high value = leakage.")
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
    # both bubbles overlaid
    ax.plot(_xi, b_ref_pair, label="train/val")
    ax.plot(_xi, b_test_pair, label="test")
    ax.set_title(f"Both bubbles (sim={best_simil:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1, 2]
    xs = np.linspace(0, 1, 2000)
    for tag, pool_idx, c in [("train/val", pool_pairs[0], "C0"),
                             ("test", pool_pairs[1], "C1")]:
        p = PiecewiseDiffusion(np.asarray(ds["metadata"]["piece_edges"][pool_idx]),
                               np.asarray(ds["metadata"]["piece_values"][pool_idx]))
        ax.plot(xs, step_eps(p, xs), c, lw=1.6, label=tag)
    ax.set_title("Both eps(x)"); ax.legend(); ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Closest H1-cosine pair: best_simil = {best_simil:.6f}")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "closest_pair.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'closest_pair.png'}")

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

    # ---- 8. apply to the real problem f = 1 + x on test-set profiles ----
    n_test = len(ds["test"]["constant"]["b"])
    apply_idx = np.linspace(0, n_test - 1, N_APPLY_SAMPLES).astype(int).tolist()
    print()
    print("=" * 66)
    print(f"APPLICATION: -(eps u')' = 1 + x   (P1 mesh: {N_APPLY_EL} elements, "
          f"{len(apply_idx)} test profiles)")
    print("=" * 66)
    for i_test in apply_idx:
        pool_idx = ds["metadata"]["split_indices"]["test"][i_test]
        piece_edges = ds["metadata"]["piece_edges"][pool_idx]
        piece_values = ds["metadata"]["piece_values"][pool_idx]
        profile = PiecewiseDiffusion(np.asarray(piece_edges),
                                     np.asarray(piece_values))
        eps_ratios = ds["test"]["constant"]["eps_ratios"][i_test]

        x_ref, u_ref, sols, errors = solve_f1px(model, profile, eps_ratios)

        print()
        print(f"  -- test sample #{i_test} (pool idx {pool_idx}) --")
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
        ax.set_title(f"Solution of -(eps u')' = 1 + x  (test #{i_test}, "
                     f"pool {pool_idx})")
        ax = axes[1]
        ax.plot(x_ref, step_eps(profile, x_ref), "C2", lw=1.6)
        ax.set_ylabel("eps(x)"); ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = FIG_DIR / f"solution_f1px_test{i_test}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"    Saved: {fname}")

    # ---- 9. interpolation test: new profiles inside the train contrast band ----
    # Generate *new* eps(x) profiles that are not part of the dataset pool but
    # whose contrast c = eps_max/eps_min lies inside the train band. The model
    # is then evaluated on genuinely unseen profiles inside its training
    # contrast interval (interpolation), using the same Reference / Galerkin /
    # Gal+bubble(exact) / Gal+bubble(KAN) pipeline as the test-set application.
    train_band = [float(x) for x in
                  ds["metadata"]["contrast_band_edges"]["train"]]
    pool_edges = ds["metadata"]["piece_edges"]
    pool_values = ds["metadata"]["piece_values"]
    interp_rng = np.random.default_rng(SEED + 1000)   # fresh stream → new shapes
    print()
    print("=" * 66)
    print(f"INTERPOLATION: new profiles with contrast in the train band "
          f"[{train_band[0]:.3f}, {train_band[1]:.3f}]  ({N_APPLY_SAMPLES} profiles)")
    print("=" * 66)
    for k in range(N_APPLY_SAMPLES):
        profile, c = random_profile_in_contrast_band(
            interp_rng, train_band, n_pieces=MIN_PIECES,
            eps_range=EPS_RANGE, min_width=MIN_PIECE_WIDTH)
        if not is_new_profile(pool_edges, pool_values, profile):
            raise RuntimeError("interpolation profile collides with the dataset pool")
        eps_ratios = make_profile_features(profile, N_PROFILE_FEATURES,
                                           FEATURE_KIND)

        x_ref, u_ref, sols, errors = solve_f1px(model, profile, eps_ratios)

        simil = max_h1_simil_to_pool(profile, ds)
        print()
        print(f"  -- interpolation profile #{k} --  contrast c = {c:.4f} "
              f"(train band [{train_band[0]:.3f}, {train_band[1]:.3f}])")
        print(f"    new shape: not in pool, max H1-similarity to train/val "
              f"pool = {simil:.4f}")
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
        ax.set_title(f"INTERPOLATION (c={c:.3f} in train band): "
                     f"solution of -(eps u')' = 1 + x  (profile #{k})")
        ax = axes[1]
        ax.plot(x_ref, step_eps(profile, x_ref), "C2", lw=1.6)
        ax.set_ylabel("eps(x)"); ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = FIG_DIR / f"solution_f1px_interp{k}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"    Saved: {fname}")


if __name__ == "__main__":
    main()
