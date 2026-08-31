#!/usr/bin/env python
"""Darcy variable-diffusion tutorial: load data, train KAN, evaluate, save.

Solves  -(epsilon(x) u'(x))' = f(x)  on (0, 1),  u(0) = u(1) = 0,
where epsilon(x) is a random piecewise-constant profile.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation_darcy_variable import generate_and_save_dataset
from src.dataset_generation import load_dataset
from src.rfb_bubble import MultiKANBubble1D
from src.training import train_multi_bubble_on_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "darcy_piecewise_combo_cband5k"
DATASET_SUBDIR = "data_darcy_variable"
MODEL_PATH = Path(f"models/{DATASET_NAME}_kan.pt")

N_SAMPLES = 5000
N_FD_POINTS = 801
N_PROFILE_FEATURES = 24
MIN_PIECES = 2
MAX_PIECES = 16
FEATURE_KIND = "scaled_combo"
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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def main():
    meta_path = Path(DATASET_SUBDIR) / f"{DATASET_NAME}_metadata.json"

    if not meta_path.exists():
        print("Generating dataset...")
        ds = generate_and_save_dataset(
            name=DATASET_NAME, subdir=DATASET_SUBDIR,
            n_samples=N_SAMPLES, n_fd_points=N_FD_POINTS,
            n_profile_features=N_PROFILE_FEATURES,
            min_pieces=MIN_PIECES, max_pieces=MAX_PIECES,
            val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED,
            feature_kind=FEATURE_KIND, split_strategy=SPLIT_STRATEGY,
        )
    else:
        ds = load_dataset(DATASET_NAME, subdir=DATASET_SUBDIR)

    n_eps = ds["train"]["constant"]["eps_ratios"].shape[1]
    print(f"Dataset: {DATASET_NAME}")
    print(f"  Train / val / test: {len(ds['train']['constant']['b'])} / "
          f"{len(ds['val']['constant']['b'])} / {len(ds['test']['constant']['b'])}")
    print(f"  FD grid: {len(ds['train']['constant']['xi'])} pts, "
          f"eps_ratios: {n_eps} features")

    model = MultiKANBubble1D(
        n_bubbles=2, n_hidden=N_HIDDEN, n_grid=N_GRID,
        n_eps=n_eps, eps_transform=EPS_TRANSFORM,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KAN: {n_params} parameters ({n_params // 2} per mode)")

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

    print()
    print(f"{'split':>5} {'mode':>8} {'mean RMSE':>12} {'mean rel L2':>12} {'worst rel L2':>12}")
    print("-" * 55)
    for split in ("train", "val", "test"):
        for mi, mode in enumerate(("constant", "xi")):
            rmse, rel_l2 = evaluate_split(model, ds[split][mode], mi)
            print(f"{split:>5} {mode:>8} {rmse.mean():12.4e} {rel_l2.mean():12.4e} "
                  f"{rel_l2.max():12.4e}")


if __name__ == "__main__":
    main()
