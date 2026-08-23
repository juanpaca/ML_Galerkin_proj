# %% [markdown]
# # KAN Darcy Variable-Diffusion Tutorial
#
# Notebook-style walkthrough for
#
# ```text
# -(epsilon(x) u'(x))' = f(x)  on (0, L),
# u(0) = u(L) = 0.
# ```
#
# The reference solutions use random piecewise-constant diffusion profiles
# with epsilon in [0.1, 100]. The dataset is split by normalized solution
# shape, not by random samples, and is audited for train/test twins.
#
# Run cells sequentially in VS Code, Jupyter, or Colab. Training is performed
# in cell 7 when `RUN_TRAINING = True`.

# %% [markdown]
# ## 0. Colab Setup

# %%
import os
import sys

# Local run: Colab clone/pull disabled.
# if os.path.isdir("/content") and not os.path.isdir("/content/ML_Galerkin_proj"):
#     os.system("git clone https://github.com/juanpaca/ML_Galerkin_proj.git "
#               "/content/ML_Galerkin_proj")
# if os.path.isdir("/content/ML_Galerkin_proj"):
#     os.system("git -C /content/ML_Galerkin_proj pull")
#     sys.path.insert(0, "/content/ML_Galerkin_proj")
# else:
#     sys.path.insert(0, ".")
sys.path.insert(0, ".")

# %% [markdown]
# ## 1. Imports and Experiment Configuration

# %%
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from data_generation_darcy_variable import generate_and_save_dataset
from src.darcy_variable import PiecewiseDiffusion, solve_darcy_1d
from src.dataset_generation import bubble_similarity_analysis, load_dataset
from src.rfb_bubble import MultiKANBubble1D
from src.training import train_multi_bubble_on_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_SUBDIR = "data_darcy_variable"
DATASET_ROOT = (
    "/content/ML_Galerkin_proj/datasets"
    if os.path.isdir("/content/ML_Galerkin_proj/datasets")
    else "datasets"
)
DATASET_SUBDIR_PATH = os.path.join(DATASET_ROOT, DATASET_SUBDIR)

# Set these before running the cells.
GENERATE_DATA = False       # True regenerates the dataset pool.
RUN_TRAINING = True          # Set False to inspect/test an existing checkpoint.
NO_PLOTS = False

# Profile-diversity controls for the piecewise diffusion pools.
DATASET_NAME = "darcy_piecewise_combo_cband5k"
MIN_PIECES = 2
MAX_PIECES = 16
MIN_PIECE_WIDTH = 0.0              # unconstrained piece widths (i.i.d. pool)
MODEL_PATH = Path(f"models/{DATASET_NAME}_kan.pt")
FEATURE_KIND = "scaled_combo"      # profile summary fed to the KAN
EPS_TRANSFORM = "none"             # features are pre-scaled to [-1, 1]
SPLIT_STRATEGY = "contrast_band"   # contiguous bands of realized contrast

N_SAMPLES = 5000
N_FD_POINTS = 801
N_PROFILE_FEATURES = 24
THETA = 0.99
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

N_EPOCHS = 1400
BATCH_SIZE = 256
TRAIN_QUAD = 160
LEARNING_RATE = 1e-3
LR_SCHEDULER = "cosine"
N_HIDDEN = 32
N_GRID = 12

print(f"Using device: {DEVICE}")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# %% [markdown]
# ## 2. Generate or Load the Dataset
#
# The generator uses all samples that survive the no-twin filter for training.
# Validation and test are reserved first as the most atypical solution shapes.

# %%
metadata_path = Path(DATASET_SUBDIR_PATH) / f"{DATASET_NAME}_metadata.json"
dataset_needs_generation = GENERATE_DATA or not metadata_path.exists()
if not dataset_needs_generation:
    existing = load_dataset(DATASET_NAME, subdir=DATASET_SUBDIR)
    dataset_needs_generation = not all(
        mode in existing.get("train", {}) for mode in ("constant", "xi")
    )
    if dataset_needs_generation:
        print("Existing Darcy dataset has only one mode; regenerating both modes.")

if dataset_needs_generation:
    ds = generate_and_save_dataset(
        name=DATASET_NAME,
        subdir=DATASET_SUBDIR,
        n_samples=N_SAMPLES,
        n_fd_points=N_FD_POINTS,
        n_profile_features=N_PROFILE_FEATURES,
        min_pieces=MIN_PIECES,
        max_pieces=MAX_PIECES,
        theta=THETA,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=SEED,
        feature_kind=FEATURE_KIND,
        min_width=MIN_PIECE_WIDTH,
        split_strategy=SPLIT_STRATEGY,
    )
else:
    ds = existing

train_data = ds["train"]["constant"]
xi = train_data["xi"]
print(f"Dataset: {DATASET_NAME}")
print(f"Grid: {len(xi)} points")
print(f"Profile features: {train_data['eps_ratios'].shape[1]}")
print(f"Splits: {len(ds['train']['constant']['b'])} train / "
      f"{len(ds['val']['constant']['b'])} val / "
      f"{len(ds['test']['constant']['b'])} test")

# %% [markdown]
# ## 3. Similarity Analysis and Leakage Audit

# %%
similarity = bubble_similarity_analysis(ds, mode="constant", verbose=False)
cross = similarity["cross"]["train_vs_test"]
stats = cross["stats"]
max_train_test_similarity = cross["max_similarity"]

print("=" * 72)
print("LEAKAGE AUDIT")
print("=" * 72)
print(f"Maximum train/test similarity: {stats['max_sim_max']:.6f}")
print(f"Mean maximum train/test similarity: {stats['max_sim_mean']:.6f}")
print(f"Test twins above theta={THETA}: "
      f"{100*np.mean(max_train_test_similarity > THETA):.2f}%")
print(f"Train effective rank: "
      f"{similarity['within']['train']['effective_rank']['effective_rank']:.3f}")
split_strategy = ds["metadata"].get("split_strategy", "no_twin_shape")
if split_strategy == "no_twin_shape":
    assert not np.any(max_train_test_similarity > THETA)
else:
    print(f"[info] split_strategy={split_strategy}: shape twins across splits are "
          f"allowed by design; leakage guard skipped.")

# %% [markdown]
# ## 4. Visualize Diffusion Profiles, Solution Shapes, and Split Geometry

# %%
if not NO_PLOTS:
    colors = {"train": "C0", "val": "C1", "test": "C3"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for split_name, color in colors.items():
        data = ds[split_name]["constant"]
        for row in data["b"][: min(40, len(data["b"]))]:
            axes[0, 0].plot(xi, row, color=color, alpha=0.12)
        axes[0, 1].scatter(
            data["eps_ratios"][:, 0], data["eps_ratios"][:, -1],
            s=8, alpha=0.35, color=color, label=split_name,
        )
    axes[0, 0].set_title("Normalized Darcy solution shapes")
    axes[0, 0].set_xlabel("xi")
    axes[0, 0].set_ylabel("u / u(0.5L)")
    axes[0, 1].set_title("Diffusion-profile feature extremes")
    axes[0, 1].set_xlabel("left profile feature")
    axes[0, 1].set_ylabel("right profile feature")
    axes[0, 1].legend()

    axes[1, 0].hist(max_train_test_similarity, bins=40, color="C3", alpha=0.8)
    axes[1, 0].axvline(THETA, color="k", ls="--", label="twin threshold")
    axes[1, 0].set_title("Best train similarity for test shapes")
    axes[1, 0].set_xlabel("maximum cosine similarity")
    axes[1, 0].legend()

    for split_name, color in colors.items():
        data = ds[split_name]["constant"]
        axes[1, 1].scatter(
            np.mean(data["eps_ratios"], axis=1),
            np.std(data["eps_ratios"], axis=1),
            s=8, alpha=0.35, color=color, label=split_name,
        )
    axes[1, 1].set_title("Profile mean versus variation")
    axes[1, 1].set_xlabel("mean normalized profile feature")
    axes[1, 1].set_ylabel("profile feature standard deviation")
    axes[1, 1].legend()
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Build the KAN Model

# %%
model = MultiKANBubble1D(
    n_bubbles=2,
    n_hidden=N_HIDDEN,
    n_grid=N_GRID,
    n_eps=train_data["eps_ratios"].shape[1],
    eps_transform=EPS_TRANSFORM,
).to(DEVICE)
print(f"KAN parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Training samples: {len(train_data['pe'])}")

# %% [markdown]
# ## 6. Train the Model
#
# The model learns normalized Darcy solution shapes from the sampled diffusion
# profile. The gradient term is disabled, as in the stable training workflow.

# %%
# Protect against a stale ``ds`` variable when only this cell is rerun in
# Colab. Both P1 source modes must exist before training starts.
if not all(mode in ds.get("train", {}) for mode in ("constant", "xi")):
    print("Stale single-mode dataset detected before training; regenerating.")
    ds = generate_and_save_dataset(
        name=DATASET_NAME,
        subdir=DATASET_SUBDIR,
        n_samples=N_SAMPLES,
        n_fd_points=N_FD_POINTS,
        n_profile_features=N_PROFILE_FEATURES,
        min_pieces=MIN_PIECES,
        max_pieces=MAX_PIECES,
        theta=THETA,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=SEED,
        feature_kind=FEATURE_KIND,
        min_width=MIN_PIECE_WIDTH,
        split_strategy=SPLIT_STRATEGY,
    )
    train_data = ds["train"]["constant"]

history = {}
if RUN_TRAINING:
    t0 = time.time()
    history = train_multi_bubble_on_dataset(
        model,
        ds["train"],
        mode_names=("constant", "xi"),
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        grad_weight=0.0,
        n_quad=TRAIN_QUAD,
        verbose=True,
        device=DEVICE,
        lr_scheduler=LR_SCHEDULER,
    )
    sync()
    print(f"Training time: {time.time() - t0:.1f} s")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
elif MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded model: {MODEL_PATH}")
else:
    print("Training skipped and no checkpoint found; evaluating an untrained model.")

# %% [markdown]
# ## 7. Plot Training History

# %%
if not NO_PLOTS:
    plt.figure(figsize=(6, 4))
    if history:
        for mode_name, losses in history.items():
            plt.semilogy(losses, label=mode_name)
        plt.title(f"Darcy KAN training loss: {min(min(v) for v in history.values()):.4e}")
        plt.xlabel("epoch")
        plt.ylabel("value MSE")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "training skipped", ha="center", va="center")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Test on Train, Validation, and Leak-Free Test Shapes

# %%
def evaluate_split(data, mode_index, chunk_size=64):
    model.eval()
    n, q = len(data["b"]), len(data["xi"])
    xi_t = torch.tensor(data["xi"], dtype=torch.float32, device=DEVICE)
    eps = torch.tensor(data["eps_ratios"], dtype=torch.float32, device=DEVICE)
    prediction = np.empty((n, q), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            m = stop - start
            xi_flat = xi_t.unsqueeze(0).expand(m, -1).reshape(-1)
            pe_flat = torch.zeros(m * q, dtype=torch.float32, device=DEVICE)
            rho_flat = torch.zeros(m * q, dtype=torch.float32, device=DEVICE)
            eps_flat = eps[start:stop].unsqueeze(1).expand(-1, q, -1).reshape(m * q, -1)
            pred = model.bubbles[mode_index](
                xi_flat, pe_flat, rho_flat, eps_ratios=eps_flat,
            ).reshape(m, q)
            prediction[start:stop] = pred.cpu().numpy()
    target = torch.tensor(data["b"], dtype=torch.float32)
    pred_t = torch.tensor(prediction)
    error = pred_t - target
    rmse = torch.sqrt(torch.mean(error * error, dim=1)).numpy()
    relative_l2 = (
        torch.linalg.vector_norm(error, dim=1) /
        torch.clamp(torch.linalg.vector_norm(target, dim=1), min=1e-12)
    ).numpy()
    return prediction, rmse, relative_l2


errors = {}
for split_name in ("train", "val", "test"):
    errors[split_name] = {}
    for mode_index, mode_name in enumerate(("constant", "xi")):
        prediction, rmse, relative_l2 = evaluate_split(
            ds[split_name][mode_name], mode_index,
        )
        errors[split_name][mode_name] = {
            "prediction": prediction,
            "rmse": rmse,
            "relative_l2": relative_l2,
        }
        print(f"{split_name:>5} / {mode_name:>8}: mean RMSE={rmse.mean():.4e}, "
              f"mean relative L2={relative_l2.mean():.4e}, "
              f"worst relative L2={relative_l2.max():.4e}")

if not NO_PLOTS:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, mode_name in enumerate(("constant", "xi")):
        for split_name, color in {"train": "C0", "val": "C1", "test": "C3"}.items():
            data = ds[split_name][mode_name]
            axes[0, col].scatter(
                data["eps_ratios"][:, 0], errors[split_name][mode_name]["rmse"],
                s=8, alpha=0.35, color=color, label=split_name,
            )
            axes[1, col].hist(
                errors[split_name][mode_name]["relative_l2"], bins=40,
                alpha=0.4, color=color, label=split_name,
            )
        axes[0, col].set_title(f"{mode_name}: per-sample RMSE")
        axes[0, col].set_xlabel("left profile feature")
        axes[0, col].set_ylabel("RMSE")
        axes[1, col].set_title(f"{mode_name}: relative L2 error")
        axes[1, col].set_xlabel("relative L2")
        axes[0, col].legend()
        axes[1, col].legend()
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Representative KAN Versus Reference Shapes

# %%
representatives = [("train", 0), ("train", -1), ("test", 0), ("test", -1)]
for mode_name in ("constant", "xi"):
    for split_name, index in representatives:
        data = ds[split_name][mode_name]
        i = index if index >= 0 else len(data["b"]) + index
        rel = errors[split_name][mode_name]["relative_l2"][i]
        print(f"{mode_name:>8} {split_name}[{i}] relative L2 error = {rel:.4e}")

if not NO_PLOTS:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for row, mode_name in enumerate(("constant", "xi")):
        for col, (split_name, index) in enumerate(representatives):
            data = ds[split_name][mode_name]
            i = index if index >= 0 else len(data["b"]) + index
            ax = axes[row, col]
            ax.plot(xi, data["b"][i], "k--", label="reference FD")
            ax.plot(xi, errors[split_name][mode_name]["prediction"][i],
                    "C3", label="KAN")
            ax.set_title(f"{mode_name}, {split_name}[{i}]")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.supxlabel("xi")
    fig.supylabel("normalized solution")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. Physical Darcy Solutions for Representative Profiles
#
# The network predicts normalized shapes. For this diagnostic, the reference
# midpoint scale is restored to compare physical solutions without hiding the
# shape-learning error behind amplitude normalization.

# %%
if not NO_PLOTS:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    piece_edges = ds["metadata"]["piece_edges"]
    piece_values = ds["metadata"]["piece_values"]
    split_indices = ds["metadata"]["split_indices"]
    for row, mode_name in enumerate(("constant", "xi")):
      for col, (split_name, index) in enumerate(representatives):
        data = ds[split_name][mode_name]
        i = index if index >= 0 else len(data["b"]) + index
        pool_index = split_indices[split_name][i]
        profile = PiecewiseDiffusion(
            np.asarray(piece_edges[pool_index]),
            np.asarray(piece_values[pool_index]),
        )
        source = 1.0 if mode_name == "constant" else lambda x, L=float(data["length"][i]): x / L
        reference = solve_darcy_1d(profile, length=float(data["length"][i]),
                                   source=source, n_points=len(xi))
        predicted = (errors[split_name][mode_name]["prediction"][i]
                     * reference["center"])
        x = reference["x"]
        ax = axes[row, col]
        ax.plot(x, reference["u"], "k--", label="reference FD")
        ax.plot(x, predicted, "C3", label="KAN shape + reference scale")
        ax.set_title(f"{mode_name}, {split_name}[{i}]")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.supxlabel("x")
    fig.supylabel("u(x)")
    fig.suptitle("Physical Darcy solution reconstruction")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Conclusion
#
# The test split is held out in solution-function space. Therefore the final
# test errors measure generalization to new piecewise diffusion/solution
# shapes, rather than merely testing another random sample from a profile
# family already represented by training twins.
