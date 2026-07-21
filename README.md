# ML_Galerkin_proj

Machine Learning Project: **ML-enhanced FE spaces for advection-diffusion-reaction PDEs**.
Learn KAN-parameterized Residual-Free Bubbles (b̂ = L⁻¹(1), b̃ = L⁻¹(ξ)),
statically condensed into P1 FEM (mesh‑independent, via Pe, ρ).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy
```

## Quick start

```python
# 1. Generate a dataset (both modes: constant and xi)
from src.dataset_generation import generate_dataset, save_dataset
dataset = generate_dataset(n_samples=5000, sigma_range=(0.0, 10.0))

# 2. Train both bubbles
from src.dataset_generation import train_multi_bubble_on_dataset
from src.rfb_bubble import MultiKANBubble1D
import torch

model = MultiKANBubble1D(n_bubbles=2, n_hidden=5).to("cuda")
histories = train_multi_bubble_on_dataset(
    model, dataset["train"], n_epochs=700, batch_size=256,
    device=torch.device("cuda"),
)
```

Or use the existing 1000‑sample dataset:

```bash
source venv/bin/activate
python train_1k.py    # trains both modes, 400 epochs, early stopping
python train_xi.py    # xi mode only, saves model to models/
```

---

## Dataset generation

### `generate_dataset()` — full API

```python
from src.dataset_generation import generate_dataset, DatasetConfig, save_dataset, load_dataset

# Option A: keyword overrides (quick)
dataset = generate_dataset(
    n_samples=5000,           # number of (eps, beta, sigma) samples
    eps_range=(1e-6, 1.0),    # diffusion coefficient range
    beta_range=(1.0, 1.0),    # advection coefficient (fixed = Pe varies via h/eps)
    sigma_range=(0.0, 10.0),  # reaction coefficient range
    h=1/16,                   # element length
    strategy="lhs",           # "lhs" | "stratified" | "grid"
    split_strategy="cell",    # "cell" | "stratified" | "random"
    n_val_cells=3,
    n_test_cells=3,
    variable_eps_fraction=0.0,  # fraction of samples with non-constant eps
    n_fd_points=400,          # FD grid resolution for reference solves
    val_split=0.15,
    test_split=0.15,
    seed=42,
)

# Option B: DatasetConfig
config = DatasetConfig(
    n_samples=5000,
    eps_range=(1e-6, 1.0),
    sigma_range=(0.0, 10.0),
    strategy="lhs",
    split_strategy="cell",
)
dataset = generate_dataset(config)

# Save to disk (creates NPZ files + JSON metadata)
path = save_dataset(dataset, name="rfb_5k")

# Load back
dataset = load_dataset("rfb_5k")
```

### Dataset structure

```python
dataset["train"]["constant"]   # b̂ = L⁻¹(1) mode
dataset["train"]["xi"]         # b̃ = L⁻¹(ξ) mode
dataset["val"]                 # same shape, held-out cells
dataset["test"]                # same shape, held-out cells
dataset["metadata"]            # full config + split indices + cell map
dataset["scaler"]              # DataScaler (if standardize=True)
```

Each mode dict contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `pe` | `(N,)` | Péclet number βh/(2ε) |
| `rho` | `(N,)` | Reaction number σh²/ε |
| `b` | `(N, n_fd)` | Bubble values on FD grid (normalized, b(0.5)=1) |
| `db` | `(N, n_fd)` | Derivative d b/dξ |
| `xi` | `(n_fd,)` | FD grid points in [0, 1] |

### Sampling strategies

```python
# LHS — uniform coverage of the parameter hypercube
strategy="lhs"

# Stratified — control per-decade allocation for epsilon
strategy="stratified"
# With weights: allocate 40% samples to most advection-dominated decade
n_stratified_decade_weights=[0.4, 0.3, 0.2, 0.1, 0.0, 0.0]

# Grid — full factorial (systematic, can be large)
strategy="grid"
```

### Split strategies

```python
# Cell-based (default): entire (Pe decade, ρ range) cells held out
split_strategy="cell"
# Every sample from an unseen (Pe, ρ) cell is completely held out — no leakage.

# Stratified: preserve Pe regime proportions
split_strategy="stratified"

# Random: simple random shuffle
split_strategy="random"
```

### Variable ε profiles

Set `variable_eps_fraction > 0` to include samples with non-constant diffusion:

```python
dataset = generate_dataset(
    variable_eps_fraction=0.2,     # 20% of samples
    variable_eps_profile="sinusoidal",  # "sinusoidal" | "layered" | "smooth_random"
    variable_eps_n_quad=5,         # quadrature points for ε sampling
)
```

The bubble model with `n_eps > 0` receives `eps_ratios[i] = ε(ξ_i)/ε_avg` as additional input features.

---

## Training

### Using the high-level API

```python
from src.dataset_generation import train_multi_bubble_on_dataset, train_bubble_on_dataset

# Two-bubble model (constant + xi modes)
model = MultiKANBubble1D(n_bubbles=2, n_hidden=5, n_grid=8, spline_order=3)
histories = train_multi_bubble_on_dataset(
    model,
    dataset["train"],
    mode_names=("constant", "xi"),
    n_epochs=700,
    batch_size=256,
    lr=1e-3,
    grad_weight=0.0,              # gradient-matching weight (0 = value-only)
    n_quad=80,                    # quadrature points per sample
    verbose=True,
    device=torch.device("cuda"),
)

# Single bubble
model = KANBubble1D(n_hidden=10, n_grid=8, spline_order=3)
losses = train_bubble_on_dataset(
    model,
    dataset["train"]["constant"],  # single mode data dict
    n_epochs=700, batch_size=256, lr=1e-3,
    device=torch.device("cuda"),
)
```

### Using the train scripts

```bash
# Train both modes on the existing 1k dataset
python train_1k.py

# Train xi mode only and save
python train_xi.py
```

These scripts implement early stopping, cosine annealing LR schedule,
and test-set evaluation on the full FD grid.

### Model architecture

```
Input: (pe_s, rho_s, xi_s, [eps_s])  →  each scaled to [-1, 1]
    ↓
KANLayer(3+n_eps → n_hidden)         learnable edge functions
    ↓
KANLayer(n_hidden → 1)               learnable edge functions
    ↓
softplus(raw) + delta                positivity
    ↓
4·ξ·(1-ξ) · value / norm(0.5)       envelope + normalization
    ↓
Output: b(ξ) ∈ [0, 1], b(0)=b(1)=0, b(0.5)=1
```

### Model evaluation

```python
# Single (pe, rho) pair, many xi points
xi = torch.linspace(0, 1, 101)
b = model(xi, pe=torch.tensor(100.0), rho=torch.tensor(0.0))

# Batched: multiple (pe, rho) pairs
# Each pair gets all xi points
bs = 16
xi_flat = xi.unsqueeze(0).expand(bs, -1).reshape(-1)       # (bs*101,)
pe_flat = pe_batch.unsqueeze(1).expand(-1, 101).reshape(-1) # (bs*101,)
rho_flat = rho_batch.unsqueeze(1).expand(-1, 101).reshape(-1)
b_flat = model(xi_flat, pe_flat, rho_flat)
b_reshaped = b_flat.reshape(bs, 101)

# With precomputed norm_factor (avoids redundant KAN pass)
nf = model.norm_at_mid(pe_batch, rho_batch)  # (bs,)
b_batch = model(xi_flat, pe_flat, rho_flat, norm_factor=nf).reshape(bs, 101)

# Derivative via autograd
xi_g = torch.linspace(0, 1, 101, requires_grad=True)
b = model(xi_g, pe=torch.tensor(100.0), rho=torch.tensor(0.0))
db = torch.autograd.grad(b.sum(), xi_g, create_graph=False)[0]

# NumPy interface (includes derivative)
xi_np = np.linspace(0, 1, 101)
b_np, db_np = model.value_grad_numpy(xi_np, pe=100.0, rho=0.0)

# Multi-bubble evaluation
multi = MultiKANBubble1D(n_bubbles=2)
b_both = multi(xi, pe, rho)      # shape: (2, 101)
b_np_both, db_np_both = multi.value_grad_numpy(xi_np, 100.0, 0.0)
```

### Model persistence

```python
# Save
torch.save(model.state_dict(), "models/kan_bubble_constant.pt")

# Load (must create model with same architecture first)
model = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
model.load_state_dict(torch.load("models/kan_bubble_constant.pt", map_location="cuda"))
model.eval()

# Multi-bubble
torch.save(multi.state_dict(), "models/multi_kan_bubble.pt")
```

---

## Static condensation assembly

The learned bubbles are designed to be used as enrichment functions in a
P1 finite element method. The static condensation eliminates bubble DOFs
element-by-element.

```python
from src.rfb_assembly import assemble_rfb_condensed_system

# Given mesh parameters
pe, rho = 312.5, 0.0  # from local_parameters(eps=1e-4, beta=1, sigma=0, h=1/16)
A_cond, f_cond = assemble_rfb_condensed_system(pe, rho, bubble_provider=model)

# Solve
u_nodal = torch.linalg.solve(A_cond, f_cond)

# Recover bubble coefficients for post-processing
from src.rfb_assembly import recover_bubble_coefficients
local_data = None  # returned by assemble_rfb_condensed_system
u_bubbles = recover_bubble_coefficients(u_nodal, mesh, local_data)

# Reconstruct enriched solution
from src.rfb_assembly import RFBSolution1D
solution = RFBSolution1D(u_nodal, u_bubbles, mesh, bubble_provider=model, pde=pde)
```

See `test_assembly_pipeline.py` for a complete end-to-end example that
compares the learned bubble assembly against the exact RFB solution.

---

## Running tests

```bash
source venv/bin/activate
python test_all.py                # 98 unit tests
python test_assembly_pipeline.py  # end-to-end static condensation
```

---

## Colab usage

In a Colab notebook:

```python
import os
# Clone (first run) or pull (subsequent runs)
if not os.path.exists("/content/ML_Galerkin_proj"):
    !git clone https://github.com/juanpaca/ML_Galerkin_proj.git
    %cd /content/ML_Galerkin_proj
else:
    %cd /content/ML_Galerkin_proj
    !git pull

import sys; sys.path.insert(0, "/content/ML_Galerkin_proj")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load existing 1k dataset and train
from src.dataset_generation import load_dataset, train_multi_bubble_on_dataset
from src.rfb_bubble import MultiKANBubble1D

dataset = load_dataset("rfb_1k")
model = MultiKANBubble1D(n_bubbles=2, n_hidden=10).to(device)
histories = train_multi_bubble_on_dataset(
    model, dataset["train"], n_epochs=700, batch_size=256, device=device,
)
```

To generate a new (larger) dataset on Colab (this calls the FD solver):

```python
from src.dataset_generation import generate_dataset, save_dataset
dataset = generate_dataset(n_samples=2000, sigma_range=(0.0, 10.0))
path = save_dataset(dataset, name="rfb_2k")
```

---

## Parameter reference

### `KANBubble1D`

| Param | Default | Description |
|-------|---------|-------------|
| `n_hidden` | 5 | Width of hidden KAN layer |
| `n_grid` | 8 | Number of B-spline intervals |
| `spline_order` | 3 | B-spline order (k=3 = quadratic) |
| `delta` | 1e-4 | Softplus offset for numerical stability |
| `n_eps` | 0 | Number of ε profile samples (0 = constant) |

Parameter count: `2 × n_hidden × (3+n_eps + n_hidden) × (n_grid + spline_order - 1) + 2 × n_hidden × (3+n_eps + 1)`

For the default (hidden=5, grid=8, k=3, n_eps=0): **240 parameters**.

### Training hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `n_epochs` | 300–700 | With early stopping |
| `batch_size` | 128–256 | GPU memory |
| `lr` | 1e-3 | Adam or AdamW |
| `grad_weight` | 0.0 | Use 0 (value-only); gradient term diverges |
| `n_quad` | 80 | Target interpolation points per sample |

---

## Architecture

```
src/
├── kan.py                 KAN1D, _eval_bspline_basis, _extend_knots
├── rfb_bubble.py          KANLayer, KANBubble1D, MultiKANBubble1D
├── rfb_local.py           solve_reference_rfb (FD), local_parameters
├── rfb_exact.py           ExactRFBubble1D (ground truth)
├── rfb_training.py        Low‑level data generation helpers
├── rfb_assembly.py        Static condensation: A_cond = A_LL − A_Lb·inv(A_bb)·A_bL
├── dataset_generation.py  Full pipeline: sampling → FD → split → train
├── mesh.py                P1 mesh utilities
├── quadrature.py          Gauss–Legendre quadrature on [0,1]
├── pde.py                 PDE coefficient handling
├── basis.py               Piecewise linear hat functions
└── errors.py              L2, H1, energy norm errors
```
