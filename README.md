# ML_Galerkin_proj

Machine Learning Project: **ML-enhanced FE spaces for advection-diffusion-reaction PDEs**.

Learn KAN-parameterized Residual-Free Bubbles (b̂ = L⁻¹(1), b̃ = L⁻¹(ξ)),
statically condensed into P1 FEM — mesh-independent, parameterized by Péclet and
reaction numbers (Pe, ρ).

## Problem

The advection-diffusion-reaction equation:

```
-ε u'' + β u' + σ u = f    on [0,1],   u(0) = u(1) = 0
```

with `Pe = βh/(2ε)` (advection dominance) and `ρ = σh²/ε` (reaction dominance).
When Pe >> 1 or ρ >> 1, standard P1 FEM suffers spurious oscillations.

**Residual-Free Bubbles (RFB)** add per-element enrichment functions that
exactly capture sub-element behavior, eliminating oscillations without mesh
refinement. We parameterize these bubbles with KANs so they generalize across
the (Pe, ρ) parameter space.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib
```

## Quick start

```python
# 1. Generate a frame-split dataset (log-uniform in Pe×ρ)
from src.dataset_generation import generate_dataset, save_dataset
dataset = generate_dataset(
    n_samples=5000,
    sampling="log_pe_rho",
    split_strategy="frame",
)

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

## Dataset generation

### Frame split (recommended)

The frame split divides the parameter domain D = log(Pe) × log(ρ) into a
**training core D'** (centered 90% subrectangle) and a **test frame D\D'**
(corners with unseen extreme Pe/ρ combinations). This evaluates genuine
out-of-distribution generalization.

```
┌──────────────────────────┐
│ T          T          T  │  T = test (corners)
│                            │
│     D' (train + val)      │
│                            │
│ T          T          T  │  D' = centered 90% subrectangle
└──────────────────────────┘
```

```python
from src.dataset_generation import generate_dataset, save_dataset, frame_split

dataset = generate_dataset(
    n_samples=5000,
    strategy="lhs",
    split_strategy="frame",       # geometric corner split
    frame_d_prime_fraction=0.90,  # D' covers central 90% in each axis
    frame_val_fraction=500/4050,  # fraction of D' held out for validation
    sigma_range=(0.0, 10000.0),
)

# D' = train+val (inner 90%),  D\D' = test (corners)
print(f"Train: {len(dataset['train']['pe'])} | "
      f"Val: {len(dataset['val']['pe'])} | "
      f"Test: {len(dataset['test']['pe'])}")
```

### Log-Pe, log-ρ sampling (recommended)

Samples uniformly in log(Pe) × log(ρ) space, then back-computes (ε, σ).
Avoids diagonal-band bias from naive (ε, σ) LHS sampling:

```python
dataset = generate_dataset(
    n_samples=5000,
    sampling="log_pe_rho",  # uniform in log(Pe)×log(ρ)
    sigma_range=(0.0, 10000.0),
)
```

### `generate_dataset()` — full API

```python
from src.dataset_generation import generate_dataset, DatasetConfig, save_dataset, load_dataset

# Option A: keyword overrides
dataset = generate_dataset(
    n_samples=5000,
    eps_range=(1e-6, 1.0),
    beta_range=(1.0, 1.0),
    sigma_range=(0.0, 10000.0),
    h=1/16,
    strategy="lhs",           # "lhs" | "stratified" | "grid"
    split_strategy="frame",   # "frame" | "cell" | "stratified" | "random"
    n_fd_points=400,
    seed=42,
)

# Option B: DatasetConfig
config = DatasetConfig(n_samples=5000, split_strategy="frame", sigma_range=(0.0, 10000.0))
dataset = generate_dataset(config)

# Save/load
path = save_dataset(dataset, name="rfb_5k_frame")
dataset = load_dataset("rfb_5k_frame")
```

### Dataset structure

```python
dataset["train"]["constant"]   # b̂ = L⁻¹(1) mode
dataset["train"]["xi"]         # b̃ = L⁻¹(ξ) mode
dataset["val"]                 # same shape, held-out cells
dataset["test"]                # same shape, frame corners
dataset["metadata"]            # full config + split indices + cell map
```

Each mode dict contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `pe` | `(N,)` | Péclet number βh/(2ε) |
| `rho` | `(N,)` | Reaction number σh²/ε |
| `b` | `(N, n_fd)` | Bubble values on FD grid (normalized, b(0.5)=1) |
| `db` | `(N, n_fd)` | Derivative d b/dξ |
| `xi` | `(n_fd,)` | FD grid points in [0, 1] |

### Split strategies

```python
# Frame (recommended): geometric split in log(Pe)×log(ρ) space
# D' = centered 90% subrectangle (train+val), D\D' = corners (test)
split_strategy="frame"

# Cell-based: entire (Pe decade, ρ range) cells held out — no leakage
split_strategy="cell"

# Stratified: preserve Pe regime proportions
split_strategy="stratified"

# Random: simple random shuffle
split_strategy="random"
```

### Sampling strategies

```python
# Log-uniform in Pe×ρ (recommended)
strategy="lhs"  # with sampling="log_pe_rho" in generate_dataset()

# LHS — uniform coverage of parameter hypercube
strategy="lhs"

# Stratified — control per-decade allocation
strategy="stratified"
n_stratified_decade_weights=[0.4, 0.3, 0.2, 0.1, 0.0, 0.0]

# Grid — full factorial
strategy="grid"
```

---

## Model architecture

### KAN Bubble (single mode)

```
Input: (Pe_s, ρ_s, ξ_s)          each scaled to [-1, 1]
    ↓
KANLayer(3 → n_hidden)            learnable edge functions
    ↓
KANLayer(n_hidden → 1)            learnable edge functions
    ↓
softplus(raw) + delta              positivity
    ↓
4·ξ·(1-ξ) · value / norm(0.5)    envelope + normalization at ξ=0.5
    ↓
Output: b(ξ) ∈ [0, 1],            b(0)=b(1)=0, b(0.5)=1
```

### KAN edge function

Each edge computes:

```
φ(x) = w_b · SiLU(x) + w_s · Σ_i c_i · B_i^k(x)
```

where `B_i^k` are quadratic B-splines (G=8 intervals, k=3) on [-1, 1].
Nodes sum their incoming edges — no activation between layers.

### Model evaluation

```python
# Single (pe, rho) pair, many ξ points
xi = torch.linspace(0, 1, 101)
b = model(xi, pe=torch.tensor(100.0), rho=torch.tensor(0.0))

# Batched: multiple (pe, rho) pairs
bs = 16
xi_flat = xi.unsqueeze(0).expand(bs, -1).reshape(-1)
pe_flat = pe_batch.unsqueeze(1).expand(-1, 101).reshape(-1)
rho_flat = rho_batch.unsqueeze(1).expand(-1, 101).reshape(-1)
b_flat = model(xi_flat, pe_flat, rho_flat)
b_reshaped = b_flat.reshape(bs, 101)

# With precomputed norm_factor (avoids redundant KAN pass)
nf = model.norm_at_mid(pe_batch, rho_batch)
b_batch = model(xi_flat, pe_flat, rho_flat, norm_factor=nf).reshape(bs, 101)

# Derivative via autograd
xi_g = torch.linspace(0, 1, 101, requires_grad=True)
b = model(xi_g, pe=torch.tensor(100.0), rho=torch.tensor(0.0))
db = torch.autograd.grad(b.sum(), xi_g, create_graph=False)[0]

# NumPy interface
xi_np = np.linspace(0, 1, 101)
b_np, db_np = model.value_grad_numpy(xi_np, pe=100.0, rho=0.0)

# Multi-bubble evaluation
multi = MultiKANBubble1D(n_bubbles=2)
b_both = multi(xi, pe, rho)       # shape: (2, 101)
b_np_both, db_np_both = multi.value_grad_numpy(xi_np, 100.0, 0.0)
```

### Model persistence

```python
# Save
torch.save(model.state_dict(), "models/kan_bubble.pt")

# Load (must match architecture)
model = KANBubble1D(n_hidden=5, n_grid=8, spline_order=3)
model.load_state_dict(torch.load("models/kan_bubble.pt", map_location="cuda"))
model.eval()
```

---

## Training

### High-level API

```python
from src.dataset_generation import train_multi_bubble_on_dataset, train_bubble_on_dataset
from src.rfb_bubble import MultiKANBubble1D, KANBubble1D
import torch

# Two-bubble model (constant + xi modes)
model = MultiKANBubble1D(n_bubbles=2, n_hidden=5, n_grid=8, spline_order=3)
histories = train_multi_bubble_on_dataset(
    model,
    dataset["train"],
    mode_names=("constant", "xi"),
    n_epochs=700,
    batch_size=256,
    lr=1e-3,
    grad_weight=0.0,    # gradient-matching weight (0 = value-only; gradient diverges)
    n_quad=80,
    verbose=True,
    device=torch.device("cuda"),
)

# Single bubble
model = KANBubble1D(n_hidden=10, n_grid=8, spline_order=3)
losses = train_bubble_on_dataset(
    model,
    dataset["train"]["constant"],
    n_epochs=700, batch_size=256, lr=1e-3,
    device=torch.device("cuda"),
)
```

### Training notes

- **Value-only loss** (`grad_weight=0.0`): gradient term with `create_graph=True` causes
  divergence. Value-only MSE loss is stable and recommended.
- **Input scaling**: `Pe_s = log1p(Pe)/6`, `ρ_s = log1p(ρ)/6`, `ξ_s = 2ξ−1` → all in [-1, 1].
- **Envelope normalization** at ξ=0.5 is evaluated per unique (Pe,ρ) pair (not per quadrature
  point).
- **GPU speedup**: KANLayer uses 3D tensor + `F.linear` — ~20-42× faster than per-edge loops.
  Full training ~1 min for 700 epochs on T4.

---

## Static condensation assembly

Learned bubbles are used as enrichment functions in a P1 FEM. Static condensation
eliminates bubble DOFs element-by-element via the Schur complement:

```
A_cond = A_LL − A_Lb · inv(A_bb) · A_bL
```

```python
from src.rfb_assembly import assemble_rfb_condensed_system

pe, rho = 312.5, 0.0
A_cond, f_cond = assemble_rfb_condensed_system(pe, rho, bubble_provider=model)
u_nodal = torch.linalg.solve(A_cond, f_cond)

# Recover bubble coefficients for post-processing
from src.rfb_assembly import recover_bubble_coefficients, RFBSolution1D
local_data = None
u_bubbles = recover_bubble_coefficients(u_nodal, mesh, local_data)
solution = RFBSolution1D(u_nodal, u_bubbles, mesh, bubble_provider=model, pde=pde)
```

See `test_assembly_pipeline.py` for a complete end-to-end example.

---

## Convergence study

```python
from src.convergence import convergence_study, print_table, plot_convergence

result = convergence_study(
    eps=1e-4, beta=1.0, sigma=0.0, f_func=lambda x: 1.0,
    h_values=[1/8, 1/16, 1/32, 1/64],
    bubble_provider=model,   # trained MultiKANBubble1D
    device=device,
)
print_table(result)
plot_convergence(result, save_path="convergence.png")
```

---

## Colab usage (GPU training)

Datasets are stored on Google Drive and loaded via symlink:

```python
# In Colab:
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/ML_Galerkin_proj/datasets
!ln -sf /content/drive/MyDrive/ML_Galerkin_proj/datasets/* /content/ML_Galerkin_proj/datasets/

# Clone repo
!git clone https://github.com/juanpaca/ML_Galerkin_proj.git
%cd ML_Galerkin_proj
!pip install torch numpy scipy matplotlib

# Load frame-split dataset and train on GPU
from src.dataset_generation import load_dataset, train_multi_bubble_on_dataset
from src.rfb_bubble import MultiKANBubble1D
import torch

device = torch.device("cuda")
dataset = load_dataset("rfb_5k_frame")
model = MultiKANBubble1D(n_bubbles=2, n_hidden=5).to(device)
histories = train_multi_bubble_on_dataset(
    model, dataset["train"], n_epochs=700, batch_size=256, device=device,
)

# Save trained model back to Drive
torch.save(model.state_dict(), "models/multi_kan_700ep_5k.pt")
!cp models/multi_kan_700ep_5k.pt /content/drive/MyDrive/ML_Galerkin_proj/models/
```

---

## Running tests

```bash
source venv/bin/activate
python test_all.py                # 98 unit tests
python test_assembly_pipeline.py  # end-to-end static condensation
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

### `MultiKANBubble1D`

| Param | Default | Description |
|-------|---------|-------------|
| `n_bubbles` | 2 | Number of bubble modes (constant + xi) |
| `n_hidden` | 5 | Width of hidden KAN layer |

### Training hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `n_epochs` | 300–700 | With early stopping |
| `batch_size` | 128–256 | GPU memory |
| `lr` | 1e-3 | Adam |
| `grad_weight` | 0.0 | Use 0 (value-only); gradient term diverges |
| `n_quad` | 80 | Target interpolation points per sample |
| `sampling` | `"lhs"` | `"lhs"` or `"log_pe_rho"` for log-uniform Pe×ρ |

---

## Source files

```
src/
├── kan.py                 KAN1D, _eval_bspline_basis, _extend_knots
├── rfb_bubble.py          KANLayer (3D tensor), KANBubble1D [3→5→1], MultiKANBubble1D
├── rfb_local.py           FD reference solver, local_parameters(ε,β,σ,h) → (Pe,ρ)
├── rfb_exact.py           ExactRFBubble1D (ground truth)
├── rfb_training.py        Low-level data generation helpers
├── rfb_assembly.py        Static condensation: A_cond = A_LL − A_Lb·inv(A_bb)·A_bL
├── dataset_generation.py  Full pipeline: sampling → FD → frame split → train
├── convergence.py         Convergence study: P1 vs exact RFB vs KAN-RFB
├── manufactured_solutions.py  Manufactured solutions for verification
├── mesh.py                P1 mesh utilities
├── quadrature.py          Gauss–Legendre quadrature on [0,1]
├── pde.py                 PDE coefficient handling
├── basis.py               Piecewise linear hat functions
└── errors.py              L2, H1, energy norm errors
```

### Trained models

Models are saved to `models/` (created on first save). Not tracked in git.
Upload to Google Drive for Colab use.
