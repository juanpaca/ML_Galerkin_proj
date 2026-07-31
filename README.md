# ML_Galerkin_proj

ML-enhanced FE spaces for advection-diffusion-reaction PDEs. Learn
KAN-parameterized Residual-Free Bubbles (b̂ = L⁻¹(1), b̃ = L⁻¹(ξ)) and use them
as enrichment functions in P1 FEM, parameterized by Péclet and reaction numbers
(Pe, ρ).

## Problem

```
-ε u'' + β u' + σ u = f    on [0,1],   u(0) = u(1) = 0
```

with `Pe = βh/(2ε)` (advection dominance) and `ρ = σh²/ε` (reaction dominance).
When Pe >> 1 or ρ >> 1, standard P1 FEM suffers spurious oscillations.
Residual-Free Bubbles (RFB) add per-element enrichment functions that capture
sub-element behavior, eliminating oscillations without mesh refinement.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib
```

## Quick start

```python
# 1. Load the pre-generated frame-split dataset
from src.dataset_generation import load_dataset, dataset_summary
ds = load_dataset("rfb_5k_frame")     # 5000 samples: 3544/499/957 (train/val/test)
dataset_summary(ds)

# 2. Check bubble redundancy / train-test leakage in function space
from src.dataset_generation import bubble_similarity_analysis
bubble_similarity_analysis(ds, mode="constant")

# 3. Train both bubbles (value-only loss)
import torch
from src.rfb_bubble import MultiKANBubble1D
from src.dataset_generation import train_multi_bubble_on_dataset

model = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
histories = train_multi_bubble_on_dataset(
    model, ds["train"], mode_names=("constant", "xi"),
    n_epochs=700, batch_size=256, lr=1e-3, grad_weight=0.0, n_quad=80,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
)
```

## Dataset

### Existing dataset

`datasets/rfb_5k_frame_*` — the only dataset present (not tracked in git;
load it from Google Drive in Colab). Frame split, log-uniform in Pe × ρ:

| | log Pe | log ρ |
|---|---|---|
| D′ (train + val) | centered 90% | centered 90% |
| D\D′ (test) | frame/corners | frame/corners |

- 5000 samples, FD reference on 400 points, `n_fd_points=400`.
- Pe ∈ [0.31, 100], ρ ∈ [0.26, 100] (h = 1/16, β = 1).

### Dataset structure

```python
ds["train"]["constant"]   # b̂ = L⁻¹(1) mode
ds["train"]["xi"]         # b̃ = L⁻¹(ξ) mode
ds["val"]                 # same shape
ds["test"]                # frame corners
ds["metadata"]            # config + split_indices + frame_meta
```

Each mode dict contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `pe` | `(N,)` | Péclet number βh/(2ε) |
| `rho` | `(N,)` | Reaction number σh²/ε |
| `b` | `(N, n_fd)` | Bubble values on FD grid (normalized, b(0.5)=1) |
| `db` | `(N, n_fd)` | Derivative d b/dξ |
| `xi` | `(n_fd,)` | FD grid points in [0, 1] |

### Generating a new dataset

```python
from src.dataset_generation import generate_dataset, save_dataset, load_dataset

dataset = generate_dataset(
    n_samples=5000,
    strategy="log_pe_rho",         # uniform in log(Pe) × log(ρ)
    pe_range=(0.3, 100.0),
    rho_range=(0.2, 100.0),
    split_strategy="frame",        # frame | stratified | cell | random
    frame_d_prime_fraction=0.90,
    n_fd_points=400,
    seed=42,
)
path = save_dataset(dataset, name="my_dataset")
dataset = load_dataset("my_dataset")
```

`generate_dataset(config=None, **overrides)` accepts a `DatasetConfig` or keyword
overrides. Sampling strategies: `"lhs"`, `"stratified"`, `"grid"`,
`"log_pe_rho"`. Split strategies: `"frame"`, `"cell"`, `"stratified"`,
`"random"`.

## Bubble redundancy / leakage analysis

The dataset is normalized so every bubble has b(0.5) = 1. `bubble_similarity_analysis`
computes the L2 cosine similarity `C[i,j] = ⟨b_i,b_j⟩/(‖b_i‖‖b_j‖)` within each
split, the spectral effective rank (how many genuinely distinct bubble shapes the
split spans), and, for every val/test bubble, its maximum similarity to any train
bubble (function-space leakage).

```python
res = bubble_similarity_analysis(ds, mode="constant")   # or mode="xi"
res["within"]["train"]["effective_rank"]
res["cross"]["train_vs_test"]["stats"]["frac_gt_0.99"]
```

On `rfb_5k_frame` the bubbles turn out to be heavily redundant (effective rank
≈ 1.2, mean similarity ≈ 0.95) and 100% of test bubbles have a train twin with
similarity > 0.99: the frame test set is effectively in-distribution in function
space, even though its (Pe, ρ) lie outside D′.

## Model architecture

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

Each KAN edge function is a B-spline network

```
φ(x) = w_b · SiLU(x) + w_s · Σ_i c_i · B_i^k(x)
```

with quadratic B-splines (G = 8 intervals, k = 3) on [-1, 1]. Nodes sum their
incoming edges — no activation between layers.

### Model evaluation

```python
xi = torch.linspace(0, 1, 101)
b = model(xi, pe=torch.tensor(100.0), rho=torch.tensor(0.0))   # shape (101,)

# Derivative via autograd
xi_g = torch.linspace(0, 1, 101, requires_grad=True)
b = model(xi_g, pe=torch.tensor(100.0), rho=torch.tensor(0.0))
db = torch.autograd.grad(b.sum(), xi_g)[0]

# NumPy interface
b_np, db_np = model.value_grad_numpy(np.linspace(0, 1, 101), pe=100.0, rho=0.0)

# Multi-bubble model
multi = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
b_both = multi(xi, pe, rho)                    # shape (2, 101)
b_np_both, db_np_both = multi.value_grad_numpy(xi_np, 100.0, 0.0)
```

### Persistence

```python
torch.save(model.state_dict(), "models/kan_bubble.pt")

model = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
model.load_state_dict(torch.load("models/kan_bubble.pt", map_location="cpu"))
```

A trained checkpoint exists at `models/multi_kan_700ep_5k.pt` (matches
`MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)`).
`models/multi_bubble_model_1k.pt` is an old-format checkpoint and does **not**
load with the current architecture.

## Training

```python
# Two-bubble model (constant + xi modes)
histories = train_multi_bubble_on_dataset(
    model, ds["train"],
    mode_names=("constant", "xi"),
    n_epochs=700, batch_size=256, lr=1e-3,
    grad_weight=0.0,    # value-only loss; the gradient term diverges
    n_quad=80,
    device=device,
)

# Single bubble
from src.dataset_generation import train_bubble_on_dataset
losses = train_bubble_on_dataset(model.bubbles[0], ds["train"]["constant"],
                                 n_epochs=700, batch_size=256, lr=1e-3, device=device)
```

Training notes:

- **Value-only loss** (`grad_weight=0.0`): the gradient-matching term with
  `create_graph=True` diverges. Value-only MSE is stable.
- Input scaling: `Pe_s = log1p(Pe)/6`, `ρ_s = log1p(ρ)/6`, `ξ_s = 2ξ − 1`.

## Static condensation assembly

Bubbles are used as enrichment functions in P1 FEM; bubble DOFs are eliminated
element-by-element via the Schur complement:

```
A_cond = A_LL − A_Lb · inv(A_bb) · A_bL
```

```python
from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.rfb_assembly import (assemble_rfb_condensed_system,
                              recover_bubble_coefficients, RFBSolution1D)

mesh = Mesh1D(0.0, 1.0, 8)                       # 8 elements
quad = GaussLegendre(16)
pde = AdvectionDiffusion1D(1e-3, 1.0, 0.0)
pde.set_source_from_function(lambda x: np.ones_like(x))

A_cond, f_cond, local_data = assemble_rfb_condensed_system(mesh, quad, pde, model)
u_nodal = np.linalg.solve(A_cond, f_cond)

u_bubbles = recover_bubble_coefficients(u_nodal, mesh, local_data)
solution = RFBSolution1D(u_nodal, u_bubbles, mesh, model, pde)
```

## Convergence study

Sweeps element counts and compares Classical P1 vs Exact RFB (vs KAN-RFB if a
model is provided) against a fine FD reference.

```python
from src.convergence import convergence_study, print_table, plot_convergence

results = convergence_study(eps=1e-3, beta=1.0, sigma=0.0,
                            mesh_sizes=[4, 8, 16, 32, 64],   # element counts
                            kan_model=model)                  # or None
print_table(results, title="Convergence")
plot_convergence(results, save_path="convergence.png")
```

Standalone script (Classical + Exact RFB only):

```bash
python convergence_study.py
```

## Tests

```bash
python test_all.py                # 114 checks (models, dataset, training, similarity)
python test_assembly_pipeline.py  # end-to-end static condensation vs exact RFB
```

## Source files

```
src/
├── kan.py                 KAN1D edge function (B-spline + SiLU)
├── rfb_bubble.py          KANLayer (3D tensor), KANBubble1D, MultiKANBubble1D
├── rfb_local.py           FD reference solver, local_parameters(ε,β,σ,h) → (Pe,ρ)
├── rfb_exact.py           ExactRFBubbleSet1D (ground truth)
├── rfb_training.py        Low-level training helpers
├── rfb_assembly.py        Static condensation assembly
├── dataset_generation.py  Sampling → FD solves → split → train, similarity analysis
├── convergence.py         Convergence study (P1 vs exact RFB vs KAN-RFB)
├── manufactured_solutions.py  Manufactured solutions for verification
├── mesh.py, quadrature.py, pde.py, basis.py   P1 FEM infrastructure
└── errors.py              L2, H1, energy norm errors
```
