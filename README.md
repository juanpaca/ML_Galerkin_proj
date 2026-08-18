# ML_Galerkin_proj

ML-enhanced finite-element spaces for **advection–diffusion–reaction** PDEs.
A KAN learns **Residual-Free Bubbles** — sub-element enrichment functions that
are statically condensed into a P1 FEM — parameterized by the Péclet number
`Pe` and the reaction number `ρ`.

The whole repository is organized around a standard ML loop:

```
0. Test that everything works   test_all.py, test_assembly_pipeline.py
1. Generate data                data_generation.py
2. Verify the data              check_fd_accuracy.py, dataset_summary, similarity analysis
3. Train the KAN                train_multi_bubble_on_dataset / tutorial.py
4. Evaluate the results         OOD test set, assembly, convergence_study.py
```

## Problem

```
-ε u'' + β u' + σ u = f   on [0,1],   u(0) = u(1) = 0
```

with `Pe = βh/(2ε)` (advection dominance) and `ρ = σh²/ε` (reaction
dominance). When `Pe ≫ 1` or `ρ ≫ 1`, plain P1 FEM oscillates. Residual-Free
Bubbles add per-element enrichment functions that resolve the sub-element
behavior and are eliminated locally (static condensation), so the method stays
mesh-independent and oscillation-free.

### Variable diffusion

Diffusion may be constant, array-valued, or a vectorized callable:

```python
import numpy as np

pde.set_diffusion_from_function(
    lambda x: np.where(x < 0.5, 0.1, 1.0)
)
```

The reference solver uses the conservative operator `-(ε b')'` with harmonic
face diffusion, including for piecewise profiles. Variable-diffusion training
uses `generate_rfb_training_data_variable_eps`; the KAN receives a fixed-size
sampled profile through `n_eps`. Use the same `n_eps` when assembling a trained
variable-diffusion model.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib
```

Every script is run through the venv: `venv/bin/python <script>.py`.

---

## 0. Test that everything works

Run these first — a clean checkout should pass all of them.

```bash
venv/bin/python test_all.py                # 128 checks: KAN, bubbles, training, similarity, FD-vs-analytic
venv/bin/python test_assembly_pipeline.py  # end-to-end static condensation (untrained KAN vs exact RFB)
```

Optional full walkthrough (also works in Google Colab, see
[tutorial.py](#5-full-walkthrough)):

```bash
venv/bin/python tutorial.py
```

---

## 1. Data generation

### The shipped dataset

`datasets/rfb_5k_noleak_*` is ready to use:

| | value |
|---|---|
| Pool | 5000 bubbles, log-uniform in Pe × ρ |
| Ranges | Pe ∈ [0.3, 100], ρ ∈ [0.2, 100] (h = 1/16, β = 1) |
| Reference | FD solution on **3200 points** (`n_fd_points=3200`) |
| Split | 1298 train / 750 val / 500 test |
| Split strategy | `no_twin_shape` — shape-based, leak-free |
| Leakage | 0% of test bubbles have a train twin (worst train–test similarity 0.98) |
| OOD by design | train Pe ∈ [2.3, 16.7] (diffusion), test Pe ∈ [51, 100] (boundary layer) |

The train set contains rounded, diffusion-dominated bubbles; the test set
contains sharp boundary-layer bubbles the model has never seen. ρ is fully
shared between splits.

### Generate a new dataset (CLI)

```bash
venv/bin/python data_generation.py \
    --n-samples 5000 \
    --n-fd-points 3200 \
    --theta 0.99 \
    --name rfb_5k_noleak
```

Constant diffusion is the default. To generate a fully piecewise-variable
diffusion pool, use:

```bash
venv/bin/python data_generation.py \
    --piecewise-diffusion \
    --variable-eps-n-quad 8 \
    --name rfb_5k_piecewise
```

The equivalent general switch is
`--diffusion-profile {constant,sinusoidal,layered,smooth_random}`. Nonconstant
profiles default to `--variable-eps-fraction 1`; use a fraction in `[0, 1]`
to mix constant and variable samples.

### Darcy variable-diffusion tutorial

For the source-only conservative problem
`-(epsilon(x)u')' = f` on `(0, L)`, run:

```bash
venv/bin/python tutorial_darcy_variable.py --no-plot
```

This creates `datasets/data_darcy_variable/`, stores piecewise profile
geometry and normalized solution shapes, and audits train/test shape twins.
The generated default dataset contains 5,000 profiles and uses every profile
that remains safe after the no-twin filter for training. Use
`--train-frac` only when a fixed training size is required. The default has
0% test twins at the `0.99` similarity threshold. The script prints the
external KAN training recipe but does not train locally.

| Flag | Default | Meaning |
|---|---|---|
| `--n-samples` | 5000 | Pool size before the no-twin filtering |
| `--n-fd-points` | 400 | FD grid for the reference bubbles. **Use 3200**: 400 under-resolves the Pe>50 boundary layer |
| `--pe-range` / `--rho-range` | (0.3,100) / (0.2,100) | Parameter ranges (log-uniform) |
| `--theta` | 0.99 | Twin-similarity threshold for the no-twin split |
| `--train-frac` / `--val-frac` / `--test-frac` | 0.60 / 0.15 / 0.25 | Target split sizes |
| `--seed` | 42 | RNG seed |
| `--name` | `rfb_5k_noleak` | Dataset name (`datasets/<name>_*`) |

The pipeline: sample a log-uniform pool → solve every bubble on the FD grid →
drop near-duplicate shapes → shape-based no-twin split → save.

### Generate a new dataset (API)

```python
from src.dataset_generation import generate_dataset, save_dataset, load_dataset

dataset = generate_dataset(
    n_samples=5000,
    strategy="log_pe_rho",
    pe_range=(0.3, 100.0),
    rho_range=(0.2, 100.0),
    n_fd_points=3200,
    seed=42,
)
save_dataset(dataset, name="my_dataset")
dataset = load_dataset("my_dataset")
```

### Dataset structure

```python
ds["train"]["constant"]   # b̂ = L⁻¹(1) mode
ds["train"]["xi"]         # b̃ = L⁻¹(ξ) mode
ds["val"] / ds["test"]    # same layout
ds["metadata"]            # config, split_indices, similarity_theta, ...
```

Each mode dict contains:

| Key | Shape | Description |
|---|---|---|
| `pe` | `(N,)` | Péclet number βh/(2ε) |
| `rho` | `(N,)` | Reaction number σh²/ε |
| `b` | `(N, n_fd)` | Bubble values on the FD grid (normalized, b(0.5)=1) |
| `db` | `(N, n_fd)` | Derivative db/dξ |
| `xi` | `(n_fd,)` | FD grid points in [0, 1] |

---

## 2. Verify the data

### 2a. FD reference vs analytic solution

`src/rfb_analytic.py` implements the exact constant-coefficient bubble, so the
FD reference can be audited independently:

```bash
venv/bin/python check_fd_accuracy.py --name rfb_5k_noleak
```

This runs (a) a Pe × ρ sweep of FD vs analytic error, (b) a resolution study,
and (c) a full audit of the dataset, reporting per split: median/max L2 error,
boundary-layer error, % of flagged samples, and whether any bubble oscillates
(negative values). A clean dataset should show < 0.5% error everywhere and 0
oscillating bubbles.

### 2b. Summarize the split

```python
from src.dataset_generation import load_dataset, dataset_summary
ds = load_dataset("rfb_5k_noleak")
dataset_summary(ds)     # prints ranges and split counts
```

### 2c. Leakage / redundancy analysis

Bubbles are normalized so b(0.5) = 1. `bubble_similarity_analysis` computes the
L2 cosine similarity `C[i,j] = ⟨b_i,b_j⟩/(‖b_i‖‖b_j‖)` within each split, the
spectral **effective rank** (how many genuinely distinct shapes a split spans),
and — for every val/test bubble — its **maximum similarity to any train
bubble**. A test bubble with a train twin (C > 0.99) is not a genuine OOD
benchmark.

```python
from src.dataset_generation import bubble_similarity_analysis

res = bubble_similarity_analysis(ds, mode="constant")   # or mode="xi"
res["within"]["train"]["effective_rank"]                # e.g. 1.06
res["cross"]["train_vs_test"]["stats"]["frac_gt_0.99"]  # e.g. 0.0
```

Expected on `rfb_5k_noleak`: `frac_gt_0.99 = 0.0` (no test twins, worst
train–test similarity ≈ 0.98) even though the train set itself is redundant
(effective rank ≈ 1.1 — a single diffusion shape family). The old
`rfb_5k_frame` dataset, by contrast, shows **100%** of test bubbles with a
train twin and is kept only for comparison.

Similarity uses trapezoidal quadrature. Cross-split maximum similarities and
the production no-twin split are computed blockwise, so they do not require a
full `N × N` matrix. Request `return_matrices=True` only for small diagnostic
datasets.

---

## 3. Training

### Model

```python
import torch
from src.rfb_bubble import MultiKANBubble1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
model.to(device)     # 960 params total (480 per mode)
```

For large one-dimensional FEM systems, pass `sparse_output=True` to
`assemble_classical_system` or `assemble_rfb_condensed_system` to receive a
CSR matrix. Dense output remains the compatibility default.

### Train both modes

```python
from src.training import train_multi_bubble_on_dataset

histories = train_multi_bubble_on_dataset(
    model, ds["train"],
    mode_names=("constant", "xi"),
    n_epochs=700, batch_size=256, lr=1e-3,
    grad_weight=0.0,   # value-only loss — the gradient term diverges
    n_quad=100,
    device=device,
)                    # returns {"constant": [loss...], "xi": [loss...]}
```

Or a single mode:

```python
from src.dataset_generation import train_bubble_on_dataset
losses = train_bubble_on_dataset(model.bubbles[0], ds["train"]["constant"],
                                 n_epochs=700, batch_size=256, lr=1e-3, device=device)
```

### Save / load

```python
torch.save(model.state_dict(), "models/multi_kan_700ep_5k_noleak.pt")

model = MultiKANBubble1D(n_bubbles=2, n_hidden=10, n_grid=8, spline_order=3)
model.load_state_dict(torch.load("models/multi_kan_700ep_5k_noleak.pt", map_location="cpu"))
model.to(device)
```

Training notes:

- **Value-only loss** (`grad_weight=0.0`): the gradient-matching term with
  `create_graph=True` diverges. Value-only MSE is stable.
- Input scaling: `Pe_s = log1p(Pe)/6`, `ρ_s = log1p(ρ)/6`, `ξ_s = 2ξ − 1`.
- The tutorial (`tutorial.py`) contains a timed benchmark (expect < 1 ms per
  single forward on GPU) and loss-curve plots.

---

## 4. Results: evaluate the trained model

### 4a. Bubble-level error on the OOD test set

Every test bubble is extrapolation (Pe 51–100 vs train ≤ 16.7). Compare the
KAN output against the exact bubble for a (Pe, ρ):

```python
import numpy as np
from src.rfb_exact import ExactRFBubbleSet1D

xi = np.linspace(0, 1, 201)
pe, rho = 95.0, 50.0
b_kan, _ = model.value_grad_numpy(xi, pe, rho)           # (2, 201): [b̂, b̃]

H = 1/16
eps = H / (2 * pe)                                     # βH/(2Pe), with β=1
sigma = rho * eps / H**2
exact = ExactRFBubbleSet1D(eps, 1.0, sigma, H,
                           residual_modes=("constant", "xi"), n_points=4000)
b_ex, _ = exact.value_grad_numpy(xi, pe, rho)
mse = np.mean((b_kan - b_ex) ** 2, axis=1)
```

### 4b. PDE-level error: static condensation assembly

Compare classical P1, exact-RFB and KAN-RFB against a fine FD reference:

```python
from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.rfb_assembly import (
    assemble_classical_system, assemble_rfb_condensed_system,
    recover_bubble_coefficients, RFBSolution1D,
)
from src.errors import compute_l2_error

mesh = Mesh1D(0.0, 1.0, 16)
quad = GaussLegendre(16)
pde = AdvectionDiffusion1D(eps, 1.0, sigma)
pde.set_source_from_function(lambda x: np.ones_like(x))

A_cond, f_cond, local_data = assemble_rfb_condensed_system(mesh, quad, pde, model)
u_nodal = np.linalg.solve(A_cond, f_cond)

u_bubbles = recover_bubble_coefficients(u_nodal, mesh, local_data)
solution = RFBSolution1D(u_nodal, u_bubbles, mesh, model, pde)
l2, norm = compute_l2_error(solution, exact_u)           # compare to FD reference
```

### 4c. Convergence study

```bash
venv/bin/python convergence_study.py              # Classical + Exact RFB only
venv/bin/python convergence_study.py --train-kan  # also train KAN on-the-fly
```

Or in code:

```python
from src.convergence import convergence_study, print_table, plot_convergence
results = convergence_study(eps=1e-3, beta=1.0, sigma=0.0,
                            mesh_sizes=[4, 8, 16, 32, 64],
                            kan_model=model)          # or None
print_table(results, title="Convergence")
plot_convergence(results, save_path="convergence.png")
```

---

## 5. Full walkthrough

`tutorial.py` runs the whole loop end-to-end (standalone or in Colab):

| Section | Content |
|---|---|
| 0–1 | Colab setup + imports |
| 2 | Load `rfb_5k_noleak`, summary |
| 2B | **Leakage confirmation plots**: train-vs-test similarity heatmaps, twin-rate histograms for the leaking `rfb_5k_frame` vs the leak-free no-leak split |
| 3 | Bubble shapes: rounded low-Pe train bubbles vs boundary-layer high-Pe test bubbles |
| 4 | Split visualization + overlap check + Pe-range table |
| 5 | Timed forward benchmark + training (700 epochs, both modes) |
| 5D | Per-sample RMSE on the OOD test set, colored by ρ |
| 6–9 | KAN vs exact bubble shapes and PDE solutions on 2 interpolation + 2 extrapolation samples, with assembly errors |

In Colab, `git clone` + `git pull`, mount Drive and symlink
`datasets/` automatically.

---

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

`model` is the `MultiKANBubble1D` from section 3 — row 0 = b̂ (constant), row 1 = b̃ (xi):

```python
xi = torch.linspace(0, 1, 101)
b = model(xi, pe=torch.tensor(100.0), rho=torch.tensor(0.0))   # shape (2, 101)

# Derivative via autograd
xi_g = torch.linspace(0, 1, 101, requires_grad=True)
b = model(xi_g, pe=torch.tensor(100.0), rho=torch.tensor(0.0))
db = torch.autograd.grad(b.sum(), xi_g)[0]                     # shape (2, 101)

# NumPy interface
b_np, db_np = model.value_grad_numpy(np.linspace(0, 1, 101), pe=100.0, rho=0.0)

# A single mode only
b_const = model.bubbles[0](xi, pe=torch.tensor(100.0), rho=torch.tensor(0.0))  # shape (101,)
```

## Notes / known issues

- `models/multi_kan_700ep_5k.pt` was trained on the **leaking** frame dataset
  (`rfb_5k_frame`); retrain on `rfb_5k_noleak` for a genuine OOD benchmark.
- `models/multi_bubble_model_1k.pt` is an old-format checkpoint and does **not**
  load with the current architecture.
- `datasets/rfb_5k_frame_*` is legacy: 100% of its test bubbles have a training
  twin (function-space leakage). It is kept only so the tutorial can plot the
  leak vs the no-leak fix.

## Source files

```
data_generation.py       CLI: pool generation + no-twin shape split + save
check_fd_accuracy.py     CLI: FD-vs-analytic sweep, resolution study, dataset audit
tutorial.py              Full guided walkthrough (Colab-compatible)
test_all.py              Unit tests (KAN, bubbles, training, similarity, FD-vs-analytic)
test_assembly_pipeline.py  End-to-end static condensation test
convergence_study.py     Convergence study CLI

src/
├── kan.py                 KAN1D edge function (B-spline + SiLU)
├── rfb_bubble.py          KANBubble1D, MultiKANBubble1D
├── rfb_local.py           FD reference solver, local_parameters(ε,β,σ,h) → (Pe,ρ)
├── rfb_analytic.py        Exact constant-coefficient bubble — ground truth for FD validation
├── rfb_exact.py           ExactRFBubbleSet1D (FD ground truth at runtime)
├── rfb_training.py        Low-level training helpers
├── rfb_assembly.py        Static condensation assembly
├── dataset_generation.py  Sampling → FD solves → split → batch training, similarity analysis
├── convergence.py         Convergence study (P1 vs exact RFB vs KAN-RFB)
├── manufactured_solutions.py  Manufactured solutions for verification
├── mesh.py, quadrature.py, pde.py, basis.py   P1 FEM infrastructure
└── errors.py              L2, H1, energy norm errors
```
