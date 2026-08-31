# ML_Galerkin_proj

ML-enhanced finite-element spaces for **variable-diffusion Darcy problems**.

A KAN learns **Residual-Free Bubbles** — sub-element enrichment functions b̂ = L⁻¹(1),
b̃ = L⁻¹(ξ) — statically condensed into a P1 FEM. The diffusion coefficient
ε(x) is *piecewise constant* and varies from sample to sample, so the bubbles
are parameterized by the profile itself (not by fixed Péclet/reaction numbers).
This makes the method **mesh-independent by design**: all resolution lives in
the bubbles, P1 is only the condenser.

The repository is organized around a standard ML loop:

```
0. Test that everything works      tests/run_all.py
1. Generate data                   data_generation_darcy_variable.py
2. Verify / audit the data         enrichment gate, no-twin / similarity analysis
3. Train the KAN                   train_multi_bubble_on_dataset / tutorial_darcy_variable.py
4. Evaluate / apply                OOD test set, static condensation, solve f = 1 + x
```

> **Scope.** This README focuses on the pure diffusion problem
> `-(ε(x) u')' = f`. Advection (β) and reaction (σ) terms — the generalized
> `-εu'' + βu' + σu = f` setup — are possible generalization directions,
> explored later, and are *not* part of the current pipeline.

## Problem

```
-(ε(x) u'(x))' = f(x)   on (0, 1),   u(0) = u(1) = 0
```

with a **positive piecewise-constant** diffusion `ε(x)` and `f ∈ span{1, x}`.
Two residual-source modes are learned:

| mode | bubble | definition |
|---|---|---|
| `constant` | b̂ | L⁻¹(1) |
| `xi` | b̃ | L⁻¹(x) |

Bubbles are full-domain solutions of the Darcy operator, then **normalized by
their midpoint** `b/b(0.5)` so the learning target is the *shape* (scale-free).
The bubbles are restricted per element onto a coarse P1 mesh and eliminated
locally (static condensation via the Schur complement `A_LL − A_Lb·A_bb⁻¹·A_bL`),
so the discretization stays oscillation-free and mesh-independent.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib
```

Every script runs through the venv: `venv/bin/python <script>.py`. For the
fastest configuration use CUDA (a GPU cuts each 1400-epoch training run from
hours to tens of minutes).

---

## 0. Test that everything works

A clean checkout should pass the whole suite:

```bash
venv/bin/python tests/run_all.py      # full pytest suite (159 checks)
```

Optional full walkthrough (generate → train → evaluate → apply):

```bash
venv/bin/python tutorial_darcy_variable.py
```

---

## 1. Data generation

### The problem setup (dataset convention)

The default training spec (`tutorial_darcy_variable.py`) is a **5-piece**
constant profile:

| parameter | value |
|---|---|
| pieces | exactly 5 |
| min piece width | 0.1 (= ℓ/10) |
| ε values | log-uniform in [0.1, 10] (contrast c = ε_max/ε_min ∈ [1, 100]) |
| FD grid | `n_fd_points = 6401` |
| profile features | `scaled_combo_v2` (8 dims) |
| split | `contrast_band` 70/15/15 |

`n_fd_points = 6401` is the resolution convention: the conservative FD solver
under-resolves thin high-contrast boundary layers at coarser grids (e.g. 801
pts → ~6e-3 rel-L2, 3201 → ~1.7e-3), so the reference bubbles are generated
at 6401 for extra headroom with the enrichment gate disabled.

### Data-generation script

```bash
venv/bin/python data_generation_darcy_variable.py \
    --name darcy_piecewise_5pc \
    --n-samples 5000 \
    --n-fd-points 6401 \
    --min-pieces 5 --max-pieces 5 \
    --min-width 0.1 \
    --eps-min 0.1 --eps-max 10.0 \
    --feature-kind scaled_combo_v2 \
    --split-strategy no_twin_shape
```

It writes `datasets/data_darcy_variable/<name>_*.npz` + `_metadata.json`.

| Flag | Default | Meaning |
|---|---|---|
| `--n-samples` | 5000 | Pool size |
| `--n-fd-points` | 801 | FD grid for reference bubbles (tutorial uses 6401) |
| `--min-pieces` / `--max-pieces` | 2 / 8 | Range of piece counts |
| `--min-width` | 0.0 | Minimum piece measure on the normalized interval |
| `--eps-min` / `--eps-max` | 0.1 / 100.0 | ε value range (log-uniform) |
| `--feature-kind` | `gauss_ratio` | Profile feature: `gauss_ratio`, `resistivity_cdf`, `scaled_combo`, `scaled_combo_v2` |
| `--split-strategy` | `no_twin_shape` | `no_twin_shape`, `random`, `contrast_band` |
| `--verify-enrichment` | on | Run the pre-training enrichment gate over the pool |
| `--gate-threshold` | 1e-2 | Max enriched rel-L2 per sample; above it → drop (`--gate-drop`) or abort |
| `--theta` | 0.99 | Similarity threshold for the no-twin split |
| `--seed` | 42 | RNG seed |

### Enrichment gate (pre-training data-quality bar)

Before training, every bubble can be audited to *enrich* the P1 Galerkin
solution. `audit_enrichment_gate` runs the **deployed assembly** — a uniform
`n_el = 8`-node P1 mesh with ONE global coefficient pair per mode, statically
condensed — against an independent fine reference (`n_gate_ref`), per source
mode. This is the same code path used in the demo/application, so it measures
bubble quality in the real consumption path.

**The reference recomputation is optional.** The default tutorial run disables
it (`verify_enrichment=False`) and instead redirects that compute budget into
generating finer reference bubbles (`n_fd_points = 6401`). To re-enable the
gate for an experiment, keep it on via the generation **CLI**
(`--verify-enrichment`, `--gate-threshold`, `--gate-n-ref`, `--gate-drop`) or
the **API** (`generate_and_save_dataset(..., verify_enrichment=True, ...)`);
it is fully tested and remains available.

Key facts:

- Enrichment of the P1 solution is **mesh-independent**: at fixed `n_fd` the
  enriched rel-L2 is flat across `n_el = 4/8/16` — bubbles carry the resolution.
- For pure diffusion with a mesh aligned to the ε-pieces, `A_Lb ≡ 0`, so
  recovery is span-only `c_e = A_bb⁻¹F_b`; the generic
  `A_bb⁻¹(F_b − A_LbᵀU)` recovery amplifies FD-gradient noise at coarse grids.
- If enabled at `n_fd=3201`/`n_gate_ref=32001`: mean ≈ 1.5e-3, p95 ≈ 5e-3,
  worst ≈ 2.2e-2; the worst ~0.5–1% are dropped by `--gate-drop` at the
  default threshold.

### Data-generation API

```python
from data_generation_darcy_variable import generate_and_save_dataset
from src.dataset_generation import load_dataset

ds = generate_and_save_dataset(
    name="darcy_piecewise_5pc", n_samples=5000, n_fd_points=6401,
    min_pieces=5, max_pieces=5, min_width=0.1,
    eps_range=(0.1, 10.0), feature_kind="scaled_combo_v2",
    split_strategy="no_twin_shape",
)
# later, just reload:
ds = load_dataset("darcy_piecewise_5pc", subdir="data_darcy_variable")
```

---

## 2. Verify / audit the data

### 2a. Enrichment gate

Tagged as a metadata report; every generated bubble satisfies the gate
(rel-L2 < threshold) unless `--gate-drop` was used to remove the offenders.

### 2b. Similarity / no-twin audit

The no-twin split reserves val/test as the most *atypical* bubble shapes and
trains on everything that is not a H¹-similarity twin of them. The similarity
metric is the **derivative-aware H¹ cosine**

```
C[i,j] = <b_i, b_j>_H1 / (||b_i||_H1 ||b_j||_H1),
<b_i, b_j>_H1 = ∫ b_i b_j + λ² ∫ b_i' b_j'         (default λ = 0.2)
```

computed on the shared trapezoid quadrature (centered-difference derivative).
The H¹ inner product distinguishes boundary-layer *sharpness* that plain L²
cosine would confuse with near-duplication. `lambda_deriv=0` reverts to L².

```python
from src.dataset_generation import bubble_similarity_analysis
res = bubble_similarity_analysis(ds, mode="constant")   # or mode="xi"
res["cross"]["train_vs_test"]["stats"]["frac_gt_0.99"]  # 0.0 on a clean set
```

### 2c. Closest H¹-cosine pair (leakage check)

`tutorial_darcy_variable.py` computes the **closest** (maximum) H¹ cosine
across (train+val) vs test — the most-similar train/val and test bubbles — and
plots them plus both ε(x) profiles (`figures/closest_pair.png`). A high
similarity here (near 1) would signal test-set leakage; a value well below the
twin threshold confirms the test set is genuinely OOD.

---

## 3. Training

### Model

```python
import torch
from src.rfb_bubble import MultiKANBubble1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiKANBubble1D(n_bubbles=2, n_hidden=32, n_grid=12,
                         n_eps=8, eps_transform="none").to(device)
```

`n_eps` = number of profile features (8 for `scaled_combo_v2`).
**Reuse the same `eps_transform` at training and at inference**; `"none"`
matches the pre-mapped `scaled_combo*` features. `pe`, `rho` are kept as
zero inputs for API compatibility — the Darcy KAN is driven by the profile
features only.

### Train both modes

```python
from src.training import train_multi_bubble_on_dataset

history = train_multi_bubble_on_dataset(
    model, ds["train"],
    mode_names=("constant", "xi"),
    n_epochs=1400, batch_size=256, lr=1e-3,
    grad_weight=0.0,     # value-only MSE — the gradient term diverges
    energy_weight=0.01,  # optional FD-derivative regularizer
    n_quad=160,
    lr_scheduler="cosine",          # per-epoch cosine annealing
    device=device,
)  # -> {"constant": [loss...], "xi": [loss...]}
```

Training notes:

- **Value-only loss** (`grad_weight=0.0`): the gradient-matching term with
  `create_graph=True` diverges; the optional `energy_weight` term is stable
  because it works on finite differences of values instead of second-order
  autograd.
- Observed GPU cost ~ tens of minutes per 1400-epoch mode at ~14k train samples
  (scale down `n_epochs`/pool size for a quick smoke run).
- The checkpoint is saved to `models/` and reloaded by setting
  `TRAIN = False` in the tutorial.

### Save / load

```python
torch.save(model.state_dict(), "models/darcy_piecewise_5pc_kan.pt")

model = MultiKANBubble1D(n_bubbles=2, n_hidden=32, n_grid=12,
                         n_eps=8, eps_transform="none")
model.load_state_dict(torch.load("models/darcy_piecewise_5pc_kan.pt",
                                 map_location="cpu"))
model.to(device)
```

---

## 4. Evaluate / apply the trained model

### 4a. Bubble-level error on the test split

The tutorial prints mean/worst relative-L2 per split and mode, and the
`figures/bubbles_train.png` panel compares KAN vs target FD bubbles (plus the
local ε(x)) for a few training samples.

### 4b. PDE-level application: solve `-(ε u')' = 1 + x`

`tutorial_darcy_variable.py` section 8 compares **Reference** (fine FD),
**Galerkin** (P1, no bubbles), **Gal+bubble(exact)**, and **Gal+bubble(KAN)**
on a coarse P1 mesh (`n_el = 8`), reporting relative L2 and H1 vs the
reference, and plots the solutions over the applied ε(x).

```python
from src.darcy_assembly import assemble_p1, assemble_enriched, eval_enriched
from src.darcy_variable import solve_darcy_1d

mesh = np.linspace(0, 1, 9)                       # n_el = 8
source = lambda x: np.asarray(x, float) + 1.0     # f = 1 + x

# reference: fine independent FD solve on the bubble grid
b_exact = np.stack([
    solve_darcy_1d(profile, source=1.0,        n_points=6401)["u_norm"],
    solve_darcy_1d(profile, source=lambda y: np.asarray(y), n_points=6401)["u_norm"],
])

# KAN bubbles predicted from the profile features (n_eps-dim)
b_kan = np.stack([predict_bubble(model, 0, xi, eps_ratios),
                  predict_bubble(model, 1, xi, eps_ratios)])

u_nodes, c = assemble_enriched(mesh, profile, source, b_kan, xi_bub, dx_bub)
u_kan      = eval_enriched(u_nodes, c, b_kan, mesh, x_bub)   # -> (n_fd,)
```

`assemble_enriched` requires `(n_fd_points − 1) % n_el == 0` so the bubble grid
aligns with the P1 mesh.

---

## 5. Full walkthrough

`tutorial_darcy_variable.py` runs the whole loop end-to-end:

| Section | Content |
|---|---|
| 1 | Load (or generate) the dataset; print split sizes / FD grid / feature count |
| 2 | Build the `MultiKANBubble1D` model |
| 3 | Train both modes (constant, xi) with cosine LR; save to `models/`. Skips training if a checkpoint already exists (set `FORCE_RETRAIN = True` to retrain) |
| 4 | Plot training losses |
| 5 | Per-split (train/val/test) mean/worst relative-L2 table, per mode |
| 6 | Closest H¹-cosine pair (train/val vs test): `best_simil` leakage check + bubble & ε plots |
| 7 | Plot learned vs target bubbles on training samples (+ ε panels) |
| 8 | Apply to `-(εu')' = 1 + x`: Reference / Galerkin / Gal+bubble(exact) / Gal+bubble(KAN), rel-L2 & rel-H1 |

---

## Model architecture

```
Input: (ξ_s, ε-features)        ξ scaled to [-1,1]; ε-features pre-mapped to [-1,1]
    ↓
KANLayer(n_in → n_hidden)        learnable edge functions (32 hidden)
    ↓
KANLayer(n_hidden → 1)           learnable edge functions
    ↓
softplus(raw) + delta            positivity
    ↓
4·ξ·(1−ξ) · value / norm(0.5)    envelope + midpoint normalization
    ↓
Output: b(ξ), b(0)=b(1)=0, b(0.5)=1
```

Each KAN edge function is a B-spline network

```
φ(x) = w_b · SiLU(x) + w_s · Σ_i c_i · B_i^k(x)
```

with quadratic B-splines (k = 3 → reproduced degree k−1 = 2) on a grid of
`n_grid` intervals over [-1, 1]. Nodes sum their incoming edges with no
activations between layers. `KAN1D`'s grid domain defaults to [-1, 1]; RFB
bubbles always use the physical domain [0, 1].

## Profile-feature representations

The Darcy solution depends on the profile `ε(x)` only through the cumulative
resistivity `R(x) = I₀(x)/I₀(1)`, `I₀(x) = ∫₀ˣ dξ/ε(ξ)`. The pool stores,
per sample:

| kind | meaning |
|---|---|
| `gauss_ratio` | ε/ε̄ (Gauss-weighted mean) at fixed points — robust bulk, but thin resistive layers alias → tail failures |
| `resistivity_cdf` | `R(x)` — exact sufficient statistic, fixes extremes but hurts mid-distribution bulk |
| `scaled_combo` | log-ratios ⊕ CDF, both mapped to [−1,1]; best on every metric |
| `scaled_combo_v2` | same but log-ratios scaled by 4.0 instead of 3.0 — no clip saturation at contrast ~1000; use for high-contrast regimes |

The recommended choice is `scaled_combo_v2` with `eps_transform="none"`.
Per-model error correlation between the gauss-ratio view and the CDF view is
weak (~0.32): complementary failure modes are what make the combined input win
(blending two separately-trained models does NOT help).

## Split strategies

`build_split_dataset(..., strategy=...)`:

- `no_twin_shape` (default): val/test = most-atypical bubble shapes by H¹
  cosine; train = everything with similarity ≤ θ (default 0.99) to all
  val/test; twins dropped. OOD by construction.
- `random`: i.i.d. diagnostic baseline.
- `contrast_band`: since rescaling ε leaves the normalized solution unchanged,
  **contrast c = ε_max/ε_min is the difficulty axis**. Train/val/test are
  contiguous contrast intervals (70/15/15) of c ∈ [1, 1000] — a clean
  controlled OOD probe. (Twin audit is informational for this strategy.)

> Datasets produced under different split strategies or feature kinds are NOT
> interchangeable: the metric-dependent OOD selection changes which bubbles go
> to train vs test, so new splits must be regenerated.

## Known results (reference numbers)

- **No-twin split, 5pc (no_twin_shape)**: worst train–test bubble H¹ similarity
  ≈ 0.98 (leak-free), effective rank ≈ 1.1 (dominant diffusion-shape family).
- **Contrast-band data-scaling study** (scaled_combo_v2 + energy_weight=0.01,
  n_hidden=32, n_grid=12, 1400 epochs; test mean rel-L2, constant mode):

| n_train | test mean | median | p95 |
|---|---|---|---|
| 3.5k | 25.4% | 20.8% | 0.56 |
| 7k | 18.8% | 15.7% | 0.38 |
| 14k | 15.8% | 13.0% | 0.33 |

Monotone power-law-ish scaling with diminishing returns past ~7–10k (standard
iteration ~8k). Error-vs-contrast is flat across the extrapolation range —
graceful degradation holds at 6× more data.

- **Galerkin baseline** on the applied problem: ~15% rel-L2; the enriched
  solutions (exact or KAN bubbles) drop this toward the FD-resolution floor.

---

## Future work / generalization

The pipeline is specific to pure diffusion `-(εu')' = f` with 5-piece constant
profiles and `f ∈ span{1, x}`. Natural generalizations, currently out of scope:

- **Advection and reaction** terms (`-εu'' + βu' + σu = f`) — reintroduce
  local Péclet/reaction parameters alongside the profile features.
- More profile classes (smooth, sinusoidal, layered; polynomial ε).
- Higher-dimension bubbles / richer residual sources beyond `span{1, x}`.

---

## Source files

```
data_generation_darcy_variable.py  CLI: Darcy pool + splits + enrichment gate
tutorial_darcy_variable.py         Load data, train KAN, evaluate, apply, audit
check_fd_accuracy.py               (legacy constant-coefficient FD audit)
export_bubble_figures.py           Paper figure generation
tests/run_all.py                   One-command pytest suite

src/
├── darcy_variable.py         PiecewiseDiffusion, FD Darcy solver, profile features, pool gen
├── darcy_assembly.py         P1 assembly, static condensation, enrichment gate/audit
├── kan.py                    KAN1D edge function (B-spline + SiLU)
├── rfb_bubble.py             KANBubble1D, MultiKANBubble1D
├── dataset_generation.py     Sampling -> FD solves -> split, batch training, similarity analysis
├── training.py               train_multi_bubble_on_dataset (API)
├── rfb_training.py           Low-level training helpers
└── ...                       P1 FEM infrastructure (mesh, quadrature, pde, basis, errors)
```

Datasets live in `datasets/data_darcy_variable/`; checkpoints in `models/`.
Both directories are created on first save.
