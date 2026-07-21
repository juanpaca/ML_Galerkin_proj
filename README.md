# ML_Galerkin_proj

PhD research: **ML-enhanced FE spaces for advection-diffusion-reaction PDEs**.
Learn KAN-parameterized Residual-Free Bubbles (b̂ = L⁻¹(1), b̃ = L⁻¹(ξ)),
statically condensed into P1 FEM (mesh‑independent, via Pe, ρ).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy
```

## Dataset generation

Generate both modes (constant and xi) in one call:

```python
from src.dataset_generation import generate_dataset, save_dataset, load_dataset

# Quick: override fields directly
dataset = generate_dataset(
    n_samples=5000,
    eps_range=(1e-6, 1.0),
    sigma_range=(0.0, 10.0),
    strategy="lhs",
    split_strategy="cell",
)

# Or use DatasetConfig
from src.dataset_generation import DatasetConfig
config = DatasetConfig(
    n_samples=5000,
    eps_range=(1e-6, 1.0),
    sigma_range=(0.0, 10.0),
    strategy="lhs",
    split_strategy="cell",
)
dataset = generate_dataset(config)

# Save for later
path = save_dataset(dataset, name="rfb_5k")

# Load back
dataset = load_dataset("rfb_5k")
```

The dataset contains:

| Key | Contents |
|-----|----------|
| `dataset["train"]["constant"]` | b̂ = L⁻¹(1) — pe, rho, b, db, xi |
| `dataset["train"]["xi"]` | b̃ = L⁻¹(ξ) — same keys |
| `dataset["val"]`, `dataset["test"]` | Same structure, held-out cells |
| `dataset["metadata"]` | Config, split indices, cell map |
| `dataset["scaler"]` | DataScaler if `standardize=True` |

## Training

### With the existing 1k dataset

```bash
source venv/bin/activate
python train_1k.py    # trains both modes, 400 epochs
python train_xi.py    # xi mode only, saves model
```

### Programmatic API

```python
from src.dataset_generation import train_multi_bubble_on_dataset
from src.rfb_bubble import MultiKANBubble1D
import torch

model = MultiKANBubble1D(n_bubbles=2, n_hidden=5).to("cuda")
histories = train_multi_bubble_on_dataset(
    model, dataset["train"],
    n_epochs=700, batch_size=256, lr=1e-3,
    device=torch.device("cuda"),
)
```

Or train a single bubble:

```python
from src.dataset_generation import train_bubble_on_dataset
from src.rfb_bubble import KANBubble1D

model = KANBubble1D(n_hidden=10).to("cuda")
losses = train_bubble_on_dataset(
    model, dataset["train"]["xi"],
    n_epochs=700, batch_size=256, lr=1e-3,
    device=torch.device("cuda"),
)
```

## Running tests

```bash
source venv/bin/activate
python test_all.py                # 98 unit tests
python test_assembly_pipeline.py  # end-to-end static condensation test
```

## Architecture

```
src/
├── kan.py                 KAN1D: w_b·SiLU(x) + w_s·Σc_i·B_i^k(x)
├── rfb_bubble.py          KANBubble1D, MultiKANBubble1D, KANLayer
├── rfb_local.py           Reference FD solver, local_parameters
├── rfb_exact.py           ExactRFBubble1D (ground truth)
├── rfb_training.py        Data generation helpers
├── rfb_assembly.py        Static condensation
├── dataset_generation.py  Full dataset pipeline + training
├── mesh.py, quad.py,
│   pde.py, basis.py       P1 FEM infrastructure
└── errors.py              L2, H1, energy errors
```

## Key details

- **Residual modes**: `constant` (r̂₀=1), `xi` (r̂₁=ξ)
- **KAN**: efficient-kan 3D tensor B-spline: `(batch, in, n_basis)` → `F.linear`
- **Input scaling**: Pe_s = log1p(Pe)/6, ρ_s = log1p(ρ)/6, ξ_s = 2ξ−1 → all in [-1,1]
- **Normalization**: envelope 4ξ(1-ξ) · softplus(raw) / norm(0.5)
- **Training**: value-only MSE (gradient term diverges with create_graph=True)
- **Cell‑based split**: entire (Pe decade, ρ range) cells held out — no leakage
