# ML_Galerkin_proj — AGENTS.md

## Project

PhD research: **ML-enhanced FE spaces for advection-diffusion-reaction PDEs**.
Learn KAN-parameterized Residual-Free Bubbles (b̂ = L⁻¹(1), b̃ = L⁻¹(ξ)),
statically condensed into P1 FEM (mesh‑independent, via Pe, ρ).

## How to run

```bash
source venv/bin/activate
venv/bin/python script.py
```

Key scripts:
- `data_generation.py` — generate leakage-free dataset (shape-based split, no-twin guarantee), replaces `data.py`
- `check_fd_accuracy.py` — validate FD bubble solver vs analytic solution (`src/rfb_analytic.py`), audit datasets
- `tutorial.py` — full guided walkthrough: leakage plots, bubble shapes, training, OOD test, assembly
- `test_all.py` — unit tests (includes analytic-FD cross-validation)
- `test_assembly_pipeline.py` — end‑to‑end assembly test (untrained KAN vs exact RFB)
- `convergence_study.py` — convergence study (Classical/Exact RFB, `--train-kan` for KAN)

Training is done via the API (`train_multi_bubble_on_dataset` in
`src/dataset_generation.py`) — see README section 3 and `tutorial.py` section 5.

## Architecture

```
src/kan.py              KAN1D: w_b·SiLU(x) + w_s·Σc_i·B_i^k(x), B-spline eval
src/rfb_bubble.py       KANBubble1D: envelope·softplus(raw)/norm, MultiKANBubble1D
src/rfb_local.py        reference FD solver, local_parameters(eps,beta,sigma,h) → (pe, rho)
src/rfb_analytic.py     exact constant-coefficient bubble b(pe,rho,mode,xi) — ground truth for FD validation
src/rfb_exact.py        ExactRFBubble1D — ground truth (FD-solved, normalized)
src/rfb_training.py     supervised: generate_rfb_training_data, train_bubble_model
src/rfb_assembly.py     statically condensed assembly: A_cond = A_LL − A_Lb·inv(A_bb)·A_bL
src/mesh.py, quad.py, pde.py, basis.py → P1 FEM infrastructure
src/errors.py           L2, H1, energy error computation
```

## Key details

- **KAN1D** grid domain defaults to `[-1, 1]`. RFB bubbles always use `[0, 1]`.
- **Static condensation**: bubble DOFs eliminated per-element via Schur complement.
- **Residual modes**: `constant` (r̂₀=1), `xi` (r̂₁=ξ).
- **Training**: value‑only MSE loss (gradient term `create_graph=True` diverges).
- **No‑twin split** (`data_generation.py`): test/val = farthest-from-typical bubble shapes, train = everything with similarity ≤ θ (=0.99) to all test/val bubbles; twins dropped. OOD by construction (not a cell/frame split).
- **FD accuracy**: n_fd_points=400 under-resolves the Pe>50 boundary layer (~2 pts inside; 1.5–2% bubble error at Pe=100). Dataset regenerated at **n_fd_points=3200** → <0.4% error everywhere (validated vs `src/rfb_analytic.py`). No oscillating bubbles (upwind scheme is monotone).

## Dataset

`datasets/rfb_5k_noleak_*` — 5000-sample pool (log-uniform), Pe∈[0.3,100], ρ∈[0.2,100], bubbles on 3200-pt FD grid. Split: 1298 train / 750 val / 500 test, leak-free (worst train–test bubble sim 0.98). Train Pe∈[2.5,16] (diffusion bubbles), test Pe∈[50,100] (boundary layer).

## Assembler conventions

- `assemble_rfb_condensed_system(pe, rho)` uses `local_parameters` internally.
- `RFBSolution1D(nodal_coeffs, bubble_coeffs, mesh, bubble_provider, pde)` evaluates enriched solution.
- `recover_bubble_coefficients(nodal_coeffs, mesh, local_data)` solves local systems element-by-element.

## Trained models

Models saved to `models/` by `train_xi.py` and `train_1k.py`. Directory created on first save.
