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
- `train_1k.py` — train KANBubble1D on 1k dataset (both modes)
- `train_xi.py` — train xi mode only, saves model
- `test_all.py` — 98 unit tests
- `test_assembly_pipeline.py` — end‑to‑end assembly test (untrained KAN vs exact RFB)

## Architecture

```
src/kan.py              KAN1D: w_b·SiLU(x) + w_s·Σc_i·B_i^k(x), B-spline eval
src/rfb_bubble.py       KANBubble1D: envelope·softplus(raw)/norm, MultiKANBubble1D
src/rfb_local.py        reference FD solver, local_parameters(eps,beta,sigma,h) → (pe, rho)
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
- **Cell‑based split**: Pe decades × ρ ranges, entire cells held out (no leakage).

## Dataset

`datasets/rfb_1k_*` — 1000 samples, Pe∈[0.3,3e4], ρ∈[0,3.6e4], 20/30 cells populated.

## Assembler conventions

- `assemble_rfb_condensed_system(pe, rho)` uses `local_parameters` internally.
- `RFBSolution1D(nodal_coeffs, bubble_coeffs, mesh, bubble_provider, pde)` evaluates enriched solution.
- `recover_bubble_coefficients(nodal_coeffs, mesh, local_data)` solves local systems element-by-element.

## Trained models

Models saved to `models/` by `train_xi.py` and `train_1k.py`. Directory created on first save.
