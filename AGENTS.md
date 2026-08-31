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
- `data_generation_darcy_variable.py` — generate piecewise-diffusion Darcy pool, contrast-band splits, enrichment gate
- `check_fd_accuracy.py` — validate FD bubble solver vs analytic solution (`src/rfb_analytic.py`), audit datasets
- `tutorial_darcy_variable.py` — load data, train KAN, evaluate, save model (120 lines)
- `convergence_study.py` — convergence study (Classical/Exact RFB, `--train-kan` for KAN)
- `export_bubble_figures.py` — paper figure generation (bubbles + enriched solutions)
- `tests/run_all.py` — one-command full test suite (runs the whole `tests/` dir): `venv/bin/python tests/run_all.py`

Training is done via the API (`train_multi_bubble_on_dataset` in
`src/dataset_generation.py`) — see README section 3 and `tutorial.py` section 5.
Optional args: `lr_scheduler="cosine"` (per-epoch cosine annealing),
`grad_weight` stays 0 for stability, and `energy_weight` adds
0.01·∫ε|d(b̂−b_target)/dx|²dx via input-space central differences of values
(same discrete operator on both sides; no second-order autograd; requires
`eps_profile` in mode_data; default 0 = legacy behavior identical).

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
src/darcy_variable.py   piecewise epsilon profiles, conservative Darcy solver, dataset pool
```

## Key details

- **KAN1D** grid domain defaults to `[-1, 1]`. RFB bubbles always use `[0, 1]`.
- **Static condensation**: bubble DOFs eliminated per-element via Schur complement.
- **Residual modes**: `constant` (r̂₀=1), `xi` (r̂₁=ξ).
- **Training**: value‑only MSE loss (gradient term `create_graph=True` diverges); optional `energy_weight` term uses FD derivatives of values only — stable.
- **No‑twin split** (`data_generation.py`): test/val = farthest-from-typical bubble shapes, train = everything with similarity ≤ θ (=0.99) to all test/val bubbles; twins dropped. OOD by construction (not a cell/frame split).
- **Similarity metric = derivative-aware H¹ cosine** (`bubble_gram_matrix`/`bubble_cosine_similarity`/`max_cross_similarity`/`_bubble_h1_features` in `src/dataset_generation.py`, `shape_no_leak_split_from_pool` in `data_generation.py`): ⟨b,b'⟩_H1 = ∫bb' + λ²∫b'b'' on the trapezoid quadrature (centered-difference derivative), default λ=0.2 — everywhere (splitter, auditor, tests). Plain L² cosine calls up to ~10% of all pairs on `darcy_piecewise_5pc` "twins" (>0.99) that are genuinely distinct in derivative space; H¹ fixes that (77k L²-twin pairs → 670). `lambda_deriv=0` reverts to L²; CLI flag `--lambda-deriv` on `data_generation.py`. Empirically H¹ ≤ L² pairwise on the real bubbles (0 violations in 2.6M pairs), so the existing L²-split datasets still satisfy the H¹ no-twin guarantee — but the choice still changes *which* bubbles are selected as OOD and how many train candidates survive (metric-dependent centrality), so it is not just "fewer twins" in one direction. New splits must regenerate.
- **FD accuracy**: n_fd_points=400 under-resolves the Pe>50 boundary layer (~2 pts inside; 1.5–2% bubble error at Pe=100). Dataset regenerated at **n_fd_points=3200** → <0.4% error everywhere (validated vs `src/rfb_analytic.py`). No oscillating bubbles (upwind scheme is monotone).
- **KAN spline degree convention** (`src/kan.py`): `k` = number of active bases ⇒ reproduced degree is k−1 (k=3 → quadratic). pykan/Blealtan use spline_order = degree, so their cubic is k=4 here. Endpoint folds in `KAN1D._eval_bspline_basis` and `KANLayer.b_splines` (src/rfb_bubble.py) make basis sums exact at x=x_min/x_max (one-hot last basis at the right edge); validated vs `scipy.interpolate.BSpline.design_matrix`.

## Dataset

`datasets/rfb_5k_noleak_*` — 5000-sample pool (log-uniform), Pe∈[0.3,100], ρ∈[0.2,100], bubbles on 3200-pt FD grid. Split: 1298 train / 750 val / 500 test, leak-free (worst train–test bubble sim 0.98). Train Pe∈[2.5,16] (diffusion bubbles), test Pe∈[50,100] (boundary layer).

## Darcy variable-coefficient experiments

Piecewise-constant ε(x) profiles (2–16 pieces, contrast ≤1000×), conservative FD solver,
bubbles on an 801-pt grid; generated by `data_generation_darcy_variable.py`
(`--feature-kind`), trained/audited by `tutorial_darcy_variable.py`.

### Pre-training enrichment gate (data-quality bar)

Before training, every generated bubble must demonstrably *enrich* the P1 Galerkin
solution. `src/darcy_assembly.py` provides the machinery: full-domain residual bubbles b̂=L⁻¹(1), b̃=L⁻¹(ξ) are
restricted per element onto a coarse P1 mesh and statically condensed
(`assemble_enriched`/`eval_enriched`); `enrichment_h1_error` (shared-grid audit) and
`enrichment_l2_gate` (independent fine reference, per-source-mode max rel L2) quantify
‖u−u_hb‖. Wiring: `data_generation_darcy_variable.py --verify-enrichment` (default on)
runs the gate over the whole pool against an independent n_gate_ref reference and
**fails with the worst offender unless `--gate-drop` is given**, which filters all
samples above `--gate-threshold`.

Key facts:

- **Gate semantics = the deployed assembly**: uniform `n_el=8` P1 mesh, ONE *global*
  coefficient pair per mode (not per-element local bubbles). This matches the demo/
  tutorial deployment exactly, so the gate measures bubble data quality in the real
  consumption path. (Experimental alternatives measured and rejected: per-element
  restriction of the global bubbles + local condensation blows up numerically
  (rel-L2 ~1e1–1e2, thin-element near-dependence); a mesh aligned to the ε-pieces
  with global coefficients underperforms uniformly because it has fewer P1 DOFs.)
- Empirical distribution (5-piece spec, n_fd=3201, n_ref≥2e4, seed 42): enriched
  rel-L2 mean ≈ 1.5e-3, p95 ≈ 5e-3, worst ≈ 2.2e-2 — the worst ~0.5–1% of samples
  are FD-under-resolved boundary layers and are dropped by `--gate-drop` at the
  default `--gate-threshold 1e-2`. The remaining pool passes the no-twin audit.
- For pure diffusion with a mesh aligned to the ε-pieces, A_Lb ≡ 0 for element-local
  bubbles (∫ε φ′b′ = εφ′(b(1)−b(0)) = 0), so local-bubble condensation recovery is
  span-only c_e = A_bb⁻¹F_b; a generic (Pe,ρ)-style recovery `A_bb⁻¹(F_b − A_LbᵀU)`
  amplifies FD-gradient noise at coarse grids. Bubbles *are* global (full
  discontinuous coefficient); the 2 modes {1, ξ} enrich to "almost perfect" only
  when the element residual is affine (mesh ⊇ piece boundaries or residues ∈ span{1,ξ}).
- Enrichment of the P1 solution is **mesh-independent**: at fixed n_fd the enriched
  rel L2 is flat across n_el = 4/8/16 (6.1e-3 / 6.5e-3 / 6.4e-3 at n_fd=801) — the
  bubbles carry all resolution, P1 is just the condenser.
- Grid resolution (5-piece, ε∈[0.1,10], min width 0.1, independent 32k reference):
  enriched rel L2 ~ 2.3e-2 (n_fd=201) → 1.2e-2 (401) → 6.5e-3 (801) → 3.4e-3 (1601)
  → 1.7e-3 (3201); rel H1 ~ 1.9e-2 at 3201 (localized kink/interface features, fine
  for value-MSE training). **Dataset convention: n_fd_points=3201, 5 pieces,
  min_width=0.1, ε∈[0.1,10]** (`datasets/data_darcy_variable/darcy_piecewise_5pc`).
- Where the sweep/gate numbers live: gate source-specs are f∈{1, ξ} per residual mode.

Profile features fed to the KAN (`make_profile_features` in `src/darcy_variable.py`):

- `gauss_ratio` ε/ε̄: robust bulk behavior, but thin resistive layers alias → catastrophic tail failures (rel. L2 up to ~10×).
- `resistivity_cdf` R(ξ)=I₀(ξ)/I₀(1): exact sufficient statistic (∝∫dξ/ε) — fixes extreme cases (train err 6.5%→2.1%) but *hurts* the bulk (monotone, highly-correlated inputs generalize worse mid-distribution).
- `scaled_combo` (recommended): log-ratios ⊕ CDF, both pre-mapped to [−1,1]; consume with `eps_transform="none"`. Best on every metric: test mean rel L2 (constant/xi) 43%/34% vs ~50% single-view; worst case bounded at 2.35 vs 9.8+.
- `scaled_combo_v2`: same but log-ratios divided by 4.0 instead of 3.0 — no clip saturation at contrast ~1000 (log10(827)/3 ≈ 0.97 was the failure mode); use for high-contrast regimes.

Key facts:

- Per-model error correlation across the two views is weak (~0.32): complementary failure modes are what make the combo input win. Blending two trained models does NOT help (cres is uniformly worse mid-bulk).
- `KANBubble1D(n_eps, eps_transform="log"|"linear"|"none")` scales eps features; default `"log"` keeps legacy behavior. Models must be rebuilt with the same transform used at training.
- 20k-pool runs use n_hidden=32, n_grid=12, 1400 epochs, cosine scheduler, value-only loss (~45 min/mode on GPU at 14k train samples). Datasets: `datasets/data_darcy_variable/darcy_piecewise_{rich20k,cres20k,combo20k}` share identical pools/splits/targets (features recomputed only); checkpoints in `models/*_kan.pt`.
- Remaining headroom: moderate-contrast profiles where plain ratios beat CDF inputs; learned per-sample gating or extra capacity is the natural next step.

### Contrast-band OOD split & data-scaling study

`--split-strategy contrast_band` (seeded, quantile cut): i.i.d. pool with ε∈[0.01,10]
(contrast c=εmax/εmin∈[1,1000]), train/val/test bands are contiguous contrast intervals
(e.g. [1,430]/[430,598]/[598,998] for 70/15/15). Since rescaling ε leaves the normalized
solution unchanged, contrast is *the* difficulty axis — a clean controlled OOD probe.
Twin audit is skipped for non-`no_twin_shape` strategies (informational print only).

Scaling curve (same recipe: scaled_combo_v2 + energy_weight=0.01, n_hidden=32,
n_grid=12, 1400 epochs; identical val/test; test mean rel L2, constant mode):

| n_train | test mean | median | p95 | time |
|---|---|---|---|---|
| 3.5k | 25.4% | 20.8% | 0.56 | 23 min |
| 7k | 18.8% | 15.7% | 0.38 | 45 min |
| 14k | 15.8% | 13.0% | 0.33 | 91 min |

Takeaways: monotone power-law-ish scaling with diminishing returns past ~7–10k
(standardize iteration on ~8k pools); train error also drops with n → genuine sample
efficiency, not under-training. Error-vs-contrast is flat across the extrapolation range
(median 0.115→0.138, Spearman 0.11) — graceful degradation held at 6× more data.
Recipe tweaks (v2 features + energy loss) contributed little vs data quantity at equal n
(3.5k arm ≈ old 5k baseline). Heavy tail worsened slightly with size (worst 1.4→3.9;
p95 improved) — single-seed caveat, ±1–2%. Datasets:
`darcy_piecewise_combo_cband{5k,20k_v2}`; checkpoints `models/darcy_piecewise_combo_cband*_kan.pt`.

## Assembler conventions

- `assemble_rfb_condensed_system(pe, rho)` uses `local_parameters` internally.
- `RFBSolution1D(nodal_coeffs, bubble_coeffs, mesh, bubble_provider, pde)` evaluates enriched solution.
- `recover_bubble_coefficients(nodal_coeffs, mesh, local_data)` solves local systems element-by-element.

## Trained models

Models saved to `models/` by `train_xi.py` and `train_1k.py`. Directory created on first save.
