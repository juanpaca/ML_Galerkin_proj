# PLAN_CVAL.md — Contrast-band CV + Hyperparameter Optimization

## 1. Objective

Evaluate KAN generalization to **unseen contrast levels** c = ε_max / ε_min ∈ [1, 100]
on 5-piece constant piecewise-ε Darcy profiles. The train/val portion (85%) covers
low-to-moderate contrast; the held-out test (15%) covers the highest-contrast profiles.
Cross-validation on the 85% selects hyperparameters; the final model is retrained on
the full 85% and evaluated on the test band.

Unlike the no-twin shape split, the data here is **i.i.d.** — generated randomly and
partitioned purely by contrast, with no cosine-similarity filtering.

---

## 2. Data Generation

### Pool specification (5-piece, as previously agreed)

| Parameter | Value |
|---|---|
| Number of pieces | exactly 5 |
| ε range | [0.1, 10] |
| Contrast range c = ε_max/ε_min | [1, 100] |
| Piece widths | random, sum = 1, minimum width = 0.1 |
| Breakpoints | uniform random inside [0, 1], filtered for min width |
| Source modes | f ∈ span{1, x} → constant and xi residual modes |
| FD grid | n_fd = 3201 uniform points in [0, 1] |
| Bubble normalization | b(0.5) = 1 |
| Enrichment gate | OFF (i.i.d. data, no quality filter) |

### Pool size

Target N = 5000 samples before splitting. The enrichment gate is **not** applied
(this is i.i.d. data, not a no-twin pool). Profiles that produce poorly-resolved
FD bubbles are kept — the KAN must learn them as-is.

### Split: contrast bands (contiguous, 70 / 15 / 15)

1. Compute realized contrast per sample: c_i = max(eps_profile_i) / min(eps_profile_i).
2. Sort all N samples by c (ascending).
3. Assign contiguous bands:
   - **Train**: first 70% → c ∈ [1, c_70]
   - **Val**: next 15% → c ∈ [c_70, c_85]
   - **Test**: last 15% → c ∈ [c_85, 100]

These three bands form the **outer split**. The test band is held out for final
evaluation only. All CV happens within the train+val (85%) pool.

### Feature representation

Use `scaled_combo_v2` (log-ratios ⊕ CDF, both mapped to [-1, 1], 16 `eps_ratios`
features per profile). Consume with `eps_transform="none"`.

---

## 3. Cross-Validation Procedure

### Outer split (already done by contrast bands)

```
[c = 1  ···············  c_70  ···  c_85  ···········  c = 100]
|◄──── Train (70%) ────►|◄─ Val (15%) ─►|◄── Test (15%) ──►|
|◄──────────── 85% pool (CV here) ─────────────►|           |
```

### Inner split: 5-fold CV on the 85% pool

1. Combine train + val indices → 85% pool (size ≈ 0.85 × N).
2. Shuffle the pool with a fixed seed (`CV_SEED = 0`).
3. Split into 5 equal folds (floor/ceiling for integer sizes).
4. For each of the 5 folds:
   - **Train fold**: 4/5 of the 85% pool (~68% of total)
   - **Val fold**: 1/5 of the 85% pool (~17% of total)

Standard k-fold: each sample appears in exactly one val fold across the 5 runs.

---

## 4. Hyperparameter Grid

| Parameter | Values | Notes |
|---|---|---|
| `n_hidden` | 16, 32 | KAN hidden layer width |
| `n_grid` | 8, 12 | B-spline grid resolution |
| `lr` | 5e-4, 1e-3 | Adam learning rate |
| `energy_weight` | 0, 0.01 | H1-derivative regularization weight |

**Total combinations**: 2 × 2 × 2 × 2 = **16**
**Total training runs**: 16 combos × 5 folds = **80**

### Fixed training parameters

| Parameter | Value |
|---|---|
| `n_epochs` | 1400 |
| `batch_size` | 256 |
| `lr_scheduler` | "cosine" |
| `grad_weight` | 0.0 |
| `n_quad` | 160 |
| `mode_names` | ("constant", "xi") |

### Selection criterion

Pick the HP combination with the **lowest mean relative H1** across the 5 folds,
averaged over both modes (constant and xi):

```
score(hp) = mean over folds of [ mean over modes of rel_H1_val ]
best_hp = argmin score(hp)
```

---

## 5. Relative H1 Metric on Bubble Arrays

The rel-H1 is computed directly on the discrete bubble arrays (KAN prediction vs
FD target), using the shared FD grid:

```
rel_H1_i = sqrt( ∫(b̂_i - b_i)² dξ + λ² ∫(b̂'_i - b'_i)² dξ )
            ─────────────────────────────────────────────────────
            sqrt( ∫b_i² dξ + λ² ∫b'_i² dξ )
```

where:
- `b̂_i` = KAN-predicted bubble for sample i (shape: [n_fd])
- `b_i` = FD-solved target bubble (shape: [n_fd])
- `∫` ≈ trapezoid quadrature on the uniform FD grid
- `d/dξ` ≈ centered finite differences (np.gradient)
- `λ = 0.2` (derivative weight, matching the H¹ inner product convention)

**Per-sample** rel-H1 is computed; the fold score is the **mean** over samples in the
val fold. Report per-mode (constant, xi) and their average.

---

## 6. Script: `scripts/contrast_cv_hpo.py`

### Input / output

**Input**: generates a fresh pool internally (or loads an existing contrast-band
dataset if available). Uses the existing infrastructure:
- `data_generation_darcy_variable.generate_and_save_dataset` (pool + split)
- `src.dataset_generation.train_multi_bubble_on_dataset` (training)
- `src.rfb_bubble.MultiKANBubble1D` (model)

**Output** (saved to `models/` and `results/`):
- `models/contrast_5pc_cv_hpo_final_kan.pt` — final model state dict
- `results/contrast_5pc_cv_hpo.json` — full CV results:
  ```json
  {
    "dataset": "contrast_5pc_pool_5k",
    "pool_size": 5000,
    "split": {"train": 3500, "val": 750, "test": 750},
    "contrast_bands": {"train": [1.0, c_70], "val": [c_70, c_85], "test": [c_85, 100]},
    "cv_seed": 0,
    "n_folds": 5,
    "param_grid": [...],
    "cv_results": [
      {"hp": {"n_hidden": 16, "n_grid": 8, "lr": 5e-4, "energy_weight": 0},
       "fold_scores": [{"constant": ..., "xi": ..., "avg": ...}, ...],
       "mean_score": 0.xxx,
       "std_score": 0.xxx},
      ...
    ],
    "best_hp": {"n_hidden": 32, "n_grid": 12, "lr": 5e-4, "energy_weight": 0.01},
    "best_cv_mean": 0.xxx,
    "test_scores": {"constant": ..., "xi": ..., "avg": ...}
  }
  ```

### Pseudocode

```python
# === Config ===
N_SAMPLES = 5000
N_FD_POINTS = 3201
EPS_RANGE = (0.1, 10.0)
N_PIECES = 5
FEATURE_KIND = "scaled_combo_v2"
EPS_TRANSFORM = "none"
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CV_SEED = 0
N_FOLDS = 5

PARAM_GRID = [
    {"n_hidden": 16, "n_grid":  8, "lr": 5e-4, "energy_weight": 0.0},
    {"n_hidden": 16, "n_grid":  8, "lr": 5e-4, "energy_weight": 0.01},
    {"n_hidden": 16, "n_grid":  8, "lr": 1e-3, "energy_weight": 0.0},
    {"n_hidden": 16, "n_grid":  8, "lr": 1e-3, "energy_weight": 0.01},
    {"n_hidden": 16, "n_grid": 12, "lr": 5e-4, "energy_weight": 0.0},
    {"n_hidden": 16, "n_grid": 12, "lr": 5e-4, "energy_weight": 0.01},
    {"n_hidden": 16, "n_grid": 12, "lr": 1e-3, "energy_weight": 0.0},
    {"n_hidden": 16, "n_grid": 12, "lr": 1e-3, "energy_weight": 0.01},
    {"n_hidden": 32, "n_grid":  8, "lr": 5e-4, "energy_weight": 0.0},
    {"n_hidden": 32, "n_grid":  8, "lr": 5e-4, "energy_weight": 0.01},
    {"n_hidden": 32, "n_grid":  8, "lr": 1e-3, "energy_weight": 0.0},
    {"n_hidden": 32, "n_grid":  8, "lr": 1e-3, "energy_weight": 0.01},
    {"n_hidden": 32, "n_grid": 12, "lr": 5e-4, "energy_weight": 0.0},
    {"n_hidden": 32, "n_grid": 12, "lr": 5e-4, "energy_weight": 0.01},
    {"n_hidden": 32, "n_grid": 12, "lr": 1e-3, "energy_weight": 0.0},
    {"n_hidden": 32, "n_grid": 12, "lr": 1e-3, "energy_weight": 0.01},
]

# === Step 1: Generate pool + contrast-band split ===
ds = generate_and_save_dataset(
    n_samples=N_SAMPLES, n_fd_points=N_FD_POINTS,
    eps_range=EPS_RANGE, n_pieces=N_PIECES,
    feature_kind=FEATURE_KIND, eps_transform=EPS_TRANSFORM,
    val_frac=VAL_FRAC, test_frac=TEST_FRAC,
    split_strategy="contrast_band", seed=42,
    name="contrast_5pc_pool_5k",
)

# === Step 2: Combine train+val → 85% pool ===
pool_85 = concat_mode_arrays(ds["train"], ds["val"], mode_names)
test_data = ds["test"]
n_85 = pool_85["constant"]["b"].shape[0]

# === Step 3: 5-fold CV splits ===
rng = np.random.default_rng(CV_SEED)
fold_idx = np.array_split(rng.permutation(n_85), N_FOLDS)

# === Step 4: CV loop ===
cv_results = []
for hp in PARAM_GRID:
    fold_scores = []
    for fold in range(N_FOLDS):
        val_mask = fold_idx[fold]
        train_mask = np.concatenate([fold_idx[i] for i in range(N_FOLDS) if i != fold])

        model = MultiKANBubble1D(
            n_bubbles=2, n_hidden=hp["n_hidden"], n_grid=hp["n_grid"],
            n_eps=n_eps, eps_transform=EPS_TRANSFORM,
        ).to(DEVICE)

        train_split = index_split(pool_85, train_mask)
        train_multi_bubble_on_dataset(
            model, train_split, mode_names=("constant", "xi"),
            n_epochs=1400, batch_size=256, lr=hp["lr"],
            grad_weight=0.0, n_quad=160, verbose=False,
            device=DEVICE, lr_scheduler="cosine",
            energy_weight=hp["energy_weight"],
        )

        val_split = index_split(pool_85, val_mask)
        scores = evaluate_rel_h1(model, val_split, xi, DEVICE)
        fold_scores.append(scores)

    mean_score = np.mean([f["avg"] for f in fold_scores])
    cv_results.append({"hp": hp, "fold_scores": fold_scores, "mean_score": mean_score})

# === Step 5: Pick best HP, retrain on full85% ===
best = min(cv_results, key=lambda r: r["mean_score"])
best_hp = best["hp"]

final_model = MultiKANBubble1D(n_bubbles=2, **best_hp, n_eps=n_eps, eps_transform=EPS_TRANSFORM).to(DEVICE)
train_multi_bubble_on_dataset(final_model, pool_85, ...)

# === Step 6: Evaluate on test ===
test_scores = evaluate_rel_h1(final_model, test_data, xi, DEVICE)

# === Step 7: Save ===
torch.save(final_model.state_dict(), "models/contrast_5pc_cv_hpo_final_kan.pt")
save_json({"cv_results": cv_results, "best_hp": best_hp, "test_scores": test_scores, ...},
          "results/contrast_5pc_cv_hpo.json")
```

### Helper functions to implement

1. **`concat_mode_arrays(split_a, split_b, mode_names)`** — concatenate two split dicts
   per mode, stacking all array fields (pe, rho, b, db, xi, eps_ratios, eps_profile).

2. **`index_split(split_dict, indices)`** — return a new split dict with all per-mode
   arrays indexed by `indices`.

3. **`evaluate_rel_h1(model, split, xi, device)`** — for each mode, predict bubbles
   for all samples in the split (batched), compute per-sample rel-H1 vs targets,
   return `{"constant": mean_rel_h1, "xi": mean_rel_h1, "avg": mean}`.

4. **`rel_h1_per_sample(b_pred, b_true, xi, lam=0.2)`** — per-sample relative H1
   on bubble arrays using trapezoid quadrature + centered differences.

---

## 7. Runtime Estimate

| Component | Estimate |
|---|---|
| Pool generation + split | ~5 min |
| Per CV run (3400 train, 1400 epochs) | ~10–15 min on GPU |
| Total CV (80 runs) | ~13–20 hours on 1 GPU |
| Final retrain (4250 samples) | ~15 min |
| Final evaluation | ~1 min |
| **Total** | **~14–21 hours** |

Recommended: run overnight. If multiple GPUs available, parallelize the outer HP
loop (each combo is independent across folds).

---

## 8. File Structure After Execution

```
ML_Project/
├── datasets/data_darcy_variable/
│   └── contrast_5pc_pool_5k_{train,val,test}_{constant,xi}.npz
│   └── contrast_5pc_pool_5k_metadata.json
├── models/
│   └── contrast_5pc_cv_hpo_final_kan.pt
├── results/
│   └── contrast_5pc_cv_hpo.json
├── scripts/
│   └── contrast_cv_hpo.py          ← new
└── PLAN_CVAL.md                    ← this file
```

---

## 9. Key Differences from Previous Experiments

| Aspect | No-twin shape split | This experiment |
|---|---|---|
| Split criterion | Cosine similarity (H¹ or L²) | Contrast c = ε_max/ε_min |
| Data selection | OOD by shape, twins dropped | i.i.d., no filtering |
| Train contrast range | Mixed (central shape bulk) | Low c ∈ [1, c_70] |
| Test contrast range | Extreme shapes | High c ∈ [c_85, 100] |
| Generalization test | Shape novelty | Extrapolation to harder contrast |
| CV | None | 5-fold on 85% |
| Metric | rel-L2 (value only) | rel-H1 (value + derivative) |

---

## 10. Notes

- The `generate_and_save_dataset` function with `--split-strategy contrast_band`
  already computes contrasts, sorts, and creates the contiguous bands. The script
  reuses this directly.

- The `--eps-min 0.1 --eps-max 10` flags set the ε range (contrast ∈ [1, 100]).
  Previous cband datasets used `--eps-min 0.01` (contrast up to 1000); this experiment
  uses the tighter5-piece spec.

- The enrichment gate is OFF (`--no-verify-enrichment` or equivalent) since we want
  i.i.d. data without quality filtering.

- No cosine-similarity audit is needed for this experiment (the split is by contrast,
  not by shape).

- The `scaled_combo_v2` features are pre-computed during pool generation and stored
  in the npz files as `eps_ratios` (shape: [N, 16]).

- PyTorch is the only ML framework used. No sklearn, optuna, or other HPO libraries
  are needed — the grid is small enough for manual enumeration.
