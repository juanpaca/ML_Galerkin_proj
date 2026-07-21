import numpy as np
from src.dataset_generation import generate_dataset, save_dataset, load_dataset
from src.dataset_generation import DatasetConfig

# Stratified split: preserves all Pe regimes proportionally in train/val/test.
# Use this for daily training — val/test metrics are reliable across all regimes.
config = DatasetConfig(
    n_samples=5000,
    eps_range=(1e-6, 1.0),
    sigma_range=(0.0, 10.0),
    strategy="lhs",
    split_strategy="stratified",  # ← all regimes in every split
    val_split=0.10,
    test_split=0.10,
)
dataset = generate_dataset(config)
path = save_dataset(dataset, name="rfb_5k")

n = dataset["metadata"]["n_total"]
tr = dataset["metadata"]["n_train"]
va = dataset["metadata"]["n_val"]
te = dataset["metadata"]["n_test"]
print(f"\n=== RESULT: {n} total → {tr}tr ({tr/n*100:.1f}%) / {va}va ({va/n*100:.1f}%) / {te}te ({te/n*100:.1f}%) ===")

# To switch to cell-based split (for final paper evaluation):
# - Change split_strategy to "cell"
# - Search seeds until held-out cells cover all regimes
# - Example: uncomment below and try seed=123, 456, ...
#
# config = DatasetConfig(
#     n_samples=5000,
#     eps_range=(1e-6, 1.0),
#     sigma_range=(0.0, 10.0),
#     strategy="lhs",
#     split_strategy="cell",
#     n_val_cells=3,
#     n_test_cells=3,
#     pe_bins=(0, 0.1, 1, 10, 100, 1000, 1e4, np.inf),
#     rho_bins=(-np.inf, 0, 0.1, 1, 10, 100, 1000, np.inf),
#     seed=123,
# )