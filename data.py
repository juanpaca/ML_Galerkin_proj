import numpy as np
from src.dataset_generation import generate_dataset, save_dataset, DatasetConfig

# Frame split with log(Pe) x log(rho) uniform sampling.
# D' = centered 95% in log-space = train + val.
# T = D \ D' = frame = test set (corners with extreme Pe/rho the model never saw).
#
# d'=0.95 → frame ~10% of area → ~500 test samples.
# val_fraction=500/4500 ≈ 0.11 → ~500 val samples.
# train = remaining ~4000 samples.
config = DatasetConfig(
    n_samples=5000,
    h=1/16,
    eps_range=(1e-4, 1e-1),
    beta_range=(1.0, 1.0),
    sigma_range=(0.0, 10000.0),
    strategy="log_pe_rho",
    pe_range=(0.3, 312.0),
    rho_range=(0.01, 380.0),
    split_strategy="frame",
    frame_d_prime_fraction=0.90,
    frame_val_fraction=500 / 4050,  # ≈ 0.123 → 500 val out of 4050 inside D'
)
dataset = generate_dataset(config)
path = save_dataset(dataset, name="rfb_5k_frame")

meta = dataset["metadata"]
n = meta["n_total"]
tr = meta["n_train"]
va = meta["n_val"]
te = meta["n_test"]
fm = meta.get("frame_meta", {})
print(f"\n=== FRAME SPLIT: {n} total -> {tr}tr ({tr/n*100:.1f}%) / {va}va ({va/n*100:.1f}%) / {te}te ({te/n*100:.1f}%) ===")
print(f"    D' (train+val) inside: {fm.get('n_inside', '?')} samples")
print(f"    Frame T (test) outside: {fm.get('n_frame', '?')} samples")
print(f"    log10(Pe) D' bounds: [{fm.get('d_prime_pe_bounds', ['?','?'])[0]:.2f}, {fm.get('d_prime_pe_bounds', ['?','?'])[1]:.2f}]")
print(f"    log10(rho) D' bounds: [{fm.get('d_prime_rho_bounds', ['?','?'])[0]:.2f}, {fm.get('d_prime_rho_bounds', ['?','?'])[1]:.2f}]")

# Verify corner coverage
pe_te = dataset['test']['constant']['pe']
rho_te = dataset['test']['constant']['rho']
log_pe_te = np.log10(pe_te)
log_rho_te = np.log10(rho_te)
pe_lo, pe_hi = fm['d_prime_pe_bounds']
rho_lo, rho_hi = fm['d_prime_rho_bounds']
c1 = ((log_pe_te <= pe_lo) & (log_rho_te <= rho_lo)).sum()
c2 = ((log_pe_te >= pe_hi) & (log_rho_te <= rho_lo)).sum()
c3 = ((log_pe_te <= pe_lo) & (log_rho_te >= rho_hi)).sum()
c4 = ((log_pe_te >= pe_hi) & (log_rho_te >= rho_hi)).sum()
print(f"\n    Corner coverage in test set:")
print(f"      (low Pe, low rho):  {c1}")
print(f"      (high Pe, low rho): {c2}")
print(f"      (low Pe, high rho): {c3}")
print(f"      (high Pe, high rho):{c4}")
