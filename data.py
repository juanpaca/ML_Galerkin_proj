import numpy as np
from src.dataset_generation import generate_dataset, save_dataset, DatasetConfig

# eps in [1e-3, 1e-1], sigma in [1, 100], beta=1, h=1/16
# Pe = h/(2*eps) in [0.3125, 31.25]
# rho = sigma*eps/h^2 in [1*1e-3*256, 100*1e-1*256] = [0.256, 2560]
config = DatasetConfig(
    n_samples=5000,
    h=1/16,
    eps_range=(1e-3, 1e-1),
    beta_range=(1.0, 1.0),
    sigma_range=(1.0, 100.0),
    strategy="log_pe_rho",
    pe_range=(0.3125, 31.25),
    rho_range=(0.256, 2560.0),
    split_strategy="frame",
    frame_d_prime_fraction=0.90,
    frame_val_fraction=500 / 4050,
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

# Verify parameter ranges
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
