#!/usr/bin/env python3
"""Generate a leakage-free RFB dataset using cosine-similarity shape analysis.

Improvement over the old ``data.py``. A purely geometric (Pe, rho) frame
split leaks in *function space*: bubble shapes evolve continuously with
(Pe, rho), so every test bubble sits next to a nearly identical train bubble
(similarity > 0.99) and the test error can be "memorized". This script

  1. generates a large pool of bubbles (log-uniform in Pe x rho),
  2. computes the L2 cosine-similarity matrix of the bubbles,
  3. splits the pool by *shape* instead of by (Pe, rho):
       train = the dense central bulk of shape space,
       val/test = the bubbles farthest from that bulk,
     and drops any val/test bubble that still has a near-duplicate
     (similarity > theta) in train — a dataset with NO function-space leakage,
  4. re-runs the similarity analysis to confirm the dataset is leak-free,
  5. saves it (default name ``rfb_5k_noleak``).

Usage:
    python data_generation.py                              # 5000 samples, theta=0.99
    python data_generation.py --n-samples 8000 --theta 0.98
    python data_generation.py --name rfb_8k_noleak --val-frac 0.1 --test-frac 0.3
"""

import argparse
import time

import numpy as np

from src.dataset_generation import (
    DatasetConfig,
    DataScaler,
    bubble_cosine_similarity,
    bubble_similarity_analysis,
    _effective_rank,
    _off_diagonal_stats,
    generate_dataset,
    load_dataset,
    save_dataset,
)

DTYPE = np.float32


# ---------------------------------------------------------------------------
# Shape-based split with a no-twin guarantee
# ---------------------------------------------------------------------------

def shape_no_leak_split(C, n_train, n_val, n_test, theta=0.99):
    """Split pool indices by bubble shape with a no-twin guarantee.

    A val/test bubble is a *twin* of the train set when its cosine
    similarity to some train bubble exceeds ``theta``. The returned split
    satisfies: every kept val/test bubble has similarity <= theta to *all*
    train bubbles (verified in both modes when ``C`` is their elementwise
    maximum), and vice versa. Samples that would leak are dropped.

    Strategy (reverse of the naive "bulk = train" idea, which always leaks):
      1. test = the n_test bubbles FARTHEST from the typical bubble
         (extreme / OOD shapes),
      2. val  = the next n_val bubbles farthest from the typical bubble
         (a second OOD slice, adjacent to test — a good proxy for test
         performance during model selection),
      3. train = every pool sample that is NOT a near-duplicate of any
         test or val bubble (this excludes the near-OOD band automatically,
         and makes the no-twin guarantee hold symmetrically).

    Parameters
    ----------
    C : ndarray, shape (N, N)
        Cosine-similarity matrix (elementwise max over the analyzed modes).
    n_train, n_val, n_test : int
        Target split sizes (the leak filter may shrink train).
    theta : float
        Twin-similarity threshold. Similarity > theta means "near-duplicate".

    Returns
    -------
    dict with keys ``train``, ``val``, ``test`` (kept indices),
    ``dropped`` (dropped indices) and ``stats``.
    """
    C = np.asarray(C, dtype=float)
    N = C.shape[0]

    # Distance from the "typical" bubble = highest-centrality sample.
    centrality = C.mean(axis=1)
    centroid = int(np.argmax(centrality))
    d = 1.0 - C[:, centroid]

    # 1-2. OOD slices: test and val are the most extreme bubble shapes.
    order = np.argsort(-d)                      # farthest first
    test = order[:n_test].tolist()
    val = order[n_test:n_test + n_val].tolist()
    ood = test + val

    # 3. Train = everything with no twin in test or val.
    max_sim_to_ood = C[:, ood].max(axis=1)
    train = (np.arange(N)[max_sim_to_ood <= theta]).tolist()
    train = [i for i in train if i not in set(ood)]
    if not train:
        raise RuntimeError("No training sample survives the no-twin filter; "
                           "lower theta, shrink val/test, or enlarge the pool.")

    dropped = [int(i) for i in range(N)
               if i not in set(train) and i not in set(ood)]

    stats = {
        "n_pool": N,
        "theta": theta,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "dropped": {"train": len(dropped), "val": 0, "test": 0},
    }
    return {"train": train, "val": val, "test": test,
            "dropped": {"train": dropped, "val": [], "test": []},
            "stats": stats}


# ---------------------------------------------------------------------------
# Pool generation and helpers
# ---------------------------------------------------------------------------

def generate_pool(n_samples, pe_range, rho_range, seed, n_fd_points=400):
    """Generate a pool with a throwaway random split, then merge it back."""
    config = DatasetConfig(
        n_samples=n_samples,
        h=1 / 16,
        strategy="log_pe_rho",
        pe_range=pe_range,
        rho_range=rho_range,
        split_strategy="random",
        standardize=False,
        seed=seed,
        n_fd_points=n_fd_points,
    )
    ds = generate_dataset(config)
    return ds, config


def merge_pool(ds):
    """Concatenate the random-split arrays into a single unlabeled pool."""
    modes = ds["mode_names"]
    pool = {}
    for m in modes:
        arrays = {}
        for k in ds["train"][m].keys():
            if k == "xi":
                arrays[k] = ds["train"][m][k]
            else:
                arrays[k] = np.concatenate(
                    [ds[s][m][k] for s in ("train", "val", "test") if s in ds]
                )
        pool[m] = arrays
    n = len(pool[modes[0]]["pe"])
    for m in modes:
        pool[m]["idx"] = np.arange(n)
    return pool, modes, n


def pool_similarity_matrix(pool, modes):
    """Elementwise-max cosine similarity across modes (binding constraint)."""
    C = None
    for m in modes:
        Cm = bubble_cosine_similarity(pool[m]["b"], pool[m]["xi"])
        C = Cm if C is None else np.maximum(C, Cm)
    return C


def fit_pool_scaler(pool, modes):
    pe = pool[modes[0]]["pe"]
    rho = pool[modes[0]]["rho"]
    X = np.column_stack([
        np.log(np.maximum(pe, 1e-15)),
        np.log(np.maximum(np.abs(rho), 1e-15)),
    ])
    return DataScaler().fit(X, feature_names=["pe_log", "rho_log"])


def build_split_dicts(pool, modes, split, scaler):
    """Assemble the new train/val/test dataset dicts from pool arrays."""
    dataset = {}
    for sname in ("train", "val", "test"):
        idx = np.asarray(split[sname], dtype=int)
        if len(idx) == 0:
            continue
        dataset[sname] = {}
        for m in modes:
            arrays = {}
            for k, v in pool[m].items():
                arrays[k] = v if k == "xi" else v[idx]
            if scaler is not None and "pe" in arrays:
                arrays["input_scaled"] = scaler.transform(np.column_stack([
                    np.log(np.maximum(arrays["pe"], 1e-15)),
                    np.log(np.maximum(np.abs(arrays["rho"]), 1e-15)),
                ]))
            dataset[sname][m] = arrays
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--n-fd-points", type=int, default=400,
                    help="FD grid resolution for the reference bubbles "
                         "(400 is under-resolved for the Pe>50 boundary "
                         "layer; use 3200 for accurate high-Pe bubbles)")
    ap.add_argument("--pe-range", nargs=2, type=float, default=[0.3, 100.0])
    ap.add_argument("--rho-range", nargs=2, type=float, default=[0.2, 100.0])
    ap.add_argument("--theta", type=float, default=0.99,
                    help="twin-similarity threshold (default 0.99)")
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="rfb_5k_noleak")
    args = ap.parse_args()

    t0 = time.time()

    # ---- 1. Pool generation --------------------------------------------
    print("=" * 72)
    print(f"STEP 1 — Generating pool: {args.n_samples} bubbles "
          f"(log-uniform Pe x rho)")
    print("=" * 72)
    ds, config = generate_pool(args.n_samples, tuple(args.pe_range),
                               tuple(args.rho_range), args.seed,
                               n_fd_points=args.n_fd_points)
    pool, modes, n = merge_pool(ds)
    print(f"  pool: {n} samples, modes {modes}, "
          f"FD resolution = {args.n_fd_points} points")

    # ---- 2. Cosine-similarity analysis (the problem) --------------------
    print("\n" + "=" * 72)
    print("STEP 2 — Bubble cosine-similarity analysis of the pool")
    print("=" * 72)
    C = pool_similarity_matrix(pool, modes)
    od = _off_diagonal_stats(C)
    er = _effective_rank(C)
    print(f"  pool off-diagonal similarity: mean={od['mean']:.4f}  "
          f"median={od['median']:.4f}")
    print(f"  pairs with C > 0.90: {100 * od['frac_gt_0.90']:.1f}%  "
          f"C > 0.99: {100 * od['frac_gt_0.99']:.1f}%")
    print(f"  effective rank = {er['effective_rank']:.2f}  "
          f"(top-1 eigenvalue = {100 * er['top1_energy']:.1f}% of energy)")
    print("  -> the pool is heavily redundant in function space; a geometric "
          "(Pe, rho) split would leak.")

    # ---- 3. Shape-based no-leak split -----------------------------------
    print("\n" + "=" * 72)
    print(f"STEP 3 — Shape-based split (theta = {args.theta})")
    print("=" * 72)
    n_train = int(round(args.train_frac * n))
    n_val = int(round(args.val_frac * n))
    n_test = int(round(args.test_frac * n))
    split = shape_no_leak_split(C, n_train, n_val, n_test, theta=args.theta)
    st = split["stats"]
    print(f"  target: {n_train} train / {n_val} val / {n_test} test")
    print(f"  after leak filter: {st['n_train']} train / {st['n_val']} val / "
          f"{st['n_test']} test")
    print(f"  dropped as twins: {st['dropped']}")

    # ---- 4. Rebuild, verify, save ---------------------------------------
    scaler = fit_pool_scaler(pool, modes)
    new_ds = build_split_dicts(pool, modes, split, scaler)
    new_ds["mode_names"] = list(modes)

    metadata = dict(config._asdict()) if hasattr(config, "_asdict") else dict(config.__dict__)
    metadata.update({
        "mode_names": list(modes),
        "n_total": n,
        "n_train": st["n_train"],
        "n_val": st["n_val"],
        "n_test": st["n_test"],
        "split_indices": {
            "train": split["train"],
            "val": split["val"],
            "test": split["test"],
        },
        "split_strategy": "no_twin_shape",
        "similarity_theta": args.theta,
        "leak_dropped": st["dropped"],
        "split_by_shape": True,
    })
    metadata.pop("frame_meta", None)
    metadata.pop("cell_map", None)
    new_ds["metadata"] = metadata
    new_ds["scaler"] = scaler

    path = save_dataset(new_ds, name=args.name)
    print(f"\n  saved -> {path}")

    # ---- 5. Confirm leak-free -------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4 — Leakage verification on the new dataset")
    print("=" * 72)
    check = load_dataset(args.name)
    for m in modes:
        res = bubble_similarity_analysis(check, mode=m, verbose=False)
        leak = res["cross"]["train_vs_test"]["stats"]
        frac_gt = leak["frac_gt_0.99"]
        worst = leak["max_sim_max"]
        verdict = "LEAK-FREE" if worst <= args.theta else "STILL LEAKING"
        print(f"\n  [{m}] test bubbles with a train twin (C > 0.99): "
              f"{100 * frac_gt:.2f}%  worst twin sim = {worst:.4f}  "
              f"->  {verdict} (theta = {args.theta})")

    print(f"\n  total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
