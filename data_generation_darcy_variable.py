#!/usr/bin/env python3
"""Generate a leakage-free dataset for variable-diffusion Darcy problems.

Problem:
    -(epsilon(x) u'(x))' = f(x) on (0, L),  u(0)=u(L)=0.

The generated target is the normalized solution shape u/u(0.5L). Diffusion
profiles are piecewise constant, with values in [0.1, 100].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_generation import shape_no_leak_split_from_pool
from src.darcy_variable import generate_darcy_pool
from src.dataset_generation import bubble_similarity_analysis, load_dataset, save_dataset


DATA_SUBDIR = "data_darcy_variable"


def build_split_dataset(
    pool: dict,
    theta: float = 0.99,
    val_frac: float = 0.15,
    test_frac: float = 0.25,
    train_frac: float | None = None,
    strategy: str = "no_twin_shape",
    seed: int = 0,
) -> dict:
    """Split a Darcy pool by normalized solution shape.

    ``strategy="no_twin_shape"`` (default) reserves validation/test as the
    most atypical solution shapes and trains on everything that is not a
    twin of them. ``strategy="random"`` draws an i.i.d. split instead
    (diagnostic baseline). ``strategy="contrast_band"`` orders samples by
    realized contrast c = eps_max/eps_min and cuts contiguous bands:
    train on the low-contrast subinterval [c_min, c_train] of [1, c_max],
    validation on the next band, test on the top band (complement).
    ``train_frac`` optionally requests a fixed size.
    """
    if val_frac < 0.0 or test_frac < 0.0 or val_frac + test_frac >= 1.0:
        raise ValueError("validation/test fractions must leave training samples")
    if strategy not in ("no_twin_shape", "random", "contrast_band"):
        raise ValueError(f"unknown split strategy: {strategy}")
    n = len(pool["constant"]["pe"])
    n_train = None if train_frac is None else int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    mode_names = tuple(pool.get("mode_names", ("constant", "xi")))
    band_edges = None
    if strategy == "no_twin_shape":
        split = shape_no_leak_split_from_pool(
            {mode: pool[mode] for mode in mode_names}, mode_names,
            n_train, n_val, n_test, theta=theta,
        )
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        split = {
            "test": perm[:n_test],
            "val": perm[n_test:n_test + n_val],
            "train": (perm[n_test + n_val:] if n_train is None
                      else perm[n_test + n_val:n_test + n_val + n_train]),
            "dropped": {"train": np.array([], dtype=int)},
        }
    else:
        contrasts = np.array([
            float(np.max(v)) / float(np.min(v)) for v in pool["piece_values"]
        ])
        order = np.argsort(contrasts, kind="stable")
        split = {
            "test": order[n - n_test:],
            "val": order[n - n_test - n_val:n - n_test],
            "train": (order[:n - n_test - n_val] if n_train is None
                      else order[:n_train]),
            "dropped": {"train": np.array([], dtype=int)},
        }
        band_edges = {
            "train": [float(contrasts[order[0]]),
                      float(contrasts[order[n - n_test - n_val - 1]])],
            "val": [float(contrasts[order[n - n_test - n_val]]),
                    float(contrasts[order[n - n_test - 1]])],
            "test": [float(contrasts[order[n - n_test]]),
                     float(contrasts[order[-1]])],
        }

    dataset = {"mode_names": list(mode_names), "metadata": dict(pool["metadata"])}
    for split_name in ("train", "val", "test"):
        idx = np.asarray(split[split_name], dtype=int)
        dataset[split_name] = {}
        for mode in mode_names:
            dataset[split_name][mode] = {}
            for key, values in pool[mode].items():
                dataset[split_name][mode][key] = (
                    values if key == "xi" else values[idx]
                )

    dataset["metadata"].update({
        "name": "darcy_piecewise",
        "mode_names": list(mode_names),
        "split_strategy": strategy,
        "similarity_theta": theta if strategy == "no_twin_shape" else None,
        "contrast_band_edges": band_edges,
        "n_total": n,
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "n_test": len(split["test"]),
        "split_indices": {k: [int(i) for i in split[k]]
                          for k in ("train", "val", "test")},
        "dropped_count": len(split["dropped"]["train"]),
        "piece_edges": [np.asarray(v).tolist() for v in pool["piece_edges"]],
        "piece_values": [np.asarray(v).tolist() for v in pool["piece_values"]],
    })
    return dataset


def generate_and_save_dataset(
    name: str = "darcy_piecewise",
    subdir: str = DATA_SUBDIR,
    n_samples: int = 5000,
    n_fd_points: int = 801,
    n_profile_features: int = 8,
    min_pieces: int = 2,
    max_pieces: int = 8,
    length_range: tuple[float, float] = (0.5, 2.0),
    theta: float = 0.99,
    val_frac: float = 0.15,
    test_frac: float = 0.25,
    train_frac: float | None = None,
    seed: int = 42,
    feature_kind: str = "gauss_ratio",
    min_width: float = 0.0,
    split_strategy: str = "no_twin_shape",
    eps_range: tuple[float, float] = (0.1, 100.0),
) -> dict:
    """Generate, split, save, reload, and audit a Darcy dataset."""
    pool = generate_darcy_pool(
        n_samples=n_samples,
        n_fd_points=n_fd_points,
        n_profile_features=n_profile_features,
        length_range=length_range,
        eps_range=eps_range,
        n_pieces_range=(min_pieces, max_pieces),
        seed=seed,
        feature_kind=feature_kind,
        min_width=min_width,
    )
    dataset = build_split_dataset(pool, theta, val_frac, test_frac, train_frac,
                                  strategy=split_strategy)
    dataset["metadata"]["name"] = name
    save_dataset(dataset, name=name, subdir=subdir)
    loaded = load_dataset(name, subdir=subdir)
    if split_strategy == "no_twin_shape":
        reports = {
            mode: bubble_similarity_analysis(loaded, mode=mode, verbose=False)
            for mode in loaded["mode_names"]
        }
        max_similarity = np.maximum.reduce([
            report["cross"]["train_vs_test"]["max_similarity"]
            for report in reports.values()
        ])
        if np.any(max_similarity > theta):
            raise RuntimeError("Darcy train/test leakage audit failed")
        loaded["leakage_report"] = reports
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="darcy_piecewise")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-fd-points", type=int, default=801)
    parser.add_argument("--n-profile-features", type=int, default=8)
    parser.add_argument("--min-pieces", type=int, default=2)
    parser.add_argument("--max-pieces", type=int, default=8)
    parser.add_argument("--length-min", type=float, default=0.5)
    parser.add_argument("--length-max", type=float, default=2.0)
    parser.add_argument("--theta", type=float, default=0.99)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-kind",
                        choices=("gauss_ratio", "resistivity_cdf", "scaled_combo"),
                        default="gauss_ratio")
    parser.add_argument("--min-width", type=float, default=0.0,
                        help="minimum piece measure on the normalized interval")
    parser.add_argument("--split-strategy",
                        choices=("no_twin_shape", "random", "contrast_band"),
                        default="no_twin_shape")
    parser.add_argument("--eps-min", type=float, default=0.1,
                        help="lower bound of piecewise-constant eps values")
    parser.add_argument("--eps-max", type=float, default=100.0,
                        help="upper bound of piecewise-constant eps values")
    args = parser.parse_args()

    ds = generate_and_save_dataset(
        name=args.name,
        n_samples=args.n_samples,
        n_fd_points=args.n_fd_points,
        n_profile_features=args.n_profile_features,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        length_range=(args.length_min, args.length_max),
        theta=args.theta,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        train_frac=args.train_frac,
        seed=args.seed,
        feature_kind=args.feature_kind,
        min_width=args.min_width,
        split_strategy=args.split_strategy,
        eps_range=(args.eps_min, args.eps_max),
    )
    print(f"Saved datasets/{DATA_SUBDIR}/{args.name}_*.npz")
    if args.split_strategy == "contrast_band":
        edges = ds["metadata"]["contrast_band_edges"]
        for band in ("train", "val", "test"):
            lo, hi = edges[band]
            print(f"{band:5s}: contrast in [{lo:.3f}, {hi:.2f}] "
                  f"({ds['metadata']['n_' + band]} samples)")
    else:
        stats = ds["leakage_report"]["constant"]["cross"]["train_vs_test"]["stats"]
        print(f"Splits: {stats['n_other']} test samples; "
              f"max train-test similarity={stats['max_sim_max']:.6f}")
        print(f"Test twins above theta={args.theta}: "
              f"{100 * stats['frac_gt_0.99']:.2f}% when theta=0.99")


if __name__ == "__main__":
    main()
