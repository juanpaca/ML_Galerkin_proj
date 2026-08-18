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
) -> dict:
    """Split a Darcy pool by normalized solution shape.

    By default, all samples that are not twins of validation/test shapes are
    used for training. ``train_frac`` optionally requests a fixed target size.
    """
    if val_frac < 0.0 or test_frac < 0.0 or val_frac + test_frac >= 1.0:
        raise ValueError("validation/test fractions must leave training samples")
    n = len(pool["constant"]["pe"])
    n_train = None if train_frac is None else int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    split = shape_no_leak_split_from_pool(
        {"constant": pool["constant"]}, ("constant",),
        n_train, n_val, n_test, theta=theta,
    )

    dataset = {"mode_names": ["constant"], "metadata": dict(pool["metadata"])}
    for split_name in ("train", "val", "test"):
        idx = np.asarray(split[split_name], dtype=int)
        dataset[split_name] = {"constant": {}}
        for key, values in pool["constant"].items():
            dataset[split_name]["constant"][key] = (
                values if key == "xi" else values[idx]
            )

    dataset["metadata"].update({
        "name": "darcy_piecewise",
        "mode_names": ["constant"],
        "split_strategy": "no_twin_shape",
        "similarity_theta": theta,
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
) -> dict:
    """Generate, split, save, reload, and audit a Darcy dataset."""
    pool = generate_darcy_pool(
        n_samples=n_samples,
        n_fd_points=n_fd_points,
        n_profile_features=n_profile_features,
        length_range=length_range,
        eps_range=(0.1, 100.0),
        n_pieces_range=(min_pieces, max_pieces),
        seed=seed,
    )
    dataset = build_split_dataset(pool, theta, val_frac, test_frac, train_frac)
    save_dataset(dataset, name=name, subdir=subdir)
    loaded = load_dataset(name, subdir=subdir)
    report = bubble_similarity_analysis(loaded, mode="constant", verbose=False)
    max_similarity = report["cross"]["train_vs_test"]["max_similarity"]
    if np.any(max_similarity > theta):
        raise RuntimeError("Darcy train/test leakage audit failed")
    loaded["leakage_report"] = report
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
    )
    stats = ds["leakage_report"]["cross"]["train_vs_test"]["stats"]
    print(f"Saved datasets/{DATA_SUBDIR}/{args.name}_*.npz")
    print(f"Splits: {stats['n_other']} test samples; "
          f"max train-test similarity={stats['max_sim_max']:.6f}")
    print(f"Test twins above theta={args.theta}: "
          f"{100 * stats['frac_gt_0.99']:.2f}% when theta=0.99")


if __name__ == "__main__":
    main()
