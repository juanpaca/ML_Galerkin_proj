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
from src.darcy_assembly import enrichment_l2_gate
from src.darcy_variable import PiecewiseDiffusion, generate_darcy_pool
from src.dataset_generation import bubble_similarity_analysis, load_dataset, save_dataset


DATA_SUBDIR = "data_darcy_variable"

_GATE_SOURCES = {
    "constant": lambda x: np.ones_like(np.asarray(x, dtype=float)),
    "xi": lambda x: np.asarray(x, dtype=float),
}


def _profile_from_pool(pool: dict, i: int) -> PiecewiseDiffusion:
    return PiecewiseDiffusion(np.asarray(pool["piece_edges"][i], dtype=float),
                              np.asarray(pool["piece_values"][i], dtype=float))


def _filter_pool(pool: dict, keep: np.ndarray) -> dict:
    """Keep the pool samples with ``keep[i]`` True (all arrays stay in sync)."""
    keep = np.asarray(keep, dtype=bool)
    modes = {}
    for mode in pool["mode_names"]:
        d = pool[mode]
        modes[mode] = {
            k: (v if k == "xi" else v[keep]) for k, v in d.items()
        }
    out = {
        **modes,
        "mode_names": pool["mode_names"],
        "piece_edges": [e for e, k in zip(pool["piece_edges"], keep) if k],
        "piece_values": [v for v, k in zip(pool["piece_values"], keep) if k],
        "metadata": dict(pool["metadata"]),
    }
    out["metadata"]["n_samples"] = int(keep.sum())
    return out


def audit_enrichment_gate(
    pool: dict,
    threshold: float = 1e-2,
    n_ref: int = 32001,
    n_el: int = 8,
    drop: bool = False,
) -> tuple[dict, dict]:
    """Pre-training quality gate: every bubble must enrich P1 to rel-L2<thr.

    Runs the static-condensation enrichment with the pool's exact FD bubbles
    (full-domain b_hat, b_tilde), on a uniform ``n_el``+1-node P1 mesh (the
    deployment assembly in the demo), against an independent ``n_ref``
    reference, per source mode.  Returns ``(pool' , report)`` where
    badly-behaved samples are dropped when ``drop=True``, otherwise a hard
    failure is raised listing the worst offender.
    """
    n = len(pool["constant"]["pe"])
    per_mode = {mode: np.zeros(n) for mode in pool["mode_names"]}
    worst = {"mode": None, "idx": -1, "rel_l2": 0.0}
    for i in range(n):
        profile = _profile_from_pool(pool, i)
        bubbles = np.stack([
            pool[mode]["b"][i] for mode in pool["mode_names"]
        ])
        result = enrichment_l2_gate(
            profile, bubbles, _GATE_SOURCES, n_ref=n_ref, n_el=n_el,
        )
        for mode, err in result.items():
            per_mode[mode][i] = err
            if err > worst["rel_l2"]:
                worst = {"mode": mode, "idx": i, "rel_l2": err}

    mx = {mode: float(np.max(per_mode[mode])) for mode in per_mode}
    report = {
        "threshold": threshold,
        "n_ref": n_ref,
        "mesh": f"uniform_n_el={n_el}",
        "per_mode_max": mx,
        "per_mode_mean": {m: float(np.mean(v)) for m, v in per_mode.items()},
        "per_mode_p95": {m: float(np.quantile(v, 0.95)) for m, v in per_mode.items()},
        "worst": worst,
        "n_checked": n,
    }
    failed = np.any(np.stack([per_mode[m] for m in per_mode]) > threshold, axis=0)
    report["n_failed"] = int(failed.sum())
    if report["n_failed"]:
        if not drop:
            raise RuntimeError(
                f"enrichment gate failed for {failed.sum()} samples "
                f"(worst {worst['mode']} #{worst['idx']}: "
                f"rel-L2={worst['rel_l2']:.3e} > {threshold:g})")
        print(f"[gate] dropping {failed.sum()}/{n} samples above "
              f"threshold {threshold:g}")
        pool = _filter_pool(pool, ~failed)
        report["n_dropped"] = int(failed.sum())
    return pool, report


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
    verify_enrichment: bool = True,
    gate_threshold: float = 1e-2,
    gate_n_ref: int = 32001,
    gate_n_el: int = 8,
    gate_drop: bool = False,
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
    gate_report = None
    if verify_enrichment:
        pool, gate_report = audit_enrichment_gate(
            pool, threshold=gate_threshold, n_ref=gate_n_ref,
            n_el=gate_n_el, drop=gate_drop,
        )
        print("[gate] enrichment rel-L2  "
              + "  ".join(
                  f"{m}: mean={gate_report['per_mode_mean'][m]:.2e} "
                  f"p95={gate_report['per_mode_p95'][m]:.2e} "
                  f"max={gate_report['per_mode_max'][m]:.2e}"
                  for m in pool["mode_names"]))
        print(f"[gate] worst #{gate_report['worst']['idx']} "
              f"({gate_report['worst']['mode']}): "
              f"rel-L2={gate_report['worst']['rel_l2']:.3e}  "
              f"threshold={gate_threshold:g}  "
              f"checked={gate_report['n_checked']}")
    dataset = build_split_dataset(pool, theta, val_frac, test_frac, train_frac,
                                  strategy=split_strategy)
    dataset["metadata"]["name"] = name
    if gate_report is not None:
        dataset["metadata"]["enrichment_gate"] = gate_report
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
                        choices=("gauss_ratio", "resistivity_cdf", "scaled_combo",
                                 "scaled_combo_v2"),
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
    parser.add_argument("--verify-enrichment", dest="verify_enrichment",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="audit every bubble via P1+condensation enrichment "
                             "against an independent fine reference")
    parser.add_argument("--gate-threshold", type=float, default=1e-2,
                        help="max allowed enriched rel-L2 per sample")
    parser.add_argument("--gate-n-ref", type=int, default=32001,
                        help="independent reference resolution for the gate")
    parser.add_argument("--gate-n-elements", type=int, default=8,
                        help="P1 elements in the gate's condensation mesh")
    parser.add_argument("--gate-drop", action="store_true",
                        help="drop failing samples instead of aborting")
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
        verify_enrichment=args.verify_enrichment,
        gate_threshold=args.gate_threshold,
        gate_n_ref=args.gate_n_ref,
        gate_n_el=args.gate_n_elements,
        gate_drop=args.gate_drop,
    )
    print(f"Saved datasets/{DATA_SUBDIR}/{args.name}_*.npz")
    if args.split_strategy == "contrast_band":
        edges = ds["metadata"]["contrast_band_edges"]
        for band in ("train", "val", "test"):
            lo, hi = edges[band]
            print(f"{band:5s}: contrast in [{lo:.3f}, {hi:.2f}] "
                  f"({ds['metadata']['n_' + band]} samples)")
    elif "leakage_report" in ds and "train_vs_test" in ds["leakage_report"]["constant"]["cross"]:
        stats = ds["leakage_report"]["constant"]["cross"]["train_vs_test"]["stats"]
        print(f"Splits: {stats['n_other']} test samples; "
              f"max train-test similarity={stats['max_sim_max']:.6f}")
        print(f"Test twins above theta={args.theta}: "
              f"{100 * stats['frac_gt_0.99']:.2f}% when theta=0.99")
    else:
        print(f"Splits: {[k for k in ('train','val','test')]} "
              f"{[ds['metadata']['n_' + k] for k in ('train','val','test')]} samples")


if __name__ == "__main__":
    main()
