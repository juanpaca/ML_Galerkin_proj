import os
from pathlib import Path

import numpy as np
import torch

from src.rfb_bubble import KANBubble1D, MultiKANBubble1D
from src.rfb_local import solve_reference_rfb, interpolate_target, local_parameters

DATASET_SUBDIR = "datasets"
DTYPE = np.float32


def _dataset_path(name: str) -> Path:
    path = Path(DATASET_SUBDIR) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pack_sample(sample: dict) -> dict:
    """Convert a sample dict to a flat dict of numpy arrays for NPZ storage.

    Key convention: single-letter prefixes for compact storage.
    Arrays that are constant across the dataset (like xi) are stored once
    under the first sample's keys.
    """
    out = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[f"p_{sk}"] = np.atleast_1d(np.asarray(sv, dtype=DTYPE))
        elif isinstance(v, np.ndarray):
            out[k] = np.asarray(v, dtype=DTYPE)
        elif np.isscalar(v):
            out[k] = np.atleast_1d(np.asarray(v, dtype=DTYPE))
        else:
            out[k] = np.atleast_1d(np.asarray(v, dtype=DTYPE))
    return out


def _unpack_sample(data: dict, idx: int) -> dict:
    """Reconstruct a sample dict from a loaded NPZ archive at index idx."""
    sample = {}
    for k in data.keys():
        arr = data[k]
        if k == "arr_0":
            continue
        if k.startswith("p_"):
            sk = k[2:]
            if "params" not in sample:
                sample["params"] = {}
            val = arr[idx] if arr.ndim > 1 else arr[0]
            sample["params"][sk] = val.item() if np.ndim(val) == 0 else val
        else:
            val = arr[idx] if arr.ndim > 1 else arr
            if val.ndim == 1 and val.shape[0] == 1:
                sample[k] = val.item()
            else:
                sample[k] = val
    for key in ("xi",):
        sample.pop(key, None)
    return sample


def save_training_data(samples: list[dict], name: str) -> str:
    """Save a list of sample dicts to a NPZ file in datasets/.

    Parameters
    ----------
    samples : list[dict]
        Training samples (as returned by generate_rfb_training_data*).
    name : str
        Dataset name (without extension). Saved as datasets/{name}.npz.

    Returns
    -------
    str : full path to the saved file.
    """
    packed = [_pack_sample(s) for s in samples]
    merged: dict[str, list[np.ndarray]] = {}
    for p in packed:
        for k, v in p.items():
            merged.setdefault(k, []).append(v)
    arrays = {}
    for k, lst in merged.items():
        try:
            arrays[k] = np.stack(lst, axis=0)
        except ValueError:
            arrays[k] = np.stack([np.asarray(v, dtype=DTYPE).ravel() for v in lst], axis=0)
    path = _dataset_path(name)
    np.savez_compressed(str(path), **arrays)
    return str(path)


def load_training_data(name: str) -> list[dict]:
    """Load training samples saved with save_training_data.

    Parameters
    ----------
    name : str
        Dataset name (with or without .npz extension).

    Returns
    -------
    list[dict] : loaded samples.
    """
    path = _dataset_path(name)
    if not path.exists():
        explicit = Path(name)
        if explicit.exists():
            path = explicit
        else:
            raise FileNotFoundError(f"Dataset not found: {name} (tried {path} and {name})")
    data = np.load(str(path))
    n = max(v.shape[0] for v in data.values() if v.ndim >= 1 and v.shape[0] > 1) // 3 * 3
    samples = [_unpack_sample(data, i) for i in range(n)]
    for s in samples:
        s["xi"] = np.linspace(0.0, 1.0, 400)
    return samples


def _gauss_legendre_01(n: int) -> np.ndarray:
    """Gauss-Legendre nodes on [0, 1]."""
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0)


def generate_rfb_training_data(
    n_samples: int,
    h: float,
    eps_range: tuple[float, float] = (1e-3, 5e-2),
    beta_range: tuple[float, float] = (1.0, 1.0),
    sigma_range: tuple[float, float] = (0.0, 0.0),
    residual_mode: str = "constant",
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        eps = 10.0 ** rng.uniform(np.log10(eps_range[0]), np.log10(eps_range[1]))
        beta = rng.uniform(*beta_range)
        sigma = rng.uniform(*sigma_range)
        target = solve_reference_rfb(eps, beta, sigma, h, residual_mode=residual_mode)
        pe, rho = local_parameters(eps, beta, sigma, h)
        target["pe"] = pe
        target["rho"] = rho
        samples.append(target)
    return samples


def generate_rfb_training_data_variable_eps(
    n_samples: int,
    h: float,
    eps_fn: callable,
    n_eps: int = 5,
    residual_mode: str = "constant",
    beta: float = 1.0,
    sigma: float = 0.0,
    n_fd_points: int = 400,
    seed: int = 0,
) -> list[dict]:
    """Generate training samples for a variable diffusion coefficient.

    The diffusion coefficient is eps_fn(xi) where xi in [0, 1] is the
    reference-element coordinate.  The FD bubble solver uses the actual
    per-point eps values, and each sample stores eps_ratios for the KAN.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    h : float
        Element length.
    eps_fn : callable
        Function eps(xi) returning the diffusion coefficient at each
        reference-element coordinate xi in [0, 1].  Called as
        ``eps_fn(np.linspace(0, 1, n_fd_points))`` to get an array.
        If eps_fn returns a scalar (constant), it is broadcast.
    n_eps : int
        Number of sample points for the eps profile (KAN input).
    residual_mode : str
        Residual mode ("constant", "xi", "one_minus_xi").
    beta, sigma : float
        Constant advection and reaction.
    n_fd_points : int
        Number of FD grid points for the reference solution.
    seed : int
        Random seed (used for variation in eps_fn — a constant scale
        is applied per sample).
    """
    rng = np.random.default_rng(seed)
    xi_fd = np.linspace(0.0, 1.0, n_fd_points)
    xi_eps = _gauss_legendre_01(n_eps)

    samples = []
    for _ in range(n_samples):
        scale = 10.0 ** rng.uniform(-2.0, 0.0)
        eps_on_xi = np.asarray(eps_fn(xi_fd), dtype=float)
        if eps_on_xi.ndim == 0:
            eps_on_xi = np.full(n_fd_points, eps_on_xi)
        eps_on_xi *= scale  # random scaling

        eps_avg = float(np.mean(eps_on_xi))
        target = solve_reference_rfb(eps_on_xi, beta, sigma, h,
                                      residual_mode=residual_mode,
                                      n_points=n_fd_points)

        eps_at_sample = np.interp(xi_eps, xi_fd, eps_on_xi)
        target["eps_ratios"] = np.asarray(eps_at_sample / eps_avg, dtype=float)
        target["pe"] = float(abs(beta) * h / (2.0 * eps_avg)) if eps_avg > 0 else np.inf
        target["rho"] = float(sigma * h * h / eps_avg) if eps_avg > 0 else np.inf
        samples.append(target)
    return samples


def generate_rfb_training_data_by_mode(
    n_samples: int,
    h: float,
    residual_modes: tuple[str, ...] = ("constant", "xi"),
    eps_range: tuple[float, float] = (1e-3, 5e-2),
    beta_range: tuple[float, float] = (1.0, 1.0),
    sigma_range: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
) -> list[list[dict]]:
    return [
        generate_rfb_training_data(
            n_samples=n_samples,
            h=h,
            eps_range=eps_range,
            beta_range=beta_range,
            sigma_range=sigma_range,
            residual_mode=mode,
            seed=seed + 7919 * i,
        )
        for i, mode in enumerate(residual_modes)
    ]


def generate_rfb_training_data_cs(
    n_samples: int,
    h: float,
    eps_range: tuple[float, float] = (1e-3, 5e-2),
    beta_range: tuple[float, float] = (1.0, 1.0),
    sigma_range: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
) -> list[list[dict]]:
    """Generate training data for the three companion/source bubbles.

    Returns three lists of samples corresponding to:
        b_1  : companion bubble from -L phi_1  (mode "companion_1")
        b_2  : companion bubble from -L phi_2  (mode "companion_2")
        b^f  : source bubble from f            (mode "constant")

    The source bubble uses a unit RHS (f = 1).  After normalization the
    result is independent of f because the scaling cancels, so the trained
    bubble serves for any source amplitude in the assembly.

    Returns
    -------
    list[list[dict]]
        Three inner lists, one per bubble with ``n_samples`` dicts each.
    """
    modes = ("companion_1", "companion_2", "constant")
    return [
        generate_rfb_training_data(
            n_samples=n_samples,
            h=h,
            eps_range=eps_range,
            beta_range=beta_range,
            sigma_range=sigma_range,
            residual_mode=mode,
            seed=seed + 7919 * i,
        )
        for i, mode in enumerate(modes)
    ]


def train_bubble_model(
    model: KANBubble1D,
    samples: list[dict],
    n_epochs: int = 300,
    lr: float = 1e-3,
    n_quad: int = 80,
    grad_weight: float = 1e-3,
    max_principle_weight: float = 0.0,
    verbose: bool = True,
) -> list[float]:
    xi_np = np.linspace(0.0, 1.0, n_quad)
    xi_np[0] = 1e-6
    xi_np[-1] = 1.0 - 1e-6
    xi_base = torch.tensor(xi_np, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total = 0.0
        for sample in samples:
            xi = xi_base.clone().detach().requires_grad_(True)
            pe = torch.full_like(xi, float(sample["pe"]))
            rho = torch.full_like(xi, float(sample["rho"]))
            eps_ratios = sample.get("eps_ratios")
            if eps_ratios is not None:
                eps_ratios_t = torch.tensor(np.asarray(eps_ratios, dtype=float), dtype=torch.float32)
            else:
                eps_ratios_t = None
            pred = model(xi, pe, rho, eps_ratios=eps_ratios_t)
            dpred = torch.autograd.grad(
                pred, xi, torch.ones_like(pred), create_graph=True
            )[0]

            b_target, db_target = interpolate_target(sample, xi_np)
            b_t = torch.tensor(b_target, dtype=torch.float32)
            db_t = torch.tensor(db_target, dtype=torch.float32)
            total = total + torch.mean((pred - b_t) ** 2)
            total = total + grad_weight * torch.mean((dpred - db_t) ** 2)
            if max_principle_weight > 0.0:
                lower = float(np.min(b_target))
                upper = float(np.max(b_target))
                lower_t = torch.tensor(lower, dtype=torch.float32)
                upper_t = torch.tensor(upper, dtype=torch.float32)
                mp_loss = torch.mean(torch.relu(lower_t - pred) ** 2)
                mp_loss = mp_loss + torch.mean(torch.relu(pred - upper_t) ** 2)
                total = total + max_principle_weight * mp_loss

        loss = total / len(samples)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch + 1}/{n_epochs}: loss={losses[-1]:.6e}")
    return losses


def train_multi_bubble_model(
    model: MultiKANBubble1D,
    samples_by_mode: list[list[dict]],
    n_epochs: int = 300,
    lr: float = 1e-3,
    n_quad: int = 80,
    grad_weight: float = 1e-3,
    max_principle_weight: float = 0.0,
    verbose: bool = True,
) -> list[list[float]]:
    histories = []
    for i, samples in enumerate(samples_by_mode):
        if verbose:
            print(f"Training bubble mode {i + 1}/{len(samples_by_mode)}")
        history = train_bubble_model(
            model.bubbles[i],
            samples,
            n_epochs=n_epochs,
            lr=lr,
            n_quad=n_quad,
            grad_weight=grad_weight,
            max_principle_weight=max_principle_weight,
            verbose=verbose,
        )
        histories.append(history)
    return histories
