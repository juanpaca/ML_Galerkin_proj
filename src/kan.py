import torch
import torch.nn as nn
import numpy as np


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _extend_knots(x_min: float, x_max: float, G: int, k: int) -> torch.Tensor:
    interior = torch.linspace(x_min, x_max, G + 1)
    left = torch.full((k - 1,), x_min)
    right = torch.full((k - 1,), x_max)
    return torch.cat([left, interior, right])


def _eval_bspline_basis(x: torch.Tensor, knots: torch.Tensor, k: int) -> torch.Tensor:
    """GPU-friendly B-spline basis evaluation.

    Returns (B, nb) basis matrix — only k non-zero entries per row.
    Uses x.contiguous() + torch.where for GPU efficiency.
    Only knot lookup is no_grad; the Cox-de Boor recurrence retains
    gradient tracking through x for d(basis)/dx.
    """
    B = x.shape[0]
    n_knots = knots.shape[0]
    nb = n_knots - k
    device = x.device
    x_c = x.contiguous()

    with torch.no_grad():
        s = torch.searchsorted(knots, x_c, right=True) - 1
        s = s.clamp(0, nb - 1).long()

    Nk = [torch.ones(B, device=device)]

    for j in range(1, k):
        rng = torch.arange(j, device=device, dtype=s.dtype)
        idx_l = (s[:, None] - rng[None, :]).clamp(0, n_knots - 1)
        idx_r = (s[:, None] + 1 + rng[None, :]).clamp(0, n_knots - 1)
        left = x_c[:, None] - knots[idx_l]
        right = knots[idx_r] - x_c[:, None]
        saved = torch.zeros(B, device=device)
        new_Nk = []
        for r in range(j):
            denom = right[:, r] + left[:, j - 1 - r]
            inv = torch.where(denom.abs() > 1e-15, 1.0 / denom, torch.zeros_like(denom))
            temp = Nk[r] * inv
            N_jr = saved + right[:, r] * temp
            saved = left[:, j - 1 - r] * temp
            new_Nk.append(N_jr)
        new_Nk.append(saved)
        Nk = new_Nk

    basis = torch.zeros(B, nb, device=device)
    offsets = s[:, None] - (k - 1) + torch.arange(k, device=device, dtype=s.dtype)[None, :]
    for r in range(k):
        gi = offsets[:, r].clamp(0, nb - 1).long()
        basis[torch.arange(B, device=device), gi] = Nk[r]
    return basis


class KAN1D(nn.Module):
    """Learnable 1D function: phi(x) = w_b * SiLU(x) + w_s * Σ c_i B_i^k(x)

    Gradients track through all parameters (w_b, w_s, c).
    Accepts optional precomputed_basis to skip B-spline evaluation.
    """

    def __init__(
        self,
        n_grid: int = 8,
        k: int = 3,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ):
        super().__init__()
        self.n_grid = n_grid
        self.k = k
        self.n_basis = n_grid + k - 1

        self.register_buffer("knots", _extend_knots(x_min, x_max, n_grid, k))
        self.c = nn.Parameter(torch.randn(self.n_basis) * 0.1)
        self.w_s = nn.Parameter(torch.tensor(1.0))
        self.w_b = nn.Parameter(torch.empty(1, 1))
        nn.init.xavier_uniform_(self.w_b)
        self.w_b = nn.Parameter(self.w_b.flatten())

    def forward(self, x: torch.Tensor, precomputed_basis: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2 and x.shape[1] == 1:
            x = x[:, 0]
        x = x.flatten()
        base = self.w_b * silu(x)
        if precomputed_basis is not None:
            spline = precomputed_basis @ self.c
        else:
            basis = _eval_bspline_basis(x, self.knots, self.k)
            spline = basis @ self.c
        return base + self.w_s * spline

    def forward_with_deriv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (g(x), g'(x)) both differentiable w.r.t. θ."""
        x_in = x.detach().clone().requires_grad_(True)
        g = self.forward(x_in)
        dg = torch.autograd.grad(g, x_in, torch.ones_like(g), create_graph=True)[0]
        return g, dg

    def get_coefficients(self):
        return {
            "w_b": self.w_b.item(),
            "w_s": self.w_s.item(),
            "c": self.c.detach().cpu().numpy(),
        }
