import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kan import _extend_knots


class KANLayer(nn.Module):
    """True KAN layer: maps n_in → n_out with learnable edge functions.

    Forward: out_j = Σ_{i=1}^{n_in} φ_{j,i}(in_i)

    Each φ_{j,i}(x) = w_b·SiLU(x) + w_s·Σc_k·B_k(x).

    GPU-efficient: B-spline bases for ALL input columns are computed
    in one 3D tensor operation (B, n_in, n_basis), weighted with a
    single F.linear call.  No Python loops over edge functions.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_grid: int = 8,
        k: int = 3,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.n_basis = n_grid + k - 1

        self.register_buffer("knots", _extend_knots(x_min, x_max, n_grid, k))

        self.base_weight = nn.Parameter(torch.empty(n_out, n_in))
        nn.init.xavier_uniform_(self.base_weight)

        self.spline_weight = nn.Parameter(torch.randn(n_out, n_in, self.n_basis) * 0.1)
        self.spline_scaler = nn.Parameter(torch.ones(n_out, n_in))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        x_3d = x.unsqueeze(-1)
        k_row = self.knots.view(1, 1, -1)

        bases = ((x_3d >= k_row[..., :-1]) & (x_3d < k_row[..., 1:])).to(x.dtype)

        for deg in range(1, self.k):
            left_den = k_row[..., deg:-1] - k_row[..., :-(deg+1)] + 1e-12
            right_den = k_row[..., deg+1:] - k_row[..., 1:(-deg)] + 1e-12
            left = (x_3d - k_row[..., :-(deg+1)]) / left_den * bases[..., :-1]
            right = (k_row[..., deg+1:] - x_3d) / right_den * bases[..., 1:]
            bases = left + right

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.contiguous()

        base = F.linear(F.silu(x), self.base_weight)

        bases = self.b_splines(x)
        w_scaled = (self.spline_weight * self.spline_scaler.unsqueeze(-1)).reshape(self.n_out, -1)
        spline = F.linear(bases.reshape(batch, -1), w_scaled)

        return base + spline


def _scale_pe_rho(pe: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pe_s = torch.log1p(torch.clamp(pe, min=0.0)) / 6.0
    rho_s = torch.log1p(torch.clamp(rho, min=0.0)) / 6.0
    return pe_s, rho_s


class KANBubble1D(nn.Module):
    """True KAN bubble: maps (pe, rho, xi) → b(xi) through KAN layers.

    Architecture: [3+n_eps, n_hidden, 1] KAN → softplus → envelope → normalize.
    All inputs (pe_s, rho_s, xi_s, eps_s) are mapped to [-1, 1] for the KAN.

    Parameters
    ----------
    n_hidden : int
        Width of the single hidden KAN layer.
    n_grid : int
        Number of B-spline grid intervals per edge function.
    spline_order : int
        B-spline order (default 3 = cubic).
    delta : float
        Small offset in softplus to avoid division by zero.
    n_eps : int
        Number of eps profile samples (0 = constant eps).
    """

    def __init__(
        self,
        n_hidden: int = 5,
        n_grid: int = 8,
        spline_order: int = 3,
        delta: float = 1e-4,
        n_eps: int = 0,
    ):
        super().__init__()
        self.delta = delta
        self.n_eps = n_eps
        n_in = 3 + n_eps

        self.kan = nn.Sequential(
            KANLayer(n_in, n_hidden, n_grid=n_grid, k=spline_order),
            KANLayer(n_hidden, 1, n_grid=n_grid, k=spline_order),
        )

    def _build_input(self, xi, pe, rho, eps_ratios=None):
        xi = xi.flatten()
        pe = torch.as_tensor(pe, dtype=xi.dtype, device=xi.device)
        rho = torch.as_tensor(rho, dtype=xi.dtype, device=xi.device)
        if pe.dim() == 0:
            pe = pe.expand_as(xi)
        elif pe.dim() == 1 and pe.shape[0] == 1:
            pe = pe.expand_as(xi)
        if rho.dim() == 0:
            rho = rho.expand_as(xi)
        elif rho.dim() == 1 and rho.shape[0] == 1:
            rho = rho.expand_as(xi)

        pe_s, rho_s = _scale_pe_rho(pe, rho)
        xi_s = 2.0 * xi - 1.0  # map [0, 1] → [-1, 1]

        scaled = torch.stack([pe_s, rho_s, xi_s], dim=-1)
        if eps_ratios is not None:
            eps_ratios = torch.as_tensor(eps_ratios, dtype=xi.dtype, device=xi.device)
            if eps_ratios.dim() == 1:
                eps_ratios = eps_ratios.expand(xi.shape[0], -1)
            eps_s = torch.log1p(torch.clamp(eps_ratios, min=0.0)) / 6.0
            scaled = torch.cat([scaled, eps_s], dim=-1)
        return scaled

    def _raw(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.kan(x_in).squeeze(-1)

    def norm_at_mid(self, pe, rho, eps_ratios=None):
        dev = next(self.parameters()).device
        pe_t = torch.as_tensor(pe, device=dev)
        rho_t = torch.as_tensor(rho, device=dev)
        if eps_ratios is not None:
            eps_ratios = torch.as_tensor(eps_ratios, device=dev)
        mid = torch.full_like(pe_t, 0.5)
        x_mid = self._build_input(mid, pe_t, rho_t, eps_ratios)
        raw_mid = self._raw(x_mid)
        return F.softplus(raw_mid) + self.delta

    def forward(self, xi, pe, rho, eps_ratios=None, norm_factor=None):
        dev = next(self.parameters()).device
        xi = torch.as_tensor(xi, device=dev).flatten()
        pe = torch.as_tensor(pe, device=dev)
        rho = torch.as_tensor(rho, device=dev)
        if eps_ratios is not None:
            eps_ratios = torch.as_tensor(eps_ratios, device=dev)
        x_in = self._build_input(xi, pe, rho, eps_ratios)
        raw = self._raw(x_in)
        positive = F.softplus(raw) + self.delta

        if norm_factor is not None:
            bs = norm_factor.shape[0]
            q = xi.shape[0] // bs
            pos = positive.view(bs, q)
            nf = norm_factor.unsqueeze(-1)
            env = (4.0 * xi * (1.0 - xi)).view(bs, q)
            return (env * pos / nf).reshape(-1)

        mid = torch.full_like(xi, 0.5)
        x_mid = self._build_input(mid, pe, rho, eps_ratios)
        raw_mid = self._raw(x_mid)
        positive_mid = F.softplus(raw_mid) + self.delta

        envelope = 4.0 * xi * (1.0 - xi)
        return envelope * positive / positive_mid

    def value_grad_numpy(
        self,
        xi: np.ndarray,
        pe: float,
        rho: float,
        eps_ratios: np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        dev = next(self.parameters()).device
        xi_t = torch.tensor(np.asarray(xi, dtype=float), dtype=dtype, device=dev, requires_grad=True)
        pe_t = torch.full_like(xi_t, float(pe))
        rho_t = torch.full_like(xi_t, float(rho))
        eps_t = None
        if eps_ratios is not None:
            eps_t = torch.tensor(np.asarray(eps_ratios, dtype=float), dtype=dtype, device=dev)
            eps_t = eps_t.expand(xi_t.shape[0], -1)
        b = self.forward(xi_t, pe_t, rho_t, eps_ratios=eps_t)
        db = torch.autograd.grad(b, xi_t, torch.ones_like(b), create_graph=False)[0]
        return b.detach().cpu().numpy(), db.detach().cpu().numpy()


class MultiKANBubble1D(nn.Module):
    """Collection of learned bubbles, one per residual mode."""

    def __init__(self, n_bubbles: int = 2, n_eps: int = 0, **bubble_kwargs):
        super().__init__()
        self.n_bubbles = n_bubbles
        self.n_eps = n_eps
        self.bubbles = nn.ModuleList(
            [KANBubble1D(n_eps=n_eps, **bubble_kwargs) for _ in range(n_bubbles)]
        )

    def norm_at_mid(self, pe, rho, eps_ratios=None):
        return torch.stack([b.norm_at_mid(pe, rho, eps_ratios=eps_ratios) for b in self.bubbles], dim=0)

    def forward(self, xi, pe, rho, eps_ratios=None, norm_factor=None):
        if norm_factor is not None:
            return torch.stack([
                b(xi, pe, rho, eps_ratios=eps_ratios, norm_factor=nf)
                for b, nf in zip(self.bubbles, norm_factor)
            ], dim=0)
        return torch.stack([bubble(xi, pe, rho, eps_ratios=eps_ratios) for bubble in self.bubbles], dim=0)

    def value_grad_numpy(
        self,
        xi: np.ndarray,
        pe: float,
        rho: float,
        eps_ratios: np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        vals, grads = [], []
        for bubble in self.bubbles:
            b, db = bubble.value_grad_numpy(xi, pe, rho, eps_ratios=eps_ratios, dtype=dtype)
            vals.append(b)
            grads.append(db)
        return np.vstack(vals), np.vstack(grads)
