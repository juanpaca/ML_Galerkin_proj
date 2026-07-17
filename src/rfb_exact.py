import numpy as np

from src.rfb_local import solve_reference_rfb, interpolate_target


class ExactRFBubble1D:
    """Reference residual-free bubble provider for one residual mode.

    This class exposes the same value_grad_numpy interface as KANBubble1D, so
    it can be used directly by the statically condensed RFB assembler.

    Supported ``residual_mode`` values:
        * ``"constant"``     — RHS = 1  (same as the source bubble for f=1)
        * ``"xi"``           — RHS = xi
        * ``"one_minus_xi"`` — RHS = 1 - xi
        * ``"companion_1"``  — RHS = beta/h - sigma*(1-xi)  (from -L phi_1)
        * ``"companion_2"``  — RHS = -beta/h - sigma*xi      (from -L phi_2)
    """

    def __init__(
        self,
        eps: float,
        beta: float,
        sigma: float,
        h: float,
        residual_mode: str = "constant",
        n_points: int = 4000,
    ):
        self.eps = float(eps)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.h = float(h)
        self.residual_mode = residual_mode
        self.n_points = n_points
        self._target = solve_reference_rfb(
            self.eps,
            self.beta,
            self.sigma,
            self.h,
            residual_mode=residual_mode,
            n_points=n_points,
        )

    def value_grad_numpy(
        self,
        xi: np.ndarray,
        pe: float | None = None,
        rho: float | None = None,
        eps_ratios: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return interpolate_target(self._target, np.asarray(xi, dtype=float))


class ExactRFBubbleSet1D:
    """Multiple exact RFB modes, e.g. companion bubbles (b_1, b_2) and source."""

    def __init__(
        self,
        eps: float,
        beta: float,
        sigma: float,
        h: float,
        residual_modes: tuple[str, ...] = ("constant", "xi"),
        n_points: int = 4000,
    ):
        self.residual_modes = residual_modes
        self.bubbles = [
            ExactRFBubble1D(
                eps=eps,
                beta=beta,
                sigma=sigma,
                h=h,
                residual_mode=mode,
                n_points=n_points,
            )
            for mode in residual_modes
        ]

    def value_grad_numpy(
        self,
        xi: np.ndarray,
        pe: float | None = None,
        rho: float | None = None,
        eps_ratios: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        vals, grads = [], []
        for bubble in self.bubbles:
            b, db = bubble.value_grad_numpy(xi, pe, rho, eps_ratios=eps_ratios)
            vals.append(b)
            grads.append(db)
        return np.vstack(vals), np.vstack(grads)
