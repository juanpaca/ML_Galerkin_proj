import numpy as np


class GaussLegendre:
    """Gauss–Legendre quadrature on the reference interval [-1, 1]."""

    def __init__(self, n_points: int):
        self.n_points = n_points
        self.ref_points, self.ref_weights = np.polynomial.legendre.leggauss(
            n_points
        )

    def map_to_interval(
        self, x_left: float, x_right: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map quadrature from [-1,1] to [x_left, x_right].

        Returns (physical_points, physical_weights).
        """
        h = (x_right - x_left) / 2.0
        x = x_left + h * (1.0 + self.ref_points)
        w = h * self.ref_weights
        return x, w
