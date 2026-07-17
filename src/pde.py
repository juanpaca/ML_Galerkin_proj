import numpy as np


class AdvectionDiffusion1D:
    """1D advection-diffusion-reaction problem.

    -(eps u')' + b u' + sigma u = f  in (0,1)

    with homogeneous Dirichlet BCs: u(0) = u(1) = 0.

    All coefficients can be set as constants (constructor) or as functions
    of x via set_*_from_function().  When a function is set it takes
    precedence over the constant value.
    """

    def __init__(self, eps: float = 1.0, beta: float = 1.0, sigma: float = 0.0):
        self.eps = eps
        self.beta = beta
        self.sigma = sigma
        self._source_fn = None
        self._diffusion_fn = None
        self._advection_fn = None
        self._reaction_fn = None

    def diffusion(self, x: np.ndarray | float) -> np.ndarray | float:
        if self._diffusion_fn is not None:
            return self._diffusion_fn(x)
        arr = np.asarray(x, dtype=float)
        return np.full_like(arr, self.eps)

    def advection(self, x: np.ndarray | float) -> np.ndarray | float:
        if self._advection_fn is not None:
            return self._advection_fn(x)
        arr = np.asarray(x, dtype=float)
        return np.full_like(arr, self.beta)

    def reaction(self, x: np.ndarray | float) -> np.ndarray | float:
        if self._reaction_fn is not None:
            return self._reaction_fn(x)
        arr = np.asarray(x, dtype=float)
        return np.full_like(arr, self.sigma)

    def source(self, x: np.ndarray | float) -> np.ndarray | float:
        if self._source_fn is not None:
            return self._source_fn(x)
        arr = np.asarray(x, dtype=float)
        return np.zeros_like(arr)

    def set_diffusion_from_function(self, f):
        self._diffusion_fn = f

    def set_advection_from_function(self, f):
        self._advection_fn = f

    def set_reaction_from_function(self, f):
        self._reaction_fn = f

    def set_source_from_function(self, f):
        self._source_fn = f
