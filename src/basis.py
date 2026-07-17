import numpy as np
from abc import ABC, abstractmethod
from src.mesh import Mesh1D


class Basis1D(ABC):
    """Abstract interface for a 1D finite element basis."""

    @abstractmethod
    def num_dofs(self) -> int:
        ...

    @abstractmethod
    def eval(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate φ_i at points x. Returns shape (len(x),)."""
        ...

    @abstractmethod
    def grad(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate φ_i' at points x. Returns shape (len(x),)."""
        ...


class LagrangeBasis1D(Basis1D):
    """Classical 1D Lagrange P1 basis (hat functions).

    Node i is at mesh.nodes[i]; φ_i is supported on [x_{i-1}, x_{i+1}].
    """

    def __init__(self, mesh: Mesh1D):
        self.mesh = mesh
        self._n = mesh.n_nodes
        self._h = mesh.h

    def num_dofs(self) -> int:
        return self._n

    def eval(self, x: np.ndarray, i: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        h = self._h
        nodes = self.mesh.nodes

        if i == 0:
            mask = (x >= nodes[0]) & (x <= nodes[1])
            out[mask] = (nodes[1] - x[mask]) / h
        elif i == self._n - 1:
            mask = (x >= nodes[-2]) & (x <= nodes[-1])
            out[mask] = (x[mask] - nodes[-2]) / h
        else:
            xl, xc, xr = nodes[i - 1], nodes[i], nodes[i + 1]
            left = (x >= xl) & (x <= xc)
            out[left] = (x[left] - xl) / h
            right = (x > xc) & (x <= xr)
            out[right] = (xr - x[right]) / h

        return out

    def grad(self, x: np.ndarray, i: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        h = self._h
        nodes = self.mesh.nodes

        if i == 0:
            mask = (x >= nodes[0]) & (x <= nodes[1])
            out[mask] = -1.0 / h
        elif i == self._n - 1:
            mask = (x >= nodes[-2]) & (x <= nodes[-1])
            out[mask] = 1.0 / h
        else:
            xl, xc, xr = nodes[i - 1], nodes[i], nodes[i + 1]
            left = (x >= xl) & (x <= xc)
            out[left] = 1.0 / h
            right = (x > xc) & (x <= xr)
            out[right] = -1.0 / h

        return out
