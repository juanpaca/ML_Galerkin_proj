import numpy as np


class Mesh1D:
    """Uniform 1D mesh on [x_min, x_max] with n_elements elements."""

    def __init__(self, x_min: float, x_max: float, n_elements: int):
        if n_elements < 1:
            raise ValueError("need at least 1 element")
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.n_elements = n_elements
        self.h = (x_max - x_min) / n_elements
        self.nodes = np.linspace(x_min, x_max, n_elements + 1)
        self.n_nodes = n_elements + 1
        self.elements = [(i, i + 1) for i in range(n_elements)]

    def element_vertices(self, e: int) -> tuple[float, float]:
        return self.nodes[self.elements[e][0]], self.nodes[self.elements[e][1]]

    def element_dofs(self, e: int, degree: int = 1) -> list[int]:
        if degree == 1:
            return [self.elements[e][0], self.elements[e][1]]
        raise NotImplementedError("only degree 1 supported")

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        for e in range(self.n_elements):
            xl, xr = self.element_vertices(e)
            ax.plot([xl, xr], [0, 0], 'k.-', markersize=8)
        return ax
