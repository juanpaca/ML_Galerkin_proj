import numpy as np

from src.mesh import Mesh1D
from src.quadrature import GaussLegendre
from src.pde import AdvectionDiffusion1D
from src.basis import LagrangeBasis1D
from src.rfb_bubble import KANBubble1D, MultiKANBubble1D
from src.rfb_local import reference_p1_basis, local_parameters


def _apply_dirichlet_zero(A: np.ndarray, f: np.ndarray):
    for idx in [0, A.shape[0] - 1]:
        A[idx, :] = 0.0
        A[:, idx] = 0.0
        A[idx, idx] = 1.0
        f[idx] = 0.0


def _gauss_legendre_01(n: int) -> np.ndarray:
    """Return n-point Gauss-Legendre nodes on [0, 1]."""
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0)


def _eps_profile(
    xl: float, xr: float, pde: AdvectionDiffusion1D, n_eps: int
) -> tuple[float, np.ndarray | None]:
    """Compute element-averaged diffusion and (optional) eps ratios at sample points.

    Returns
    -------
    eps_avg : float
        Arithmetic mean of eps(x) over the element (sampled at 10 points).
    eps_ratios : ndarray or None
        eps(x_sample) / eps_avg at n_eps Gauss-Legendre nodes, or None if n_eps == 0.
    """
    h = xr - xl
    xi_coarse = np.linspace(0.0, 1.0, 10)
    eps_vals = pde.diffusion(xl + h * xi_coarse)
    eps_avg = float(np.mean(eps_vals))

    if n_eps <= 0:
        return eps_avg, None

    xi_sample = _gauss_legendre_01(n_eps)
    eps_sample = pde.diffusion(xl + h * xi_sample)
    return eps_avg, np.asarray(eps_sample / eps_avg, dtype=float)


def assemble_classical_system(
    mesh: Mesh1D,
    quad: GaussLegendre,
    pde: AdvectionDiffusion1D,
    apply_bc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    N = mesh.n_nodes
    A = np.zeros((N, N), dtype=float)
    f = np.zeros(N, dtype=float)

    xi_ref = 0.5 * (quad.ref_points + 1.0)
    w_ref = 0.5 * quad.ref_weights
    phi_ref, dphi_dxi = reference_p1_basis(xi_ref)

    for e in range(mesh.n_elements):
        xl, xr = mesh.element_vertices(e)
        h = xr - xl
        x_q = xl + h * xi_ref
        w_q = h * w_ref
        eps = pde.diffusion(x_q)
        beta = pde.advection(x_q)
        sigma = pde.reaction(x_q)
        src = pde.source(x_q)
        dphi_dx = dphi_dxi / h
        dofs = mesh.element_dofs(e)

        A_e = np.zeros((2, 2), dtype=float)
        f_e = np.zeros(2, dtype=float)
        for i in range(2):
            for j in range(2):
                integrand = (
                    eps * dphi_dx[j] * dphi_dx[i]
                    + beta * dphi_dx[j] * phi_ref[i]
                    + sigma * phi_ref[j] * phi_ref[i]
                )
                A_e[i, j] = np.sum(w_q * integrand)
            f_e[i] = np.sum(w_q * src * phi_ref[i])

        for i, gi in enumerate(dofs):
            f[gi] += f_e[i]
            for j, gj in enumerate(dofs):
                A[gi, gj] += A_e[i, j]

    if apply_bc:
        _apply_dirichlet_zero(A, f)
    return A, f


def local_enriched_matrices(
    xl: float,
    xr: float,
    quad: GaussLegendre,
    pde: AdvectionDiffusion1D,
    bubble: KANBubble1D | MultiKANBubble1D,
) -> tuple[np.ndarray, np.ndarray]:
    h = xr - xl
    xi = 0.5 * (quad.ref_points + 1.0)
    w_ref = 0.5 * quad.ref_weights
    x_q = xl + h * xi
    w_q = h * w_ref

    # Coefficient values at quadrature points
    eps_q = pde.diffusion(x_q)
    beta_q = pde.advection(x_q)
    sigma_q = pde.reaction(x_q)
    src_q = pde.source(x_q)

    # Element-averaged diffusion for bubble parameters
    n_eps = getattr(bubble, 'n_eps', 0)
    eps_avg, eps_ratios = _eps_profile(xl, xr, pde, n_eps)
    pe, rho = local_parameters(eps_avg, float(np.mean(beta_q)), float(np.mean(sigma_q)), h)

    # P1 basis and bubble
    phi_ref, dphi_dxi = reference_p1_basis(xi)
    dphi_dx = dphi_dxi / h
    b, db_dxi = bubble.value_grad_numpy(xi, pe, rho, eps_ratios=eps_ratios)
    b = np.asarray(b, dtype=float)
    db_dxi = np.asarray(db_dxi, dtype=float)
    if b.ndim == 1:
        b = b[None, :]
        db_dxi = db_dxi[None, :]
    db_dx = db_dxi / h

    vals = np.vstack([phi_ref, b])
    grads = np.vstack([dphi_dx, db_dx])

    n_local = vals.shape[0]
    A = np.zeros((n_local, n_local), dtype=float)
    F = np.zeros(n_local, dtype=float)
    for i in range(n_local):
        for j in range(n_local):
            integrand = (
                eps_q * grads[j] * grads[i]
                + beta_q * grads[j] * vals[i]
                + sigma_q * vals[j] * vals[i]
            )
            A[i, j] = np.sum(w_q * integrand)
        F[i] = np.sum(w_q * src_q * vals[i])
    return A, F


def assemble_rfb_condensed_system(
    mesh: Mesh1D,
    quad: GaussLegendre,
    pde: AdvectionDiffusion1D,
    bubble: KANBubble1D | MultiKANBubble1D,
    apply_bc: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    N = mesh.n_nodes
    A_global = np.zeros((N, N), dtype=float)
    F_global = np.zeros(N, dtype=float)
    local_data = []

    for e in range(mesh.n_elements):
        xl, xr = mesh.element_vertices(e)
        A_e, F_e = local_enriched_matrices(xl, xr, quad, pde, bubble)
        A_LL = A_e[:2, :2]
        A_Lb = A_e[:2, 2:]
        A_bL = A_e[2:, :2]
        A_bb = A_e[2:, 2:]
        F_L = F_e[:2]
        F_b = F_e[2:]

        inv_A_bb = np.linalg.inv(A_bb)
        A_cond = A_LL - A_Lb @ inv_A_bb @ A_bL
        F_cond = F_L - (A_Lb @ inv_A_bb @ F_b).reshape(2)
        dofs = mesh.element_dofs(e)
        for i, gi in enumerate(dofs):
            F_global[gi] += F_cond[i]
            for j, gj in enumerate(dofs):
                A_global[gi, gj] += A_cond[i, j]

        local_data.append(
            {
                "element": e,
                "A_bL": A_bL,
                "A_bb": A_bb,
                "F_b": F_b,
                "xl": xl,
                "xr": xr,
            }
        )

    if apply_bc:
        _apply_dirichlet_zero(A_global, F_global)
    return A_global, F_global, local_data


def recover_bubble_coefficients(coeffs: np.ndarray, mesh: Mesh1D, local_data: list[dict]) -> np.ndarray:
    n_bubbles = local_data[0]["A_bb"].shape[0]
    ub = np.zeros((mesh.n_elements, n_bubbles), dtype=float)
    for item in local_data:
        e = item["element"]
        dofs = mesh.element_dofs(e)
        U_L = coeffs[dofs]
        val = np.linalg.solve(item["A_bb"], item["F_b"] - item["A_bL"] @ U_L)
        ub[e, :] = val
    if n_bubbles == 1:
        return ub[:, 0]
    return ub


class RFBSolution1D:
    def __init__(
        self,
        nodal_coeffs: np.ndarray,
        bubble_coeffs: np.ndarray | None,
        mesh: Mesh1D,
        bubble: KANBubble1D | MultiKANBubble1D | None = None,
        pde: AdvectionDiffusion1D | None = None,
    ):
        self.nodal_coeffs = np.asarray(nodal_coeffs, dtype=float)
        self.bubble_coeffs = None if bubble_coeffs is None else np.asarray(bubble_coeffs, dtype=float)
        self.mesh = mesh
        self.bubble = bubble
        self.pde = pde
        self.classical_basis = LagrangeBasis1D(mesh)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for i in range(self.mesh.n_nodes):
            out += self.nodal_coeffs[i] * self.classical_basis.eval(x, i)

        if self.bubble is None or self.bubble_coeffs is None or self.pde is None:
            return out

        coeffs = self.bubble_coeffs
        if coeffs.ndim == 1:
            coeffs = coeffs[:, None]
        n_eps = getattr(self.bubble, 'n_eps', 0)

        for e in range(self.mesh.n_elements):
            xl, xr = self.mesh.element_vertices(e)
            mask = (x >= xl) & (x <= xr) if e == self.mesh.n_elements - 1 else (x >= xl) & (x < xr)
            if not np.any(mask):
                continue
            h = xr - xl
            xi = (x[mask] - xl) / h

            eps_avg, eps_ratios = _eps_profile(xl, xr, self.pde, n_eps)
            pe, rho = local_parameters(eps_avg, self.pde.beta, self.pde.sigma, h)
            b, _ = self.bubble.value_grad_numpy(xi, pe, rho, eps_ratios=eps_ratios)
            b = np.asarray(b, dtype=float)
            if b.ndim == 1:
                b = b[None, :]
            out[mask] += coeffs[e, :] @ b
        return out
