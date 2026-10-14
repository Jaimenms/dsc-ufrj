"""Esquemas nodais da tese: Galerkin, Petrov-Galerkin, diferenças centrais e
upwind (solver_edp_galerkin.m, solver_edp_petrovgalerkin.m,
solver_edp_diferencas.m, solver_edp_diferencas_upwind.m).

Todos partem da forma quase-linear  dy/dt + A(y) dy/dx + B(y) = 0  nos
slots do modelo e produzem  M dy/dt = -R(y).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..geometry import node_angles
from .base import PipeScheme


class NodalScheme(PipeScheme):
    """Base para esquemas com variáveis nos N+1 nós da malha."""

    @property
    def n_nodes(self):
        return self.N + 1

    @property
    def x_nodes(self):
        return np.asarray(self.mesh["s"], float)

    @property
    def theta_nodes(self):
        return node_angles(self.theta_elem)

    def _slots(self, Y):
        return np.asarray(Y, float)[:, list(self.model.slots)]

    def _expand(self, dys, Y=None):
        dY = np.zeros((self.n_nodes, 4))
        dY[:, list(self.model.slots)] = dys
        if Y is not None and 0 in self.model.slots:
            # limitador suave da fração de área junto aos limites físicos
            p = self.model.params
            band = float(self.settings.get("alpha_band", 0.02))
            aG = np.asarray(Y, float)[:, 0]
            r_lo = np.clip((aG - p.alpha_min) / band, 0.0, 1.0)
            r_hi = np.clip((p.alpha_max - aG) / band, 0.0, 1.0)
            d = dY[:, 0]
            dY[:, 0] = np.where(d < 0, d * r_lo, d * r_hi)
        return dY


class FDCentralScheme(NodalScheme):
    """Diferenças finitas centrais nos nós interiores, laterais nas extremidades."""

    name = "fd_central"

    def rhs(self, Y):
        A, B = self.model.quasi_linear(Y, self.theta_nodes)
        ys = self._slots(Y)
        h = self.h
        dydx = np.empty_like(ys)
        dydx[0] = (ys[1] - ys[0]) / h[0]
        dydx[-1] = (ys[-1] - ys[-2]) / h[-1]
        dydx[1:-1] = (ys[2:] - ys[:-2]) / (h[:-1] + h[1:])[:, None]
        dys = -(np.einsum("nij,nj->ni", A, dydx) + B)
        return self._expand(dys, Y)


class FDUpwindScheme(NodalScheme):
    """Diferenças regressivas (upwind para escoamento no sentido da aresta)."""

    name = "fd_upwind"

    def rhs(self, Y):
        A, B = self.model.quasi_linear(Y, self.theta_nodes)
        ys = self._slots(Y)
        h = self.h
        dydx = np.empty_like(ys)
        dydx[0] = (ys[1] - ys[0]) / h[0]
        dydx[1:] = (ys[1:] - ys[:-1]) / h[:, None]
        dys = -(np.einsum("nij,nj->ni", A, dydx) + B)
        return self._expand(dys, Y)


def _mass_matrix(h):
    """Matriz de massa consistente de elementos lineares (N+1 x N+1)."""
    N = h.size
    M = np.zeros((N + 1, N + 1))
    for e in range(N):
        M[e, e] += 2 * h[e] / 6
        M[e + 1, e + 1] += 2 * h[e] / 6
        M[e, e + 1] += h[e] / 6
        M[e + 1, e] += h[e] / 6
    return M


class GalerkinScheme(NodalScheme):
    """Galerkin em elementos finitos lineares com quadratura de Gauss.

    R_i = sum_e sum_q 0.5 w_q phi_i(xi_q) [A(y_q) (y_{e+1} - y_e) + h_e B(y_q)]
    M dy/dt = -R
    """

    name = "galerkin"

    def __init__(self, model, mesh, settings=None):
        super().__init__(model, mesh, settings)
        nq = int(self.settings.get("n_quad", 3))
        a, w = leggauss(nq)
        self.xi = 0.5 + 0.5 * a      # coordenada local em [0, 1]
        self.wq = 0.5 * w            # pesos normalizados (soma 1)
        self.M = _mass_matrix(self.h)
        self.Minv = np.linalg.inv(self.M)
        self._phi_right = self.xi          # phi do nó direito
        self._phi_left = 1.0 - self.xi     # phi do nó esquerdo

    def _quad_points(self, Y):
        """Estados (N*nq x 4) e ângulos nos pontos de quadratura."""
        Y = np.asarray(Y, float)
        yl = Y[:-1][:, None, :]
        yr = Y[1:][:, None, :]
        Yq = yl + (yr - yl) * self.xi[None, :, None]
        thq = np.repeat(self.theta_elem, self.xi.size)
        return Yq.reshape(-1, 4), thq

    def residual(self, Y):
        nq = self.xi.size
        N = self.N
        Yq, thq = self._quad_points(Y)
        A, B = self.model.quasi_linear(Yq, thq)
        nv = self.model.nv
        A = A.reshape(N, nq, nv, nv)
        B = B.reshape(N, nq, nv)
        ys = self._slots(Y)
        dy = ys[1:] - ys[:-1]                                   # (N, nv)
        Ady = np.einsum("eqij,ej->eqi", A, dy)                  # (N, nq, nv)
        hB = self.h[:, None, None] * B
        integrand = Ady + hB                                    # (N, nq, nv)
        R_left = np.einsum("q,eqi->ei", self.wq * self._phi_left, integrand)
        R_right = np.einsum("q,eqi->ei", self.wq * self._phi_right, integrand)
        R = np.zeros((N + 1, nv))
        R[:-1] += R_left
        R[1:] += R_right
        return R

    def rhs(self, Y):
        R = self.residual(Y)
        dys = -(self.Minv @ R)
        return self._expand(dys, Y)


class PetrovGalerkinScheme(GalerkinScheme):
    """Petrov-Galerkin (ponderação assimétrica com fator alfa, como na tese)."""

    name = "petrov_galerkin"

    def __init__(self, model, mesh, settings=None):
        super().__init__(model, mesh, settings)
        self.alpha = float(self.settings.get("pg_alpha", 0.1))
        Mp = np.zeros_like(self.M)
        for e in range(self.N):
            he = self.h[e]
            Mp[e, e] += -he / 4
            Mp[e + 1, e] += he / 4
            Mp[e, e + 1] += -he / 4
            Mp[e + 1, e + 1] += he / 4
        self.Minv = np.linalg.inv(self.M + self.alpha * Mp)

    def residual(self, Y):
        nq = self.xi.size
        N = self.N
        Yq, thq = self._quad_points(Y)
        A, B = self.model.quasi_linear(Yq, thq)
        nv = self.model.nv
        A = A.reshape(N, nq, nv, nv)
        B = B.reshape(N, nq, nv)
        ys = self._slots(Y)
        dy = ys[1:] - ys[:-1]
        a2 = self.alpha / 2.0
        # pesos modificados dependem do sinal de cada coeficiente
        wl = self.wq[None, :, None, None] * (self._phi_left[None, :, None, None] - a2 * np.sign(A))
        wr = self.wq[None, :, None, None] * (self._phi_right[None, :, None, None] + a2 * np.sign(A))
        Al = np.einsum("eqij,eqij->eij", wl, A)
        Ar = np.einsum("eqij,eqij->eij", wr, A)
        wlB = self.wq[None, :, None] * (self._phi_left[None, :, None] - a2 * np.sign(B))
        wrB = self.wq[None, :, None] * (self._phi_right[None, :, None] + a2 * np.sign(B))
        Bl = np.einsum("eqi,eqi->ei", wlB, B) * self.h[:, None]
        Br = np.einsum("eqi,eqi->ei", wrB, B) * self.h[:, None]
        R = np.zeros((N + 1, nv))
        R[:-1] += np.einsum("eij,ej->ei", Al, dy) + Bl
        R[1:] += np.einsum("eij,ej->ei", Ar, dy) + Br
        return R
