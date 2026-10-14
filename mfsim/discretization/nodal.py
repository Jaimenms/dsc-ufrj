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

    def __init__(self, model, mesh, settings=None):
        super().__init__(model, mesh, settings)
        xp = self.xp
        self._slots = list(self.model.slots)
        self._h_x = xp.asarray(self.h)
        self._theta_x = xp.asarray(self.theta_nodes)
        self._limit = 0 in self._slots
        self._alpha_band = float(self.settings.get("alpha_band", 0.02))

    @property
    def n_nodes(self):
        return self.N + 1

    @property
    def x_nodes(self):
        return np.asarray(self.mesh["s"], float)

    @property
    def theta_nodes(self):
        return node_angles(self.theta_elem)

    def _slot_values(self, Y):
        return Y[:, self._slots]

    def _expand(self, dys, Y=None):
        xp, bk = self.xp, self.bk
        dY = bk.set(xp.zeros((self.n_nodes, 4)), (slice(None), self._slots), dys)
        if Y is not None and self._limit:
            # limitador suave da fração de área junto aos limites físicos
            p = self.model.params
            band = self._alpha_band
            aG = Y[:, 0]
            r_lo = xp.clip((aG - p.alpha_min) / band, 0.0, 1.0)
            r_hi = xp.clip((p.alpha_max - aG) / band, 0.0, 1.0)
            d = dY[:, 0]
            dY = bk.set(dY, (slice(None), 0), xp.where(d < 0, d * r_lo, d * r_hi))
        return dY


class FDCentralScheme(NodalScheme):
    """Diferenças finitas centrais nos nós interiores, laterais nas extremidades."""

    name = "fd_central"

    def rhs(self, Y):
        xp = self.xp
        Y = self.bk.asarray(Y)
        A, B = self.model.quasi_linear(Y, self._theta_x)
        ys = self._slot_values(Y)
        h = self._h_x
        first = (ys[1] - ys[0]) / h[0]
        last = (ys[-1] - ys[-2]) / h[-1]
        mid = (ys[2:] - ys[:-2]) / (h[:-1] + h[1:])[:, None]
        dydx = xp.concatenate([first[None, :], mid, last[None, :]], axis=0)
        dys = -(xp.einsum("nij,nj->ni", A, dydx) + B)
        return self._expand(dys, Y)


class FDUpwindScheme(NodalScheme):
    """Diferenças regressivas (upwind para escoamento no sentido da aresta)."""

    name = "fd_upwind"

    def rhs(self, Y):
        xp = self.xp
        Y = self.bk.asarray(Y)
        A, B = self.model.quasi_linear(Y, self._theta_x)
        ys = self._slot_values(Y)
        h = self._h_x
        first = (ys[1] - ys[0]) / h[0]
        rest = (ys[1:] - ys[:-1]) / h[:, None]
        dydx = xp.concatenate([first[None, :], rest], axis=0)
        dys = -(xp.einsum("nij,nj->ni", A, dydx) + B)
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
        self._thq = np.repeat(self.theta_elem, self.xi.size)
        self._to_backend()

    def _to_backend(self):
        xp = self.xp
        self._xi_x = xp.asarray(self.xi)
        self._wq_x = xp.asarray(self.wq)
        self._phi_l_x = xp.asarray(self._phi_left)
        self._phi_r_x = xp.asarray(self._phi_right)
        self._Minv_x = xp.asarray(self.Minv)
        self._thq_x = xp.asarray(self._thq)

    def _quad_points(self, Y):
        """Estados (N*nq x 4) e ângulos nos pontos de quadratura."""
        yl = Y[:-1][:, None, :]
        yr = Y[1:][:, None, :]
        Yq = yl + (yr - yl) * self._xi_x[None, :, None]
        return Yq.reshape(-1, 4), self._thq_x

    def residual(self, Y):
        xp, bk = self.xp, self.bk
        Y = bk.asarray(Y)
        nq = self.xi.size
        N = self.N
        Yq, thq = self._quad_points(Y)
        A, B = self.model.quasi_linear(Yq, thq)
        nv = self.model.nv
        A = A.reshape(N, nq, nv, nv)
        B = B.reshape(N, nq, nv)
        ys = self._slot_values(Y)
        dy = ys[1:] - ys[:-1]                                   # (N, nv)
        Ady = xp.einsum("eqij,ej->eqi", A, dy)                  # (N, nq, nv)
        hB = self._h_x[:, None, None] * B
        integrand = Ady + hB                                    # (N, nq, nv)
        R_left = xp.einsum("q,eqi->ei", self._wq_x * self._phi_l_x, integrand)
        R_right = xp.einsum("q,eqi->ei", self._wq_x * self._phi_r_x, integrand)
        R = bk.add(bk.add(xp.zeros((N + 1, nv)), slice(None, -1), R_left), slice(1, None), R_right)
        return R

    def rhs(self, Y):
        Y = self.bk.asarray(Y)
        R = self.residual(Y)
        dys = -(self._Minv_x @ R)
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
        self._to_backend()

    def residual(self, Y):
        xp, bk = self.xp, self.bk
        Y = bk.asarray(Y)
        nq = self.xi.size
        N = self.N
        Yq, thq = self._quad_points(Y)
        A, B = self.model.quasi_linear(Yq, thq)
        nv = self.model.nv
        A = A.reshape(N, nq, nv, nv)
        B = B.reshape(N, nq, nv)
        ys = self._slot_values(Y)
        dy = ys[1:] - ys[:-1]
        a2 = self.alpha / 2.0
        wq, phi_l, phi_r, h = self._wq_x, self._phi_l_x, self._phi_r_x, self._h_x
        # pesos modificados dependem do sinal de cada coeficiente
        sA = xp.sign(A)
        wl = wq[None, :, None, None] * (phi_l[None, :, None, None] - a2 * sA)
        wr = wq[None, :, None, None] * (phi_r[None, :, None, None] + a2 * sA)
        Al = xp.einsum("eqij,eqij->eij", wl, A)
        Ar = xp.einsum("eqij,eqij->eij", wr, A)
        sB = xp.sign(B)
        wlB = wq[None, :, None] * (phi_l[None, :, None] - a2 * sB)
        wrB = wq[None, :, None] * (phi_r[None, :, None] + a2 * sB)
        Bl = xp.einsum("eqi,eqi->ei", wlB, B) * h[:, None]
        Br = xp.einsum("eqi,eqi->ei", wrB, B) * h[:, None]
        R_left = xp.einsum("eij,ej->ei", Al, dy) + Bl
        R_right = xp.einsum("eij,ej->ei", Ar, dy) + Br
        R = bk.add(bk.add(xp.zeros((N + 1, nv)), slice(None, -1), R_left), slice(1, None), R_right)
        return R
