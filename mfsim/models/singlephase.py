"""Modelos monofásicos (modelo_duto_L.m e modelo_duto_G.m).

Variáveis de estado: PT (slot 1) e q (slot 2 para líquido, 3 para gás).

  dPT/dt = -F1 dq/dx
  dq/dt  = -(F2 dPT/dx + F3 dq/dx) - F4
"""

from __future__ import annotations

import numpy as np

from ..friction import fanning, vabs
from .base import G, PipeModel, SLOT_P, SLOT_QG, SLOT_QL


class _SinglePhase(PipeModel):
    A_pattern = np.array([[0, 1],
                          [2, 3]])
    B_pattern = np.array([0, 4])
    nF = 4
    phase = "L"

    def _props(self, P):
        rhoL, rhoG, CL, CG = self.fluid.props(P)
        if self.phase == "L":
            return rhoL, CL, self.fluid.muL
        return rhoG, CG, self.fluid.muG

    def coefficients(self, y, theta, outputs=False):
        xp, bk = self.xp, self.bk
        p = self.params
        y = xp.asarray(y, dtype=float).reshape(-1, 4)
        PT = y[:, SLOT_P]
        q = y[:, self.slots[1]]
        n = PT.shape[0]
        th = xp.asarray(theta, dtype=float) * xp.ones(n)
        P = p.PR * PT
        rho, C, mu = self._props(P)
        Area = xp.pi * p.D**2 / 4.0
        v = q / (rho * Area)
        Per = xp.pi * p.D
        f, Re = fanning(v, rho, mu, p.D, p.eps, bk=bk)
        tau = f * rho * v * vabs(v, bk) / 2.0
        Gamma = Per * tau
        F = xp.stack([
            (1.0 / Area / C / p.PR),
            Area * (1.0 - C * v * xp.abs(v)) * p.PR,
            2.0 * xp.abs(v),
            Gamma + G * Area * rho * xp.sin(th),
        ])
        if not outputs:
            return F
        key = self.phase
        out = {
            "inclinacao": th, "diametro": np.full(n, p.D), "pressao": PT * p.PR / 1e5,
            f"vazaoMassa{key}": q, f"vazaoVol{key}": q / rho, f"velocidade{key}": v,
            f"velSom{key}": 1 / np.sqrt(C), f"carga{key}": F[3] / Area * 100 / 1e5,
            f"densidade{key}": rho, f"viscosidade{key}": np.full(n, mu),
            "alfaL": np.full(n, 1.0 if key == "L" else 0.0), "alfaG": np.full(n, 0.0 if key == "L" else 1.0),
            f"Gamma{key}": Gamma, f"tau{key}": tau,
        }
        return F, out

    def conservative_terms(self, y, theta):
        xp, bk = self.xp, self.bk
        p = self.params
        y = xp.asarray(y, dtype=float).reshape(-1, 4)
        P = p.PR * y[:, SLOT_P]
        rho, C, mu = self._props(P)
        Area = xp.pi * p.D**2 / 4.0
        q = y[:, self.slots[1]]
        v = q / (rho * Area)
        f, _ = fanning(v, rho, mu, p.D, p.eps, bk=bk)
        Gamma = xp.pi * p.D * f * rho * v * vabs(v, bk) / 2.0
        return dict(rho=rho, C=C, v=v, Gamma=Gamma, Area=Area, P=P)


class SinglePhaseLiquid(_SinglePhase):
    name = "liquid"
    slots = (SLOT_P, SLOT_QL)
    phase = "L"


class SinglePhaseGas(_SinglePhase):
    name = "gas"
    slots = (SLOT_P, SLOT_QG)
    phase = "G"
