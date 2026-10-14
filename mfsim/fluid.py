"""Propriedades dos fluidos (equação de estado politrópica)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fluid:
    """Par líquido/gás com equação de estado politrópica.

    rho_N = rho_N0 (P/P0)^(1/gamma_N);  C_N = d rho_N/dP = rho_N/(gamma_N P)

    Atributos
    ---------
    rhoL0, rhoG0 : densidades de referência [kg/m3] na pressão P0 [Pa]
    muL, muG     : viscosidades dinâmicas [Pa.s]
    gammaL, gammaG : expoentes politrópicos (gammaL=1000 ~ líquido quase incompressível)
    P0           : pressão de referência [Pa]
    sigma        : tensão interfacial [N/m]
    """

    rhoL0: float = 860.0
    rhoG0: float = 80.0
    muL: float = 0.008
    muG: float = 2e-5
    gammaL: float = 1000.0
    gammaG: float = 1.2
    P0: float = 100e5
    sigma: float = 0.03

    def rhoL(self, P):
        return self.rhoL0 * (P / self.P0) ** (1.0 / self.gammaL)

    def rhoG(self, P):
        return self.rhoG0 * (P / self.P0) ** (1.0 / self.gammaG)

    def props(self, P):
        """Retorna (rhoL, rhoG, CL, CG) para a pressão absoluta P [Pa] (numpy ou JAX)."""
        rhoL = self.rhoL(P)
        rhoG = self.rhoG(P)
        CL = rhoL / (P * self.gammaL)
        CG = rhoG / (P * self.gammaG)
        return rhoL, rhoG, CL, CG

    def sound_speeds(self, P):
        rhoL, rhoG, CL, CG = self.props(P)
        return 1.0 / np.sqrt(CL), 1.0 / np.sqrt(CG)

    @classmethod
    def from_legacy(cls, d):
        """Constrói a partir do struct `dados.fluido` do MFSIM/MATLAB."""
        return cls(
            rhoL0=float(d["densidadeL0"]),
            rhoG0=float(d["densidadeG0"]),
            muL=float(d["viscosidadeL"]),
            muG=float(d["viscosidadeG"]),
            gammaL=float(d["gamaL"]),
            gammaG=float(d["gamaG"]),
            P0=float(d["pressao0"]),
            sigma=float(d["tensao"]),
        )
