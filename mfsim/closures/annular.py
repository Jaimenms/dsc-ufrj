"""Fechamento para padrão anular (modelo_anular.m)."""

from __future__ import annotations

import numpy as np

from ..friction import fanning, vabs
from .common import ClosureInputs, ClosureTerms
from .stratified import interfacial_factor_stratified


def annular(inp: ClosureInputs) -> ClosureTerms:
    aG, aL = inp.alphaG, inp.alphaL
    D = inp.D
    rhoL, rhoG = inp.rhoL, inp.rhoG
    n = inp.size

    hL = D / 2.0 * (1.0 - np.sqrt(aG))
    beta = np.full(n, 2.0 * np.pi)

    PerG = np.zeros(n)
    PerL = np.pi * D
    PerI = np.pi * D * np.sqrt(aG)

    DG = 4.0 * inp.AreaG / (PerG + PerI)
    DL = 4.0 * inp.AreaL / PerL

    f_L, Re_L = fanning(inp.vL, rhoL, inp.muL, DL, inp.eps)
    f_G, Re_G = fanning(inp.vG, rhoG, inp.muG, DG, inp.eps)
    # Kowalski (1987) para o filme de líquido
    Re_L_safe = np.where(Re_L == 0, 1e-30, Re_L)
    f_L = np.where(
        Re_L == 0, 0.0,
        np.maximum(24.0 / Re_L_safe, 0.263 / (1.0 - aG) * np.sqrt(DL / Re_L_safe / D)),
    )
    f_I = interfacial_factor_stratified(f_G, inp.vGS, hL / D)

    tauG = f_G * rhoG * inp.vG * vabs(inp.vG) / 2.0
    tauL = f_L * rhoL * inp.vL * vabs(inp.vL) / 2.0
    dv = inp.vG - inp.vL
    tauIG = f_I * rhoG * dv * vabs(dv) / 2.0
    tauIL = -tauIG

    PerI = np.maximum(PerI, 1e-8)
    PerL = np.maximum(PerL, 1e-8)
    PerG = np.maximum(PerG, 1e-8)
    GammaL = PerI * tauIL + PerL * tauL
    GammaG = PerI * tauIG + PerG * tauG

    z = np.zeros(n)
    return ClosureTerms(
        GammaL=GammaL, GammaG=GammaG, dPL=z, dPG=z.copy(),
        ddPLdP=z.copy(), ddPGdP=z.copy(), ddPLdalphaG=z.copy(), ddPGdalphaG=z.copy(),
        dPI=z.copy(), hL=hL, beta=beta, tauG=tauG, tauL=tauL,
    )
