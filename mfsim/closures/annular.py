"""Fechamento para padrão anular (modelo_anular.m)."""

from __future__ import annotations

from ..backend import NUMPY
from ..friction import fanning, vabs
from .common import ClosureInputs, ClosureTerms
from .stratified import interfacial_factor_stratified


def annular(inp: ClosureInputs, bk=NUMPY) -> ClosureTerms:
    xp = bk.xp
    aG, aL = inp.alphaG, inp.alphaL
    D = inp.D
    rhoL, rhoG = inp.rhoL, inp.rhoG
    n = inp.size

    hL = D / 2.0 * (1.0 - bk.safe_sqrt(aG))
    beta = xp.full(n, 2.0 * xp.pi)

    PerG = xp.zeros(n)
    PerL = xp.pi * D
    PerI = xp.pi * D * bk.safe_sqrt(aG)

    DG = 4.0 * inp.AreaG / (PerG + PerI)
    DL = 4.0 * inp.AreaL / PerL

    f_L, Re_L = fanning(inp.vL, rhoL, inp.muL, DL, inp.eps, bk=bk)
    f_G, Re_G = fanning(inp.vG, rhoG, inp.muG, DG, inp.eps, bk=bk)
    # Kowalski (1987) para o filme de líquido
    Re_L_safe = xp.where(Re_L == 0, 1e-30, Re_L)
    f_L = xp.where(
        Re_L == 0, 0.0,
        xp.maximum(24.0 / Re_L_safe, 0.263 / (1.0 - aG) * xp.sqrt(DL / Re_L_safe / D)),
    )
    f_I = interfacial_factor_stratified(f_G, inp.vGS, hL / D, bk=bk)

    tauG = f_G * rhoG * inp.vG * vabs(inp.vG, bk) / 2.0
    tauL = f_L * rhoL * inp.vL * vabs(inp.vL, bk) / 2.0
    dv = inp.vG - inp.vL
    tauIG = f_I * rhoG * dv * vabs(dv, bk) / 2.0
    tauIL = -tauIG

    PerI = xp.maximum(PerI, 1e-8)
    PerL = xp.maximum(PerL, 1e-8)
    PerG = xp.maximum(PerG, 1e-8)
    GammaL = PerI * tauIL + PerL * tauL
    GammaG = PerI * tauIG + PerG * tauG

    z = xp.zeros(n)
    return ClosureTerms(
        GammaL=GammaL, GammaG=GammaG, dPL=z, dPG=z,
        ddPLdP=z, ddPGdP=z, ddPLdalphaG=z, ddPGdalphaG=z,
        dPI=z, hL=hL, beta=beta, tauG=tauG, tauL=tauL,
    )
