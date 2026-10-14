"""Fechamento para padrão bolhas (modelo_bolhas.m)."""

from __future__ import annotations

import numpy as np

from ..friction import fanning, vabs
from .common import ClosureInputs, ClosureTerms, G_CLOSURE


SLIP_MAX = 3.0  # m/s


def bubble(inp: ClosureInputs) -> ClosureTerms:
    aG, aL = inp.alphaG, inp.alphaL
    D = inp.D
    rhoL, rhoG = inp.rhoL, inp.rhoG
    n = inp.size

    beta = np.full(n, 2.0 * np.pi)
    hL = D.copy() if isinstance(D, np.ndarray) else np.full(n, D)

    vM = inp.vGS + inp.vLS
    f_L, _ = fanning(inp.vL, rhoL, inp.muL, D, inp.eps)
    f_M, _ = fanning(vM, rhoL, inp.muL, D, inp.eps)
    f_G = 0.0

    # diâmetro máximo de bolha (Hinze / Calderbank) e diâmetro de equilíbrio
    ratio = np.abs(inp.vGS / np.where(vM == 0, 1e-30, vM))
    dbmax = (0.725 + 4.15 * np.sqrt(ratio)) * (inp.sigma / rhoL) ** (3.0 / 5.0) * \
        np.abs(2.0 * f_M * vM**3 / D) ** (-2.0 / 5.0)
    db = np.sqrt(16.0 * inp.sigma / G_CLOSURE / (rhoL - rhoG))
    db = np.minimum(db, dbmax)
    # velocidade terminal: tese (modelo_bolhas.m, sem g e sigma/rhoL) ou Mendelson / Fan & Tsuchiya
    phys = np.zeros(n) if inp.bubble_drag is None else np.asarray(inp.bubble_drag, float) * np.ones(n)
    vb_tese = np.sqrt(0.51 * db + 2.14 / db)
    vb_fis = np.sqrt(2.14 * inp.sigma / (rhoL * db) + 0.505 * G_CLOSURE * db)
    vb = np.where(phys > 0, vb_fis, vb_tese)

    PerL = np.pi * D * aL
    PerG = np.pi * D * aG
    PerI = np.pi * D**2 * aG / db

    # Drew & Passman (1999); escorregamento limitado (salvaguarda para fase gás
    # quase extinta, em que vG = qG/(rhoG alphaG A) cresce sem limite)
    slip = np.clip(inp.vG - inp.vL, -SLIP_MAX, SLIP_MAX)
    dPL = 0.25 * rhoL * slip ** 2
    dPG = np.zeros(n)
    ddPLdP = dPL / rhoL * inp.CL
    z = np.zeros(n)

    CDb = 4.0 / 3.0 * ((rhoL - rhoG) / rhoL) * G_CLOSURE * db / vb**2
    # tese: f_I = 1.5 alphaG rhoL CD/db com rhoG na tensão (força ~ rhoG alphaG^2);
    # físico: PerI tauIG = A (3/4) CD alphaG rhoL |vr| vr / db  <=>  f_I = 3/8 CD (adimensional) com rhoL na tensão
    f_I = np.where(phys > 0, 0.375 * CDb, 1.5 * aG * rhoL * CDb / db)
    rho_I = np.where(phys > 0, rhoL, rhoG)

    tauG = f_G * rhoG * inp.vG * vabs(inp.vG) / 2.0
    tauL = f_L * rhoL * inp.vL * vabs(inp.vL) / 2.0
    dv = inp.vG - inp.vL
    tauIG = f_I * rho_I * dv * vabs(dv) / 2.0
    tauIL = -tauIG

    PerI = np.maximum(PerI, 1e-9)
    PerL = np.maximum(PerL, 1e-9)
    PerG = np.maximum(PerG, 1e-9)
    GammaL = PerI * tauIL + PerL * tauL
    GammaG = PerI * tauIG + PerG * tauG

    return ClosureTerms(
        GammaL=GammaL, GammaG=GammaG, dPL=dPL, dPG=dPG,
        ddPLdP=ddPLdP, ddPGdP=z.copy(), ddPLdalphaG=z.copy(), ddPGdalphaG=z.copy(),
        dPI=z.copy(), hL=hL, beta=beta, tauG=np.broadcast_to(tauG, (n,)).copy(), tauL=tauL,
    )
