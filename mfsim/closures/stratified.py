"""Fechamento para padrão estratificado (modelo_estratificado.m)."""

from __future__ import annotations

import numpy as np

from ..friction import fanning, vabs
from .common import ClosureInputs, ClosureTerms, G_CLOSURE


def interfacial_factor_stratified(f_G, vGS, hD, v_GT=5.5):
    """Andritsos & Hanratty (1987): f_I = f_G (1 + 15 max(0, |vGS|/vGT - 1) sqrt(h/D))."""
    return f_G * (1.0 + np.maximum(0.0, 15.0 * (np.abs(vGS) / v_GT - 1.0) * np.sqrt(hD)))


def stratified(inp: ClosureInputs) -> ClosureTerms:
    aG, aL = inp.alphaG, inp.alphaL
    D, beta, hD = inp.D, inp.beta, inp.hD
    rhoL, rhoG = inp.rhoL, inp.rhoG
    cos_t = np.cos(inp.theta)

    # perímetros
    PerL = beta / 2.0 * D
    PerG = np.pi * D - PerL
    PerI = D * np.sin(beta / 2.0)

    # diâmetros hidráulicos
    DG = 4.0 * inp.AreaG / (PerG + PerI)
    DL = np.where(aG < 1, 4.0 * inp.AreaL / np.maximum(PerL, 1e-30), 0.0)

    # centros de massa das fases (medidos do fundo do tubo)
    sb = np.sin(beta / 2.0)
    zcml = D * (0.5 - 1.0 / 3.0 / np.pi / aL * sb**3)
    zcmg = D * (0.5 + 1.0 / 3.0 / np.pi / aG * sb**3)

    hL = hD * D
    dPLG = ((rhoL - rhoG) * hL - rhoL * zcml + rhoG * zcmg) * G_CLOSURE * cos_t

    dPL = aG * dPLG
    dPG = -aL * dPLG
    dPI = (aL * rhoL * (zcml - hL) + aG * rhoG * (zcmg - hL)) * G_CLOSURE * cos_t

    # derivadas em relação à pressão
    ddPLGdP = ((inp.CL - inp.CG) * hL - inp.CL * zcml + inp.CG * zcmg) * G_CLOSURE * cos_t
    ddPLdP = aG * ddPLGdP
    ddPGdP = -aL * ddPLGdP

    # derivadas em relação à fração de área de gás
    dbetadaG = -2.0 * np.pi / (1.0 - np.cos(beta))
    dhLdaG = 0.5 * sb * dbetadaG
    cb = np.cos(beta / 2.0)
    dzcmldaG = -D / (2.0 * np.pi * aL) * sb**2 * (2.0 / 3.0 / aL * sb + cb * dbetadaG)
    dzcmgdaG = -D / (2.0 * np.pi * aG) * sb**2 * (2.0 / 3.0 / aG * sb - cb * dbetadaG)
    ddPLGdaG = ((rhoL - rhoG) * dhLdaG - rhoL * dzcmldaG + rhoG * dzcmgdaG) * G_CLOSURE * cos_t
    ddPLdaG = aG * ddPLGdaG + dPLG
    ddPGdaG = -aL * ddPLGdaG + dPLG

    # atrito fluido-parede
    f_L, Re_L = fanning(inp.vL, rhoL, inp.muL, DL, inp.eps)
    f_G, Re_G = fanning(inp.vG, rhoG, inp.muG, DG, inp.eps)
    # atrito interfacial (Andritsos & Hanratty com v_GT; v_GT = inf -> Taitel & Dukler, f_I = f_G)
    v_gt = 5.5 if inp.fi_vgt is None else inp.fi_vgt
    f_I = interfacial_factor_stratified(f_G, inp.vGS, hD, v_GT=v_gt)
    if inp.fi_mult is not None:
        f_I = f_I * inp.fi_mult

    tauG = f_G * rhoG * inp.vG * vabs(inp.vG) / 2.0
    tauL = f_L * rhoL * inp.vL * vabs(inp.vL) / 2.0
    dv = inp.vG - inp.vL
    tauIG = f_I * rhoG * dv * vabs(dv) / 2.0
    tauIL = -tauIG

    GammaL = PerI * tauIL + PerL * tauL
    GammaG = PerI * tauIG + PerG * tauG

    return ClosureTerms(
        GammaL=GammaL, GammaG=GammaG, dPL=dPL, dPG=dPG,
        ddPLdP=ddPLdP, ddPGdP=ddPGdP, ddPLdalphaG=ddPLdaG, ddPGdalphaG=ddPGdaG,
        dPI=dPI, hL=hL, beta=beta, tauG=tauG, tauL=tauL,
    )
