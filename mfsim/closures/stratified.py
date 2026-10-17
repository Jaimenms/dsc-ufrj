"""Fechamento para padrão estratificado (modelo_estratificado.m)."""

from __future__ import annotations

from ..backend import NUMPY
from ..friction import fanning, vabs
from .common import ClosureInputs, ClosureTerms, G_CLOSURE, smooth_abs, smooth_max


def interfacial_factor_stratified(f_G, vGS, hD, v_GT=5.5, smooth=None, bk=NUMPY):
    """Andritsos & Hanratty (1987): f_I = f_G (1 + 15 max(0, |vGS|/vGT - 1) sqrt(h/D)). Com `smooth` (issue 052) o
    módulo e o máximo são suaves (largura 0,05 m/s e 0,02 em |vGS|/vGT - 1)."""
    xp = bk.xp
    if smooth is None:
        return f_G * (1.0 + xp.maximum(0.0, 15.0 * (xp.abs(vGS) / v_GT - 1.0) * bk.safe_sqrt(hD)))
    x = smooth_abs(vGS, 0.05, bk) / v_GT - 1.0
    return f_G * (1.0 + 15.0 * smooth_max(x, 0.0, 0.02, bk) * bk.safe_sqrt(hD))


def stratified(inp: ClosureInputs, bk=NUMPY) -> ClosureTerms:
    xp = bk.xp
    aG, aL = inp.alphaG, inp.alphaL
    D, beta, hD = inp.D, inp.beta, inp.hD
    rhoL, rhoG = inp.rhoL, inp.rhoG
    cos_t = xp.cos(inp.theta)

    # perímetros
    PerL = beta / 2.0 * D
    PerG = xp.pi * D - PerL
    PerI = D * xp.sin(beta / 2.0)

    # diâmetros hidráulicos
    DG = 4.0 * inp.AreaG / (PerG + PerI)
    DL = xp.where(aG < 1, 4.0 * inp.AreaL / xp.maximum(PerL, 1e-30), 0.0)

    # centros de massa das fases (medidos do fundo do tubo)
    sb = xp.sin(beta / 2.0)
    zcml = D * (0.5 - 1.0 / 3.0 / xp.pi / aL * sb**3)
    zcmg = D * (0.5 + 1.0 / 3.0 / xp.pi / aG * sb**3)

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
    dbetadaG = -2.0 * xp.pi / (1.0 - xp.cos(beta))
    dhLdaG = 0.5 * sb * dbetadaG
    cb = xp.cos(beta / 2.0)
    dzcmldaG = -D / (2.0 * xp.pi * aL) * sb**2 * (2.0 / 3.0 / aL * sb + cb * dbetadaG)
    dzcmgdaG = -D / (2.0 * xp.pi * aG) * sb**2 * (2.0 / 3.0 / aG * sb - cb * dbetadaG)
    ddPLGdaG = ((rhoL - rhoG) * dhLdaG - rhoL * dzcmldaG + rhoG * dzcmgdaG) * G_CLOSURE * cos_t
    ddPLdaG = aG * ddPLGdaG + dPLG
    ddPGdaG = -aL * ddPLGdaG + dPLG

    # atrito fluido-parede
    f_L, Re_L = fanning(inp.vL, rhoL, inp.muL, DL, inp.eps, bk=bk)
    f_G, Re_G = fanning(inp.vG, rhoG, inp.muG, DG, inp.eps, bk=bk)
    # atrito interfacial (Andritsos & Hanratty com v_GT; v_GT = inf -> Taitel & Dukler, f_I = f_G)
    v_gt = 5.5 if inp.fi_vgt is None else inp.fi_vgt
    f_I = interfacial_factor_stratified(f_G, inp.vGS, hD, v_GT=v_gt, smooth=inp.smooth_kinks, bk=bk)
    if inp.fi_mult is not None:
        f_I = f_I * inp.fi_mult

    tauG = f_G * rhoG * inp.vG * vabs(inp.vG, bk) / 2.0
    tauL = f_L * rhoL * inp.vL * vabs(inp.vL, bk) / 2.0
    dv = inp.vG - inp.vL
    tauIG = f_I * rhoG * dv * vabs(dv, bk) / 2.0
    tauIL = -tauIG

    GammaL = PerI * tauIL + PerL * tauL
    GammaG = PerI * tauIG + PerG * tauG

    return ClosureTerms(
        GammaL=GammaL,
        GammaG=GammaG,
        dPL=dPL,
        dPG=dPG,
        ddPLdP=ddPLdP,
        ddPGdP=ddPGdP,
        ddPLdalphaG=ddPLdaG,
        ddPGdalphaG=ddPGdaG,
        dPI=dPI,
        hL=hL,
        beta=beta,
        tauG=tauG,
        tauL=tauL,
    )
