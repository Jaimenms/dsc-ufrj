"""Fechamento para padrão bolhas (modelo_bolhas.m) e a opção de arrasto físico."""

from __future__ import annotations

from ..backend import NUMPY
from ..friction import fanning, vabs
from .common import ClosureInputs, ClosureTerms, G_CLOSURE


SLIP_MAX = 3.0  # m/s


def bubble(inp: ClosureInputs, bk=NUMPY) -> ClosureTerms:
    xp = bk.xp
    aG, aL = inp.alphaG, inp.alphaL
    D = inp.D
    rhoL, rhoG = inp.rhoL, inp.rhoG
    n = inp.size

    beta = xp.full(n, 2.0 * xp.pi)
    hL = D * xp.ones(n)

    vM = inp.vGS + inp.vLS
    f_L, _ = fanning(inp.vL, rhoL, inp.muL, D, inp.eps, bk=bk)
    f_M, _ = fanning(vM, rhoL, inp.muL, D, inp.eps, bk=bk)
    f_G = 0.0

    # diâmetro máximo de bolha (Hinze / Calderbank) e diâmetro de equilíbrio.
    # Com a mistura parada (vM = 0) a dissipação é nula e dbmax é infinito: o `where`
    # devolve inf sem formar o produto (cujo tangente seria 0 * inf = NaN na
    # diferenciação automática); db = min(db, dbmax) não muda de valor
    ratio = xp.abs(inp.vGS / xp.where(vM == 0, 1e-30, vM))
    arg = xp.abs(2.0 * f_M * vM**3 / D)
    small = arg == 0
    arg_safe = xp.where(small, 1.0, arg)
    dbmax = xp.where(small, xp.inf, (0.725 + 4.15 * bk.safe_sqrt(ratio)) * (inp.sigma / rhoL) ** (3.0 / 5.0)
                     * arg_safe ** (-2.0 / 5.0))
    db = xp.sqrt(16.0 * inp.sigma / G_CLOSURE / (rhoL - rhoG))
    db = xp.minimum(db, dbmax)
    # velocidade terminal: tese (modelo_bolhas.m, sem g e sigma/rhoL) ou Mendelson / Fan & Tsuchiya
    phys = xp.zeros(n) if inp.bubble_drag is None else inp.bubble_drag * xp.ones(n)
    vb_tese = xp.sqrt(0.51 * db + 2.14 / db)
    vb_fis = xp.sqrt(2.14 * inp.sigma / (rhoL * db) + 0.505 * G_CLOSURE * db)
    vb = xp.where(phys > 0, vb_fis, vb_tese)

    PerL = xp.pi * D * aL
    PerG = xp.pi * D * aG
    PerI = xp.pi * D**2 * aG / db

    # Drew & Passman (1999); escorregamento limitado (salvaguarda para fase gás
    # quase extinta, em que vG = qG/(rhoG alphaG A) cresce sem limite)
    slip = xp.clip(inp.vG - inp.vL, -SLIP_MAX, SLIP_MAX)
    dPL = 0.25 * rhoL * slip ** 2
    dPG = xp.zeros(n)
    ddPLdP = dPL / rhoL * inp.CL
    z = xp.zeros(n)

    CDb = 4.0 / 3.0 * ((rhoL - rhoG) / rhoL) * G_CLOSURE * db / vb**2
    # tese: f_I = 1.5 alphaG rhoL CD/db com rhoG na tensão (força ~ rhoG alphaG^2);
    # físico: PerI tauIG = A (3/4) CD alphaG rhoL |vr| vr / db  <=>  f_I = 3/8 CD (adimensional) com rhoL na tensão
    f_I = xp.where(phys > 0, 0.375 * CDb, 1.5 * aG * rhoL * CDb / db)
    rho_I = xp.where(phys > 0, rhoL, rhoG)

    tauG = f_G * rhoG * inp.vG * vabs(inp.vG, bk) / 2.0
    tauL = f_L * rhoL * inp.vL * vabs(inp.vL, bk) / 2.0
    dv = inp.vG - inp.vL
    tauIG = f_I * rho_I * dv * vabs(dv, bk) / 2.0
    tauIL = -tauIG

    PerI = xp.maximum(PerI, 1e-9)
    PerL = xp.maximum(PerL, 1e-9)
    PerG = xp.maximum(PerG, 1e-9)
    GammaL = PerI * tauIL + PerL * tauL
    GammaG = PerI * tauIG + PerG * tauG

    return ClosureTerms(
        GammaL=GammaL, GammaG=GammaG, dPL=dPL, dPG=dPG,
        ddPLdP=ddPLdP, ddPGdP=z, ddPLdalphaG=z, ddPGdalphaG=z,
        dPI=z, hL=hL, beta=beta, tauG=tauG * xp.ones(n), tauL=tauL,
    )
