"""Fechamento para padrão intermitente (modelo_intermitente.m).

Unidade de slug (comprimento lu) composta por um pistão de líquido aerado
(comprimento ls, modelado como bolhas) e uma bolha de Taylor com filme
estratificado (comprimento lf). Os termos de fechamento são ponderados
pelos comprimentos.
"""

from __future__ import annotations

from ..backend import NUMPY
from ..friction import fanning
from ..geometry import stratified_level
from .bubble import bubble
from .common import ClosureInputs, ClosureTerms, G_CLOSURE
from .stratified import stratified


def slug_frequency(uLS, uGS, rhoL, rhoG, muL, D, sigma, eps, bk=NUMPY):
    """Frequência de slugs (Zabaras 2000 / Gregory-Scott generalizada)."""
    xp = bk.xp
    fL, _ = fanning(uLS, rhoL, muL, D, eps, bk=bk)
    Eo = xp.sqrt(G_CLOSURE * D**2 * (rhoL - rhoG) / sigma)
    Nf = (D**1.5) * xp.sqrt(rhoL * (rhoL - rhoG) * G_CLOSURE) / muL
    n = Nf / (260.0 + 0.85 * Nf)
    uGS_s = xp.where(uGS == 0, 1e-30, uGS)
    uLS_s = xp.where(uLS == 0, 1e-30, uLS)
    fL_s = xp.where(fL == 0, 1e-30, fL)
    base = 0.0323 / 2.0 * uLS_s / uGS_s * xp.sqrt(rhoL / (rhoL - rhoG)) * fL_s ** (-0.5) / Eo**0.2
    return xp.abs(uGS_s) / D * xp.abs(base) ** n


def slug_liquid_fraction(D, eps, theta, sigma, rhoL, rhoG, muL, uLS, uGS, bk=NUMPY):
    """Fração de líquido no pistão (Barnea & Brauner 1985)."""
    xp = bk.xp
    uS = uLS + uGS
    ReS = D * xp.abs(uS) * rhoL / muL
    ReS = xp.where(ReS == 0, 1e-30, ReS)
    A = (2.457 * xp.log(1.0 / (0.27 * eps / D + (7.0 / ReS) ** 0.9))) ** 16
    B = (37530.0 / ReS) ** 16
    fS = 2.0 * ((8.0 / ReS) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    alphas = 0.058 * (
        2.0 * xp.sqrt(0.4 * sigma / G_CLOSURE / (rhoL - rhoG))
        * bk.safe_pow(2.0 * fS / D * xp.abs(uS) ** 3, 2.0 / 5.0) * (rhoL / sigma) ** (3.0 / 5.0) - 0.725
    ) ** 2
    alphas = xp.clip(alphas, 0.0, 1.0 - 0.48)
    return 1.0 - alphas


def translational_velocity(D, theta, uLS, uGS, bk=NUMPY):
    """Velocidade translacional (Bendiksen 1984 com C0 = 1.2)."""
    xp = bk.xp
    gD = xp.sqrt(G_CLOSURE * D)
    ud = 0.35 * gD * xp.sin(theta) + 0.54 * gD * xp.cos(theta)
    return 1.2 * (uLS + uGS) + ud


def slug_length(D, bk=NUMPY):
    """Comprimento do pistão (Nicholson 1978; correlação para D > 38 mm)."""
    xp = bk.xp
    ls = 18.1 * D
    big = D > 0.038
    return xp.where(big, xp.exp(-3.287 + 4.589 * xp.sqrt(xp.log(xp.maximum(D, 0.0381) / 0.0254))), ls)


ALPHA_FILM_MAX = 0.95
VEL_LIMIT_FACTOR = 5.0


def intermittent(inp: ClosureInputs, bk=NUMPY) -> ClosureTerms:
    xp = bk.xp
    aG = inp.alphaG
    # correlações da unidade de slug formuladas para escoamento co-corrente;
    # para refluxo usam-se os módulos (as tensões conservam o sinal via v|v|)
    uLS, uGS = xp.abs(inp.vLS), xp.abs(inp.vGS)
    D = inp.D
    rhoL, rhoG = inp.rhoL, inp.rhoG

    Rs = slug_liquid_fraction(D, inp.eps, inp.theta, inp.sigma, rhoL, rhoG, inp.muL, uLS, uGS, bk=bk)
    alphas = 1.0 - Rs
    ls = slug_length(D, bk=bk)
    wS = slug_frequency(uLS, uGS, rhoL, rhoG, inp.muL, D, inp.sigma, inp.eps, bk=bk)
    ut = translational_velocity(D, inp.theta, uLS, uGS, bk=bk)

    lu = ut / xp.where(wS == 0, 1e-30, wS)
    ls = xp.minimum(ls, lu)
    lf = lu - ls
    lf = xp.maximum(lf, ls * 1e-3)
    lu = lf + ls

    alphaf = lu / lf * aG - ls / lf * alphas
    # salvaguarda: fração de gás do filme limitada (no código original o limite
    # 1-1e-6 produz velocidades de filme ~1e6 m/s e forças de atrito ~1e11 N/m
    # quando a célula unitária é inconsistente com alphaG elevado)
    alphaf = xp.clip(alphaf, 1e-6, ALPHA_FILM_MAX)
    Rf = 1.0 - alphaf

    uL = (uLS + ut * (aG - alphas)) / Rs
    ub = (uGS - ut * (aG - alphas)) / xp.where(alphas == 0, 1e-30, alphas)
    uf = ut - (ut - uL) * Rs / Rf
    uG = ut - (ut - ub) * alphas / alphaf
    # limita velocidades locais a um múltiplo das velocidades superficiais
    vmax = VEL_LIMIT_FACTOR * (xp.abs(ut) + uLS + uGS) + 1e-3
    uL, ub, uf, uG = (xp.clip(v, -vmax, vmax) for v in (uL, ub, uf, uG))
    # recupera o sentido do escoamento de cada fase
    sL = xp.where(inp.vLS < 0, -1.0, 1.0)
    sG = xp.where(inp.vGS < 0, -1.0, 1.0)
    uL, uf = sL * uL, sL * uf
    ub, uG = sG * ub, sG * uG

    Area = xp.pi * D**2 / 4.0
    inp_s = inp.replace(
        alphaG=1.0 - Rs, alphaL=Rs, vL=uL, vLS=uL * Rs, vG=ub, vGS=ub * (1.0 - Rs),
        AreaL=Area * Rs, AreaG=Area * (1.0 - Rs),
    )
    Ss = bubble(inp_s, bk=bk)

    hD, beta = stratified_level(1.0 - Rf, bk=bk)
    inp_f = inp.replace(
        alphaG=1.0 - Rf, alphaL=Rf, vL=uf, vLS=uf * Rf, vG=uG, vGS=uG * (1.0 - Rf),
        AreaL=Area * Rf, AreaG=Area * (1.0 - Rf), beta=beta, hD=hD,
    )
    Sf = stratified(inp_f, bk=bk)

    def mix(a, b):
        return (a * ls + b * lf) / lu

    return ClosureTerms(
        GammaL=mix(Ss.GammaL, Sf.GammaL), GammaG=mix(Ss.GammaG, Sf.GammaG),
        dPL=mix(Ss.dPL, Sf.dPL), dPG=mix(Ss.dPG, Sf.dPG),
        ddPLdP=mix(Ss.ddPLdP, Sf.ddPLdP), ddPGdP=mix(Ss.ddPGdP, Sf.ddPGdP),
        ddPLdalphaG=mix(Ss.ddPLdalphaG, Sf.ddPLdalphaG),
        ddPGdalphaG=mix(Ss.ddPGdalphaG, Sf.ddPGdalphaG),
        dPI=mix(Ss.dPI, Sf.dPI), hL=hD * D, beta=beta,
        tauG=mix(Ss.tauG, Sf.tauG), tauL=mix(Ss.tauL, Sf.tauL),
    )
