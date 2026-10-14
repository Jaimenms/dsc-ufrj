"""Física em JAX: fluido, atrito, fechamentos, modelos de duto e elementos de rede.

Versões `jax.numpy` dos módulos numpy correspondentes (fluid.py, friction.py,
geometry.stratified_level, closures/*.py, models/*.py, network/elements.py),
escritas para reproduzir os valores a precisão de máquina e para serem
diferenciáveis (sem ramificações em Python sobre valores traçados; "double
where" onde a derivada do ramo não escolhido seria infinita).

Cada fechamento recebe um dict `inp` com as mesmas chaves de ClosureInputs e
devolve um dict com as chaves de ClosureTerms.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .closures.common import G_CLOSURE
from .closures.intermittent import ALPHA_FILM_MAX, VEL_LIMIT_FACTOR
from .closures.regime import (_HD_TEST, _HD_TEST2, REGIME_ANNULAR, REGIME_AUTO, REGIME_BUBBLE,
                              REGIME_INTERMITTENT, REGIME_STRAT_SMOOTH, REGIME_STRAT_WAVY, V_SOBE, EPS_SMAX)
from .friction import V_EPS
from .geometry import _P_BETA
from .models.base import G

TERM_NAMES = ("GammaL", "GammaG", "dPL", "dPG", "ddPLdP", "ddPGdP", "ddPLdalphaG", "ddPGdalphaG",
              "dPI", "hL", "beta", "tauG", "tauL")
_NAN_TO_ZERO = ("ddPLdalphaG", "ddPGdalphaG", "dPL", "dPG", "dPI", "ddPLdP", "ddPGdP")


# ----------------------------------------------------------------------------
# fluido e atrito
# ----------------------------------------------------------------------------
def fluid_props(fl, P):
    """(rhoL, rhoG, CL, CG) como Fluid.props; fl é o Fluid (parâmetros Python)."""
    rhoL = fl.rhoL0 * (P / fl.P0) ** (1.0 / fl.gammaL)
    rhoG = fl.rhoG0 * (P / fl.P0) ** (1.0 / fl.gammaG)
    return rhoL, rhoG, rhoL / (P * fl.gammaL), rhoG / (P * fl.gammaG)


def vabs(v):
    return jnp.sqrt(v * v + V_EPS**2)


def fanning(v, rho, mu, D, eps):
    """Churchill, como friction.fanning com V_EPS > 0 (nunca há v == 0 exato). Devolve (f, Re)."""
    Dd = jnp.where(D <= 0, 1e-12, D)
    Re = Dd * vabs(v) * rho / mu
    A = (2.457 * jnp.log(1.0 / (0.27 * eps / Dd + (7.0 / Re) ** 0.9))) ** 16
    B = (37530.0 / Re) ** 16
    f = 2.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    return f, Re


def stratified_level(aG, n_iter=12):
    """hD e beta (geometry.stratified_level) com Newton de iterações fixas."""
    aL = 1.0 - aG
    beta = jnp.clip(jnp.polyval(jnp.asarray(_P_BETA), aG), 0.0, 2 * jnp.pi)
    only_liq = aL > 1 - 1e-6
    only_gas = aL < 1e-6
    b0 = jnp.where(only_liq, 2 * jnp.pi * aL, jnp.where(only_gas, 2 * jnp.pi * aG, beta))

    def body(_, bm):
        F0 = jnp.sin(bm) - bm + 2 * jnp.pi * aL
        dF = jnp.cos(bm) - 1.0
        dF = jnp.where(jnp.abs(dF) < 1e-14, -1e-14, dF)
        return bm - F0 / dF

    bm = jax.lax.fori_loop(0, n_iter, body, b0)
    b = jnp.where(only_liq | only_gas, b0, bm)
    return 0.5 * (1.0 - jnp.cos(b / 2.0)), b


def _safe_sqrt(x):
    """sqrt com derivada finita em 0 (double where)."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _safe_pow(x, p):
    """x**p para x >= 0 com derivada finita em 0 quando p < 1 (double where)."""
    pos = x > 0
    return jnp.where(pos, jnp.where(pos, x, 1.0) ** p, 0.0 if p > 0 else jnp.inf)


# ----------------------------------------------------------------------------
# fechamentos
# ----------------------------------------------------------------------------
def _zeros_like_terms(n, **kw):
    z = jnp.zeros(n)
    out = {k: z for k in TERM_NAMES}
    out.update(kw)
    return out


def interfacial_factor_stratified(f_G, vGS, hD, v_GT=5.5):
    return f_G * (1.0 + jnp.maximum(0.0, 15.0 * (jnp.abs(vGS) / v_GT - 1.0) * _safe_sqrt(hD)))


def stratified(inp):
    aG, aL = inp["alphaG"], inp["alphaL"]
    D, beta, hD = inp["D"], inp["beta"], inp["hD"]
    rhoL, rhoG = inp["rhoL"], inp["rhoG"]
    cos_t = jnp.cos(inp["theta"])
    PerL = beta / 2.0 * D
    PerG = jnp.pi * D - PerL
    PerI = D * jnp.sin(beta / 2.0)
    DG = 4.0 * inp["AreaG"] / (PerG + PerI)
    DL = jnp.where(aG < 1, 4.0 * inp["AreaL"] / jnp.maximum(PerL, 1e-30), 0.0)
    sb = jnp.sin(beta / 2.0)
    zcml = D * (0.5 - 1.0 / 3.0 / jnp.pi / aL * sb**3)
    zcmg = D * (0.5 + 1.0 / 3.0 / jnp.pi / aG * sb**3)
    hL = hD * D
    dPLG = ((rhoL - rhoG) * hL - rhoL * zcml + rhoG * zcmg) * G_CLOSURE * cos_t
    dPL = aG * dPLG
    dPG = -aL * dPLG
    dPI = (aL * rhoL * (zcml - hL) + aG * rhoG * (zcmg - hL)) * G_CLOSURE * cos_t
    ddPLGdP = ((inp["CL"] - inp["CG"]) * hL - inp["CL"] * zcml + inp["CG"] * zcmg) * G_CLOSURE * cos_t
    ddPLdP = aG * ddPLGdP
    ddPGdP = -aL * ddPLGdP
    dbetadaG = -2.0 * jnp.pi / (1.0 - jnp.cos(beta))
    dhLdaG = 0.5 * sb * dbetadaG
    cb = jnp.cos(beta / 2.0)
    dzcmldaG = -D / (2.0 * jnp.pi * aL) * sb**2 * (2.0 / 3.0 / aL * sb + cb * dbetadaG)
    dzcmgdaG = -D / (2.0 * jnp.pi * aG) * sb**2 * (2.0 / 3.0 / aG * sb - cb * dbetadaG)
    ddPLGdaG = ((rhoL - rhoG) * dhLdaG - rhoL * dzcmldaG + rhoG * dzcmgdaG) * G_CLOSURE * cos_t
    ddPLdaG = aG * ddPLGdaG + dPLG
    ddPGdaG = -aL * ddPLGdaG + dPLG
    f_L, _ = fanning(inp["vL"], rhoL, inp["muL"], DL, inp["eps"])
    f_G, _ = fanning(inp["vG"], rhoG, inp["muG"], DG, inp["eps"])
    f_I = interfacial_factor_stratified(f_G, inp["vGS"], hD, v_GT=inp.get("fi_vgt", 5.5)) * inp.get("fi_mult", 1.0)
    tauG = f_G * rhoG * inp["vG"] * vabs(inp["vG"]) / 2.0
    tauL = f_L * rhoL * inp["vL"] * vabs(inp["vL"]) / 2.0
    dv = inp["vG"] - inp["vL"]
    tauIG = f_I * rhoG * dv * vabs(dv) / 2.0
    return dict(GammaL=-PerI * tauIG + PerL * tauL, GammaG=PerI * tauIG + PerG * tauG, dPL=dPL, dPG=dPG,
                ddPLdP=ddPLdP, ddPGdP=ddPGdP, ddPLdalphaG=ddPLdaG, ddPGdalphaG=ddPGdaG,
                dPI=dPI, hL=hL, beta=beta, tauG=tauG, tauL=tauL)


def annular(inp):
    aG = inp["alphaG"]
    D = inp["D"]
    rhoL, rhoG = inp["rhoL"], inp["rhoG"]
    n = aG.shape[0]
    sqa = _safe_sqrt(aG)
    hL = D / 2.0 * (1.0 - sqa)
    PerG = jnp.zeros(n)
    PerL = jnp.pi * D
    PerI = jnp.pi * D * sqa
    DG = 4.0 * inp["AreaG"] / (PerG + PerI)
    DL = 4.0 * inp["AreaL"] / PerL
    f_L0, Re_L = fanning(inp["vL"], rhoL, inp["muL"], DL, inp["eps"])
    f_G, _ = fanning(inp["vG"], rhoG, inp["muG"], DG, inp["eps"])
    Re_L_safe = jnp.where(Re_L == 0, 1e-30, Re_L)
    f_L = jnp.where(Re_L == 0, 0.0, jnp.maximum(24.0 / Re_L_safe, 0.263 / (1.0 - aG) * jnp.sqrt(DL / Re_L_safe / D)))
    f_I = interfacial_factor_stratified(f_G, inp["vGS"], hL / D)
    tauG = f_G * rhoG * inp["vG"] * vabs(inp["vG"]) / 2.0
    tauL = f_L * rhoL * inp["vL"] * vabs(inp["vL"]) / 2.0
    dv = inp["vG"] - inp["vL"]
    tauIG = f_I * rhoG * dv * vabs(dv) / 2.0
    PerI = jnp.maximum(PerI, 1e-8)
    PerL = jnp.maximum(PerL, 1e-8)
    PerG = jnp.maximum(PerG, 1e-8)
    return _zeros_like_terms(n, GammaL=-PerI * tauIG + PerL * tauL, GammaG=PerI * tauIG + PerG * tauG,
                             hL=hL, beta=jnp.full(n, 2.0 * jnp.pi), tauG=tauG, tauL=tauL)


def bubble(inp, slip_max=3.0):
    aG, aL = inp["alphaG"], inp["alphaL"]
    D = inp["D"]
    rhoL, rhoG = inp["rhoL"], inp["rhoG"]
    n = aG.shape[0]
    vM = inp["vGS"] + inp["vLS"]
    f_L, _ = fanning(inp["vL"], rhoL, inp["muL"], D, inp["eps"])
    f_M, _ = fanning(vM, rhoL, inp["muL"], D, inp["eps"])
    vM_safe = jnp.where(vM == 0, 1e-30, vM)
    ratio = jnp.abs(inp["vGS"] / vM_safe)
    arg = jnp.abs(2.0 * f_M * vM**3 / D)
    small = arg < 1e-300
    arg_safe = jnp.where(small, 1.0, arg)
    dbmax = jnp.where(small, jnp.inf, (0.725 + 4.15 * _safe_sqrt(ratio)) * (inp["sigma"] / rhoL) ** (3.0 / 5.0)
                      * arg_safe ** (-2.0 / 5.0))
    db = jnp.sqrt(16.0 * inp["sigma"] / G_CLOSURE / (rhoL - rhoG))
    db = jnp.minimum(db, dbmax)
    phys = inp.get("bubble_drag", 0.0) * jnp.ones(n)
    vb = jnp.where(phys > 0, jnp.sqrt(2.14 * inp["sigma"] / (rhoL * db) + 0.505 * G_CLOSURE * db), jnp.sqrt(0.51 * db + 2.14 / db))
    PerL = jnp.maximum(jnp.pi * D * aL, 1e-9)
    PerG = jnp.maximum(jnp.pi * D * aG, 1e-9)
    PerI = jnp.maximum(jnp.pi * D**2 * aG / db, 1e-9)
    slip = jnp.clip(inp["vG"] - inp["vL"], -slip_max, slip_max)
    dPL = 0.25 * rhoL * slip**2
    ddPLdP = dPL / rhoL * inp["CL"]
    CDb = 4.0 / 3.0 * ((rhoL - rhoG) / rhoL) * G_CLOSURE * db / vb**2
    f_I = jnp.where(phys > 0, 0.375 * CDb, 1.5 * aG * rhoL * CDb / db)
    rho_I = jnp.where(phys > 0, rhoL, rhoG)
    tauL = f_L * rhoL * inp["vL"] * vabs(inp["vL"]) / 2.0
    dv = inp["vG"] - inp["vL"]
    tauIG = f_I * rho_I * dv * vabs(dv) / 2.0
    return _zeros_like_terms(n, GammaL=-PerI * tauIG + PerL * tauL, GammaG=PerI * tauIG, dPL=dPL,
                             ddPLdP=ddPLdP, hL=D * jnp.ones(n), beta=jnp.full(n, 2.0 * jnp.pi), tauL=tauL)


def _slug_frequency(uLS, uGS, rhoL, rhoG, muL, D, sigma, eps):
    fL, _ = fanning(uLS, rhoL, muL, D, eps)
    Eo = jnp.sqrt(G_CLOSURE * D**2 * (rhoL - rhoG) / sigma)
    Nf = (D**1.5) * jnp.sqrt(rhoL * (rhoL - rhoG) * G_CLOSURE) / muL
    n = Nf / (260.0 + 0.85 * Nf)
    uGS_s = jnp.where(uGS == 0, 1e-30, uGS)
    uLS_s = jnp.where(uLS == 0, 1e-30, uLS)
    fL_s = jnp.where(fL == 0, 1e-30, fL)
    base = 0.0323 / 2.0 * uLS_s / uGS_s * jnp.sqrt(rhoL / (rhoL - rhoG)) * fL_s ** (-0.5) / Eo**0.2
    return jnp.abs(uGS_s) / D * jnp.abs(base) ** n


def _slug_liquid_fraction(D, eps, sigma, rhoL, rhoG, muL, uLS, uGS):
    uS = uLS + uGS
    ReS = D * jnp.abs(uS) * rhoL / muL
    ReS = jnp.where(ReS == 0, 1e-30, ReS)
    A = (2.457 * jnp.log(1.0 / (0.27 * eps / D + (7.0 / ReS) ** 0.9))) ** 16
    B = (37530.0 / ReS) ** 16
    fS = 2.0 * ((8.0 / ReS) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    alphas = 0.058 * (2.0 * jnp.sqrt(0.4 * sigma / G_CLOSURE / (rhoL - rhoG))
                      * _safe_pow(2.0 * fS / D * jnp.abs(uS) ** 3, 2.0 / 5.0) * (rhoL / sigma) ** (3.0 / 5.0) - 0.725) ** 2
    return 1.0 - jnp.clip(alphas, 0.0, 1.0 - 0.48)


def _slug_length(D):
    ls = 18.1 * D
    big = D > 0.038
    return jnp.where(big, jnp.exp(-3.287 + 4.589 * jnp.sqrt(jnp.log(jnp.maximum(D, 0.0381) / 0.0254))), ls)


def intermittent(inp):
    aG = inp["alphaG"]
    uLS, uGS = jnp.abs(inp["vLS"]), jnp.abs(inp["vGS"])
    D = inp["D"]
    rhoL, rhoG = inp["rhoL"], inp["rhoG"]
    Rs = _slug_liquid_fraction(D, inp["eps"], inp["sigma"], rhoL, rhoG, inp["muL"], uLS, uGS)
    alphas = 1.0 - Rs
    ls = _slug_length(D)
    wS = _slug_frequency(uLS, uGS, rhoL, rhoG, inp["muL"], D, inp["sigma"], inp["eps"])
    gD = jnp.sqrt(G_CLOSURE * D)
    ut = 1.2 * (uLS + uGS) + 0.35 * gD * jnp.sin(inp["theta"]) + 0.54 * gD * jnp.cos(inp["theta"])
    lu = ut / jnp.where(wS == 0, 1e-30, wS)
    ls = jnp.minimum(ls, lu)
    lf = lu - ls
    lf = jnp.maximum(lf, ls * 1e-3)
    lu = lf + ls
    alphaf = lu / lf * aG - ls / lf * alphas
    alphaf = jnp.clip(alphaf, 1e-6, ALPHA_FILM_MAX)
    Rf = 1.0 - alphaf
    uL = (uLS + ut * (aG - alphas)) / Rs
    ub = (uGS - ut * (aG - alphas)) / jnp.where(alphas == 0, 1e-30, alphas)
    uf = ut - (ut - uL) * Rs / Rf
    uG = ut - (ut - ub) * alphas / alphaf
    vmax = VEL_LIMIT_FACTOR * (jnp.abs(ut) + uLS + uGS) + 1e-3
    uL, ub, uf, uG = (jnp.clip(v, -vmax, vmax) for v in (uL, ub, uf, uG))
    sL = jnp.where(inp["vLS"] < 0, -1.0, 1.0)
    sG = jnp.where(inp["vGS"] < 0, -1.0, 1.0)
    uL, uf = sL * uL, sL * uf
    ub, uG = sG * ub, sG * uG
    Area = jnp.pi * D**2 / 4.0
    inp_s = dict(inp, alphaG=1.0 - Rs, alphaL=Rs, vL=uL, vLS=uL * Rs, vG=ub, vGS=ub * (1.0 - Rs),
                 AreaL=Area * Rs, AreaG=Area * (1.0 - Rs))
    Ss = bubble(inp_s)
    hD, beta = stratified_level(1.0 - Rf)
    inp_f = dict(inp, alphaG=1.0 - Rf, alphaL=Rf, vL=uf, vLS=uf * Rf, vG=uG, vGS=uG * (1.0 - Rf),
                 AreaL=Area * Rf, AreaG=Area * (1.0 - Rf), beta=beta, hD=hD)
    Sf = stratified(inp_f)
    out = {k: (Ss[k] * ls + Sf[k] * lf) / lu for k in TERM_NAMES}
    out["hL"] = hD * D
    out["beta"] = beta
    return out


CLOSURE_FUNCS = {"Est": stratified, "Anu": annular, "Int": intermittent, "Bub": bubble}


def fixed_flags(regime, n):
    """RegimeFlags de fixed_regime como dict de arrays."""
    z = jnp.zeros(n)
    o = jnp.ones(n)
    fl = dict(Est=z, EsS=z, EsW=z, Anu=z, Bub=z, Int=z)
    if regime == REGIME_ANNULAR:
        fl["Anu"] = o
    elif regime == REGIME_BUBBLE:
        fl["Bub"] = o
    elif regime == REGIME_INTERMITTENT:
        fl["Int"] = o
    elif regime == REGIME_STRAT_SMOOTH:
        fl["Est"], fl["EsS"] = o, o
    elif regime == REGIME_STRAT_WAVY:
        fl["Est"], fl["EsW"] = o, o
    else:
        raise ValueError(f"regime fixo inválido: {regime}")
    return fl


def _closures_used(regime, *extra_regimes):
    names = set()
    for r in (regime,) + tuple(r for r in extra_regimes if r):
        names.add({REGIME_ANNULAR: "Anu", REGIME_BUBBLE: "Bub", REGIME_INTERMITTENT: "Int",
                   REGIME_STRAT_SMOOTH: "Est", REGIME_STRAT_WAVY: "Est"}[r])
    return names


# ----------------------------------------------------------------------------
# identificação de padrão (regime 6): equilíbrio de Taitel/Pots e critérios da tese
# ----------------------------------------------------------------------------
_G_REG = 9.81          # closures.regime.G


def _table_geometry(hD, geometry, n=0.2, m=0.2):
    """FI e LAMBDA de closures.regime._lockhart_vLS: só dependem de hD (tabelas), logo
    são constantes pré-calculadas em numpy com as mesmas expressões."""
    At = np.pi / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if geometry == "stratified":
            c = 2.0 * hD - 1.0
            root = np.sqrt(np.clip(1.0 - c**2, 0.0, None))
            ALt = (np.pi - np.arccos(c) + c * root) / 4.0
            AGt = (np.arccos(c) - c * root) / 4.0
            SLt = np.pi - np.arccos(c)
            SGt = np.arccos(c)
            SIt = root
        else:
            alpha = (1.0 - 2.0 * hD) ** 2
            ALt = At * (1.0 - alpha)
            AGt = At - ALt
            SLt = np.pi * np.ones_like(hD)
            SGt = np.zeros_like(hD)
            SIt = np.pi * np.sqrt(alpha)
        uLt = At / ALt
        uGt = At / AGt
        DGt = 4.0 * AGt / (SGt + SIt)
        DLt = 4.0 * ALt / SLt
        FI = (SGt / AGt + SIt / ALt + SIt / AGt) * (uGt * DGt) ** (-m) * uGt**2
        LAMBDA = (uLt * DLt) ** (-n) * uLt**2 * SLt / ALt
    return FI, LAMBDA


_FI_EST, _LAM_EST = _table_geometry(_HD_TEST, "stratified")
_FI_ANU, _LAM_ANU = _table_geometry(_HD_TEST2, "annular")


def _equilibrium_vLS(vGS, D, eta, muL, muG, rhoL, rhoG, FI, LAMBDA, CL=0.046, CG=0.046, n=0.2, m=0.2):
    """vLS de equilíbrio (n x n_hD) para uma tabela de hD (closures.regime._lockhart_vLS)."""
    Re_GS = D * jnp.abs(vGS) * rhoG / muG
    Re_GS = jnp.where(Re_GS == 0, 1e-30, Re_GS)
    f_GS = CG * Re_GS ** (-m)
    tau_GS = f_GS * rhoG * vGS * jnp.abs(vGS) / 2.0
    tau_s = jnp.where(tau_GS == 0, 1e-30, tau_GS)
    Y = (rhoL - rhoG) * _G_REG * jnp.sin(eta) / (4.0 / D * tau_s)
    FI, LAMBDA = jnp.asarray(FI), jnp.asarray(LAMBDA)
    if FI.ndim == 1:                       # tabela completa (nh,) ou entradas escolhidas por linha (n, m)
        FI, LAMBDA = FI[None, :], LAMBDA[None, :]
    X2 = (4.0 * Y[:, None] + FI) / LAMBDA
    X2 = jnp.where(X2 < 0, jnp.nan, X2)
    tau_LS = X2 * tau_GS[:, None]
    return ((tau_LS / CL) ** (1.0 / (2.0 - n)) * (D * rhoL / muL)[:, None] ** (n / (2.0 - n))
            * (rhoL / 2.0)[:, None] ** (1.0 / (n - 2.0)))


def _table_lookup(v, hd_table, vLS, after_last_nan):
    """Índice da tabela cujo vLS de equilíbrio é o mais próximo de vLS (a busca de
    equilibrium_fractions: no estratificado só a partir do último NaN da linha) e,
    para a interpolação do modo suave, os índices dos candidatos vizinhos.
    Devolve (idx, prev, next, has_cand, has_prev, has_next)."""
    nan = jnp.isnan(v)
    nh = v.shape[1]
    j = jnp.arange(nh)
    if after_last_nan:
        last = jnp.max(jnp.where(nan, j[None, :], -1), axis=1)
        cand = (j[None, :] > last[:, None]) & ~nan
    else:
        cand = ~nan
    err = jnp.where(cand, jnp.minimum(jnp.abs(v - vLS[:, None]), 1e300), jnp.inf)
    idx = jnp.argmin(err, axis=1)
    nxt = jnp.min(jnp.where(cand & (j[None, :] > idx[:, None]), j[None, :], nh), axis=1)
    prv = jnp.max(jnp.where(cand & (j[None, :] < idx[:, None]), j[None, :], -1), axis=1)
    return idx, jnp.clip(prv, 0, nh - 1), jnp.clip(nxt, 0, nh - 1), jnp.any(cand, axis=1), prv >= 0, nxt < nh


def _interp_pick(sel, vLS, hd_table, FI, LAM, args, smooth):
    """hD escolhido: entrada mais próxima (tese) ou interpolação linear entre os
    candidatos que enquadram vLS (modo suave), com o vLS de equilíbrio reavaliado
    com gradiente nas (até) três entradas envolvidas."""
    idx, prv, nxt, has, has_p, has_n = sel
    hd = jnp.asarray(hd_table)
    if not smooth:
        return jnp.where(has, hd[idx], 1e-6)
    cols = jnp.stack([prv, idx, nxt], axis=1)
    v3 = _equilibrium_vLS(*args, jnp.asarray(FI)[cols], jnp.asarray(LAM)[cols])       # (n, 3) com gradiente
    vp, vj, vn = v3[:, 0], v3[:, 1], v3[:, 2]
    hp, hj, hn = hd[prv], hd[idx], hd[nxt]
    use_n = has_n & ((vj - vLS) * (vn - vLS) <= 0)
    use_p = ~use_n & has_p & ((vp - vLS) * (vj - vLS) <= 0)
    va = jnp.where(use_n, vj, vp)
    vb = jnp.where(use_n, vn, vj)
    ha = jnp.where(use_n, hj, hp)
    hb = jnp.where(use_n, hn, hj)
    dv = vb - va
    ok = (use_n | use_p) & (dv != 0)
    dv_s = jnp.where(ok, dv, 1.0)
    h_int = ha + (vLS - va) / dv_s * (hb - ha)
    return jnp.where(has, jnp.where(ok, h_int, hj), 1e-6)


def equilibrium_fractions(inp, smooth=False):
    """closures.regime.equilibrium_fractions: (hD_est, hD_anu, aG_est, aG_anu, aG_bol).

    As buscas em tabela são avaliadas sem gradiente (stop_gradient: constantes por
    partes no estado, e evita propagar tangentes pelas n x 700 avaliações das
    tabelas na Jacobiana); no modo suave o hD interpolado é reavaliado com gradiente
    nas entradas escolhidas (_interp_pick)."""
    sg = jax.lax.stop_gradient
    vLS_s = jnp.abs(inp["vLS"]) if smooth else inp["vLS"]
    vGS_s = jnp.abs(inp["vGS"]) if smooth else inp["vGS"]
    D, th, muL, muG = inp["D"], inp["theta"], inp["muL"], inp["muG"]
    rhoL, rhoG = inp["rhoL"], inp["rhoG"]
    args_sg = (sg(vGS_s), D, -th, muL, muG, sg(rhoL), sg(rhoG))
    v_est = _equilibrium_vLS(*args_sg, _FI_EST, _LAM_EST)
    v_anu = _equilibrium_vLS(*args_sg, _FI_ANU, _LAM_ANU)
    sel_est = _table_lookup(v_est, _HD_TEST, sg(vLS_s), True)
    sel_anu = _table_lookup(v_anu, _HD_TEST2, sg(vLS_s), False)
    args = (vGS_s, D, -th, muL, muG, rhoL, rhoG)
    hD_est = _interp_pick(sel_est, vLS_s, _HD_TEST, _FI_EST, _LAM_EST, args, smooth)
    hD_anu = _interp_pick(sel_anu, vLS_s, _HD_TEST2, _FI_ANU, _LAM_ANU, args, smooth)
    beta_est = 2.0 * jnp.arccos(1.0 - 2.0 * hD_est)
    aG_est = 1.0 - (beta_est - jnp.sin(beta_est)) / (2.0 * jnp.pi)
    aG_anu = (1.0 - 2.0 * hD_anu) ** 2
    sigma = inp["sigma"]
    ub = 1.53 * (sigma * _G_REG * jnp.abs(rhoL - rhoG) / rhoL**2) ** 0.25 * jnp.sin(th)
    vM = inp["vLS"] + inp["vGS"]
    vG = 1.25 * vM + ub
    if smooth:
        aG_bol = jnp.clip(jnp.abs(inp["vGS"]) / (jnp.abs(vG) + 1e-3), 1e-6, 1 - 1e-6)
    else:
        aG_bol = jnp.clip(inp["vGS"] / jnp.where(vG == 0, 1e-30, vG), 1e-6, 1 - 1e-6)
    return hD_est, hD_anu, aG_est, aG_anu, aG_bol


def _sigmoid(k, x1, x2, relative=False):
    """closures.regime.sigmoid. A forma 1/(1 + exp(clip(-k (x1 - x2), -700, 700))) do numpy
    tem tangente NaN quando o argumento encosta no corte (exp ~ 1e304 vezes uma derivada
    grande dá inf/inf); a logística do JAX tem os mesmos valores (diferença < 1e-300)
    e derivada s (1 - s) estável. relative=True: largura relativa ao valor crítico."""
    d = x1 - x2
    if relative:
        d = d / (jnp.abs(x2) + 1e-300)
    return jax.nn.sigmoid(k * d)


def identify_regime(inp, k_switch, eq, smooth=False, local=False):
    """closures.regime.identify_regime: indicadores (Est, EsS, EsW, Anu, Bub, Int) como
    dict de arrays. Raízes de grandezas que se anulam com as vazões (NL, T, ...) com
    derivada finita em zero (_safe_sqrt); os valores são os do numpy."""
    def _sig(x1, x2):
        return _sigmoid(k_switch, x1, x2, relative=smooth)
    hD_est, hD_anu, aG_est, aG_anu, aG_bol = eq
    if local:
        hD_est = jnp.clip(inp["hD"], 1e-6, 1 - 1e-6)
        aG_est = jnp.clip(inp["alphaG"], 1e-6, 1 - 1e-6)
        a, b = aG_bol, inp["alphaG"]
        aG_bol = jnp.clip(0.5 * (a + b + jnp.sqrt((a - b) ** 2 + EPS_SMAX**2)), 1e-6, 1 - 1e-6)
    vLS, vGS = inp["vLS"], inp["vGS"]
    vM = vLS + vGS
    D, eps, th = inp["D"], inp["eps"], inp["theta"]
    rhoL, rhoG, muL, sigma = inp["rhoL"], inp["rhoG"], inp["muL"], inp["sigma"]
    aL_est = 1.0 - aG_est
    aL_anu = 1.0 - aG_anu
    aL_bol = 1.0 - aG_bol
    if smooth:
        flag_sobe = 0.5 * (1.0 + jnp.tanh(vM / V_SOBE))
    else:
        flag_sobe = (jnp.sign(jnp.cos(th) * (vM + 1e-10)) + 1.0) / 2.0
    flag_desce = 1.0 - flag_sobe
    flag_horizontal = (jnp.abs(th) < (10.0 / 180.0 * jnp.pi)).astype(float)
    flag_vertical = (jnp.abs(jnp.rad2deg(jnp.abs(th)) - 90.0) < 10.0).astype(float)
    flag_naovertical = 1.0 - flag_vertical
    fS_L, ReS_L = fanning(vLS, rhoL, muL, D, eps)
    FL = jnp.sqrt(rhoL / (rhoL - rhoG) / 9.8 / D) * jnp.abs(vLS)
    FG = jnp.sqrt(rhoG / (rhoL - rhoG) / 9.8 / D) * jnp.abs(vGS)
    cos_t = jnp.cos(th)
    FGast = FG / jnp.sqrt(cos_t)
    beta = 2.0 * jnp.arccos(1.0 - 2.0 * hD_est)
    FGast_c = jnp.sqrt(((1.0 - hD_est) ** 2) * (aG_est**3) * (jnp.pi * D / 4.0) / (D * jnp.sin(beta / 2.0)))
    flag_naoest1 = _sig(FGast, FGast_c)
    NL = _safe_sqrt(fS_L * FL**2 / cos_t)
    f_L_est, _ = fanning(vLS / aL_est, rhoL, muL, D, eps)
    fS_L_s = jnp.where(fS_L == 0, 1e-30, fS_L)
    NL_c = 1.0 * jnp.sqrt(1.0 - hD_est) * aL_est / jnp.sqrt(f_L_est / fS_L_s)
    flag_naoest2 = _sig(NL, NL_c) * flag_desce
    K = FG * jnp.sqrt(ReS_L / cos_t)
    K_c = 2.0 * jnp.sqrt(aL_est) * aG_est / jnp.sqrt(0.01)
    flag_wavy1 = _sig(K, K_c)
    FL_c = 1.5 * jnp.sqrt(hD_est) * aL_est
    flag_wavy2 = _sig(FL, FL_c)
    EOD = D * jnp.sqrt((rhoL - rhoG) * _G_REG / sigma)
    FG_c = jnp.sqrt(75.0 / (EOD**2) * (jnp.sqrt(1.0 + (EOD / 75.0) ** 2 / 0.0065) - 1.0))
    flag_naoanular = _sig(aL_anu, 0.35)
    flag_anular = _sig(FG, FG_c)
    T = _safe_sqrt(2.0 * fS_L * FL**2 / cos_t)
    T_c = jnp.sqrt(2.0 * jnp.pi * (aL_est**2 * aG_est) / jnp.sin(beta / 2.0) / (f_L_est / fS_L_s))
    flag_bubble3 = _sig(T, T_c)
    rhoM = aL_bol * rhoL + aG_bol * rhoG
    f_M, _ = fanning(vM, rhoM, muL, D, eps)
    vM_s = jnp.where(vM == 0, 1e-30, vM)
    dmax = ((0.725 + 4.15 * _safe_sqrt(jnp.abs(vGS / vM_s))) * (sigma / rhoL) ** (3.0 / 5.0)
            * jnp.abs(2.0 * f_M * vM**3 / D) ** (-2.0 / 5.0))
    ddef = 2.0 * jnp.sqrt(0.4 * sigma / jnp.abs(rhoL - rhoG) / _G_REG)
    dmigr = (3.0 / 8.0) * rhoL / jnp.abs(rhoL - rhoG) * f_M * vM**2 / _G_REG / cos_t
    flag_intermitente1 = _sig(aG_bol, 0.52)
    flag_bubble1 = _sig(0.25, aG_bol)
    flag_checkbolhas = (1.0 - flag_bubble1) * (1.0 - flag_intermitente1)
    flag_bubble2 = _sig(ddef, dmax) * _sig(dmigr, dmax) * flag_checkbolhas
    Est = (1.0 - flag_naoest1) * flag_sobe + (1.0 - flag_naoest2) * (1.0 - flag_sobe)
    p_anu = (1.0 - flag_naoanular) * (flag_naovertical + flag_anular * flag_vertical)
    if local:
        Est = Est * flag_naovertical
        a_hi = _sig(inp["alphaG"], 0.9)
        p_anu = 1.0 - (1.0 - p_anu) * (1.0 - a_hi)
    EsW = Est * (flag_wavy1 * flag_sobe + flag_wavy2 * (1.0 - flag_sobe))
    EsS = Est * (1.0 - EsW)
    Anu = (1.0 - Est) * p_anu
    p_bub = (flag_bubble1 * (1.0 - flag_horizontal)
             + flag_bubble2 * (1.0 - flag_horizontal) * (1.0 - flag_intermitente1) * (1.0 - flag_bubble1)
             + flag_bubble3 * flag_horizontal)
    Bub = (1.0 - Est) * (1.0 - Anu) * p_bub
    Int = (1.0 - Est) * (1.0 - Anu) * (1.0 - p_bub)
    return dict(Est=Est, EsS=EsS, EsW=EsW, Anu=Anu, Bub=Bub, Int=Int)


def blend_closures(inp, flags, used, threshold):
    """closures.regime.blend_closures: soma ponderada só onde flag > threshold."""
    n = inp["alphaG"].shape[0]
    out = {k: jnp.zeros(n) for k in TERM_NAMES}
    for nm in ("Est", "Anu", "Int", "Bub"):
        if nm not in used:
            continue
        w = jnp.where(flags[nm] > threshold, flags[nm], 0.0)
        S = CLOSURE_FUNCS[nm](inp)
        for k in TERM_NAMES:
            out[k] = out[k] + jnp.where(w > 0, w * S[k], 0.0)
    for k in _NAN_TO_ZERO:
        out[k] = jnp.where(jnp.isfinite(out[k]), out[k], 0.0)
    return out


# ----------------------------------------------------------------------------
# modelo a dois fluidos
# ----------------------------------------------------------------------------
class TwoFluidJ:
    """Versão JAX de TwoFluidModel (padrão fixo com misturas ou identificação automática)."""

    def __init__(self, model):
        p = model.params
        self.p = p
        self.fl = model.fluid
        self.auto = p.regime == REGIME_AUTO
        if self.auto:
            # identificação automática: os quatro fechamentos entram ponderados pelos
            # indicadores (mais a mistura de fração baixa, se pedida; a de ângulo não se aplica)
            self.used = {"Est", "Anu", "Int", "Bub"}
            self.blended = bool(p.low_void_regime or p.ipe_smooth or p.ipe_local)
        else:
            self.used = _closures_used(p.regime, p.regime_vertical, p.low_void_regime, p.high_void_regime)
            self.blended = bool((p.regime_vertical and p.vertical_blend > 0) or p.low_void_regime or p.high_void_regime)
        self.thr = 1e-6 if self.blended else 0.1
        self.Area = float(np.pi * p.D**2 / 4.0)
        self.nv = 4
        self.slots = (0, 1, 2, 3)
        self.A_pattern = model.A_pattern
        self.B_pattern = model.B_pattern

    def vertical_weight(self, theta):
        p = self.p
        ath = jnp.rad2deg(jnp.abs(theta))
        if p.vertical_blend > 0:
            u = jnp.clip((ath - (p.vertical_angle - p.vertical_blend)) / (2.0 * p.vertical_blend), 0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)
        return (ath > p.vertical_angle).astype(float)

    def local_state(self, Y, theta):
        p, fl = self.p, self.fl
        aG = jnp.clip(Y[:, 0], p.alpha_min, p.alpha_max)
        aL = 1.0 - aG
        n = aG.shape[0]
        P = p.PR * Y[:, 1]
        rhoL, rhoG, CL, CG = fluid_props(fl, P)
        hD, beta = stratified_level(aG)
        Area = self.Area
        vLS = Y[:, 2] / (rhoL * Area)
        vGS = Y[:, 3] / (rhoG * Area)
        ones = jnp.ones(n)
        inp = dict(alphaG=aG, alphaL=aL, vL=vLS / aL, vLS=vLS, vG=vGS / aG, vGS=vGS,
                   AreaL=Area * aL, AreaG=Area * aG, D=p.D * ones, eps=p.eps * ones,
                   theta=theta * ones, rhoL=rhoL, rhoG=rhoG, muL=fl.muL * ones, muG=fl.muG * ones,
                   CL=CL, CG=CG, sigma=fl.sigma * ones, beta=beta, hD=hD,
                   fi_mult=p.fi_mult * ones, fi_vgt=p.fi_vgt * ones,
                   bubble_drag=(1.0 if p.bubble_drag == "fisico" else 0.0) * ones)
        return inp, P

    def closures(self, inp, return_flags=False):
        p = self.p
        n = inp["alphaG"].shape[0]
        if self.auto:
            flags = identify_regime(inp, p.k_switch, equilibrium_fractions(inp, smooth=p.ipe_smooth), smooth=p.ipe_smooth, local=p.ipe_local)
            if p.low_void_regime:
                u = jnp.clip((2.0 * p.low_void_alpha - inp["alphaG"]) / p.low_void_alpha, 0.0, 1.0)
                wl = u * u * (3.0 - 2.0 * u)
                flv = fixed_flags(p.low_void_regime, n)
                flags = {k: (1.0 - wl) * flags[k] + wl * flv[k] for k in flags}
        else:
            flags = fixed_flags(p.regime, n)
            if p.regime_vertical:
                w = self.vertical_weight(inp["theta"])
                fv = fixed_flags(p.regime_vertical, n)
                flags = {k: (1.0 - w) * flags[k] + w * fv[k] for k in flags}
            if p.low_void_regime:
                u = jnp.clip((2.0 * p.low_void_alpha - inp["alphaG"]) / p.low_void_alpha, 0.0, 1.0)
                wl = u * u * (3.0 - 2.0 * u)
                flv = fixed_flags(p.low_void_regime, n)
                flags = {k: (1.0 - wl) * flags[k] + wl * flv[k] for k in flags}
            if p.high_void_regime:
                u = jnp.clip((inp["alphaG"] - (p.high_void_alpha - 0.05)) / 0.1, 0.0, 1.0)
                wh = u * u * (3.0 - 2.0 * u)
                fhv = fixed_flags(p.high_void_regime, n)
                flags = {k: (1.0 - wh) * flags[k] + wh * fhv[k] for k in flags}
        S = blend_closures(inp, flags, self.used, self.thr)
        if p.interfacial_pressure > 0:
            aG, aL = inp["alphaG"], inp["alphaL"]
            rhoL, rhoG, CL, CG = inp["rhoL"], inp["rhoG"], inp["CL"], inp["CG"]
            slip = jnp.clip(inp["vG"] - inp["vL"], -p.ip_slip_max, p.ip_slip_max)
            den = aG * rhoL + aL * rhoG
            g = aG * aL / den
            dP = p.interfacial_pressure * rhoG * rhoL * g * slip**2
            dg_daG = ((aL - aG) * den - aG * aL * (rhoL - rhoG)) / den**2
            ddP_daG = p.interfacial_pressure * rhoG * rhoL * dg_daG * slip**2
            dden_dP = aG * CL + aL * CG
            ddP_dP = p.interfacial_pressure * slip**2 * aG * aL * ((CG * rhoL + rhoG * CL) * den - rhoG * rhoL * dden_dP) / den**2
            S = dict(S)
            S["dPL"] = S["dPL"] + dP
            S["dPG"] = S["dPG"] + dP
            S["ddPLdalphaG"] = S["ddPLdalphaG"] + ddP_daG
            S["ddPGdalphaG"] = S["ddPGdalphaG"] + ddP_daG
            S["ddPLdP"] = S["ddPLdP"] + ddP_dP
            S["ddPGdP"] = S["ddPGdP"] + ddP_dP
        if return_flags:
            return S, flags
        return S

    def coefficients(self, Y, theta):
        """F (12, n) como TwoFluidModel.coefficients."""
        p, fl = self.p, self.fl
        inp, P = self.local_state(Y, theta)
        S = self.closures(inp)
        aG, aL = inp["alphaG"], inp["alphaL"]
        rhoL, rhoG, CL, CG = inp["rhoL"], inp["rhoG"], inp["CL"], inp["CG"]
        vL, vG = inp["vL"], inp["vG"]
        AreaL, AreaG = inp["AreaL"], inp["AreaG"]
        Area = self.Area
        sin_t = jnp.sin(inp["theta"])
        auxr = aL / fl.gammaL + aG / fl.gammaG
        rL = (aL / fl.gammaL) / auxr
        rG = (aG / fl.gammaG) / auxr
        rL2 = (1.0 / fl.gammaL) / auxr
        rG2 = (1.0 / fl.gammaG) / auxr
        if p.legacy_convective:
            vL2, vG2 = vL * jnp.abs(vL), vG * jnp.abs(vG)
            cL, cG = 2.0 * jnp.abs(vL), 2.0 * jnp.abs(vG)
        else:
            vL2, vG2 = vL * vL, vG * vG
            cL, cG = 2.0 * vL, 2.0 * vG
        F = [
            -rG / (Area * rhoL),
            rL / (Area * rhoG),
            rL2 / (Area * p.PR * CL),
            rG2 / (Area * p.PR * CG),
            -Area * (S["dPL"] - aL * S["ddPLdalphaG"] - vL2 * rhoL),
            p.PR * AreaL * (1.0 - vL2 * CL + S["ddPLdP"]),
            cL,
            Area * (S["dPG"] + aG * S["ddPGdalphaG"] - vG2 * rhoG),
            p.PR * AreaG * (1.0 - vG2 * CG + S["ddPGdP"]),
            cG,
            S["GammaL"] + G * AreaL * rhoL * sin_t,
            S["GammaG"] + G * AreaG * rhoG * sin_t,
        ]
        return jnp.stack(F)

    def conservative_terms(self, Y, theta):
        inp, P = self.local_state(Y, theta)
        S, flags = self.closures(inp, return_flags=True)
        return dict(inp=inp, S=S, P=P, Area=self.Area, vm_weight=flags["Bub"])


# ----------------------------------------------------------------------------
# modelo de deriva (Beggs & Brill)
# ----------------------------------------------------------------------------
def _bb_sigmoid(k, x1, x2):
    return jax.nn.sigmoid(k * (x1 - x2))     # ver _sigmoid: mesma função, tangente estável


def beggs_brill(P, qL, qG, D, eps, theta, fl, regime):
    g = 9.80665
    P = jnp.maximum(1e5, P)
    rhoL = fl.rhoL0 * jnp.ones_like(P)
    rhoG = fl.rhoG0 * (P / fl.P0) ** (1.0 / fl.gammaG)
    Area = jnp.pi * D**2 / 4.0
    vSL = qL / rhoL / Area
    vSG = qG / rhoG / Area
    direction = jnp.where(vSL > 0, 1.0, -1.0)
    vSL = jnp.abs(vSL)
    vSG = jnp.abs(vSG)
    vm = vSL + vSG
    lambdaL = vSL / (vm + 1e-9)
    lambdaG = vSG / (vm + 1e-9)
    ixL = lambdaL > 1
    lambdaL = jnp.where(ixL, 1.0, lambdaL)
    lambdaG = jnp.where(ixL, 0.0, lambdaG)
    ixG = lambdaG > 1
    lambdaG = jnp.where(ixG, 1.0, lambdaG)
    lambdaL = jnp.where(ixG, 0.0, lambdaL)
    rhoN = rhoL * lambdaL + rhoG * lambdaG
    muN = fl.muL * lambdaL + fl.muG * lambdaG
    NFR = vm**2 / (g * D)
    NLV = vSL * (rhoL / fl.sigma / g) ** 0.25
    NRE = rhoN * vm * D / muN
    L1 = 316.0 * lambdaL**0.3020
    L2 = 0.0009252 * lambdaL ** (-2.4684)
    L3 = 0.10 * lambdaL ** (-1.4516)
    L4 = 0.5 * lambdaL ** (-6.7380)
    k = 1e1
    tA = _bb_sigmoid(k, lambdaL / 0.01, 1)
    tB = _bb_sigmoid(k, lambdaL / 0.4, 1)
    tC = _bb_sigmoid(k, L1 / NFR, 1)
    tD = _bb_sigmoid(k, L2 / NFR, 1)
    tE = _bb_sigmoid(k, L3 / NFR, 1)
    tF = _bb_sigmoid(k, L4 / NFR, 1)
    fEst = jnp.clip((1 - tA) * tC + tA * tD, 0, 1)
    fTra = jnp.clip(tA * tE * (1 - tD), 0, 1)
    fInt = jnp.clip((1 - tB) * tA * tC * (1 - tE) + tB * tF * (1 - tE), 0, 1)
    fDis = jnp.clip((1 - tB) * (1 - tC) + tB * (1 - tF), 0, 1)
    if regime != 6:
        z = jnp.zeros_like(lambdaL)
        o = jnp.ones_like(lambdaL)
        fEst, fTra, fInt, fDis = z, z, z, z
        if regime == 1:
            fTra = o
        elif regime == 2:
            fDis = o
        elif regime == 3:
            fInt = o
        elif regime in (4, 5):
            fEst = o
    HL0_Est = jnp.clip(jnp.maximum(0.98 * lambdaL**0.4846 / NFR**0.0868, lambdaL), lambdaL, 1)
    HL0_Int = jnp.clip(jnp.maximum(0.845 * lambdaL**0.5351 / NFR**0.0173, lambdaL), lambdaL, 1)
    HL0_Dis = jnp.clip(jnp.maximum(1.065 * lambdaL**0.5824 / NFR**0.0609, lambdaL), lambdaL, 1)
    flagAsc = (1 + jnp.sign(direction) * jnp.sign(theta)) / 2.0
    flagDesc = 1 - flagAsc
    s18 = jnp.sin(1.8 * theta) - 0.333 * jnp.sin(1.8 * theta) ** 3

    def corr(Cd, Ce, Cf, Cg):
        C = (1 - lambdaL) * jnp.log(Cd * lambdaL**Ce * NLV**Cf * NFR**Cg)
        C = jnp.where(jnp.isnan(C), 0.0, C)
        C = jnp.where(C == -jnp.inf, 0.0, C)
        C = jnp.maximum(C, 0.0)
        return 1 + C * s18

    qsiDesc = corr(4.70, -0.3692, 0.1244, -0.5056)
    qsi_Est = corr(0.011, -3.768, 3.539, -1.614) * flagAsc + qsiDesc * flagDesc
    qsi_Int = corr(2.96, 0.305, -0.4473, 0.0978) * flagAsc + qsiDesc * flagDesc
    qsi_Dis = 1.0 * flagAsc + qsiDesc * flagDesc
    HL_Est = HL0_Est * qsi_Est
    HL_Int = HL0_Int * qsi_Int
    HL_Dis = HL0_Dis * qsi_Dis
    fi = (L3 - NFR) / (L3 - L2)
    HL_Tra = fi * HL_Est + (1 - fi) * HL_Int
    HL = HL_Est * fEst + HL_Int * fInt + HL_Dis * fDis + HL_Tra * fTra
    HL = jnp.clip(jnp.where(jnp.isnan(HL), lambdaL, HL), 1e-9, 1 - 1e-9)
    HG = 1 - HL
    ix1 = HG < 1e-6
    HG = jnp.where(ix1, 1e-6, HG)
    HL = jnp.where(ix1, 1 - 1e-6, HL)
    ix2 = HL < 1e-6
    HL = jnp.where(ix2, 1e-6, HL)
    HG = jnp.where(ix2, 1 - 1e-6, HG)
    vL = direction * vSL / HL
    vG = direction * vSG / HG
    rhoS = HL * rhoL + HG * rhoG
    NRE_s = jnp.where(NRE == 0, 1e-30, NRE)
    A1 = (2.457 * jnp.log(1.0 / (0.27 * eps / D + (7.0 / NRE_s) ** 0.9))) ** 16
    B1 = (37530.0 / NRE_s) ** 16
    f = 4.0 * (2.0 * ((8.0 / NRE_s) ** 12 + 1.0 / (A1 + B1) ** 1.5) ** (1.0 / 12.0))
    kk = jnp.log(lambdaL / HL**2)
    S = jnp.log(2.2 * jnp.exp(kk) - 1.2)
    ix = (kk <= 0) | (kk >= 0.1823)
    kk_safe = jnp.where(kk == 0, 1e-300, kk)
    S_alt = (-0.0523 / kk_safe + 3.1820 - 0.8725 * kk + 0.01853 * kk**3) ** (-1)
    S = jnp.where(ix, S_alt, S)
    S = jnp.where(jnp.isfinite(S), S, 0.0)
    S = jnp.where(ix1 | ix2, 0.0, S)
    ftp = f * jnp.exp(S)
    tension = direction * (ftp * rhoS * vm**2) / 8.0
    vM = (HL * rhoL * vL + HG * rhoG * vG) / (HL * rhoL + HG * rhoG)
    return dict(tension=tension, rhoS=rhoS, vM=vM, HL=HL, HG=HG, rhoL=rhoL, rhoG=rhoG, vL=vL, vG=vG, vm=vm)


class BeggsBrillJ:
    """Versão JAX de BeggsBrillModel. As derivadas internas do fechamento (momento,
    fluxo de momento e hold-up em relação a P, qL e qG) são calculadas por
    diferenciação automática (`jax.jvp`, bb_derivatives="exact") ou, para reproduzir
    a tese, por diferenças finitas de 1e-3 relativo (bb_derivatives="fd")."""

    def __init__(self, model):
        self.p = model.params
        self.fl = model.fluid
        self.nv = 3
        self.slots = (1, 2, 3)
        self.A_pattern = model.A_pattern
        self.B_pattern = model.B_pattern
        self.exact = getattr(model.params, "bb_derivatives", "exact") == "exact"

    def coefficients(self, Y, theta):
        p, fl = self.p, self.fl
        PT = Y[:, 1]
        qL = jnp.maximum(0.001, Y[:, 2])
        qG = jnp.maximum(0.001, Y[:, 3])
        n = PT.shape[0]
        th = theta * jnp.ones(n)
        P = p.PR * PT
        bb = beggs_brill(P, qL, qG, p.D, p.eps, th, fl, p.regime)
        rhoL, rhoG = bb["rhoL"], bb["rhoG"]
        cL = rhoL / (P * fl.gammaL)
        cG = rhoG / (P * fl.gammaG)
        alphaL, alphaG = bb["HL"], bb["HG"]
        Mom = bb["rhoS"] * bb["vM"]
        Vel = bb["rhoS"] * bb["vM"] ** 2
        if self.exact:
            # derivadas internas exatas: jvp do fechamento (nó a nó, pois cada nó só
            # depende dos seus próprios P, qL, qG) nas três direções
            def g(P_, qL_, qG_):
                b = beggs_brill(P_, qL_, qG_, p.D, p.eps, th, fl, p.regime)
                return b["rhoS"] * b["vM"], b["rhoS"] * b["vM"] ** 2, b["HG"]
            o, z = jnp.ones_like(P), jnp.zeros_like(P)
            _, (dMomdP, dVeldP, daGdP) = jax.jvp(g, (P, qL, qG), (o, z, z))
            _, (dMomdqL, dVeldqL, daGdqL) = jax.jvp(g, (P, qL, qG), (z, o, z))
            _, (dMomdqG, dVeldqG, daGdqG) = jax.jvp(g, (P, qL, qG), (z, z, o))
        else:
            P2 = P * (1 + 1e-3)
            qL2 = qL * (1 + 1e-3) + 1e-6
            qG2 = qG * (1 + 1e-3) + 1e-6
            bP = beggs_brill(P2, qL, qG, p.D, p.eps, th, fl, p.regime)
            bL = beggs_brill(P, qL2, qG, p.D, p.eps, th, fl, p.regime)
            bG = beggs_brill(P, qL, qG2, p.D, p.eps, th, fl, p.regime)
            dMomdP = (bP["rhoS"] * bP["vM"] - Mom) / (P2 - P)
            dMomdqL = (bL["rhoS"] * bL["vM"] - Mom) / (qL2 - qL)
            dMomdqG = (bG["rhoS"] * bG["vM"] - Mom) / (qG2 - qG)
            dVeldP = (bP["rhoS"] * bP["vM"] ** 2 - Vel) / (P2 - P)
            dVeldqL = (bL["rhoS"] * bL["vM"] ** 2 - Vel) / (qL2 - qL)
            dVeldqG = (bG["rhoS"] * bG["vM"] ** 2 - Vel) / (qG2 - qG)
            daGdP = (bP["HG"] - alphaG) / (P2 - P)
            daGdqL = (bL["HG"] - alphaG) / (qL2 - qL)
            daGdqG = (bG["HG"] - alphaG) / (qG2 - qG)
        Area = jnp.pi * p.D**2 / 4.0
        sgn = jnp.sign(bb["vM"])
        z = jnp.zeros(n)
        MA = jnp.stack([jnp.stack([p.PR * (alphaL * cL - rhoL * daGdP), -rhoL * daGdqL, -rhoL * daGdqG], axis=1),
                        jnp.stack([p.PR * (alphaG * cG + rhoG * daGdP), rhoG * daGdqL, rhoG * daGdqG], axis=1),
                        jnp.stack([p.PR * dMomdP, dMomdqL, dMomdqG], axis=1)], axis=1)
        MB = jnp.stack([jnp.stack([z, jnp.full(n, 1.0 / Area), z], axis=1),
                        jnp.stack([z, z, jnp.full(n, 1.0 / Area)], axis=1),
                        jnp.stack([p.PR * (1.0 + sgn * dVeldP), sgn * dVeldqL, sgn * dVeldqG], axis=1)], axis=1)
        MC = jnp.stack([z, z, jnp.pi * p.D * bb["tension"] / Area + bb["rhoS"] * G * jnp.sin(th)], axis=1)
        invA = jnp.linalg.inv(MA)
        bad = ~jnp.all(jnp.isfinite(invA), axis=(1, 2))
        pinvA = jnp.linalg.pinv(MA, rcond=1e-12)
        invA = jnp.where(bad[:, None, None], pinvA, invA)
        Amat = invA @ MB
        Bvec = jnp.einsum("nij,nj->ni", invA, MC)
        return jnp.concatenate([Amat.reshape(n, 9).T, Bvec.T], axis=0)


    def conservative_terms(self, Y, theta):
        """BeggsBrillModel.conservative_terms: hold-up e derivadas (jvp exatas, ou diferenças
        finitas iguais às do numpy com bb_derivatives="fd"), rhoS, vM, tensão e propriedades
        do fluido (Fluid.props) para as equações de massa."""
        p, fl = self.p, self.fl
        n = Y.shape[0]
        th = theta * jnp.ones(n)
        P = jnp.maximum(p.PR * Y[:, 1], 1e3)
        qL = Y[:, 2]
        qG = Y[:, 3]
        Area = jnp.pi * p.D**2 / 4.0

        def hg(P_, qL_, qG_):
            return beggs_brill(P_, qL_, qG_, p.D, p.eps, th, fl, p.regime)["HG"]
        bb = beggs_brill(P, qL, qG, p.D, p.eps, th, fl, p.regime)
        HG = bb["HG"]
        if self.exact:
            # derivadas exatas avaliadas a 1e-6 kg/s do lado do escoamento: em vazão de gás
            # exatamente nula a correlação está na região cortada (HG = 1e-6, derivada zero)
            # e o coeficiente da restrição de escorregamento cairia no piso; a derivada
            # lateral (como a diferença finita avançada da tese) é a sensibilidade real
            o, z = jnp.ones_like(P), jnp.zeros_like(P)
            qLs = qL + 1e-6 * jnp.sign(qL + 1e-300)
            qGs = qG + 1e-6 * jnp.sign(qG + 1e-300)
            _, dHGdP = jax.jvp(hg, (P, qLs, qGs), (o, z, z))
            _, dHGdqL = jax.jvp(hg, (P, qLs, qGs), (z, o, z))
            _, dHGdqG = jax.jvp(hg, (P, qLs, qGs), (z, z, o))
        else:
            dP_ = 1e-3 * P
            dqL = 1e-3 * jnp.abs(qL) + 1e-6
            dqG = 1e-3 * jnp.abs(qG) + 1e-6
            dHGdP = (hg(P + dP_, qL, qG) - HG) / dP_
            dHGdqL = (hg(P, qL + dqL, qG) - HG) / dqL
            dHGdqG = (hg(P, qL, qG + dqG) - HG) / dqG
        rhoL, rhoG, CL, CG = fluid_props(fl, P)
        HL = bb["HL"]
        inp = dict(rhoL=rhoL, rhoG=rhoG, CL=CL, CG=CG, alphaG=HG, alphaL=HL)
        return dict(Area=Area, P=P, inp=inp, HG=HG, HL=HL, rhoS=bb["rhoS"], vM=bb["vM"], tension=bb["tension"],
                    dHGdP=dHGdP, dHGdqL=dHGdqL, dHGdqG=dHGdqG, C_mix=HL * CL + HG * CG)


# ----------------------------------------------------------------------------
# modelos monofásicos
# ----------------------------------------------------------------------------
class SinglePhaseJ:
    def __init__(self, model):
        self.p = model.params
        self.fl = model.fluid
        self.phase = model.phase
        self.nv = 2
        self.slots = tuple(model.slots)
        self.A_pattern = model.A_pattern
        self.B_pattern = model.B_pattern

    def _props(self, P):
        rhoL, rhoG, CL, CG = fluid_props(self.fl, P)
        if self.phase == "L":
            return rhoL, CL, self.fl.muL
        return rhoG, CG, self.fl.muG

    def coefficients(self, Y, theta):
        p = self.p
        PT = Y[:, 1]
        q = Y[:, self.slots[1]]
        n = PT.shape[0]
        th = theta * jnp.ones(n)
        P = p.PR * PT
        rho, C, mu = self._props(P)
        Area = jnp.pi * p.D**2 / 4.0
        v = q / (rho * Area)
        f, _ = fanning(v, rho, mu, p.D, p.eps)
        Gamma = jnp.pi * p.D * f * rho * v * vabs(v) / 2.0
        return jnp.stack([1.0 / Area / C / p.PR,
                          Area * (1.0 - C * v * jnp.abs(v)) * p.PR,
                          2.0 * jnp.abs(v),
                          Gamma + G * Area * rho * jnp.sin(th)])

    def conservative_terms(self, Y, theta):
        """Termos para o esquema de volumes finitos (_SinglePhase.conservative_terms)."""
        p = self.p
        P = p.PR * Y[:, 1]
        rho, C, mu = self._props(P)
        Area = jnp.pi * p.D**2 / 4.0
        q = Y[:, self.slots[1]]
        v = q / (rho * Area)
        f, _ = fanning(v, rho, mu, p.D, p.eps)
        Gamma = jnp.pi * p.D * f * rho * v * vabs(v) / 2.0
        return dict(rho=rho, C=C, v=v, Gamma=Gamma, Area=Area, P=P)


def quasi_linear(mj, Y, theta):
    """(A (n, nv, nv), B (n, nv)) a partir dos padrões do modelo (PipeModel.quasi_linear)."""
    F = mj.coefficients(Y, theta)
    n = F.shape[1]
    nv = mj.nv
    Ap, Bp = np.asarray(mj.A_pattern), np.asarray(mj.B_pattern)
    zero = jnp.zeros(n)
    rows = []
    for i in range(nv):
        rows.append(jnp.stack([F[Ap[i, j] - 1] if Ap[i, j] > 0 else zero for j in range(nv)], axis=1))
    A = jnp.stack(rows, axis=1)
    B = jnp.stack([F[Bp[i] - 1] if Bp[i] > 0 else zero for i in range(nv)], axis=1)
    return A, B


def make_model_j(model):
    name = model.name
    if name == "twofluid":
        return TwoFluidJ(model)
    if name == "beggs_brill":
        return BeggsBrillJ(model)
    if name in ("liquid", "gas"):
        return SinglePhaseJ(model)
    raise NotImplementedError(f"rhs JAX: modelo {name}")


# ----------------------------------------------------------------------------
# elementos de rede (network/elements.py)
# ----------------------------------------------------------------------------
def valve_upstream_pressure(P_down, qL, qG, opening, CV, fl, PR):
    qM = qL + qG
    P = P_down * PR
    rhoL, rhoG, _, _ = fluid_props(fl, jnp.maximum(P, 1e3))
    qM_safe = jnp.where(qM != 0, qM, 1.0)
    rhoM = jnp.where(qM != 0, (qL * rhoL + qG * rhoG) / qM_safe, rhoL)
    ab = jnp.maximum(opening, 1e-6)
    dP = qM * jnp.abs(qM) / rhoM / (0.0075 * CV * ab) ** 2
    ZP = (P + dP * 1e5) / PR
    return jnp.maximum(1e-3, ZP)


def reservoir_flows(P_node, IPL, IPG, P_res_bar, PR):
    dP = P_res_bar - P_node * PR / 1e5
    return jnp.maximum(0.0, IPL * dP), jnp.maximum(0.0, IPG * dP)


def check_valve_flows(P_up, P_down, HG, CV, dP_min, fl, PR):
    dP = (P_up - P_down) * PR / 1e5
    P_mean = 0.5 * (P_up + P_down) * PR
    rhoL, rhoG, _, _ = fluid_props(fl, jnp.maximum(P_mean, 1e3))
    HL = 1.0 - HG
    rhoM = HL * rhoL + HG * rhoG
    eps = 1e-3
    qM = 0.0075 * CV * jnp.sqrt(rhoM) * dP / jnp.sqrt(jnp.abs(dP) + eps)
    if dP_min <= 0.0:
        w = 0.5 * (1.0 + jnp.tanh(dP / eps))
    else:
        w = 1.0 / (1.0 + jnp.exp(jnp.clip(-(dP - dP_min) / (0.25 * eps + 0.02 * max(dP_min, 0.0)), -700, 700)))
    return w * HL * qM, w * HG * qM
