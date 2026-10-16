"""Identificação do padrão de escoamento (IPE) e ponderação dos fechamentos.

Porta determinaAlfaEstavel.m (modelos de equilíbrio de Taitel e Pots),
identificaRegimeEstavel.m (critérios de estabilidade com chaveamento
sigmoidal) e montaRegime.m/filtra_entradas.m. Uma só implementação para numpy e
JAX (`bk`): as buscas nas tabelas de equilíbrio são vetorizadas (sem laço por nó)
e, no JAX, avaliadas sem gradiente (`stop_gradient`); a ponderação dos fechamentos
avalia cada fechamento só onde o seu indicador conta (`bk.masked_apply`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..backend import NUMPY
from ..friction import fanning
from .annular import annular
from .bubble import bubble
from .common import ClosureInputs, ClosureTerms
from .intermittent import intermittent
from .stratified import stratified

REGIME_ANNULAR = 1
REGIME_BUBBLE = 2
REGIME_INTERMITTENT = 3
REGIME_STRAT_SMOOTH = 4
REGIME_STRAT_WAVY = 5
REGIME_AUTO = 6

G = 9.81

CLOSURE_FUNCS = {"Est": stratified, "Anu": annular, "Int": intermittent, "Bub": bubble}
REGIME_FLAG = {
    REGIME_ANNULAR: "Anu",
    REGIME_BUBBLE: "Bub",
    REGIME_INTERMITTENT: "Int",
    REGIME_STRAT_SMOOTH: "Est",
    REGIME_STRAT_WAVY: "Est",
}
# termos em que NaN/inf (estados extremos) são postos a zero depois da ponderação
NAN_TO_ZERO = ("ddPLdalphaG", "ddPGdalphaG", "dPL", "dPG", "dPI", "ddPLdP", "ddPGdP")


@dataclass
class RegimeFlags:
    Est: np.ndarray
    EsS: np.ndarray
    EsW: np.ndarray
    Anu: np.ndarray
    Bub: np.ndarray
    Int: np.ndarray

    @property
    def reg1(self):
        return self.Anu * 1 + self.Bub * 2 + self.Int * 3 + self.Est * 4

    @property
    def reg2(self):
        return self.Anu * 1 + self.Bub * 2 + self.Int * 3 + self.EsS * 4 + self.EsW * 5

    def names(self):
        return [f.name for f in fields(self)]


def sigmoid(k, x1, x2, relative=False, bk=NUMPY):
    """1/(1+exp(-k (x1-x2))): ~1 se x1 > x2, ~0 se x1 < x2.

    relative=True: o argumento é k (x1 - x2)/|x2| -- a largura da transição passa a
    ser uma fração do valor crítico (~2/k), igual para critérios de escalas muito
    diferentes (FG* ~ 0.02, K ~ 1..10), em vez de absoluta como na tese."""
    xp = bk.xp
    d = x1 - x2
    if relative:
        d = d / (xp.abs(x2) + 1e-300)
    return bk.sigmoid(k * d)


def fixed_regime(regime, n, bk=NUMPY):
    xp = bk.xp
    z = xp.zeros(n)
    o = xp.ones(n)
    fl = RegimeFlags(Est=z, EsS=z, EsW=z, Anu=z, Bub=z, Int=z)
    if regime == REGIME_ANNULAR:
        fl.Anu = o
    elif regime == REGIME_BUBBLE:
        fl.Bub = o
    elif regime == REGIME_INTERMITTENT:
        fl.Int = o
    elif regime == REGIME_STRAT_SMOOTH:
        fl.Est, fl.EsS = o, o
    elif regime == REGIME_STRAT_WAVY:
        fl.Est, fl.EsW = o, o
    else:
        raise ValueError(f"regime fixo inválido: {regime}")
    return fl


def closures_used(regime, *extra_regimes):
    """Conjunto estático de fechamentos que podem ter peso não nulo para um padrão fixo e
    os padrões das regras (vertical, fração baixa/alta); regime 6 -> todos."""
    if regime == REGIME_AUTO:
        return frozenset(CLOSURE_FUNCS)
    names = set()
    for r in (regime,) + tuple(r for r in extra_regimes if r):
        names.add(REGIME_FLAG[int(r)])
    return frozenset(names)


# ----------------------------------------------------------------------------
# Modelos de equilíbrio (Taitel & Dukler 1976 / Pots) para hD de equilíbrio
# ----------------------------------------------------------------------------
_AUX = (10.0 ** np.arange(-3, 0.0001, 0.01)) / 2.0
_HD_TEST = np.concatenate([_AUX, 1.0 - _AUX[::-1]])
_HD_TEST2 = np.arange(0.01, 0.9901, 0.01)
V_SOBE = 0.02  # m/s: largura da regularização do indicador de sentido (vM > 0) no modo suave
EPS_SMAX = 0.02  # largura do máximo suave entre a fração de bolhas de equilíbrio e a fração local (modo local)


def _table_geometry(hD, geometry, n=0.2, m=0.2):
    """FI e LAMBDA de Lockhart-Martinelli: só dependem de hD (tabelas), logo são
    constantes pré-calculadas em numpy."""
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


def _equilibrium_vLS(vGS, D, eta, muL, muG, rhoL, rhoG, FI, LAMBDA, CL=0.046, CG=0.046, n=0.2, m=0.2, bk=NUMPY):
    """vLS de equilíbrio (n x n_hD) para uma tabela de hD (FI, LAMBDA (nh,)) ou para
    entradas escolhidas por linha (FI, LAMBDA (n, k))."""
    xp = bk.xp
    Re_GS = D * xp.abs(vGS) * rhoG / muG
    Re_GS = xp.where(Re_GS == 0, 1e-30, Re_GS)
    f_GS = CG * Re_GS ** (-m)
    tau_GS = f_GS * rhoG * vGS * xp.abs(vGS) / 2.0
    tau_s = xp.where(tau_GS == 0, 1e-30, tau_GS)
    Y = (rhoL - rhoG) * G * xp.sin(eta) / (4.0 / D * tau_s)
    FI, LAMBDA = xp.asarray(FI), xp.asarray(LAMBDA)
    if FI.ndim == 1:
        FI, LAMBDA = FI[None, :], LAMBDA[None, :]
    with bk.errstate(divide="ignore", invalid="ignore"):
        X2 = (4.0 * Y[:, None] + FI) / LAMBDA
        X2 = xp.where(X2 < 0, xp.nan, X2)
        tau_LS = X2 * tau_GS[:, None]
        vLS = (
            (tau_LS / CL) ** (1.0 / (2.0 - n))
            * (D * rhoL / muL)[:, None] ** (n / (2.0 - n))
            * (rhoL / 2.0)[:, None] ** (1.0 / (n - 2.0))
        )
    return vLS


def _lockhart_vLS(vGS, hD, D, eta, muL, muG, rhoL, rhoG, geometry, CL=0.046, CG=0.046, n=0.2, m=0.2):
    """Forma antiga (broadcasting (n,1) x (1,nh)): mantida para referência/testes."""
    FI, LAMBDA = _table_geometry(np.asarray(hD).ravel(), geometry, n, m)
    return _equilibrium_vLS(
        np.asarray(vGS).ravel(),
        np.asarray(D).ravel(),
        np.asarray(eta).ravel(),
        np.asarray(muL).ravel(),
        np.asarray(muG).ravel(),
        np.asarray(rhoL).ravel(),
        np.asarray(rhoG).ravel(),
        FI,
        LAMBDA,
        CL,
        CG,
        n,
        m,
    )


def smooth_max(a, b, eps=EPS_SMAX, bk=NUMPY):
    """max(a, b) com derivada contínua: 0.5 (a + b + sqrt((a - b)^2 + eps^2))."""
    return 0.5 * (a + b + bk.xp.sqrt((a - b) ** 2 + eps**2))


def _table_lookup(v, vLS, after_last_nan, bk):
    """Índice da tabela cujo vLS de equilíbrio é o mais próximo de vLS (no estratificado só
    a partir do último NaN da linha) e os índices dos candidatos vizinhos (modo suave).
    Devolve (idx, prev, next, has_cand, has_prev, has_next)."""
    xp = bk.xp
    nan = xp.isnan(v)
    nh = v.shape[1]
    j = xp.arange(nh)
    if after_last_nan:
        last = xp.max(xp.where(nan, j[None, :], -1), axis=1)
        cand = (j[None, :] > last[:, None]) & ~nan
    else:
        cand = ~nan
    err = xp.where(cand, xp.minimum(xp.abs(v - vLS[:, None]), 1e300), xp.inf)
    idx = xp.argmin(err, axis=1)
    nxt = xp.min(xp.where(cand & (j[None, :] > idx[:, None]), j[None, :], nh), axis=1)
    prv = xp.max(xp.where(cand & (j[None, :] < idx[:, None]), j[None, :], -1), axis=1)
    return idx, xp.clip(prv, 0, nh - 1), xp.clip(nxt, 0, nh - 1), xp.any(cand, axis=1), prv >= 0, nxt < nh


def _interp_pick(sel, v_table, vLS, hd_table, FI, LAM, args, smooth, bk):
    """hD escolhido: entrada mais próxima (tese) ou interpolação linear entre os
    candidatos que enquadram vLS (modo suave). No numpy os vLS das entradas vêm da
    tabela já calculada; no JAX são reavaliados com gradiente nas (até) três entradas."""
    xp = bk.xp
    idx, prv, nxt, has, has_p, has_n = sel
    hd = xp.asarray(hd_table)
    if not smooth:
        return xp.where(has, hd[idx], 1e-6)
    cols = xp.stack([prv, idx, nxt], axis=1)
    if bk.has_ad:
        v3 = _equilibrium_vLS(*args, xp.asarray(FI)[cols], xp.asarray(LAM)[cols], bk=bk)  # (n, 3) com gradiente
    else:
        v3 = xp.take_along_axis(v_table, cols, axis=1)
    vp, vj, vn = v3[:, 0], v3[:, 1], v3[:, 2]
    hp, hj, hn = hd[prv], hd[idx], hd[nxt]
    use_n = has_n & ((vj - vLS) * (vn - vLS) <= 0)
    use_p = ~use_n & has_p & ((vp - vLS) * (vj - vLS) <= 0)
    va = xp.where(use_n, vj, vp)
    vb = xp.where(use_n, vn, vj)
    ha = xp.where(use_n, hj, hp)
    hb = xp.where(use_n, hn, hj)
    dv = vb - va
    ok = (use_n | use_p) & (dv != 0)
    dv_s = xp.where(ok, dv, 1.0)
    h_int = ha + (vLS - va) / dv_s * (hb - ha)
    return xp.where(has, xp.where(ok, h_int, hj), 1e-6)


def equilibrium_fractions(inp: ClosureInputs, smooth=False, bk=NUMPY):
    """Frações de área de equilíbrio para estratificado (Taitel), anular (Pots)
    e bolhas (Zuber). Retorna (hD_est, hD_anu, aG_est, aG_anu, aG_bol).

    smooth=True (modo suave da identificação): os equilíbrios são avaliados com os
    módulos das velocidades superficiais (co-corrente: com vGS < 0 os modelos dão
    NaN e o código original cai em hD = 1e-6, o que faz os indicadores saltarem a
    cada inversão do gás), hD é interpolado na tabela e a fração de bolhas usa o
    módulo da velocidade do gás com um piso de 1 mm/s."""
    xp = bk.xp
    sg = bk.stop_gradient
    vLS_s = xp.abs(inp.vLS) if smooth else inp.vLS
    vGS_s = xp.abs(inp.vGS) if smooth else inp.vGS
    D, th, muL, muG = inp.D, inp.theta, inp.muL, inp.muG
    rhoL, rhoG = inp.rhoL, inp.rhoG
    args_sg = (sg(vGS_s), D, -th, muL, muG, sg(rhoL), sg(rhoG))
    v_est = _equilibrium_vLS(*args_sg, _FI_EST, _LAM_EST, bk=bk)
    v_anu = _equilibrium_vLS(*args_sg, _FI_ANU, _LAM_ANU, bk=bk)
    sel_est = _table_lookup(v_est, sg(vLS_s), True, bk)
    sel_anu = _table_lookup(v_anu, sg(vLS_s), False, bk)
    args = (vGS_s, D, -th, muL, muG, rhoL, rhoG)
    hD_est = _interp_pick(sel_est, v_est, vLS_s, _HD_TEST, _FI_EST, _LAM_EST, args, smooth, bk)
    hD_anu = _interp_pick(sel_anu, v_anu, vLS_s, _HD_TEST2, _FI_ANU, _LAM_ANU, args, smooth, bk)

    beta_est = 2.0 * xp.arccos(1.0 - 2.0 * hD_est)
    aG_est = 1.0 - (beta_est - xp.sin(beta_est)) / (2.0 * xp.pi)
    aG_anu = (1.0 - 2.0 * hD_anu) ** 2
    ub = 1.53 * (inp.sigma * G * xp.abs(inp.rhoL - inp.rhoG) / inp.rhoL**2) ** 0.25 * xp.sin(inp.theta)
    vM = inp.vLS + inp.vGS
    vG = 1.25 * vM + ub
    if smooth:
        aG_bol = xp.clip(xp.abs(inp.vGS) / (xp.abs(vG) + 1e-3), 1e-6, 1 - 1e-6)
    else:
        aG_bol = xp.clip(inp.vGS / xp.where(vG == 0, 1e-30, vG), 1e-6, 1 - 1e-6)
    return hD_est, hD_anu, aG_est, aG_anu, aG_bol


def _froude(rhoL, rhoG, D, vLS, vGS, bk):
    xp = bk.xp
    FL = xp.sqrt(rhoL / (rhoL - rhoG) / 9.8 / D) * xp.abs(vLS)
    FG = xp.sqrt(rhoG / (rhoL - rhoG) / 9.8 / D) * xp.abs(vGS)
    return FL, FG


def identify_regime(inp: ClosureInputs, k_switch=1e6, eq=None, smooth=False, local=False, local_film=False, bk=NUMPY):
    """Identificação do padrão de escoamento (identificaRegimeEstavel.m).

    k_switch controla a suavidade do chaveamento (1e6 no original = quase
    descontínuo). smooth=True é o modo suave (PipeParams.ipe_smooth, para uso
    dinâmico): sigmoides com largura relativa ao valor crítico (k ~ 20), indicador
    de sentido com tanh(vM / V_SOBE) em vez de sign(vM), e os equilíbrios de
    equilibrium_fractions(smooth=True). local=True (PipeParams.ipe_local): os
    critérios usam o ESTADO LOCAL em vez dos equilíbrios de regime permanente --
    o nível/fração reais (inp.hD, inp.alphaG) no lugar de hD_est/aG_est
    (Kelvin-Helmholtz, Jeffreys, trajetória e turbulência avaliados na interface
    que existe) e a fração de bolhas max(aG_bol, alphaG) (um bolsão com 80 % de
    gás não é escoamento de bolhas por ter velocidades superficiais pequenas; os
    mapas de regime permanente só olham as velocidades). Retorna (flags, fatores)
    onde fatores é um dict com os grupos adimensionais e valores críticos.

    Duas correções em relação ao código original, sem efeito quando os
    indicadores são 0/1: (i) o indicador de duto vertical compara o ângulo em
    graus (o original comparava radianos com 90, e nunca era vertical, de modo
    que o critério de Froude do anular nunca era aplicado em risers); (ii) os
    pesos formam uma partição da unidade também com indicadores fracionários
    (o original usava 1 - Bub, com Bub já multiplicado por (1 - Est)(1 - Anu),
    no peso do intermitente, e a soma dos pesos chegava a 1.4 nos empates).

    local_film=True (com local; PipeParams.ipe_local_film, issue 046): o critério de ponte do filme do
    anular usa também a fração de líquido LOCAL, alphaL_filme = max(alphaL de equilíbrio de Pots, alphaL da
    célula): uma célula com mais líquido do que o filme que faz ponte (0.35) não pode ser anular, por
    pequenas que sejam as velocidades superficiais (sem isso a célula do ponto baixo quase cheia de
    líquido, onde o bolsão encontra o líquido, era identificada como anular).
    """
    xp = bk.xp
    if eq is None:
        eq = equilibrium_fractions(inp, smooth=smooth, bk=bk)

    def sig(x1, x2):
        return sigmoid(k_switch, x1, x2, relative=smooth, bk=bk)

    hD_est, hD_anu, aG_est, aG_anu, aG_bol = eq
    if local:
        hD_est = xp.clip(inp.hD, 1e-6, 1 - 1e-6)
        aG_est = xp.clip(inp.alphaG, 1e-6, 1 - 1e-6)
        aG_bol = xp.clip(smooth_max(aG_bol, inp.alphaG, bk=bk), 1e-6, 1 - 1e-6)
        if local_film:
            aG_anu = xp.minimum(aG_anu, xp.clip(inp.alphaG, 1e-6, 1 - 1e-6))
    vLS, vGS = inp.vLS, inp.vGS
    vM = vLS + vGS
    D, eps, th = inp.D, inp.eps, inp.theta
    rhoL, rhoG, muL, sigma = inp.rhoL, inp.rhoG, inp.muL, inp.sigma

    aL_est = 1.0 - aG_est
    aL_anu = 1.0 - aG_anu
    aL_bol = 1.0 - aG_bol

    if smooth:
        # sentido do escoamento: só o sinal de vM importa (cos theta > 0 para |theta| < 90°,
        # e vale 6e-17 no riser vertical, onde anularia o argumento)
        flag_sobe = 0.5 * (1.0 + xp.tanh(vM / V_SOBE))
    else:
        flag_sobe = (xp.sign(xp.cos(th) * (vM + 1e-10)) + 1.0) / 2.0
    flag_desce = 1.0 - flag_sobe
    flag_horizontal = xp.where(xp.abs(th) < (10.0 / 180.0 * xp.pi), 1.0, 0.0)
    flag_vertical = xp.where(xp.abs(xp.rad2deg(xp.abs(th)) - 90.0) < 10.0, 1.0, 0.0)
    flag_naovertical = 1.0 - flag_vertical

    fS_L, ReS_L = fanning(vLS, rhoL, muL, D, eps, bk=bk)
    FL, FG = _froude(rhoL, rhoG, D, vLS, vGS, bk)

    cos_t = xp.cos(th)
    # Kelvin-Helmholtz
    FGast = FG / xp.sqrt(cos_t)
    beta = 2.0 * xp.arccos(1.0 - 2.0 * hD_est)
    FGast_c = xp.sqrt(((1.0 - hD_est) ** 2) * (aG_est**3) * (xp.pi * D / 4.0) / (D * xp.sin(beta / 2.0)))
    flag_naoest1 = sig(FGast, FGast_c)

    # trajetória (descendente)
    NL = bk.safe_sqrt(fS_L * FL**2 / cos_t)
    f_L_est, _ = fanning(vLS / aL_est, rhoL, muL, D, eps, bk=bk)
    fS_L_s = xp.where(fS_L == 0, 1e-30, fS_L)
    NL_c = 1.0 * xp.sqrt(1.0 - hD_est) * aL_est / xp.sqrt(f_L_est / fS_L_s)
    flag_naoest2 = sig(NL, NL_c) * flag_desce

    # Jeffreys (ondas por vento)
    K = FG * xp.sqrt(ReS_L / cos_t)
    K_c = 2.0 * xp.sqrt(aL_est) * aG_est / xp.sqrt(0.01)
    flag_wavy1 = sig(K, K_c)
    # ondas de gravidade
    FL_c = 1.5 * xp.sqrt(hD_est) * aL_est
    flag_wavy2 = sig(FL, FL_c)

    # anular
    EOD = D * xp.sqrt((rhoL - rhoG) * G / sigma)
    FG_c = xp.sqrt(75.0 / (EOD**2) * (xp.sqrt(1.0 + (EOD / 75.0) ** 2 / 0.0065) - 1.0))
    flag_naoanular = sig(aL_anu, 0.35)
    flag_anular = sig(FG, FG_c)

    # bolhas (turbulência)
    T = bk.safe_sqrt(2.0 * fS_L * FL**2 / cos_t)
    betahEst = 2.0 * xp.arccos(1.0 - 2.0 * hD_est)
    T_c = xp.sqrt(2.0 * xp.pi * (aL_est**2 * aG_est) / xp.sin(betahEst / 2.0) / (f_L_est / fS_L_s))
    flag_bubble3 = sig(T, T_c)

    rhoM = aL_bol * rhoL + aG_bol * rhoG
    f_M, _ = fanning(vM, rhoM, muL, D, eps, bk=bk)
    vM_s = xp.where(vM == 0, 1e-30, vM)
    # mistura parada: dmax infinito sem formar o produto (tangente 0 * inf = NaN em AD)
    arg = xp.abs(2.0 * f_M * vM**3 / D)
    small = arg == 0
    dmax = xp.where(
        small,
        xp.inf,
        (0.725 + 4.15 * bk.safe_sqrt(xp.abs(vGS / vM_s)))
        * (sigma / rhoL) ** (3.0 / 5.0)
        * xp.where(small, 1.0, arg) ** (-2.0 / 5.0),
    )
    ddef = 2.0 * xp.sqrt(0.4 * sigma / xp.abs(rhoL - rhoG) / G)
    dmigr = (3.0 / 8.0) * rhoL / xp.abs(rhoL - rhoG) * f_M * vM**2 / G / cos_t

    flag_intermitente1 = sig(aG_bol, 0.52)
    flag_bubble1 = sig(0.25, aG_bol)
    flag_checkbolhas = (1.0 - flag_bubble1) * (1.0 - flag_intermitente1)
    flag_bubble2 = sig(ddef, dmax) * sig(dmigr, dmax) * flag_checkbolhas

    Est = (1.0 - flag_naoest1) * flag_sobe + (1.0 - flag_naoest2) * (1.0 - flag_sobe)
    p_anu = (1.0 - flag_naoanular) * (flag_naovertical + flag_anular * flag_vertical)
    if local:
        # estado local: estratificado é impossível num duto vertical (os critérios de
        # K-H/trajetória degeneram com cos theta -> 0) e uma fração de gás > 0.9 é um
        # filme de parede (anular) qualquer que seja a velocidade do gás
        Est = Est * flag_naovertical
        a_hi = sig(inp.alphaG, 0.9)
        p_anu = 1.0 - (1.0 - p_anu) * (1.0 - a_hi)
    EsW = Est * (flag_wavy1 * flag_sobe + flag_wavy2 * (1.0 - flag_sobe))
    EsS = Est * (1.0 - EsW)
    Anu = (1.0 - Est) * p_anu
    p_bub = (
        flag_bubble1 * (1.0 - flag_horizontal)
        + flag_bubble2 * (1.0 - flag_horizontal) * (1.0 - flag_intermitente1) * (1.0 - flag_bubble1)
        + flag_bubble3 * flag_horizontal
    )
    Bub = (1.0 - Est) * (1.0 - Anu) * p_bub
    Int = (1.0 - Est) * (1.0 - Anu) * (1.0 - p_bub)
    flags = RegimeFlags(Est=Est, EsS=EsS, EsW=EsW, Anu=Anu, Bub=Bub, Int=Int)
    factors = dict(
        FL=FL,
        FG=FG,
        NL=NL,
        K=K,
        T=T,
        FGcosteta=FGast,
        FL_c=FL_c,
        NL_c=NL_c,
        K_c=K_c,
        T_c=T_c,
        FG_c=FG_c,
        FGast_c=FGast_c,
    )
    return flags, factors


def blend_closures(inp: ClosureInputs, flags: RegimeFlags, threshold=0.1, used=None, bk=NUMPY) -> ClosureTerms:
    """Soma ponderada dos fechamentos pelos indicadores de padrão.

    Cada fechamento é avaliado apenas nos pontos onde o indicador > threshold (ou é
    NaN, como em filtra_entradas.m / montaRegime.m), evitando NaN de estados
    extremos; `used` (conjunto estático) restringe os fechamentos avaliados."""
    xp = bk.xp
    n = inp.size
    out = {k: xp.zeros(n) for k in ClosureTerms.zeros(1).names()}
    for nm in ("Est", "Anu", "Int", "Bub"):
        if used is not None and nm not in used:
            continue
        flag = getattr(flags, nm)
        mask = (flag > threshold) | xp.isnan(flag)
        if not bk.is_jax and not bool(xp.any(mask)):
            continue
        S = bk.masked_apply(lambda sub, _f=CLOSURE_FUNCS[nm]: _f(sub, bk=bk), mask, inp, ClosureTerms)
        for k in out:
            out[k] = out[k] + xp.where(mask, flag * getattr(S, k), 0.0)
    for k in NAN_TO_ZERO:
        out[k] = xp.where(xp.isfinite(out[k]), out[k], 0.0)
    return ClosureTerms(**out)
