"""Identificação do padrão de escoamento (IPE) e ponderação dos fechamentos.

Porta determinaAlfaEstavel.m (modelos de equilíbrio de Taitel e Pots),
identificaRegimeEstavel.m (critérios de estabilidade com chaveamento
sigmoidal) e montaRegime.m/filtra_entradas.m.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

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


def sigmoid(k, x1, x2, relative=False):
    """1/(1+exp(-k (x1-x2))): ~1 se x1 > x2, ~0 se x1 < x2.

    relative=True: o argumento é k (x1 - x2)/|x2| -- a largura da transição passa a
    ser uma fração do valor crítico (~2/k), igual para critérios de escalas muito
    diferentes (FG* ~ 0.02, K ~ 1..10), em vez de absoluta como na tese."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    d = x1 - x2
    if relative:
        d = d / (np.abs(x2) + 1e-300)
    arg = np.clip(-k * d, -700, 700)
    return 1.0 / (1.0 + np.exp(arg))


def fixed_regime(regime, n):
    z = np.zeros(n)
    o = np.ones(n)
    fl = RegimeFlags(Est=z.copy(), EsS=z.copy(), EsW=z.copy(), Anu=z.copy(), Bub=z.copy(), Int=z.copy())
    if regime == REGIME_ANNULAR:
        fl.Anu = o
    elif regime == REGIME_BUBBLE:
        fl.Bub = o
    elif regime == REGIME_INTERMITTENT:
        fl.Int = o
    elif regime == REGIME_STRAT_SMOOTH:
        fl.Est, fl.EsS = o, o.copy()
    elif regime == REGIME_STRAT_WAVY:
        fl.Est, fl.EsW = o, o.copy()
    else:
        raise ValueError(f"regime fixo inválido: {regime}")
    return fl


# ----------------------------------------------------------------------------
# Modelos de equilíbrio (Taitel & Dukler 1976 / Pots) para hD de equilíbrio
# ----------------------------------------------------------------------------

def _lockhart_vLS(vGS, hD, D, eta, muL, muG, rhoL, rhoG, geometry, CL=0.046, CG=0.046, n=0.2, m=0.2):
    """Velocidade superficial de líquido de equilíbrio para dado hD e vGS.

    geometry = 'stratified' (Taitel) ou 'annular' (Pots).
    Entradas vetorizadas por broadcasting (vGS[:, None] x hD[None, :]).
    """
    Re_GS = D * np.abs(vGS) * rhoG / muG
    Re_GS = np.where(Re_GS == 0, 1e-30, Re_GS)
    f_GS = CG * Re_GS ** (-m)
    tau_GS = f_GS * rhoG * vGS * np.abs(vGS) / 2.0
    tau_GS_safe = np.where(tau_GS == 0, 1e-30, tau_GS)
    Y = (rhoL - rhoG) * G * np.sin(eta) / (4.0 / D * tau_GS_safe)

    At = np.pi / 4.0
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
    with np.errstate(divide="ignore", invalid="ignore"):
        uLt = At / ALt
        uGt = At / AGt
        DGt = 4.0 * AGt / (SGt + SIt)
        DLt = 4.0 * ALt / SLt
        FI = (SGt / AGt + SIt / ALt + SIt / AGt) * (uGt * DGt) ** (-m) * uGt**2
        LAMBDA = (uLt * DLt) ** (-n) * uLt**2 * SLt / ALt
        X2 = (4.0 * Y + FI) / LAMBDA
        X2 = np.where(X2 < 0, np.nan, X2)
        tau_LS = X2 * tau_GS
        vLS = (tau_LS / CL) ** (1.0 / (2.0 - n)) * (D * rhoL / muL) ** (n / (2.0 - n)) * (rhoL / 2.0) ** (1.0 / (n - 2.0))
    return vLS


_AUX = (10.0 ** np.arange(-3, 0.0001, 0.01)) / 2.0
_HD_TEST = np.concatenate([_AUX, 1.0 - _AUX[::-1]])
_HD_TEST2 = np.arange(0.01, 0.9901, 0.01)
V_SOBE = 0.02       # m/s: largura da regularização do indicador de sentido (vM > 0) no modo suave
EPS_SMAX = 0.02     # largura do máximo suave entre a fração de bolhas de equilíbrio e a fração local (modo local)


def smooth_max(a, b, eps=EPS_SMAX):
    """max(a, b) com derivada contínua: 0.5 (a + b + sqrt((a - b)^2 + eps^2))."""
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + eps**2))


def _table_pick(v, hd, vLS, smooth):
    """hD da tabela (candidatos v, hd finitos, em ordem) cujo vLS de equilíbrio é o mais
    próximo de vLS; no modo suave, interpolação linear entre os dois candidatos que
    enquadram vLS (hD contínuo no estado em vez de constante por partes)."""
    j = int(np.argmin(np.abs(v - vLS)))
    if not smooth or v.size < 2:
        return hd[j]
    if j + 1 < v.size and (v[j] - vLS) * (v[j + 1] - vLS) <= 0:
        a, b = j, j + 1
    elif j > 0 and (v[j - 1] - vLS) * (v[j] - vLS) <= 0:
        a, b = j - 1, j
    else:
        return hd[j]
    dv = v[b] - v[a]
    if dv == 0:
        return hd[j]
    return hd[a] + (vLS - v[a]) / dv * (hd[b] - hd[a])


def equilibrium_fractions(inp: ClosureInputs, smooth=False):
    """Frações de área de equilíbrio para estratificado (Taitel), anular (Pots)
    e bolhas (Zuber). Retorna (hD_est, hD_anu, aG_est, aG_anu, aG_bol).

    smooth=True (modo suave da identificação): os equilíbrios são avaliados com os
    módulos das velocidades superficiais (co-corrente: com vGS < 0 os modelos dão
    NaN e o código original cai em hD = 1e-6, o que faz os indicadores saltarem a
    cada inversão do gás), hD é interpolado na tabela e a fração de bolhas usa o
    módulo da velocidade do gás com um piso de 1 mm/s."""
    vLS_s = np.abs(inp.vLS) if smooth else inp.vLS
    vGS_s = np.abs(inp.vGS) if smooth else inp.vGS
    vLS = vLS_s[:, None]
    vGS = vGS_s[:, None]
    D = inp.D[:, None]
    th = inp.theta[:, None]
    rhoL, rhoG = inp.rhoL[:, None], inp.rhoG[:, None]
    muL, muG = inp.muL[:, None], inp.muG[:, None]

    vl_est = _lockhart_vLS(vGS, _HD_TEST[None, :], D, -th, muL, muG, rhoL, rhoG, "stratified")
    vl_anu = _lockhart_vLS(vGS, _HD_TEST2[None, :], D, -th, muL, muG, rhoL, rhoG, "annular")

    n = inp.size
    hD_est = np.full(n, 1e-6)
    hD_anu = np.full(n, 1e-6)
    for i in range(n):
        v = vl_est[i]
        hd = _HD_TEST
        nan = np.isnan(v)
        if nan.any():
            last = np.flatnonzero(nan)[-1]
            v, hd = v[last:], hd[last:]
            keep = ~np.isnan(v)
            v, hd = v[keep], hd[keep]
        if v.size:
            hD_est[i] = _table_pick(np.real(v), hd, vLS_s[i], smooth)
        v2 = vl_anu[i]
        keep = ~np.isnan(v2)
        if keep.any():
            hD_anu[i] = _table_pick(v2[keep], _HD_TEST2[keep], vLS_s[i], smooth)

    beta_est = 2.0 * np.arccos(1.0 - 2.0 * hD_est)
    aG_est = 1.0 - (beta_est - np.sin(beta_est)) / (2.0 * np.pi)
    aG_anu = (1.0 - 2.0 * hD_anu) ** 2
    ub = 1.53 * (inp.sigma * G * np.abs(inp.rhoL - inp.rhoG) / inp.rhoL**2) ** 0.25 * np.sin(inp.theta)
    vM = inp.vLS + inp.vGS
    vG = 1.25 * vM + ub
    if smooth:
        aG_bol = np.clip(np.abs(inp.vGS) / (np.abs(vG) + 1e-3), 1e-6, 1 - 1e-6)
    else:
        aG_bol = np.clip(inp.vGS / np.where(vG == 0, 1e-30, vG), 1e-6, 1 - 1e-6)
    return hD_est, hD_anu, aG_est, aG_anu, aG_bol


def _froude(rhoL, rhoG, D, vLS, vGS):
    FL = np.sqrt(rhoL / (rhoL - rhoG) / 9.8 / D) * np.abs(vLS)
    FG = np.sqrt(rhoG / (rhoL - rhoG) / 9.8 / D) * np.abs(vGS)
    return FL, FG


def identify_regime(inp: ClosureInputs, k_switch=1e6, eq=None, smooth=False, local=False):
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
    """
    if eq is None:
        eq = equilibrium_fractions(inp, smooth=smooth)

    def sig(x1, x2):
        return sigmoid(k_switch, x1, x2, relative=smooth)
    hD_est, hD_anu, aG_est, aG_anu, aG_bol = eq
    if local:
        hD_est = np.clip(inp.hD, 1e-6, 1 - 1e-6)
        aG_est = np.clip(inp.alphaG, 1e-6, 1 - 1e-6)
        aG_bol = np.clip(smooth_max(aG_bol, inp.alphaG), 1e-6, 1 - 1e-6)
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
        flag_sobe = 0.5 * (1.0 + np.tanh(vM / V_SOBE))
    else:
        flag_sobe = (np.sign(np.cos(th) * (vM + 1e-10)) + 1.0) / 2.0
    flag_desce = 1.0 - flag_sobe
    flag_horizontal = (np.abs(th) < (10.0 / 180.0 * np.pi)).astype(float)
    flag_vertical = (np.abs(np.rad2deg(np.abs(th)) - 90.0) < 10.0).astype(float)
    flag_naovertical = 1.0 - flag_vertical

    fS_L, ReS_L = fanning(vLS, rhoL, muL, D, eps)
    FL, FG = _froude(rhoL, rhoG, D, vLS, vGS)

    cos_t = np.cos(th)
    # Kelvin-Helmholtz
    FGast = FG / np.sqrt(cos_t)
    beta = 2.0 * np.arccos(1.0 - 2.0 * hD_est)
    FGast_c = np.sqrt(((1.0 - hD_est) ** 2) * (aG_est**3) * (np.pi * D / 4.0) / (D * np.sin(beta / 2.0)))
    flag_naoest1 = sig(FGast, FGast_c)

    # trajetória (descendente)
    NL = np.sqrt(fS_L * FL**2 / cos_t)
    f_L_est, _ = fanning(vLS / aL_est, rhoL, muL, D, eps)
    fS_L_s = np.where(fS_L == 0, 1e-30, fS_L)
    NL_c = 1.0 * np.sqrt(1.0 - hD_est) * aL_est / np.sqrt(f_L_est / fS_L_s)
    flag_naoest2 = sig(NL, NL_c) * flag_desce

    # Jeffreys (ondas por vento)
    K = FG * np.sqrt(ReS_L / cos_t)
    K_c = 2.0 * np.sqrt(aL_est) * aG_est / np.sqrt(0.01)
    flag_wavy1 = sig(K, K_c)
    # ondas de gravidade
    FL_c = 1.5 * np.sqrt(hD_est) * aL_est
    flag_wavy2 = sig(FL, FL_c)

    # anular
    EOD = D * np.sqrt((rhoL - rhoG) * G / sigma)
    FG_c = np.sqrt(75.0 / (EOD**2) * (np.sqrt(1.0 + (EOD / 75.0) ** 2 / 0.0065) - 1.0))
    flag_naoanular = sig(aL_anu, 0.35)
    flag_anular = sig(FG, FG_c)

    # bolhas (turbulência)
    T = np.sqrt(2.0 * fS_L * FL**2 / cos_t)
    betahEst = 2.0 * np.arccos(1.0 - 2.0 * hD_est)
    T_c = np.sqrt(2.0 * np.pi * (aL_est**2 * aG_est) / np.sin(betahEst / 2.0) / (f_L_est / fS_L_s))
    flag_bubble3 = sig(T, T_c)

    rhoM = aL_bol * rhoL + aG_bol * rhoG
    f_M, _ = fanning(vM, rhoM, muL, D, eps)
    vM_s = np.where(vM == 0, 1e-30, vM)
    dmax = (0.725 + 4.15 * np.sqrt(np.abs(vGS / vM_s))) * (sigma / rhoL) ** (3.0 / 5.0) * \
        np.abs(2.0 * f_M * vM**3 / D) ** (-2.0 / 5.0)
    ddef = 2.0 * np.sqrt(0.4 * sigma / np.abs(rhoL - rhoG) / G)
    dmigr = (3.0 / 8.0) * rhoL / np.abs(rhoL - rhoG) * f_M * vM**2 / G / cos_t

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
    p_bub = (flag_bubble1 * (1.0 - flag_horizontal)
             + flag_bubble2 * (1.0 - flag_horizontal) * (1.0 - flag_intermitente1) * (1.0 - flag_bubble1)
             + flag_bubble3 * flag_horizontal)
    Bub = (1.0 - Est) * (1.0 - Anu) * p_bub
    Int = (1.0 - Est) * (1.0 - Anu) * (1.0 - p_bub)
    flags = RegimeFlags(Est=Est, EsS=EsS, EsW=EsW, Anu=Anu, Bub=Bub, Int=Int)
    factors = dict(FL=FL, FG=FG, NL=NL, K=K, T=T, FGcosteta=FGast,
                   FL_c=FL_c, NL_c=NL_c, K_c=K_c, T_c=T_c, FG_c=FG_c, FGast_c=FGast_c)
    return flags, factors


def blend_closures(inp: ClosureInputs, flags: RegimeFlags, threshold=0.1) -> ClosureTerms:
    """Soma ponderada dos fechamentos pelos indicadores de padrão.

    Cada fechamento é avaliado apenas nos pontos onde o indicador > threshold
    (filtra_entradas.m / montaRegime.m), evitando NaN de estados extremos.
    """
    n = inp.size
    out = ClosureTerms.zeros(n)
    names = out.names()
    for flag, func in ((flags.Est, stratified), (flags.Anu, annular),
                       (flags.Int, intermittent), (flags.Bub, bubble)):
        mask = (flag > threshold) | np.isnan(flag)
        if not mask.any():
            continue
        sub = func(inp.subset(mask))
        w = flag[mask]
        for nm in names:
            arr = getattr(out, nm)
            arr[mask] = arr[mask] + w * getattr(sub, nm)
    for nm in ("ddPLdalphaG", "ddPGdalphaG", "dPL", "dPG", "dPI", "ddPLdP", "ddPGdP"):
        arr = getattr(out, nm)
        arr[~np.isfinite(arr)] = 0.0
    return out
