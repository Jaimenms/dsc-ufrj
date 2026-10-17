"""Fechamento para padrão intermitente (modelo_intermitente.m).

Unidade de slug (comprimento lu) composta por um pistão de líquido aerado
(comprimento ls, modelado como bolhas) e uma bolha de Taylor com filme
estratificado (comprimento lf). Os termos de fechamento são ponderados
pelos comprimentos.
"""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY
from ..friction import fanning
from ..geometry import stratified_level
from .bubble import SLIP_MAX, bubble
from .common import ClosureInputs, ClosureTerms, G_CLOSURE
from .stratified import stratified


def slug_frequency(uLS, uGS, rhoL, rhoG, muL, D, sigma, eps, bk=NUMPY):
    """Frequência de golfada da tese (eq. 5.83, atribuída lá a Manolis 1995; frequenciaSlugs.m):

        w_s = (u_GS/D) [0,0323/2 (u_LS/u_GS) (rhoL/Δrho)^1/2 f_L^-1/2 Eo^-0,2]^n,  n = N_f/(260 + 0,85 N_f),

    com Eo = D (g Δrho/sigma)^1/2, N_f = D^1,5 (rhoL Δrho g)^1/2/muL e f_L o fator de Fanning do líquido a u_LS.
    O texto da tese escreve n = 0,85 + 260/N_f; o código MATLAB usa o inverso, que é o que se mantém (com N_f grande
    n tende a 1/0,85 ≈ 1,18, o expoente de Gregory e Scott). Não é a correlação de Zabaras (2000), que está em
    `slug_frequency_zabaras` (opção `closures.intermittent.slug_frequency`, issue 049)."""
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


def slug_frequency_zabaras(uLS, uGS, D, theta, bk=NUMPY):
    """Frequência de golfada de Zabaras (2000): a de Gregory e Scott (1969) com o fator de inclinação,

        f_s = 0,0226 [u_LS/(g D) (19,75/u_M + u_M)]^1,2 [0,836 + 2,75 (sin θ)^1/4]   (SI; 19,75 m²/s²),

    com u_M = u_LS + u_GS. O fator foi ajustado em dutos de 0 a 11° ascendentes; em dutos descendentes usa-se o
    horizontal (sin θ truncado a 0). u_M leva 1e-6 m/s para o duto parado."""
    xp = bk.xp
    uM = uLS + uGS + 1e-6
    incl = 0.836 + 2.75 * bk.safe_pow(xp.maximum(xp.sin(theta), 0.0), 0.25)
    return 0.0226 * bk.safe_pow(uLS / (G_CLOSURE * D) * (19.75 / uM + uM), 1.2) * incl


def slug_liquid_fraction(D, eps, theta, sigma, rhoL, rhoG, muL, uLS, uGS, bk=NUMPY):
    """Fração de líquido no pistão (Barnea & Brauner 1985)."""
    xp = bk.xp
    uS = uLS + uGS
    ReS = D * xp.abs(uS) * rhoL / muL
    # piso 1e-12 (era 1e-30 só em ReS = 0): com 1e-30 fS = inf e fS |uS|^3 = inf * 0 = NaN no duto parado (issue 057)
    ReS = xp.where(ReS < 1e-12, 1e-12, ReS)
    A = (2.457 * xp.log(1.0 / (0.27 * eps / D + (7.0 / ReS) ** 0.9))) ** 16
    B = (37530.0 / ReS) ** 16
    fS = 2.0 * ((8.0 / ReS) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    alphas = (
        0.058
        * (
            2.0
            * xp.sqrt(0.4 * sigma / G_CLOSURE / (rhoL - rhoG))
            * bk.safe_pow(2.0 * fS / D * xp.abs(uS) ** 3, 2.0 / 5.0)
            * (rhoL / sigma) ** (3.0 / 5.0)
            - 0.725
        )
        ** 2
    )
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


def slug_unit(aG, alphas, ls, lu, consistent=False, bk=NUMPY):
    """Geometria da unidade de golfada: (alphas, ls, lf, lu, alphaf) a partir da fração de gás da célula, da do
    pistão, do comprimento do pistão e do da unidade (v_t/w_s).

    consistent=False (tese, modelo_intermitente.m): ls = min(ls, lu), lf = max(lu - ls, 1e-3 ls) e a fração de gás
    do filme do balanço de massa cortada a [1e-6, ALPHA_FILM_MAX]; quando a célula tem mais gás do que a unidade
    consegue guardar, o corte quebra o balanço (e as velocidades locais deixam de reproduzir as vazões da célula).

    consistent=True (issue 050): alphaG lu = alphas ls + alphaf lf sempre, com alphas <= alphaG (o pistão não guarda
    mais gás do que a célula) e alphaf <= ALPHA_FILM_MAX (o pistão encurta até o filme guardar o resto,
    ls <= lu (ALPHA_FILM_MAX - alphaG)/(ALPHA_FILM_MAX - alphas)). No limite rico em líquido a unidade tem o estado
    da célula; acima de ALPHA_FILM_MAX é só filme."""
    xp = bk.xp
    if not consistent:
        ls = xp.minimum(ls, lu)
        lf = lu - ls
        lf = xp.maximum(lf, ls * 1e-3)
        lu = lf + ls
        alphaf = lu / lf * aG - ls / lf * alphas
        # salvaguarda: fração de gás do filme limitada (no código original o limite
        # 1-1e-6 produz velocidades de filme ~1e6 m/s e forças de atrito ~1e11 N/m
        # quando a célula unitária é inconsistente com alphaG elevado)
        alphaf = xp.clip(alphaf, 1e-6, ALPHA_FILM_MAX)
        return alphas, ls, lf, lu, alphaf
    alphas = xp.minimum(alphas, aG)
    ls_max = lu * (ALPHA_FILM_MAX - aG) / (ALPHA_FILM_MAX - alphas)
    ls = xp.clip(xp.minimum(ls, ls_max), 0.0, lu)
    lf = xp.maximum(lu - ls, 1e-6 * lu)
    lu = ls + lf
    alphaf = xp.clip(aG + ls / lf * (aG - alphas), 1e-6, 1.0 - 1e-6)
    return alphas, ls, lf, lu, alphaf


FILM_BLEND_DEG = 10.0  # ascendentes de 0 a 10°: do filme estratificado (equilíbrio) ao filme em queda (Brotz)
BROTZ = 9.916  # velocidade terminal do filme em queda livre, u_f = -9.916 [g D sin θ (1 - α_TB^1/2)]^1/2
R_DEGEN = 0.05  # a perda de aceleração anula-se (smoothstep) a menos de 5 % das unidades degeneradas
SLUG_FILM_GAP = 0.1  # o pistão é pelo menos isto menos arejado do que o filme: r varia continuamente entre 1 e 0
SLUG_SLIP_MAX = (
    0.3  # m/s (~velocidade terminal): escorregamento máximo das bolhas no corpo do pistão na força de bolhas
)
EPS_ALPHA = 0.02  # largura dos máximos e mínimos suaves nas frações da unidade (quinas fazem o Newton divergir)
EPS_R = 0.01  # idem para a fração de pistão
V_SIGN = 0.05  # m/s: largura do sentido suave das velocidades superficiais na unidade com o filme de equilíbrio


def _smax(a, b, eps, bk=NUMPY):
    """max(a, b) com derivada contínua (>= max; iguala-o a menos de eps/2 no empate)."""
    return 0.5 * (a + b + bk.xp.sqrt((a - b) ** 2 + eps**2))


def _smin(a, b, eps, bk=NUMPY):
    return 0.5 * (a + b - bk.xp.sqrt((a - b) ** 2 + eps**2))


def _smoothstep(x, bk=NUMPY):
    u = bk.xp.clip(x, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def falling_film_fraction(q, ut, D, theta, bk=NUMPY):
    """Fração de gás da bolha de Taylor num duto ascendente (Taitel, Barnea e Dukler 1980): a cinemática da unidade,
    (v_t - u_f)(1 - α) = q com q = (v_t - u_L)(1 - α_s) = v_t (1 - α_G) - u_LS, e o filme em queda livre de Brotz com a
    gravidade ao longo do duto, u_f = -9,916 [g D sin θ (1 - α^1/2)]^1/2. F(α) é decrescente em (0, 1): bissecção sem
    gradiente e um passo de Newton no fim, que dá a derivada exata pelo teorema da função implícita."""
    xp = bk.xp
    c = BROTZ * xp.sqrt(G_CLOSURE * D * xp.maximum(xp.sin(theta), 0.0))
    q = xp.maximum(q, 1e-9)

    def F(a):
        return (ut + c * bk.safe_sqrt(1.0 - xp.sqrt(a))) * (1.0 - a) - q

    qs, uts, cs = (bk.stop_gradient(v) for v in (q, ut, c))

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        pos = (uts + cs * bk.safe_sqrt(1.0 - xp.sqrt(mid))) * (1.0 - mid) - qs > 0.0
        return xp.where(pos, mid, lo), xp.where(pos, hi, mid)

    lo, hi = bk.fori_loop(0, 48, body, (0.0 * qs, 0.0 * qs + 1.0))
    a = xp.clip(0.5 * (lo + hi), 1e-6, 1.0 - 1e-6)
    s = xp.sqrt(1.0 - xp.sqrt(a))
    dF = -c * (1.0 - a) / (4.0 * s * xp.sqrt(a)) - (ut + c * s)
    return xp.clip(a - F(a) / dF, 1e-6, 1.0 - 1e-6)


def film_unit(aG, alphas0, uLS, uGS, ut, theta, D, aG_strat, bk=NUMPY):
    """Unidade de golfada com o filme de equilíbrio (issue 051). O gás do pistão é o de Barnea e Brauner limitado ao da
    célula; o do filme vem da física do filme: equilíbrio estratificado de Taitel e Dukler nos troços horizontais e
    descendentes, bolha de Taylor com filme em queda (`falling_film_fraction`) nos ascendentes, com transição suave
    entre 0 e 10°, e nunca menos do que o da célula. A fração de pistão r = l_s/l_u sai do balanço de massa,
    α_G = r α_s + (1 - r) α_f, e as velocidades locais das relações de massa da unidade, que devolvem as vazões da
    célula. Devolve (α_s, α_f, r, u_L, u_b, u_f, u_G, a), com a = (1 - α_s)(α_f - α_s)(v_t - u_L)²/(1 - α_f) o fator
    da perda de aceleração do filme na frente do pistão (Taitel e Barnea 1990: Δp = ρ_L a por unidade), anulado
    suavemente quando a unidade degenera (r a menos de R_DEGEN de 0 ou de 1)."""
    xp = bk.xp
    a_tb = falling_film_fraction(ut * (1.0 - aG) - uLS, ut, D, theta, bk=bk)
    ok = xp.isfinite(aG_strat)
    a_st = xp.where(ok, xp.where(ok, aG_strat, 0.5), aG)
    w_up = _smoothstep(theta / np.deg2rad(FILM_BLEND_DEG), bk)
    # o filme de equilíbrio nunca passa de ALPHA_FILM_MAX (acima disso é anular, coberto pelos pesos dos padrões):
    # sem o limite, um bolsão quase parado (q -> 0) dá alpha_tb -> 1, R_f -> 0 e velocidades de filme ~1/R_f;
    # uma célula mais rica em gás do que o filme fica só filme, no estado da célula
    a_model = _smin((1.0 - w_up) * a_st + w_up * a_tb, ALPHA_FILM_MAX, EPS_ALPHA, bk)
    alphaf = xp.clip(_smax(a_model, aG, EPS_ALPHA, bk), 1e-6, 1.0 - 1e-6)
    # o pistão não guarda mais gás do que a célula nem chega à aeração do filme: sem a folga, quando a aeração de
    # Barnea e Brauner iguala o filme de equilíbrio (ponto triplo) r salta de 0 a 1 e a célula passa de só filme a
    # só pistão com o escorregamento inteiro no fechamento de bolhas (forças de 1e5 N/m)
    alphas = _smax(_smin(_smin(alphas0, aG, EPS_ALPHA, bk), alphaf - SLUG_FILM_GAP, EPS_ALPHA, bk), 0.0, EPS_ALPHA, bk)
    alphas = xp.clip(alphas, 1e-6, 1.0 - 1e-6)
    Rs = 1.0 - alphas
    uL = (uLS + ut * (aG - alphas)) / Rs
    ub = (uGS - ut * (aG - alphas)) / alphas
    r = _smin(_smax((alphaf - aG) / xp.maximum(alphaf - alphas, 1e-9), 0.0, EPS_R, bk), 1.0, EPS_R, bk)
    r = xp.clip(r, 0.0, 1.0)
    Rf = 1.0 - alphaf
    uf = ut - (ut - uL) * Rs / Rf
    uG = ut - (ut - ub) * alphas / alphaf
    acc = Rs * (alphaf - alphas) * (ut - uL) ** 2 / Rf
    acc = acc * _smoothstep(r / R_DEGEN, bk) * _smoothstep((1.0 - r) / R_DEGEN, bk)
    return alphas, alphaf, r, uL, ub, uf, uG, acc


def intermittent(inp: ClosureInputs, bk=NUMPY) -> ClosureTerms:
    xp = bk.xp
    aG = inp.alphaG
    # correlações da unidade de slug formuladas para escoamento co-corrente;
    # para refluxo usam-se os módulos (as tensões conservam o sinal via v|v|)
    if inp.slug_film is not None:
        # módulos e sentidos suaves (issue 051): com |v| e sign(v) a unidade salta quando o líquido do riser
        # oscila em torno de velocidade nula, o que faz o Newton divergir
        uLS = xp.sqrt(inp.vLS**2 + V_SIGN**2)
        uGS = xp.sqrt(inp.vGS**2 + V_SIGN**2)
    else:
        uLS, uGS = xp.abs(inp.vLS), xp.abs(inp.vGS)
    D = inp.D
    rhoL, rhoG = inp.rhoL, inp.rhoG

    Rs = slug_liquid_fraction(D, inp.eps, inp.theta, inp.sigma, rhoL, rhoG, inp.muL, uLS, uGS, bk=bk)
    alphas = 1.0 - Rs
    ls = slug_length(D, bk=bk)
    if inp.slug_frequency is None:
        wS = slug_frequency(uLS, uGS, rhoL, rhoG, inp.muL, D, inp.sigma, inp.eps, bk=bk)
    else:
        wS = slug_frequency_zabaras(uLS, uGS, D, inp.theta, bk=bk)
    ut = translational_velocity(D, inp.theta, uLS, uGS, bk=bk)

    acc = None
    if inp.slug_film is not None:
        # unidade com o filme de equilíbrio (issue 051): pesos r e 1 - r; a frequência entra na perda de aceleração
        alphas, alphaf, ls, uL, ub, uf, uG, acc = film_unit(
            aG, alphas, uLS, uGS, ut, inp.theta, D, inp.aG_film_eq, bk=bk
        )
        Rs, Rf = 1.0 - alphas, 1.0 - alphaf
        lf = 1.0 - ls
        lu = ls + lf
    else:
        lu = ut / xp.where(wS == 0, 1e-30, wS)
        alphas, ls, lf, lu, alphaf = slug_unit(aG, alphas, ls, lu, consistent=inp.slug_unit is not None, bk=bk)
        if inp.slug_unit is not None:
            Rs = 1.0 - alphas  # na da tese Rs fica intacto (bit a bit)
        Rf = 1.0 - alphaf

        uL = (uLS + ut * (aG - alphas)) / Rs
        ub = (uGS - ut * (aG - alphas)) / xp.where(alphas == 0, 1e-30, alphas)
        uf = ut - (ut - uL) * Rs / Rf
        uG = ut - (ut - ub) * alphas / alphaf
    # limita velocidades locais a um múltiplo das velocidades superficiais
    vmax = VEL_LIMIT_FACTOR * (xp.abs(ut) + uLS + uGS) + 1e-3
    uL, ub, uf, uG = (xp.clip(v, -vmax, vmax) for v in (uL, ub, uf, uG))
    # recupera o sentido do escoamento de cada fase
    if inp.slug_film is not None:
        sL = xp.tanh(inp.vLS / V_SIGN)
        sG = xp.tanh(inp.vGS / V_SIGN)
    else:
        sL = xp.where(inp.vLS < 0, -1.0, 1.0)
        sG = xp.where(inp.vGS < 0, -1.0, 1.0)
    uL, uf = sL * uL, sL * uf
    ub, uG = sG * ub, sG * uG

    Area = xp.pi * D**2 / 4.0
    if inp.slug_film is not None:
        # no corpo do pistão as bolhas dispersas escorregam pouco; um escorregamento cinemático maior é gás da
        # bolha de Taylor, que o filme já representa: limita-o na força de bolhas (as vazões são os estados)
        ub = uL + SLUG_SLIP_MAX * xp.tanh((ub - uL) / SLUG_SLIP_MAX)
    inp_s = inp.replace(
        alphaG=1.0 - Rs,
        alphaL=Rs,
        vL=uL,
        vLS=uL * Rs,
        vG=ub,
        vGS=ub * (1.0 - Rs),
        AreaL=Area * Rs,
        AreaG=Area * (1.0 - Rs),
    )
    Ss = bubble(inp_s, bk=bk)

    hD, beta = stratified_level(1.0 - Rf, bk=bk)
    inp_f = inp.replace(
        alphaG=1.0 - Rf,
        alphaL=Rf,
        vL=uf,
        vLS=uf * Rf,
        vG=uG,
        vGS=uG * (1.0 - Rf),
        AreaL=Area * Rf,
        AreaG=Area * (1.0 - Rf),
        beta=beta,
        hD=hD,
    )
    Sf = stratified(inp_f, bk=bk)

    def mix(a, b):
        return (a * ls + b * lf) / lu

    GammaL = mix(Ss.GammaL, Sf.GammaL)
    dPL, dPG = mix(Ss.dPL, Sf.dPL), mix(Ss.dPG, Sf.dPG)
    ddPLdP, ddPGdP = mix(Ss.ddPLdP, Sf.ddPLdP), mix(Ss.ddPGdP, Sf.ddPGdP)
    if acc is not None:
        # perda de aceleração do filme apanhado pela frente do pistão, uma vez por unidade (1/l_u = w_s/v_t), contra
        # o sentido do líquido
        GammaL = GammaL + sL * Area * rhoL * acc * wS / xp.maximum(ut, 1e-3)
        # pressão interfacial de Stuhmiller com o escorregamento REAL da célula e peso 1 (é a pressão interfacial da
        # mistura, não a do corpo do pistão): no riser o filme estratificado tem cos θ = 0 (sem gradiente de nível)
        # e, pesado por r e com o escorregamento limitado, o termo do pistão não bastava — o modelo ficava mal posto
        # (famílias de ondas instáveis de 0,3–2,5 s no riser) e o BDF em passos de ms
        slip_c = xp.clip(inp.vG - inp.vL, -SLIP_MAX, SLIP_MAX)
        dPL_c = 0.25 * rhoL * slip_c**2
        dPL = dPL_c + lf * Sf.dPL
        dPG = lf * Sf.dPG
        ddPLdP = dPL_c / rhoL * inp.CL + lf * Sf.ddPLdP
        ddPGdP = lf * Sf.ddPGdP

    return ClosureTerms(
        GammaL=GammaL,
        GammaG=mix(Ss.GammaG, Sf.GammaG),
        dPL=dPL,
        dPG=dPG,
        ddPLdP=ddPLdP,
        ddPGdP=ddPGdP,
        ddPLdalphaG=mix(Ss.ddPLdalphaG, Sf.ddPLdalphaG),
        ddPGdalphaG=mix(Ss.ddPGdalphaG, Sf.ddPGdalphaG),
        dPI=mix(Ss.dPI, Sf.dPI),
        hL=hD * D,
        beta=beta,
        tauG=mix(Ss.tauG, Sf.tauG),
        tauL=mix(Ss.tauL, Sf.tauL),
    )
