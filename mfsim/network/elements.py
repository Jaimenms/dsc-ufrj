"""Relações das arestas de comprimento zero: válvula (choke), reservatório (IPR) e
válvula de retenção. Funções puras sobre o backend injetado (`bk`): escalares 0-d em
numpy (lado da rede) ou traçadas em JAX (lado direito compilado); os valores coincidem."""

from __future__ import annotations

from ..backend import NUMPY

VALVE_LAWS = ("homogeneous", "thesis")
Q_SMOOTH = 1e-6  # m³/s: escala de regularização da lei homogénea perto de vazão nula


def valve_upstream_pressure(P_down, qL, qG, opening, CV, fluid, PR, law="homogeneous", q_smooth=Q_SMOOTH, bk=NUMPY):
    """Pressão adimensional a montante da válvula e queda dP [bar].

    Lei homogénea (padrão; issue 039): a mistura atravessa o choke como um fluido com a
    densidade sem escorregamento, ponderada por volume, rho = qM / (|qL|/rhoL + |qG|/rhoG),
    logo dP [bar] = qM Q / (0.0075 CV ab)^2 com Q = |qL|/rhoL + |qG|/rhoG a vazão
    volumétrica. Não há divisão pela vazão total, dP tem o sinal de qM e é monótona em cada
    vazão; os módulos são suaves, |x| ~ x tanh(x/q_s), exatos a menos de exp(-|x|/q_s).

    Lei da tese (`law="thesis"`, modelo_valvula.m, bit a bit com o MATLAB): densidade média
    ponderada pela vazão MÁSSICA, rhoM = (qL rhoL + qG rhoG) / qM, e
    dP [bar] = qM |qM| / rhoM / (0.0075 CV ab)^2. Fica perto da do líquido mesmo com o gás a
    dominar o volume (o choke quase não resiste ao gás) e é singular quando qL + qG -> 0.

    As densidades são as da pressão de jusante nas duas leis (como na tese)."""
    xp = bk.xp
    qM = qL + qG
    P = P_down * PR
    rhoL, rhoG, _, _ = fluid.props(xp.maximum(P, 1e3))
    ab = xp.maximum(opening, 1e-6)
    if law == "thesis":
        qM_safe = xp.where(qM != 0, qM, 1.0)
        rhoM = xp.where(qM != 0, (qL * rhoL + qG * rhoG) / qM_safe, rhoL)
        dP = qM * xp.abs(qM) / rhoM / (0.0075 * CV * ab) ** 2
    elif law == "homogeneous":
        QL, QG = qL / rhoL, qG / rhoG
        Q = QL * xp.tanh(QL / q_smooth) + QG * xp.tanh(QG / q_smooth)
        dP = qM * Q / (0.0075 * CV * ab) ** 2
    else:
        raise ValueError(f"lei de válvula desconhecida: {law!r} (use {VALVE_LAWS})")
    ZP = (P + dP * 1e5) / PR
    return xp.maximum(1e-3, ZP), dP


def reservoir_flows(P_node, IPL, IPG, P_res_bar, PR, bk=NUMPY):
    """Vazões de líquido e gás do reservatório (IPR linear, >= 0) e dP [bar]."""
    xp = bk.xp
    dP = P_res_bar - P_node * PR / 1e5
    return xp.maximum(0.0, IPL * dP), xp.maximum(0.0, IPG * dP), dP


def check_valve_flows(P_up, P_down, HG, CV, dP_min, fluid, PR, bk=NUMPY):
    """Vazões (L, G) através de uma válvula de retenção (modelo_valvulaRet.m) e dP [bar].

    qM = 0.0075 CV sign(dP) sqrt(rhoM |dP|), dP em bar, só se dP > dP_min.
    """
    xp = bk.xp
    dP = (P_up - P_down) * PR / 1e5
    P_mean = 0.5 * (P_up + P_down) * PR
    rhoL, rhoG, _, _ = fluid.props(xp.maximum(P_mean, 1e3))
    HL = 1.0 - HG
    rhoM = HL * rhoL + HG * rhoG
    # sqrt regularizada (derivada finita em dP -> 0) e abertura suave em dP_min
    eps = 1e-3  # bar
    qM = 0.0075 * CV * xp.sqrt(rhoM) * dP / xp.sqrt(xp.abs(dP) + eps)
    if dP_min <= 0.0:
        w = 0.5 * (1.0 + xp.tanh(dP / eps))
    else:
        w = bk.sigmoid((dP - dP_min) / (0.25 * eps + 0.02 * max(dP_min, 0.0)))
    return w * HL * qM, w * HG * qM, dP
