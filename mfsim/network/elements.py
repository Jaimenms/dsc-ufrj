"""Especificações especiais: válvula (choke), reservatório e válvula de retenção."""

from __future__ import annotations

import numpy as np


def valve_upstream_pressure(P_down, qL, qG, opening, CV, fluid, PR):
    """Pressão adimensional a montante da válvula (modelo_valvula.m).

    dP [bar] = qM |qM| / rhoM / (0.0075 CV ab)^2
    """
    qM = qL + qG
    P = P_down * PR
    rhoL, rhoG, _, _ = fluid.props(np.maximum(P, 1e3))
    if qM != 0:
        rhoM = (qL * rhoL + qG * rhoG) / qM
    else:
        rhoM = rhoL
    ab = max(float(opening), 1e-6)
    dP = qM * abs(qM) / rhoM / (0.0075 * CV * ab) ** 2
    ZP = (P + dP * 1e5) / PR
    return max(1e-3, ZP), dP


def reservoir_flows(P_node, IPL, IPG, P_res_bar, PR):
    """Vazões de líquido e gás do reservatório (IPR linear, >= 0)."""
    dP = P_res_bar - P_node * PR / 1e5
    return max(0.0, IPL * dP), max(0.0, IPG * dP), dP


def check_valve_flows(P_up, P_down, HG, CV, dP_min, fluid, PR):
    """Vazões (L, G) através de uma válvula de retenção (modelo_valvulaRet.m).

    qM = 0.0075 CV sign(dP) sqrt(rhoM |dP|), dP em bar, só se dP > dP_min.
    """
    dP = (P_up - P_down) * PR / 1e5
    P_mean = 0.5 * (P_up + P_down) * PR
    rhoL, rhoG, _, _ = fluid.props(max(P_mean, 1e3))
    HL = 1.0 - HG
    rhoM = HL * rhoL + HG * rhoG
    # sqrt regularizada (derivada finita em dP -> 0) e abertura suave em dP_min
    eps = 1e-3  # bar
    qM = 0.0075 * CV * np.sqrt(rhoM) * dP / np.sqrt(abs(dP) + eps)
    w = 1.0 / (1.0 + np.exp(np.clip(-(dP - dP_min) / (0.25 * eps + 0.02 * max(dP_min, 0.0)), -700, 700)))
    if dP_min <= 0.0:
        w = 1.0 if dP > 0 else 0.0
        w = 0.5 * (1.0 + np.tanh(dP / eps))
    return w * HL * qM, w * HG * qM, dP
