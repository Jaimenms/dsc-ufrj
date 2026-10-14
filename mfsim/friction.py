"""Fatores de atrito de Fanning."""

from __future__ import annotations

import numpy as np

# Regularização de |v| em torno de zero (m/s): torna a tensão de cisalhamento
# tau = f(Re) rho v |v| / 2 suave em v = 0 (no regime laminar tau -> 8 mu v / D,
# linear em v, em vez de ter derivada descontínua). Para |v| >> V_EPS o efeito
# é desprezível (erro relativo ~ (V_EPS/v)^2 / 2).
V_EPS = 1e-4


def vabs(v):
    """|v| regularizado: sqrt(v^2 + V_EPS^2)."""
    return np.sqrt(np.asarray(v, float) ** 2 + V_EPS**2)


def fanning(v, rho, mu, D, eps, correlation="churchill"):
    """Fator de atrito de Fanning e número de Reynolds (fator.m).

    Para v == 0 retorna f = 0 (como no código original).
    D pode ser um diâmetro hidráulico (array).
    """
    v = np.asarray(v, float)
    D = np.asarray(D, float)
    zero = np.zeros(np.broadcast(v, D).shape, dtype=bool) if V_EPS > 0 else (v == 0)
    vv = np.where(zero, 1e-6, v)
    Dd = np.where(D <= 0, 1e-12, D)
    Re = Dd * (vabs(vv) if V_EPS > 0 else np.abs(vv)) * rho / mu
    if correlation == "churchill":
        A = (2.457 * np.log(1.0 / (0.27 * eps / Dd + (7.0 / Re) ** 0.9))) ** 16
        B = (37530.0 / Re) ** 16
        f = 2.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    elif correlation == "haaland":
        f = 0.07716 / (np.log10(6.9 / Re + (eps / Dd / 3.7) ** 1.11)) ** 2
    elif correlation == "laminar":
        f = 16.0 / Re
    elif correlation == "blasius":
        f = 0.046 * Re ** (-0.2)
    else:
        raise ValueError(correlation)
    f = np.where(zero, 0.0, f)
    Re = np.where(zero, 0.0, Re)
    return f, Re
