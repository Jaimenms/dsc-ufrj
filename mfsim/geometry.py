"""Geometria: nível estratificado em seção circular e traçado do duto."""

from __future__ import annotations

import numpy as np

# Polinômios (calcula_nivel_estratificado.m) usados como estimativa inicial
_P_H = np.array(
    [0.00000000000197, -3.66048527474641, 9.15121318685793, -8.89104962298284,
     4.18536124761860, -1.75567780293538, 0.98531913309307]
)
_P_BETA = 1.0e2 * np.array(
    [0.00000000000056, -0.67451330764088, 1.68628326909991, -1.58733054088785,
     0.69471254223251, -0.17280444315654, 0.05824216671205]
)


def stratified_level(alphaG, tol=1e-9, max_iter=100):
    """Altura adimensional de líquido hD=h/D e ângulo molhado beta para a
    fração de área de gás alphaG numa seção circular.

    Resolve  sin(beta) - beta + 2*pi*alphaL = 0  por Newton (vetorizado),
    partindo da aproximação polinomial do código original.
    """
    alphaG = np.asarray(alphaG, dtype=float)
    alphaL = 1.0 - alphaG
    beta = np.polyval(_P_BETA, alphaG)
    beta = np.clip(beta, 0.0, 2 * np.pi)
    # casos extremos tratados como no código original
    only_liq = alphaL > 1 - 1e-6
    only_gas = alphaL < 1e-6
    mid = ~(only_liq | only_gas)
    b = beta.copy()
    b[only_liq] = 2 * np.pi * alphaL[only_liq]
    b[only_gas] = 2 * np.pi * alphaG[only_gas]
    bm = b[mid]
    aL = alphaL[mid]
    for _ in range(max_iter):
        F0 = np.sin(bm) - bm + 2 * np.pi * aL
        dF = np.cos(bm) - 1.0
        dF = np.where(np.abs(dF) < 1e-14, -1e-14, dF)
        step = F0 / dF
        bm = bm - step
        if np.all(np.abs(step) <= tol):
            break
    b[mid] = bm
    hD = 0.5 * (1.0 - np.cos(b / 2.0))
    return hD, b


def alpha_from_level(hD):
    """Fração de área de gás a partir de h/D (estratificado)."""
    beta = 2.0 * np.arccos(1.0 - 2.0 * np.asarray(hD, dtype=float))
    return 1.0 - (beta - np.sin(beta)) / (2 * np.pi), beta


def pipe_mesh(x, y, z, n_elements, grading=None):
    """Malha de um duto poligonal.

    Parâmetros
    ----------
    x, y, z : coordenadas dos vértices do traçado (m), z vertical para cima
    n_elements : número de elementos em cada trecho (len = len(x)-1)
    grading : razão h_último/h_primeiro em cada trecho (progressão geométrica dos
        comprimentos; 1 = uniforme; < 1 refina para o fim do trecho, > 1 para o
        início). None = uniforme em todos.

    Retorna dict com: s (coordenada curvilínea dos nós), h (comprimento dos
    elementos), theta (ângulo de inclinação de cada elemento, rad),
    xyz dos nós, comprimento total L.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    n_elements = np.atleast_1d(np.asarray(n_elements, int))
    if n_elements.size == 1 and x.size - 1 > 1:
        n_elements = np.repeat(n_elements, x.size - 1)
    assert n_elements.size == x.size - 1, "n_elements deve ter len(x)-1 entradas"
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    dL = np.sqrt(dx**2 + dy**2 + dz**2)
    theta_seg = np.arcsin(np.clip(dz / dL, -1, 1))
    if grading is None:
        grading = np.ones(n_elements.size)
    grading = np.atleast_1d(np.asarray(grading, float))
    if grading.size == 1 and n_elements.size > 1:
        grading = np.repeat(grading, n_elements.size)
    assert grading.size == n_elements.size, "grading deve ter uma razão por trecho"
    h_list = []
    for i, n in enumerate(n_elements):
        g = grading[i]
        if n == 1 or abs(g - 1.0) < 1e-12:
            h_list.append(np.full(n, dL[i] / n))
        else:
            q = g ** (1.0 / (n - 1))                         # razão entre elementos consecutivos
            h1 = dL[i] * (1.0 - q) / (1.0 - q**n)
            h_list.append(h1 * q ** np.arange(n))
    h = np.concatenate(h_list)
    theta = np.concatenate([np.full(n, theta_seg[i]) for i, n in enumerate(n_elements)])
    nodes = [np.array([x[0], y[0], z[0]])]
    for i, n in enumerate(n_elements):
        frac = np.cumsum(h_list[i]) / dL[i]
        frac[-1] = 1.0
        seg = np.stack([x[i] + dx[i] * frac, y[i] + dy[i] * frac, z[i] + dz[i] * frac], axis=1)
        nodes.append(seg)
    xyz = np.vstack([nodes[0][None, :]] + nodes[1:])
    s = np.concatenate([[0.0], np.cumsum(h)])
    return {
        "s": s,
        "h": h,
        "theta": theta,
        "xyz": xyz,
        "L": float(s[-1]),
        "n_elements": int(h.size),
        "segments": n_elements,
    }


def node_angles(theta_elem):
    """Ângulo nos nós (média dos elementos vizinhos), como interpola_angulo.m."""
    th = np.asarray(theta_elem, float)
    out = np.empty(th.size + 1)
    out[0] = th[0]
    out[-1] = th[-1]
    out[1:-1] = 0.5 * (th[1:] + th[:-1])
    return out
