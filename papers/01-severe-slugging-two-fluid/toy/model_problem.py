"""Problema-modelo do artigo 01: uma linha de transmissão terminada num elemento de resistência negativa.

Duas EDP lineares de coeficientes constantes na linha 0 < x < 1 (pressão p e fluxo w),

    C ∂p/∂t + ∂w/∂x = 0,        L ∂w/∂t + ∂p/∂x = −R w,

alimentada por uma fonte linear, w(0, t) = k (P_r − p(0, t)) (o análogo da IPR), e terminada por um elemento de um
só estado y (o «riser») com uma resistência linear a (o «choke»),

    m dy/dt = p(1, t) − f(y) − a y,    w(1, t) = y,    f(y) = g0 − β (y − y0) + (y − y0)³/3.

A única não-linearidade é a característica cúbica f: a inclinação −β em torno de y0 é uma resistência negativa (mais
fluxo, menos contrapressão, como uma coluna que fica mais leve quando o gás entra) e o termo cúbico limita a amplitude.
É a linha terminada num díodo de túnel (a origem das equações de FitzHugh–Nagumo): fechar o choke estabiliza; ao
abrir, o par lento atravessa o eixo imaginário (Hopf, ciclo quase sinusoidal) e, mais aberto, colide no eixo real
(ciclo de relaxação, a «golfada»).

Discretização escalonada como a do modelo a dois fluidos: p nas N células, w nas N − 1 faces interiores, a face de
entrada dada pela fonte e a de saída pelo estado y. O lado direito devolve d(M x)/dt (cargas C p, momentos L w e m y),
como o modo conservativo do simulador, e a linearização é M·δẋ = J·δx.

Uso (grava em runs/paper01/toy/; o guião das figuras só lê):

    uv run python papers/01-severe-slugging-two-fluid/toy/model_problem.py run
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "runs", "paper01", "toy")

# parâmetros adimensionais; P_r tal que o fluxo estacionário vale y0 = 1 com a = 1
PARAMS = dict(N=50, C=10.0, L=0.05, R=0.05, m=0.1, k=0.5, beta=1.0, g0=1.0, y0=1.0)
A_RUNS = (1.2, 0.95, 0.85, 0.5)  # estável, Hopf, ciclo maior, relaxação
A_SCAN = tuple(np.round(np.r_[np.arange(0.2, 0.7, 0.05), np.arange(0.7, 1.5001, 0.01)], 3))


def params(**kw):
    p = dict(PARAMS, **kw)
    if "Pr" not in p:
        y, a = p["y0"], 1.0
        p["Pr"] = f(y, p) + a * y + p["R"] * y * (p["N"] - 1) / p["N"] + y / p["k"]
    return p


def f(y, p):
    """Característica do «riser»: resistência negativa −β em torno de y0, saturação cúbica."""
    return p["g0"] - p["beta"] * (y - p["y0"]) + (y - p["y0"]) ** 3 / 3.0


def df(y, p):
    return -p["beta"] + (y - p["y0"]) ** 2


def mass(p):
    """Diagonal da matriz de massa: C nas células, L nas faces interiores, m no «riser»."""
    N = p["N"]
    return np.concatenate([np.full(N, p["C"]), np.full(N - 1, p["L"]), [p["m"]]])


def unpack(x, p):
    N = p["N"]
    return x[:N], x[N : 2 * N - 1], x[2 * N - 1]


def rhs(x, a, p):
    """d(M x)/dt do sistema semi-discreto."""
    N = p["N"]
    h = 1.0 / N
    pc, w, y = unpack(x, p)
    wf = np.concatenate([[p["k"] * (p["Pr"] - pc[0])], w, [y]])
    return np.concatenate([-(wf[1:] - wf[:-1]) / h, -(pc[1:] - pc[:-1]) / h - p["R"] * w, [pc[-1] - f(y, p) - a * y]])


def jac(x, a, p):
    """Jacobiana exata de `rhs` (densa)."""
    N = p["N"]
    h = 1.0 / N
    n = 2 * N
    J = np.zeros((n, n))
    iw = np.arange(N, 2 * N - 1)  # faces interiores 1..N−1 (face i entre as células i−1 e i, base 0)
    # células: −(w_{i+1} − w_i)/h, com w_0 = k (Pr − p_0) e w_N = y
    J[0, 0] = -p["k"] / h  # a face de entrada w_0 = k (Pr − p_0) entra como +w_0/h na célula 0
    for i in range(N):
        if i < N - 1:
            J[i, iw[i]] -= 1.0 / h  # −w_{i+1}/h (face à direita da célula i)
        else:
            J[i, n - 1] -= 1.0 / h  # face de saída: y
        if i > 0:
            J[i, iw[i - 1]] += 1.0 / h  # +w_i/h (face à esquerda)
    for j in range(N - 1):
        J[iw[j], j] += 1.0 / h
        J[iw[j], j + 1] -= 1.0 / h
        J[iw[j], iw[j]] -= p["R"]
    J[n - 1, N - 1] = 1.0
    J[n - 1, n - 1] = -df(x[-1], p) - a
    return J


def steady(a, p):
    """Estacionário: fluxo uniforme y, pressão linear na linha."""
    N = p["N"]
    h = 1.0 / N

    def g(y):
        pN = f(y, p) + a * y
        return y - p["k"] * (p["Pr"] - (pN + p["R"] * y * (N - 1) * h))

    y = brentq(g, -5.0, 10.0)
    pN = f(y, p) + a * y
    pc = pN + p["R"] * y * h * np.arange(N - 1, -1, -1)
    return np.concatenate([pc, np.full(N - 1, y), [y]])


def spectrum(a, p, with_mass=True):
    """Autovalores da linearização no estacionário: de M⁻¹J (com a matriz de massa) ou de J sozinha."""
    x = steady(a, p)
    J = jac(x, a, p)
    A = J / mass(p)[:, None] if with_mass else J
    w = np.linalg.eigvals(A)
    return w[np.argsort(-w.real)]


def lumped(a, p):
    """Aproximação concentrada (linha como uma capacitância C, coluna como inércia m): traço e determinante de
    [[−k/C, −1/C], [1/m, −r/m]] com r = a + f'(y*) a resistência líquida da terminação."""
    y = steady(a, p)[-1]
    r = a + df(y, p)
    tr = -p["k"] / p["C"] - r / p["m"]
    det = (p["k"] * r + 1.0) / (p["C"] * p["m"])
    return r, tr, det


def run(a, p, tf=300.0, dt_out=0.1, perturb=1e-3, max_step=0.5):
    """Transitório a partir do estacionário perturbado (pressões × (1 + perturb)) com o BDF do simulador, resíduo
    conservativo T(x) = M x."""
    import scipy.sparse as sp

    from mfsim.integrators.implicit import ImplicitBDF

    x0 = steady(a, p)
    x0[: p["N"]] *= 1.0 + perturb
    Mv = mass(p)
    n = x0.size
    bdf = ImplicitBDF(
        lambda t, x: rhs(x, a, p),
        S=sp.csc_matrix(np.ones((n, n))),
        rtol=1e-4,
        atol=1e-6,
        max_step=max_step,
        transform=lambda x: Mv * x,
        inverse=lambda y: y / Mv,
        transform_jac=lambda x: sp.diags(Mv),
        jac_fun=lambda t, x: jac(x, a, p),
    )
    T, X, info = bdf.integrate(np.arange(0.0, tf + 0.5 * dt_out, dt_out), x0)
    return T, X, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--out", default=OUT)
    a_ = ap.parse_args(argv)
    os.makedirs(a_.out, exist_ok=True)
    p = params()
    for a in A_RUNS:
        T, X, info = run(a, p)
        path = os.path.join(a_.out, f"a{a:.2f}.npz")
        np.savez(path, T=T, X=X, a=a, n_steps=info["n_steps"], **{f"param_{k}": v for k, v in p.items()})
        print(f"a = {a:.2f}: {info['n_steps']} passos -> {path}")


if __name__ == "__main__":
    main()
