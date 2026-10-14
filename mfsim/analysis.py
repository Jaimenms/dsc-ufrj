"""Análises: Jacobiana densa e autovalores do sistema discretizado."""

from __future__ import annotations

import numpy as np


def dense_jacobian(model, X, t=0.0, rel=1e-7, abs_=1e-9):
    X = np.asarray(X, float)
    f0 = model.rhs(t, X)
    n = X.size
    J = np.empty((n, n))
    for j in range(n):
        h = rel * abs(X[j]) + abs_
        xp = X.copy()
        xp[j] += h
        J[:, j] = (model.rhs(t, xp) - f0) / h
    return J


def jacobian_eigenvalues(model, X, t=0.0):
    """Autovalores da Jacobiana dX/dt em (t, X), ordenados por parte real decrescente."""
    J = dense_jacobian(model, X, t)
    ev = np.linalg.eigvals(J)
    return ev[np.argsort(-ev.real)]


def stability_report(model, X, t=0.0, tol=1e-9):
    ev = jacobian_eigenvalues(model, X, t)
    unstable = ev[ev.real > tol]
    return dict(eigenvalues=ev, n_unstable=int(unstable.size), max_real=float(ev.real.max()),
                dominant=unstable[:5], period=(2 * np.pi / abs(unstable[0].imag) if unstable.size and unstable[0].imag != 0 else None))


def mass_balance(res, k=0, phase="L"):
    """Balanço de massa de uma fase num duto: estoque(t) - estoque(0) e integral
    acumulada de (entrada - saída) [kg].

    Usa os fluxos de massa efetivos nas faces de contorno: no modo conservativo do
    esquema fv_upwind são as vazões vezes o limitador da célula doadora (os mesmos
    fluxos das equações de massa); nos demais, as próprias vazões das faces.
    Devolve (mass, net, err) com err = (mass - mass[0]) - net."""
    model = res.model
    pb = model.pipes[k]
    p = res.pipe(k)
    T = res.T
    Y = p["Y"]                                    # (nt, n_nodes, 4), primitivas
    sch = pb.scheme
    A = np.pi * pb.pipe.D**2 / 4
    h = pb.mesh["h"]
    if sch.name.startswith("fv_"):
        cells = slice(1, sch.N + 1)
        P = Y[:, cells, 1] * model.PR
        aG = np.clip(Y[:, cells, 0], 0.0, 1.0)
        rhoL, rhoG, _, _ = model.fluid.props(P)
        rho_a = rhoG * aG if phase == "G" else rhoL * (1.0 - aG)
        mass = np.sum(rho_a * A * h[None, :], axis=1)
        if getattr(sch, "conservative", False):
            F = np.array([sch.mass_fluxes_nodal(Y[it])[0 if phase == "L" else 1] for it in range(T.size)])
            qin, qout = F[:, 0], F[:, -1]
        else:
            slot = 2 if phase == "L" else 3
            qin, qout = Y[:, 0, slot], Y[:, sch.N, slot]
    else:
        # esquemas nodais: regra do trapézio nos nós
        P = Y[:, :, 1] * model.PR
        aG = np.clip(Y[:, :, 0], 0.0, 1.0)
        rhoL, rhoG, _, _ = model.fluid.props(P)
        rho_a = rhoG * aG if phase == "G" else rhoL * (1.0 - aG)
        mass = np.sum(0.5 * (rho_a[:, 1:] + rho_a[:, :-1]) * A * h[None, :], axis=1)
        slot = 2 if phase == "L" else 3
        qin, qout = Y[:, 0, slot], Y[:, -1, slot]
    dT = np.diff(T)
    net = np.concatenate([[0.0], np.cumsum(0.5 * ((qin[1:] - qout[1:]) + (qin[:-1] - qout[:-1])) * dT)])
    return mass, net, (mass - mass[0]) - net
