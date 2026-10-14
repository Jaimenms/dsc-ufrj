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


def blowdowns(T, P_in, min_gap=200.0):
    """Instantes dos blowdowns de uma série de pressão na entrada: mínimos locais abaixo de
    30 % da faixa, separados por pelo menos `min_gap` s (um blowdown tem mínimos
    secundários nos segundos seguintes). Mais robusto do que contar picos de vazão.
    (Era `blowdowns` no notebook nb_severe_slug e em scratch/cycle_stats.py.)"""
    T = np.asarray(T, float); P_in = np.asarray(P_in, float)
    lvl = P_in.min() + 0.3 * (P_in.max() - P_in.min())
    low = P_in < lvl
    out = []
    i = 0
    while i < T.size:
        if low[i]:
            j = i
            while j < T.size and low[j]:
                j += 1
            k = i + int(np.argmin(P_in[i:j]))
            if not out or T[k] - out[-1] > min_gap:
                out.append(T[k])
            i = j
        else:
            i += 1
    return np.array(out)


def cycle_signature(T, P_in, q_out, min_gap=200.0, t_skip=0.0):
    """Assinatura de um ciclo de intermitência severa (para comparações tolerantes ao
    caos): instante do primeiro blowdown, intervalo médio, número de blowdowns, P_in
    mínima e máxima, pico de q_out depois de t_skip. P_in em bar, q_out em kg/s."""
    T = np.asarray(T, float); P_in = np.asarray(P_in, float); q_out = np.asarray(q_out, float)
    tb = blowdowns(T, P_in, min_gap)
    sel = T >= t_skip
    return dict(t_first=float(tb[0]) if tb.size else np.nan,
                interval=float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan,
                n_blowdowns=int(tb.size), P_min=float(P_in[sel].min()), P_max=float(P_in[sel].max()),
                q_peak=float(q_out[sel].max()))


def gas_front_widths(x, aG, ib):
    """Largura (m) da frente de gás no riser em cada instante em que há gás acima da base
    (fração > 0.5 em algum ponto e < 0.05 noutro): distância entre os pontos em que a
    fração cruza 0.1 e 0.9. (Era parte de scratch/cycle_stats.py.)"""
    widths = []
    xs = x[ib:]
    for it in range(aG.shape[1]):
        seg = aG[ib:, it]
        if seg.max() > 0.5 and seg.min() < 0.05:
            j1 = np.argmax(seg > 0.1)
            j9 = np.argmax(seg > 0.9) if (seg > 0.9).any() else len(seg) - 1
            if j9 > j1:
                widths.append(xs[j9] - xs[j1])
    return np.array(widths)


def cycle_stats(run, t_skip=100.0, min_gap=200.0):
    """Estatísticas de ciclo de uma corrida gravada do severe slug (partida bloqueada, duto
    único): instantes dos blowdowns, intervalo médio, pico e média da vazão de líquido na
    saída, faixa da pressão de entrada (bar) depois de `t_skip`, largura mediana da frente
    de gás no riser, passos e CPU do integrador. (Era scratch/cycle_stats.py.)"""
    p = run.pipe(0)
    T = run.T
    P1 = p["out"]["P_bar"][1]
    q = p["out"]["qL"][-1]
    aG = p["out"]["alphaG"]
    x = p["x"]
    tb = blowdowns(T, P1, min_gap)
    per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    sel = T > t_skip
    widths = gas_front_widths(x, aG, run.ib)
    return dict(blowdowns=tb, interval=per, q_peak=float(q[sel].max()), q_mean=float(q[sel].mean()),
                P_min=float(P1[sel].min()), P_max=float(P1[sel].max()),
                front_width=float(np.median(widths)) if widths.size else np.nan,
                n_steps=run.meta.get("n_steps", run.info.get("n_steps")), cpu=run.meta.get("cpu", run.info.get("cpu")))


CYCLE_STATS_HEADER = (f"{'trajetória':48s} {'passos':>7s} {'CPU':>6s} {'blowdowns':>24s} {'interv.':>7s} {'pico':>6s} "
                      f"{'P_in':>11s} {'prod.':>6s} {'frente':>6s}")


def format_cycle_stats(name, st):
    """Linha da tabela de `cycle_stats` (formato de scratch/cycle_stats.py)."""
    short = name.replace("severe_slug_blocked_", "")
    n_steps = int(st["n_steps"]) if st["n_steps"] is not None else -1
    cpu = float(st["cpu"]) if st["cpu"] is not None else np.nan
    tb = np.asarray(st["blowdowns"], float)
    return (f"{short:48s} {n_steps:7d} {cpu:5.0f}s {str(tb.round(0).astype(int).tolist()):>24s} {st['interval']:6.0f}s "
            f"{st['q_peak']:5.2f} {st['P_min']:5.2f}-{st['P_max']:4.2f} {st['q_mean']:6.3f} {st['front_width']:5.1f}m")
