"""Integração com scipy.integrate.solve_ivp (BDF/Radau), análogo ao ode15s usado na tese."""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp


def integrate_scipy(
    model,
    X0,
    t_eval,
    method="BDF",
    rtol=1e-4,
    atol=1e-6,
    max_step=np.inf,
    use_sparsity=True,
    progress=True,
    first_step=None,
    max_cpu=None,
):
    """Integra dX/dt = model.rhs(t, X) e devolve (T, X[n_states x nt], info). `max_cpu` (s) é um orçamento de
    tempo de relógio: a integração faz-se em blocos de instantes de saída e pára entre blocos quando ele se
    esgota (`success = False`), com a trajetória até aí."""
    t_eval = np.asarray(t_eval, float)
    jac_sparsity = model.jac_sparsity() if use_sparsity else None
    t_start = time.time()
    n_calls = [0]

    def f(t, x):
        n_calls[0] += 1
        return model.rhs(t, x)

    def solve(te, x0):
        return solve_ivp(
            f,
            (te[0], te[-1]),
            np.asarray(x0, float),
            method=method,
            t_eval=te,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            jac_sparsity=jac_sparsity,
            first_step=first_step,
        )

    if max_cpu is None:
        sol = solve(t_eval, X0)
        T, Y, ok, message = sol.t, sol.y, sol.success, sol.message
        nfev, njev, nlu = sol.nfev, sol.njev, sol.nlu
        stopped = False
    else:
        # orçamento de tempo: integra em blocos de instantes de saída e pára entre blocos quando se esgota
        # (o solve_ivp não tem orçamento; um evento terminal dependente do relógio não é contínuo em t)
        chunks = np.array_split(np.arange(t_eval.size), max(1, min(50, t_eval.size - 1)))
        T, Y, x0 = [t_eval[:1]], [np.asarray(X0, float)[:, None]], np.asarray(X0, float)
        ok, message, stopped, nfev, njev, nlu = True, "", False, 0, 0, 0
        last = 0
        for idx in chunks:
            idx = idx[idx > last]
            if idx.size == 0:
                continue
            te = t_eval[np.concatenate([[last], idx])]
            sol = solve(te, x0)
            nfev, njev, nlu = nfev + sol.nfev, njev + sol.njev, nlu + sol.nlu
            T.append(sol.t[1:])
            Y.append(sol.y[:, 1:])
            ok, message = sol.success, sol.message
            if not sol.success or sol.t.size < te.size:
                break
            x0, last = sol.y[:, -1], int(idx[-1])
            if time.time() - t_start > float(max_cpu):
                stopped = last < t_eval.size - 1
                break
        T, Y = np.concatenate(T), np.concatenate(Y, axis=1)
    info = dict(
        success=bool(ok) and not stopped,
        message=message,
        n_rhs=n_calls[0],
        n_steps=int(np.size(T)),
        cpu=time.time() - t_start,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
    )
    if stopped:
        info["error"] = f"RuntimeError: orçamento de tempo ({max_cpu:g} s) excedido em t={float(T[-1]):.4g}"
    if progress:
        print(f"[scipy {method}] {message} | rhs={n_calls[0]} | njev={njev} | {info['cpu']:.1f}s")
    return np.asarray(T, float), np.asarray(Y, float).reshape(len(X0), -1), info
