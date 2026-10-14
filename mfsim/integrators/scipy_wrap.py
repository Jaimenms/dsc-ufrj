"""Integração com scipy.integrate.solve_ivp (BDF/Radau), análogo ao ode15s usado na tese."""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp


def integrate_scipy(model, X0, t_eval, method="BDF", rtol=1e-4, atol=1e-6, max_step=np.inf,
                    use_sparsity=True, progress=True, first_step=None):
    """Integra dX/dt = model.rhs(t, X) e devolve (T, X[n_states x nt], info)."""
    t_eval = np.asarray(t_eval, float)
    jac_sparsity = model.jac_sparsity() if use_sparsity else None
    t_start = time.time()
    n_calls = [0]

    def f(t, x):
        n_calls[0] += 1
        return model.rhs(t, x)

    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), np.asarray(X0, float), method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step,
                    jac_sparsity=jac_sparsity, first_step=first_step)
    info = dict(success=sol.success, message=sol.message, n_rhs=n_calls[0],
                n_steps=sol.t.size, cpu=time.time() - t_start, nfev=sol.nfev, njev=sol.njev,
                nlu=sol.nlu)
    if progress:
        print(f"[scipy {method}] {sol.message} | rhs={n_calls[0]} | njev={sol.njev} | {info['cpu']:.1f}s")
    return sol.t, sol.y, info
