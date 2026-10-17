"""Estado estacionário: solução de rhs(t0, X) = 0.

Estratégias:
  1. Newton amortecido com Jacobiana esparsa (colorida) e busca linear;
  2. continuação pseudo-transiente (integração implícita com passos
     crescentes) como fallback robusto.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .integrators.implicit import ImplicitBDF
from .integrators.jacobian import color_groups, colored_jacobian


def newton_steady(model, X0, t=0.0, tol=1e-9, max_iter=50, verbose=True, bounds=None, jax=False):
    """Newton amortecido; jax=True usa o lado direito compilado e a Jacobiana exata (densa). `info["cpu"]` é o tempo
    de relógio da chamada (inclui a compilação JAX se for a primeira)."""
    t_start = time.time()
    if jax:
        from .compiled import get_jax

        rhs, jac = get_jax(model)
    else:
        rhs, jac = model.rhs, None
        S = model.jac_sparsity(exact_galerkin=True)
        groups = color_groups(S)
    X = np.asarray(X0, float).copy()
    F = rhs(t, X)
    res = np.linalg.norm(F)
    hist = [res]
    if verbose:
        print(f"  it 0  |F| = {res:.3e}")
    for it in range(1, max_iter + 1):
        if res < tol:
            break
        if jac is not None:
            Jd = jac(t, X)
            try:
                dX = np.linalg.solve(Jd, F)
            except np.linalg.LinAlgError:
                dX = np.linalg.lstsq(Jd, F, rcond=None)[0]
        else:
            J = colored_jacobian(lambda y: rhs(t, y), X, f0=F, S=S, groups=groups)
            try:
                dX = spla.spsolve(J.tocsc(), F)
            except Exception:
                dX = np.linalg.lstsq(J.toarray(), F, rcond=None)[0]
        if not np.all(np.isfinite(dX)):
            break
        # busca linear simples
        lam = 1.0
        best = None
        for _ in range(12):
            Xn = X - lam * dX
            if bounds is not None:
                Xn = np.minimum(np.maximum(Xn, bounds[0]), bounds[1])
            Fn = rhs(t, Xn)
            rn = np.linalg.norm(Fn)
            if np.isfinite(rn) and rn < res:
                best = (Xn, Fn, rn)
                break
            lam *= 0.5
        if best is None:
            if verbose:
                print("  busca linear falhou")
            break
        X, F, res = best
        hist.append(res)
        if verbose:
            print(f"  it {it}  |F| = {res:.3e}  (lambda={lam:g})")
    return X, dict(residual=res, iterations=len(hist) - 1, history=hist, converged=res < tol, cpu=time.time() - t_start)


def pseudo_transient(
    model, X0, t=0.0, t_final=1e5, rtol=1e-6, atol=1e-8, verbose=True, bounds=None, max_step=np.inf, jax=False
):
    """Integra o sistema congelando as especificações em t até o regime permanente."""
    S = model.jac_sparsity()
    kw = {}
    if jax:
        from .compiled import get_jax

        rhs_j, jac_j = get_jax(model)
        kw["jac_fun"] = lambda tt, x: jac_j(t, x)
        rhs_t = rhs_j
    else:
        rhs_t = model.rhs

    def frozen_rhs(tt, x):
        return rhs_t(t, x)

    bdf = ImplicitBDF(frozen_rhs, S=S, rtol=rtol, atol=atol, max_step=max_step, bounds=bounds, verbose=verbose, **kw)
    T, X, info = bdf.integrate([0.0, t_final], X0)
    Xf = X[:, -1]
    res = np.linalg.norm(model.rhs(t, Xf))
    if verbose:
        print(f"  pseudo-transiente: |F| = {res:.3e} ({info['n_steps']} passos)")
    return Xf, dict(residual=res, info=info)


def steady_state(model, X0, t=0.0, tol=1e-8, verbose=True, bounds=None, pt_time=1e4, jax=False):
    """Newton; se falhar, pseudo-transiente seguido de Newton. jax=True usa mfsim.compiled."""
    X, info = newton_steady(model, X0, t=t, tol=tol, verbose=verbose, bounds=bounds, jax=jax)
    if info["converged"]:
        return X, info
    if verbose:
        print("Newton não convergiu; usando continuação pseudo-transiente")
    X, info2 = pseudo_transient(model, X0, t=t, t_final=pt_time, verbose=verbose, bounds=bounds, jax=jax)
    X, info3 = newton_steady(model, X, t=t, tol=tol, verbose=verbose, bounds=bounds, jax=jax)
    info3["pseudo_transient"] = info2
    return X, info3
