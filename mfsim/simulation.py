"""Rotinas de alto nível: estado estacionário e simulação dinâmica de um caso."""

from __future__ import annotations

import numpy as np

from .integrators import integrate_implicit, integrate_scipy
from .network.model import NetworkModel
from .results import Results
from . import steady as _steady


def build_model(case, method=None, scheme_settings=None):
    return NetworkModel(case, method=method, scheme_settings=scheme_settings)


def state_bounds(model):
    """Limites físicos dos estados (frações em [0,1], pressões > 0)."""
    lo = np.full(model.n_states, -np.inf)
    hi = np.full(model.n_states, np.inf)
    a_min, a_max = 0.0, 1.0
    for pb in model.pipes:
        slots = pb.free % 4
        seg = slice(pb.offset, pb.offset + pb.n_states)
        lo_s = lo[seg]; hi_s = hi[seg]
        # limites consistentes com o corte interno do modelo (alpha_min/alpha_max)
        pa = getattr(pb.model, "params", None)
        a_min = getattr(pa, "alpha_min", 1e-3) if pa is not None else 1e-3
        a_max = getattr(pa, "alpha_max", 1 - 1e-3) if pa is not None else 1 - 1e-3
        if pb.scheme.name == "fv_upwind_cons":
            lo_s[slots == 0] = 0.0          # massas por volume >= 0
            lo_s[slots == 1] = 0.0
            continue
        if pb.scheme.name == "fv_upwind" and getattr(pb.scheme, "conservative", False):
            # modo conservativo: pequeno undershoot/overshoot admitido (a projeção em
            # 0/1 tornaria o resíduo conservativo insatisfazível quando a fase some e
            # forçaria passos minúsculos); o limite inferior fica a meio caminho da
            # singularidade do mapa de massas em alphaG = -gammaG/gammaL (-0.0105 com
            # gammaL = 100; -0.001 com gammaL = 1000, onde a banda fixa de -0.005 a cruzava
            # e a pressão da célula colapsava para o piso)
            fl_ = pb.model.fluid
            lo_s[slots == 0] = -min(0.005, 0.5 * fl_.gammaG / fl_.gammaL); hi_s[slots == 0] = 1.005
        elif pb.scheme.name == "fv_upwind":
            lo_s[slots == 0] = 0.0; hi_s[slots == 0] = 1.0   # coeficientes de massa consistentes em 0 e 1
        else:
            lo_s[slots == 0] = a_min; hi_s[slots == 0] = a_max
        lo_s[slots == 1] = 1e-4
    for k, vi in enumerate(model.vA):
        iH, iP = model.vertex_A_index(model.vertices[vi].index)
        if getattr(model, "conservative", False):
            lo[iH], hi[iH] = -min(0.005, 0.5 * model.fluid.gammaG / model.fluid.gammaL), 1.005     # undershoot/overshoot admitidos (como nas células conservativas)
        else:
            lo[iH], hi[iH] = a_min, a_max
        lo[iP] = 1e-4
    return lo, hi


def steady_state(case, method=None, X0=None, verbose=True, update_case=True, scheme_settings=None, **kw):
    """Estado estacionário em t0 (anulando as derivadas temporais)."""
    model = build_model(case, method, scheme_settings)
    if isinstance(X0, str) and X0 == "auto":
        from .initial_guess import initial_guess
        X0 = initial_guess(model)
    elif X0 is None:
        try:
            X0 = model.initial_state()
        except ValueError:
            from .initial_guess import initial_guess
            X0 = initial_guess(model)
    X, info = _steady.steady_state(model, X0, t=case.settings.t0, verbose=verbose,
                                   bounds=state_bounds(model), **kw)
    if update_case:
        model.state_to_case_ic(X, case.settings.t0)
    return model, X, info


def simulate(case, method=None, integrator=None, X0=None, t_eval=None, rtol=None, atol=None,
             max_step=None, progress=True, scheme_settings=None, jax=False, tol_range=None, **kw):
    """Simulação dinâmica do caso. Retorna Results. jax=True: lado direito compilado e
    Jacobiana exata (só com integrator="implicit")."""
    model = build_model(case, method, scheme_settings)
    st = case.settings
    if X0 is None:
        X0 = model.initial_state()
    if t_eval is None:
        t_eval = np.arange(st.t0, st.tf + 0.5 * st.dt_out, st.dt_out)
    rtol = st.rtol if rtol is None else rtol
    atol = st.atol if atol is None else atol
    max_step = st.max_step if max_step is None else max_step
    integrator = integrator or st.integrator
    if integrator in ("bdf", "BDF", "Radau", "radau", "LSODA"):
        meth = {"bdf": "BDF", "radau": "Radau"}.get(integrator, integrator)
        T, X, info = integrate_scipy(model, X0, t_eval, method=meth, rtol=rtol, atol=atol,
                                     max_step=max_step, progress=progress, **kw)
    elif integrator == "implicit":
        T, X, info = integrate_implicit(model, X0, t_eval, rtol=rtol, atol=atol, max_step=max_step,
                                        bounds=state_bounds(model), progress=progress, jax=jax, tol_range=tol_range, **kw)
    else:
        raise ValueError(integrator)
    return Results(model, T, X, info)
