"""Rotinas de alto nível: estado estacionário e simulação dinâmica de um caso."""

from __future__ import annotations

import numpy as np

from .integrators import integrate_scipy
from .network.model import NetworkModel
from .results import Results
from . import steady as _steady


def build_model(case, method=None, scheme_settings=None):
    """Sistema da rede a partir de um `Network`."""
    return NetworkModel(case, method=method, scheme_settings=scheme_settings)


def state_bounds(model):
    """Limites físicos dos estados (frações em [0,1], pressões > 0)."""
    lo = np.full(model.n_states, -np.inf)
    hi = np.full(model.n_states, np.inf)
    a_min, a_max = 0.0, 1.0
    for pb in model.pipes:
        slots = pb.free % 4
        seg = slice(pb.offset, pb.offset + pb.n_states)
        lo_s = lo[seg]
        hi_s = hi[seg]
        # limites consistentes com o corte interno do modelo (alpha_min/alpha_max)
        pa = getattr(pb.model, "params", None)
        a_min = getattr(pa, "alpha_min", 1e-3) if pa is not None else 1e-3
        a_max = getattr(pa, "alpha_max", 1 - 1e-3) if pa is not None else 1 - 1e-3
        if pb.scheme.name == "fv_upwind_cons":
            lo_s[slots == 0] = 0.0  # massas por volume >= 0
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
            lo_s[slots == 0] = -min(0.005, 0.5 * fl_.gammaG / fl_.gammaL)
            hi_s[slots == 0] = 1.005
        elif pb.scheme.name == "fv_upwind":
            lo_s[slots == 0] = 0.0
            hi_s[slots == 0] = 1.0  # coeficientes de massa consistentes em 0 e 1
        else:
            lo_s[slots == 0] = a_min
            hi_s[slots == 0] = a_max
        lo_s[slots == 1] = 1e-4
    for k, vi in enumerate(model.vA):
        iH, iP = model.vertex_A_index(model.vertices[vi].id)
        if getattr(model, "conservative", False):
            lo[iH], hi[iH] = (
                -min(0.005, 0.5 * model.fluid.gammaG / model.fluid.gammaL),
                1.005,
            )  # undershoot/overshoot admitidos (como nas células conservativas)
        else:
            lo[iH], hi[iH] = a_min, a_max
        lo[iP] = 1e-4
    return lo, hi


def steady_state(case, method=None, X0=None, verbose=True, update_case=True, scheme_settings=None, **kw):
    """Estado estacionário em t0 (anulando as derivadas temporais); com `update_case` o estado
    é escrito nas condições iniciais do grafo (e do registo legado, se foi esse o caso dado)."""
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
    X, info = _steady.steady_state(model, X0, t=model.t0, verbose=verbose, bounds=state_bounds(model), **kw)
    if update_case:
        model.state_to_case_ic(X, model.t0)
    return model, X, info


def _integrate_bdf(model, X0, t_eval, spec, progress=True, verbose=False, partial=False):
    """BDF implícito construído a partir do `IntegrationSpec` do caso (os campos a None usam os
    defaults do `ImplicitBDF`): tolerância por faixa ou por variável, limites físicos,
    mapa conservativo, Jacobiana exata em JAX, acumulador dos fluxos de contorno,
    `slow_exit`/`predict_space`/`jt_refresh`, primeiro passo e pontos de quebra."""
    import time as _time

    from .integrators.implicit import ImplicitBDF

    X0 = np.asarray(X0, float)
    rtol = float(spec.rtol)
    if spec.tol_range is not None:
        atol = float(spec.tol_range) * model.state_scales(X0)
    else:
        atol = float(spec.atol)
        if spec.atol_P is not None or spec.atol_q is not None:
            atol = atol * np.ones(X0.size)
            slots = model.state_slots()
            if spec.atol_P is not None:
                atol[slots == 1] = spec.atol_P
            if spec.atol_q is not None:
                atol[(slots == 2) | (slots == 3)] = spec.atol_q
    lo, hi = state_bounds(model)
    if spec.tight_bounds:
        slots = model.pipes[0].free % 4
        lo[slots == 0] = 1e-3
        hi[slots == 0] = 1 - 1e-3
    kw = {}
    if model.conservative:
        kw.update(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
    fun, jac_fun = model.rhs, None
    if spec.backend == "jax":
        from .compiled import get_jax

        fun, jac_fun = get_jax(model)
    bdf = ImplicitBDF(
        fun,
        S=model.jac_sparsity(),
        rtol=rtol,
        atol=atol,
        max_step=float(spec.max_step),
        min_step=1e-8 if spec.min_step is None else float(spec.min_step),
        newton_tol=1e-3 if spec.newton_tol is None else float(spec.newton_tol),
        max_newton=12 if spec.max_newton is None else int(spec.max_newton),
        max_order=2 if spec.max_order is None else int(spec.max_order),
        bounds=(lo, hi),
        verbose=verbose,
        jac_fun=jac_fun,
        **kw,
    )
    if spec.slow_exit is not None:
        bdf.slow_exit = bool(spec.slow_exit)
    if spec.max_cpu is not None:
        bdf.max_cpu = float(spec.max_cpu)
    if spec.predict_space is not None:
        bdf.predict_space = spec.predict_space
    if spec.jt_refresh is not None:
        bdf.jt_refresh = spec.jt_refresh
    for key in ("newton_damping", "jac_policy", "predictor", "error_estimate"):  # issues 053, 055, 058
        if getattr(spec, key) is not None:
            setattr(bdf, key, str(getattr(spec, key)))
    if spec.jac_rate is not None:
        bdf.jac_rate = float(spec.jac_rate)
    if spec.jac_refresh_max is not None:
        bdf.jac_refresh_max = int(spec.jac_refresh_max)
    if spec.accumulate:
        bdf.accum = model.boundary_net_fluxes
    t_start = _time.time()
    try:
        T, X, info = bdf.integrate(t_eval, X0, first_step=spec.first_step, breakpoints=model.spec_breakpoints())
    except Exception as e:
        if not partial:
            raise
        out = getattr(bdf, "_out", None) or []
        X = np.array(out).T if out else np.zeros((X0.size, 0))
        T = np.asarray(getattr(bdf, "_t_eval", t_eval), float)[: X.shape[1]]
        info = dict(
            success=False,
            error=f"{type(e).__name__}: {e}",
            n_steps=bdf.n_steps,
            n_rejected=bdf.n_rejected,
            n_rhs=bdf.n_rhs,
            n_jac=bdf.n_jac,
            cpu=_time.time() - t_start,
        )
    if progress and info.get("success", True):
        print(
            f"[implicit BDF] steps={info['n_steps']} rejected={info['n_rejected']} rhs={info['n_rhs']} "
            f"jac={info['n_jac']} | {info['cpu']:.1f}s"
        )
    return T, X, info


def output_times(t0, tf, dt_out):
    """Instantes de saída: `t0`, `t0 + dt_out`, ... e sempre `tf` (também quando `tf` não é múltiplo
    de `dt_out`, p. ex. `--tf 0.5` com saída a cada 10 s dá `[0, 0.5]`)."""
    t = np.arange(t0, tf + 0.5 * dt_out, dt_out)
    if t.size == 0 or tf - t[-1] > 1e-9 * max(1.0, abs(dt_out)):
        t = np.append(t, tf)
    return t


def simulate(
    case,
    method=None,
    scheme_settings=None,
    X0=None,
    t_eval=None,
    progress=True,
    verbose=False,
    partial=False,
    jax=None,
    **overrides,
):
    """Simulação dinâmica de um caso (`Network`, ou o registo legado): o caminho único.

    Tudo vem do caso — tempo (`time`), integrador e tolerâncias (`integration`) e a política
    de estado inicial (`initial`: condições iniciais dos dutos/tanques ou estimativa);
    `overrides` (rtol, atol, max_step, integrator, backend, tol_range, first_step, max_order,
    max_newton, newton_tol, min_step, slow_exit, accumulate, ...) substituem campos de
    `integration` só nesta chamada (`jax=True` é atalho de `backend="jax"`). `X0`/`t_eval`
    substituem o estado inicial e os instantes de saída (por omissão `output_times`: de `t0` a
    `tf` de `dt_out` em `dt_out`, e `tf` sempre incluído). Com `partial=True` uma falha do
    integrador devolve a trajetória integrada até aí (`info["success"] = False`). Devolve
    `Results`."""
    from dataclasses import replace

    model = build_model(case, method, scheme_settings)
    spec = model.integration
    if jax is not None:
        overrides["backend"] = "jax" if jax else "numpy"
    if overrides:
        spec = replace(spec, **overrides)
    start_info = None
    if X0 is None:
        if model.network.initial == "guess":
            from .initial_guess import initial_guess

            X0 = initial_guess(model)
        elif model.network.initial == "steady":
            X0, start_info = steady_start(model, jax=spec.backend == "jax")
        else:
            X0 = model.initial_state()
    settle_ok = t_eval is None  # instantes explícitos desligam a paragem adaptativa
    if t_eval is None:
        t_eval = output_times(model.t0, model.tf, model.dt_out)
    t_eval = np.asarray(t_eval, float)
    if t_eval.size < 2:  # tf == t0: a "trajetória" é o estado inicial (nada a integrar)
        X0 = np.asarray(X0, float)
        info = dict(success=True, message="tf == t0: estado inicial", n_steps=0, n_rhs=0, cpu=0.0)
        if start_info is not None:
            info["initial"] = start_info
        return Results(model, t_eval, X0[:, None], info)
    settle = dict(model.network.time.settle) if (settle_ok and model.network.time.settle) else {}
    if settle:
        T, X, info = _integrate_settled(model, X0, spec, settle, progress=progress, verbose=verbose, partial=partial)
    else:
        T, X, info = _integrate(model, X0, t_eval, spec, progress=progress, verbose=verbose, partial=partial)
    if start_info is not None:
        info = dict(info, initial=start_info)
    return Results(model, T, X, info)


def _integrate(model, X0, t_eval, spec, progress=True, verbose=False, partial=False):
    integrator = spec.integrator
    if integrator in ("bdf", "BDF", "Radau", "radau", "LSODA"):
        meth = {"bdf": "BDF", "radau": "Radau"}.get(integrator, integrator)
        return integrate_scipy(
            model,
            X0,
            t_eval,
            method=meth,
            rtol=spec.rtol,
            atol=spec.atol,
            max_step=spec.max_step,
            progress=progress,
            max_cpu=spec.max_cpu,
            partial=partial,
        )
    if integrator == "implicit":
        return _integrate_bdf(model, X0, t_eval, spec, progress=progress, verbose=verbose, partial=partial)
    raise ValueError(integrator)


SETTLE_DEFAULTS = dict(window=2000.0, amp_tol=0.05, rel_tol=0.1)
_SIGNAL_SLOT = {"alphaG": 0, "P": 1, "qL": 2, "qG": 3}


def _signal(model, T, X, sig):
    """Série do sinal monitorizado (P em bar; alphaG; qL, qG em kg/s) num nó de um duto."""
    edge = sig.get("edge")
    k = 0 if edge is None else next(i for i, pb in enumerate(model.pipes) if str(pb.pipe.id) == str(edge))
    node, slot = int(sig.get("node", 1)), _SIGNAL_SLOT[sig.get("var", "P")]
    y = np.array([model.assemble(T[i], X[:, i])[1][k][node, slot] for i in range(T.size)])
    return y * model.PR / 1e5 if slot == 1 else y


def window_signature(T, y):
    """Assinatura de uma janela: amplitude, média, número de cruzamentos da média e período
    médio entre o primeiro e o último cruzamento ascendente (interpolados; NaN com menos de
    dois)."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    m = float(np.mean(y))
    d = y - m
    sg = np.sign(d)
    nz = sg[sg != 0]
    n = int(np.sum(nz[1:] != nz[:-1]))
    up = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    t_up = T[up] - d[up] * (T[up + 1] - T[up]) / (d[up + 1] - d[up])
    return dict(
        t_end=float(T[-1]),
        amp=float(np.max(y) - np.min(y)),
        mean=m,
        crossings=n,
        period=float((t_up[-1] - t_up[0]) / (t_up.size - 1)) if t_up.size >= 2 else float("nan"),
    )


def settled(windows, amp_tol, rel_tol):
    """ "steady" se a amplitude da última janela é menor do que amp_tol; "cycle" se as duas
    últimas janelas têm pelo menos 4 cruzamentos e amplitude e período iguais a rel_tol;
    None caso contrário (incluindo uma deriva sem ciclo)."""
    w = windows[-1]
    if w["amp"] < amp_tol:
        return "steady"
    if len(windows) >= 2:
        a = windows[-2]
        if a["crossings"] >= 4 and w["crossings"] >= 4:
            if (
                abs(a["amp"] - w["amp"]) <= rel_tol * w["amp"]
                and abs(a["period"] - w["period"]) <= rel_tol * w["period"]
            ):
                return "cycle"
    return None


def _integrate_settled(model, X0, spec, settle, progress=True, verbose=False, partial=False):
    """Paragem adaptativa (issue 041): integra em janelas de `window` segundos a partir do
    estado final da anterior e pára, depois de `t_min`, quando o sinal monitorizado fica
    permanente ou o ciclo se repete (`settled`); `tf` é o tempo máximo. As saídas ficam nos
    instantes de `dt_out`, o acumulador dos fluxos de contorno é contínuo entre janelas e a
    informação junta as contagens e a lista de assinaturas em info["settle"]."""
    opt = dict(SETTLE_DEFAULTS, **settle)
    window = float(opt["window"])
    t_min = float(opt.get("t_min", window))
    sig = dict(opt.get("signal") or {})
    t0, tf, dt = model.t0, model.tf, model.dt_out
    Ts, Xs, accs, windows = [np.array([t0])], [np.asarray(X0, float)[:, None]], [], []
    tot = dict(n_steps=0, n_rejected=0, n_rhs=0, n_jac=0, cpu=0.0)
    acc_off, reason, info = None, None, {}
    t_cur, X_cur = t0, np.asarray(X0, float)
    while t_cur < tf - 1e-9 * max(1.0, dt):
        t_end = min(t_cur + window, tf)
        te = output_times(t_cur, t_end, dt)
        T, X, info = _integrate(model, X_cur, te, spec, progress=False, verbose=verbose, partial=partial)
        for k in tot:
            tot[k] += info.get(k, 0) or 0
        Ts.append(np.asarray(T)[1:])
        Xs.append(np.asarray(X)[:, 1:])
        if info.get("accum") is not None:
            a = np.asarray(info["accum"], float)
            if acc_off is None:
                accs.append(a)
            else:
                accs.append(a[:, 1:] + acc_off[:, None])
            acc_off = (accs[-1][:, -1]).copy()
        if not info.get("success", True) or np.asarray(T).size < 2:
            reason = "failure"
            break
        windows.append(window_signature(np.asarray(T), _signal(model, np.asarray(T), np.asarray(X), sig)))
        t_cur, X_cur = float(T[-1]), np.asarray(X)[:, -1]
        if t_cur - t0 >= t_min - 1e-9:
            reason = settled(windows, float(opt["amp_tol"]), float(opt["rel_tol"]))
            if reason:
                break
    out = dict(info)
    out.update(tot)
    out["success"] = reason != "failure" and info.get("success", True)
    if accs:
        out["accum"] = np.concatenate(accs, axis=1)
    out["settle"] = dict(reason=reason or "tf", t_end=float(t_cur), windows=windows)
    if progress:
        print(
            f"[settle] {out['settle']['reason']} em t={t_cur:.0f} s ({len(windows)} janelas, {tot['n_steps']} passos)"
        )
    return np.concatenate(Ts), np.concatenate(Xs, axis=1), out


def steady_start(model, jax=False):
    """Estado inicial `initial_state.source = "steady"` (issue 040): Newton para rhs(t0, X) = 0
    a partir de `from` ("edges": as condições iniciais do caso, por exemplo o estacionário de um
    ponto vizinho de uma varredura; "guess": a estimativa automática), sem continuação
    pseudo-transiente (não converge para estacionários instáveis, que são os que interessam a
    um mapa de estabilidade). `perturb` multiplica as pressões por (1 + perturb) para tirar a
    trajetória de um equilíbrio instável; `on_failure` = "error" (padrão) ou "start" (segue a
    partir de `from`). Devolve (X0, info do Newton)."""
    from .initial_guess import initial_guess
    from .steady import newton_steady

    opts = dict(model.network.initial_options or {})
    src = opts.get("from", "edges")
    if src not in ("edges", "guess"):
        raise ValueError(f"initial_state.from inválido: {src!r}")
    X_start = initial_guess(model) if src == "guess" else model.initial_state()
    X, info = newton_steady(
        model,
        X_start,
        t=model.t0,
        tol=float(opts.get("tol", 1e-9)),
        verbose=False,
        bounds=state_bounds(model),
        jax=jax,
        max_iter=int(opts.get("max_iter", 100)),
    )
    out = dict(
        source="steady",
        start=src,
        converged=bool(info["converged"]),
        iterations=int(info["iterations"]),
        residual=float(info["residual"]),
    )
    if not info["converged"]:
        if opts.get("on_failure", "error") != "start":
            raise RuntimeError(
                f"estado estacionário não convergiu a partir de '{src}' (|F| = {info['residual']:.2e}); "
                "use initial_state.on_failure = 'start' para seguir a partir do estado de partida"
            )
        X = np.asarray(X_start, float)
    perturb = float(opts.get("perturb", 0.0))
    if perturb:
        X = np.array(X, float)
        X[model.state_slots() == 1] *= 1.0 + perturb
    out["perturb"] = perturb
    return X, out
