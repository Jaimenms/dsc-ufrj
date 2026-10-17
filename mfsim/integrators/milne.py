"""BDF implícito com a estimativa de erro de Milne na ordem 2 (issue 058).

O `ImplicitBDF` extrapola o preditor da ordem 2 por dois pontos e estima o erro local por r/(1 + 2r)·(x − pred); como
x − pred é O(h²), a estimativa é a de um método de 1.ª ordem e o custo cresce como rtol^(−1/2). Aqui o preditor é o
polinómio quadrático pelos três estados do histórico (que o integrador já guarda) e a constante do erro local vem das
posições dos nós (dispositivo de Milne; 2/11 no passo constante), o que devolve ao BDF2 o escalamento rtol^(−1/3) e,
no problema-modelo, 20–50 % menos tentativas a erro global igual. Na ordem 1, nos primeiros passos e depois de uma
projeção (histórico de dois pontos) nada muda.

Ficheiro à parte por decisão do dono (2026-10-17): `implicit.py` estava em edição noutra sessão. `integrate` é uma
cópia do laço da classe-mãe com as alterações marcadas «058»; `integrate_bdf_milne` replica a construção do
`mfsim.simulation._integrate_bdf` com esta classe. Quando o ficheiro estiver livre, isto dobra-se numa opção
`integration.error_estimate = "thesis" | "milne"` do `ImplicitBDF` e esta cópia desaparece.
"""

from __future__ import annotations

import time

import numpy as np

from .implicit import ImplicitBDF


def _dlagrange(tau, j):
    """Derivada em 0 do polinómio de base de Lagrange do nó j (nós `tau`, com tau[0] = 0 o novo instante)."""
    s = 0.0
    for m in range(tau.size):
        if m == j:
            continue
        prod = 1.0
        for kk in range(tau.size):
            if kk in (j, m):
                continue
            prod *= (0.0 - tau[kk]) / (tau[j] - tau[kk])
        s += prod / (tau[j] - tau[m])
    return s


def milne_order2(t_new, t_hist, vals):
    """Preditor quadrático pelos três últimos pontos ((t_hist[-3], vals[0]) ... (t_hist[-1], vals[2])) avaliado em
    `t_new`, e a constante C de Milne do BDF2 nestes nós: erro local ≈ C·(x − pred). A constante sai da aplicação do
    corretor e do preditor a t³: o resíduo do corretor R = Σ α_j τ_j³ (α_j = derivadas das bases de Lagrange nos nós
    do corretor) dá erro −R/α_0; o do preditor, o produto das distâncias aos seus nós; C é a razão entre o primeiro
    e a diferença dos dois. No passo constante, C = 2/11."""
    t2, t1, t0 = t_hist[-3], t_hist[-2], t_hist[-1]
    v2, v1, v0 = vals
    w0 = (t_new - t1) * (t_new - t2) / ((t0 - t1) * (t0 - t2))
    w1 = (t_new - t0) * (t_new - t2) / ((t1 - t0) * (t1 - t2))
    w2 = (t_new - t0) * (t_new - t1) / ((t2 - t0) * (t2 - t1))
    pred = w0 * v0 + w1 * v1 + w2 * v2
    tau = np.array([0.0, t0 - t_new, t1 - t_new])
    alpha = np.array([_dlagrange(tau, j) for j in range(3)])
    lte = -(alpha[1] * tau[1] ** 3 + alpha[2] * tau[2] ** 3) / alpha[0]
    prod = (t_new - t0) * (t_new - t1) * (t_new - t2)
    return pred, lte / (lte + prod)


class ImplicitBDFMilne(ImplicitBDF):
    """`ImplicitBDF` com preditor quadrático e constante de Milne na ordem 2 (issue 058).

    Opções próprias (atributos): `newton_start` — iterado inicial do Newton, "linear" (a extrapolação por dois pontos
    da classe-mãe; omissão: nas frentes o preditor quadrático erra por mais e o Newton diverge dele) ou "quadratic"
    (o preditor de Milne); `newton_fail_cap` — depois de uma falha do Newton no passo h_f, o passo fica limitado a
    `newton_fail_cap`·h_f e o limite cresce 1,25× por passo aceite durante `newton_fail_steps` passos (None: só o
    arrefecimento da classe-mãe, que deixa o passo voltar a h_f em dois passos e falhar outra vez)."""

    estimate = "milne"  # "thesis": preditor linear e r/(1 + 2r), como a classe-mãe (para isolar as outras opções)
    newton_start = "linear"
    newton_fail_cap = None
    newton_fail_steps = 8

    def integrate(self, t_eval, x0, first_step=None, breakpoints=()):
        t_eval = np.asarray(t_eval, float)
        x = self._project(np.asarray(x0, float).copy())
        t = t_eval[0]
        t_end = t_eval[-1]
        bps = np.array(sorted(set(np.asarray(breakpoints, float).tolist())))
        bps = bps[(bps > t) & (bps < t_end)]
        out = [x.copy()]
        i_out = 1
        hist = [x.copy()]
        t_hist = [t]
        acc_hist = [np.zeros(np.asarray(self.accum(t, x), float).shape)] if self.accum is not None else None
        out_acc = [acc_hist[0].copy()] if acc_hist is not None else None
        self._out, self._t_eval = out, t_eval
        h = first_step or min(1e-2 * (t_eval[1] - t_eval[0]) if t_eval.size > 1 else 1e-2, self.max_step)
        if self.fixed_step:
            h = min(h, self.fixed_step)
        order = 1
        t_start = time.time()
        f0 = self._f(t, x)
        self._jacobian(t, x, f0)
        next_bp = 0
        n_fail_row = 0
        cooldown = 0
        at_bound = False
        self._clipped = False
        self._err_coef = 0.5
        h_cap, cap_steps = np.inf, 0  # 058: tampão do passo depois de uma falha do Newton
        while t < t_end - 1e-12 * max(1.0, abs(t_end)):
            if cap_steps > 0:
                h = min(h, h_cap)
            h = min(h, self.max_step, t_end - t)
            if next_bp < bps.size and t + h > bps[next_bp] - 1e-12:
                h = max(bps[next_bp] - t, self.min_step)
                hit_bp = True
            else:
                hit_bp = False
            t_new = t + h

            def predict(ordr):
                Tx = self._T(x)
                in_T = self.transform is not None and self.predict_space == "T"
                if ordr == 1 or len(hist) < 2:
                    pred_ = self._Tinv(Tx + h * f0) if in_T else x + h * f0
                    self._hist_w = (1.0, 0.0)
                    self._err_coef = 0.5
                    self._pred_lin = pred_
                    return pred_, Tx, 1.0, 1.0
                h_old = t_hist[-1] - t_hist[-2]
                rr = h / h_old
                xm1 = hist[-2]
                Txm1 = self._T(xm1)
                w1, w2 = (1 + rr) ** 2 / (1 + 2 * rr), -(rr**2) / (1 + 2 * rr)
                self._hist_w = (w1, w2)
                pred_lin = self._Tinv(Tx + rr * (Tx - Txm1)) if in_T else x + rr * (x - xm1)
                self._pred_lin = pred_lin
                if len(hist) >= 3 and self.estimate == "milne":
                    # 058: preditor quadrático pelos três estados do histórico e constante de Milne
                    vals = (self._T(hist[-3]), Txm1, Tx) if in_T else (hist[-3], xm1, x)
                    pred_, self._err_coef = milne_order2(t_new, t_hist, vals)
                    if in_T:
                        pred_ = self._Tinv(pred_)
                else:
                    pred_ = pred_lin
                    self._err_coef = rr / (1 + 2 * rr)
                return pred_, w1 * Tx + w2 * Txm1, (1 + rr) / (1 + 2 * rr), rr

            if at_bound:
                order = 1
            pred, c_hist, gcoef, r = predict(order)
            self._slow = False
            if self.jac_policy == "step" and self._jac_age > 0:
                self._jacobian(t, x, f0)
            x_start = None
            if self.predictor == "none":
                x_start = x
            elif self.predictor == "euler":
                x_start = self._project(x + h * f0)
            elif self.newton_start == "linear":
                x_start = self._pred_lin  # 058: o Newton parte da extrapolação linear; o quadrático só estima o erro
            x_new, ok, nit = self._newton(t_new, h, pred, c_hist, gcoef, x_start=x_start)
            if not ok and self._clipped and order > 1:
                order = 1
                pred, c_hist, gcoef, r = predict(order)
                x_start = (
                    self._pred_lin if self.newton_start == "linear" and self.predictor == "extrapolate" else x_start
                )
                x_new, ok, nit = self._newton(t_new, h, pred, c_hist, gcoef, x_start=x_start)
            if not ok:
                if self._jac_age > 0 or self._slow:
                    xj = pred if np.all(np.isfinite(pred)) else x
                    try:
                        fj = self._f(t_new, xj)
                        if not np.all(np.isfinite(fj)):
                            raise ValueError
                    except Exception:
                        xj, fj = x, f0
                    self._jacobian(t_new, xj, fj)
                    self._slow = False
                    x_new, ok, nit = self._newton(t_new, h, pred, c_hist, gcoef, x_start=x_start)
            if not ok or not np.all(np.isfinite(x_new)):
                if self.newton_fail_cap is not None:
                    h_cap, cap_steps = self.newton_fail_cap * h, self.newton_fail_steps
                h *= 0.5 if n_fail_row == 0 else 0.25
                self.n_rejected += 1
                n_fail_row += 1
                cooldown = self.cooldown_steps
                if n_fail_row >= 2:
                    order = 1
                if h < self.min_step:
                    raise RuntimeError(f"passo mínimo atingido em t={t:.4g}")
                continue
            if self.transform is not None and (self.error_space or self.predict_space) == "T":
                xe, pe = self._T(x_new), self._T(pred)
                scale = self.atol + self.rtol * np.maximum(np.abs(xe), np.abs(self._T(x)))
            else:
                xe, pe = x_new, pred
                scale = self.atol + self.rtol * np.maximum(np.abs(x_new), np.abs(x))
            # 058: a constante do erro local vem do preditor usado (0,5 na ordem 1; Milne ou r/(1 + 2r) na ordem 2)
            err_est = (0.5 if order == 1 else self._err_coef) * (xe - pe)
            err = np.sqrt(np.mean((err_est / scale) ** 2))
            self._last_err_vec = err_est / scale
            if self.fixed_step:
                err = 0.5
            if err > 1.0:
                h *= max(0.2, 0.9 * err ** (-1.0 / (order + 1)))
                self.n_rejected += 1
                n_fail_row += 1
                cooldown = self.cooldown_steps
                if n_fail_row >= 3:
                    order = 1
                if h < self.min_step:
                    raise RuntimeError(f"passo mínimo atingido em t={t:.4g}")
                continue
            n_fail_row = 0
            if cap_steps > 0:
                h_cap, cap_steps = 1.25 * h_cap, cap_steps - 1
            at_bound = self._clipped or (
                self.bounds is not None and bool(np.any((x_new <= self.bounds[0]) | (x_new >= self.bounds[1])))
            )
            if acc_hist is not None:
                w1, w2 = self._hist_w
                acc_new = (
                    w1 * acc_hist[-1]
                    + (w2 * acc_hist[-2] if w2 != 0.0 else 0.0)
                    + gcoef * h * np.asarray(self.accum(t_new, x_new), float)
                )
                acc_hist.append(acc_new)
                if len(acc_hist) > 3:
                    acc_hist.pop(0)
            t, x = t_new, x_new
            f0 = self._f(t, x)
            hist.append(x.copy())
            t_hist.append(t)
            if len(hist) > 3:
                hist.pop(0)
                t_hist.pop(0)
            self.n_steps += 1
            if self.max_cpu is not None and time.time() - t_start > self.max_cpu:
                raise RuntimeError(f"orçamento de tempo ({self.max_cpu:g} s) excedido em t={t:.4g}")
            self._jac_age += 1
            if self._jac_age >= self.jac_reuse or nit > 4:
                self._jacobian(t, x, f0)
            while i_out < t_eval.size and t_eval[i_out] <= t + 1e-12 * max(1.0, abs(t)):
                if abs(t_eval[i_out] - t) < 1e-9 * max(1.0, abs(t)) or len(hist) < 2:
                    out.append(x.copy())
                    if out_acc is not None:
                        out_acc.append(acc_hist[-1].copy())
                else:
                    tp, xp = t_hist[-2], hist[-2]
                    w = (t_eval[i_out] - tp) / (t - tp)
                    out.append(xp + w * (x - xp))
                    if out_acc is not None:
                        out_acc.append(acc_hist[-2] + w * (acc_hist[-1] - acc_hist[-2]))
                i_out += 1
            if hit_bp:
                next_bp += 1
                order = 1
            else:
                order = min(self.max_order, order + 1)
            fac = 0.9 * err ** (-1.0 / (order + 1)) if err > 0 else 2.0
            if cooldown > 0:
                fac = min(fac, 1.0)
                cooldown -= 1
            h *= min(2.0, max(0.3, fac))
            if self.fixed_step:
                h = min(self.fixed_step, h)
            if self.verbose and self.n_steps % 50 == 0:
                print(
                    f"  t={t:10.3f} h={h:9.3e} ord={order} steps={self.n_steps} rej={self.n_rejected} "
                    f"rhs={self.n_rhs} jac={self.n_jac} ({time.time()-t_start:.1f}s)"
                )
        X = np.array(out).T
        info = dict(
            success=True,
            n_steps=self.n_steps,
            n_rejected=self.n_rejected,
            n_rhs=self.n_rhs,
            n_jac=self.n_jac,
            cpu=time.time() - t_start,
        )
        if out_acc is not None:
            info["accum"] = np.array(out_acc).T
        return t_eval, X, info


def integrate_bdf_milne(model, X0, t_eval, spec, progress=True, verbose=False, partial=False, **milne_options):
    """Réplica de `mfsim.simulation._integrate_bdf` com o `ImplicitBDFMilne` (até a opção entrar no simulador);
    `milne_options` são atributos da subclasse (`newton_start`, `newton_fail_cap`, `newton_fail_steps`)."""
    from ..simulation import state_bounds

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
        from ..compiled import get_jax

        fun, jac_fun = get_jax(model)
    bdf = ImplicitBDFMilne(
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
    for key, val in milne_options.items():
        if not hasattr(ImplicitBDFMilne, key):
            raise TypeError(f"opção desconhecida do ImplicitBDFMilne: {key}")
        setattr(bdf, key, val)
    if spec.slow_exit is not None:
        bdf.slow_exit = bool(spec.slow_exit)
    if spec.max_cpu is not None:
        bdf.max_cpu = float(spec.max_cpu)
    if spec.predict_space is not None:
        bdf.predict_space = spec.predict_space
    if spec.jt_refresh is not None:
        bdf.jt_refresh = spec.jt_refresh
    for key in ("newton_damping", "jac_policy", "predictor"):
        if getattr(spec, key) is not None:
            setattr(bdf, key, str(getattr(spec, key)))
    if spec.jac_rate is not None:
        bdf.jac_rate = float(spec.jac_rate)
    if spec.jac_refresh_max is not None:
        bdf.jac_refresh_max = int(spec.jac_refresh_max)
    if spec.accumulate:
        bdf.accum = model.boundary_net_fluxes
    t_start = time.time()
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
            cpu=time.time() - t_start,
        )
    if progress and info.get("success", True):
        print(
            f"[implicit BDF, Milne] steps={info['n_steps']} rejected={info['n_rejected']} rhs={info['n_rhs']} "
            f"jac={info['n_jac']} | {info['cpu']:.1f}s"
        )
    return T, X, info
