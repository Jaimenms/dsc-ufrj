"""Integrador implícito próprio: BDF de ordem variável (1-2) com passo
adaptativo, Newton modificado e Jacobiana esparsa por coloração.

Formulação (passo h, ordem k):
   k=1: X_{n+1} - X_n = h f(t_{n+1}, X_{n+1})
   k=2 (passo variável, razão r = h/h_old):
        X_{n+1} - (1+r)^2/(1+2r) X_n + r^2/(1+2r) X_{n-1} = h (1+r)/(1+2r) f(t_{n+1}, X_{n+1})

Estimativa de erro: diferença entre a solução e o preditor por extrapolação.
Após cada passo os estados são projetados nos limites físicos opcionais.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .jacobian import color_groups, colored_jacobian


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
    `t_new`, e a constante C de Milne do BDF2 nestes nós: erro local ≈ C·(x − pred) (issue 058). A constante sai da
    aplicação do corretor e do preditor a t³: o resíduo do corretor R = Σ α_j τ_j³ (α_j = derivadas das bases de
    Lagrange nos nós do corretor) dá erro −R/α_0; o do preditor, o produto das distâncias aos seus nós; C é a razão
    entre o primeiro e a diferença dos dois. No passo constante, C = 2/11."""
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


class ImplicitBDF:
    def __init__(
        self,
        fun,
        S=None,
        rtol=1e-4,
        atol=1e-6,
        max_step=np.inf,
        min_step=1e-8,
        newton_tol=1e-3,
        max_newton=12,
        max_order=2,
        bounds=None,
        verbose=False,
        jac_reuse=20,
        fixed_step=None,
        transform=None,
        inverse=None,
        transform_jac=None,
        jac_fun=None,
    ):
        """fixed_step: se dado, integra com passo fixo (Euler implícito se
        max_order=1) sem controle de erro; o passo só é reduzido temporariamente
        quando o Newton falha.

        transform / inverse: mapa conservativo T(x) (e seu inverso) entre o estado
        primitivo x e as quantidades conservadas (p. ex. massas por volume nas
        células). Quando dados, o resíduo de cada passo é escrito como
            T(x_{n+1}) - c_hist[T] - gamma f(x_{n+1}) = 0
        (com f fornecendo d T/dt para os componentes transformados), o que torna
        o balanço discreto exatamente conservativo mantendo o Newton nas
        variáveis primitivas (suaves). T é local (bloco-diagonal), com a mesma
        esparsidade da Jacobiana de f."""
        self.fun = fun
        self.fixed_step = fixed_step
        self.transform = transform
        self.inverse = inverse
        self.transform_jac = transform_jac
        self.slow_exit = True  # recálculo da Jacobiana dentro do Newton quando a convergência é lenta (necessário nos casos de rede com eventos, ex. gas lift; os scripts do severe slug desligam)
        self.predict_space = "x"  # preditor e erro local em x (primitivas) ou "T" (conservadas)
        self.error_space = None  # espaço da estimativa de erro local (None: o mesmo do preditor)
        self.accum = None  # g(t, x) opcional, integrado com a recorrência BDF (ver integrate)
        self._hist_w = (1.0, 0.0)
        self._last_err_vec = None
        self.jt_refresh = (
            "call"  # dT/dx reavaliada: a cada Newton ("call"), a cada iteração ("iter") ou só com J ("jac")
        )
        self.fd_rel, self.fd_abs = 1e-7, 1e-9  # passo das diferenças finitas da Jacobiana colorida
        self.S = S
        self.rtol, self.atol = rtol, atol
        self.max_step, self.min_step = max_step, min_step
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.max_order = max_order
        self.bounds = bounds
        self.verbose = verbose
        self.jac_reuse = jac_reuse
        self.jac_fun = jac_fun  # opcional: jac_fun(t, x) -> Jacobiana densa/esparsa de f (ex.: JAX)
        self.cooldown_steps = 3  # passos sem crescimento do passo após uma rejeição
        # issue 053 (opções, omissão = comportamento anterior):
        self.newton_damping = "legacy"  # "legacy": amortece só se o resíduo cresce (nunca na 1.ª iteração); "armijo": sempre;  # "armijo_tol": Armijo só enquanto a correção está acima de newton_tol (issue 055)
        self.jac_policy = "legacy"  # "legacy": J reusada jac_reuse passos; "step": J nova em cada passo; "rate": nova dentro do Newton quando a contração é fraca
        self.predictor = (
            "extrapolate"  # chute inicial do Newton: "extrapolate" (preditor BDF), "none" (x_n), "euler" (x_n + h f_n)
        )
        # issue 058: estimativa do erro local na ordem 2 — "thesis": preditor linear e r/(1 + 2r)·(x − pred), que é
        # O(h²) e controla o BDF2 como um método de 1.ª ordem; "milne": preditor quadrático pelos três estados do
        # histórico e constante de Milne das posições dos nós (só para a estimativa: o Newton parte do linear)
        self.error_estimate = "thesis"
        self._err_coef = 0.5
        self._pred_lin = None
        # issue 059: "state" propaga S = ∂x_n/∂x_0 (n × n) com a recorrência de cada passo aceite, com uma Jacobiana
        # própria no estado aceite (a do Newton não muda: a trajetória é a mesma com e sem sensibilidades); S final
        # em info["S"] — a matriz de monodromia quando se integra um período
        self.sensitivity = None
        self.jac_rate = 0.3  # "rate": contração err/err_prev acima da qual J é reavaliada no iterado atual
        self.jac_refresh_max = 3  # "rate": reavaliações por Newton, no máximo
        self.n_backtracks = 0  # diagnóstico: reduções de passo de Newton pela busca em linha
        self.max_cpu = None  # s: orçamento de tempo de relógio da integração (None = sem limite); excedido -> erro com a trajetória parcial
        self.groups = color_groups(S) if S is not None else None
        self.n_rhs = 0
        self.n_jac = 0
        self.n_steps = 0
        self.n_rejected = 0
        self._J = None
        self._LU = None
        self._LU_h = None
        self._jac_age = 0
        self._slow = False

    def _f(self, t, x):
        self.n_rhs += 1
        return self.fun(t, x)

    def _T(self, x):
        return x if self.transform is None else self.transform(x)

    def _Tinv(self, y):
        return y if self.inverse is None else self.inverse(y)

    def _jacobian(self, t, x, f0):
        n = x.size
        if self.transform is not None and self.transform_jac is not None:
            self._JT = sp.csc_matrix(self.transform_jac(x))
        elif self.transform is not None:
            # Jacobiana (bloco-diagonal) do mapa conservativo
            self._JT = (
                colored_jacobian(self.transform, x, f0=self.transform(x), S=self.S, groups=self.groups)
                if self.S is not None
                else sp.csc_matrix(
                    np.column_stack(
                        [
                            (self.transform(x + np.eye(n)[j] * (1e-7 * abs(x[j]) + 1e-9)) - self.transform(x))
                            / (1e-7 * abs(x[j]) + 1e-9)
                            for j in range(n)
                        ]
                    )
                )
            )
        if self.jac_fun is not None:
            self._J = sp.csc_matrix(self.jac_fun(t, x))
        elif self.S is None:
            J = np.empty((n, n))
            for j in range(n):
                h = 1e-7 * abs(x[j]) + 1e-9
                xp = x.copy()
                xp[j] += h
                J[:, j] = (self._f(t, xp) - f0) / h
            self._J = sp.csc_matrix(J)
        else:
            self._J = colored_jacobian(
                lambda y: self._f(t, y), x, f0=f0, S=self.S, groups=self.groups, rel=self.fd_rel, abs_=self.fd_abs
            )
        self.n_jac += 1
        self._jac_age = 0
        self._LU = None

    def _jac_at(self, t, x, f0):
        """Jacobiana (csc) de f em (t, x) à parte da do Newton: não toca em `_J`, `_LU` nem `_jac_age` (issue 059)."""
        n = x.size
        if self.jac_fun is not None:
            J = sp.csc_matrix(self.jac_fun(t, x))
        elif self.S is None:
            J = np.empty((n, n))
            for j in range(n):
                h = 1e-7 * abs(x[j]) + 1e-9
                xp = x.copy()
                xp[j] += h
                J[:, j] = (self._f(t, xp) - f0) / h
            J = sp.csc_matrix(J)
        else:
            J = colored_jacobian(
                lambda y: self._f(t, y), x, f0=f0, S=self.S, groups=self.groups, rel=self.fd_rel, abs_=self.fd_abs
            )
        self.n_jac += 1
        return J

    def _JT_dense(self, x):
        """Jacobiana densa do mapa conservativo em x (identidade sem mapa; issue 059)."""
        if self.transform is None:
            return np.eye(x.size)
        if self.transform_jac is None:
            raise NotImplementedError(
                "sensibilidades (issue 059) exigem transform_jac, a Jacobiana do mapa conservativo"
            )
        return np.asarray(sp.csc_matrix(self.transform_jac(x)).todense())

    def _factor(self, gamma):
        n = self._J.shape[0]
        I_or_JT = sp.identity(n, format="csc") if self.transform is None else self._JT
        A = (I_or_JT - gamma * self._J).tocsc()
        self._LU = spla.splu(A)
        self._LU_h = gamma

    def _project(self, x):
        if self.bounds is None:
            return x
        lo, hi = self.bounds
        return np.minimum(np.maximum(x, lo), hi)

    def _newton(self, t, h, pred, c_hist, gamma_coef, x_start=None):
        """Resolve X - c_hist = gamma_coef h f(t, X) a partir do preditor (ou de `x_start`, issue 053)."""
        x = pred.copy() if x_start is None else np.asarray(x_start, float).copy()
        gamma = gamma_coef * h
        if self.transform is not None and self.transform_jac is not None and self.jt_refresh in ("call", "iter"):
            # a Jacobiana (analítica, barata) do mapa conservativo é reavaliada no
            # preditor: dT/dx muda muito quando alphaG e P variam depressa, e um
            # valor envelhecido (da última Jacobiana de f) degrada o Newton
            self._JT = sp.csc_matrix(self.transform_jac(x))
            self._factor(gamma)
        elif self._LU is None or self._LU_h != gamma:
            self._factor(gamma)
        scale = self.atol + self.rtol * np.abs(pred)
        self._clipped = False
        err_prev = np.inf
        n_grow = 0
        n_refresh = 0
        self._err_hist = []  # diagnóstico: erro (RMS escalado) por iteração
        self._last_dx = None
        for it in range(self.max_newton):
            try:
                f = self._f(t, x)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                self._fail_reason = "exception"
                return x, False, it + 1
            if not np.all(np.isfinite(f)):
                self._fail_reason = "nonfinite_f"
                return x, False, it + 1
            res = self._T(x) - c_hist - gamma * f
            if it > 0 and self.transform is not None and self.transform_jac is not None and self.jt_refresh == "iter":
                self._JT = sp.csc_matrix(self.transform_jac(x))
                self._factor(gamma)
            dx = self._LU.solve(res)
            if not np.all(np.isfinite(dx)):
                self._fail_reason = "nonfinite_dx"
                return x, False, it + 1
            err = np.sqrt(np.mean((dx / scale) ** 2))
            self._err_hist.append(float(err))
            self._last_dx = dx
            self._last_dx_scaled = dx / scale  # diagnóstico: componentes que dominam a última atualização de Newton
            # divergência: atualização enorme ou crescendo
            if err > 1e6:
                self._fail_reason = "huge"
                return x, False, it + 1
            if err > 2.0 * err_prev:
                n_grow += 1
                if n_grow >= 2:
                    self._fail_reason = "diverge"
                    return x, False, it + 1
            # issue 053, "rate": contração fraca -> Jacobiana nova no iterado atual (até 3 vezes por Newton)
            if self.jac_policy == "rate" and it >= 1 and err > self.jac_rate * err_prev and err > self.newton_tol:
                if n_refresh < self.jac_refresh_max:
                    self._jacobian(t, x, f)
                    self._factor(gamma)
                    n_refresh += 1
                    err_prev = np.inf
                    continue
            # convergência lenta (Jacobiana desatualizada ou forte não-linearidade):
            # recalcula a Jacobiana no iterado atual e continua (até 2 vezes);
            # depois disso desiste para que o chamador reduza o passo
            if self.slow_exit and it >= 2 and err > 0.5 * err_prev and err > self.newton_tol:
                if n_refresh < 2:
                    self._jacobian(t, x, f)
                    self._factor(gamma)
                    n_refresh += 1
                    err_prev = np.inf
                    continue
                self._slow = True
                self._fail_reason = "slow"
                return x, False, it + 1
            err_prev_bt = err_prev
            err_prev = err
            # amortecimento (só quando a convergência não é monotônica): se a norma
            # do resíduo cresce, reduz o passo de Newton
            res_norm = np.sqrt(np.mean((res / scale) ** 2))
            lam = 1.0
            x_try = self._project(x - dx)
            if self.newton_damping == "armijo_tol" and err < self.newton_tol:
                pass  # issue 055: correção já convergida, aceita sem exigir descida do resíduo (que pode não descer)
            elif self.newton_damping in ("armijo", "armijo_tol"):
                # issue 053: busca em linha de Armijo em todas as iterações (também na primeira, a que diverge nas
                # frentes): aceita o passo se o resíduo escalado desce; senão reduz lambda até 4 vezes
                accepted = False
                for _bt in range(5):
                    if res_norm <= 1e-12:
                        accepted = True
                        break
                    try:
                        f_try = self._f(t, x_try)
                    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                        f_try = None
                    if f_try is not None and np.all(np.isfinite(f_try)):
                        res_try = self._T(x_try) - c_hist - gamma * f_try
                        if np.sqrt(np.mean((res_try / scale) ** 2)) <= (1.0 - 1e-4 * lam) * res_norm:
                            accepted = True
                            break
                    lam *= 0.5
                    self.n_backtracks += 1
                    x_try = self._project(x - lam * dx)
                if not accepted:
                    self._fail_reason = "diverge"
                    return x, False, it + 1
            else:
                for _bt in range(3):
                    if res_norm <= 1e-12 or (it == 0) or err < 0.5 * err_prev_bt:
                        break
                    try:
                        f_try = self._f(t, x_try)
                    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                        f_try = None
                    if f_try is not None and np.all(np.isfinite(f_try)):
                        res_try = self._T(x_try) - c_hist - gamma * f_try
                        if np.sqrt(np.mean((res_try / scale) ** 2)) <= 2.0 * res_norm:
                            break
                    lam *= 0.5
                    x_try = self._project(x - lam * dx)
            xp = x_try
            if np.any(xp != x - lam * dx):
                self._clipped = True
            x = xp
            if err < self.newton_tol:
                self._fail_reason = None
                return x, True, it + 1
        self._fail_reason = "maxit"
        return x, False, self.max_newton

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
        # acumulador opcional: integra g(t, x) com a MESMA recorrência BDF dos estados
        # (I_{n+1} = w1 I_n + w2 I_{n-1} + gamma h g_{n+1}); usado para verificar a
        # conservação de massa de forma exata (fluxos de contorno vs estoque)
        acc_hist = [np.zeros(np.asarray(self.accum(t, x), float).shape)] if self.accum is not None else None
        out_acc = [acc_hist[0].copy()] if acc_hist is not None else None
        self._out, self._t_eval = out, t_eval  # referências vivas: trajetória parcial acessível se a integração falhar
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
        sens = self.sensitivity == "state"
        if sens:
            S_prev2, S_prev = None, np.eye(x.size)
            JTS_prev2, JTS_prev = None, self._JT_dense(x) @ S_prev
        while t < t_end - 1e-12 * max(1.0, abs(t_end)):
            h = min(h, self.max_step, t_end - t)
            if next_bp < bps.size and t + h > bps[next_bp] - 1e-12:
                h = max(bps[next_bp] - t, self.min_step)
                hit_bp = True
            else:
                hit_bp = False
            t_new = t + h

            def predict(ordr):
                # histórico (c_hist) sempre nas variáveis conservadas T(x): é ele que
                # garante o balanço exato; o preditor (chute inicial do Newton) pode
                # ser extrapolado nas variáveis primitivas x (predict_space="x"), o
                # que evita a amplificação, por T^-1, de erros de extrapolação da
                # massa de líquido em erros grandes de pressão (fator gamma_L)
                Tx = self._T(x)
                in_T = self.transform is not None and self.predict_space == "T"
                if ordr == 1 or len(hist) < 2:
                    pred_ = self._Tinv(Tx + h * f0) if in_T else x + h * f0
                    self._hist_w = (1.0, 0.0)
                    self._err_coef, self._pred_lin = 0.5, pred_
                    return pred_, Tx, 1.0, 1.0
                h_old = t_hist[-1] - t_hist[-2]
                rr = h / h_old
                xm1 = hist[-2]
                Txm1 = self._T(xm1)
                if in_T:
                    pred_ = self._Tinv(Tx + rr * (Tx - Txm1))
                else:
                    pred_ = x + rr * (x - xm1)
                w1, w2 = (1 + rr) ** 2 / (1 + 2 * rr), -(rr**2) / (1 + 2 * rr)
                self._hist_w = (w1, w2)
                self._err_coef, self._pred_lin = rr / (1 + 2 * rr), pred_
                if self.error_estimate == "milne" and len(hist) >= 3:
                    # issue 058: o preditor quadrático só serve à estimativa do erro; o Newton parte do linear
                    vals = (self._T(hist[-3]), Txm1, Tx) if in_T else (hist[-3], xm1, x)
                    pred_, self._err_coef = milne_order2(t_new, t_hist, vals)
                    if in_T:
                        pred_ = self._Tinv(pred_)
                return pred_, w1 * Tx + w2 * Txm1, (1 + rr) / (1 + 2 * rr), rr

            if at_bound:
                order = 1  # histórico do BDF2 inconsistente com estados projetados nos limites
            pred, c_hist, gcoef, r = predict(order)
            self._slow = False
            if self.jac_policy == "step" and self._jac_age > 0:
                self._jacobian(t, x, f0)  # issue 053: Jacobiana nova em cada passo (no estado aceite)
            # issue 053: iterado inicial do Newton; o preditor extrapolado continua a servir à estimativa de erro
            x_start = None
            if self.predictor == "none":
                x_start = x
            elif self.predictor == "euler":
                x_start = self._project(x + h * f0)
            elif self.error_estimate == "milne":
                x_start = self._pred_lin  # issue 058: do quadrático o Newton diverge nas frentes
            x_new, ok, nit = self._newton(t_new, h, pred, c_hist, gcoef, x_start=x_start)
            if not ok and self._clipped and order > 1:
                # projeção ativa: repete o passo com BDF1 (sem extrapolação do histórico)
                order = 1
                pred, c_hist, gcoef, r = predict(order)
                if self.predictor == "extrapolate" and self.error_estimate == "milne":
                    x_start = self._pred_lin
                x_new, ok, nit = self._newton(t_new, h, pred, c_hist, gcoef, x_start=x_start)
            if not ok:
                # tenta com Jacobiana atualizada (no preditor, se finito)
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
                h *= 0.5 if n_fail_row == 0 else 0.25
                self.n_rejected += 1
                n_fail_row += 1
                cooldown = self.cooldown_steps
                if n_fail_row >= 2:
                    order = 1  # recua para BDF1 (mais robusto) após falhas seguidas
                if h < self.min_step:
                    raise RuntimeError(f"passo mínimo atingido em t={t:.4g}")
                continue
            # erro local: diferença solução - preditor (BDF: err ~ C (x - pred))
            if self.transform is not None and (self.error_space or self.predict_space) == "T":
                # erro medido nas quantidades conservadas (a inversa T^-1 amplifica,
                # por gamma_L, erros de truncamento da massa de líquido em pressão)
                xe, pe = self._T(x_new), self._T(pred)
                scale = self.atol + self.rtol * np.maximum(np.abs(xe), np.abs(self._T(x)))
            else:
                xe, pe = x_new, pred
                scale = self.atol + self.rtol * np.maximum(np.abs(x_new), np.abs(x))
            if order == 1:
                err_est = 0.5 * (xe - pe)
            else:
                # r/(1 + 2r) com o preditor linear (tese) ou a constante de Milne com o quadrático (058)
                err_est = self._err_coef * (xe - pe) if len(hist) >= 2 else 0.5 * (xe - pe)
            err = np.sqrt(np.mean((err_est / scale) ** 2))
            self._last_err_vec = err_est / scale  # diagnóstico
            if self.fixed_step:
                err = 0.5  # sem controle de erro em passo fixo
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
            # aceita
            n_fail_row = 0
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
            if sens:
                # issue 059: (J_T − γJ)(x_{n+1}) S_{n+1} = w1 J_T(x_n) S_n + w2 J_T(x_{n−1}) S_{n−1}, a derivada do
                # passo aceite (ordem e pesos do histórico iguais aos do passo); estados presos num limite não
                # dependem do estado inicial
                w1, w2 = self._hist_w
                rhs_s = w1 * JTS_prev
                if w2 != 0.0 and JTS_prev2 is not None:
                    rhs_s = rhs_s + w2 * JTS_prev2
                JT_acc = self._JT_dense(x)
                A_s = (sp.csc_matrix(JT_acc) - gcoef * h * self._jac_at(t, x, f0)).tocsc()
                S_new = spla.splu(A_s).solve(np.asarray(rhs_s, float))
                if self.bounds is not None:
                    S_new[(x <= self.bounds[0]) | (x >= self.bounds[1]), :] = 0.0
                S_prev2, S_prev = S_prev, S_new
                JTS_prev2, JTS_prev = JTS_prev, JT_acc @ S_new
            self.n_steps += 1
            if self.max_cpu is not None and time.time() - t_start > self.max_cpu:
                raise RuntimeError(f"orçamento de tempo ({self.max_cpu:g} s) excedido em t={t:.4g}")
            self._jac_age += 1
            if self._jac_age >= self.jac_reuse or nit > 4:
                self._jacobian(t, x, f0)
            # saída
            while i_out < t_eval.size and t_eval[i_out] <= t + 1e-12 * max(1.0, abs(t)):
                if abs(t_eval[i_out] - t) < 1e-9 * max(1.0, abs(t)) or len(hist) < 2:
                    out.append(x.copy())
                    if out_acc is not None:
                        out_acc.append(acc_hist[-1].copy())
                else:
                    # interpolação linear entre os dois últimos pontos
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
                # após uma rejeição, não deixa o passo crescer por alguns passos
                # (evita o ciclo "sucesso em h, falha em 2h")
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
                if h < 1e-3 and self._last_err_vec is not None:
                    # diagnóstico de passo rastejante: componentes que dominam o erro local e a última atualização de Newton
                    ev = np.abs(self._last_err_vec)
                    top = np.argsort(ev)[-5:][::-1]
                    print(
                        "    erro local dominante (índice: erro/escala, x):",
                        ", ".join(f"{int(i)}: {ev[i]:.2f}, {x[i]:.4g}" for i in top),
                    )
                    dxs = getattr(self, "_last_dx_scaled", None)
                    if dxs is not None:
                        ad = np.abs(dxs)
                        topn = np.argsort(ad)[-5:][::-1]
                        print(
                            f"    último Newton ({getattr(self, '_fail_reason', '?')}, {len(self._err_hist)} it, erros {np.round(self._err_hist[-4:], 3).tolist()}): dx/escala dominante:",
                            ", ".join(f"{int(i)}: {ad[i]:.2f}" for i in topn),
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
            info["accum"] = np.array(out_acc).T  # (m, nt): integral BDF-consistente de g
        if sens:
            info["S"] = S_prev  # (n, n): ∂x(t_end)/∂x(t_0) da trajetória discreta (issue 059)
        return t_eval, X, info


def integrate_implicit(
    model,
    X0,
    t_eval,
    rtol=1e-4,
    atol=1e-6,
    max_step=np.inf,
    bounds=None,
    verbose=False,
    progress=True,
    jax=False,
    tol_range=None,
    **kw,
):
    """Integra o modelo com o BDF implícito; jax=True usa o lado direito compilado e a
    Jacobiana exata (mfsim.compiled) em vez de diferenças finitas coloridas.

    tol_range: tolerância absoluta como fração da faixa de cada estado
    (model.state_scales: 1 para frações, pressão e vazões de referência do caso);
    substitui `atol` e dispensa ajustar tolerâncias por variável."""
    S = model.jac_sparsity()
    if tol_range is not None:
        atol = float(tol_range) * model.state_scales(np.asarray(X0, float))
    if getattr(model, "conservative", False):
        kw.setdefault("transform", model.transform)
        kw.setdefault("inverse", model.inverse_transform)
        kw.setdefault("transform_jac", model.transform_jacobian)
    fun = model.rhs
    if jax:
        from ..compiled import get_jax

        fun, jac_fun = get_jax(model)
        kw.setdefault("jac_fun", jac_fun)
    bdf = ImplicitBDF(fun, S=S, rtol=rtol, atol=atol, max_step=max_step, bounds=bounds, verbose=verbose, **kw)
    T, X, info = bdf.integrate(t_eval, X0, breakpoints=model.spec_breakpoints())
    if progress:
        print(
            f"[implicit BDF] steps={info['n_steps']} rejected={info['n_rejected']} rhs={info['n_rhs']} "
            f"jac={info['n_jac']} | {info['cpu']:.1f}s"
        )
    return T, X, info
