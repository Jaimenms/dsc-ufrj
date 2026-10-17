"""Variantes do BDF no problema-modelo (estudo): o que faria o integrador do simulador gastar menos passos.

Sobre o protótipo em JAX da issue 056 (BDF de passo variável em `lax.while_loop`, Newton com Armijo e a regra da 055,
Jacobiana por `jacfwd`, LU denso) ensaiam-se, uma de cada vez e combinadas:

- **estimativa de erro de ordem consistente** («order»): na ordem 2 o simulador extrapola o preditor por dois pontos
  (erro O(h²)) e estima o erro local por r/(1 + 2r)·(x − pred), que é O(h²) e não o erro O(h³) do BDF2 — o controlo
  comporta-se como o de um método de 1.ª ordem. A alternativa extrapola por q + 1 pontos e usa a constante do erro
  que sai das posições dos nós (Milne), 2/11 no passo constante da ordem 2;
- **ordem até 5**: BDF de coeficientes variáveis (derivada do polinómio de Lagrange nos últimos q + 1 instantes),
  ordem a subir um degrau por passo aceite e a recuar como no simulador;
- **controlador de passo**: o elementar do simulador (0,9 err^(−1/(q+1)) em [0,3, 2], arrefecimento de 3 passos) ou o
  H211b de Söderlind (filtro PI nos dois últimos erros e na razão dos passos), com ou sem o limite de crescimento;
- **tolerância do Newton** (1e−3 ou 1e−2 na correção escalada).

Cada variante corre nos três casos oscilantes do artigo (a = 0,95, 0,85, 0,5) com rtol de 1e−3 a 1e−6 (atol =
rtol/100); o custo (passos, rejeições, iterações de Newton, relógio) é lido contra o erro global medido face a uma
referência Radau (scipy, rtol 1e−10) — diagramas trabalho–precisão em vez de contagens de passos soltas, porque a
estimativa de erro muda a exatidão entregue a tolerância igual.

Uso (grava em runs/paper01/toy/variants/; o guião das figuras só lê):

    uv run python papers/01-severe-slugging-two-fluid/toy/model_problem_bdf_variants.py run [--out DIR] [--R 5]
    uv run python papers/01-severe-slugging-two-fluid/toy/model_problem_bdf_variants.py plot [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_problem as mp  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.scipy.linalg import lu_factor, lu_solve  # noqa: E402

OUT = os.path.join(ROOT, "runs", "paper01", "toy", "variants")
REF_DIR = os.path.join(ROOT, "runs", "paper01", "toy", "reference")
WINDOW = 100.0
K_HIST = 7  # estados guardados (ordem máxima 5 precisa de 6 + o atual)

VARIANTS = {
    "thesis": dict(),  # o integrador do simulador (ordem 2, preditor linear, r/(1+2r), controlador elementar)
    "order2": dict(estimate="order"),
    "order3": dict(estimate="order", max_order=3),
    "order4": dict(estimate="order", max_order=4),
    "order5": dict(estimate="order", max_order=5),
    "thesis_h211b": dict(controller="h211b"),
    "order2_h211b": dict(estimate="order", controller="h211b"),
    "order5_h211b": dict(estimate="order", max_order=5, controller="h211b"),
    "thesis_grow5": dict(growth_max=5.0, cooldown=0),
    "order5_ntol1e-2": dict(estimate="order", max_order=5, newton_tol=1e-2),
}
RTOLS = (1e-3, 1e-4, 1e-5, 1e-6)
EARLY = 30.0  # s: janela inicial do erro «local» (antes de a deriva de fase se acumular)
A_CASES = (0.95, 0.85, 0.5)


def build(
    p,
    max_order=2,
    estimate="thesis",
    controller="basic",
    growth_max=2.0,
    cooldown=3,
    newton_tol=1e-3,
    max_step=0.5,
    max_newton=20,
    max_steps=200000,
):
    """`solve(a, rtol, atol, t_eval, y_guess)` compilado para uma variante (opções estáticas)."""
    N = int(p["N"])
    hx = 1.0 / N
    n = 2 * N
    Mv = jnp.asarray(mp.mass(p))
    k, Pr, R, g0, beta, y0 = (float(p[s]) for s in ("k", "Pr", "R", "g0", "beta", "y0"))
    min_step = 1e-10
    assert 1 <= max_order <= 5 and estimate in ("thesis", "order") and controller in ("basic", "h211b")
    if estimate == "thesis":
        assert max_order <= 2

    def fchar(y):
        return g0 - beta * (y - y0) + (y - y0) ** 3 / 3.0

    def dfchar(y):
        return -beta + (y - y0) ** 2

    def rhs(x, a):
        pc, w, y = x[:N], x[N : 2 * N - 1], x[2 * N - 1]
        wf = jnp.concatenate([jnp.array([k * (Pr - pc[0])]), w, jnp.array([y])])
        return jnp.concatenate(
            [-(wf[1:] - wf[:-1]) / hx, -(pc[1:] - pc[:-1]) / hx - R * w, jnp.array([pc[-1] - fchar(y) - a * y])]
        )

    jac = jax.jacfwd(rhs, argnums=0)

    def steady(a, y_guess):
        def g(y):
            return y - k * (Pr - (fchar(y) + a * y + R * y * (N - 1) * hx))

        def dg(y):
            return 1.0 + k * (dfchar(y) + a + R * (N - 1) * hx)

        y = lax.fori_loop(0, 6, lambda _, yy: yy - g(yy) / dg(yy), y_guess)
        pN = fchar(y) + a * y
        pc = pN + R * y * hx * jnp.arange(N - 1, -1, -1)
        return jnp.concatenate([pc, jnp.full(N - 1, y), jnp.array([y])])

    def rms(v, scale):
        return jnp.sqrt(jnp.mean((v / scale) ** 2))

    # ---------------------------------------------------------------- coeficientes do BDF de ordem q (q estático)
    def lagrange_weights(tau, j):
        """Peso do nó j do polinómio de Lagrange avaliado em 0 (nós tau)."""
        w = 1.0
        for kk in range(tau.shape[0]):
            if kk != j:
                w = w * (0.0 - tau[kk]) / (tau[j] - tau[kk])
        return w

    def lagrange_dweights(tau, j):
        """Derivada em 0 do polinómio de Lagrange do nó j (nós tau)."""
        s = 0.0
        for m in range(tau.shape[0]):
            if m == j:
                continue
            prod = 1.0
            for kk in range(tau.shape[0]):
                if kk in (j, m):
                    continue
                prod = prod * (0.0 - tau[kk]) / (tau[j] - tau[kk])
            s = s + prod / (tau[j] - tau[m])
        return s

    def make_branch(q):
        """(pred, c_hist, gamma, C_err) da ordem q a partir do histórico; o operando é (xh, th, t_new, h, f0)."""

        def branch(op):
            xh, th, t_new, h, f0 = op
            x = xh[0]
            if q == 1:
                pred = x + h * f0
                return pred, Mv * x, h, 0.5
            tau = jnp.concatenate([jnp.zeros(1), th[:q] - t_new])  # nós do corretor: 0 e os q últimos
            alpha = jnp.stack([lagrange_dweights(tau, j) for j in range(q + 1)])
            gamma = 1.0 / alpha[0]
            c_hist = -jnp.sum((alpha[1:] / alpha[0])[:, None] * (Mv[None, :] * xh[:q]), axis=0)
            if estimate == "thesis":  # q == 2: preditor linear e r/(1 + 2r)
                rr = h / (th[0] - th[1])
                pred = x + rr * (x - xh[1])
                return pred, c_hist, gamma, rr / (1.0 + 2.0 * rr)
            taup = th[: q + 1] - t_new  # nós do preditor: os q + 1 últimos
            wts = jnp.stack([lagrange_weights(taup, j) for j in range(q + 1)])
            pred = jnp.sum(wts[:, None] * xh[: q + 1], axis=0)
            Rres = jnp.sum(alpha[1:] * tau[1:] ** (q + 1))
            lte = -Rres / alpha[0]
            prod = jnp.prod(-taup)
            return pred, c_hist, gamma, lte / (lte + prod)

        return branch

    branches = [make_branch(q) for q in range(1, max_order + 1)]

    def newton(x_pred, c_hist, gamma, a, scale):
        A = jnp.diag(Mv) - gamma * jac(x_pred, a)
        lu = lu_factor(A)

        def resid(x):
            return Mv * x - c_hist - gamma * rhs(x, a)

        def cond(st):
            x, it, done, fail, err_prev, n_grow, nbt = st
            return (~done) & (~fail) & (it < max_newton)

        def body(st):
            x, it, done, fail, err_prev, n_grow, nbt = st
            res = resid(x)
            dx = lu_solve(lu, res)
            err = rms(dx, scale)
            res_norm = rms(res, scale)

            def bt_cond(s):
                lam, acc, kk = s
                return (~acc) & (kk < 5)

            def bt_body(s):
                lam, acc, kk = s
                r_try = rms(resid(x - lam * dx), scale)
                acc = (r_try <= (1.0 - 1e-4 * lam) * res_norm) | (res_norm <= 1e-12)
                lam = jnp.where(acc, lam, 0.5 * lam)
                return lam, acc, kk + 1

            lam, acc, kk = lax.cond(
                err < newton_tol,
                lambda: (1.0, jnp.bool_(True), jnp.int32(1)),
                lambda: lax.while_loop(bt_cond, bt_body, (1.0, jnp.bool_(False), jnp.int32(0))),
            )
            x_new = x - lam * dx
            grow = (it >= 1) & (err > 2.0 * err_prev)
            n_grow = n_grow + grow.astype(jnp.int32)
            fail = (~acc) | (err > 1e6) | (~jnp.all(jnp.isfinite(dx))) | (n_grow >= 2)
            done = err < newton_tol
            return x_new, it + 1, done, fail, err, n_grow, nbt + kk - 1

        st0 = (x_pred, jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.inf, jnp.int32(0), jnp.int32(0))
        x, it, done, fail, _, _, nbt = lax.while_loop(cond, body, st0)
        return x, done & (~fail), it, nbt

    def solve(a, rtol, atol, t_eval, y_guess, perturb=1e-3):
        t_eval = jnp.asarray(t_eval, float)
        n_out = t_eval.size
        t_end = t_eval[-1]
        x0 = steady(a, y_guess)
        x0 = x0.at[:N].multiply(1.0 + perturb)
        c0 = dict(
            t=t_eval[0],
            xh=jnp.zeros((K_HIST, n)).at[0].set(x0),
            th=jnp.full(K_HIST, t_eval[0]) - jnp.arange(K_HIST, dtype=float),  # instantes fictícios decrescentes
            nhist=jnp.int32(1),
            h=jnp.minimum(1e-2 * (t_eval[1] - t_eval[0]), max_step),
            h_prev=jnp.minimum(1e-2 * (t_eval[1] - t_eval[0]), max_step),
            err_prev=1.0,
            order=jnp.int32(1),
            cooldown=jnp.int32(0),
            nfail=jnp.int32(0),
            n_steps=jnp.int32(0),
            n_rej=jnp.int32(0),
            n_nit=jnp.int32(0),
            n_bt=jnp.int32(0),
            i_out=jnp.int32(1),
            out=jnp.zeros((n_out, n)).at[0].set(x0),
            dead=jnp.bool_(False),
        )

        def cond(c):
            return (
                (c["t"] < t_end - 1e-12 * jnp.maximum(1.0, jnp.abs(t_end)))
                & (c["n_steps"] + c["n_rej"] < max_steps)
                & (~c["dead"])
            )

        def body(c):
            h = jnp.minimum(jnp.minimum(c["h"], max_step), t_end - c["t"])
            t_new = c["t"] + h
            x = c["xh"][0]
            order = jnp.minimum(c["order"], jnp.maximum(c["nhist"] - 1, 1))
            f0 = rhs(x, a) / Mv
            pred, c_hist, gamma, C_err = lax.switch(order - 1, branches, (c["xh"], c["th"], t_new, h, f0))
            scale_n = atol + rtol * jnp.abs(pred)
            x_new, ok, nit, nbt = newton(pred, c_hist, gamma, a, scale_n)
            ok = ok & jnp.all(jnp.isfinite(x_new))
            scale = atol + rtol * jnp.maximum(jnp.abs(x_new), jnp.abs(x))
            err = jnp.where(ok, rms(C_err * (x_new - pred), scale), jnp.inf)
            accept = ok & (err <= 1.0)
            qf = order.astype(float)

            def reject(c):
                nfail = c["nfail"] + 1
                h_new = jnp.where(
                    ok,
                    h * jnp.maximum(0.2, 0.9 * err ** (-1.0 / (qf + 1.0))),
                    h * jnp.where(c["nfail"] == 0, 0.5, 0.25),
                )
                order_new = jnp.where(ok, jnp.where(nfail >= 3, 1, order), jnp.where(nfail >= 2, 1, order))
                return dict(
                    c,
                    h=h_new,
                    order=order_new,
                    nfail=nfail,
                    cooldown=jnp.int32(cooldown),
                    n_rej=c["n_rej"] + 1,
                    n_nit=c["n_nit"] + nit,
                    n_bt=c["n_bt"] + nbt,
                    dead=h_new < min_step,
                )

            def accept_(c):
                def out_cond(s):
                    i, out = s
                    return (i < n_out) & (
                        t_eval[jnp.minimum(i, n_out - 1)] <= t_new + 1e-12 * jnp.maximum(1.0, jnp.abs(t_new))
                    )

                def out_body(s):
                    i, out = s
                    wgt = (t_eval[i] - c["t"]) / h
                    return i + 1, out.at[i].set(x + wgt * (x_new - x))

                i_out, out = lax.while_loop(out_cond, out_body, (c["i_out"], c["out"]))
                nhist = jnp.minimum(c["nhist"] + 1, K_HIST)
                order_new = jnp.minimum(jnp.minimum(max_order, order + 1), nhist - 1)
                qn = order_new.astype(float)
                err_s = jnp.maximum(err, 1e-10)
                if controller == "basic":
                    fac = 0.9 * err_s ** (-1.0 / (qn + 1.0))
                else:  # H211b (Söderlind 2003), b = 4: filtro nos dois últimos erros e na razão dos passos
                    b1 = 1.0 / (4.0 * (qn + 1.0))
                    fac = (
                        (0.8 / err_s) ** b1
                        * (0.8 / jnp.maximum(c["err_prev"], 1e-10)) ** b1
                        * (h / c["h_prev"]) ** (-0.25)
                    )
                fac = jnp.where(c["cooldown"] > 0, jnp.minimum(fac, 1.0), fac)
                h_new = h * jnp.minimum(growth_max, jnp.maximum(0.3, fac))
                return dict(
                    c,
                    t=t_new,
                    xh=jnp.roll(c["xh"], 1, axis=0).at[0].set(x_new),
                    th=jnp.roll(c["th"], 1).at[0].set(t_new),
                    nhist=nhist,
                    h=h_new,
                    h_prev=h,
                    err_prev=err_s,
                    order=order_new,
                    cooldown=jnp.maximum(c["cooldown"] - 1, 0),
                    nfail=jnp.int32(0),
                    n_steps=c["n_steps"] + 1,
                    n_nit=c["n_nit"] + nit,
                    n_bt=c["n_bt"] + nbt,
                    i_out=i_out,
                    out=out,
                )

            return lax.cond(accept, accept_, reject, c)

        c = lax.while_loop(cond, body, c0)
        return c["out"], dict(n_steps=c["n_steps"], n_rej=c["n_rej"], n_nit=c["n_nit"], n_bt=c["n_bt"], dead=c["dead"])

    return solve


def functional(t_eval, y):
    """Variância do fluxo y nos últimos WINDOW s (trapézios na grelha de saída)."""
    w = (t_eval >= t_eval[-1] - WINDOW).astype(float)
    dt = np.diff(t_eval)
    wm = 0.5 * (w[1:] + w[:-1]) * dt
    ybar = np.sum(wm * 0.5 * (y[1:] + y[:-1])) / np.sum(wm)
    v = (y - ybar) ** 2
    return float(np.sum(wm * 0.5 * (v[1:] + v[:-1])) / np.sum(wm))


def reference(a, p, t_eval, perturb=1e-3):
    """Solução de referência: Radau (scipy) com rtol 1e−10 na forma explícita ẋ = f(x)/M; em cache."""
    from scipy.integrate import solve_ivp

    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(
        REF_DIR, f"a{a:.2f}.npz" if abs(p["R"] - mp.PARAMS["R"]) < 1e-12 else f"a{a:.2f}_R{p['R']:g}.npz"
    )
    if os.path.exists(path):
        d = np.load(path)
        if d["T"].size == t_eval.size and np.allclose(d["T"], t_eval):
            return d["X"]
    x0 = mp.steady(a, p)
    x0[: p["N"]] *= 1.0 + perturb
    Mv = mp.mass(p)
    t0 = time.time()
    sol = solve_ivp(
        lambda t, x: mp.rhs(x, a, p) / Mv,
        (t_eval[0], t_eval[-1]),
        x0,
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
        jac=lambda t, x: mp.jac(x, a, p) / Mv[:, None],
        t_eval=t_eval,
    )
    assert sol.success, sol.message
    np.savez(path, T=t_eval, X=sol.y, a=a, nfev=sol.nfev, wall=time.time() - t0)
    print(f"referência a = {a:.2f}: Radau rtol 1e-10, {sol.t.size} saídas, {time.time() - t0:.0f} s", flush=True)
    return sol.y


def plot(out):
    """Diagramas trabalho–precisão (tentativas = passos + rejeições contra o erro) e tabela do custo a exatidão igual à
    da variante «thesis» com rtol 1e−4, a partir de `summary.json`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(os.path.join(out, "summary.json")))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    As = sorted({r["a"] for r in rows}, reverse=True)

    def series(v, a, key):
        rr = sorted([r for r in rows if r["variant"] == v and r["a"] == a and not r["dead"]], key=lambda r: -r["rtol"])
        return rr, np.array([r["n_steps"] + r["n_rej"] for r in rr], float), np.array([r[key] for r in rr], float)

    def cost_at(cost, err, target):
        o = np.argsort(err)
        e, c = err[o], cost[o]
        if e.size < 2 or not (e.min() <= target <= e.max()):
            return np.nan
        return float(np.exp(np.interp(np.log(target), np.log(e), np.log(c))))

    table = {}
    for key in ("err_early", "err_G", "err_rms"):
        print(f"\n=== tentativas a exatidão igual à de «thesis» com rtol 1e-4 ({key})")
        print(f"{'variante':18s}" + "".join(f"{f'a={a:g}':>22s}" for a in As))
        for v in variants:
            line = f"{v:18s}"
            for a in As:
                rr0, c0, e0 = series("thesis", a, key)
                k0 = [i for i, r in enumerate(rr0) if abs(r["rtol"] - 1e-4) < 1e-12]
                target, c_ref = (e0[k0[0]], c0[k0[0]]) if k0 else (np.nan, np.nan)
                rr, c, e = series(v, a, key)
                ca = cost_at(c, e, target)
                table[(key, v, a)] = ca / c_ref if np.isfinite(ca) else None
                line += f"{ca:10.0f} ({ca / c_ref:4.2f}×)".rjust(22) if np.isfinite(ca) else f"{'fora':>22s}"
            print(line)
    fig, axes = plt.subplots(len(As), 3, figsize=(13, 3.6 * len(As)), squeeze=False)
    colors = plt.get_cmap("tab10")
    for i, a in enumerate(As):
        for j, key in enumerate(("err_early", "err_G", "err_rms")):
            ax = axes[i, j]
            for kv, v in enumerate(variants):
                rr, c, e = series(v, a, key)
                ax.loglog(e, c, "-o", ms=3, lw=1, color=colors(kv % 10), label=v)
            ax.set_xlabel(key)
            ax.set_ylabel("attempts (steps + rejections)")
            ax.set_title(f"a = {a:g}")
            ax.grid(True, which="both", lw=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    path = os.path.join(out, "work_precision.png")
    fig.savefig(path, dpi=120)
    json.dump(
        {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in table.items()},
        open(os.path.join(out, "cost_ratio.json"), "w"),
        indent=1,
    )
    print(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["run", "plot"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--a", nargs="*", type=float, default=list(A_CASES))
    ap.add_argument("--rtol", nargs="*", type=float, default=list(RTOLS))
    ap.add_argument("--tf", type=float, default=300.0)
    ap.add_argument("--dt-out", type=float, default=0.1)
    ap.add_argument("--R", type=float, default=None, help="resistência da linha (omissão: a do problema-modelo)")
    args = ap.parse_args(argv)
    if args.command == "plot":
        return plot(args.out)
    os.makedirs(args.out, exist_ok=True)
    p = mp.params() if args.R is None else mp.params(R=args.R)
    t_eval = np.arange(0.0, args.tf + 0.5 * args.dt_out, args.dt_out)
    refs = {a: reference(a, p, t_eval) for a in args.a}
    rows = []
    for name in args.variants:
        solve = jax.jit(lambda a, rtol, atol, yg, _s=build(p, **VARIANTS[name]): _s(a, rtol, atol, t_eval, yg))
        for a in args.a:
            yg = float(mp.steady(a, p)[-1])
            y_ref = refs[a][-1]
            amp = float(np.ptp(y_ref[t_eval >= t_eval[-1] - WINDOW]))
            G_ref = functional(t_eval, y_ref)
            for rtol in args.rtol:
                atol = rtol * 1e-2
                out, cnt = solve(a, rtol, atol, yg)
                out.block_until_ready()
                t0 = time.time()
                out, cnt = solve(a, rtol, atol, yg)
                out.block_until_ready()
                wall = time.time() - t0
                y = np.asarray(out[:, -1])
                row = dict(
                    variant=name,
                    a=a,
                    rtol=rtol,
                    wall=wall,
                    err_rms=float(np.sqrt(np.mean((y - y_ref) ** 2)) / amp),
                    err_early=float(np.sqrt(np.mean((y - y_ref)[t_eval <= EARLY] ** 2)) / amp),
                    err_max=float(np.max(np.abs(y - y_ref)) / amp),
                    err_G=abs(functional(t_eval, y) - G_ref) / G_ref,
                    **{k: int(v) for k, v in cnt.items()},
                )
                rows.append(row)
                print(
                    f"{name:16s} a {a:.2f} rtol {rtol:.0e}: {row['n_steps']:6d} passos {row['n_rej']:5d} rej. "
                    f"{row['n_nit']:6d} it. {row['n_bt']:4d} bt | erro rms {row['err_rms']:.2e} (até {EARLY:.0f} s {row['err_early']:.2e}) máx {row['err_max']:.2e} "
                    f"G {row['err_G']:.2e} | {wall:.2f} s{'  MORTO' if row['dead'] else ''}",
                    flush=True,
                )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
