"""Problema-modelo integrado por um BDF escrito todo em JAX (estudo: issue 056).

O mesmo BDF de ordem 1–2 com passo variável do simulador (`mfsim.integrators.implicit.ImplicitBDF`: resíduo
conservativo M x − c_hist − γ h f(x), preditor extrapolado, erro local pela diferença corretor − preditor, controlo de
passo 0,9 err^(−1/(q+1)) com arrefecimento depois de rejeições, recuo a BDF1 depois de falhas), mas com o laço de passos
em `lax.while_loop`, o Newton com busca em linha de Armijo (053) e a Jacobiana exata por `jacfwd` com LU denso —
tudo compilado numa só função de `a`. Mede:

1. custo contra o BDF do simulador (laço em Python, LU esparsa, Jacobiana analítica) nos quatro valores de `a`;
2. a sensibilidade dG/da de um funcional do ciclo (variância do fluxo y nos últimos 100 s) pelo modo direto (`jvp`)
   através de toda a integração adaptativa, contra diferenças finitas;
3. a mesma sensibilidade pelo modo inverso (`grad`) com a reposição (*replay*) em grelha congelada: os passos aceites
   pela integração adaptativa são reintegrados por `lax.scan` com Newton de contagem fixa, que é diferenciável no
   sentido inverso (o `while_loop` não é).

Uso (grava em runs/paper01/toy/jaxbdf/; o guião das figuras só lê):

    uv run python papers/01-severe-slugging-two-fluid/toy/model_problem_jaxbdf.py run [--out DIR] [--a 0.85 ...]
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

OUT = os.path.join(ROOT, "runs", "paper01", "toy", "jaxbdf")
WINDOW = 100.0  # s: janela final do funcional (variância do fluxo y)


def build(p, rtol=1e-4, atol=1e-6, max_step=0.5, newton_tol=1e-3, max_newton=20, max_order=2, max_steps=120000):
    """Funções compiladas do problema-modelo com os parâmetros `p` fixos; `a` é o argumento diferenciável."""
    N = int(p["N"])
    hx = 1.0 / N
    n = 2 * N
    Mv = jnp.asarray(mp.mass(p))
    k, Pr, R, g0, beta, y0 = (float(p[s]) for s in ("k", "Pr", "R", "g0", "beta", "y0"))
    min_step = 1e-10

    def fchar(y):
        return g0 - beta * (y - y0) + (y - y0) ** 3 / 3.0

    def dfchar(y):
        return -beta + (y - y0) ** 2

    def rhs(x, a):
        """d(M x)/dt, como `model_problem.rhs`."""
        pc, w, y = x[:N], x[N : 2 * N - 1], x[2 * N - 1]
        wf = jnp.concatenate([jnp.array([k * (Pr - pc[0])]), w, jnp.array([y])])
        return jnp.concatenate(
            [-(wf[1:] - wf[:-1]) / hx, -(pc[1:] - pc[:-1]) / hx - R * w, jnp.array([pc[-1] - fchar(y) - a * y])]
        )

    jac = jax.jacfwd(rhs, argnums=0)

    def steady(a, y_guess):
        """Estacionário diferenciável em `a`: Newton escalar a partir da raiz numérica (brentq) do guião original."""

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

    def newton(x_pred, c_hist, gamma, a, scale, armijo):
        """Newton modificado (Jacobiana no preditor, LU reusada) com busca em linha de Armijo opcional."""
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
            if armijo:

                def bt_cond(s):
                    lam, acc, kk = s
                    return (~acc) & (kk < 5)

                def bt_body(s):
                    lam, acc, kk = s
                    r_try = rms(resid(x - lam * dx), scale)
                    acc = (r_try <= (1.0 - 1e-4 * lam) * res_norm) | (res_norm <= 1e-12)
                    lam = jnp.where(acc, lam, 0.5 * lam)
                    return lam, acc, kk + 1

                # sem busca em linha quando a correção já convergiu (issue 055): no fundo da convergência o resíduo
                # está no erro de arredondamento e não desce
                lam, acc, kk = lax.cond(
                    err < newton_tol,
                    lambda: (1.0, jnp.bool_(True), jnp.int32(1)),
                    lambda: lax.while_loop(bt_cond, bt_body, (1.0, jnp.bool_(False), jnp.int32(0))),
                )
            else:
                lam, acc, kk = 1.0, jnp.bool_(True), jnp.int32(1)
            x_new = x - lam * dx
            grow = (it >= 1) & (err > 2.0 * err_prev)
            n_grow = n_grow + grow.astype(jnp.int32)
            fail = (~acc) | (err > 1e6) | (~jnp.all(jnp.isfinite(dx))) | (n_grow >= 2)
            done = err < newton_tol
            return x_new, it + 1, done, fail, err, n_grow, nbt + kk - 1

        st0 = (x_pred, jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.inf, jnp.int32(0), jnp.int32(0))
        x, it, done, fail, _, _, nbt = lax.while_loop(cond, body, st0)
        return x, done & (~fail), it, nbt

    def solve(a, t_eval, y_guess, perturb=1e-3, armijo=True):
        """Integração adaptativa completa em JAX. Devolve a trajetória nos instantes `t_eval` (interpolação linear
        entre passos aceites, como o simulador), os contadores e a grelha aceite (h e ordem por passo)."""
        t_eval = jnp.asarray(t_eval, float)
        n_out = t_eval.size
        t_end = t_eval[-1]
        x0 = steady(a, y_guess)
        x0 = x0.at[:N].multiply(1.0 + perturb)
        c0 = dict(
            t=t_eval[0],
            x=x0,
            x_prev=x0,
            t_prev=t_eval[0] - 1.0,
            h=jnp.minimum(1e-2 * (t_eval[1] - t_eval[0]), max_step),
            order=jnp.int32(1),
            nhist=jnp.int32(1),
            cooldown=jnp.int32(0),
            nfail=jnp.int32(0),
            n_steps=jnp.int32(0),
            n_rej=jnp.int32(0),
            n_nit=jnp.int32(0),
            n_bt=jnp.int32(0),
            i_out=jnp.int32(1),
            out=jnp.zeros((n_out, n)).at[0].set(x0),
            h_log=jnp.zeros(max_steps),
            q_log=jnp.zeros(max_steps, jnp.int32),
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
            x, xm1 = c["x"], c["x_prev"]
            order = jnp.where(c["nhist"] < 2, 1, c["order"])
            h_old = c["t"] - c["t_prev"]
            rr = jnp.where(order == 2, h / h_old, 0.0)
            f0 = rhs(x, a) / Mv
            pred = jnp.where(order == 2, x + rr * (x - xm1), x + h * f0)
            w1 = jnp.where(order == 2, (1.0 + rr) ** 2 / (1.0 + 2.0 * rr), 1.0)
            w2 = jnp.where(order == 2, -(rr**2) / (1.0 + 2.0 * rr), 0.0)
            c_hist = w1 * Mv * x + w2 * Mv * xm1
            gcoef = jnp.where(order == 2, (1.0 + rr) / (1.0 + 2.0 * rr), 1.0)
            scale_n = atol + rtol * jnp.abs(pred)
            x_new, ok, nit, nbt = newton(pred, c_hist, gcoef * h, a, scale_n, armijo)
            ok = ok & jnp.all(jnp.isfinite(x_new))
            scale = atol + rtol * jnp.maximum(jnp.abs(x_new), jnp.abs(x))
            coef = jnp.where(order == 2, rr / (1.0 + 2.0 * rr), 0.5)
            err = jnp.where(ok, rms(coef * (x_new - pred), scale), jnp.inf)
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
                    cooldown=jnp.int32(3),
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
                order_new = jnp.minimum(max_order, order + 1)
                fac = jnp.where(err > 0.0, 0.9 * err ** (-1.0 / (order_new.astype(float) + 1.0)), 2.0)
                fac = jnp.where(c["cooldown"] > 0, jnp.minimum(fac, 1.0), fac)
                h_new = h * jnp.minimum(2.0, jnp.maximum(0.3, fac))
                return dict(
                    c,
                    t=t_new,
                    x=x_new,
                    x_prev=x,
                    t_prev=c["t"],
                    h=h_new,
                    order=order_new,
                    nhist=jnp.minimum(c["nhist"] + 1, 3),
                    cooldown=jnp.maximum(c["cooldown"] - 1, 0),
                    nfail=jnp.int32(0),
                    n_steps=c["n_steps"] + 1,
                    n_nit=c["n_nit"] + nit,
                    n_bt=c["n_bt"] + nbt,
                    i_out=i_out,
                    out=out,
                    h_log=c["h_log"].at[c["n_steps"]].set(h),
                    q_log=c["q_log"].at[c["n_steps"]].set(order),
                )

            return lax.cond(accept, accept_, reject, c)

        c = lax.while_loop(cond, body, c0)
        return (
            c["out"],
            dict(n_steps=c["n_steps"], n_rej=c["n_rej"], n_nit=c["n_nit"], n_bt=c["n_bt"], dead=c["dead"]),
            c["h_log"],
            c["q_log"],
        )

    def replay(a, t_eval, y_guess, h_log, q_log, n_acc, perturb=1e-3, n_newton=6):
        """Reposição em grelha congelada: os passos aceites (h, ordem) são reintegrados por `lax.scan` com Newton de
        contagem fixa (diferenciável no sentido inverso). Devolve y nos instantes `t_eval` por interpolação."""
        t_eval = jnp.asarray(t_eval, float)
        x0 = steady(a, y_guess)
        x0 = x0.at[:N].multiply(1.0 + perturb)
        idx = jnp.arange(h_log.size)
        live = idx < n_acc
        h_seq = jnp.where(live, h_log, 0.0)
        q_seq = jnp.where(live, q_log, 1)

        def step(carry, inp):
            x, xm1, t, h_old = carry
            h, order, alive = inp
            rr = jnp.where(order == 2, h / jnp.maximum(h_old, 1e-300), 0.0)
            f0 = rhs(x, a) / Mv
            pred = jnp.where(order == 2, x + rr * (x - xm1), x + h * f0)
            w1 = jnp.where(order == 2, (1.0 + rr) ** 2 / (1.0 + 2.0 * rr), 1.0)
            w2 = jnp.where(order == 2, -(rr**2) / (1.0 + 2.0 * rr), 0.0)
            c_hist = w1 * Mv * x + w2 * Mv * xm1
            gamma = jnp.where(order == 2, (1.0 + rr) / (1.0 + 2.0 * rr), 1.0) * h
            A = jnp.diag(Mv) - gamma * jac(pred, a)
            lu = lu_factor(A)
            x_new = lax.fori_loop(
                0, n_newton, lambda _, xx: xx - lu_solve(lu, Mv * xx - c_hist - gamma * rhs(xx, a)), pred
            )
            x_new = jnp.where(alive, x_new, x)
            t_new = t + h
            return (x_new, jnp.where(alive, x, xm1), t_new, jnp.where(alive, h, h_old)), (t_new, x_new[-1])

        (_, _, _, _), (t_acc, y_acc) = lax.scan(step, (x0, x0, t_eval[0], 1.0), (h_seq, q_seq, live))
        # instantes padded (depois do fim) crescem com o índice para a interpolação ficar bem definida
        t_acc = jnp.where(live, t_acc, t_eval[-1] + 1.0 + idx.astype(float))
        t_all = jnp.concatenate([t_eval[:1], t_acc])
        y_all = jnp.concatenate([x0[-1:], y_acc])
        return jnp.interp(t_eval, t_all, y_all)

    def functional(t_eval, y):
        """Variância do fluxo y nos últimos WINDOW s (trapézios na grelha de saída)."""
        t_eval = jnp.asarray(t_eval, float)
        w = (t_eval >= t_eval[-1] - WINDOW).astype(float)
        dt = jnp.diff(t_eval)
        wm = 0.5 * (w[1:] + w[:-1]) * dt
        ybar = jnp.sum(wm * 0.5 * (y[1:] + y[:-1])) / jnp.sum(wm)
        v = (y - ybar) ** 2
        return jnp.sum(wm * 0.5 * (v[1:] + v[:-1])) / jnp.sum(wm)

    return dict(solve=solve, replay=replay, functional=functional, n=n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--a", nargs="*", type=float, default=list(mp.A_RUNS))
    ap.add_argument("--tf", type=float, default=300.0)
    ap.add_argument("--dt-out", type=float, default=0.1)
    ap.add_argument("--fd", type=float, default=1e-3, help="passo das diferenças finitas centrais em a")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    p = mp.params()
    t_eval = np.arange(0.0, args.tf + 0.5 * args.dt_out, args.dt_out)
    fx = build(p)
    solve = jax.jit(lambda a, yg: fx["solve"](a, t_eval, yg))
    solve_legacy = jax.jit(lambda a, yg: fx["solve"](a, t_eval, yg, armijo=False))

    def G_adaptive(a, yg):
        out, _, _, _ = fx["solve"](a, t_eval, yg)
        return fx["functional"](t_eval, out[:, -1])

    G_jvp = jax.jit(lambda a, yg: jax.jvp(lambda aa: G_adaptive(aa, yg), (a,), (1.0,)))
    G_replay = jax.jit(
        lambda a, yg, h_log, q_log, n_acc: fx["functional"](t_eval, fx["replay"](a, t_eval, yg, h_log, q_log, n_acc))
    )
    dG_replay = jax.jit(jax.grad(lambda a, yg, h_log, q_log, n_acc: G_replay(a, yg, h_log, q_log, n_acc)))
    results = []
    for a in args.a:
        yg = float(mp.steady(a, p)[-1])
        row = dict(a=a)
        # 1. BDF do simulador (laço em Python)
        t0 = time.time()
        T, X, info = mp.run(a, p, tf=args.tf, dt_out=args.dt_out)
        row["mfsim_wall"] = time.time() - t0
        row["mfsim_steps"], row["mfsim_rej"] = int(info["n_steps"]), int(info["n_rejected"])
        # 2. BDF em JAX: compilação (1.ª chamada) e execução quente
        t0 = time.time()
        out, cnt, h_log, q_log = solve(a, yg)
        out.block_until_ready()
        row["jax_first_wall"] = time.time() - t0
        t0 = time.time()
        out, cnt, h_log, q_log = solve(a, yg)
        out.block_until_ready()
        row["jax_wall"] = time.time() - t0
        row.update({f"jax_{k}": int(v) for k, v in cnt.items()})
        y_j, y_m = np.asarray(out[:, -1]), X[-1]
        amp = float(np.ptp(y_m[T >= T[-1] - WINDOW]))
        row["max_dy"], row["amplitude"] = float(np.max(np.abs(y_j - y_m))), amp
        row["G_jax"] = float(fx["functional"](t_eval, out[:, -1]))
        row["G_mfsim"] = float(fx["functional"](t_eval, jnp.asarray(y_m)))
        # sem busca em linha (Newton do simulador antes da 053)
        out_l, cnt_l, _, _ = solve_legacy(a, yg)
        out_l.block_until_ready()
        t0 = time.time()
        out_l, cnt_l, _, _ = solve_legacy(a, yg)
        out_l.block_until_ready()
        row["jax_legacy_wall"] = time.time() - t0
        row.update({f"jax_legacy_{k}": int(v) for k, v in cnt_l.items()})
        # 3. sensibilidade: jvp através da integração adaptativa
        t0 = time.time()
        G0, dG = G_jvp(a, yg)
        dG.block_until_ready()
        row["jvp_first_wall"] = time.time() - t0
        t0 = time.time()
        G0, dG = G_jvp(a, yg)
        dG.block_until_ready()
        row["jvp_wall"], row["dG_jvp"] = time.time() - t0, float(dG)
        # 4. diferenças finitas centrais (duas integrações adaptativas)
        d = args.fd
        Gp = fx["functional"](t_eval, solve(a + d, float(mp.steady(a + d, p)[-1]))[0][:, -1])
        Gm = fx["functional"](t_eval, solve(a - d, float(mp.steady(a - d, p)[-1]))[0][:, -1])
        row["dG_fd"] = float((Gp - Gm) / (2 * d))
        # 5. reposição em grelha congelada: grad (modo inverso)
        n_acc = int(cnt["n_steps"])
        h_log, q_log = np.asarray(h_log)[:n_acc], np.asarray(q_log)[:n_acc]  # forma estática: recompila por a
        t0 = time.time()
        Gr = G_replay(a, yg, h_log, q_log, n_acc)
        Gr.block_until_ready()
        row["replay_first_wall"] = time.time() - t0
        row["G_replay"] = float(Gr)
        t0 = time.time()
        g = dG_replay(a, yg, h_log, q_log, n_acc)
        g.block_until_ready()
        row["grad_first_wall"] = time.time() - t0
        t0 = time.time()
        g = dG_replay(a, yg, h_log, q_log, n_acc)
        g.block_until_ready()
        row["grad_wall"], row["dG_replay"] = time.time() - t0, float(g)
        results.append(row)
        np.savez(
            os.path.join(args.out, f"a{a:.2f}.npz"),
            T=t_eval,
            X=np.asarray(out).T,
            X_mfsim=X,
            h_log=h_log,
            q_log=q_log,
            **{k: v for k, v in row.items()},
        )
        print(
            f"a = {a:.2f}: mfsim {row['mfsim_steps']} passos / {row['mfsim_wall']:.1f} s | JAX {row['jax_n_steps']} passos "
            f"({row['jax_n_rej']} rej., {row['jax_n_bt']} backtracks) / {row['jax_wall']:.2f} s quente, "
            f"{row['jax_first_wall']:.1f} s com compilação | sem Armijo {row['jax_legacy_n_steps']} passos / {row['jax_legacy_wall']:.2f} s | "
            f"max|Δy| {row['max_dy']:.2e} (amplitude {amp:.3f}) | dG/da: jvp {row['dG_jvp']:+.4e} ({row['jvp_wall']:.2f} s), "
            f"DF {row['dG_fd']:+.4e}, replay {row['dG_replay']:+.4e} ({row['grad_wall']:.2f} s); G {row['G_jax']:.4e} "
            f"(replay {row['G_replay']:.4e}, mfsim {row['G_mfsim']:.4e})",
            flush=True,
        )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
