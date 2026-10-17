"""Órbitas periódicas do caso offshore do artigo 01 por *shooting* simples (issue 059; método do artigo 03).

Incógnitas: o estado x₀ (200 no caso de 2 km) e o período T, com φ_T(x₀) = x₀ e a condição de fase de uma secção de
Poincaré na pressão de entrada da linha, x₀[iP] = P_sec. Newton no sistema bordado

    [M − I, ẋ(x_T); e_iPᵀ, 0] [δx₀; δT] = −[φ_T(x₀) − x₀; x₀[iP] − P_sec],

com M = ∂φ_T/∂x₀ a matriz de monodromia, propagada dentro do BDF próprio com a opção `integration.sensitivity =
"state"` (a derivada da trajetória discreta, com os passos que o integrador escolhe). Passo de Newton amortecido
(RMS da correção nas escalas das tolerâncias limitado a 2) e projetado nos limites físicos. Multiplicadores de
Floquet pelos valores próprios de M (um deles é ≈ 1, a direção da própria órbita). Semente: o último ciclo de uma
corrida da campanha (`runs/paper01/offshore_dynamics/ab{a}.npz`), x₀ no cruzamento ascendente da média de P_in e T o
intervalo entre fins de sopro (duplo com `--period2`); sem corrida, a órbita anterior da lista (continuação).

Grava `runs/paper01/offshore_orbits/orbit_a{a:.3f}.npz` (x₀, T, resíduos, multiplicadores, séries de P_in e q_L da
órbita convergida) e `branch.json`; o guião das figuras só lê.

    uv run python tools/periodic_orbit.py --openings 0.3 0.6 1.0 [--period2 1.0] [--max-iter 10] [--tol 0.3]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "runs", "paper01", "offshore_orbits")
SEED_DIR = os.path.join(ROOT, "runs", "paper01", "offshore_dynamics")
CASE = os.path.join(ROOT, "cases", "paper01", "offshore.json")
WINDOW = 8000.0  # s: janela final da corrida-semente
SMOOTH = 20.0


def build(a):
    from mfsim.io import case_from_json
    from mfsim.simulation import build_model

    return build_model(case_from_json(CASE, overrides={"network.edges.choke_2.opening": float(a)}))


def blowdown_minima(T, P, smooth=SMOOTH, t_min=300.0):
    from mfsim.analysis import moving_average

    Ps = moving_average(T, P, smooth)
    lo = np.percentile(Ps, 30)
    out = []
    for j in range(1, T.size - 1):
        if (
            Ps[j] < Ps[j - 1]
            and Ps[j] <= Ps[j + 1]
            and Ps[j] < lo
            and T[j] > t_min
            and (not out or T[j] - T[out[-1]] > 150)
        ):
            out.append(j)
    return out, Ps


def seed_from_run(a, iP, period2=False, seed_dir=SEED_DIR):
    """(x₀, T₀, P_sec) do último ciclo completo da corrida da campanha nesta abertura, ou None."""
    from mfsim.runs import load_run

    path = os.path.join(seed_dir, f"ab{a:.3f}.npz")
    if not os.path.exists(path):
        return None
    run = load_run(path)
    T = np.asarray(run.T)
    X = np.asarray(run.X)
    P = X[iP] * run.model.PR / 1e5
    sel = np.flatnonzero(T >= T[-1] - WINDOW)
    mins, Ps = blowdown_minima(T[sel], P[sel])
    if len(mins) < (3 if period2 else 2):
        return None
    j0, j1 = (mins[-3], mins[-1]) if period2 else (mins[-2], mins[-1])
    j0, j1 = sel[j0], sel[j1]
    T0 = float(T[j1] - T[j0])
    # secção: cruzamento ascendente da média do ciclo (depois do mínimo que o abre)
    Pm = float(np.mean(P[j0:j1]))
    k = next(k for k in range(j0 + 1, j1) if P[k - 1] < Pm <= P[k])
    return np.asarray(X[:, k], float), T0, float(X[iP, k])


def integrate(model, x0, T, sens, dt_out=None):
    from mfsim.simulation import _integrate_bdf, output_times

    t_eval = output_times(0.0, T, dt_out) if dt_out else np.array([0.0, T])
    spec = dataclasses.replace(model.integration, sensitivity="state" if sens else None)
    return _integrate_bdf(model, x0, t_eval, spec, progress=False, partial=False)


def outputs(model, X):
    """Séries de P_in (bar), q_L à saída (kg/s) e fração de líquido média do riser ao longo da órbita."""
    N = model.pipes[0].scheme.N
    th = np.degrees(np.asarray(model.pipes[0].mesh["theta"]))
    h = np.asarray(model.pipes[0].mesh["h"])
    riser = th > 89.0
    P, q, hold = [], [], []
    for k in range(X.shape[1]):
        Y = model.assemble(0.0, X[:, k])[1][0]
        P.append(Y[1, 1] * model.PR / 1e5)
        q.append(Y[N, 2])
        aG = np.asarray(Y[1 : N + 1, 0])
        hold.append(1.0 - float(np.average(aG[riser], weights=h[riser])))
    return np.array(P), np.array(q), np.array(hold)


def shoot(model, x0, T, P_sec, iP, tol=0.3, max_iter=10, verbose=True):
    """Newton amortecido em (x₀, T). Devolve dict com x0, T, convergido, histórico do resíduo, M e ẋ_T."""
    from mfsim.simulation import state_bounds

    lo, hi = state_bounds(model)
    n = x0.size
    scale = model.state_scales(x0)
    hist = []
    best = None
    for it in range(max_iter):
        t0 = time.time()
        _, X, info = integrate(model, x0, T, sens=True)
        xT, M = X[:, -1], info["S"]
        r = xT - x0
        g = x0[iP] - P_sec
        err = float(np.sqrt(np.mean((r / scale) ** 2)))
        hist.append(err)
        if verbose:
            print(
                f"   it {it}: resíduo {err:.4f} (fase {g:+.2e}), T {T:.1f} s, {info['n_steps']} passos, "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
        if best is None or err < best["err"]:
            best = dict(x0=x0.copy(), T=T, err=err, M=M, it=it)
        if err < tol:
            return dict(x0=x0, T=T, converged=True, err_hist=hist, M=M, n_iter=it + 1)
        f = model.rhs(T, xT)
        xdot = spla.spsolve(sp.csc_matrix(model.transform_jacobian(xT)), f)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = M - np.eye(n)
        A[:n, n] = xdot
        A[n, iP] = 1.0
        d = np.linalg.solve(A, -np.concatenate([r, [g]]))
        m = float(np.sqrt(np.mean((d[:n] / scale) ** 2)))
        lam = 1.0 if m <= 2.0 else 2.0 / m
        x0 = np.minimum(np.maximum(x0 + lam * d[:n], lo), hi)
        T = float(np.clip(T + lam * d[n], 0.5 * T, 1.5 * T))
    return dict(x0=best["x0"], T=best["T"], converged=False, err_hist=hist, M=best["M"], n_iter=max_iter)


def floquet(M, k=8):
    mu = np.linalg.eigvals(M)
    mu = mu[np.argsort(-np.abs(mu))]
    return mu[:k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--openings", nargs="+", type=float, required=True)
    ap.add_argument(
        "--period2", nargs="*", type=float, default=[], help="aberturas em que a semente usa o período duplo"
    )
    ap.add_argument("--seed-dir", default=SEED_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tol", type=float, default=0.3, help="RMS do resíduo φ_T(x₀) − x₀ nas escalas das tolerâncias")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--dt-out", type=float, default=2.0)
    ap.add_argument("--suffix", default="", help="sufixo do nome da órbita (ex.: _p2 para o ramo de período 2)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    branch_path = os.path.join(a.out, "branch.json")
    branch = json.load(open(branch_path)) if os.path.exists(branch_path) else {}
    prev = None
    for ab in a.openings:
        model = build(ab)
        slots = model.state_slots()
        iP = int(np.flatnonzero(slots == 1)[0])
        seed = seed_from_run(ab, iP, period2=ab in a.period2, seed_dir=a.seed_dir)
        if seed is None and prev is not None:
            seed = (prev["x0"], prev["T"], float(prev["x0"][iP]))
            src = f"órbita anterior (a = {prev['a']:g})"
        else:
            src = "corrida da campanha" if seed is not None else None
        if seed is None:
            print(f"a = {ab:.3f}: sem semente (nem corrida nem órbita anterior)", flush=True)
            continue
        x0, T0, P_sec = seed
        print(f"a = {ab:.3f}: semente da {src}, T₀ {T0:.0f} s, P_sec {P_sec * model.PR / 1e5:.2f} bar", flush=True)
        t_start = time.time()
        res = shoot(model, x0, T0, P_sec, iP, tol=a.tol, max_iter=a.max_iter)
        wall = time.time() - t_start
        mu = floquet(res["M"])
        _, X, _ = integrate(model, res["x0"], res["T"], sens=False, dt_out=a.dt_out)
        P, q, hold = outputs(model, X)
        mu_sorted = mu[np.argsort(-np.abs(mu))]
        trivial = int(np.argmin(np.abs(mu_sorted - 1.0)))
        others = np.delete(mu_sorted, trivial)
        stable = bool(np.abs(others).max() < 1.0) if others.size else True
        summ = dict(
            a=ab,
            T=res["T"],
            converged=res["converged"],
            residual=res["err_hist"][-1],
            n_iter=res["n_iter"],
            wall=wall,
            P_min=float(P.min()),
            P_max=float(P.max()),
            P_mean=float(P.mean()),
            q_mean=float(q.mean()),
            q_peak=float(q.max()),
            holdup_min=float(hold.min()),
            mu_abs=np.abs(mu_sorted).tolist(),
            mu_lead=[float(others[0].real), float(others[0].imag)] if others.size else [np.nan, np.nan],
            stable=stable,
            seed=src,
        )
        np.savez(
            os.path.join(a.out, f"orbit_a{ab:.3f}{a.suffix}.npz"),
            x0=res["x0"],
            T=res["T"],
            err_hist=np.array(res["err_hist"]),
            mu_re=mu.real,
            mu_im=mu.imag,
            t=np.linspace(0.0, res["T"], X.shape[1]),
            P_in=P,
            q_L=q,
            holdup=hold,
            X=X,
            **{k: v for k, v in summ.items() if not isinstance(v, list) and k not in ("T", "a")},
        )
        branch[f"{ab:.3f}{a.suffix}"] = summ
        with open(branch_path, "w") as f:
            json.dump(branch, f, indent=1)
        print(
            f"a = {ab:.3f}: {'convergiu' if res['converged'] else 'NÃO convergiu'} em {res['n_iter']} it. "
            f"(resíduo {summ['residual']:.3f}, {wall:.0f} s): T {res['T']:.0f} s, P_in {summ['P_min']:.2f}–{summ['P_max']:.2f} bar, "
            f"q_L média {summ['q_mean']:.2f} pico {summ['q_peak']:.1f}, holdup mín {summ['holdup_min']:.2f}, "
            f"|μ| {np.round(summ['mu_abs'][:4], 3).tolist()} → {'estável' if stable else 'instável'}",
            flush=True,
        )
        prev = dict(a=ab, x0=res["x0"], T=res["T"])
    print(branch_path)


if __name__ == "__main__":
    main()
