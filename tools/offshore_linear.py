"""Estabilidade linear do sistema offshore do artigo 01 ao longo da abertura do choke (não integra no tempo).

Semente: o estado final de uma corrida que assentou (por omissão a de menor abertura do lote da dinâmica), levado ao
estacionário exato por Newton; continuação em abertura com bissecção do passo; em cada ponto o espetro generalizado
(`mfsim.analysis.linear_spectrum`, matriz de massa do modo conservativo) e os modos lentos globais
(`slow_global_modes`), com a participação somada sobre os dois dutos. Grava `runs/paper01/offshore_linear/hopf.npz`
(linhas [abertura, Re1, Im1, Re2, Im2, Re_outro, Im_outro, n_eff_outro, nó_outro, P_in (bar), q_L (kg/s),
P_topo (bar, entrada do choke), fração de gás média da linha, fração de líquido média do riser]).

    uv run python tools/offshore_linear.py [--seed runs/paper01/offshore_dynamics/ab0.100.npz] [--openings ...]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "runs", "paper01", "offshore_linear")
OPENINGS = tuple(float(a) for a in np.round(np.r_[np.arange(0.10, 0.40, 0.02), np.arange(0.40, 1.0001, 0.05)], 3))


def participation(model, X):
    """Autovalores (Im >= 0), nós efetivos e nó de maior peso, com a participação somada sobre todos os dutos."""
    from mfsim.analysis import linear_spectrum

    w, V = linear_spectrum(model, X, jax=True, vectors=True)
    keep = w.imag >= -1e-12
    w, V = w[keep], V[:, keep]
    sc = model.state_scales(X)
    blocks = []
    for pb in model.pipes:
        sl = slice(pb.offset, pb.offset + pb.n_states)
        v2 = (np.abs(V[sl]) / sc[sl, None]) ** 2
        nodes = pb.free // 4
        pn = np.zeros((int(nodes.max()) + 1, w.size))
        np.add.at(pn, nodes, v2)
        blocks.append(pn)
    pn = np.vstack(blocks)
    pn /= pn.sum(axis=0)
    return w, 1.0 / np.sum(pn**2, axis=0), np.argmax(pn, axis=0)


def solve(src, ab, X0):
    from mfsim.io import case_from_json
    from mfsim.simulation import build_model, state_bounds
    from mfsim.steady import newton_steady

    model = build_model(case_from_json(copy.deepcopy(src), overrides={"network.edges.choke_2.opening": float(ab)}))
    X, info = newton_steady(model, X0, tol=1e-9, verbose=False, bounds=state_bounds(model), jax=True, max_iter=60)
    return model, (X if info["converged"] else None)


def path(src, a0, a1, X0, depth=0):
    """Continuação de a0 a a1 com bissecção geométrica do passo (até 5 níveis)."""
    model, X1 = solve(src, a1, X0)
    if X1 is not None or depth >= 5:
        return model, X1
    am = float(np.sqrt(a0 * a1))
    _, Xm = path(src, a0, am, X0, depth + 1)
    if Xm is None:
        return model, None
    return path(src, am, a1, Xm, depth + 1)


def main(argv=None):
    from mfsim.analysis import slow_global_modes
    from mfsim.runs import load_run

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", default=os.path.join(ROOT, "runs", "paper01", "offshore_dynamics", "ab0.100.npz"))
    ap.add_argument("--openings", nargs="*", type=float, default=list(OPENINGS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    run = load_run(a.seed)
    src = run.case_json
    ab0 = float(next(e for e in src["network"]["edges"] if e.get("type") == "choke")["opening"])
    model, X = solve(src, ab0, np.asarray(run.X[:, -1], float))
    if X is None:
        raise SystemExit(f"Newton não convergiu na semente (abertura {ab0}): a corrida não assentou?")
    print(f"semente {os.path.basename(a.seed)}: estacionário em a = {ab0}", flush=True)
    rows = {}
    for up in (True, False):
        Xr, ar = X, ab0
        seq = sorted([o for o in a.openings if (o >= ab0 - 1e-9) == up], reverse=not up)
        for ab in seq:
            if abs(ab - ar) < 1e-9:
                Xn, m = Xr, model
            else:
                m, Xn = path(src, ar, ab, Xr)
            if Xn is None:
                print(f"a = {ab:.3f}: Newton falhou", flush=True)
                break
            Xr, ar = Xn, ab
            w, n_eff, node = participation(m, Xn)
            g = slow_global_modes(w, n_eff, node)
            Y = m.assemble(0.0, Xn)[1][0]
            N = m.pipes[0].scheme.N
            # fração de gás média da linha (células antes da curva) e fração de líquido média do riser, para o
            # critério de Jansen et al. (1996) no artigo
            th = np.degrees(np.asarray(m.pipes[0].mesh["theta"]))
            h = np.asarray(m.pipes[0].mesh["h"])
            line = np.arange(N) < int(np.flatnonzero(th > 1.0)[0])
            riser = th > 89.0
            aG = np.asarray(Y[1 : N + 1, 0])
            rows[ab] = [
                ab,
                g["re1"],
                g["im1"],
                g["re2"],
                g["im2"],
                g["re_other"],
                g["im_other"],
                g["n_eff_other"],
                g["node_other"],
                Y[1, 1] * m.PR / 1e5,
                Y[N, 2],
                Y[N, 1] * m.PR / 1e5,  # pressão no topo do riser (entrada do choke), para a queda no choke
                float(np.average(aG[line], weights=h[line])),
                1.0 - float(np.average(aG[riser], weights=h[riser])),
            ]
            per = 2 * np.pi / g["im1"] if g["im1"] > 0 else np.nan
            print(
                f"a = {ab:.3f}: Re1 {g['re1']:+.3e} T {per:7.0f} s {g['cls']:6s} P_in {Y[1, 1] * m.PR / 1e5:6.2f} bar q_L {Y[N, 2]:.3f} kg/s",
                flush=True,
            )
    os.makedirs(a.out, exist_ok=True)
    arr = np.array([rows[k] for k in sorted(rows)])
    np.savez(os.path.join(a.out, "hopf.npz"), rows=arr, seed=os.path.basename(a.seed))
    print(os.path.join(a.out, "hopf.npz"))


if __name__ == "__main__":
    main()
