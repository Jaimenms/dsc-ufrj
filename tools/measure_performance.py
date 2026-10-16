"""Tempos por chamada do lado direito e da Jacobiana em numpy e JAX num caso (artigo 01, secção de backends).

Mede, num estado do caso (o inicial, ou o último de uma corrida gravada com `--run`): o lado direito em numpy,
a Jacobiana numpy por diferenças finitas coloridas (a do integrador), a compilação JAX (primeira chamada do
lado direito e da Jacobiana) e as chamadas JAX seguintes (lado direito e Jacobiana exata por `jacfwd`); grava
um JSON com as medianas. Não simula.

uso: uv run python tools/measure_performance.py cases/severe_slug/well_choke_auto.json \
         --set network.edges.choke_2.opening=1.0 --out runs/paper01/performance/timings.json [--run corrida.npz]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _median_time(fn, repeat=30):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure(case_path, overrides=None, run_path=None, repeat=30):
    from mfsim.compiled import get_jax
    from mfsim.integrators.jacobian import color_groups, colored_jacobian
    from mfsim.io import case_from_json
    from mfsim.simulation import build_model

    net = case_from_json(case_path, overrides=overrides)
    model = build_model(net)
    if run_path:
        from mfsim.runs import load_run

        X = np.asarray(load_run(run_path).X[:, -1], float)
    else:
        X = model.initial_state()
    S = model.jac_sparsity()
    groups = color_groups(S)
    f0 = model.rhs(0.0, X)
    out = dict(
        case=os.path.relpath(case_path, ROOT),
        overrides=list(overrides or []),
        n_states=int(X.size),
        n_colors=int(len(groups)),
    )
    out["numpy_rhs_ms"] = 1e3 * _median_time(lambda: model.rhs(0.0, X), repeat)
    out["numpy_jac_ms"] = 1e3 * _median_time(
        lambda: colored_jacobian(lambda y: model.rhs(0.0, y), X, f0=f0, S=S, groups=groups), max(5, repeat // 6)
    )
    t0 = time.perf_counter()
    rhs_j, jac_j = get_jax(model)
    np.asarray(rhs_j(0.0, X))
    out["jax_compile_rhs_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    np.asarray(jac_j(0.0, X))
    out["jax_compile_jac_s"] = time.perf_counter() - t0
    out["jax_rhs_ms"] = 1e3 * _median_time(lambda: np.asarray(rhs_j(0.0, X)), repeat)
    out["jax_jac_ms"] = 1e3 * _median_time(lambda: np.asarray(jac_j(0.0, X)), repeat)
    out["parity_rhs"] = float(np.max(np.abs(np.asarray(rhs_j(0.0, X)) - f0)) / max(1.0, float(np.max(np.abs(f0)))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--set", action="append", default=[], help="override caminho=valor (como mfsim run --set)")
    ap.add_argument("--run", help="corrida gravada cujo último estado é o ponto de medida")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=30)
    a = ap.parse_args(argv)
    res = measure(a.case, a.set, a.run, a.repeat)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
