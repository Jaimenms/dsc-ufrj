"""Estimativa de erro de Milne (058) nas janelas do Newton do caso offshore (`cases/paper01/newton/*.json`, 150 s a
partir de estados gravados): `integration.error_estimate = "thesis"` contra `"milne"`, com rtol 3e−3 (o do artigo) e
1e−3, todos com a busca em linha de Armijo (`--damping`, omissão `armijo_tol`). Passos, rejeições, Jacobianas, lados
direitos e relógio, e o desvio de cada corrida face a uma referência de conveniência (`--ref`, a corrida mais apertada
com a estimativa da tese), nas escalas das tolerâncias. Grava em `runs/paper01/offshore_newton_milne/`.

    uv run python tools/milne_windows.py [--windows blocked blowout blowout_end] [--variants base milne ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "runs", "paper01", "offshore_newton_milne")
# nome -> alterações ao IntegrationSpec do caso
VARIANTS = {
    "base": dict(error_estimate="thesis", rtol=3e-3),
    "milne": dict(error_estimate="milne", rtol=3e-3),
    "base_tight": dict(error_estimate="thesis", rtol=1e-3),
    "milne_tight": dict(error_estimate="milne", rtol=1e-3),
}


def main(argv=None):
    from mfsim.io import case_from_json
    from mfsim.simulation import _integrate_bdf, build_model, output_times

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--windows", nargs="*", default=["blocked", "blowout", "blowout_end"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--ref", default="base_tight", help="variante de referência dos desvios")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--damping", default="armijo_tol")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for w in a.windows:
        net = case_from_json(os.path.join(ROOT, "cases", "paper01", "newton", f"{w}.json"))
        model = build_model(net)
        X0 = model.initial_state()
        t_eval = output_times(net.time.t0, net.time.tf, net.time.dt_out)
        scale = model.state_scales(X0)
        res = {}
        for name in a.variants:
            spec = dataclasses.replace(model.integration, newton_damping=a.damping, **VARIANTS[name])
            t0 = time.time()
            T, X, info = _integrate_bdf(model, X0, t_eval, spec, progress=False, partial=True)
            wall = time.time() - t0
            res[name] = (T, X)
            rows.append(
                dict(
                    window=w,
                    variant=name,
                    rtol=spec.rtol,
                    success=bool(info.get("success", True)),
                    t_end=float(T[-1]) if len(T) else float("nan"),
                    n_steps=int(info["n_steps"]),
                    n_rejected=int(info["n_rejected"]),
                    n_jac=int(info["n_jac"]),
                    n_rhs=int(info["n_rhs"]),
                    wall=wall,
                    error=info.get("error"),
                )
            )
            np.savez(
                os.path.join(a.out, f"{w}_{name}.npz"), T=T, X=X, **{k: v for k, v in rows[-1].items() if v is not None}
            )
            print(
                f"{w:12s} {name:12s} rtol {spec.rtol:.0e}: {rows[-1]['n_steps']:5d} passos {rows[-1]['n_rejected']:5d} rej. "
                f"{rows[-1]['n_jac']:5d} jac {rows[-1]['n_rhs']:6d} rhs | {wall:6.1f} s"
                + ("" if rows[-1]["success"] else f"  FALHOU em {rows[-1]['t_end']:.0f} s: {rows[-1]['error']}"),
                flush=True,
            )
        if a.ref not in res:
            continue
        # desvio face à referência, nas escalas das tolerâncias (RMS no fim e máximo ao longo do tempo)
        T_ref, X_ref = res[a.ref]
        for name in a.variants:
            if name == a.ref:
                continue
            T, X = res[name]
            n = min(X.shape[1], X_ref.shape[1])
            if n == 0:
                continue
            d = (X[:, :n] - X_ref[:, :n]) / scale[:, None]
            r = next(r for r in rows if r["window"] == w and r["variant"] == name)
            r["dev_end"] = float(np.sqrt(np.mean(d[:, -1] ** 2)))
            r["dev_max"] = float(np.max(np.sqrt(np.mean(d**2, axis=0))))
            print(
                f"   {name:12s} desvio face a {a.ref}: RMS no fim {r['dev_end']:.3e}, máx no tempo {r['dev_max']:.3e}"
            )
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(os.path.join(a.out, "summary.json"))


if __name__ == "__main__":
    main()
