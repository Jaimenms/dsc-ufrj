"""Estimativa de erro de Milne (058) nas janelas do Newton do caso offshore (`cases/paper01/newton/*.json`, 150 s a
partir de estados gravados): `ImplicitBDF` contra `ImplicitBDFMilne` e as suas opções (iterado inicial do Newton
linear ou quadrático, tampão do passo depois de falhas do Newton), com rtol 3e−3 (o do artigo) e 1e−3, todos com a
busca em linha de Armijo (053). Passos, rejeições, Jacobianas, lados direitos e relógio, e o desvio de cada corrida
face a uma referência de conveniência (`--ref`, a corrida mais apertada da classe-mãe), nas escalas das tolerâncias.
Grava em `runs/paper01/offshore_newton_milne/`.

    uv run python tools/milne_windows.py [--windows blocked blowout blowout_end] [--variants base milne_lin ...]
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
# nome -> (classe de Milne?, rtol, opções da subclasse)
VARIANTS = {
    "base": (False, 3e-3, {}),
    "base_tight": (False, 1e-3, {}),
    "milne": (True, 3e-3, dict(newton_start="quadratic")),
    "milne_tight": (True, 1e-3, dict(newton_start="quadratic")),
    "milne_lin": (True, 3e-3, dict(newton_start="linear")),
    "milne_lin_tight": (True, 1e-3, dict(newton_start="linear")),
    "milne_cap": (True, 3e-3, dict(newton_start="linear", newton_fail_cap=0.8)),
    "base_cap": (True, 3e-3, dict(estimate="thesis", newton_start="linear", newton_fail_cap=0.8)),
}


def main(argv=None):
    from mfsim.integrators.milne import integrate_bdf_milne
    from mfsim.io import case_from_json
    from mfsim.simulation import _integrate_bdf, build_model, output_times

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--windows", nargs="*", default=["blocked", "blowout", "blowout_end"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--ref", default="base_tight", help="variante de referência dos desvios")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--damping", default="armijo")
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
            milne, rtol, opts = VARIANTS[name]
            spec = dataclasses.replace(model.integration, rtol=rtol, newton_damping=a.damping)
            t0 = time.time()
            if milne:
                T, X, info = integrate_bdf_milne(model, X0, t_eval, spec, progress=False, partial=True, **opts)
            else:
                T, X, info = _integrate_bdf(model, X0, t_eval, spec, progress=False, partial=True)
            wall = time.time() - t0
            res[name] = (T, X)
            rows.append(
                dict(
                    window=w,
                    variant=name,
                    rtol=rtol,
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
                f"{w:12s} {name:16s} rtol {rtol:.0e}: {rows[-1]['n_steps']:5d} passos {rows[-1]['n_rejected']:5d} rej. "
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
                f"   {name:16s} desvio face a {a.ref}: RMS no fim {r['dev_end']:.3e}, máx no tempo {r['dev_max']:.3e}"
            )
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(os.path.join(a.out, "summary.json"))


if __name__ == "__main__":
    main()
