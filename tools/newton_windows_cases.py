"""Casos das janelas do Newton (053/058) a partir de uma corrida do caso offshore: três estados do último ciclo com o
choke aberto gravados nos nós do esquema (`cases/paper01/newton/{blocked,blowout,blowout_end}.json`) e o lote
`batches/paper01_offshore_newton.json` (150 s por janela e variante do Newton).

Os instantes vêm da pressão de entrada da linha com média móvel de 20 s, no último ciclo completo da corrida: «sopro»
no máximo (o gás começa a entrar no riser), «fim do sopro» no mínimo seguinte, «bloqueio» a meio da subida anterior
ao máximo. Com `--times` os instantes são dados à mão.

    uv run python tools/newton_windows_cases.py [--run runs/paper01/offshore_dynamics/ab1.000.npz] [--times 2500 4100 4600]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CASES = os.path.join(ROOT, "cases", "paper01", "newton")
BATCH = os.path.join(ROOT, "batches", "paper01_offshore_newton.json")
WINDOW = 150.0
# variantes do Newton em relação ao caso (que já traz a busca em linha e a estimativa de Milne do gerador)
VARIANTS = {
    "reference": [],
    "no_line_search": ["integration.newton_damping=legacy"],
    "armijo_all": ["integration.newton_damping=armijo"],
    "estimate_thesis": ["integration.error_estimate=thesis"],
    "jac_rate": ["integration.jac_policy=rate"],
    "pred_none": ["integration.predictor=none"],
}


def pick_times(run, smooth=20.0):
    """Instantes do último ciclo completo: (bloqueio, sopro, fim do sopro)."""
    from mfsim.analysis import moving_average

    T = np.asarray(run.T)
    P = np.asarray(run.pipe(0)["out"]["P_bar"][1])
    Ps = moving_average(T, P, smooth)
    lo, hi = np.percentile(Ps, 30), np.percentile(Ps, 70)
    mins = [j for j in range(1, T.size - 1) if Ps[j] < Ps[j - 1] and Ps[j] <= Ps[j + 1] and Ps[j] < lo and T[j] > 300]
    tm = []
    for j in mins:
        if not tm or T[j] - T[tm[-1]] > 150:
            tm.append(j)
    if len(tm) < 2:
        raise SystemExit("menos de dois fins de sopro na corrida: dá os instantes com --times")
    j0, j1 = tm[-2], tm[-1]  # último ciclo completo: entre dois mínimos
    jp = j0 + int(np.argmax(Ps[j0:j1]))  # máximo = início do sopro
    t_block = 0.5 * (T[j0] + T[jp])
    return float(t_block), float(T[jp]), float(T[j1])


def main(argv=None):
    from mfsim.io import case_from_json, case_to_json
    from mfsim.runs import load_run
    from mfsim.simulation import build_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "paper01", "offshore_dynamics", "ab1.000.npz"))
    ap.add_argument("--times", nargs=3, type=float, default=None, metavar=("BLOCKED", "BLOWOUT", "END"))
    ap.add_argument("--window", type=float, default=WINDOW, help="duração de cada janela (s)")
    a = ap.parse_args(argv)
    window = float(a.window)
    run = load_run(a.run)
    T = np.asarray(run.T)
    times = tuple(a.times) if a.times else pick_times(run)
    os.makedirs(CASES, exist_ok=True)
    runs = []
    for label, t_pick in zip(("blocked", "blowout", "blowout_end"), times):
        k = int(np.argmin(np.abs(T - t_pick)))
        X0, t0 = np.asarray(run.X[:, k], float), float(T[k])
        d = copy.deepcopy(run.case_json)
        for e in d["network"]["edges"]:
            e.pop("ic", None)
        d["initial_state"] = {"source": "guess"}
        d["name"] = f"newton_{label}"
        d["description"] = (
            f"Janela do Newton (053/058): estado da corrida {os.path.basename(a.run)} do artigo 01 aos {t0:.0f} s gravado "
            f"nos nós do esquema; {window:.0f} s de integração com as opções do artigo."
        )
        net = case_from_json(d)
        model = build_model(net)
        model.state_to_case_ic(X0, t0, exact=True)
        net.initial = "edges"
        net.set_time(t0=t0, tf=t0 + window, dt_out=2.0)
        out = os.path.join(CASES, f"{label}.json")
        case_to_json(net, out, ic_file=f"{label}.ic.npz")
        print(f"{label}: t0 {t0:.0f} s -> {os.path.relpath(out, ROOT)}")
        for var, sets in VARIANTS.items():
            runs.append({"name": f"{label}_{var}", "case": os.path.relpath(out, ROOT), "set": sets})
    spec = {
        "description": (
            "Artigo 01: o Newton do BDF em três janelas de 150 s do ciclo com o choke aberto (bloqueio, sopro, fim do "
            f"sopro, {window:.0f} s cada): referência (busca em linha de Armijo até à convergência e estimativa de Milne), sem busca em linha, "
            "Armijo em todas as iterações, estimativa da tese, Jacobiana pela taxa e estado anterior como iterado inicial "
            "(issues 053, 055, 058)."
        ),
        "out_dir": "runs/paper01/offshore_newton",
        "runs": runs,
    }
    with open(BATCH, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"lote: {os.path.relpath(BATCH, ROOT)} ({len(runs)} corridas)")


if __name__ == "__main__":
    main()
