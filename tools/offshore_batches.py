"""Casos e lotes do artigo 01 sobre o sistema offshore (`generators.offshore`).

Gera `cases/paper01/offshore.json` (ponto base, partida bloqueada gravada) e os lotes `batches/paper01_offshore_*.json`:
`dynamics` (aberturas do choke no ponto base), `gor` (mais e menos gás do poço), `ablation` (ingredientes do modelo e do
solver, um de cada vez), `discretization` (escalonado conservativo contra primitivo), `mesh` e `backend`. Os lotes
correm com `mfsim batch`; o guião das figuras só lê `runs/paper01/offshore_<lote>/`.

    uv run python tools/offshore_batches.py --P-res 110 --GOR 50 [--tf 8000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generators.offshore import offshore_case, write_case  # noqa: E402

CASES = os.path.join(ROOT, "cases", "paper01")
BATCHES = os.path.join(ROOT, "batches")
OPENINGS = (1.0, 0.6, 0.3, 0.2, 0.1)
GORS = (30.0, 50.0, 80.0, 122.13)
ABLATION = {
    "primitive": [
        "network.edges.1.scheme.options.conservative=false",
        "network.edges.well.scheme.options.conservative=false",
        "network.edges.1.scheme.options.relax_target=raw",
        "network.edges.well.scheme.options.relax_target=raw",
    ],
    "no_art_visc": ["network.edges.1.scheme.options.art_visc=0.0", "network.edges.well.scheme.options.art_visc=0.0"],
    "closure_mean": [
        "network.edges.1.scheme.options.closure_alpha=mean",
        "network.edges.well.scheme.options.closure_alpha=mean",
    ],
    "no_low_void": [
        "network.edges.1.model.regime.low_void.alpha=1e-12",
        "network.edges.well.model.regime.low_void.alpha=1e-12",
    ],
    "no_virtual_mass": [
        "network.edges.1.model.options.virtual_mass.C_vm=0.0",
        "network.edges.well.model.options.virtual_mass.C_vm=0.0",
    ],
    "bubble_drag_thesis": [
        "network.edges.1.model.closures.bubble.drag=thesis",
        "network.edges.well.model.closures.bubble.drag=thesis",
    ],
    "unit_consistent": [
        "network.edges.1.model.closures.intermittent.unit=consistent",
        "network.edges.well.model.closures.intermittent.unit=consistent",
    ],
    "unit_thesis": [
        "network.edges.1.model.closures.intermittent.unit=thesis",
        "network.edges.well.model.closures.intermittent.unit=thesis",
    ],
    "freq_thesis": [
        "network.edges.1.model.closures.intermittent.slug_frequency=thesis",
        "network.edges.well.model.closures.intermittent.slug_frequency=thesis",
    ],
    "k_switch_80": ["network.edges.1.model.regime.k_switch=80.0", "network.edges.well.model.regime.k_switch=80.0"],
    "bdf1": ["integration.max_order=1"],
    "uniform_atol": ["integration.tol_range=null", "integration.atol=1e-6"],
}


def _batch(name, description, runs, tf):
    d = {"description": description, "out_dir": f"runs/paper01/offshore_{name}", "runs": []}
    for run in runs:
        r = {"name": run["name"], "case": run["case"], "set": ["time.dt_out=2.0"] + run.get("set", []), "tf": tf}
        d["runs"].append(r)
    path = os.path.join(BATCHES, f"paper01_offshore_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--P-res", type=float, required=True)
    ap.add_argument("--GOR", type=float, required=True)
    ap.add_argument(
        "--PI", type=float, default=None, help="índice de produtividade em Sm³/d/bar (omissão: o do gerador)"
    )
    ap.add_argument("--tf", type=float, default=8000.0)
    a = ap.parse_args(argv)
    os.makedirs(CASES, exist_ok=True)
    PI = {} if a.PI is None else {"PI": a.PI}
    base = write_case(
        offshore_case(P_res_bar=a.P_res, GOR=a.GOR, choke_opening=1.0, **PI),
        os.path.join(CASES, "offshore.json"),
        ic="blocked",
    )
    base_rel = os.path.relpath(base, ROOT)
    gor_cases = {}
    for g in GORS:
        if abs(g - a.GOR) < 1e-9:
            gor_cases[g] = base_rel
            continue
        p = write_case(
            offshore_case(P_res_bar=a.P_res, GOR=g, choke_opening=1.0, **PI),
            os.path.join(CASES, f"offshore_gor{g:g}.json"),
            ic="blocked",
        )
        gor_cases[g] = os.path.relpath(p, ROOT)
    out = []
    out.append(
        _batch(
            "dynamics",
            f"Artigo 01, sistema offshore: aberturas do choke no ponto base (P_res {a.P_res:g} bar, RGO {a.GOR:g} Sm³/Sm³), partida bloqueada gravada no caso, saída a 2 s.",
            [
                {"name": f"ab{ab:.3f}", "case": base_rel, "set": [f"network.edges.choke_2.opening={ab}"]}
                for ab in OPENINGS
            ],
            a.tf,
        )
    )
    out.append(
        _batch(
            "gor",
            f"Artigo 01, sistema offshore: mais e menos gás do poço (RGO) com o choke aberto e a {a.P_res:g} bar.",
            [{"name": f"gor{g:g}", "case": c} for g, c in gor_cases.items()],
            a.tf,
        )
    )
    out.append(
        _batch(
            "ablation",
            "Artigo 01, sistema offshore: o ponto base com o choke aberto menos um ingrediente de cada vez.",
            [{"name": "base", "case": base_rel}]
            + [{"name": k, "case": base_rel, "set": v} for k, v in ABLATION.items()],
            a.tf,
        )
    )
    out.append(
        _batch(
            "mesh",
            "Artigo 01, sistema offshore: malha grossa e fina no ponto base com o choke aberto.",
            [
                {
                    "name": "coarse",
                    "case": os.path.relpath(
                        write_case(
                            offshore_case(P_res_bar=a.P_res, GOR=a.GOR, n_h1=4, n_down=3, n_riser=8, n_well=8, **PI),
                            os.path.join(CASES, "offshore_coarse.json"),
                            ic="blocked",
                        ),
                        ROOT,
                    ),
                },
                {
                    "name": "fine",
                    "case": os.path.relpath(
                        write_case(
                            offshore_case(
                                P_res_bar=a.P_res, GOR=a.GOR, n_h1=16, n_down=10, n_elbow=8, n_riser=32, n_well=30, **PI
                            ),
                            os.path.join(CASES, "offshore_fine.json"),
                            ic="blocked",
                        ),
                        ROOT,
                    ),
                },
            ],
            a.tf,
        )
    )
    out.append(
        _batch(
            "backend",
            "Artigo 01, sistema offshore: numpy contra JAX no ponto base com o choke aberto (2000 s).",
            [
                {"name": "jax", "case": base_rel},
                {"name": "numpy", "case": base_rel, "set": ["integration.backend=numpy"]},
            ],
            min(a.tf, 2000.0),
        )
    )
    for p in out:
        print(os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
