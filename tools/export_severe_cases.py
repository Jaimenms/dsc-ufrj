"""Migração (já feita em 2026-10-14): reexporta os casos JSON de `cases/severe_slug` a partir
das corridas gravadas que os definem (`case_json` de cada corrida, com `integration` e o meta
do gerador), com a condição inicial como receita nos dutos únicos e as variantes como remendo
`$extends` sobre o caso base. Cada JSON guarda a corrida de origem em `meta.run`.

uso: uv run python tools/export_severe_cases.py [--out cases/severe_slug]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mfsim.io.spec import case_to_json  # noqa: E402
from mfsim.runs import load_run  # noqa: E402

CASES_DIR = os.path.join(ROOT, "cases", "severe_slug")

# corridas gravadas que definem os casos de referência (nome em notebooks/data -> nome do JSON)
REFERENCE_RUNS = {
    "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys": "reference",
    "severe_slug_blocked_conservative_arc_tol_hv_vm_lv": "reference_thesis_bubbles",
    "severe_slug_blocked_regime6_local": "reference_auto_regime",
    "severe_slug_blocked_conservative_arc_boe_bub_rules": "long_line_2bar",
    "severe_slug_blocked_network_v005": "reference_network",
    "severe_slug_lab_fabre1_fisico": "lab_fabre1",
    "severe_slug_lab_fabre2_fisico": "lab_fabre2",
    "severe_slug_lab_fabre2_tese": "lab_fabre2_thesis_bubbles",
    "severe_slug_lab_jansen_fisico": "lab_jansen",
    "severe_slug_lab_jansen_tese": "lab_jansen_thesis_bubbles",
    "severe_slug_lab_teng1_fisico": "lab_tengesdal1",
}


# variantes gravadas como remendo (`"$extends"`) sobre o caso base da mesma pasta
EXTENDS = {
    "reference_thesis_bubbles": "reference",
    "reference_auto_regime": "reference",
    "long_line_2bar": "reference",
    "lab_fabre1": "lab_fabre2",
    "lab_fabre2_thesis_bubbles": "lab_fabre2",
    "lab_jansen_thesis_bubbles": "lab_jansen",
}


def export_run_cases(out_dir=CASES_DIR, runs=None):
    """Exporta para JSON os casos das corridas gravadas (o `case_json` de cada corrida, com a
    secção `integration` e o meta do gerador), com a condição inicial como receita nos dutos
    únicos; as variantes de `EXTENDS` ficam como remendo sobre o caso base."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    items = sorted((runs or REFERENCE_RUNS).items(), key=lambda it: it[1] in EXTENDS)  # bases primeiro
    for run_name, name in items:
        run = load_run(run_name)
        net = copy.deepcopy(run.case)  # já traz a secção integration e o meta do gerador (conversão v2)
        net.name = name
        if not run.meta.get("network", False) and run.meta.get("start", "blocked") in ("blocked", "shut_in"):
            # duto único: a condição inicial fica como receita (ficheiro auto-contido)
            m = net.meta
            pipe = net.pipes[0]
            if run.meta.get("start", "blocked") == "blocked":
                pipe.ic_recipe = {
                    "recipe": "blocked",
                    "s_front": float(m["s_front"]),
                    "s_top": float(m["s_top"]),
                    "pocket_alpha": float(m["pocket_alpha"]),
                    "P_out_bar": float(m["P_out_bar"]),
                    "dP_margin_bar": float(run.meta.get("dP_margin", 0.3)),
                    "qL": float(m["qL"]),
                    "qG": float(m["qG"]),
                }
            else:
                pipe.ic_recipe = {
                    "recipe": "shut_in",
                    "s_elbow0": float(m["s_elbow0"]),
                    "alpha_line": 0.5,
                    "alpha_riser": 0.01,
                    "P_out_bar": float(m["P_out_bar"]),
                }
            pipe.ic = None
        path = os.path.join(out_dir, f"{name}.json")
        base = EXTENDS.get(name)
        case_to_json(net, path, ic_file=f"{name}.ic.npz", extends=f"{base}.json" if base else None)
        written.append(path)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=CASES_DIR)
    a = ap.parse_args()
    for p in export_run_cases(a.out):
        print("escrito", p)
