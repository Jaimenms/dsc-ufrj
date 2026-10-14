"""Conversão dos casos legados (.mf2 do MFSIM/MATLAB) para JSON `mfsim-case/1`.

uso: uv run python -m mfsim.io.legacy_mf2 [--out cases/chapter10]
Escreve `<chave>.json` + `<chave>.ic.npz` por caso do capítulo 10 (com as composições
e correções que `load_chapter10_case` aplica: 10.3.2a = válvulas do ficheiro "a" sobre a
rede "b"; `LEGACY_OVERRIDES`), registadas em `meta.provenance`.
"""
from __future__ import annotations

import argparse
import os

from ..cases.chapter10 import CHAPTER10, load_chapter10_case
from .spec import case_to_json

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cases", "chapter10")


def export_chapter10(out_dir=DEFAULT_OUT, keys=None):
    """Exporta os casos do capítulo 10 para JSON; devolve os caminhos escritos."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key in (keys or list(CHAPTER10)):
        info = CHAPTER10[key]
        case, raw = load_chapter10_case(key)
        case.name = f"chapter10_{key}"
        case.meta = {"provenance": {"source": info.file, "section": info.section, "title": info.title, "figures": info.figures,
                                    "note": "convertido de legacy/mfsim23/Files pelo mfsim.io.legacy_mf2 (inclui as correções de load_chapter10_case)"},
                     "description": info.description}
        fname = key.replace(".", "_")
        path = os.path.join(out_dir, f"{fname}.json")
        case_to_json(case, path, ic_file=f"{fname}.ic.npz")
        written.append(path)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    for p in export_chapter10(a.out, a.keys):
        print("escrito", p)
