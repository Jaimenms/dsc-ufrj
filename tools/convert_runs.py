"""Converte corridas gravadas no formato antigo (v1: `meta` com o dialeto do driver, ou os
ficheiros do driver com `opts`) para o formato v2 (`case_json`, `info_json`, proveniência).

uso: uv run python tools/convert_runs.py antiga.npz [...] [--out-dir runs/]   (por omissão substitui no lugar)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legacy_runs import convert_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out-dir", dest="out_dir", help="escreve as cópias convertidas nesta pasta")
    a = ap.parse_args(argv)
    rc = 0
    for src in a.files:
        dst = os.path.join(a.out_dir, os.path.basename(src)) if a.out_dir else None
        try:
            written = convert_run(src, dst)
            print(f"{src} -> {written}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{src}: FALHOU ({e})", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
