"""Manifesto de dados não versionados (ou binários) do repositório.

Fixa a proveniência de notebooks/data/*.npz, dos casos legados .mf2, dos PDFs/DOC e dos
npz de scratch/arquivo: sha256, tamanho e, para npz, as chaves, as formas de T/X e o
`meta` (JSON). Escreve MANIFEST.sha256 (formato `sha256sum`) e MANIFEST.json (índice
legível, que serve de índice de corridas: nome -> meta).

uso: uv run python tools/manifest.py --write        # (re)gera os dois ficheiros
     uv run python tools/manifest.py --check        # verifica; sai com 1 se algum ficheiro mudou
     uv run python tools/manifest.py --check --strict   # também falha se faltar algum ficheiro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    "notebooks/data/*.npz",
    "thesis/data/*.mf2",
    "docs/*.pdf",
    "tests/golden/**/*.npz",
]
MANIFEST_SHA = ROOT / "MANIFEST.sha256"
MANIFEST_JSON = ROOT / "MANIFEST.json"


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def npz_summary(path: Path) -> dict:
    """Chaves, formas de T/X e meta de um npz (carrega só esses arrays)."""
    out = {}
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            out["keys"] = sorted(z.files)
            for key in ("T", "X"):
                if key in z.files:
                    out[f"{key}_shape"] = list(z[key].shape)
            if "meta" in z.files:
                try:
                    out["meta"] = json.loads(str(z["meta"]))
                except (json.JSONDecodeError, TypeError):
                    out["meta"] = str(z["meta"])
    except Exception as e:  # ficheiro corrompido ou formato inesperado: regista o erro
        out["error"] = repr(e)
    return out


def collect() -> list[Path]:
    files = set()
    for pat in PATTERNS:
        files.update(p for p in ROOT.glob(pat) if p.is_file())
    return sorted(files)


def write():
    entries = {}
    for p in collect():
        rel = p.relative_to(ROOT).as_posix()
        entry = {"size": p.stat().st_size, "sha256": sha256_of(p)}
        if p.suffix == ".npz":
            entry.update(npz_summary(p))
        entries[rel] = entry
    with open(MANIFEST_SHA, "w") as f:
        for rel, e in entries.items():
            f.write(f"{e['sha256']}  {rel}\n")
    with open(MANIFEST_JSON, "w") as f:
        json.dump({"root": ROOT.name, "patterns": PATTERNS, "files": entries}, f, indent=1, ensure_ascii=False)
    total = sum(e["size"] for e in entries.values())
    print(f"manifesto: {len(entries)} ficheiros, {total/1e6:.0f} MB -> {MANIFEST_SHA.name}, {MANIFEST_JSON.name}")


def check(strict=False) -> int:
    if not MANIFEST_SHA.exists():
        print("sem MANIFEST.sha256 (correr --write)")
        return 2
    expected = {}
    for line in MANIFEST_SHA.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            expected[rel] = h
    changed, missing = [], []
    for rel, h in expected.items():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256_of(p) != h:
            changed.append(rel)
    new = [p.relative_to(ROOT).as_posix() for p in collect() if p.relative_to(ROOT).as_posix() not in expected]
    for rel in changed:
        print("ALTERADO:", rel)
    for rel in missing:
        print("em falta:", rel)
    for rel in new:
        print("novo (fora do manifesto):", rel)
    ok = not changed and not (strict and missing)
    print(
        f"verificados {len(expected)}: {len(changed)} alterados, {len(missing)} em falta, {len(new)} novos -> {'OK' if ok else 'FALHA'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    sys.exit(write() or 0 if a.write else check(a.strict))
