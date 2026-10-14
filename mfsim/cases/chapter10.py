"""Casos do capítulo 10 (Resultados de Simulações Dinâmicas) da tese.

A fonte são os ficheiros JSON em `cases/chapter10/` (exportados dos `.mf2` originais por
`mfsim.io.legacy_mf2`, com as correções registadas em `meta.provenance`): a descrição de
cada caso (secção, título, figuras, descrição) vive no `meta` do JSON, não em código. Os
resultados do MATLAB, para comparação com os notebooks e os testes, são lidos do `.mf2`
indicado em `meta.provenance.source` (`legacy_results`).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

CASES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cases", "chapter10"
)
PREFIX = "chapter10_"


@dataclass(frozen=True)
class CaseInfo:
    """Descrição de um caso, lida de `meta` do JSON."""

    key: str
    section: str
    title: str
    description: str
    figures: str
    source: str  # ficheiro .mf2 com a entrada e os resultados do MATLAB


def chapter10_json_path(key: str) -> str:
    return os.path.join(CASES_DIR, key.replace(".", "_") + ".json")


def chapter10_keys() -> list[str]:
    """Chaves dos casos ('10.1.1', ..., '10.3.3') a partir dos ficheiros em `cases/chapter10/`."""
    keys = []
    for path in sorted(glob.glob(os.path.join(CASES_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            name = json.load(f).get("name", "")
        if name.startswith(PREFIX):
            keys.append(name[len(PREFIX) :])
    return keys


def chapter10_info(key: str) -> CaseInfo:
    with open(chapter10_json_path(key), encoding="utf-8") as f:
        d = json.load(f)
    meta = d.get("meta") or {}
    prov = meta.get("provenance") or {}
    return CaseInfo(
        key=key,
        section=prov.get("section", key),
        title=prov.get("title", ""),
        description=meta.get("description", d.get("description", "")),
        figures=prov.get("figures", ""),
        source=prov.get("source", ""),
    )


def chapter10_case(key: str):
    """O grafo (`Network`) do caso `key`, do JSON exportado."""
    from ..io.spec import case_from_json

    path = chapter10_json_path(key)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: exportar com `uv run mfsim export-case --chapter10`")
    return case_from_json(path)


def legacy_results(case_or_key):
    """Dicionário do `.mf2` original (`dados`, `saidas`, `estados`) com a entrada e os
    resultados do MATLAB do caso, para comparação; o ficheiro vem de `meta.provenance.source`."""
    from ..io_legacy import load_mf2

    if isinstance(case_or_key, str):
        source = chapter10_info(case_or_key).source
    else:
        source = ((case_or_key.meta or {}).get("provenance") or {}).get("source", "")
    if not source:
        raise ValueError("o caso não indica o .mf2 de origem em meta.provenance.source")
    return load_mf2(source)
