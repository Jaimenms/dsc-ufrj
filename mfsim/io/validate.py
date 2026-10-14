"""Validação dos casos JSON contra o schema `mfsim-case/1` (`io/schema/mfsim-case-1.schema.json`).

O schema amarra o vocabulário (modelos, padrões, esquemas e as suas opções, tipos de porta,
tipos de aresta, receitas de condição inicial, campos de integração) e os tipos/limites dos
valores; a coerência estrutural (ids referidos, tamanhos de tabelas) é verificada por
`Network.validate` e pelos construtores. `case_from_json` valida ao carregar e `case_to_json`
antes de escrever; `mfsim validate` valida ficheiros avulsos.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema", "mfsim-case-1.schema.json")


@lru_cache(maxsize=1)
def schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator

    sch = schema()
    Draft202012Validator.check_schema(sch)
    return Draft202012Validator(sch)


def case_errors(d: dict) -> list:
    """Mensagens de erro (vazia se o dicionário obedece ao schema). Chaves internas (`_base_dir`)
    são ignoradas."""
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    errs = sorted(_validator().iter_errors(d), key=lambda e: list(e.absolute_path))
    out = []
    for e in errs:
        path = "/".join(str(p) for p in e.absolute_path) or "(raiz)"
        msg = e.message if len(e.message) < 200 else e.message[:200] + "..."
        out.append(f"{path}: {msg}")
    return out


def validate_case(d: dict, source: str = "") -> dict:
    """Levanta ValueError com todas as violações, ou devolve o próprio dicionário."""
    errs = case_errors(d)
    if errs:
        head = f"{source}: " if source else ""
        raise ValueError(f"{head}caso JSON inválido para mfsim-case/1 ({len(errs)} erro(s)):\n  " + "\n  ".join(errs))
    return d
