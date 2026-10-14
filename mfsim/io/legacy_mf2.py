"""Conversão dos casos legados (.mf2 do MFSIM/MATLAB) para JSON `mfsim-case/1`.

uso: uv run mfsim export-case --chapter10 [--out cases/chapter10]   (ou python -m mfsim.io.legacy_mf2)
Escreve `<chave>.json` + `<chave>.ic.npz` por caso do capítulo 10. `network_from_mf2` aplica
as composições e correções dos ficheiros originais (10.3.2a = válvulas do ficheiro "a" sobre a
rede "b"; `LEGACY_OVERRIDES` de `io_legacy`), registadas em `meta.provenance`. A descrição de
cada caso (título, figuras, descrição) vive no JSON: a exportação preserva o `meta` do ficheiro
existente e só atualiza `provenance.source`, `provenance.section` e a nota.
"""

from __future__ import annotations

import argparse
import json
import os

from .spec import case_to_json

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cases", "chapter10"
)

# chave do caso -> ficheiro .mf2 original (legacy/mfsim23/Files)
MF2_SOURCES = {
    "10.1.1": "CASO10-1-1.mf2",
    "10.1.2": "CASO10-1-2.mf2",
    "10.1.3": "CASO10-1-3.mf2",
    "10.1.4": "CASO10-1-4.mf2",
    "10.2.1A": "CASO10-2-1.mf2",
    "10.2.1B": "CASO10-2-1b.mf2",
    "10.2.2": "CASO10-2-2.mf2",
    "10.3.1": "CASO10-3-1.mf2",
    "10.3.2a": "CASO10-3-2.mf2",
    "10.3.2b": "CASO10-3-2b.mf2",
    "10.3.3": "CASO10-3-3b.mf2",
}
NOTE = "convertido de legacy/mfsim23/Files pelo mfsim.io.legacy_mf2 (network_from_mf2: correções e composição do 10.3.2a)"


def network_from_mf2(key, **kw):
    """(network, raw) do caso `key` a partir do .mf2 original, com as correções: condição
    inicial de `estados.X[:, 0]`, `LEGACY_OVERRIDES` e, no 10.3.2a, os chokes do ficheiro "a"
    sobre a rede do ficheiro "b" (o ficheiro "a" tem o vértice 8 estático, editado após a
    simulação; a rede da tese usa vértice dinâmico com válvula de retenção, como no "b")."""
    from ..io_legacy import load_case
    from ..network.graph import Choke

    if key == "10.3.2a":
        kw = dict(kw, ic_from="file")  # o ficheiro original não monta (retenção em vértice estático)
    net, raw = load_case(MF2_SOURCES[key], **kw)
    if key == "10.3.2a":
        net_b, _ = load_case(MF2_SOURCES["10.3.2b"], ic_from="legacy_t0")
        chokes_a = {e.v_to: e for e in net.edges_of(Choke)}
        net_b.edges = [chokes_a.get(e.v_to, e) if isinstance(e, Choke) else e for e in net_b.edges]
        net = net_b
    net.name = f"chapter10_{key}"
    return net, raw


def export_chapter10(out_dir=DEFAULT_OUT, keys=None):
    """Exporta os casos do capítulo 10 para JSON (preservando o `meta` já existente); devolve
    os caminhos escritos."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key in keys or list(MF2_SOURCES):
        case, raw = network_from_mf2(key)
        fname = key.replace(".", "_")
        path = os.path.join(out_dir, f"{fname}.json")
        meta = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                meta = json.load(f).get("meta") or {}
        prov = dict(meta.get("provenance") or {})
        prov.update(source=MF2_SOURCES[key], section=prov.get("section") or key.rstrip("ABab"), note=NOTE)
        case.meta = dict(meta, provenance=prov)
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
