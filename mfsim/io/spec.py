"""Casos em JSON (`mfsim-case/1`): todas as entradas de uma simulação num ficheiro.

Secções: schema, name, description, meta, fluid, reference (PR, state_ref), time,
library (models, schemes reutilizáveis por "$ref"), network (vertices, edges),
integration, initial_state. Um ficheiro pode herdar de outro com "$extends" e trazer só o
que muda (`io/extends.py`); o schema (`io/schema/`) valida o caso expandido. Unidades SI; pressões nas especificações como "P" (Pa) ou
"PT" (P/PR, para round-trip exato dos casos legados); arrays de condição inicial num
ficheiro npz ao lado (`"ic": {"arrays": {"file": ..., "key": ...}}`) ou receita.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from ..core.options import IntegrationSpec, ModelSpec, ReferenceSpec, SchemeSpec, TimeSpec
from ..core.slots import PortKind, SLOT_NAMES
from ..fluid import Fluid
from ..network.graph import (
    BoundaryVertex,
    CheckValve,
    Choke,
    Connector,
    JunctionVertex,
    Network,
    Pipe,
    Reservoir,
    TankVertex,
)
from ..network.schedules import Schedule

SCHEMA = "mfsim-case/1"


# ----------------------------------------------------------------------------- $ref / $include
def _resolve_refs(obj, root):
    if isinstance(obj, dict):
        if "$ref" in obj and len(obj) == 1:
            target = root
            for part in obj["$ref"].split("."):
                target = target[part]
            return _resolve_refs(target, root)
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    return obj


def _relocate_ic_files(d, from_dir, to_dir):
    """Os `ic.arrays.file` de um caso carregado de outra pasta passam a caminhos absolutos
    para continuarem a resolver depois de fundidos num caso desta pasta."""
    if os.path.abspath(from_dir) == os.path.abspath(to_dir):
        return
    for e in d.get("network", {}).get("edges", []):
        arrays = (e.get("ic") or {}).get("arrays")
        if arrays and arrays.get("file") and not os.path.isabs(arrays["file"]):
            arrays["file"] = os.path.join(os.path.abspath(from_dir), arrays["file"])


def load_json(path, _stack=()):
    """Lê o JSON, funde `$include` (bibliotecas externas), aplica o ficheiro sobre as bases de
    `$extends` (`mfsim.io.extends.merge_case`; a cadeia fica em `meta.extends`) e resolve `$ref`."""
    from .extends import merge_case

    path = os.fspath(path)
    key = os.path.abspath(path)
    if key in _stack:
        raise ValueError(f"ciclo em $extends: {' -> '.join(_stack + (key,))}")
    with open(path) as f:
        d = json.load(f)
    base = os.path.dirname(key)
    for inc in d.pop("$include", []) or []:
        with open(os.path.join(base, inc)) as f:
            lib = json.load(f)
        d.setdefault("library", {})
        for k, v in lib.get("library", {}).items():
            d["library"].setdefault(k, {}).update(v)
    extends = d.pop("$extends", None)
    if extends:
        bases = [extends] if isinstance(extends, str) else list(extends)
        merged, chain = None, []
        for b in bases:
            bd = load_json(os.path.join(base, b), _stack + (key,))
            _relocate_ic_files(bd, bd.pop("_base_dir"), base)
            chain += [b] + list((bd.get("meta") or {}).pop("extends", []))
            merged = bd if merged is None else merge_case(merged, bd)
        d = merge_case(merged, d)
        d.setdefault("meta", {})["extends"] = chain
    d["_base_dir"] = base
    return _resolve_refs(d, d)


def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")


# ----------------------------------------------------------------------------- vértices e arestas
def _spec_to_json(spec, PR):
    out = {}
    for k, sch in spec.items():
        out[k] = sch.to_json()
    return out


def _spec_from_json(d, PR):
    out = {}
    for k, v in (d or {}).items():
        if k == "P":  # pressão em Pa -> PT adimensional
            sch = Schedule.from_json(v)
            from ..network.schedules import Constant

            if not isinstance(sch, Constant):
                raise ValueError("especificação P variável: use PT (P/PR) ou tabela em PT")
            out["PT"] = Constant(sch.value / PR)
        else:
            out[k] = Schedule.from_json(v)
    return out


def _vertex_to_json(v, PR):
    d = {"id": v.id, "kind": v.kind}
    if isinstance(v, TankVertex):
        d["volume"] = v.volume
        if v.spec:
            d["spec"] = _spec_to_json(v.spec, PR)
        if v.HG0 is not None or v.PT0 is not None:
            d["ic"] = {"HG": v.HG0, "PT": v.PT0}
    elif isinstance(v, BoundaryVertex):
        d["spec"] = _spec_to_json(v.spec, PR)
    elif isinstance(v, JunctionVertex) and v.spec:
        d["spec"] = _spec_to_json(v.spec, PR)
    return d


def _vertex_from_json(d, PR):
    kind = d.get("kind", "boundary")
    if kind == "tank":
        ic = d.get("ic") or {}
        return TankVertex(
            id=str(d["id"]),
            volume=float(d.get("volume", 1e-3)),
            spec=_spec_from_json(d.get("spec"), PR),
            HG0=ic.get("HG"),
            PT0=ic.get("PT"),
        )
    if kind == "junction":
        return JunctionVertex(id=str(d["id"]), spec=_spec_from_json(d.get("spec"), PR) if d.get("spec") else {})
    return BoundaryVertex(id=str(d["id"]), spec=_spec_from_json(d.get("spec"), PR))


def _pipe_to_json(p, ic_store):
    pts = [{"xyz": [float(x) for x in row]} for row in p.points]
    for i, name in enumerate(p.point_names or ()):
        if name:
            pts[i]["name"] = name
    segs = []
    for i, n in enumerate(p.n_elements):
        s = {"n": int(n)}
        if p.grading is not None and float(p.grading[i]) != 1.0:
            s["grading"] = float(p.grading[i])
        segs.append(s)
    d = {
        "id": p.id,
        "type": "pipe",
        "from": p.v_from,
        "to": p.v_to,
        "D": p.D,
        "roughness": p.roughness,
        "geometry": {"points": pts, "segments": segs},
        "model": p.model.to_json(),
        "scheme": p.scheme.to_json(),
        "ports": {
            "in": {n: k.value for n, k in zip(SLOT_NAMES, p.ports_in)},
            "out": {n: k.value for n, k in zip(SLOT_NAMES, p.ports_out)},
        },
    }
    if p.ic_recipe is not None:
        d["ic"] = dict(p.ic_recipe)
    elif p.ic is not None:
        ic_store[p.id] = np.asarray(p.ic, float)
        d["ic"] = {"arrays": {"key": p.id}}
    return d


def _pipe_from_json(d, ic_arrays):
    g = d["geometry"]
    points = np.array([pt["xyz"] for pt in g["points"]], float)
    names = tuple(pt.get("name", "") for pt in g["points"])
    n_el = np.array([int(s["n"]) for s in g["segments"]], int)
    grading = np.array([float(s.get("grading", 1.0)) for s in g["segments"]], float)
    if np.all(grading == 1.0):
        grading = None
    ports = d.get("ports") or {}
    pin = tuple(PortKind((ports.get("in") or {}).get(n, k.value)) for n, k in zip(SLOT_NAMES, Pipe.ports_in))
    pout = tuple(PortKind((ports.get("out") or {}).get(n, k.value)) for n, k in zip(SLOT_NAMES, Pipe.ports_out))
    ic_d = d.get("ic") or {}
    ic = None
    recipe = None
    if "arrays" in ic_d:
        key = ic_d["arrays"].get("key", d["id"])
        ic = np.asarray(ic_arrays[key], float)
    elif ic_d:
        recipe = dict(ic_d)
    return Pipe(
        id=str(d["id"]),
        v_from=str(d["from"]),
        v_to=str(d["to"]),
        points=points,
        n_elements=n_el,
        grading=grading,
        D=float(d.get("D", 0.1)),
        roughness=float(d.get("roughness", 1e-3)),
        model=ModelSpec.from_json(d.get("model") or {}),
        scheme=SchemeSpec.from_json(d.get("scheme") or {}),
        ports_in=pin,
        ports_out=pout,
        ic=ic,
        ic_recipe=recipe,
        point_names=names,
    )


def _edge_to_json(e):
    d = {"id": e.id, "type": e.type, "from": e.v_from, "to": e.v_to}
    if isinstance(e, Choke):
        d.update(CV=e.CV, opening=e.opening.to_json())
        if e.mode is not None:
            d["mode"] = e.mode.to_json()
        if e.setpoint is not None:
            d["setpoint"] = e.setpoint.to_json()
        pid = {k: v for k, v in e.pid.items() if v is not None and not (k == "pv_type" and v == 1)}
        if pid:
            d["pid"] = pid
    elif isinstance(e, Reservoir):
        d.update(IPL=e.IPL, IPG=e.IPG, P_bar=e.P)
    elif isinstance(e, CheckValve):
        d.update(CV=e.CV, dP_min_bar=e.dP_min)
    return d


def _edge_from_json(d, ic_arrays):
    t = d.get("type", "pipe")
    if t == "pipe":
        return _pipe_from_json(d, ic_arrays)
    base = dict(id=str(d["id"]), v_from=str(d["from"]), v_to=str(d["to"]))
    if t == "connector":
        return Connector(**base)
    if t == "choke":
        pid = dict(d.get("pid") or {})
        pid.setdefault("pv_type", 1)
        return Choke(
            **base,
            CV=float(d.get("CV", 1000.0)),
            opening=Schedule.from_json(d.get("opening", 1.0)),
            mode=Schedule.from_json(d["mode"]) if "mode" in d else None,
            setpoint=Schedule.from_json(d["setpoint"]) if "setpoint" in d else None,
            pid=pid,
        )
    if t == "reservoir":
        return Reservoir(
            **base, IPL=float(d.get("IPL", 10.0)), IPG=float(d.get("IPG", 2.0)), P=float(d.get("P_bar", 100.0))
        )
    if t == "check_valve":
        return CheckValve(**base, CV=float(d.get("CV", 1000.0)), dP_min=float(d.get("dP_min_bar", 0.0)))
    raise ValueError(f"tipo de aresta desconhecido: {t}")


# ----------------------------------------------------------------------------- rede <-> JSON
def network_to_json(net: Network, ic_file=None):
    """Dicionário JSON do grafo; os arrays de condição inicial vão para `ic_arrays`
    (gravados em `ic_file` por `case_to_json`)."""
    ic_store = {}
    PR = net.reference.PR
    d = {"schema": SCHEMA, "name": net.name}
    if net.description:
        d["description"] = net.description
    if net.meta:
        d["meta"] = _jsonable(net.meta)
    d["fluid"] = dict(
        type="polytropic",
        **{
            k: float(getattr(net.fluid, k)) for k in ("rhoL0", "rhoG0", "muL", "muG", "gammaL", "gammaG", "P0", "sigma")
        },
    )
    d["reference"] = net.reference.to_json()
    d["time"] = net.time.to_json()
    d["network"] = {
        "vertices": [_vertex_to_json(v, PR) for v in net.vertices],
        "edges": [_pipe_to_json(e, ic_store) if isinstance(e, Pipe) else _edge_to_json(e) for e in net.edges],
    }
    if ic_store:
        for e in d["network"]["edges"]:
            if e.get("ic", {}).get("arrays") is not None:
                e["ic"]["arrays"]["file"] = ic_file or f"{net.name}.ic.npz"
    d["integration"] = net.integration.to_json()
    d["initial_state"] = {"source": net.initial}
    return d, ic_store


def network_from_json(d, ic_arrays=None, base_dir=None) -> Network:
    """Grafo a partir do dicionário JSON (já com `$ref` resolvidos). Carrega os npz de
    condição inicial referidos em `ic.arrays.file` relativamente a `base_dir`."""
    if d.get("schema", SCHEMA) != SCHEMA:
        raise ValueError(f"schema {d.get('schema')} não suportado (esperado {SCHEMA})")
    base_dir = base_dir or d.get("_base_dir", ".")
    ic_arrays = dict(ic_arrays or {})
    for e in d["network"]["edges"]:
        f = ((e.get("ic") or {}).get("arrays") or {}).get("file")
        if f and ((e.get("ic") or {}).get("arrays") or {}).get("key", e["id"]) not in ic_arrays:
            with np.load(os.path.join(base_dir, f), allow_pickle=False) as z:
                for k in z.files:
                    ic_arrays[k] = z[k]
    fl = dict(d["fluid"])
    fl.pop("type", None)
    fluid = Fluid(**fl)
    ref = ReferenceSpec.from_json(d.get("reference") or {})
    net = Network(
        name=d.get("name", "case"),
        description=d.get("description", ""),
        fluid=fluid,
        time=TimeSpec.from_json(d.get("time") or {}),
        reference=ref,
        integration=IntegrationSpec.from_json(d.get("integration") or {}),
        initial=str((d.get("initial_state") or {}).get("source", "edges")),
        vertices=[_vertex_from_json(v, ref.PR) for v in d["network"]["vertices"]],
        edges=[_edge_from_json(e, ic_arrays) for e in d["network"]["edges"]],
        meta=dict(d.get("meta") or {}),
    )
    return net.validate()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- Case legado <-> JSON
def case_to_json(case, path, ic_file=None, extends=None):
    """Grava o grafo como JSON + npz de condições iniciais ao lado. Com
    `extends` (caminho de um caso base, relativo à pasta de `path`) escreve só o remendo mínimo
    sobre a base (`"$extends"`), de modo que carregar o ficheiro reconstrói o caso completo.
    Devolve o dicionário completo (validado contra o schema)."""
    net = case
    if ic_file is None:
        ic_file = os.path.splitext(os.path.basename(path))[0] + ".ic.npz"
    d, ic_store = network_to_json(net, ic_file=ic_file)
    from .validate import validate_case

    validate_case(d, source=str(path))
    out = d
    if extends:
        from .extends import make_patch, merge_case

        base_dir = os.path.dirname(os.path.abspath(path))
        base = {k: v for k, v in load_json(os.path.join(base_dir, extends)).items() if not k.startswith("_")}
        (base.get("meta") or {}).pop("extends", None)
        target = dict(d)
        patch = make_patch(base, target)
        patch.pop("schema", None)
        if merge_case(base, patch) != target:
            raise ValueError(f"remendo sobre {extends} não reconstrói o caso {net.name}")
        out = {"schema": SCHEMA, "$extends": extends, **patch}
    save_json(out, path)
    if ic_store:
        np.savez(os.path.join(os.path.dirname(os.path.abspath(path)), ic_file), **ic_store)
    return d


def case_from_json(path_or_dict, ic_arrays=None, validate=True, overrides=None):
    """Grafo (`Network`, o que `simulate`/`build_model`/`mfsim run` consomem) a partir de um JSON
    (com `$extends`/`$include`/`$ref` resolvidos), validado contra o schema `mfsim-case/1`
    (`validate=False` desliga). `overrides`: `{"integration.rtol": 1e-3}` ou lista de
    `caminho=valor` (`mfsim run --set`), aplicados ao dicionário antes de validar."""
    d = load_json(path_or_dict) if isinstance(path_or_dict, (str, os.PathLike)) else dict(path_or_dict)
    if overrides:
        from .extends import apply_overrides

        apply_overrides(d, overrides)
    if validate:
        from .validate import validate_case

        validate_case(d, source=str(path_or_dict) if isinstance(path_or_dict, (str, os.PathLike)) else "")
    return network_from_json(d, ic_arrays=ic_arrays)
