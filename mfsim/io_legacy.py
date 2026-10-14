"""Leitura dos arquivos .mf2 (MAT v5) do MFSIM/SDMPFLOW original.

Os arquivos contêm `dados` (entrada), `saidas` (resultados pós-processados),
`estados` (T, X) e `eixos`. Esta rotina converte tudo para dicts/arrays e
constrói o `Case` equivalente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

LEGACY_DIR = Path(__file__).resolve().parent.parent / "legacy" / "mfsim23" / "Files"


def _to_py(obj):
    """Converte mat_struct / arrays de objetos recursivamente em dict/list/ndarray."""
    if hasattr(obj, "_fieldnames"):
        return {f: _to_py(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_to_py(o) for o in obj.ravel()]
        return obj
    return obj


def load_mf2(path):
    """Carrega um arquivo .mf2 e retorna dict com dados/saidas/estados/eixos."""
    path = Path(path)
    if not path.exists() and (LEGACY_DIR / path.name).exists():
        path = LEGACY_DIR / path.name
    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    out = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        out[k] = _to_py(v)
    return out


def _vec(v):
    """Vetor 1-D (vazio -> array vazio)."""
    if v is None:
        return np.array([])
    a = np.atleast_1d(np.asarray(v, dtype=float))
    return a.ravel()


def _scalar(v, default=None):
    a = np.atleast_1d(np.asarray(v))
    if a.size == 0:
        return default
    return a.ravel()[0].item() if hasattr(a.ravel()[0], "item") else a.ravel()[0]


def _list_of_structs(d, key):
    v = d.get(key)
    if v is None:
        return []
    if isinstance(v, dict):
        return [v]
    return [x for x in v if isinstance(x, dict) and x]


def case_from_legacy(data, name=None):
    """Constrói um `Case` a partir do dict `dados` de um arquivo .mf2."""
    from .fluid import Fluid
    from .network.data import (
        BC_ABSENT,
        BC_DIRICHLET,
        BC_NEUMANN,
        Case,
        CheckValve,
        Coupling,
        Pipe,
        Reservoir,
        SimulatorSettings,
        Valve,
        Vertex,
    )

    sim = data["simulador"]
    fl = data["fluido"]
    fluid = Fluid.from_legacy(fl)
    method_map = {1: "fd_central", 2: "galerkin", 3: "petrov_galerkin", 4: "fd_upwind"}
    if "metodo" in sim:
        method = method_map.get(int(_scalar(sim["metodo"], 1)), "fd_central")
    else:
        # formato antigo (sem campo "metodo"): os resultados gravados foram
        # reproduzidos com Galerkin (resíduo ~1e-8 no estado inicial)
        method = "galerkin"
    settings = SimulatorSettings(
        t0=float(_scalar(sim["tempoInicial"])),
        tf=float(_scalar(sim["tempoFinal"])),
        dt_out=float(_scalar(sim["incrementoTempo"])),
        PR=float(_scalar(sim["pressaoR"])),
        method=method,
        max_step=float(_scalar(sim.get("maxStep", 1.0), 1.0)),
        atol=float(_scalar(sim.get("absTol", 1e-3), 1e-3)),
        rtol=float(_scalar(sim.get("relTol", 1e-3), 1e-3)),
        quadrature_order=int(_scalar(sim.get("ordemQuadratura", 1), 1)),
    )

    vertices = []
    for v in _list_of_structs(data, "vertice"):
        spec = {}
        for i, key in enumerate(("alphaG", "PT", "qL", "qG"), start=1):
            vals = _vec(v.get(f"especificacao{i}"))
            times = _vec(v.get(f"especificacaoT{i}"))
            if vals.size == 0:
                continue
            if vals.size == 1:
                spec[key] = float(vals[0])
            else:
                spec[key] = (times, vals)
        vertices.append(Vertex(
            index=int(_scalar(v["indice"])),
            kind="A" if int(_scalar(v["tipo"])) == 1 else "B",
            volume=float(_scalar(v.get("volume", 1e-3), 1e-3)),
            spec=spec,
            HG0=_scalar(v.get("condicaoInicial1"), None),
            PT0=_scalar(v.get("condicaoInicial2"), None),
        ))

    pipes = []
    for t in _list_of_structs(data, "tubo"):
        bc_in = tuple(int(_scalar(t[f"tipocondicaoContornoE{i}"])) for i in (1, 2, 3, 4))
        bc_out = tuple(int(_scalar(t[f"tipocondicaoContornoS{i}"])) for i in (1, 2, 3, 4))
        ic = []
        for i in (1, 2, 3, 4):
            ic.append(_vec(t.get(f"condicaoInicial{i}")))
        ic = None if any(a.size == 0 for a in ic) else np.stack(ic, axis=1)
        # Dirichlet apontando para especificação inexistente (-1) em alphaG/PT
        # -> gradiente nulo (semântica dos arquivos antigos: "conector")
        vin = int(_scalar(t["verticeOrigem"]))
        vout = int(_scalar(t["verticeDestino"]))
        vmap = {v.index: v for v in vertices}
        bc_in = list(bc_in)
        bc_out = list(bc_out)
        for s_, key in ((0, "alphaG"), (1, "PT")):
            if bc_in[s_] == BC_DIRICHLET and key not in vmap[vin].spec and vmap[vin].kind == "B":
                bc_in[s_] = BC_NEUMANN
            if bc_out[s_] == BC_DIRICHLET and key not in vmap[vout].spec and vmap[vout].kind == "B":
                bc_out[s_] = BC_NEUMANN
        bc_in, bc_out = tuple(bc_in), tuple(bc_out)
        pipes.append(Pipe(
            index=int(_scalar(t["indice"])),
            model=int(_scalar(t.get("modelo", 1), 1)),
            v_in=int(_scalar(t["verticeOrigem"])),
            v_out=int(_scalar(t["verticeDestino"])),
            x=_vec(t["x"]), y=_vec(t["y"]), z=_vec(t["z"]),
            n_elements=np.atleast_1d(np.asarray(t["numeroElementos"], int)),
            D=float(_scalar(t["diametro"])),
            roughness=float(_scalar(t["rugosidade"])) * 1e-6,
            regime=int(_scalar(t["regime"])),
            bc_in=bc_in, bc_out=bc_out,
            check=int(_scalar(t.get("check", 1), 1)) == 1,
            ic=ic,
        ))

    valves = []
    for v in _list_of_structs(data, "valvula"):
        def tbl(i):
            vals = _vec(v.get(f"especificacao{i}"))
            times = _vec(v.get(f"especificacaoT{i}"))
            if vals.size <= 1:
                return float(vals[0]) if vals.size else None
            return (times, vals)
        valves.append(Valve(
            vertex=int(_scalar(v["vertice"])), CV=float(_scalar(v["CV"])),
            opening=tbl(1), mode=tbl(2), setpoint=tbl(3),
            pv_type=int(_scalar(v.get("tipoPV", 1), 1)),
            pv_node=_scalar(v.get("nohPV"), None), pv_pipe=_scalar(v.get("linhaPV"), None),
            action=_scalar(v.get("acao"), None), Kp=_scalar(v.get("ContP"), None),
            Ki=_scalar(v.get("ContI"), None), Kd=_scalar(v.get("ContD"), None),
            bias=_scalar(v.get("ContBias"), None),
        ))

    check_valves = [
        CheckValve(vertex=int(_scalar(v["vertice"])), CV=float(_scalar(v["CV"])),
                   dP_min=float(_scalar(v.get("dPMin", 0.0), 0.0)))
        for v in _list_of_structs(data, "valvulaRet")
    ]
    reservoirs = [
        Reservoir(vertex=int(_scalar(r["vertice"])), IPL=float(_scalar(r["IPL"])),
                  IPG=float(_scalar(r["IPG"])), P=float(_scalar(r["P"])))
        for r in _list_of_structs(data, "reservatorio")
    ]
    couplings = [
        Coupling(v_up=int(_scalar(c["verticeE"])), v_down=int(_scalar(c["verticeS"])))
        for c in _list_of_structs(data, "acoplador")
    ]
    return Case(name=name or "legacy", fluid=fluid, settings=settings, vertices=vertices,
                pipes=pipes, valves=valves, check_valves=check_valves,
                reservoirs=reservoirs, couplings=couplings)


# Ajustes necessários para reproduzir os resultados gravados (os arquivos
# `dados` foram editados após as simulações originais):
#   CASO10-1-4: o modelo de Beggs & Brill foi executado com identificação
#               automática de padrão (regime 6) e não com padrão fixo 4;
#   CASO10-3-1: o poço simples foi simulado com padrão intermitente (3),
#               como descrito no texto da tese (seção 10.3).
LEGACY_OVERRIDES = {
    "CASO10-1-4": {"pipes": {1: {"regime": 6}}},
    "CASO10-3-1": {"pipes": {1: {"regime": 3}}},
}


def load_case(path, ic_from="legacy_t0", apply_overrides=True):
    """Carrega .mf2 e devolve (case, legacy) com os resultados originais.

    No MFSIM original, `dados.tubo.condicaoInicial*` é sobrescrito com o
    estado FINAL da simulação; o estado inicial da trajetória gravada está em
    `estados.X[:, 0]`. Com ic_from="legacy_t0" as condições iniciais do caso
    são tomadas desse estado (layout de estados idêntico ao original para os
    esquemas nodais). Com ic_from="file" mantém-se o que está no arquivo.
    """
    raw = load_mf2(path)
    case = case_from_legacy(raw["dados"], name=Path(path).stem)
    if apply_overrides and case.name in LEGACY_OVERRIDES:
        for idx, kv in LEGACY_OVERRIDES[case.name].get("pipes", {}).items():
            for k, v in kv.items():
                setattr(case.pipe(idx), k, v)
    for p in case.pipes:
        p.extra["ic_file"] = None if p.ic is None else p.ic.copy()
    T, X = legacy_states(raw)
    if ic_from == "legacy_t0" and X is not None and X.ndim == 2:
        from .network.model import NetworkModel
        model = NetworkModel(case)
        if model.n_states == X.shape[0]:
            model.state_to_case_ic(X[:, 0], case.settings.t0)
    return case, raw


def legacy_pipe_outputs(raw, pipe_idx):
    """Saídas do tubo pipe_idx (0-based) armazenadas no arquivo: dict de arrays (nós x tempos)."""
    saidas = raw.get("saidas", {})
    tubo = saidas.get("tubo")
    if tubo is None:
        return None
    if isinstance(tubo, dict):
        tubo = [tubo]
    return tubo[pipe_idx]


def legacy_states(raw):
    est = raw.get("estados")
    if not est:
        return None, None
    return np.asarray(est["T"], float).ravel(), np.asarray(est["X"], float)
