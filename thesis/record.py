"""Registo legado de um caso (o struct `dados` do MFSIM/MATLAB como dataclasses).

É só um formato de entrada: o leitor dos `.mf2` (`thesis.mf2`) produz este registo
e converte-o para o grafo (`network_from_record`), que é o que o simulador consome;
`record_from_network` é a inversa (round-trip = identidade, teste). Índices inteiros
de vértices e dutos, elementos "no vértice" (choke no vértice de destino, reservatório no
de origem, retenção num vértice dinâmico, conexão entre vértices) e opções em `extra`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from mfsim.core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN  # noqa: F401  (reexportados)
from mfsim.fluid import Fluid

# Especificação temporal: valor constante ou (tempos, valores) interpolados linearmente
SpecValue = Union[float, tuple]


@dataclass
class SimulatorSettings:
    t0: float = 0.0
    tf: float = 200.0
    dt_out: float = 2.0
    PR: float = 100e5  # pressão de referência (PT = P/PR)
    method: str = "galerkin"  # galerkin | fd_central | fd_upwind | petrov_galerkin | fv_upwind
    max_step: float = 1.0
    atol: float = 1e-3
    rtol: float = 1e-4
    quadrature_order: int = 1  # 1->3, 2->5, 3->8, 4->12 pontos de Gauss
    integrator: str = "bdf"  # bdf (scipy) | implicit (próprio)
    extra: dict = field(default_factory=dict)

    @property
    def n_quad(self):
        return [3, 5, 8, 12][max(0, min(3, self.quadrature_order - 1))]


@dataclass
class Vertex:
    """Vértice da rede.

    kind 'A': vértice dinâmico (estados HG, PT; volume V). Especificações
              qL, qG = vazões externas (positivas entrando).
    kind 'B': vértice estático: especificações alphaG, PT (adimensional P/PR),
              qL, qG impostas como condição de contorno Dirichlet.
    """

    index: int
    kind: str = "B"
    volume: float = 1e-3
    spec: dict = field(default_factory=dict)  # chaves: alphaG, PT, qL, qG
    HG0: Optional[float] = None
    PT0: Optional[float] = None


@dataclass
class Pipe:
    index: int
    v_in: int
    v_out: int
    x: Sequence[float]
    y: Sequence[float]
    z: Sequence[float]
    n_elements: Sequence[int]
    D: float = 0.1
    roughness: float = 1e-3  # [m]
    model: int = 1  # 1 two-fluid, 2 Beggs&Brill (3 eq), 3 líquido, 4 gás
    regime: int = 4  # 1 anular, 2 bolhas, 3 intermitente, 4 est. suave, 5 est. ondulado, 6 automático
    bc_in: tuple = (1, 1, 2, 2)  # (alphaG, PT, qL, qG) na entrada
    bc_out: tuple = (1, 2, 1, 1)  # idem na saída
    check: bool = (
        False  # campo do arquivo original (filtro anti-refluxo); SEM EFEITO: o refluxo é resolvido pelas equações
    )
    ic: Optional[np.ndarray] = None  # condição inicial nodal (n_nodes x 4) ou None
    extra: dict = field(default_factory=dict)
    grading: Optional[Sequence[float]] = None  # razão h_último/h_primeiro por trecho (malha não uniforme)


@dataclass
class Valve:
    """Válvula (choke) instalada no vértice de destino de uma aresta.

    dP[bar] = qM |qM| / rhoM / (0.0075 CV ab)^2 ; P_montante = P_vertice + dP.
    Abertura: manual (opening) ou PID (mode=1) sobre uma variável de processo.
    """

    vertex: int
    CV: float = 1000.0
    opening: SpecValue = 1.0
    mode: SpecValue = 0.0
    setpoint: SpecValue = 1.0
    pv_type: int = 1
    pv_node: Optional[int] = None
    pv_pipe: Optional[int] = None
    action: Optional[float] = None
    Kp: Optional[float] = None
    Ki: Optional[float] = None
    Kd: Optional[float] = None
    bias: Optional[float] = None


@dataclass
class CheckValve:
    """Válvula de retenção num vértice dinâmico (ex.: válvula de gas lift)."""

    vertex: int
    CV: float = 1000.0
    dP_min: float = 0.0  # bar


@dataclass
class Reservoir:
    """Reservatório (IPR linear) no vértice de origem de uma aresta: q = IP (P_res - P)."""

    vertex: int
    IPL: float = 10.0  # kg/s/bar
    IPG: float = 2.0
    P: float = 100.0  # bar


@dataclass
class Coupling:
    """Conexão entre o vértice de destino v_up de uma aresta e o vértice de
    origem v_down de outra: pressão de v_up = pressão no 1º nó da aresta a
    jusante; vazões de v_down = vazões no último nó da aresta a montante."""

    v_up: int
    v_down: int


@dataclass
class Case:
    name: str
    fluid: Fluid
    settings: SimulatorSettings
    vertices: list
    pipes: list
    valves: list = field(default_factory=list)
    check_valves: list = field(default_factory=list)
    reservoirs: list = field(default_factory=list)
    couplings: list = field(default_factory=list)

    def vertex(self, index):
        for v in self.vertices:
            if v.index == index:
                return v
        raise KeyError(index)

    def pipe(self, index):
        for p in self.pipes:
            if p.index == index:
                return p
        raise KeyError(index)


# ----------------------------------------------------------------------------- registo <-> grafo
from mfsim.core.options import IntegrationSpec, ModelSpec, ReferenceSpec, SchemeSpec, TimeSpec  # noqa: E402,F401
from abc import ABC, abstractmethod  # noqa: E402,F401
from dataclasses import dataclass, field, replace  # noqa: E402,F401
from typing import Optional  # noqa: E402,F401
import numpy as np  # noqa: E402,F401
from mfsim.core.options import IntegrationSpec, ModelSpec, ReferenceSpec, SchemeSpec, TimeSpec  # noqa: E402,F401
from mfsim.core.slots import PortKind, SLOT_NAMES  # noqa: E402,F401
from mfsim.fluid import Fluid  # noqa: E402,F401
from mfsim.network.schedules import Schedule  # noqa: E402,F401
from mfsim.network import graph as _g  # noqa: E402
from mfsim.network.graph import ports, specs  # noqa: E402,F401
from mfsim.network.schedules import Constant, PiecewiseLinear, Schedule  # noqa: E402,F401

LVertex, LPipe, LValve, LCheckValve, LReservoir, LCoupling = Vertex, Pipe, Valve, CheckValve, Reservoir, Coupling


def network_from_record(case) -> _g.Network:
    """Grafo equivalente a um `Case` legado (índices como ids). As junções implícitas
    do legado ganham vértices: `Jc<v>` para chokes (duto termina em Jc<v>, choke Jc<v> -> v)
    e `Jr<v>` para reservatórios (reservatório v -> Jr<v>, duto parte de Jr<v>)."""
    s = case.settings
    vertices = []
    for v in case.vertices:
        spec = {k: Schedule.from_legacy(val) for k, val in v.spec.items()}
        if v.kind == "A":
            vertices.append(_g.TankVertex(id=str(v.index), volume=float(v.volume), spec=spec, HG0=v.HG0, PT0=v.PT0))
        else:
            vertices.append(_g.BoundaryVertex(id=str(v.index), spec=spec))
    choke_at = {str(e.vertex): e for e in case.valves}
    res_at = {str(e.vertex): e for e in case.reservoirs}
    edges = []
    for p in case.pipes:
        e = dict(p.extra)
        e.pop("ic_file", None)  # campo do carregador legado, nunca lido
        v_in, v_out = str(p.v_in), str(p.v_out)
        if v_in in res_at:
            v_in = f"Jr{v_in}"
        if v_out in choke_at:
            v_out = f"Jc{v_out}"
        edges.append(
            _g.Pipe(
                id=str(p.index),
                v_from=v_in,
                v_to=v_out,
                points=np.stack([np.asarray(p.x, float), np.asarray(p.y, float), np.asarray(p.z, float)], axis=1),
                n_elements=np.atleast_1d(np.asarray(p.n_elements, int)),
                grading=None if p.grading is None else np.asarray(p.grading, float),
                D=float(p.D),
                roughness=float(p.roughness),
                model=ModelSpec.from_legacy(p.model, p.regime, e),
                scheme=SchemeSpec.from_legacy(s.method, e),
                ports_in=tuple(PortKind.from_legacy(c) for c in p.bc_in),
                ports_out=tuple(PortKind.from_legacy(c) for c in p.bc_out),
                ic=None if p.ic is None else np.asarray(p.ic, float),
            )
        )
    legacy_spec = {v.id: dict(v.spec) for v in vertices}
    for vid, e in choke_at.items():
        vertices.append(_g.JunctionVertex(id=f"Jc{vid}", spec=legacy_spec.get(vid, {})))
        edges.append(
            _g.Choke(
                id=f"choke_{vid}",
                v_from=f"Jc{vid}",
                v_to=vid,
                CV=float(e.CV),
                opening=Schedule.from_legacy(e.opening),
                mode=Schedule.from_legacy(e.mode),
                setpoint=Schedule.from_legacy(e.setpoint),
                pid=dict(
                    pv_type=e.pv_type,
                    pv_node=e.pv_node,
                    pv_pipe=e.pv_pipe,
                    action=e.action,
                    Kp=e.Kp,
                    Ki=e.Ki,
                    Kd=e.Kd,
                    bias=e.bias,
                ),
            )
        )
    for vid, e in res_at.items():
        vertices.append(_g.JunctionVertex(id=f"Jr{vid}", spec=legacy_spec.get(vid, {})))
        edges.append(
            _g.Reservoir(
                id=f"reservoir_{vid}", v_from=vid, v_to=f"Jr{vid}", IPL=float(e.IPL), IPG=float(e.IPG), P=float(e.P)
            )
        )
    couplings = {str(c.v_up): str(c.v_down) for c in case.couplings}
    for e in case.check_valves:
        vid = str(e.vertex)
        edges.append(
            _g.CheckValve(
                id=f"check_{vid}", v_from=vid, v_to=couplings.get(vid, vid), CV=float(e.CV), dP_min=float(e.dP_min)
            )
        )
    for c in case.couplings:
        edges.append(_g.Connector(id=f"connector_{c.v_up}_{c.v_down}", v_from=str(c.v_up), v_to=str(c.v_down)))
    ex = dict(s.extra)
    state_ref = ex.pop("state_ref", {})
    ex.pop("ic_file", None)
    gb = ex.pop("galerkin_band", None)
    # opções do integrador que os drivers guardam em settings.extra (fase 1): campos próprios
    integ = {k: ex.pop(k) for k in IntegrationSpec.OPTIONAL + IntegrationSpec.FLAGS + ("backend", "initial") if k in ex}
    integration = IntegrationSpec(
        integrator=s.integrator,
        method_quadrature=int(s.quadrature_order),
        max_step=float(s.max_step),
        atol=float(s.atol),
        rtol=float(s.rtol),
        galerkin_band=gb if gb is not None else integ.get("galerkin_band"),
        backend=integ.get("backend", "numpy"),
        accumulate=bool(integ.get("accumulate", False)),
        tight_bounds=bool(integ.get("tight_bounds", False)),
        extra=tuple(sorted(ex.items())),
        **{k: integ.get(k) for k in IntegrationSpec.OPTIONAL if k != "galerkin_band"},
    )
    initial = integ.get("initial", "edges")
    net = _g.Network(
        name=case.name,
        fluid=case.fluid,
        time=TimeSpec(float(s.t0), float(s.tf), float(s.dt_out)),
        reference=ReferenceSpec(float(s.PR), tuple(sorted(state_ref.items()))),
        integration=integration,
        vertices=vertices,
        edges=edges,
        meta=dict(getattr(case, "meta", {}) or {}),
        initial=initial,
    )
    return net.validate()


def record_from_network(net):
    """`Case` legado equivalente (o que o NetworkModel consome)."""

    net.validate()
    junction = {v.id for v in net.vertices if isinstance(v, _g.JunctionVertex)}
    # junções colapsam no vértice legado: choke Jc -> v  (dutos que chegam a Jc passam a chegar a v);
    # reservatório v -> Jr (dutos que partem de Jr passam a partir de v)
    collapse = {}
    for e in net.edges_of(_g.Choke):
        if e.v_from in junction:
            collapse[e.v_from] = e.v_to
    for e in net.edges_of(_g.Reservoir):
        if e.v_to in junction:
            collapse[e.v_to] = e.v_from
    unresolved = junction - set(collapse)
    if unresolved:
        raise ValueError(f"junções sem choke/reservatório não têm equivalente legado: {sorted(unresolved)}")
    legacy_vertices = [v for v in net.vertices if not isinstance(v, _g.JunctionVertex)]
    index = {v.id: (int(v.id) if v.id.lstrip("-").isdigit() else i + 1) for i, v in enumerate(legacy_vertices)}
    vertices = []
    for v in legacy_vertices:
        spec = {k: sch.to_legacy() for k, sch in v.spec.items()}
        if isinstance(v, _g.TankVertex):
            vertices.append(LVertex(index=index[v.id], kind="A", volume=v.volume, spec=spec, HG0=v.HG0, PT0=v.PT0))
        else:
            vertices.append(LVertex(index=index[v.id], kind="B", volume=1e-3, spec=spec))
    methods = {p.scheme.type for p in net.pipes}
    if len(methods) > 1:
        raise ValueError(f"o Case legado admite um só esquema para todos os dutos: {methods}")
    pipes = []
    for i, p in enumerate(net.pipes):
        model_id, regime_id, extra = p.model.to_legacy()
        ic = p.ic
        if ic is None and p.ic_recipe is not None:
            from ..initial.recipes import materialize

            ic = materialize(p.ic_recipe, p.points, p.n_elements, p.grading, net.fluid, net.reference.PR)
        extra.update(p.scheme.to_legacy_extra())
        v_in = collapse.get(p.v_from, p.v_from)
        v_out = collapse.get(p.v_to, p.v_to)
        pipes.append(
            LPipe(
                index=int(p.id) if p.id.isdigit() else i + 1,
                v_in=index[v_in],
                v_out=index[v_out],
                x=p.points[:, 0].copy(),
                y=p.points[:, 1].copy(),
                z=p.points[:, 2].copy(),
                n_elements=np.asarray(p.n_elements, int),
                D=p.D,
                roughness=p.roughness,
                model=model_id,
                regime=regime_id,
                bc_in=tuple(k.legacy for k in p.ports_in),
                bc_out=tuple(k.legacy for k in p.ports_out),
                ic=None if ic is None else np.asarray(ic, float),
                extra=extra,
                grading=None if p.grading is None else list(p.grading),
            )
        )
    valves = [
        Valve(
            vertex=index[e.v_to],
            CV=e.CV,
            opening=e.opening.to_legacy(),
            mode=e.mode.to_legacy() if e.mode else 0.0,
            setpoint=e.setpoint.to_legacy() if e.setpoint else 1.0,
            **{k: e.pid.get(k) for k in ("pv_node", "pv_pipe", "action", "Kp", "Ki", "Kd", "bias")},
            pv_type=e.pid.get("pv_type", 1) if e.pid.get("pv_type") is not None else 1,
        )
        for e in net.edges_of(_g.Choke)
    ]
    reservoirs = [LReservoir(vertex=index[e.v_from], IPL=e.IPL, IPG=e.IPG, P=e.P) for e in net.edges_of(_g.Reservoir)]
    check_valves = [LCheckValve(vertex=index[e.v_from], CV=e.CV, dP_min=e.dP_min) for e in net.edges_of(_g.CheckValve)]
    couplings = [Coupling(v_up=index[e.v_from], v_down=index[e.v_to]) for e in net.edges_of(_g.Connector)]
    it = net.integration
    extra = dict(it.extra)
    for k in IntegrationSpec.OPTIONAL:
        if k != "galerkin_band" and getattr(it, k) is not None:
            extra[k] = getattr(it, k)
    for k in IntegrationSpec.FLAGS:
        if getattr(it, k):
            extra[k] = True
    if it.backend != "numpy":
        extra["backend"] = it.backend
    if net.initial != "edges":
        extra["initial"] = net.initial
    if net.reference.state_ref:
        extra["state_ref"] = dict(net.reference.state_ref)
    if it.galerkin_band is not None:
        extra["galerkin_band"] = it.galerkin_band
    settings = SimulatorSettings(
        t0=net.time.t0,
        tf=net.time.tf,
        dt_out=net.time.dt_out,
        PR=net.reference.PR,
        method=next(iter(methods)) if methods else "galerkin",
        max_step=it.max_step,
        atol=it.atol,
        rtol=it.rtol,
        quadrature_order=it.method_quadrature,
        integrator=it.integrator,
        extra=extra,
    )
    case = Case(
        name=net.name,
        fluid=net.fluid,
        settings=settings,
        vertices=vertices,
        pipes=pipes,
        valves=valves,
        check_valves=check_valves,
        reservoirs=reservoirs,
        couplings=couplings,
    )
    if net.meta:
        case.meta = dict(net.meta)
    return case
