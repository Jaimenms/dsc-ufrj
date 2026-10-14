"""Grafo da rede: vértices e arestas com comportamento próprio.

Vértices: `BoundaryVertex` (especificações no tempo; legado "B"), `JunctionVertex`
(sem estados nem especificações: liga dutos a arestas de comprimento zero) e
`TankVertex` (estados HG, PT e volume; legado "A").

Arestas: `Pipe` (duto discretizado: geometria, malha, modelo, esquema, portas, condição
inicial) e as de comprimento zero `Connector` (acoplamento), `Choke` (válvula),
`Reservoir` (IPR) e `CheckValve` (retenção), todas com `v_from` e `v_to`.

O grafo é a representação que a rede consome (`network.system.NetworkSystem`, com
resolução de portas em `network.ports`) e a que os casos JSON descrevem. O `Case` legado
(`io.legacy_case`, o struct `dados` do MATLAB) converte-se sem perda por `Network.from_legacy_case`, que insere as
junções que o legado deixava implícitas (choke "no vértice de destino" -> `Jc<v>`,
reservatório "no vértice de origem" -> `Jr<v>`; a junção herda as especificações do
vértice legado para as portas que a aresta não fornece); `to_legacy_case` é a inversa
(round-trip = identidade, teste).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..core.options import IntegrationSpec, ModelSpec, ReferenceSpec, SchemeSpec, TimeSpec
from ..core.slots import PortKind, SLOT_NAMES
from ..fluid import Fluid
from .schedules import Schedule


def specs(**kw) -> dict:
    """Especificações de um vértice a partir de valores legados (escalar, (t, v)) ou `Schedule`."""
    return {k: (v if isinstance(v, Schedule) else Schedule.from_legacy(v)) for k, v in kw.items() if v is not None}


def ports(codes) -> tuple:
    """Portas de uma extremidade a partir de códigos legados (1/2/3) ou `PortKind`."""
    return tuple(c if isinstance(c, PortKind) else PortKind.from_legacy(c) for c in codes)


def as_network(obj) -> "Network":
    """O próprio `Network`, ou o grafo equivalente a um registo legado (`io.legacy_case.Case`)."""
    if isinstance(obj, Network):
        return obj
    from ..io.legacy_case import Case
    if isinstance(obj, Case):
        return Network.from_legacy_case(obj)
    raise TypeError(f"esperava Network ou Case legado, recebi {type(obj).__name__}")


# ----------------------------------------------------------------------------- vértices
@dataclass
class Vertex(ABC):
    id: str

    @property
    def n_states(self) -> int:
        return 0

    @property
    @abstractmethod
    def kind(self) -> str: ...


@dataclass
class BoundaryVertex(Vertex):
    """Vértice de fronteira: especificações alphaG, PT (P/PR), P (Pa) ou qL/qG (kg/s)."""

    spec: dict = field(default_factory=dict)      # chave -> Schedule

    @property
    def kind(self):
        return "boundary"


@dataclass
class JunctionVertex(Vertex):
    """Junção algébrica entre um duto e uma aresta de comprimento zero (sem estados).
    `spec` (opcional) dá os valores das portas que a aresta não fornece (nos casos
    convertidos do legado, as especificações do vértice que a junção desdobra)."""

    spec: dict = field(default_factory=dict)

    @property
    def kind(self):
        return "junction"


@dataclass
class TankVertex(Vertex):
    """Vértice dinâmico (tanque de mistura): estados HG e PT, volume [m³], fluxos externos."""

    volume: float = 1e-3
    spec: dict = field(default_factory=dict)
    HG0: Optional[float] = None
    PT0: Optional[float] = None

    @property
    def n_states(self):
        return 2

    @property
    def kind(self):
        return "tank"


# ----------------------------------------------------------------------------- arestas
@dataclass
class Edge(ABC):
    id: str
    v_from: str
    v_to: str

    @property
    @abstractmethod
    def type(self) -> str: ...


@dataclass
class Pipe(Edge):
    """Duto discretizado."""

    points: np.ndarray = None                    # (n_points, 3) xyz [m]
    n_elements: np.ndarray = None                # por segmento
    grading: Optional[np.ndarray] = None         # razão h_último/h_primeiro por segmento
    D: float = 0.1
    roughness: float = 1e-3                      # [m]
    model: ModelSpec = field(default_factory=ModelSpec)
    scheme: SchemeSpec = field(default_factory=SchemeSpec)
    ports_in: tuple = (PortKind.FREE, PortKind.FREE, PortKind.DIRICHLET, PortKind.DIRICHLET)
    ports_out: tuple = (PortKind.FREE, PortKind.DIRICHLET, PortKind.FREE, PortKind.FREE)
    ic: Optional[np.ndarray] = None              # (n_mesh_nodes, 4) ou None
    ic_recipe: Optional[dict] = None             # receita (fase 1b) em alternativa aos arrays
    point_names: tuple = ()                      # nomes opcionais dos pontos (marcadores)

    @property
    def type(self):
        return "pipe"

    # traçado por coordenada e código legado do padrão (só leitura; para alterar use
    # `points` e `Network.set_options(regime=...)`)
    @property
    def x(self):
        return np.asarray(self.points, float)[:, 0]

    @property
    def y(self):
        return np.asarray(self.points, float)[:, 1]

    @property
    def z(self):
        return np.asarray(self.points, float)[:, 2]

    @property
    def regime(self):
        return self.model.to_legacy()[1]

    @classmethod
    def legacy(cls, id, v_from, v_to, x, y, z, n_elements, D=0.1, roughness=1e-3, model=1, regime=4, method="galerkin",
               bc_in=(1, 1, 2, 2), bc_out=(1, 2, 1, 1), ic=None, extra=None, grading=None) -> "Pipe":
        """Duto no dialeto legado (traçado x, y, z; modelo/padrão por código; opções em `extra`)."""
        e = dict(extra or {})
        return cls(id=str(id), v_from=str(v_from), v_to=str(v_to),
                   points=np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], axis=1),
                   n_elements=np.atleast_1d(np.asarray(n_elements, int)), grading=None if grading is None else np.asarray(grading, float),
                   D=float(D), roughness=float(roughness), model=ModelSpec.from_legacy(model, regime, e), scheme=SchemeSpec.from_legacy(method, e),
                   ports_in=ports(bc_in), ports_out=ports(bc_out), ic=None if ic is None else np.asarray(ic, float))


@dataclass
class ZeroLengthEdge(Edge):
    """Aresta sem estados: relação algébrica entre os dois vértices."""


@dataclass
class Connector(ZeroLengthEdge):
    """Acoplamento: pressão de v_from = pressão no 1.º nó do duto que sai de v_to;
    vazões de v_to = vazões no último nó dos dutos que chegam a v_from."""

    @property
    def type(self):
        return "connector"


@dataclass
class Choke(ZeroLengthEdge):
    """Válvula (choke) entre a junção de saída de um duto (v_from) e o vértice de jusante (v_to):
    P_montante = P_jusante + qM|qM|/(rhoM (0.0075 CV ab)²) [bar]."""

    CV: float = 1000.0
    opening: Schedule = None
    mode: Schedule = None
    setpoint: Schedule = None
    pid: dict = field(default_factory=dict)      # pv_type, pv_node, pv_pipe, action, Kp, Ki, Kd, bias (não implementado)

    @property
    def type(self):
        return "choke"


@dataclass
class Reservoir(ZeroLengthEdge):
    """Reservatório (IPR linear) do vértice de fronteira v_from para a junção de entrada do duto (v_to):
    q = IP (P_res - P)."""

    IPL: float = 10.0
    IPG: float = 2.0
    P: float = 100.0                              # bar

    @property
    def type(self):
        return "reservoir"


@dataclass
class CheckValve(ZeroLengthEdge):
    """Válvula de retenção de um tanque (v_from) para o vértice cuja pressão é a de jusante (v_to)."""

    CV: float = 1000.0
    dP_min: float = 0.0                           # bar

    @property
    def type(self):
        return "check_valve"


# ----------------------------------------------------------------------------- rede
@dataclass
class Network:
    name: str
    fluid: Fluid
    time: TimeSpec
    reference: ReferenceSpec
    integration: IntegrationSpec
    vertices: list
    edges: list
    meta: dict = field(default_factory=dict)
    description: str = ""

    # --- construção --------------------------------------------------------
    @classmethod
    def build(cls, name, fluid, vertices, edges, *, tf, dt_out, t0=0.0, PR=1e6, integrator="bdf", quadrature_order=1,
              max_step=1.0, atol=1e-3, rtol=1e-4, state_ref=None, meta=None, **integration) -> "Network":
        """Grafo a partir das listas de vértices e arestas e das opções de tempo/referência/integração."""
        it = IntegrationSpec(integrator=integrator, method_quadrature=int(quadrature_order), max_step=float(max_step),
                             atol=float(atol), rtol=float(rtol), **integration)
        net = cls(name=name, fluid=fluid, time=TimeSpec(float(t0), float(tf), float(dt_out)),
                  reference=ReferenceSpec(float(PR), tuple(sorted((state_ref or {}).items()))), integration=it,
                  vertices=list(vertices), edges=list(edges), meta=dict(meta or {}))
        return net.validate()

    # --- consultas ---------------------------------------------------------
    def vertex(self, vid):
        vid = str(vid)
        for v in self.vertices:
            if v.id == vid:
                return v
        raise KeyError(vid)

    def pipe(self, pid):
        pid = str(pid)
        for p in self.pipes:
            if p.id == pid:
                return p
        raise KeyError(pid)

    # --- edição (em lugar; devolvem o próprio grafo) -------------------------
    def set_method(self, method):
        """Esquema `method` em todos os dutos (opções mantidas)."""
        for p in self.pipes:
            p.scheme = SchemeSpec(type=method, options=p.scheme.options)
        return self

    def set_time(self, **kw):
        self.time = replace(self.time, **kw)
        return self

    def set_integration(self, **kw):
        self.integration = replace(self.integration, **kw)
        return self

    def set_reference(self, PR=None, state_ref=None):
        self.reference = ReferenceSpec(self.reference.PR if PR is None else float(PR),
                                       self.reference.state_ref if state_ref is None else tuple(sorted(dict(state_ref).items())))
        return self

    def set_options(self, pipes=None, **legacy):
        """Opções no dialeto legado (`model`, `regime`, `method` e as chaves de `Pipe.extra`:
        k_switch, conservative, art_visc, bb_derivatives, ...) nos dutos `pipes` (ids; todos
        por omissão): reconstrói os `ModelSpec`/`SchemeSpec` com as chaves alteradas."""
        ids = None if pipes is None else {str(p) for p in pipes}
        for p in self.pipes:
            if ids is not None and p.id not in ids:
                continue
            model_id, regime_id, extra = p.model.to_legacy()
            extra.update(p.scheme.to_legacy_extra())
            kw = dict(legacy)
            model_id = kw.pop("model", model_id)
            regime_id = kw.pop("regime", regime_id)
            method = kw.pop("method", p.scheme.type)
            extra.update(kw)
            p.model = ModelSpec.from_legacy(model_id, regime_id, extra)
            p.scheme = SchemeSpec.from_legacy(method, extra)
        return self

    @property
    def pipes(self):
        return [e for e in self.edges if isinstance(e, Pipe)]

    def edges_of(self, cls):
        return [e for e in self.edges if isinstance(e, cls)]

    def validate(self):
        ids = [v.id for v in self.vertices]
        if len(set(ids)) != len(ids):
            raise ValueError(f"ids de vértices repetidos: {ids}")
        eids = [e.id for e in self.edges]
        if len(set(eids)) != len(eids):
            raise ValueError(f"ids de arestas repetidos: {eids}")
        for e in self.edges:
            for vid in (e.v_from, e.v_to):
                if vid not in ids:
                    raise ValueError(f"aresta {e.id}: vértice {vid} inexistente")
        for p in self.pipes:
            if p.points is None or p.points.shape[0] < 2:
                raise ValueError(f"duto {p.id}: geometria precisa de >= 2 pontos")
            if p.n_elements is None or len(p.n_elements) != p.points.shape[0] - 1:
                raise ValueError(f"duto {p.id}: n_elements deve ter um valor por segmento")
        return self

    # --- legado ------------------------------------------------------------
    @classmethod
    def from_legacy_case(cls, case) -> "Network":
        """Grafo equivalente a um `Case` legado (índices como ids). As junções implícitas
        do legado ganham vértices: `Jc<v>` para chokes (duto termina em Jc<v>, choke Jc<v> -> v)
        e `Jr<v>` para reservatórios (reservatório v -> Jr<v>, duto parte de Jr<v>)."""
        s = case.settings
        vertices = []
        for v in case.vertices:
            spec = {k: Schedule.from_legacy(val) for k, val in v.spec.items()}
            if v.kind == "A":
                vertices.append(TankVertex(id=str(v.index), volume=float(v.volume), spec=spec, HG0=v.HG0, PT0=v.PT0))
            else:
                vertices.append(BoundaryVertex(id=str(v.index), spec=spec))
        choke_at = {str(e.vertex): e for e in case.valves}
        res_at = {str(e.vertex): e for e in case.reservoirs}
        edges = []
        for p in case.pipes:
            e = dict(p.extra); e.pop("ic_file", None)       # campo do carregador legado, nunca lido
            v_in, v_out = str(p.v_in), str(p.v_out)
            if v_in in res_at:
                v_in = f"Jr{v_in}"
            if v_out in choke_at:
                v_out = f"Jc{v_out}"
            edges.append(Pipe(id=str(p.index), v_from=v_in, v_to=v_out, points=np.stack([np.asarray(p.x, float), np.asarray(p.y, float), np.asarray(p.z, float)], axis=1),
                              n_elements=np.atleast_1d(np.asarray(p.n_elements, int)),
                              grading=None if p.grading is None else np.asarray(p.grading, float), D=float(p.D), roughness=float(p.roughness),
                              model=ModelSpec.from_legacy(p.model, p.regime, e), scheme=SchemeSpec.from_legacy(s.method, e),
                              ports_in=tuple(PortKind.from_legacy(c) for c in p.bc_in), ports_out=tuple(PortKind.from_legacy(c) for c in p.bc_out),
                              ic=None if p.ic is None else np.asarray(p.ic, float)))
        legacy_spec = {v.id: dict(v.spec) for v in vertices}
        for vid, e in choke_at.items():
            vertices.append(JunctionVertex(id=f"Jc{vid}", spec=legacy_spec.get(vid, {})))
            edges.append(Choke(id=f"choke_{vid}", v_from=f"Jc{vid}", v_to=vid, CV=float(e.CV), opening=Schedule.from_legacy(e.opening),
                               mode=Schedule.from_legacy(e.mode), setpoint=Schedule.from_legacy(e.setpoint),
                               pid=dict(pv_type=e.pv_type, pv_node=e.pv_node, pv_pipe=e.pv_pipe, action=e.action, Kp=e.Kp, Ki=e.Ki, Kd=e.Kd, bias=e.bias)))
        for vid, e in res_at.items():
            vertices.append(JunctionVertex(id=f"Jr{vid}", spec=legacy_spec.get(vid, {})))
            edges.append(Reservoir(id=f"reservoir_{vid}", v_from=vid, v_to=f"Jr{vid}", IPL=float(e.IPL), IPG=float(e.IPG), P=float(e.P)))
        couplings = {str(c.v_up): str(c.v_down) for c in case.couplings}
        for e in case.check_valves:
            vid = str(e.vertex)
            edges.append(CheckValve(id=f"check_{vid}", v_from=vid, v_to=couplings.get(vid, vid), CV=float(e.CV), dP_min=float(e.dP_min)))
        for c in case.couplings:
            edges.append(Connector(id=f"connector_{c.v_up}_{c.v_down}", v_from=str(c.v_up), v_to=str(c.v_down)))
        ex = dict(s.extra)
        state_ref = ex.pop("state_ref", {}); ex.pop("ic_file", None)
        gb = ex.pop("galerkin_band", None)
        # opções do integrador que os drivers guardam em settings.extra (fase 1): campos próprios
        integ = {k: ex.pop(k) for k in ("tol_range", "max_newton", "slow_exit", "backend", "first_step", "max_order", "newton_tol") if k in ex}
        integration = IntegrationSpec(integrator=s.integrator, method_quadrature=int(s.quadrature_order), max_step=float(s.max_step),
                                      atol=float(s.atol), rtol=float(s.rtol), galerkin_band=gb, extra=tuple(sorted(ex.items())),
                                      tol_range=integ.get("tol_range"), max_newton=int(integ.get("max_newton", 12)),
                                      slow_exit=bool(integ.get("slow_exit", False)), backend=integ.get("backend", "numpy"),
                                      first_step=float(integ.get("first_step", 1e-3)), max_order=int(integ.get("max_order", 2)),
                                      newton_tol=float(integ.get("newton_tol", 1e-3)))
        net = cls(name=case.name, fluid=case.fluid, time=TimeSpec(float(s.t0), float(s.tf), float(s.dt_out)),
                  reference=ReferenceSpec(float(s.PR), tuple(sorted(state_ref.items()))), integration=integration,
                  vertices=vertices, edges=edges, meta=dict(getattr(case, "meta", {}) or {}))
        return net.validate()

    def to_legacy_case(self):
        """`Case` legado equivalente (o que o NetworkModel consome)."""
        from ..io.legacy_case import Case, CheckValve as LCheckValve, Coupling, Pipe as LPipe, Reservoir as LReservoir, SimulatorSettings, Valve, Vertex as LVertex
        self.validate()
        junction = {v.id for v in self.vertices if isinstance(v, JunctionVertex)}
        # junções colapsam no vértice legado: choke Jc -> v  (dutos que chegam a Jc passam a chegar a v);
        # reservatório v -> Jr (dutos que partem de Jr passam a partir de v)
        collapse = {}
        for e in self.edges_of(Choke):
            if e.v_from in junction:
                collapse[e.v_from] = e.v_to
        for e in self.edges_of(Reservoir):
            if e.v_to in junction:
                collapse[e.v_to] = e.v_from
        unresolved = junction - set(collapse)
        if unresolved:
            raise ValueError(f"junções sem choke/reservatório não têm equivalente legado: {sorted(unresolved)}")
        legacy_vertices = [v for v in self.vertices if not isinstance(v, JunctionVertex)]
        index = {v.id: (int(v.id) if v.id.lstrip("-").isdigit() else i + 1) for i, v in enumerate(legacy_vertices)}
        vertices = []
        for v in legacy_vertices:
            spec = {k: sch.to_legacy() for k, sch in v.spec.items()}
            if isinstance(v, TankVertex):
                vertices.append(LVertex(index=index[v.id], kind="A", volume=v.volume, spec=spec, HG0=v.HG0, PT0=v.PT0))
            else:
                vertices.append(LVertex(index=index[v.id], kind="B", volume=1e-3, spec=spec))
        methods = {p.scheme.type for p in self.pipes}
        if len(methods) > 1:
            raise ValueError(f"o Case legado admite um só esquema para todos os dutos: {methods}")
        pipes = []
        for i, p in enumerate(self.pipes):
            model_id, regime_id, extra = p.model.to_legacy()
            ic = p.ic
            if ic is None and p.ic_recipe is not None:
                from ..initial.recipes import materialize
                ic = materialize(p.ic_recipe, p.points, p.n_elements, p.grading, self.fluid, self.reference.PR)
            extra.update(p.scheme.to_legacy_extra())
            v_in = collapse.get(p.v_from, p.v_from); v_out = collapse.get(p.v_to, p.v_to)
            pipes.append(LPipe(index=int(p.id) if p.id.isdigit() else i + 1, v_in=index[v_in], v_out=index[v_out],
                               x=p.points[:, 0].copy(), y=p.points[:, 1].copy(), z=p.points[:, 2].copy(), n_elements=np.asarray(p.n_elements, int),
                               D=p.D, roughness=p.roughness, model=model_id, regime=regime_id,
                               bc_in=tuple(k.legacy for k in p.ports_in), bc_out=tuple(k.legacy for k in p.ports_out),
                               ic=None if ic is None else np.asarray(ic, float), extra=extra,
                               grading=None if p.grading is None else list(p.grading)))
        valves = [Valve(vertex=index[e.v_to], CV=e.CV, opening=e.opening.to_legacy(), mode=e.mode.to_legacy() if e.mode else 0.0,
                        setpoint=e.setpoint.to_legacy() if e.setpoint else 1.0, **{k: e.pid.get(k) for k in ("pv_node", "pv_pipe", "action", "Kp", "Ki", "Kd", "bias")},
                        pv_type=e.pid.get("pv_type", 1) if e.pid.get("pv_type") is not None else 1) for e in self.edges_of(Choke)]
        reservoirs = [LReservoir(vertex=index[e.v_from], IPL=e.IPL, IPG=e.IPG, P=e.P) for e in self.edges_of(Reservoir)]
        check_valves = [LCheckValve(vertex=index[e.v_from], CV=e.CV, dP_min=e.dP_min) for e in self.edges_of(CheckValve)]
        couplings = [Coupling(v_up=index[e.v_from], v_down=index[e.v_to]) for e in self.edges_of(Connector)]
        it = self.integration
        extra = dict(it.extra)
        if it.tol_range is not None or it.max_newton != 12 or it.slow_exit or it.backend != "numpy":
            extra.update(tol_range=it.tol_range, max_newton=it.max_newton, slow_exit=it.slow_exit, backend=it.backend,
                         first_step=it.first_step, max_order=it.max_order, newton_tol=it.newton_tol)
        if self.reference.state_ref:
            extra["state_ref"] = dict(self.reference.state_ref)
        if it.galerkin_band is not None:
            extra["galerkin_band"] = it.galerkin_band
        settings = SimulatorSettings(t0=self.time.t0, tf=self.time.tf, dt_out=self.time.dt_out, PR=self.reference.PR,
                                     method=next(iter(methods)) if methods else "galerkin", max_step=it.max_step, atol=it.atol, rtol=it.rtol,
                                     quadrature_order=it.method_quadrature, integrator=it.integrator, extra=extra)
        case = Case(name=self.name, fluid=self.fluid, settings=settings, vertices=vertices, pipes=pipes, valves=valves,
                    check_valves=check_valves, reservoirs=reservoirs, couplings=couplings)
        if self.meta:
            case.meta = dict(self.meta)
        return case
