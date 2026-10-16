"""Grafo da rede: vértices e arestas com comportamento próprio.

Vértices: `BoundaryVertex` (especificações no tempo; legado "B"), `JunctionVertex`
(sem estados nem especificações: liga dutos a arestas de comprimento zero) e
`TankVertex` (estados HG, PT e volume; legado "A").

Arestas: `Pipe` (duto discretizado: geometria, malha, modelo, esquema, portas, condição
inicial) e as de comprimento zero `Connector` (acoplamento), `Choke` (válvula),
`Reservoir` (IPR) e `CheckValve` (retenção), todas com `v_from` e `v_to`.

O grafo é a representação que a rede consome (`network.system.NetworkSystem`, com
resolução de portas em `network.ports`) e a que os casos JSON descrevem. O `Case` legado
(`thesis.record`, o struct `dados` do MATLAB) converte-se sem perda por `thesis.record.network_from_record`, que insere as
junções que o legado deixava implícitas (choke "no vértice de destino" -> `Jc<v>`,
reservatório "no vértice de origem" -> `Jr<v>`; a junção herda as especificações do
vértice legado para as portas que a aresta não fornece); `record_from_network` é a inversa
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
    """O próprio `Network` (o registo legado da tese converte-se com `thesis.record.network_from_record`)."""
    if isinstance(obj, Network):
        return obj
    raise TypeError(f"esperava Network, recebi {type(obj).__name__}")


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

    spec: dict = field(default_factory=dict)  # chave -> Schedule

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

    points: np.ndarray = None  # (n_points, 3) xyz [m]
    n_elements: np.ndarray = None  # por segmento
    grading: Optional[np.ndarray] = None  # razão h_último/h_primeiro por segmento
    D: float = 0.1
    roughness: float = 1e-3  # [m]
    model: ModelSpec = field(default_factory=ModelSpec)
    scheme: SchemeSpec = field(default_factory=SchemeSpec)
    ports_in: tuple = (PortKind.FREE, PortKind.FREE, PortKind.DIRICHLET, PortKind.DIRICHLET)
    ports_out: tuple = (PortKind.FREE, PortKind.DIRICHLET, PortKind.FREE, PortKind.FREE)
    ic: Optional[np.ndarray] = None  # (n_mesh_nodes, 4) ou, com ic_at = "scheme", (nós do esquema, 4)
    ic_recipe: Optional[dict] = None  # receita (fase 1b) em alternativa aos arrays
    ic_at: str = "mesh"  # "mesh": nós da malha (interpolados); "scheme": estado exato do esquema (issue 040)
    point_names: tuple = ()  # nomes opcionais dos pontos (marcadores)

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
    def legacy(
        cls,
        id,
        v_from,
        v_to,
        x,
        y,
        z,
        n_elements,
        D=0.1,
        roughness=1e-3,
        model=1,
        regime=4,
        method="galerkin",
        bc_in=(1, 1, 2, 2),
        bc_out=(1, 2, 1, 1),
        ic=None,
        extra=None,
        grading=None,
    ) -> "Pipe":
        """Duto no dialeto legado (traçado x, y, z; modelo/padrão por código; opções em `extra`)."""
        e = dict(extra or {})
        return cls(
            id=str(id),
            v_from=str(v_from),
            v_to=str(v_to),
            points=np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], axis=1),
            n_elements=np.atleast_1d(np.asarray(n_elements, int)),
            grading=None if grading is None else np.asarray(grading, float),
            D=float(D),
            roughness=float(roughness),
            model=ModelSpec.from_legacy(model, regime, e),
            scheme=SchemeSpec.from_legacy(method, e),
            ports_in=ports(bc_in),
            ports_out=ports(bc_out),
            ic=None if ic is None else np.asarray(ic, float),
        )


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
    P_montante = P_jusante + dP(qL, qG, ab) [bar]. `law` escolhe a densidade da mistura:
    "homogeneous" (padrão, ponderada por volume, regularizada com `q_smooth` [m³/s]) ou
    "thesis" (ponderada por massa, a do MATLAB; os casos da tese usam-na). Ver
    `elements.valve_upstream_pressure` e a issue 039."""

    CV: float = 1000.0
    law: str = "homogeneous"
    q_smooth: float = 1e-6
    opening: Schedule = None
    mode: Schedule = None
    setpoint: Schedule = None
    pid: dict = field(default_factory=dict)  # pv_type, pv_node, pv_pipe, action, Kp, Ki, Kd, bias (não implementado)

    @property
    def type(self):
        return "choke"


@dataclass
class Reservoir(ZeroLengthEdge):
    """Reservatório (IPR linear) do vértice de fronteira v_from para a junção de entrada do duto (v_to):
    q = IP (P_res - P)."""

    IPL: float = 10.0
    IPG: float = 2.0
    P: float = 100.0  # bar

    @property
    def type(self):
        return "reservoir"


@dataclass
class CheckValve(ZeroLengthEdge):
    """Válvula de retenção de um tanque (v_from) para o vértice cuja pressão é a de jusante (v_to)."""

    CV: float = 1000.0
    dP_min: float = 0.0  # bar

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
    initial: str = "edges"  # estado inicial: "edges" (condições iniciais) | "guess" (estimativa) | "steady"
    # opções de initial = "steady" (issue 040): from ("edges" | "guess"), perturb, max_iter, on_failure
    initial_options: dict = field(default_factory=dict)

    # --- construção --------------------------------------------------------
    @classmethod
    def build(
        cls,
        name,
        fluid,
        vertices,
        edges,
        *,
        tf,
        dt_out,
        t0=0.0,
        PR=1e6,
        integrator="bdf",
        quadrature_order=1,
        max_step=1.0,
        atol=1e-3,
        rtol=1e-4,
        state_ref=None,
        meta=None,
        initial="edges",
        **integration,
    ) -> "Network":
        """Grafo a partir das listas de vértices e arestas e das opções de tempo/referência/integração."""
        it = IntegrationSpec(
            integrator=integrator,
            method_quadrature=int(quadrature_order),
            max_step=float(max_step),
            atol=float(atol),
            rtol=float(rtol),
            **integration,
        )
        net = cls(
            name=name,
            fluid=fluid,
            time=TimeSpec(float(t0), float(tf), float(dt_out)),
            reference=ReferenceSpec(float(PR), tuple(sorted((state_ref or {}).items()))),
            integration=it,
            vertices=list(vertices),
            edges=list(edges),
            meta=dict(meta or {}),
            initial=initial,
        )
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
        self.reference = ReferenceSpec(
            self.reference.PR if PR is None else float(PR),
            self.reference.state_ref if state_ref is None else tuple(sorted(dict(state_ref).items())),
        )
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
