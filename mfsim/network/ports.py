"""Resolução de portas da rede: plano estático de avaliação das arestas de comprimento zero.

Cada vértice tem quatro portas (alphaG, PT, qL, qG) cujo valor no instante t entra no
vetor de especificações E (NV x 4) que os dutos leem nas extremidades Dirichlet e os
tanques usam como fluxos externos. As portas são preenchidas, por esta ordem:

  1. tabelas dos vértices (`Schedule`; ausente = UNSPECIFIED);
  2. tanques: vazões externas não especificadas valem 0;
  3. passos das arestas de comprimento zero (`ConnectorPressure`, `ChokeStep`,
     `ReservoirStep`, `CheckValveStep`, `ConnectorFlows`), ordenados topologicamente
     pelas portas que leem e escrevem (desempate pela ordem legada entre classes e pela
     ordem das arestas no grafo: nos casos convertidos do legado o plano reproduz a
     montagem anterior bit a bit);
  4. estados dos tanques (HG, PT) nas suas portas alphaG, PT.

A "pressão de um vértice" lida por uma aresta é a do primeiro nó de um duto que parte do
vértice (diretamente ou pela junção de um reservatório) com a pressão livre (estado), se
existir — é o que o acoplamento legado propagava —, senão a porta PT em E nesse momento;
as "vazões que chegam a um vértice" são as do último nó dos dutos que lá chegam
diretamente ou pela junção de um choke. Cada porta tem no máximo um escritor (as vazões
dos acoplamentos para o mesmo vértice somam-se num só passo); conflitos levantam
ValueError na construção, nunca sobrescrita silenciosa.
"""
from __future__ import annotations

import heapq
from abc import ABC, abstractmethod

from ..core.slots import SLOT_P, SLOT_QG, SLOT_QL
from .elements import check_valve_flows, reservoir_flows, valve_upstream_pressure
from .graph import CheckValve, Choke, Connector, Reservoir, TankVertex


def legacy_index(vid):
    """Índice legado de um vértice (inteiro quando o id é numérico; senão o próprio id)."""
    s = str(vid)
    return int(s) if s.lstrip("-").isdigit() else s


class Step(ABC):
    """Um passo do plano: lê portas/estados e escreve portas de E (função pura em `bk`)."""

    weight = 0            # ordem legada entre classes (desempate)
    order = 0             # ordem da aresta no grafo (desempate)
    reads = ()            # portas (vi, slot) de E lidas
    writes = ()           # portas (vi, slot) escritas

    @abstractmethod
    def __call__(self, x, E, opt_t, info, bk):
        """Devolve E atualizado (bk.set); `info` (dict ou None) recebe diagnósticos."""


class PressureSource:
    """Pressão de um vértice lida pelas arestas: estado do 1.º nó de um duto que parte do
    vértice com PT livre (o último desses dutos, como o acoplamento legado), senão E[v, PT]."""

    def __init__(self, vi, idx):
        self.vi, self.idx = vi, idx

    @property
    def reads(self):
        return () if self.idx is not None else ((self.vi, SLOT_P),)

    def __call__(self, x, E):
        return x[self.idx] if self.idx is not None else E[self.vi, SLOT_P]


class ConnectorPressure(Step):
    """Acoplamento, parte da pressão: E[v_from, PT] = pressão do vértice v_to (estado)."""

    weight = 0

    def __init__(self, order, vu, idx):
        self.order, self.vu, self.idx = order, vu, idx
        self.writes = ((vu, SLOT_P),)

    def __call__(self, x, E, opt_t, info, bk):
        return bk.set(E, (self.vu, SLOT_P), x[self.idx])


class ChokeStep(Step):
    """Choke J -> v: E[J, PT] = P_montante(P(v), vazões que chegam a J, abertura)."""

    weight = 1

    def __init__(self, order, edge, J, v, src, iL, iG, j, fluid, PR):
        self.order, self.edge, self.J, self.v, self.src, self.iL, self.iG, self.j, self.fluid, self.PR = order, edge, J, v, src, iL, iG, j, fluid, PR
        self.reads = src.reads
        self.writes = ((J, SLOT_P),)

    def __call__(self, x, E, opt_t, info, bk):
        qL = x[self.iL] if self.iL is not None else 0.0
        qG = x[self.iG] if self.iG is not None else 0.0
        P_down = self.src(x, E)
        opening = opt_t[self.j]
        ZP, dP = valve_upstream_pressure(P_down, qL, qG, opening, self.edge.CV, self.fluid, self.PR, bk=bk)
        if info is not None:
            info["valves"].append(dict(vertex=legacy_index(self.edge.v_to), Ab=opening, ZP=ZP * self.PR / 1e5, dP=dP,
                                       qL=qL, qG=qG, P_down=P_down * self.PR / 1e5))
        return bk.set(E, (self.J, SLOT_P), ZP)


class ReservoirStep(Step):
    """Reservatório R -> J: E[J, qL/qG] = IP (P_res - P(J))."""

    weight = 2

    def __init__(self, order, edge, J, src, PR):
        self.order, self.edge, self.J, self.src, self.PR = order, edge, J, src, PR
        self.reads = src.reads
        self.writes = ((J, SLOT_QL), (J, SLOT_QG))

    def __call__(self, x, E, opt_t, info, bk):
        P = self.src(x, E)
        ZL, ZG, dP = reservoir_flows(P, self.edge.IPL, self.edge.IPG, self.edge.P, self.PR, bk=bk)
        if info is not None:
            info["reservoirs"].append(dict(vertex=legacy_index(self.edge.v_from), ZL=ZL, ZG=ZG, dP=dP, P=P * self.PR / 1e5))
        E = bk.set(E, (self.J, SLOT_QL), ZL)
        return bk.set(E, (self.J, SLOT_QG), ZG)


class CheckValveStep(Step):
    """Retenção tanque -> v: E[tanque, qL/qG] = -q(P_tanque, P(v), HG) (saída do tanque)."""

    weight = 3

    def __init__(self, order, edge, tank, iH, iP, src, fluid, PR):
        self.order, self.edge, self.tank, self.iH, self.iP, self.src, self.fluid, self.PR = order, edge, tank, iH, iP, src, fluid, PR
        self.reads = src.reads
        self.writes = ((tank, SLOT_QL), (tank, SLOT_QG))

    def __call__(self, x, E, opt_t, info, bk):
        P_down = self.src(x, E)
        qL, qG, dP = check_valve_flows(x[self.iP], P_down, x[self.iH], self.edge.CV, self.edge.dP_min, self.fluid, self.PR, bk=bk)
        if info is not None:
            info["check_valves"].append(dict(vertex=legacy_index(self.edge.v_from), ZP=x[self.iP] * self.PR / 1e5, dP=dP, qL=qL, qG=qG))
        E = bk.set(E, (self.tank, SLOT_QL), -qL)
        return bk.set(E, (self.tank, SLOT_QG), -qG)


class ConnectorFlows(Step):
    """Acoplamentos com o mesmo v_to, parte das vazões: E[v_to, qL/qG] = soma, na ordem das
    arestas, das vazões no último nó dos dutos que chegam a cada v_from."""

    weight = 4

    def __init__(self, order, vd, pairs):
        self.order, self.vd, self.pairs = order, vd, tuple(pairs)      # pairs: (iL, iG) por duto, já na ordem legada
        self.writes = ((vd, SLOT_QL), (vd, SLOT_QG))

    def __call__(self, x, E, opt_t, info, bk):
        qL = 0.0
        qG = 0.0
        for iL, iG in self.pairs:
            if iL is not None:
                qL = qL + x[iL]
            if iG is not None:
                qG = qG + x[iG]
        E = bk.set(E, (self.vd, SLOT_QL), qL)
        return bk.set(E, (self.vd, SLOT_QG), qG)


def build_plan(system):
    """Passos das arestas de comprimento zero de `system` (NetworkSystem) em ordem
    topológica pelas portas lidas/escritas (desempate: ordem legada das classes, ordem das
    arestas). Verifica que cada porta tem um só escritor."""
    net, vpos, pipes = system.network, system.vpos, system.pipes
    fluid, PR = system.fluid, system.PR
    pipes_in, pipes_out = system.pipes_in, system.pipes_out
    arriving, leaving = system.arriving, system.leaving          # através das junções de chokes/reservatórios

    def pressure_source(vi):
        idx = None
        for k in leaving[vi]:
            i = pipes[k].state_index(pipes[k].inlet_P_node, SLOT_P)
            if i is not None:
                idx = i
        return PressureSource(vi, idx)

    steps = []
    flows_by_vd = {}
    chokes = []
    for order, e in enumerate(net.edges):
        if isinstance(e, Connector):
            vu, vd = vpos[e.v_from], vpos[e.v_to]
            src = pressure_source(vd)
            if src.idx is not None:
                steps.append(ConnectorPressure(order, vu, src.idx))
            pairs = []
            for k in arriving[vu]:
                pb = pipes[k]
                pairs.append((pb.state_index(pb.outlet_q_node, SLOT_QL), pb.state_index(pb.outlet_q_node, SLOT_QG)))
            if vd not in flows_by_vd:
                flows_by_vd[vd] = ConnectorFlows(order, vd, [])
                steps.append(flows_by_vd[vd])
            flows_by_vd[vd].pairs = flows_by_vd[vd].pairs + tuple(pairs)
        elif isinstance(e, Choke):
            J, v = vpos[e.v_from], vpos[e.v_to]
            if len(pipes_in[J]) != 1:
                raise ValueError(f"choke {e.id}: a junção {e.v_from} deve receber exatamente um duto (recebe {len(pipes_in[J])})")
            pb = pipes[pipes_in[J][0]]
            steps.append(ChokeStep(order, e, J, v, pressure_source(v), pb.state_index(pb.outlet_q_node, SLOT_QL),
                                   pb.state_index(pb.outlet_q_node, SLOT_QG), len(chokes), fluid, PR))
            chokes.append(e)
        elif isinstance(e, Reservoir):
            J = vpos[e.v_to]
            if len(pipes_out[J]) != 1:
                raise ValueError(f"reservatório {e.id}: da junção {e.v_to} deve partir exatamente um duto (partem {len(pipes_out[J])})")
            steps.append(ReservoirStep(order, e, J, pressure_source(J), PR))
        elif isinstance(e, CheckValve):
            tank = vpos[e.v_from]
            if not isinstance(net.vertices[tank], TankVertex):
                raise ValueError(f"válvula de retenção {e.id}: {e.v_from} deve ser um tanque")
            iH, iP = system.vertex_A_index(e.v_from)
            steps.append(CheckValveStep(order, e, tank, iH, iP, pressure_source(vpos[e.v_to]), fluid, PR))
    # cada porta com um só escritor
    writers = {}
    for st in steps:
        for port in st.writes:
            if port in writers:
                raise ValueError(f"porta {port} escrita por dois passos: {type(writers[port]).__name__} e {type(st).__name__}")
            writers[port] = st
    # ordenação topológica (Kahn) com prioridade (classe legada, ordem da aresta)
    deps = {id(st): {id(writers[p]) for p in st.reads if p in writers and writers[p] is not st} for st in steps}
    by_id = {id(st): st for st in steps}
    indeg = {k: len(v) for k, v in deps.items()}
    heap = [(st.weight, st.order, id(st)) for st in steps if indeg[id(st)] == 0]
    heapq.heapify(heap)
    plan = []
    while heap:
        _, _, k = heapq.heappop(heap)
        st = by_id[k]
        plan.append(st)
        for other in steps:
            if k in deps[id(other)]:
                deps[id(other)].discard(k)
                indeg[id(other)] -= 1
                if indeg[id(other)] == 0:
                    heapq.heappush(heap, (other.weight, other.order, id(other)))
    if len(plan) != len(steps):
        raise ValueError("dependência cíclica entre as arestas de comprimento zero")
    return plan, chokes
