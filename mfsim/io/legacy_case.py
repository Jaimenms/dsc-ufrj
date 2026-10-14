"""Registo legado de um caso (o struct `dados` do MFSIM/MATLAB como dataclasses).

É só um formato de entrada: o leitor dos `.mf2` (`mfsim.io_legacy`) produz este registo
e converte-o para o grafo (`Network.from_legacy_case`), que é o que o simulador consome;
`Network.to_legacy_case` é a inversa (round-trip = identidade, teste). Índices inteiros
de vértices e dutos, elementos "no vértice" (choke no vértice de destino, reservatório no
de origem, retenção num vértice dinâmico, conexão entre vértices) e opções em `extra`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from ..core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN  # noqa: F401  (reexportados)
from ..fluid import Fluid

# Especificação temporal: valor constante ou (tempos, valores) interpolados linearmente
SpecValue = Union[float, tuple]


@dataclass
class SimulatorSettings:
    t0: float = 0.0
    tf: float = 200.0
    dt_out: float = 2.0
    PR: float = 100e5                # pressão de referência (PT = P/PR)
    method: str = "galerkin"         # galerkin | fd_central | fd_upwind | petrov_galerkin | fv_upwind
    max_step: float = 1.0
    atol: float = 1e-3
    rtol: float = 1e-4
    quadrature_order: int = 1        # 1->3, 2->5, 3->8, 4->12 pontos de Gauss
    integrator: str = "bdf"          # bdf (scipy) | implicit (próprio)
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
    spec: dict = field(default_factory=dict)   # chaves: alphaG, PT, qL, qG
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
    roughness: float = 1e-3          # [m]
    model: int = 1                   # 1 two-fluid, 2 Beggs&Brill (3 eq), 3 líquido, 4 gás
    regime: int = 4                  # 1 anular, 2 bolhas, 3 intermitente, 4 est. suave, 5 est. ondulado, 6 automático
    bc_in: tuple = (1, 1, 2, 2)      # (alphaG, PT, qL, qG) na entrada
    bc_out: tuple = (1, 2, 1, 1)     # idem na saída
    check: bool = False              # campo do arquivo original (filtro anti-refluxo); SEM EFEITO: o refluxo é resolvido pelas equações
    ic: Optional[np.ndarray] = None  # condição inicial nodal (n_nodes x 4) ou None
    extra: dict = field(default_factory=dict)
    grading: Optional[Sequence[float]] = None   # razão h_último/h_primeiro por trecho (malha não uniforme)


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
    dP_min: float = 0.0     # bar


@dataclass
class Reservoir:
    """Reservatório (IPR linear) no vértice de origem de uma aresta: q = IP (P_res - P)."""

    vertex: int
    IPL: float = 10.0       # kg/s/bar
    IPG: float = 2.0
    P: float = 100.0        # bar


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
