"""Interface comum das discretizações de um duto.

Um esquema trabalha sobre um arranjo "nodal" Y (n_nodes x 4) com o layout
[alphaG, PT, qL, qG] por nó. O esquema informa:

  * n_nodes            : número de nós do arranjo
  * state_mask (n x 4) : quais entradas são variáveis de estado da EDO
                         (antes da remoção das condições de contorno)
  * fill_derived(Y)    : devolve Y com as entradas derivadas preenchidas (ex.: duplicatas)
  * rhs(Y) -> dY/dt    : derivadas temporais de todas as entradas de estado
  * boundary_fluxes(Y) : fluxos de massa (L, G) nas extremidades, para os balanços dos vértices
  * x_nodes            : coordenada curvilínea de cada nó (para saídas)
  * theta_nodes        : inclinação em cada nó

Para os esquemas nodais da tese (Galerkin, diferenças finitas) os nós são
os nós da malha (N+1). Para o esquema de volumes finitos escalonados os nós
são [contorno de entrada, N células, contorno de saída].

Backend: o esquema herda o backend numérico do modelo (`model.bk`, numpy ou JAX).
`rhs`, `fill_derived` e `boundary_fluxes` são funções puras escritas sobre `self.xp`
(sem escrita in-place; `if` só sobre configuração estática), de modo que o mesmo
código corre em numpy e traça/compila em JAX. As constantes da malha ficam em numpy
(atributos públicos) e em cópias no backend (prefixo `_x`) para o lado direito.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.slots import SLOT_QG, SLOT_QL


class PipeScheme(ABC):
    name = "base"

    def __init__(self, model, mesh, settings=None):
        self.model = model
        self.mesh = mesh
        self.settings = settings or {}
        self.bk = model.bk
        self.xp = model.xp
        self.h = np.asarray(mesh["h"], float)
        self.theta_elem = np.asarray(mesh["theta"], float)
        self.N = self.h.size

    # --- propriedades a serem definidas nas subclasses -----------------------
    @property
    @abstractmethod
    def n_nodes(self): ...

    @property
    @abstractmethod
    def x_nodes(self): ...

    @property
    @abstractmethod
    def theta_nodes(self): ...

    # nós cujo valor de pressão/vazão representa a extremidade do duto para a rede
    @property
    def inlet_P_node(self):
        return 0

    @property
    def outlet_P_node(self):
        return self.n_nodes - 1

    @property
    def inlet_q_node(self):
        return 0

    @property
    def outlet_q_node(self):
        return self.n_nodes - 1

    def state_mask(self):
        """Máscara (n_nodes x 4) das entradas que são estados do esquema."""
        mask = np.zeros((self.n_nodes, 4), dtype=bool)
        mask[:, list(self.model.slots)] = True
        return mask

    def prepare_ic(self, Y):
        """Condição inicial nodal -> estado do esquema (numpy; identidade por omissão)."""
        return np.asarray(Y, float)

    def fill_derived(self, Y):
        """Devolve Y com as entradas derivadas preenchidas (identidade por omissão)."""
        return Y

    @abstractmethod
    def rhs(self, Y): ...

    def boundary_fluxes(self, Y):
        """Fluxos de massa ((L_in, G_in), (L_out, G_out)) nas extremidades do duto: as
        vazões dos nós de entrada/saída (os esquemas conservativos devolvem os fluxos
        limitados das suas equações de massa)."""
        i0, i1 = self.inlet_q_node, self.outlet_q_node
        return (Y[i0, SLOT_QL], Y[i0, SLOT_QG]), (Y[i1, SLOT_QL], Y[i1, SLOT_QG])

    def outputs(self, Y):
        """Saídas secundárias nos nós (dict de arrays; numpy)."""
        F, out = self.model.coefficients(Y, self.theta_nodes, outputs=True)
        return out
