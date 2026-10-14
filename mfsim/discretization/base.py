"""Interface comum das discretizações de um duto.

Um esquema trabalha sobre um arranjo "nodal" Y (n_nodes x 4) com o layout
[alphaG, PT, qL, qG] por nó. O esquema informa:

  * n_nodes            : número de nós do arranjo
  * state_mask (n x 4) : quais entradas são variáveis de estado da EDO
                         (antes da remoção das condições de contorno)
  * fill_derived(Y)    : preenche entradas derivadas (ex.: duplicatas)
  * rhs(Y) -> dY/dt    : derivadas temporais de todas as entradas de estado
  * x_nodes            : coordenada curvilínea de cada nó (para saídas)
  * theta_nodes        : inclinação em cada nó

Para os esquemas nodais da tese (Galerkin, diferenças finitas) os nós são
os nós da malha (N+1). Para o esquema de volumes finitos escalonados os nós
são [contorno de entrada, N células, contorno de saída].
"""

from __future__ import annotations

import numpy as np


class PipeScheme:
    name = "base"

    def __init__(self, model, mesh, settings=None):
        self.model = model
        self.mesh = mesh
        self.settings = settings or {}
        self.h = np.asarray(mesh["h"], float)
        self.theta_elem = np.asarray(mesh["theta"], float)
        self.N = self.h.size

    # --- propriedades a serem definidas nas subclasses -----------------------
    @property
    def n_nodes(self):
        raise NotImplementedError

    @property
    def x_nodes(self):
        raise NotImplementedError

    @property
    def theta_nodes(self):
        raise NotImplementedError

    def state_mask(self):
        """Máscara (n_nodes x 4) das entradas que são estados do esquema."""
        mask = np.zeros((self.n_nodes, 4), dtype=bool)
        mask[:, list(self.model.slots)] = True
        return mask

    def fill_derived(self, Y):
        return Y

    def rhs(self, Y):
        raise NotImplementedError

    def outputs(self, Y):
        """Saídas secundárias nos nós (dict de arrays)."""
        F, out = self.model.coefficients(Y, self.theta_nodes, outputs=True)
        return out
