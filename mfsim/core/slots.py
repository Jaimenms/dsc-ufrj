"""Layout nodal de 4 variáveis por nó (alphaG, PT, qL, qG), tipos de porta nas
extremidades dos dutos e constantes físicas."""

from __future__ import annotations

from enum import Enum

SLOT_ALPHA, SLOT_P, SLOT_QL, SLOT_QG = 0, 1, 2, 3
SLOT_NAMES = ("alphaG", "PT", "qL", "qG")
G = 9.80665

# códigos legados das portas nas extremidades dos dutos (`PortKind.legacy`)
BC_ABSENT = 1  # extremidade livre (estado)
BC_DIRICHLET = 2  # valor imposto pelo vértice conectado
BC_NEUMANN = 3  # gradiente nulo (copia o nó vizinho)


class PortKind(str, Enum):
    """Condição numa extremidade do duto por variável (códigos legados 1/2/3):
    FREE = a extremidade é estado livre; DIRICHLET = valor imposto pelo vértice;
    NEUMANN = gradiente nulo (copia o nó vizinho)."""

    FREE = "free"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"

    @property
    def legacy(self) -> int:
        return {"free": 1, "dirichlet": 2, "neumann": 3}[self.value]

    @classmethod
    def from_legacy(cls, code) -> "PortKind":
        return {1: cls.FREE, 2: cls.DIRICHLET, 3: cls.NEUMANN}[int(code)]
