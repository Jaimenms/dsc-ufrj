"""MFSIM - Simulador dinâmico de escoamento bifásico em redes de dutos.

Reimplementação em Python do SDMPFLOW/MFSIM desenvolvido na tese de doutorado
de Jaime Neiva Miranda de Souza (UFRJ, 2010) (modelo a dois fluidos com 4 equações, modelo de mistura a
3 equações, modelos monofásicos, discretização espacial por Galerkin,
diferenças finitas e volumes finitos escalonados, integração implícita).
"""

from .fluid import Fluid
from .network.graph import (
    BoundaryVertex,
    CheckValve,
    Choke,
    Connector,
    JunctionVertex,
    Network,
    Pipe,
    Reservoir,
    TankVertex,
    as_network,
    ports,
    specs,
)

__all__ = [
    "Fluid",
    "Network",
    "BoundaryVertex",
    "JunctionVertex",
    "TankVertex",
    "Pipe",
    "Connector",
    "Choke",
    "Reservoir",
    "CheckValve",
    "as_network",
    "ports",
    "specs",
]


def __getattr__(name):
    # importações tardias para evitar ciclos na carga do pacote
    if name == "NetworkModel":
        from .network.model import NetworkModel

        return NetworkModel
    if name in ("simulate", "steady_state"):
        from . import simulation

        return getattr(simulation, name)
    if name == "Results":
        from .results import Results

        return Results
    raise AttributeError(name)
