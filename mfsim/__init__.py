"""MFSIM - Simulador dinâmico de escoamento bifásico em redes de dutos.

Reimplementação em Python do SDMPFLOW/MFSIM desenvolvido na tese de doutorado
de Jaime Souza (modelo a dois fluidos com 4 equações, modelo de mistura a
3 equações, modelos monofásicos, discretização espacial por Galerkin,
diferenças finitas e volumes finitos escalonados, integração implícita).
"""

from .fluid import Fluid
from .network.data import (
    Case,
    CheckValve,
    Coupling,
    Pipe,
    Reservoir,
    SimulatorSettings,
    Valve,
    Vertex,
)

__all__ = [
    "Fluid", "Case", "CheckValve", "Coupling", "Pipe", "Reservoir",
    "SimulatorSettings", "Valve", "Vertex",
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
