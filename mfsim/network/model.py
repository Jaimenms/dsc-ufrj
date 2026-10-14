"""`NetworkModel`: o sistema da rede (`NetworkSystem`, montado sobre o grafo) construído a
partir de um `Network` (o struct
`dados` do MATLAB), com as opções de `build_model` (esquema global `method`,
`scheme_settings`). O registo legado da tese converte-se fora do simulador (`thesis.record.network_from_record`;
junções explícitas para chokes e reservatórios) e o estado estacionário é escrito de volta
nas suas condições iniciais (`state_to_case_ic`)."""

from __future__ import annotations

from dataclasses import replace

from ..core.options import SchemeSpec
from .graph import Network, Pipe, TankVertex
from .system import NetworkSystem, PipeBlock  # noqa: F401  (PipeBlock: compatibilidade)


def with_method(network: Network, method: str) -> Network:
    """Cópia rasa do grafo com o esquema `method` em todos os dutos (opções mantidas)."""
    edges = [
        replace(e, scheme=SchemeSpec(type=method, options=e.scheme.options)) if isinstance(e, Pipe) else e
        for e in network.edges
    ]
    return replace(network, edges=edges)


class NetworkModel(NetworkSystem):
    def __init__(self, case, method=None, scheme_settings=None, bk=None):
        net = case
        if method is not None:
            net = with_method(net, method)
        super().__init__(net, scheme_settings=scheme_settings, bk=bk)

    def state_to_case_ic(self, X, t=0.0):
        """Escreve o estado X nas condições iniciais do grafo."""
        self.state_to_network_ic(X, t)
