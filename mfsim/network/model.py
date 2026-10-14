"""`NetworkModel`: o sistema da rede (`NetworkSystem`, montado sobre o grafo) construído a
partir de um `Case` legado (`network.data`) ou diretamente de um `Network`, com as opções
de `build_model` (esquema global `method`, `scheme_settings`). O `Case` legado é só um
formato de entrada: converte-se por `Network.from_legacy_case` (sem perda; junções
explícitas para chokes e reservatórios) e o estado estacionário é escrito de volta nas
suas condições iniciais (`state_to_case_ic`)."""

from __future__ import annotations

from dataclasses import replace

from ..core.options import SchemeSpec
from .graph import Network, Pipe, TankVertex
from .system import NetworkSystem, PipeBlock  # noqa: F401  (PipeBlock: compatibilidade)


def with_method(network: Network, method: str) -> Network:
    """Cópia rasa do grafo com o esquema `method` em todos os dutos (opções mantidas)."""
    edges = [replace(e, scheme=SchemeSpec(type=method, options=e.scheme.options)) if isinstance(e, Pipe) else e for e in network.edges]
    return replace(network, edges=edges)


class NetworkModel(NetworkSystem):
    def __init__(self, case, method=None, scheme_settings=None, bk=None):
        if isinstance(case, Network):
            net, self.case = case, None
        else:
            net, self.case = Network.from_legacy_case(case), case
        if method is not None:
            net = with_method(net, method)
        super().__init__(net, scheme_settings=scheme_settings, bk=bk)
        self.settings = getattr(case, "settings", None)

    def state_to_case_ic(self, X, t=0.0):
        """Escreve o estado X nas condições iniciais do grafo e, se o modelo veio de um
        `Case` legado, também nos seus dutos e vértices."""
        self.state_to_network_ic(X, t)
        if self.case is not None:
            for pb, lp in zip(self.pipes, self.case.pipes):
                lp.ic = pb.pipe.ic
            for vi in self.vA:
                v = self.vertices[vi]
                for lv in self.case.vertices:
                    if str(lv.index) == v.id:
                        lv.HG0, lv.PT0 = v.HG0, v.PT0
