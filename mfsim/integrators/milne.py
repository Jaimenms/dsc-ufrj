"""Estimativa de erro de Milne (issue 058): agora é a opção `integration.error_estimate = "milne"` do `ImplicitBDF`
(preditor quadrático pelos três estados do histórico e constante de Milne das posições dos nós, só para a estimativa;
o Newton parte da extrapolação linear). Este módulo nasceu como ficheiro à parte enquanto `implicit.py` estava em
edição noutra sessão e fica como atalho: `ImplicitBDFMilne` (a classe com a opção ligada), `integrate_bdf_milne`
(`_integrate_bdf` com a opção) e `milne_order2` reexportada. O tampão do passo depois de falhas do Newton que aqui
se ensaiou (`newton_fail_cap`) não compensou e não passou para o `ImplicitBDF`."""

from __future__ import annotations

import dataclasses

from .implicit import ImplicitBDF, milne_order2  # noqa: F401


class ImplicitBDFMilne(ImplicitBDF):
    """`ImplicitBDF` com `error_estimate = "milne"` ligado."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_estimate = "milne"


def integrate_bdf_milne(model, X0, t_eval, spec, progress=True, verbose=False, partial=False):
    """`mfsim.simulation._integrate_bdf` com `error_estimate = "milne"`."""
    from ..simulation import _integrate_bdf

    spec = dataclasses.replace(spec, error_estimate="milne")
    return _integrate_bdf(model, X0, t_eval, spec, progress=progress, verbose=verbose, partial=partial)
