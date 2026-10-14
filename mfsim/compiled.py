"""Lado direito da rede compilado (JAX) com Jacobiana por diferenciação automática.

`CompiledNetwork(model)` reconstrói o mesmo `NetworkSystem` (sobre o mesmo grafo) com o backend JAX (uma só
implementação da física, dos esquemas e da rede) e expõe

    rhs(t, x) -> dx/dt       (numpy, igual a NetworkModel.rhs a precisão de máquina)
    jac(t, x) -> d(rhs)/dx   (numpy denso, exato, por jacfwd)

As tabelas no tempo (especificações dos vértices, aberturas das válvulas) são avaliadas
em numpy a cada instante e passadas como argumentos ao núcleo compilado
`NetworkModel.rhs_core(x, E_t, opt_t)`. `build_jax_rhs(model)` e `get_jax(model)` mantêm a
interface anterior de `mfsim.jaxrhs`.
"""

from __future__ import annotations

import numpy as np

from .backend import get_backend
from .network.system import NetworkSystem


class CompiledNetwork:
    def __init__(self, model: NetworkSystem):
        import jax

        self.model = model
        self.backend = get_backend("jax")
        self.system = NetworkSystem(model.network, scheme_settings=model.scheme_settings, bk=self.backend)
        if self.system.n_states != model.n_states:
            raise RuntimeError("CompiledNetwork: layout de estados diferente do modelo numpy")
        self.rhs_jit = jax.jit(self.system.rhs_core)
        self.jac_jit = jax.jit(jax.jacfwd(self.system.rhs_core, argnums=0))
        self._cache = {}

    def inputs(self, t):
        """(E_t, opt_t) numpy no instante t (cache do último instante)."""
        key = float(t)
        if self._cache.get("t") == key:
            return self._cache["E"], self._cache["opt"]
        E, opt = self.model.time_inputs(t)
        self._cache.update(t=key, E=E, opt=opt)
        return E, opt

    def rhs(self, t, x):
        E, opt = self.inputs(t)
        return np.asarray(self.rhs_jit(x, E, opt))

    def jac(self, t, x):
        E, opt = self.inputs(t)
        return np.asarray(self.jac_jit(x, E, opt))


def build_jax_rhs(model):
    """(rhs, jac) compilados para um NetworkModel (interface anterior)."""
    cn = CompiledNetwork(model)

    def rhs(t, x):
        return cn.rhs(t, x)

    rhs.jit = cn.rhs_jit
    rhs.inputs = cn.inputs
    rhs.compiled = cn
    return rhs, cn.jac


def get_jax(model):
    """(rhs, jac) JAX do modelo, construídos uma vez e guardados no próprio modelo."""
    cached = getattr(model, "_jax_functions", None)
    if cached is None:
        cached = build_jax_rhs(model)
        model._jax_functions = cached
    return cached
