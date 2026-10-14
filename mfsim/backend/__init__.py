"""Backend numérico injetado nos elementos do modelo: numpy (eager) ou JAX (jit/jacfwd).

Uma só implementação da física corre nos dois: os métodos usam `bk.xp` (numpy ou
jax.numpy) para a aritmética e as primitivas do backend para o que difere entre os dois
(atualização funcional `set/add`, laços `while_loop/fori_loop`, `stop_gradient`, raízes
e sigmoide com derivada estável, inversa robusta, aplicação sob máscara, `jvp`).

Regras para os métodos de elemento: funções puras de arrays e configuração estática;
sem escrita in-place (só `bk.set/add`); `if` em Python só sobre configuração, nunca sobre
valores (usar `xp.where`); formas estáticas; `np.` só em constantes de construção; NaN
tratado com `where` em último; float64.
"""

from .base import Backend
from .numpy_backend import NUMPY, NumpyBackend

__all__ = ["Backend", "NUMPY", "NumpyBackend", "get_backend"]


def get_backend(name="numpy") -> Backend:
    """`"numpy"` (default) ou `"jax"` (import preguiçoso; liga float64 e a cache de compilação)."""
    if name in (None, "numpy", NUMPY):
        return NUMPY
    if isinstance(name, Backend):
        return name
    if name == "jax":
        from .jax_backend import JAX

        return JAX
    raise ValueError(f"backend desconhecido: {name}")
