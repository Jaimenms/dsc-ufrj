"""Especificações temporais (tabelas t x valor) com interpolação linear."""

from __future__ import annotations

import numpy as np

UNSPECIFIED = -1.0


class TimeTable:
    """Função do tempo definida por pares (t, valor), linear por partes,
    constante fora do intervalo. `None`/vazio -> UNSPECIFIED (-1)."""

    def __init__(self, spec, t0=0.0, tf=np.inf):
        if spec is None:
            self.t = np.array([t0])
            self.v = np.array([UNSPECIFIED])
        elif np.isscalar(spec):
            self.t = np.array([t0])
            self.v = np.array([float(spec)])
        else:
            t, v = spec
            t = np.atleast_1d(np.asarray(t, float)).ravel()
            v = np.atleast_1d(np.asarray(v, float)).ravel()
            if t.size == 0 and v.size == 1:
                t = np.array([t0])
            if t.size != v.size:
                raise ValueError(f"tabela de especificação inconsistente: {t.size} tempos, {v.size} valores")
            order = np.argsort(t)
            self.t, self.v = t[order], v[order]

    def __call__(self, t):
        return np.interp(t, self.t, self.v)

    @property
    def is_specified(self):
        return not (self.v.size == 1 and self.v[0] == UNSPECIFIED)

    @property
    def breakpoints(self):
        return self.t.copy()
