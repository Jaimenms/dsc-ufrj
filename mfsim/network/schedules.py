"""Especificações no tempo (schedules): constante, linear por partes (a TimeTable legada) e
rampa. Conversão de e para a forma legada (escalar | (t, v)) e JSON."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Schedule(ABC):
    """Função do tempo usada nas especificações de vértices e aberturas de chokes."""

    @abstractmethod
    def __call__(self, t): ...

    @abstractmethod
    def to_legacy(self):
        """escalar ou (tempos, valores) como `Vertex.spec`/`Valve.opening` esperam."""

    @abstractmethod
    def to_json(self): ...

    @property
    def breakpoints(self):
        return np.zeros(0)

    @staticmethod
    def from_legacy(spec) -> "Schedule":
        if spec is None:
            raise ValueError("especificação ausente")
        if np.isscalar(spec):
            return Constant(float(spec))
        t, v = spec
        return PiecewiseLinear(np.asarray(t, float).ravel(), np.asarray(v, float).ravel())

    @staticmethod
    def from_json(d) -> "Schedule":
        if isinstance(d, (int, float)):
            return Constant(float(d))
        if "ramp" in d:
            t1, t2 = d["ramp"]
            return Ramp(float(t1), float(t2), float(d.get("from", 0.0)), float(d["to"]))
        return PiecewiseLinear(np.asarray(d["t"], float), np.asarray(d["v"], float))


@dataclass(frozen=True)
class Constant(Schedule):
    value: float

    def __call__(self, t):
        return self.value

    def to_legacy(self):
        return float(self.value)

    def to_json(self):
        return float(self.value)


@dataclass(frozen=True)
class PiecewiseLinear(Schedule):
    """Linear por partes, constante fora do intervalo (semântica de np.interp)."""

    t: np.ndarray
    v: np.ndarray

    def __call__(self, t):
        return np.interp(t, self.t, self.v)

    def to_legacy(self):
        return (np.asarray(self.t, float), np.asarray(self.v, float))

    def to_json(self):
        return {"t": [float(x) for x in self.t], "v": [float(x) for x in self.v]}

    @property
    def breakpoints(self):
        return np.asarray(self.t, float)


@dataclass(frozen=True)
class Ramp(Schedule):
    """Rampa linear de `start` a `end` entre t1 e t2 (equivale à tabela [[t1, t2], [start, end]])."""

    t1: float
    t2: float
    start: float
    end: float

    def __call__(self, t):
        return np.interp(t, [self.t1, self.t2], [self.start, self.end])

    def to_legacy(self):
        return (np.array([self.t1, self.t2]), np.array([self.start, self.end]))

    def to_json(self):
        return {"ramp": [self.t1, self.t2], "from": self.start, "to": self.end}

    @property
    def breakpoints(self):
        return np.array([self.t1, self.t2])
