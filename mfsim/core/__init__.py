"""Núcleo: layout dos estados, tipos de porta, opções (dataclasses congeladas) e registo de
classes. Sem dependências do resto do pacote."""

from .slots import SLOT_ALPHA, SLOT_P, SLOT_QL, SLOT_QG, SLOT_NAMES, PortKind, G
from .options import (
    ModelSpec,
    RegimeSpec,
    SchemeSpec,
    IntegrationSpec,
    TimeSpec,
    ReferenceSpec,
    REGIME_NAMES,
    REGIME_IDS,
    MODEL_NAMES,
    MODEL_IDS,
)

__all__ = [
    "SLOT_ALPHA",
    "SLOT_P",
    "SLOT_QL",
    "SLOT_QG",
    "SLOT_NAMES",
    "PortKind",
    "G",
    "ModelSpec",
    "RegimeSpec",
    "SchemeSpec",
    "IntegrationSpec",
    "TimeSpec",
    "ReferenceSpec",
    "REGIME_NAMES",
    "REGIME_IDS",
    "MODEL_NAMES",
    "MODEL_IDS",
]
