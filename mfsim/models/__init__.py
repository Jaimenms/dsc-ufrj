"""Modelos de escoamento em dutos (coeficientes das EDPs)."""

from .base import PipeModel, SLOT_ALPHA, SLOT_P, SLOT_QL, SLOT_QG
from .twofluid import TwoFluidModel
from .singlephase import SinglePhaseLiquid, SinglePhaseGas
from .driftflux_bb import BeggsBrillModel

MODEL_BY_LEGACY_ID = {
    1: TwoFluidModel,
    2: BeggsBrillModel,
    3: SinglePhaseLiquid,
    4: SinglePhaseGas,
}

__all__ = [
    "PipeModel",
    "TwoFluidModel",
    "SinglePhaseLiquid",
    "SinglePhaseGas",
    "BeggsBrillModel",
    "MODEL_BY_LEGACY_ID",
    "SLOT_ALPHA",
    "SLOT_P",
    "SLOT_QL",
    "SLOT_QG",
]
