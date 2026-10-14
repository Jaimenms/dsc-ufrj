"""Relações de fechamento do modelo a dois fluidos por padrão de escoamento."""

from .common import ClosureInputs, ClosureTerms, G
from .stratified import stratified
from .annular import annular
from .bubble import bubble
from .intermittent import intermittent
from .regime import (
    REGIME_ANNULAR,
    REGIME_AUTO,
    REGIME_BUBBLE,
    REGIME_INTERMITTENT,
    REGIME_STRAT_SMOOTH,
    REGIME_STRAT_WAVY,
    RegimeFlags,
    blend_closures,
    equilibrium_fractions,
    fixed_regime,
    identify_regime,
)

__all__ = [
    "ClosureInputs", "ClosureTerms", "G",
    "stratified", "annular", "bubble", "intermittent",
    "RegimeFlags", "blend_closures", "equilibrium_fractions",
    "fixed_regime", "identify_regime",
    "REGIME_ANNULAR", "REGIME_AUTO", "REGIME_BUBBLE", "REGIME_INTERMITTENT",
    "REGIME_STRAT_SMOOTH", "REGIME_STRAT_WAVY",
]
