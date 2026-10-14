"""Discretização espacial das EDPs de cada duto (método das linhas): esquemas nodais da
tese (Galerkin, Petrov-Galerkin, diferenças centrais e regressivas) e volumes finitos
escalonados (dois fluidos, deriva, monofásicos; variante com massas por volume).
`SCHEMES[nome](model, mesh, settings)` constrói o esquema (fv_upwind escolhe a classe
concreta pelo modelo)."""

from .base import PipeScheme
from .nodal import (
    FDCentralScheme,
    FDUpwindScheme,
    GalerkinScheme,
    NodalScheme,
    PetrovGalerkinScheme,
)
from .staggered import (
    StaggeredDriftFluxFV,
    StaggeredFVBase,
    StaggeredFVConsScheme,
    StaggeredFVScheme,
    StaggeredSinglePhaseFV,
    StaggeredTwoFluidConsFV,
    StaggeredTwoFluidFV,
    solve_pressure,
    staggered_scheme,
)

SCHEMES = {
    "galerkin": GalerkinScheme,
    "petrov_galerkin": PetrovGalerkinScheme,
    "fd_central": FDCentralScheme,
    "fd_upwind": FDUpwindScheme,
    "fv_upwind": staggered_scheme,
    "fv_upwind_cons": StaggeredTwoFluidConsFV,
}

__all__ = [
    "SCHEMES",
    "PipeScheme",
    "NodalScheme",
    "GalerkinScheme",
    "PetrovGalerkinScheme",
    "FDCentralScheme",
    "FDUpwindScheme",
    "StaggeredFVBase",
    "StaggeredTwoFluidFV",
    "StaggeredDriftFluxFV",
    "StaggeredSinglePhaseFV",
    "StaggeredTwoFluidConsFV",
    "StaggeredFVScheme",
    "StaggeredFVConsScheme",
    "staggered_scheme",
    "solve_pressure",
]
