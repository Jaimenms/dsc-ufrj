"""Discretização espacial das EDPs de cada duto (método das linhas)."""

from .nodal import (
    FDCentralScheme,
    FDUpwindScheme,
    GalerkinScheme,
    NodalScheme,
    PetrovGalerkinScheme,
)
from .staggered import StaggeredFVConsScheme, StaggeredFVScheme

SCHEMES = {
    "galerkin": GalerkinScheme,
    "petrov_galerkin": PetrovGalerkinScheme,
    "fd_central": FDCentralScheme,
    "fd_upwind": FDUpwindScheme,
    "fv_upwind": StaggeredFVScheme,
    "fv_upwind_cons": StaggeredFVConsScheme,
}

__all__ = ["SCHEMES", "NodalScheme", "GalerkinScheme", "PetrovGalerkinScheme",
           "FDCentralScheme", "FDUpwindScheme", "StaggeredFVScheme", "StaggeredFVConsScheme"]
