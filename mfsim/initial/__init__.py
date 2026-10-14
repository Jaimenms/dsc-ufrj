"""Condições iniciais: receitas (partida bloqueada, linha fechada) materializadas na malha do
duto. Os arrays explícitos continuam a ser aceites (`Pipe.ic`)."""

from .recipes import RECIPES, blocked_ic, materialize, shut_in_ic

__all__ = ["RECIPES", "blocked_ic", "shut_in_ic", "materialize"]
