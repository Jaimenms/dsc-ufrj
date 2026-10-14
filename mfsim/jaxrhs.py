"""Compatibilidade: `build_jax_rhs`/`get_jax` vivem em `mfsim.compiled` (fase 2 da
reorganização); este módulo será removido na fase 5."""

from .compiled import CompiledNetwork, build_jax_rhs, get_jax  # noqa: F401

try:
    import jax as _jax  # noqa: F401
    HAVE_JAX = True
except ImportError:                     # pragma: no cover
    HAVE_JAX = False
