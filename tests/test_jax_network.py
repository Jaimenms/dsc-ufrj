"""O lado direito JAX da rede reproduz NetworkModel.rhs em todos os casos dos
notebooks (modelos, esquemas, elementos de rede) e a Jacobiana é finita."""

import copy

import numpy as np
import pytest

from mfsim.cases import chapter10_case
from mfsim.simulation import build_model

jax = pytest.importorskip("jax")

CASES = ["10.1.1", "10.1.2", "10.1.3", "10.1.4", "10.2.2", "10.3.1", "10.3.2b", "10.3.3"]
SCHEMES = ["galerkin", "petrov_galerkin", "fd_central", "fd_upwind"]


@pytest.mark.parametrize("cid", CASES)
def test_jax_network_matches_numpy(cid):
    from mfsim.compiled import build_jax_rhs

    case = chapter10_case(cid)
    case.set_options(bb_derivatives="fd")  # equivalência exata com o numpy (que usa diferenças finitas na tese)
    model = build_model(case)
    rhs_j, jac_j = build_jax_rhs(model)
    X0 = model.initial_state()
    rng = np.random.default_rng(0)
    t = case.time.t0 + 1.0
    for x in (X0, X0 * (1 + 0.02 * rng.standard_normal(X0.size))):
        f_np = model.rhs(t, x)
        f_jx = rhs_j(t, x)
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    assert np.all(np.isfinite(jac_j(t, X0)))


@pytest.mark.parametrize("method", SCHEMES)
def test_jax_schemes_match_numpy(method):
    from mfsim.compiled import build_jax_rhs

    case = chapter10_case("10.1.1")
    c = copy.deepcopy(case)
    c.set_method(method)
    model = build_model(c)
    rhs_j, jac_j = build_jax_rhs(model)
    X0 = model.initial_state()
    f_np = model.rhs(1.0, X0)
    assert np.max(np.abs(f_np - rhs_j(1.0, X0))) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    assert np.all(np.isfinite(jac_j(1.0, X0)))


@pytest.mark.parametrize("cid", ["10.1.4", "10.3.2b"])
def test_beggs_brill_exact_derivatives_close_to_fd(cid):
    """As derivadas internas exatas (jvp) do Beggs & Brill concordam com as diferenças
    finitas de 1e-3 da tese dentro do erro de truncamento destas (~1e-3 relativo)."""
    from mfsim.compiled import build_jax_rhs

    out = {}
    x = None
    for mode in ("fd", "exact"):
        case = chapter10_case(cid)
        case.set_options(bb_derivatives=mode)
        model = build_model(case)
        rhs_j, _ = build_jax_rhs(model)
        if x is None:
            rng = np.random.default_rng(3)
            X0 = model.initial_state()
            x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))  # o MESMO estado perturbado nos dois modos
        out[mode] = rhs_j(case.time.t0 + 1.0, x)
    scale = np.max(np.abs(out["fd"]))
    assert scale > 0
    assert np.max(np.abs(out["fd"] - out["exact"])) < 2e-2 * scale
