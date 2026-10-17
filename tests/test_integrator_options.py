"""Opções do BDF próprio da issue 053 (busca em linha de Armijo, política da Jacobiana, iterado inicial do Newton):
cada uma integra o caso de exemplo até ao mesmo estado final que a omissão (dentro da tolerância), a omissão fica
intacta e as chaves fazem o caminho JSON e o schema."""

import numpy as np
import pytest

from mfsim.core.options import IntegrationSpec
from mfsim.io import case_from_json
from mfsim.simulation import simulate

CASE = "cases/examples/hello_world.json"


def _run(**overrides):
    net = case_from_json(CASE)
    net.set_time(tf=60.0, dt_out=10.0)
    res = simulate(net, progress=False, integrator="implicit", backend="numpy", **overrides)
    return np.asarray(res.X[:, -1]), res.info


@pytest.fixture(scope="module")
def reference():
    return _run()


@pytest.mark.parametrize(
    "overrides",
    [
        {"newton_damping": "armijo"},
        {"newton_damping": "armijo_tol"},  # issue 055
        {"jac_policy": "step"},
        {"jac_policy": "rate"},
        {"predictor": "none"},
        {"predictor": "euler"},
        {"newton_damping": "armijo", "jac_policy": "rate", "predictor": "none"},
    ],
)
def test_options_reach_the_same_state(reference, overrides):
    X_ref, info_ref = reference
    X, info = _run(**overrides)
    assert info["success"] and info["n_steps"] > 0
    np.testing.assert_allclose(X, X_ref, rtol=2e-3, atol=1e-6)


def test_legacy_default_is_the_default():
    spec = IntegrationSpec()
    assert spec.newton_damping is None and spec.jac_policy is None and spec.predictor is None
    assert "newton_damping" not in spec.to_json()
    d = IntegrationSpec.from_json(
        {"integrator": "implicit", "newton_damping": "armijo", "jac_policy": "rate", "predictor": "none"}
    )
    assert (d.newton_damping, d.jac_policy, d.predictor) == ("armijo", "rate", "none")
    assert IntegrationSpec.from_json(d.to_json()) == d


def test_armijo_tol_accepts_a_converged_correction():
    """Issue 055: com o iterado inicial já convergido (correção abaixo de newton_tol) o Newton termina na 1.ª
    iteração, sem passar pela busca em linha (cujo resíduo pode não descer nas quinas dos fechamentos)."""
    from mfsim.integrators.implicit import ImplicitBDF

    def f(t, x):
        return -x

    bdf = ImplicitBDF(f, rtol=1e-3, atol=1e-6)
    bdf.newton_damping = "armijo_tol"
    x0 = np.array([1.0])
    bdf._jacobian(0.0, x0, f(0.0, x0))
    h = 1e-3
    x, ok, nit = bdf._newton(h, h, x0 / (1 + h), x0.copy(), 1.0)  # BDF1: o preditor é a solução exata
    assert ok and nit == 1
    np.testing.assert_allclose(x, x0 / (1 + h), rtol=1e-12)
