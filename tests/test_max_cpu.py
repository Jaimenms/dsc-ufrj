"""Orçamento de tempo de relógio da integração (`integration.max_cpu`, issue 045): a corrida pára com a
trajetória parcial e a mensagem do orçamento, no BDF próprio e no do SciPy."""

from pathlib import Path

import numpy as np

from mfsim.io import case_from_json
from mfsim.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]


def test_implicit_bdf_stops_on_budget():
    net = case_from_json(
        str(ROOT / "cases" / "severe_slug" / "reference.json"),
        overrides=['integration.backend="numpy"', "integration.max_cpu=2.0"],
    )
    net.set_time(tf=200.0)
    res = simulate(net, progress=False, partial=True)
    assert res.info["success"] is False
    assert "orçamento de tempo" in res.info["error"]
    assert res.T[-1] < 200.0 and np.all(np.isfinite(res.X))


def test_scipy_bdf_stops_on_budget():
    net = case_from_json(str(ROOT / "cases" / "examples" / "hello_world.json"), overrides=["integration.max_cpu=0.05"])
    res = simulate(net, progress=False, partial=True)
    assert not res.info["success"]
    assert "orçamento de tempo" in res.info.get("error", "")
    assert res.T[-1] < net.time.tf


class _FailingModel:
    """Modelo mínimo cujo lado direito falha a partir de t = 0,5 (como a fatoração LU singular do SciPy)."""

    def jac_sparsity(self):
        return None

    def rhs(self, t, x):
        if t > 0.5:
            raise RuntimeError("Factor is exactly singular")
        return -x


def test_scipy_failure_returns_partial_trajectory_with_time():
    import pytest

    from mfsim.integrators import integrate_scipy

    t_eval = np.linspace(0.0, 1.0, 21)
    T, X, info = integrate_scipy(_FailingModel(), np.ones(2), t_eval, max_cpu=1e9, partial=True, progress=False)
    assert not info["success"]
    assert "Factor is exactly singular em t=" in info["error"]
    assert 0.0 < T[-1] <= 0.5 and X.shape == (2, T.size)
    np.testing.assert_allclose(X[:, -1], np.exp(-T[-1]), rtol=5e-3)
    with pytest.raises(RuntimeError):  # sem `partial` a exceção sobe, como antes
        integrate_scipy(_FailingModel(), np.ones(2), t_eval, max_cpu=1e9, progress=False)
