"""O caso de exemplo (`cases/examples/hello_world.json`, docs/hello_world.md) obedece ao schema,
corre, dá a pressão imposta na saída e pressão maior na entrada, e a construção em Python
descrita no documento é o mesmo modelo."""

from pathlib import Path

import numpy as np
import pytest

from conftest import assert_same_model
from mfsim import BoundaryVertex, Fluid, Network, Pipe, specs
from mfsim.io.spec import case_from_json, load_json
from mfsim.io.validate import case_errors
from mfsim.network.schedules import Ramp
from mfsim.initial_guess import initial_guess
from mfsim.simulation import build_model, simulate

ROOT = Path(__file__).resolve().parents[1]
HELLO = ROOT / "cases" / "examples" / "hello_world.json"


def test_hello_world_runs():
    assert case_errors(load_json(str(HELLO))) == []
    net = case_from_json(str(HELLO))
    net.set_time(tf=50.0)
    res = simulate(net, progress=False)
    P = res.pipe(0)["out"]["P_bar"]
    assert res.info["success"] and abs(P[-1, -1] - 100.0) < 1e-9 and P[0, -1] > 100.0
    assert res.T[-1] == 50.0 and res.pipe(0)["out"]["qL"].shape == (21, 6)


def test_hello_world_python_equals_json():
    fluid = Fluid(rhoL0=860.0, rhoG0=80.0, muL=0.008, muG=2e-5, gammaL=1000.0, gammaG=1.2, P0=100e5, sigma=0.03)
    inlet = BoundaryVertex("inlet", spec=specs(qL=0.5, qG=Ramp(100.0, 130.0, 0.10, 0.13)))
    outlet = BoundaryVertex("outlet", spec=specs(PT=100e5 / 1e7))
    line = Pipe(
        "line",
        "inlet",
        "outlet",
        points=np.array([[0.0, 0, 0], [1000.0, 0, 0]]),
        n_elements=np.array([20]),
        D=0.1,
        roughness=1e-3,
    )
    net = Network.build(
        "hello_world",
        fluid,
        [inlet, outlet],
        [line],
        tf=600.0,
        dt_out=10.0,
        PR=1e7,
        integrator="bdf",
        rtol=1e-3,
        atol=1e-3,
        max_step=1.0,
        initial="guess",
    )
    ref = case_from_json(str(HELLO))
    X0 = initial_guess(build_model(ref))  # sem `ic` nos dutos: comparar no estado estimado
    assert_same_model(ref, net, X=X0)
    assert net.initial == ref.initial and net.integration == ref.integration and net.time == ref.time


@pytest.mark.parametrize("integrator", ["bdf", "implicit"])
def test_final_time_is_always_an_output(integrator):
    """`tf` entra sempre nos instantes de saída, mesmo quando não é múltiplo de `dt_out` (antes,
    `tf < dt_out` deixava o intervalo do SciPy degenerado e rebentava); `tf == t0` devolve o
    estado inicial sem integrar."""
    net = case_from_json(str(HELLO))
    net.set_time(tf=25.0).set_integration(integrator=integrator)
    res = simulate(net, progress=False)
    assert res.info["success"] and np.allclose(res.T, [0.0, 10.0, 20.0, 25.0])
    net.set_time(tf=0.5)
    res = simulate(net, progress=False)
    assert res.info["success"] and np.allclose(res.T, [0.0, 0.5]) and res.X.shape[1] == 2
    net.set_time(tf=0.0)
    res = simulate(net, progress=False)
    assert res.T.tolist() == [0.0] and res.X.shape == (res.model.n_states, 1) and res.info["n_steps"] == 0
