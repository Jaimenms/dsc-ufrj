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
