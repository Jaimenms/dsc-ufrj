"""Partida a partir do estado estacionário (issue 040): `initial_state.source = "steady"` faz o
Newton a partir de `from` antes de integrar, regista o resultado na informação da corrida,
aplica a perturbação e, sem convergência, falha ou segue do estado de partida."""

import json
from pathlib import Path

import numpy as np
import pytest

from mfsim.io import case_from_json, case_to_json
from mfsim.simulation import build_model, simulate

ROOT = Path(__file__).resolve().parents[1]


def _thesis_steady(**opts):
    from thesis.mf2 import chapter10_case

    net = chapter10_case("10.1.1")  # condição inicial gravada = estado final (estacionário)
    net.initial = "steady"
    net.initial_options = dict(opts)
    net.set_time(tf=net.time.t0)
    return net


def test_json_roundtrip_of_steady_options(tmp_path):
    net = _thesis_steady()
    net.initial_options = {"from": "edges", "perturb": 1e-3, "on_failure": "start"}
    d = case_to_json(net, str(tmp_path / "c.json"))
    assert d["initial_state"] == {"source": "steady", "from": "edges", "perturb": 1e-3, "on_failure": "start"}
    back = case_from_json(str(tmp_path / "c.json"))
    assert back.initial == "steady" and back.initial_options == {
        "from": "edges",
        "perturb": 1e-3,
        "on_failure": "start",
    }


def test_steady_start_converges_from_recorded_state():
    net = _thesis_steady()
    res = simulate(net, progress=False)
    info = res.info["initial"]
    assert info["source"] == "steady" and info["converged"] and info["residual"] < 1e-9
    model = build_model(net)
    assert np.linalg.norm(model.rhs(model.t0, res.X[:, 0])) < 1e-9


def test_perturbation_scales_pressures():
    base = simulate(_thesis_steady(), progress=False).X[:, 0]
    pert = simulate(_thesis_steady(perturb=1e-3), progress=False).X[:, 0]
    slots = build_model(_thesis_steady()).state_slots()
    assert np.allclose(pert[slots == 1], base[slots == 1] * 1.001)
    assert np.array_equal(pert[slots != 1], base[slots != 1])


def _blocked_reference(**opts):
    d = json.load(open(ROOT / "cases" / "severe_slug" / "reference.json"))
    d["initial_state"] = {"source": "steady", "from": "edges", "max_iter": 3, **opts}
    d["time"]["tf"] = 0.0
    return case_from_json(d)


def test_failure_raises_or_falls_back_to_start():
    """A partida bloqueada está longe de qualquer estacionário: 3 iterações não chegam."""
    with pytest.raises(RuntimeError):
        simulate(_blocked_reference(), progress=False)
    net = _blocked_reference(on_failure="start")
    res = simulate(net, progress=False)
    assert res.info["initial"]["converged"] is False
    assert np.array_equal(res.X[:, 0], build_model(net).initial_state())


def test_exact_state_roundtrip_through_json(tmp_path):
    """Estado do esquema escalonado -> condição inicial nos nós do esquema -> JSON -> estado:
    a identidade (nos nós da malha a ida e volta suaviza o degrau de fração de gás na curva)."""
    d = json.load(open(ROOT / "cases" / "severe_slug" / "reference.json"))
    net = case_from_json(d)
    model = build_model(net)
    X = model.initial_state()
    X = X * (1.0 + 1e-3 * np.sin(np.arange(X.size)))  # um estado qualquer, não o da receita
    model.state_to_case_ic(X, model.t0, exact=True)
    case_to_json(net, str(tmp_path / "c.json"))
    back = case_from_json(str(tmp_path / "c.json"))
    assert back.pipes[0].ic_at == "scheme" and back.pipes[0].ic_recipe is None
    assert np.array_equal(build_model(back).initial_state(), X)
    model.state_to_case_ic(X, model.t0)  # nos nós da malha: aproximado
    assert not np.array_equal(build_model(net).initial_state(), X)
