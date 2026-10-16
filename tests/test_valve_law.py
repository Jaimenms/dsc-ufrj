"""Lei de válvula homogénea (issue 039): coincide com a da tese em líquido puro, não tem
singularidade quando a vazão total passa por zero, resiste ao gás como a densidade sem
escorregamento manda, vai e volta pelo JSON e pelo registo legado, as corridas antigas
continuam com a lei da tese e o lado direito em JAX é igual ao do numpy."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mfsim.fluid import Fluid
from mfsim.io import case_from_json
from mfsim.network.elements import valve_upstream_pressure

ROOT = Path(__file__).resolve().parents[1]
FLUID = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5)
PR = 1e6
P_DOWN = 4e5 / PR


def _dp(qL, qG, law, ab=0.1, CV=15.0):
    return float(valve_upstream_pressure(P_DOWN, qL, qG, ab, CV, FLUID, PR, law=law)[1])


def test_liquid_only_matches_thesis():
    for q in (0.05, 0.53, 2.0):
        assert _dp(q, 0.0, "homogeneous") == pytest.approx(_dp(q, 0.0, "thesis"), rel=1e-12)


def test_reverse_gas_flow_is_well_behaved():
    """Refluxo de gás com pouco líquido (o fim do blowdown que fazia o integrador falhar):
    na lei da tese a densidade troca de sinal com qL + qG e a queda fica com o sinal errado e
    não monótona; a homogénea é finita, tem sempre o sinal de qM e é monótona em escoamento
    co-corrente (em contracorrente, gás a entrar e líquido a sair, a mistura homogénea não tem
    sentido físico e só se exige o sinal)."""
    qL = 1e-3
    qG = np.linspace(-3e-3, 2e-3, 501)
    qM = qL + qG
    dp_h = np.array([_dp(qL, g, "homogeneous") for g in qG])
    dp_t = np.array([_dp(qL, g, "thesis") for g in qG])
    assert np.all(np.isfinite(dp_h))
    assert np.all(dp_h * qM >= 0)
    co = qG >= 0
    assert np.all(np.diff(dp_h[co]) > 0)
    assert np.any((qM < -1e-5) & (dp_t > 0))  # tese: queda positiva com a vazão a recuar
    assert np.any(np.diff(dp_t) < 0)  # tese: não monótona


def test_gas_raises_the_pressure_drop():
    """Com 1 % de gás em massa a queda homogénea é ~3x a da tese (densidade 300 contra 890)."""
    ratio = _dp(0.5, 0.005, "homogeneous") / _dp(0.5, 0.005, "thesis")
    assert 2.5 < ratio < 3.5


def test_unknown_law_raises():
    with pytest.raises(ValueError):
        _dp(0.5, 0.0, "linear")


def _choked_reference(law=None):
    d = json.load(open(ROOT / "cases" / "severe_slug" / "reference.json"))
    net = d["network"]
    net["edges"][0]["to"] = "Jc2"
    net["vertices"].append({"id": "Jc2", "kind": "junction", "spec": {"PT": 0.4}})
    choke = {"id": "choke_2", "type": "choke", "from": "Jc2", "to": "2", "CV": 15.0, "opening": 0.2}
    if law is not None:
        choke["law"] = law
    net["edges"].append(choke)
    return d


def test_json_default_is_homogeneous_and_roundtrips(tmp_path):
    from mfsim.io import case_to_json
    from mfsim.network.graph import Choke

    net = case_from_json(_choked_reference())
    (ch,) = net.edges_of(Choke)
    assert ch.law == "homogeneous"
    d = case_to_json(net, str(tmp_path / "c.json"))
    assert next(e for e in d["network"]["edges"] if e["type"] == "choke")["law"] == "homogeneous"
    assert case_from_json(str(tmp_path / "c.json")).edges_of(Choke)[0].law == "homogeneous"
    with pytest.raises(Exception):
        case_from_json(_choked_reference(law="linear"))


def test_thesis_cases_keep_thesis_law():
    from thesis.mf2 import chapter10_case
    from thesis.record import network_from_record, record_from_network
    from mfsim.network.graph import Choke

    for key in ("10.3.1", "10.3.2b", "10.3.3"):
        net = chapter10_case(key)
        assert {e.law for e in net.edges_of(Choke)} == {"thesis"}
        assert {e.law for e in network_from_record(record_from_network(net)).edges_of(Choke)} == {"thesis"}


def test_stored_runs_without_law_use_thesis_law():
    from mfsim.runs import _legacy_valve_law

    d = _choked_reference()
    assert "law" not in d["network"]["edges"][-1]
    assert _legacy_valve_law(copy.deepcopy(d))["network"]["edges"][-1]["law"] == "thesis"
    d2 = _legacy_valve_law(_choked_reference(law="homogeneous"))
    assert d2["network"]["edges"][-1]["law"] == "homogeneous"


def test_homogeneous_choke_jax_matches_numpy():
    from mfsim.compiled import get_jax
    from mfsim.simulation import build_model

    model = build_model(case_from_json(_choked_reference()))
    X = model.initial_state()
    f_np = model.rhs(0.0, X)
    rhs, _ = get_jax(model)
    f_jx = np.asarray(rhs(0.0, X))
    assert np.allclose(f_jx, f_np, rtol=1e-10, atol=1e-12)
