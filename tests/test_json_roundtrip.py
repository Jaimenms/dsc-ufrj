"""O grafo e o JSON representam os casos sem perda: Network -> JSON (+ npz de condições
iniciais) -> Network, e o registo legado (struct `dados`) <-> grafo, dão o mesmo modelo:
mesmo número de estados, mesmo estado inicial e rhs bit-idêntico (nos 11 casos do capítulo
10 e nas corridas de intermitência severa gravadas)."""

import json

import numpy as np
import pytest

from thesis.record import network_from_record, record_from_network
from thesis.mf2 import chapter10_keys, chapter10_case
from mfsim.io import case_from_json, case_to_json
from mfsim.network.graph import Network


from conftest import assert_same_model as _assert_same


@pytest.mark.parametrize("key", chapter10_keys())
def test_chapter10_legacy_and_json_roundtrip(key, tmp_path):
    net = chapter10_case(key)
    _assert_same(net, network_from_record(record_from_network(net)))
    path = tmp_path / f"{key}.json"
    case_to_json(net, str(path))
    d = json.load(open(path))
    assert d["schema"] == "mfsim-case/1" and len(d["network"]["edges"]) >= len(net.pipes)
    _assert_same(net, case_from_json(str(path)))


@pytest.mark.data
@pytest.mark.parametrize(
    "name",
    [
        "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys",
        "severe_slug_blocked_regime6_local",
        "severe_slug_blocked_network_v005",
        "severe_slug_lab_fabre2_fisico",
        "severe_slug_blocked_conservative_arc_boe_vup20",
    ],
)
def test_severe_slug_runs_roundtrip(name, tmp_path, data_dir):
    from mfsim.runs import load_run

    path = data_dir / f"{name}.npz"
    if not path.exists():
        pytest.skip(name)
    run = load_run(str(path))
    net = run.case
    _assert_same(net, network_from_record(record_from_network(net)), X=run.X[:, 0])
    jpath = tmp_path / f"{name}.json"
    case_to_json(net, str(jpath))
    _assert_same(net, case_from_json(str(jpath)), X=run.X[:, 0])


def test_json_is_self_contained(tmp_path):
    """O JSON gravado não depende do Python que o gerou: valores explícitos de modelo, esquema,
    portas e integração para cada duto; as junções levam as especificações herdadas."""
    net = chapter10_case("10.3.2b")
    d = case_to_json(net, str(tmp_path / "c.json"))
    pipe = next(e for e in d["network"]["edges"] if e["type"] == "pipe")
    assert set(pipe) >= {"id", "from", "to", "D", "roughness", "geometry", "model", "scheme", "ports", "ic"}
    assert {e["type"] for e in d["network"]["edges"]} == {"pipe", "choke", "reservoir", "check_valve", "connector"}
    assert {v["kind"] for v in d["network"]["vertices"]} == {"boundary", "tank", "junction"}
    assert any(v["kind"] == "junction" and v.get("spec") for v in d["network"]["vertices"])


@pytest.mark.parametrize("key", chapter10_keys())
def test_chapter10_json_files_match_mf2(key):
    """Os JSON exportados em cases/chapter10 (a fonte dos casos) constroem o mesmo modelo que
    os .mf2 originais lidos com as correções do exportador."""
    from thesis.export import network_from_mf2

    net_m, raw = network_from_mf2(key)
    _assert_same(net_m, chapter10_case(key))


def test_network_editing_helpers():
    """`set_method`/`set_options`/`set_time` reconstroem as especificações sem alterar o resto
    (a identidade de `from_legacy(to_legacy())` é o que permite editar no dialeto legado)."""
    net = chapter10_case("10.1.1")
    before = net.pipes[0].model
    net2 = network_from_record(record_from_network(net)).set_options(k_switch=net.pipes[0].model.regime.k_switch)
    assert net2.pipes[0].model == before and net2.pipes[0].scheme == net.pipes[0].scheme
    net2.set_method("fd_central").set_options(conservative=True, bb_derivatives="fd").set_time(tf=10.0)
    assert net2.pipes[0].scheme.type == "fd_central" and dict(net2.pipes[0].scheme.options)["conservative"] is True
    assert net2.pipes[0].model.derivatives == "fd" and net2.time.tf == 10.0 and net.time.tf != 10.0


@pytest.mark.data
def test_severe_json_cases_match_runs(data_dir):
    """Os JSON de cases/severe_slug reconstroem os modelos das corridas gravadas que os definem
    (a corrida de origem está em `meta.run` de cada ficheiro)."""
    from pathlib import Path

    from mfsim.runs import load_run

    n = 0
    for path in sorted((Path(__file__).resolve().parents[1] / "cases" / "severe_slug").glob("*.json")):
        net_j = case_from_json(str(path))
        run_name = (net_j.meta or {}).get("run")
        if not run_name or not (data_dir / f"{run_name}.npz").exists():
            continue
        run = load_run(str(data_dir / f"{run_name}.npz"))
        _assert_same(run.case, net_j, X=run.X[:, 0])
        n += 1
        assert net_j.integration.integrator == "implicit" and net_j.integration.tol_range is not None
    assert n >= 5
