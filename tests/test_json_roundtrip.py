"""Fase 1: o grafo e o JSON representam os casos sem perda. Case legado -> grafo -> Case e
Case -> JSON (+ npz de condições iniciais) -> Case dão o mesmo modelo: mesmo número de
estados, mesmo estado inicial e rhs bit-idêntico (nos 11 casos do capítulo 10 e nas
corridas de intermitência severa gravadas)."""
import json

import numpy as np
import pytest

from mfsim.cases import CHAPTER10, load_chapter10_case
from mfsim.io import case_from_json, case_to_json
from mfsim.network.graph import Network


from conftest import assert_same_model as _assert_same


@pytest.mark.parametrize("key", list(CHAPTER10))
def test_chapter10_graph_and_json_roundtrip(key, tmp_path):
    case, raw = load_chapter10_case(key)
    net = Network.from_legacy_case(case)
    _assert_same(case, net.to_legacy_case())
    path = tmp_path / f"{key}.json"
    case_to_json(case, str(path))
    d = json.load(open(path))
    assert d["schema"] == "mfsim-case/1" and len(d["network"]["edges"]) >= len(case.pipes)
    _assert_same(case, case_from_json(str(path)))


@pytest.mark.data
@pytest.mark.parametrize("name", ["severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys", "severe_slug_blocked_regime6_local",
                                  "severe_slug_blocked_network_v005", "severe_slug_lab_fabre2_fisico", "severe_slug_blocked_conservative_arc_boe_vup20"])
def test_severe_slug_runs_roundtrip(name, tmp_path, data_dir):
    from mfsim.runs import load_run
    path = data_dir / f"{name}.npz"
    if not path.exists():
        pytest.skip(name)
    run = load_run(str(path)); case = run.case
    net = Network.from_legacy_case(case)
    _assert_same(case, net.to_legacy_case(), X=run.X[:, 0])
    jpath = tmp_path / f"{name}.json"
    case_to_json(case, str(jpath))
    _assert_same(case, case_from_json(str(jpath)), X=run.X[:, 0])


def test_json_is_self_contained(tmp_path):
    """O JSON gravado não depende do Python que o gerou: valores explícitos de modelo, esquema,
    portas e integração para cada duto."""
    case, raw = load_chapter10_case("10.3.2b")
    d = case_to_json(case, str(tmp_path / "c.json"))
    pipe = next(e for e in d["network"]["edges"] if e["type"] == "pipe")
    assert set(pipe) >= {"id", "from", "to", "D", "roughness", "geometry", "model", "scheme", "ports", "ic"}
    assert {e["type"] for e in d["network"]["edges"]} == {"pipe", "choke", "reservoir", "check_valve", "connector"}
    assert {v["kind"] for v in d["network"]["vertices"]} == {"boundary", "tank", "junction"}


@pytest.mark.parametrize("key", list(CHAPTER10))
def test_chapter10_json_files_match_mf2(key):
    """Os JSON exportados em cases/chapter10 constroem o mesmo modelo que os .mf2."""
    from mfsim.cases.chapter10 import load_chapter10_json
    case, raw = load_chapter10_case(key)
    case_j, raw_j = load_chapter10_json(key)
    _assert_same(case, case_j)


@pytest.mark.data
def test_severe_json_cases_match_runs(data_dir):
    """Os JSON de cases/severe_slug reconstroem os modelos das corridas gravadas que os definem."""
    import os
    from mfsim.cases.severe_slug import CASES_DIR, REFERENCE_RUNS
    from mfsim.runs import load_run
    n = 0
    for run_name, name in REFERENCE_RUNS.items():
        path = os.path.join(CASES_DIR, f"{name}.json")
        if not os.path.exists(path) or not (data_dir / f"{run_name}.npz").exists():
            continue
        run = load_run(str(data_dir / f"{run_name}.npz"))
        case_j = case_from_json(path)
        _assert_same(run.case, case_j, X=run.X[:, 0]); n += 1
        assert case_j.settings.integrator == "implicit" and "tol_range" in case_j.settings.extra
    assert n >= 5
