"""Herança de casos (`"$extends"`), remendo mínimo (`make_patch`/`merge_case`) e overrides por
caminho (`--set`): as variantes de `cases/severe_slug` reconstroem o caso completo, a fusão é a
inversa exata do remendo em todos os pares de casos, `null` é valor e não "sem diferença", os
arrays de condição inicial da base continuam a ser encontrados a partir de outra pasta, e o
exportador diferencial escreve só o que muda."""

import itertools
import json
import shutil
from pathlib import Path

import pytest

from conftest import assert_same_model
from mfsim.io.extends import apply_overrides, make_patch, merge_case, parse_override
from mfsim.io.spec import case_from_json, case_to_json, load_json

ROOT = Path(__file__).resolve().parents[1]
SEVERE = ROOT / "cases" / "severe_slug"


def _expanded(path):
    d = {k: v for k, v in load_json(str(path)).items() if not k.startswith("_")}
    (d.get("meta") or {}).pop("extends", None)
    return d


def test_patch_merge_roundtrip_all_pairs():
    docs = {p.stem: _expanded(p) for p in sorted(SEVERE.glob("*.json"))}
    for a, b in itertools.permutations(docs, 2):
        assert merge_case(docs[a], make_patch(docs[a], docs[b])) == docs[b], (a, b)
    assert make_patch(docs["reference"], docs["reference"]) == {}
    assert merge_case(docs["reference"], {}) == docs["reference"]


def test_null_is_a_value():
    assert make_patch({"a": 1.0}, {"a": None}) == {"a": None}
    assert merge_case({"a": 1.0, "b": {"c": 1}}, {"a": None, "b": {"d": 2}}) == {"a": None, "b": {"c": 1, "d": 2}}


def test_replace_and_id_lists():
    base = {"m": {"x": 1, "y": 2}, "l": [{"id": "a", "v": 1}, {"id": "b", "v": 2}]}
    new = {"m": {"x": 1}, "l": [{"id": "a", "v": 1}, {"id": "b", "v": 3}, {"id": "c", "v": 4}]}
    p = make_patch(base, new)
    assert p == {"m": {"x": 1, "$replace": True}, "l": [{"id": "b", "v": 3}, {"id": "c", "v": 4}]}
    assert merge_case(base, p) == new
    new2 = {"m": base["m"], "l": [{"id": "b", "v": 2}]}  # id removido: a lista vai inteira
    assert make_patch(base, new2) == {"l": {"$replace": new2["l"]}}
    assert merge_case(base, make_patch(base, new2)) == new2


def test_variant_files_expand_to_full_cases():
    n = 0
    for p in sorted(SEVERE.glob("*.json")):
        raw = json.load(open(p))
        if "$extends" not in raw:
            continue
        d = load_json(str(p))
        assert d["meta"]["extends"][0] == raw["$extends"] and d["name"] == p.stem
        base = _expanded(SEVERE / raw["$extends"])
        patch = {k: v for k, v in raw.items() if k not in ("$extends", "schema")}
        assert merge_case(base, patch) == _expanded(p)
        assert len(json.dumps(raw)) < 0.6 * len(json.dumps(d)), p.name
        n += 1
    assert n >= 6


def test_extends_from_other_directory_with_ic_arrays(tmp_path):
    base = ROOT / "cases" / "chapter10" / "10_1_1.json"
    v = tmp_path / "v.json"
    v.write_text(json.dumps({"schema": "mfsim-case/1", "$extends": str(base), "name": "v", "time": {"tf": 3.0}}))
    net = case_from_json(str(v))
    ref = case_from_json(str(base))
    assert net.time.tf == 3.0 and ref.time.tf != 3.0 and net.name == "v" and net.meta["extends"] == [str(base)]
    ref.set_time(tf=3.0)
    assert_same_model(ref, net)


def test_extends_chain_and_cycle(tmp_path):
    a, b, c = (tmp_path / f"{k}.json" for k in "abc")
    a.write_text(
        json.dumps(
            {
                "schema": "mfsim-case/1",
                "$extends": str(SEVERE / "reference.json"),
                "name": "a",
                "integration": {"rtol": 1e-5},
            }
        )
    )
    b.write_text(
        json.dumps(
            {"schema": "mfsim-case/1", "$extends": "a.json", "name": "b", "integration": {"integrator": "radau"}}
        )
    )
    net = case_from_json(str(b))
    assert net.integration.integrator == "radau" and net.integration.rtol == 1e-5
    assert net.meta["extends"] == ["a.json", str(SEVERE / "reference.json")]
    c.write_text(json.dumps({"schema": "mfsim-case/1", "$extends": "c.json", "name": "c"}))
    with pytest.raises(ValueError, match="ciclo"):
        load_json(str(c))


def test_overrides_by_path():
    ref = str(SEVERE / "reference.json")
    net = case_from_json(
        ref,
        overrides=[
            "integration.integrator=radau",
            "network.edges.1.D=0.06",
            "network.vertices.0.spec.qL=0.4",
            "time.tf=10",
        ],
    )
    assert net.integration.integrator == "radau" and net.pipes[0].D == 0.06 and net.time.tf == 10.0
    assert parse_override('a.b="x"') == ("a.b", "x") and parse_override("a=texto") == ("a", "texto")
    assert apply_overrides({"l": [{"id": "1"}, {"id": "2"}]}, {"l.1.v": 3}) == {"l": [{"id": "1", "v": 3}, {"id": "2"}]}
    with pytest.raises(ValueError):  # o schema apanha o valor inválido
        case_from_json(ref, overrides=["integration.backend=torch"])
    with pytest.raises(KeyError):
        case_from_json(ref, overrides=["network.edges.zz.D=1"])


def test_differential_export_roundtrip(tmp_path):
    shutil.copy(SEVERE / "reference.json", tmp_path / "reference.json")
    net = case_from_json(str(SEVERE / "reference.json"))
    net.name = "ref_radau"
    net.set_integration(integrator="radau", rtol=1e-5)
    case_to_json(net, str(tmp_path / "ref_radau.json"), extends="reference.json")
    raw = json.load(open(tmp_path / "ref_radau.json"))
    assert raw["$extends"] == "reference.json" and set(raw) == {"schema", "$extends", "name", "integration"}
    net2 = case_from_json(str(tmp_path / "ref_radau.json"))
    assert net2.integration.integrator == "radau" and net2.integration.rtol == 1e-5 and net2.name == "ref_radau"
    assert_same_model(net, net2)


def test_cli_set(tmp_path):
    from mfsim.cli import main
    from mfsim.runs import load_run

    out = tmp_path / "r.npz"
    rc = main(
        [
            "run",
            str(ROOT / "cases" / "chapter10" / "10_1_1.json"),
            "--tf",
            "20",
            "--set",
            "integration.rtol=1e-3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    run = load_run(str(out))
    assert run.case.integration.rtol == 1e-3 and run.T[-1] == 20.0 and run.case.time.tf == 20.0
