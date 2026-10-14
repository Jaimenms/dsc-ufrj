"""O schema `mfsim-case/1` amarra os casos JSON: o próprio schema é válido, todos os ficheiros de
`cases/` obedecem-lhe, o que o exportador escreve obedece-lhe, e erros de vocabulário ou de
tipo são apanhados com uma mensagem que aponta o caminho."""

import copy
import json
from pathlib import Path

import pytest

from mfsim.io.spec import case_from_json, case_to_json, load_json
from mfsim.io.validate import case_errors, schema, validate_case

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "cases").glob("**/*.json"))


def test_schema_itself_is_valid():
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(schema())
    assert schema()["title"] == "mfsim-case/1"


@pytest.mark.parametrize("path", CASES, ids=[p.relative_to(ROOT).as_posix() for p in CASES])
def test_case_files_obey_schema(path):
    assert case_errors(load_json(str(path))) == []


def test_generated_case_obeys_schema(tmp_path):
    from generators.severe_slug import generate

    net = generate(D=0.05, n_down=8, n_riser=4, start="blocked")
    d = case_to_json(net, str(tmp_path / "g.json"))  # valida antes de escrever
    assert case_errors(d) == []
    case_from_json(str(tmp_path / "g.json"))  # e ao carregar


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda d: d["network"]["edges"][0]["model"].__setitem__("type", "three_fluid"), "network/edges/0/model"),
        (lambda d: d["network"]["edges"][0]["scheme"]["options"].__setitem__("relax_tim", 0.05), "scheme/options"),
        (lambda d: d["integration"].__setitem__("backend", "torch"), "integration/backend"),
        (lambda d: d["network"]["edges"][0]["ports"]["in"].__setitem__("PT", "fixed"), "ports/in/PT"),
        (lambda d: d["time"].__setitem__("dt_out", 0.0), "time/dt_out"),
        (lambda d: d.__setitem__("solver", {}), "(raiz)"),
    ],
)
def test_invalid_cases_are_rejected(mutate, expect):
    d = load_json(str(ROOT / "cases" / "severe_slug" / "reference.json"))
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    bad = copy.deepcopy(d)
    mutate(bad)
    errs = case_errors(bad)
    assert errs and any(expect in e for e in errs), errs
    with pytest.raises(ValueError):
        validate_case(bad)
    assert case_errors(d) == []
