"""Linha de comando `mfsim`: `run` reproduz o golden JAX da referência (150 s) a partir do
caso JSON e grava npz v2 (caso completo em `case_json`), `stats` reproduz a tabela do antigo
`scratch/cycle_stats.py` carácter a carácter, `export-case --severe` gera um caso válido."""

import json
from pathlib import Path

import numpy as np
import pytest

from mfsim.cli import main

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.golden]


@pytest.mark.data
def test_stats_matches_old_cycle_stats(capsys, data_dir):
    names = (ROOT / "tests" / "golden" / "cli" / "cycle_stats_runs.txt").read_text().split()
    main(["stats", *names])
    out = capsys.readouterr().out
    ref = (ROOT / "tests" / "golden" / "cli" / "cycle_stats_reference.txt").read_text()
    assert out == ref


@pytest.mark.data
def test_run_reference_jax_150s_matches_golden(tmp_path, golden, data_dir):
    pytest.importorskip("jax")
    from mfsim.runs import load_run

    g = golden("severe.npz")
    out = tmp_path / "ref150.npz"
    rc = main(
        [
            "run",
            str(ROOT / "cases" / "severe_slug" / "reference.json"),
            "--backend",
            "jax",
            "--tf",
            "150",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    run = load_run(str(out))
    g.assert_equal("reference_jax_150s/T", run.T)
    g.assert_equal("reference_jax_150s/X", run.X)
    g.assert_equal("reference_jax_150s/n_steps", np.array(run.info["n_steps"]))
    # npz v2: o caso completo viaja com a corrida e reconstrói o mesmo modelo
    assert run.case_json is not None and run.case_json.get("schema") == "mfsim-case/1"
    assert run.model.n_states == run.X.shape[0]
    assert run.info["success"] and run.provenance.get("command")
    assert np.isfinite(run.signature(min_gap=50.0)["P_min"])


def test_export_severe_case(tmp_path):
    from mfsim.io.spec import case_from_json
    from mfsim.simulation import build_model

    out = tmp_path / "novo.json"
    main(
        [
            "export-case",
            "--severe",
            "--out",
            str(out),
            "--param",
            "D=0.05",
            "--param",
            "start=blocked",
            "--param",
            "n_down=8",
            "--param",
            "n_riser=4",
        ]
    )
    d = json.loads(out.read_text())
    assert d["schema"] == "mfsim-case/1" and d["network"]["edges"]
    model = build_model(case_from_json(str(out)))
    assert np.all(np.isfinite(model.rhs(0.0, model.initial_state())))
