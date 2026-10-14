"""Cada corrida gravada em notebooks/data (formato v2, `case_json`) reconstrói o modelo
exatamente como o notebook o fazia a partir do `meta` v1 (tamanho do estado, rhs e saídas no
primeiro instante, fluxos de contorno) — os golden foram gerados com uma cópia literal de
`load_blocked_run` e a conversão v1 -> v2 tem de preservá-los."""

import numpy as np
import pytest

from mfsim.runs import load_run

pytestmark = [pytest.mark.golden, pytest.mark.data]


def _names():
    from conftest import GOLDEN

    p = GOLDEN / "runs_meta.npz"
    if not p.exists():
        return []
    with np.load(p) as g:
        return [str(n) for n in g["names"]]


@pytest.mark.parametrize("name", _names())
def test_rebuild_from_meta(name, golden, data_dir):
    g = golden("runs_meta.npz")
    path = data_dir / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{name} ausente")
    run = load_run(str(path))
    model = run.model
    g.assert_equal(f"{name}/n_states", np.array(model.n_states))
    g.assert_equal(f"{name}/rhs0", model.rhs(run.T[0], run.X[:, 0]))
    from mfsim.results import Results

    r = Results(model, run.T[:1], run.X[:, :1])
    p = r.pipe(0)
    g.assert_equal(f"{name}/P_bar0", np.asarray(p["out"]["P_bar"][:, 0], float))
    g.assert_equal(f"{name}/alphaG0", np.asarray(p["out"]["alphaG"][:, 0], float))
    if g.has(f"{name}/FL0"):
        FL, FG = model.pipes[0].scheme.mass_fluxes_nodal(p["Y"][0])
        g.assert_equal(f"{name}/FL0", np.asarray(FL, float))
        g.assert_equal(f"{name}/FG0", np.asarray(FG, float))
