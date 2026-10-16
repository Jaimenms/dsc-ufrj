"""Peças do mapa de intermitência severa (issue 043): gerador com choke e IPR, lote em paralelo
igual ao sequencial e o caso base do poço."""

import json

import numpy as np

from generators.severe_slug import generate
from mfsim.io import case_from_json
from mfsim.network.graph import Choke, Reservoir
from mfsim.simulation import build_model

GEN = dict(
    D=0.05,
    L_h1=20.0,
    L_down=300.0,
    slope_down_deg=-3.0,
    H_riser=20.0,
    L_h2=5.0,
    n_h1=2,
    n_down=10,
    n_riser=4,
    n_h2=2,
    qL=0.5,
    qG=0.001,
    P_out_bar=4.0,
    regime=4,
    tf=10.0,
    dt_out=5.0,
    start="blocked",
)


def test_generator_choke_and_reservoir():
    net = generate(**GEN, choke_opening=0.2, reservoir={"P_bar": 15.0, "IPL": 0.056, "IPG": 1.2e-4})
    (ch,) = net.edges_of(Choke)
    (res,) = net.edges_of(Reservoir)
    assert ch.law == "homogeneous" and ch.v_to == "2" and net.pipes[0].v_to == "Jc2"
    assert res.P == 15.0 and net.pipes[0].v_from == "Jr1"
    model = build_model(net)
    X = model.initial_state()
    assert np.all(np.isfinite(model.rhs(0.0, X)))
    plain = build_model(generate(**GEN))
    assert plain.n_states == model.n_states  # as junções não trazem estados


def test_well_case_builds():
    net = case_from_json("cases/severe_slug/well_choke.json")
    assert {e.id for e in net.edges} == {"1", "reservoir_1", "choke_2"}
    assert net.edges_of(Choke)[0].law == "homogeneous"


def test_parallel_batch_equals_sequential(tmp_path):
    from mfsim.runs import load_run, run_batch

    runs = [{"name": f"r{k}", "case": "cases/chapter10/10_1_1.json", "tf": 5.0} for k in range(2)]
    for jobs, sub in ((1, "seq"), (2, "par")):
        spec = {"out_dir": str(tmp_path / sub), "runs": runs}
        p = tmp_path / f"{sub}.json"
        p.write_text(json.dumps(spec))
        done = run_batch(str(p), progress=False, jobs=jobs)
        assert [d[0] for d in done] == ["r0", "r1"] and all(d[2].get("success", True) for d in done)
    for k in range(2):
        a, b = load_run(str(tmp_path / "seq" / f"r{k}.npz")), load_run(str(tmp_path / "par" / f"r{k}.npz"))
        assert np.array_equal(a.X, b.X)
