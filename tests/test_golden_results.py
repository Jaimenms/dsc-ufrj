"""Golden do pós-processamento: todas as saídas de Results (dutos, vértices, elementos) em
três instantes das trajetórias golden e o balanço de massa são bit-idênticos."""

import numpy as np
import pytest

from mfsim.analysis import mass_balance
from mfsim.io_legacy import chapter10_case
from mfsim.results import Results
from mfsim.simulation import build_model

pytestmark = pytest.mark.golden


def _check_results(g, prefix, res, idx):
    r = Results(res.model, res.T[idx], res.X[:, idx])
    n = 0
    for k in range(len(res.model.pipes)):
        p = r.pipe(k)
        g.assert_equal(f"{prefix}/pipe{k}/Y", np.asarray(p["Y"]))
        for key, v in p["out"].items():
            g.assert_equal(f"{prefix}/pipe{k}/{key}", np.asarray(v, float))
            n += 1
    for vi, v in enumerate(r.vertices):
        for key, arr in v.items():
            g.assert_equal(f"{prefix}/vertex{vi}/{key}", np.asarray(arr, float))
            n += 1
    for kind, d in r.elements.items():
        for vertex, dd in d.items():
            for key, arr in dd.items():
                g.assert_equal(f"{prefix}/{kind}/{vertex}/{key}", np.asarray(arr, float))
                n += 1
    assert n == len(g.keys(prefix + "/")) - len(res.model.pipes) - len(
        [k for k in g.keys(prefix + "/") if "mass_balance" in k]
    ), "chaves de saída diferentes do golden"


@pytest.mark.parametrize("key", ["10.1.1", "10.3.1", "10.3.2b"])
def test_results_chapter10(key, golden):
    g = golden("results.npz")
    gc = golden("chapter10.npz")
    case = chapter10_case(key)
    model = build_model(case)
    T, X = gc[f"{key}/T"], gc[f"{key}/X"]
    res = Results(model, T, X)
    _check_results(g, key, res, np.array([0, T.size // 2, T.size - 1]))
    if key == "10.1.1":
        for ph in ("L", "G"):
            m, net, err = mass_balance(res, 0, ph)
            g.assert_equal(f"{key}/mass_balance_{ph}", np.stack([np.asarray(m, float), np.asarray(net, float)]))


@pytest.mark.data
def test_results_severe(golden, data_dir):
    from mfsim.runs import load_run

    g = golden("results.npz")
    gs = golden("severe.npz")
    run = load_run(str(data_dir / "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys.npz"))
    model = build_model(run.network(tf=150.0, dt_out=2.0))
    T, X = gs["reference_numpy_150s/T"], gs["reference_numpy_150s/X"]
    res = Results(model, T, X)
    _check_results(g, "severe150", res, np.array([0, T.size // 2, T.size - 1]))
    for ph in ("L", "G"):
        m, net, err = mass_balance(res, 0, ph)
        g.assert_equal(f"severe150/mass_balance_{ph}", np.stack([np.asarray(m, float), np.asarray(net, float)]))
