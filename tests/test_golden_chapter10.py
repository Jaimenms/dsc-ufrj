"""Golden do capítulo 10: as trajetórias que os notebooks calculam (run_case) e os cálculos
secundários (estacionário automático, estudo de esquemas, 10.1.4 consistente, 10.3.1 em
volumes finitos) têm de ser bit-idênticos aos gerados antes da reorganização."""
import copy

import numpy as np
import pytest

from mfsim.cases import CHAPTER10, load_chapter10_case
from mfsim.notebook_utils import run_case
from mfsim.simulation import simulate, steady_state

NB_KW = {"10.3.2a": dict(rtol=1e-3, atol=1e-5), "10.3.2b": dict(rtol=1e-3, atol=1e-5)}

pytestmark = pytest.mark.golden


@pytest.mark.parametrize("key", list(CHAPTER10))
def test_trajectory(key, golden):
    g = golden("chapter10.npz")
    case, raw, res = run_case(key, **NB_KW.get(key, {}))
    g.assert_equal(f"{key}/T", res.T)
    g.assert_equal(f"{key}/X", res.X)
    g.assert_equal(f"{key}/n_steps", np.array(res.info.get("n_steps", -1)))


def test_variant_10_1_1_steady_auto(golden):
    g = golden("variants.npz")
    case, raw = load_chapter10_case("10.1.1")
    case_ss = copy.deepcopy(case); case_ss.pipes[0].bc_in = (3, 1, 2, 2); case_ss.pipes[0].ic = None
    model_ss, X_ss, info_ss = steady_state(case_ss, X0="auto", verbose=False)
    g.assert_equal("10.1.1_steady_auto/X", X_ss)
    res2 = simulate(copy.deepcopy(case_ss), integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
    g.assert_equal("10.1.1_c2/T", res2.T); g.assert_equal("10.1.1_c2/X", res2.X)


@pytest.mark.parametrize("method,nel", [("galerkin", 5), ("galerkin", 25), ("fd_central", 5), ("fd_central", 25)])
def test_variant_10_1_1_mesh_study(method, nel, golden):
    g = golden("variants.npz")
    case, raw = load_chapter10_case("10.1.1")
    c = copy.deepcopy(case); c.pipes[0].bc_in = (3, 1, 2, 2); c.pipes[0].n_elements = np.array([nel]); c.pipes[0].ic = None; c.settings.method = method
    steady_state(c, X0="auto", verbose=False)
    r = simulate(c, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
    g.assert_equal(f"10.1.1_mesh{nel}_{method}/T", r.T); g.assert_equal(f"10.1.1_mesh{nel}_{method}/X", r.X)


def test_variant_10_1_4_steady(golden):
    g = golden("variants.npz")
    case, raw = load_chapter10_case("10.1.4")
    case_ss = copy.deepcopy(case); n = case_ss.pipes[0].n_elements[0] + 1
    case_ss.pipes[0].ic = np.stack([np.full(n, 0.3), np.full(n, 1.0), np.full(n, 0.5), np.full(n, 0.1)], axis=1)
    steady_state(case_ss, verbose=False)
    r = simulate(case_ss, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
    g.assert_equal("10.1.4_steady/T", r.T); g.assert_equal("10.1.4_steady/X", r.X)


def test_variant_10_3_1_fv_drift(golden):
    g = golden("variants.npz")
    case, raw = load_chapter10_case("10.3.1")
    case_fv = copy.deepcopy(case); case_fv.settings.method = "fv_upwind"
    case_fv.pipes[0].extra.update({"conservative": True, "q_switch": 0.05})
    steady_state(case_fv, verbose=False, update_case=True)
    r = simulate(case_fv, progress=False, integrator="implicit", rtol=1e-3, atol=1e-6)
    g.assert_equal("10.3.1_fv_drift/T", r.T); g.assert_equal("10.3.1_fv_drift/X", r.X)
    g.assert_equal("10.3.1_fv_drift/n_steps", np.array(r.info.get("n_steps", -1)))
