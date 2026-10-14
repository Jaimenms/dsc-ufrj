"""Golden da intermitência severa: a corrida de referência (configuração recomendada com
arrasto físico de bolhas) por 150 s em numpy e em JAX é bit-idêntica e tem a mesma
assinatura de ciclo; rhs, transformação conservativa, escalas e fluxos de contorno
avaliados nos estados históricos da trajetória guardada são bit-idênticos; o caso de
laboratório de Jansen por 10 s em JAX é bit-idêntico."""
import numpy as np
import pytest

from mfsim.analysis import cycle_signature
from mfsim.runs import case_from_meta, integrate_run, integrator_opts_from_meta, load_run

pytestmark = [pytest.mark.golden, pytest.mark.data]

REFERENCE_RUN = "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys"
LAB_RUN = "severe_slug_lab_jansen_tese_bloqueada"


def _reference(data_dir, tf, dt_out):
    run = load_run(str(data_dir / f"{REFERENCE_RUN}.npz"))
    return run, case_from_meta(run.meta, tf=tf, dt_out=dt_out), integrator_opts_from_meta(run.meta)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_reference_150s(backend, golden, data_dir):
    if backend == "jax":
        pytest.importorskip("jax")
    g = golden("severe.npz")
    run, case, opts = _reference(data_dir, 150.0, 2.0)
    model, T, X, info = integrate_run(case, dict(opts, jax=(backend == "jax")))
    g.assert_equal(f"reference_{backend}_150s/T", T)
    g.assert_equal(f"reference_{backend}_150s/X", X)
    g.assert_equal(f"reference_{backend}_150s/acc", info["accum"])
    g.assert_equal(f"reference_{backend}_150s/n_steps", np.array(info["n_steps"]))
    # assinatura de ciclo (comparação robusta ao caos, usada quando a ordem das operações muda)
    from mfsim.results import Results
    res = Results(model, T, X); p = res.pipe(0)
    sig = cycle_signature(T, p["out"]["P_bar"][1], p["out"]["qL"][-1], min_gap=50.0)
    Xg = g[f"reference_{backend}_150s/X"]; resg = Results(model, T, Xg); pg = resg.pipe(0)
    sig_g = cycle_signature(T, pg["out"]["P_bar"][1], pg["out"]["qL"][-1], min_gap=50.0)
    assert sig["n_blowdowns"] == sig_g["n_blowdowns"] and abs(sig["t_first"] - sig_g["t_first"]) <= 2.0
    assert abs(sig["P_min"] - sig_g["P_min"]) <= 0.01 and abs(sig["P_max"] - sig_g["P_max"]) <= 0.01
    assert abs(sig["q_peak"] - sig_g["q_peak"]) <= 0.02 * sig_g["q_peak"]


def test_reference_rhs_at_historical_states(golden, data_dir):
    g = golden("severe.npz")
    run = load_run(str(data_dir / f"{REFERENCE_RUN}.npz")); model = run.model; T, X = run.T, run.X
    idx = g["reference_samples/idx"]
    g.assert_equal("reference_samples/rhs", np.stack([model.rhs(T[i], X[:, i]) for i in idx]))
    g.assert_equal("reference_samples/transform", np.stack([model.transform(X[:, i]) for i in idx]))
    g.assert_equal("reference_samples/inverse", np.stack([model.inverse_transform(model.transform(X[:, i])) for i in idx]))
    g.assert_equal("reference_samples/transform_jac", np.stack([model.transform_jacobian(X[:, i]).toarray() for i in idx[:5]]))
    g.assert_equal("reference_samples/state_scales", model.state_scales(X[:, 0]))
    g.assert_equal("reference_samples/boundary_net_fluxes", np.stack([np.asarray(model.boundary_net_fluxes(T[i], X[:, i]), float) for i in idx]))
    g.assert_equal("reference_samples/sparsity", np.asarray(model.jac_sparsity(), dtype=float).astype(np.int8))


def test_reference_jax_at_historical_states(golden, data_dir):
    pytest.importorskip("jax")
    from mfsim.jaxrhs import build_jax_rhs
    g = golden("severe.npz")
    run = load_run(str(data_dir / f"{REFERENCE_RUN}.npz")); model = run.model; T, X = run.T, run.X
    idx = g["reference_samples/idx"]
    rhs_j, jac_j = build_jax_rhs(model)
    g.assert_equal("reference_samples/rhs_jax", np.stack([rhs_j(T[i], X[:, i]) for i in idx]))
    g.assert_equal("reference_samples/jac_jax", np.stack([np.asarray(jac_j(T[i], X[:, i])) for i in idx[:3]]))


def test_lab_jansen_jax_10s(golden, data_dir):
    pytest.importorskip("jax")
    g = golden("severe.npz")
    run = load_run(str(data_dir / f"{LAB_RUN}.npz"))
    case = case_from_meta(run.meta, tf=10.0, dt_out=0.25)
    model, T, X, info = integrate_run(case, dict(integrator_opts_from_meta(run.meta), jax=True))
    g.assert_equal("lab_jansen_jax_10s/T", T); g.assert_equal("lab_jansen_jax_10s/X", X)
    g.assert_equal("lab_jansen_jax_10s/n_steps", np.array(info["n_steps"]))
