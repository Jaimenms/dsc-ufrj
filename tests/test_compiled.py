"""`CompiledNetwork`: o mesmo `NetworkModel` com o backend JAX reproduz o numpy (1e-9) em
toda a matriz de configurações (modelos, esquemas nodais e escalonados, elementos de
rede, vértices dinâmicos), traça sem ramos em valores (`make_jaxpr`) e tem Jacobiana
finita; o esquema fv_upwind_cons passa a ser coberto pelo caminho compilado."""

import copy

import numpy as np
import pytest

from mfsim.io_legacy import chapter10_case
from generators.severe_slug import severe_slug_case
from mfsim.fluid import Fluid
from mfsim.simulation import build_model

jax = pytest.importorskip("jax")
from mfsim.compiled import CompiledNetwork  # noqa: E402


def _close(a, b, what):
    a = np.asarray(a)
    b = np.asarray(b)
    assert np.all(np.isfinite(a)), f"{what}: numpy não finito"
    assert np.max(np.abs(a - b)) < 1e-9 * max(1.0, np.max(np.abs(a))), what


def _severe(method="fv_upwind", **extra):
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    D = 0.05
    A = np.pi * D**2 / 4
    qL, qG = 0.3 * 900 * A, 0.1 * fluid.rhoG(5e5) * A
    return severe_slug_case(
        method=method,
        tf=100,
        dt_out=2.0,
        qL=qL,
        qG=qG,
        D=D,
        P_out_bar=4.0,
        L_h1=20.0,
        L_down=300.0,
        slope_down_deg=-3.0,
        H_riser=20.0,
        L_h2=5.0,
        n_h1=2,
        n_down=8,
        n_riser=4,
        n_h2=2,
        regime=4,
        single_pipe=True,
        fluid=fluid,
        roughness=1e-5,
        start="blocked",
        elbow_radius=5.0,
        n_elbow=3,
        extra={"regime_vertical": 2, "vertical_blend": 20.0, **extra},
    )


MATRIX = {
    "10.1.1": lambda: chapter10_case("10.1.1"),
    "10.1.4": lambda: chapter10_case("10.1.4"),
    "10.2.2": lambda: chapter10_case("10.2.2"),
    "10.3.2b": lambda: chapter10_case("10.3.2b"),
    "10.3.3": lambda: chapter10_case("10.3.3"),
    "severe_cons": lambda: _severe(conservative=True, art_visc=0.05),
    "severe_prim_muscl": lambda: _severe(conservative=False, fv_order=2, alpha_face="muscl"),
    "severe_fv_cons_scheme": lambda: _severe(method="fv_upwind_cons", art_visc=0.05),
    "severe_auto_regime": lambda: _severe(conservative=True, art_visc=0.05, regime=6, low_void_regime=2),
}


@pytest.mark.parametrize("key", sorted(MATRIX))
def test_compiled_matches_numpy_and_traces(key):
    case = MATRIX[key]()
    case.set_options(bb_derivatives="fd")
    model = build_model(case)
    cn = CompiledNetwork(model)
    X0 = model.initial_state()
    t = case.time.t0 + 1.0
    E, opt = model.time_inputs(t)
    jaxpr = jax.make_jaxpr(cn.system.rhs_core)(X0, E, opt)  # traça: sem ramos em valores
    assert len(jaxpr.jaxpr.eqns) > 0
    rng = np.random.default_rng(0)
    slots = np.concatenate([pb.free % 4 for pb in model.pipes])
    for k in range(3):
        x = X0 * (1 + 0.02 * rng.standard_normal(X0.size))
        if k and key.startswith("severe") and "cons_scheme" not in key:
            x[: slots.size][slots == 0] = np.clip(
                x[: slots.size][slots == 0] + 0.3 * rng.random((slots == 0).sum()), 0.0, 1.0
            )
        _close(model.rhs(t, x), cn.rhs(t, x), f"{key} rhs {k}")
    J = cn.jac(t, X0)
    assert J.shape == (X0.size, X0.size) and np.all(np.isfinite(J))


def test_specifications_info_numpy_only():
    """Os diagnósticos dos elementos (válvulas, reservatório, retenção) continuam no lado numpy."""
    case = chapter10_case("10.3.3")
    model = build_model(case)
    E, info = model.specifications(case.time.t0 + 1.0, model.initial_state())
    assert info["valves"] or info["reservoirs"] or info["check_valves"]
    for grp in info.values():
        for d in grp:
            assert all(np.isfinite(float(v)) for v in d.values() if not isinstance(v, (int, np.integer)))


@pytest.mark.data
@pytest.mark.parametrize("name", ["severe_slug_lab_jansen_fisico", "severe_slug_lab_fabre2_fisico"])
def test_lab_shut_in_jacobian_finite(name, data_dir):
    """Casos de laboratório com partida de linha fechada (fases paradas, arrasto físico de
    bolhas): a Jacobiana compilada é finita no estado inicial (regressão encontrada na
    noturna: NaN pelo diâmetro de bolha infinito)."""
    from mfsim.runs import load_run

    run = load_run(str(data_dir / f"{name}.npz"))
    model = build_model(run.network(tf=10.0))
    cn = CompiledNetwork(model)
    X0 = model.initial_state()
    assert np.all(np.isfinite(cn.rhs(0.0, X0)))
    assert np.all(np.isfinite(cn.jac(0.0, X0)))
