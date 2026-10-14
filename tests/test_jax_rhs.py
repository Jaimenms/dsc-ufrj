"""O lado direito em JAX reproduz o numpy (fv_upwind conservativo, estratificado +
bolhas com misturas, limitador, relaxação, viscosidade artificial, pressão
interfacial) e a Jacobiana automática coincide com diferenças finitas onde o
resíduo é suave."""

import numpy as np
import pytest

from generators.severe_slug import severe_slug_case
from mfsim.fluid import Fluid
from mfsim.simulation import build_model

jax = pytest.importorskip("jax")


def _model(extra):
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    D = 0.05
    A = np.pi * D**2 / 4
    qL, qG = 0.3 * 900 * A, 0.1 * fluid.rhoG(5e5) * A
    case = severe_slug_case(
        method="fv_upwind",
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
        extra={"regime_vertical": 2, "conservative": True, "vertical_blend": 20.0, **extra},
    )
    return build_model(case)


@pytest.mark.parametrize(
    "extra",
    [
        {"art_visc": 0.05},
        {"art_visc": 0.01, "low_void_regime": 2, "interfacial_pressure": 1.2},
        {"art_visc": 0.05, "relax_target": "noslip"},
        {"art_visc": 0.05, "virtual_mass": 0.5},
        {"art_visc": 0.05, "high_void_regime": 1, "low_void_regime": 2},
    ],
)
def test_jax_rhs_matches_numpy(extra):
    from mfsim.compiled import build_jax_rhs

    model = _model(extra)
    rhs_j, jac_j = build_jax_rhs(model)
    rng = np.random.default_rng(0)
    X0 = model.initial_state()
    for k in range(4):
        x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))  # estados perturbados (frações, pressões e vazões)
        pb = model.pipes[0]
        slots = pb.free % 4
        x[slots == 0] = np.clip(x[slots == 0] + 0.3 * rng.random((slots == 0).sum()), 0.0, 1.0)
        f_np = model.rhs(0.0, x)
        f_jx = rhs_j(0.0, x)
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    # Jacobiana finita também no estado inicial (frações exatamente em 0, vazões nulas)
    assert np.all(np.isfinite(jac_j(0.0, X0)))
    # e igual às diferenças finitas centradas num estado afastado das quinas
    # (clips em alpha = 0/1 e chaveamentos em q = 0 não têm derivada única)
    x = X0.copy()
    slots = model.pipes[0].free % 4
    x[slots == 0] = 0.2 + 0.6 * rng.random((slots == 0).sum())
    x[slots == 2] = 0.3 + 0.2 * rng.random((slots == 2).sum())
    x[slots == 3] = 0.002 + 0.002 * rng.random((slots == 3).sum())
    J = jac_j(0.0, x)
    for j in rng.choice(x.size, 16, replace=False):
        h = 1e-6 * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        col = (model.rhs(0.0, xp) - model.rhs(0.0, xm)) / (2 * h)
        scale = np.max(np.abs(J[:, j])) + np.max(np.abs(col)) + 1e-8
        assert np.max(np.abs(col - J[:, j])) < 1e-5 * scale


@pytest.mark.parametrize(
    "extra",
    [
        {"fv_order": 2},
        {"fv_order": 2, "alpha_face": "central", "conservative": False},
        {"alpha_face": "muscl"},
        {"fv_order": 2, "alpha_face": "muscl"},
    ],
)
def test_jax_muscl_matches_numpy(extra):
    """fv_order=2 (MUSCL/minmod na velocidade doadora), alpha_face='central' e alpha_face='muscl'
    (reconstrução TVD da fração de área nas faces) em JAX."""
    from mfsim.compiled import build_jax_rhs

    model = _model({"art_visc": 0.02, **extra})
    assert model.pipes[0].scheme.order == extra.get("fv_order", 1)
    rhs_j, jac_j = build_jax_rhs(model)
    rng = np.random.default_rng(2)
    X0 = model.initial_state()
    slots = model.pipes[0].free % 4
    for k in range(4):
        x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))
        x[slots == 0] = np.clip(x[slots == 0] + 0.3 * rng.random((slots == 0).sum()), 0.0, 1.0)
        f_np = model.rhs(0.0, x)
        f_jx = rhs_j(0.0, x)
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    assert np.all(np.isfinite(jac_j(0.0, X0)))
