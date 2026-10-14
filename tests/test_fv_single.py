"""Volumes finitos escalonados para os modelos monofásicos: o lado direito JAX
reproduz o numpy (ordens 1 e 2), a Jacobiana é finita e a integração implícita
(numpy e JAX) chega ao mesmo estado num transiente de fechamento de válvula."""

import numpy as np
import pytest

from mfsim.fluid import Fluid
from mfsim.network.data import BC_ABSENT, BC_DIRICHLET, Case, Pipe, SimulatorSettings, Vertex
from mfsim.simulation import build_model, simulate

jax = pytest.importorskip("jax")


def _case(phase="L", order=1, tf=20.0):
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=1000.0, gammaG=1.05, P0=1e5, sigma=0.05)
    PR = 1e6
    D, L, N = 0.1, 500.0, 20
    A = np.pi * D**2 / 4
    P_in = 10e5
    if phase == "L":
        q0 = 1.0 * fluid.rhoL0 * A
        model, slot = 3, 2
    else:
        q0 = 5.0 * fluid.rhoG(P_in) * A
        model, slot = 4, 3
    key = "qL" if phase == "L" else "qG"
    # vazão de saída cai à metade entre t = 2 s e t = 4 s (transiente de pressão)
    spec_out = {key: (np.array([0.0, 2.0, 4.0, tf]), np.array([q0, q0, 0.5 * q0, 0.5 * q0]))}
    vertices = [Vertex(index=1, kind="B", spec={"PT": P_in / PR}), Vertex(index=2, kind="B", spec=spec_out)]
    s = np.linspace(0, L, N + 1)
    ic = np.zeros((N + 1, 4))
    ic[:, 0] = 0.0 if phase == "L" else 1.0
    ic[:, 1] = (P_in - 0.1e5 * s / L) / PR
    ic[:, slot] = q0
    bc_in = (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
    bc_out = (BC_ABSENT, BC_ABSENT, BC_DIRICHLET, BC_ABSENT) if phase == "L" else (BC_ABSENT, BC_ABSENT, BC_ABSENT, BC_DIRICHLET)
    pipe = Pipe(index=1, v_in=1, v_out=2, x=[0.0, 0.0], y=[0.0, L], z=[0.0, 10.0], n_elements=[N], D=D, roughness=1e-5,
                model=model, regime=4, bc_in=bc_in, bc_out=bc_out, ic=ic, extra={"fv_order": order, "q_switch": 1e-2 * q0})
    settings = SimulatorSettings(t0=0.0, tf=tf, dt_out=1.0, PR=PR, method="fv_upwind", max_step=np.inf,
                                 atol=1e-6, rtol=1e-4, quadrature_order=2, integrator="implicit")
    return Case(name=f"single_{phase}", fluid=fluid, settings=settings, vertices=vertices, pipes=[pipe])


@pytest.mark.parametrize("phase, order", [("L", 1), ("G", 1), ("L", 2)])
def test_single_phase_fv_jax_matches_numpy(phase, order):
    from mfsim.jaxrhs import build_jax_rhs
    model = build_model(_case(phase, order))
    assert model.pipes[0].scheme.name == "fv_upwind" and not model.conservative
    rhs_j, jac_j = build_jax_rhs(model)
    X0 = model.initial_state()
    rng = np.random.default_rng(0)
    for k in range(4):
        x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))
        f_np = model.rhs(3.0, x)
        f_jx = rhs_j(3.0, x)
        assert np.all(np.isfinite(f_np))
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    assert np.all(np.isfinite(jac_j(3.0, X0)))


def test_single_phase_fv_integrates_numpy_and_jax():
    case = _case("L", 1, tf=20.0)
    res_np = simulate(case, progress=False, jax=False)
    res_jx = simulate(case, progress=False, jax=True)
    p_np = res_np.pipe(0)
    p_jx = res_jx.pipe(0)
    P_np, P_jx = p_np["out"]["P_bar"], p_jx["out"]["P_bar"]
    assert np.all(np.isfinite(P_np))
    # a vazão de entrada acompanha a redução da vazão de saída (novo estacionário em ~20 s)
    q0 = p_np["out"]["qL"][0, 0]
    assert abs(p_np["out"]["qL"][0, -1] - 0.5 * q0) < 0.05 * q0
    # a pressão na saída estabiliza abaixo da inicial (perda por atrito + 10 m de desnível)
    assert 8.5 < P_np[-2, -1] < P_np[-2, 0]
    assert np.max(np.abs(P_np - P_jx)) < 2e-2 * np.max(np.abs(P_np))
