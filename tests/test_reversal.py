"""Inversão do sentido do escoamento no esquema escalonado conservativo: a fração de
gás que entra pela extremidade invertida é a imposta pelo vértice (separador cheio
de líquido), o balanço fecha e o duto se enche de líquido."""

import numpy as np

from mfsim.cases.severe_slug import BC_ABSENT, BC_DIRICHLET, Case, Pipe, SimulatorSettings, Vertex
from mfsim.fluid import Fluid
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.analysis import mass_balance
from mfsim.simulation import build_model, state_bounds


def test_reversal_takes_separator_holdup():
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    D = 0.05; PR = 1e6; N = 12; L = 100.0; tf = 200.0
    settings = SimulatorSettings(t0=0.0, tf=tf, dt_out=1.0, PR=PR, method="fv_upwind", max_step=np.inf, atol=1e-6, rtol=1e-4, quadrature_order=2, integrator="implicit")
    P_in = (np.array([0.0, 60.0, 100.0, tf]), np.array([5.3e5, 5.3e5, 4.7e5, 4.7e5]) / PR)
    v1 = Vertex(index=1, kind="B", spec={"PT": P_in, "alphaG": 0.5})
    v2 = Vertex(index=2, kind="B", spec={"PT": 5.0e5 / PR, "alphaG": 0.0})
    pipe = Pipe(index=1, v_in=1, v_out=2, x=[0, 0], y=[0, L], z=[0, 0], n_elements=[N], D=D, roughness=1e-5, model=1, regime=4,
                bc_in=(BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT), bc_out=(BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT),
                extra={"conservative": True, "art_visc": 0.05, "q_switch_L": 0.01, "q_switch_G": 1e-4})
    pipe.ic = np.stack([np.full(N + 1, 0.5), np.linspace(5.3e5, 5.0e5, N + 1) / PR, np.zeros(N + 1), np.zeros(N + 1)], axis=1)
    case = Case(name="inversao", fluid=fluid, settings=settings, vertices=[v1, v2], pipes=[pipe])
    case.settings.extra["state_ref"] = dict(PT=0.6, qL=2.0, qG=0.01)
    model = build_model(case); X0 = model.initial_state()
    bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=3e-3, atol=0.01 * model.state_scales(X0), bounds=state_bounds(model), max_newton=20,
                      transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
    bdf.accum = model.boundary_net_fluxes
    T, X, info = bdf.integrate([0.0, 50.0, 200.0], X0, breakpoints=model.spec_breakpoints())
    res = Results(model, T, X); p = res.pipe(0)
    qL, aG = p["out"]["qL"], p["out"]["alphaG"]
    assert qL[-1, 1] > 0.5 and qL[-1, 2] < -0.5            # escoamento invertido no fim
    assert 0.4 < aG[N, 1] < 0.6                              # antes: fração da alimentação
    assert np.all(aG[1:N + 1, 2] < 0.02)                     # depois: duto cheio do líquido do separador
    mL, _, _ = mass_balance(res, 0, "L")
    assert abs(mL[-1] - mL[0] - info["accum"][0, -1]) < 1e-3 * mL[0]
