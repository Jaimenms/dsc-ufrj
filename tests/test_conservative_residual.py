"""O modo conservativo do integrador (resíduo T(x_{n+1}) - T(x_n) = h f) deve
fechar exatamente o balanço de massa por célula em cada passo."""

import copy

from mfsim.network.schedules import Constant

import numpy as np

from thesis.mf2 import chapter10_case
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.simulation import build_model, state_bounds, steady_state


def _mass_per_cell(model, X):
    """Massas de líquido e gás por célula [kg] a partir do estado primitivo."""
    pb = model.pipes[0]
    E, Ys, _, _ = model.assemble(0.0, X)  # inclui as condições de contorno (face de entrada)
    Y = Ys[0]
    N = pb.scheme.N
    h = pb.mesh["h"]
    A = np.pi * pb.pipe.D**2 / 4
    P = Y[1 : N + 1, 1] * model.PR
    rhoL, rhoG, _, _ = model.fluid.props(P)
    aG = Y[1 : N + 1, 0]
    return rhoL * (1 - aG) * A * h, rhoG * aG * A * h, Y


def test_conservative_mode_balances_mass_each_step():
    case = chapter10_case("10.1.1")
    c = copy.deepcopy(case)
    c.set_method("fv_upwind").set_options(conservative=True)
    c.pipes[0].ic = None
    model0, X0, _ = steady_state(c, X0="auto", verbose=False)
    # perturbação: vazão de gás 30% maior na entrada a partir de t=0 (transiente forte)
    c.vertices[0].spec["qG"] = Constant(0.13)
    model = build_model(c)
    bdf = ImplicitBDF(
        model.rhs,
        S=model.jac_sparsity(),
        rtol=1e-4,
        atol=1e-6,
        bounds=state_bounds(model),
        transform=model.transform,
        inverse=model.inverse_transform,
        transform_jac=model.transform_jacobian,
        max_order=1,
        fixed_step=0.5,
    )
    T, X, info = bdf.integrate([0.0, 0.5], X0, first_step=0.5)
    assert info["n_steps"] == 1
    h = T[-1] - T[0]
    # (i) resíduo conservativo do integrador: T(x1) - T(x0) = h f(x1)
    r = model.transform(X[:, -1]) - model.transform(X[:, 0]) - h * model.rhs(T[-1], X[:, -1])
    assert np.max(np.abs(r)) < 1e-9
    # (ii) o mesmo em termos de massa por célula [kg]: dm = h (q_in - q_out)
    mL0, mG0, Y0 = _mass_per_cell(model, X[:, 0])
    mL1, mG1, Y1 = _mass_per_cell(model, X[:, -1])
    N = model.pipes[0].scheme.N
    FL, FG = model.pipes[0].scheme.mass_fluxes_nodal(Y1)  # fluxos limitados (= vazões onde há as duas fases)
    assert np.allclose(FL, Y1[0 : N + 1, 2]) and np.allclose(FG, Y1[0 : N + 1, 3])
    fluxL = (FL[0:N] - FL[1 : N + 1]) * h
    fluxG = (FG[0:N] - FG[1 : N + 1]) * h
    assert np.max(np.abs(mG1 - mG0)) > 1e-6  # houve transiente de fato
    assert np.max(np.abs((mL1 - mL0) - fluxL)) < 1e-9 * np.max(mL0)
    assert np.max(np.abs((mG1 - mG0) - fluxG)) < 1e-9 * np.max(mG0)


def test_conservative_mode_global_balance_adaptive_bdf2():
    """Com passos adaptativos e BDF2, a variação do estoque de cada fase no duto é
    igual à integral BDF-consistente dos fluxos de contorno (acumulador do integrador)."""
    case = chapter10_case("10.1.1")
    c = copy.deepcopy(case)
    c.set_method("fv_upwind").set_options(conservative=True)
    c.pipes[0].ic = None
    model0, X0, _ = steady_state(c, X0="auto", verbose=False)
    c.vertices[0].spec["qG"] = Constant(0.13)
    model = build_model(c)
    bdf = ImplicitBDF(
        model.rhs,
        S=model.jac_sparsity(),
        rtol=1e-3,
        atol=1e-5,
        bounds=state_bounds(model),
        transform=model.transform,
        inverse=model.inverse_transform,
        transform_jac=model.transform_jacobian,
        max_order=2,
    )
    bdf.accum = model.boundary_net_fluxes
    T, X, info = bdf.integrate([0.0, 7.0, 20.0], X0)
    assert info["n_steps"] > 10
    acc = info["accum"]
    mL0, mG0, _ = _mass_per_cell(model, X[:, 0])
    mL1, mG1, _ = _mass_per_cell(model, X[:, -1])
    assert abs(mG1.sum() - mG0.sum()) > 1e-3  # transiente de fato
    assert abs((mL1.sum() - mL0.sum()) - acc[0, -1]) < 1e-9 * mL0.sum()
    assert abs((mG1.sum() - mG0.sum()) - acc[1, -1]) < 1e-9 * mG0.sum()


def test_network_vertices_conserve_mass():
    """Rede de dutos conservativos com vértices dinâmicos conservativos: o estoque total
    (dutos + vértices) segue exatamente a integral BDF-consistente de entrada - saída."""
    import numpy as np
    from generators.severe_slug import severe_slug_case
    from mfsim.fluid import Fluid
    from mfsim.analysis import mass_balance
    from mfsim.results import Results

    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    D = 0.05
    A = np.pi * D**2 / 4
    qL = 0.3 * 900 * A
    qG = 0.1 * fluid.rhoG(5e5) * A
    case = severe_slug_case(
        method="fv_upwind",
        tf=10,
        dt_out=1.0,
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
        single_pipe=False,
        fluid=fluid,
        roughness=1e-5,
        start="blocked",
        elbow_radius=5.0,
        n_elbow=3,
        extra={
            "regime_vertical": 2,
            "conservative": True,
            "vertical_blend": 20.0,
            "art_visc": 0.05,
            "vertex_volume": 0.02,
        },
    )
    model = build_model(case)
    X0 = model.initial_state()
    bdf = ImplicitBDF(
        model.rhs,
        S=model.jac_sparsity(),
        rtol=3e-3,
        atol=1e-2 * model.state_scales(X0),
        bounds=state_bounds(model),
        transform=model.transform,
        inverse=model.inverse_transform,
        transform_jac=model.transform_jacobian,
        max_newton=20,
    )
    bdf.accum = model.boundary_net_fluxes
    T, X, info = bdf.integrate([0.0, 3.0, 6.0], X0, first_step=1e-3)
    acc = info["accum"]
    res = Results(model, T, X)
    npipes = len(model.pipes)
    # entrada do sistema - saída do sistema = soma(acum. dutos) + soma(acum. vértices)
    sys_net = acc[0 : 2 * npipes : 2].sum(axis=0) + acc[2 * npipes :: 2].sum(axis=0)
    mL = sum(mass_balance(res, k, "L")[0] for k in range(npipes))
    for vi in model.vA:
        v = model.vertices[vi]
        iH, iP = model.vertex_A_index(v.id)
        mL = mL + v.volume * (1 - X[iH]) * model.fluid.rhoL(np.maximum(X[iP] * model.PR, 1e3))
    assert abs(mL[-1] - mL[0]) > 1e-3
    assert abs((mL[-1] - mL[0]) - sys_net[-1]) < 1e-8 * mL[0]
    # e cada vértice segue a sua própria integral
    for k, vi in enumerate(model.vA):
        v = model.vertices[vi]
        iH, iP = model.vertex_A_index(v.id)
        mv = v.volume * (1 - X[iH]) * model.fluid.rhoL(np.maximum(X[iP] * model.PR, 1e3))
        assert abs((mv[-1] - mv[0]) - acc[2 * npipes + 2 * k, -1]) < 1e-9 * mL[0]
