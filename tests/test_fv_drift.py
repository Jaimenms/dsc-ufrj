"""Volumes finitos escalonados para o modelo de deriva (Beggs & Brill): o estado
estacionário do poço 10.3.1 converge e é linearmente estável, o lado direito JAX
reproduz o numpy, a Jacobiana é finita, o transiente de fechamento do choke
conserva massa exatamente (acumulador BDF) e fica próximo do esquema nodal."""

import copy

import numpy as np
import pytest

from mfsim.cases import load_chapter10_case
from mfsim.simulation import build_model, state_bounds, steady_state
from mfsim.integrators.implicit import ImplicitBDF


def _fv_case(bb_derivatives="fd"):
    case, raw = load_chapter10_case("10.3.1")
    case.set_method("fv_upwind").set_options(conservative=True, q_switch=0.05, bb_derivatives=bb_derivatives)
    return case


def test_drift_fv_steady_state_stable():
    case = _fv_case()
    model, X, info = steady_state(case, verbose=False, update_case=False)
    assert np.max(np.abs(model.rhs(0.0, X))) < 1e-8
    pb = model.pipes[0]
    assert tuple(pb.scheme.cell_slots) == (0, 1)        # hold-up e pressão são estados de célula
    n = X.size
    J = np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(X[j]), 1e-3)
        xp, xm = X.copy(), X.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (model.rhs(0.0, xp) - model.rhs(0.0, xm)) / (2 * h)
    ev = np.linalg.eigvals(J)
    assert np.max(ev.real) < 0.0                        # sem modos crescentes (o sinal do coeficiente da restrição)
    # hold-up das células próximo do hold-up da correlação no esquema nodal
    E, Ys, _, _ = model.assemble(0.0, X)
    HG_cells = Ys[0][1:-1, 0]
    assert np.all((HG_cells > 0.2) & (HG_cells < 0.8))
    assert HG_cells[-1] > HG_cells[0]                   # mais gás no topo (expansão)


def test_drift_fv_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    from mfsim.compiled import build_jax_rhs
    case = _fv_case("fd")
    model, X, info = steady_state(case, verbose=False, update_case=False)
    rhs_j, jac_j = build_jax_rhs(model)
    rng = np.random.default_rng(0)
    for k in range(4):
        x = X * (1 + 0.05 * rng.standard_normal(X.size))
        f_np = model.rhs(1.0, x)
        assert np.max(np.abs(f_np - rhs_j(1.0, x))) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    J = jac_j(1.0, X)
    assert np.all(np.isfinite(J))
    assert np.max(np.linalg.eigvals(J).real) < 0.0


def test_drift_fv_transient_conserves_mass_and_matches_nodal():
    case_n, raw = load_chapter10_case("10.3.1")
    case = _fv_case()
    model, X0, info = steady_state(case, verbose=False, update_case=False)
    lo, hi = state_bounds(model)
    kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
    bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-6 * np.ones(X0.size), bounds=(lo, hi),
                      min_step=1e-9, verbose=False, max_order=2, max_newton=20, newton_tol=1e-3, **kw_t)
    bdf.accum = model.boundary_net_fluxes
    t_eval = np.arange(0.0, 300.1, 5.0)                 # fechamento do choke em 60-90 s
    T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3, breakpoints=model.spec_breakpoints())
    assert info["success"]
    # balanço exato por fase (estoque das células vs integral BDF-consistente dos fluxos de contorno)
    pb = model.pipes[0]
    A = np.pi * pb.model.params.D**2 / 4
    h = pb.mesh["h"]
    X = np.asarray(X)
    X = X.T if X.shape[0] != T.size else X
    stock = []
    for it in (0, -1):
        E, Ys, _, _ = model.assemble(T[it], X[it])
        Y = Ys[0]
        rhoL, rhoG, _, _ = model.fluid.props(Y[1:-1, 1] * model.PR)
        stock.append((np.sum((1 - Y[1:-1, 0]) * rhoL * A * h), np.sum(Y[1:-1, 0] * rhoG * A * h)))
    acc = info["accum"]
    assert abs(stock[1][0] - stock[0][0] - acc[0, -1]) < 1e-6 * stock[0][0]
    assert abs(stock[1][1] - stock[0][1] - acc[1, -1]) < 1e-6 * stock[0][1]
    # resposta ao fechamento do choke comparável à do esquema nodal da tese
    from mfsim.simulation import simulate
    res_n = simulate(copy.deepcopy(case_n), progress=False, t_eval=t_eval)
    qn = res_n.pipe(0)["out"]["qL"][-1]
    E, Ys, _, _ = model.assemble(T[-1], X[-1])
    q_fv = [model.assemble(T[i], X[i])[1][0][-2, 2] for i in range(T.size)]
    assert abs(np.min(q_fv) - np.min(qn)) < 0.2 * np.min(qn)
    assert abs(np.max(q_fv) - np.max(qn)) < 0.2 * np.max(qn)
