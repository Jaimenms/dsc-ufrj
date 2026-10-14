"""Testes de regressão: reprodução das trajetórias gravadas pelo MFSIM/MATLAB."""

import numpy as np
import pytest

from thesis.mf2 import chapter10_case, legacy_results
from thesis.mf2 import legacy_states
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators import integrate_implicit


def _run(key, tol):
    case, raw = chapter10_case(key), legacy_results(key)
    model = build_model(case)
    Tl, Xl = legacy_states(raw)
    T, X, info = integrate_implicit(
        model, Xl[:, 0], Tl, rtol=1e-4, atol=1e-6, bounds=state_bounds(model), progress=False
    )
    for pb in model.pipes:
        slots = pb.free % 4
        seg = slice(pb.offset, pb.offset + pb.n_states)
        for s in range(4):
            m = slots == s
            if m.any():
                rng = np.ptp(Xl[seg][m])
                d = np.max(np.abs(X[seg][m] - Xl[seg][m]))
                assert d <= tol * max(rng, 1e-12), f"{key} duto {pb.pipe.id} slot {s}: {d:.2e} > {tol} x {rng:.2e}"


@pytest.mark.parametrize("key,tol", [("10.1.1", 0.01), ("10.2.1A", 0.01), ("10.2.1B", 0.01), ("10.2.2", 0.02)])
def test_two_fluid_cases(key, tol):
    _run(key, tol)


def test_steady_residual_is_zero_at_legacy_state():
    for key in ("10.1.1", "10.2.2", "10.3.3"):
        case, raw = chapter10_case(key), legacy_results(key)
        model = build_model(case)
        Tl, Xl = legacy_states(raw)
        r = model.rhs(Tl[0], Xl[:, 0])
        assert np.linalg.norm(r) < 1e-6, key


def test_staggered_steady_state_unique():
    """O esquema de volumes finitos converge para o mesmo estado estacionário em malhas diferentes."""
    import copy
    from mfsim.simulation import steady_state

    case, raw = chapter10_case("10.1.1"), legacy_results("10.1.1")
    vals = []
    for nel in (10, 50):
        c = copy.deepcopy(case)
        c.pipes[0].n_elements = np.array([nel])
        c.pipes[0].ic = None
        c.set_method("fv_upwind")
        model, X, info = steady_state(c, X0="auto", verbose=False)
        assert info["converged"]
        E, Ys, _, _ = model.assemble(0.0, X)
        vals.append(Ys[0][1, 0])
    assert abs(vals[0] - vals[1]) < 1e-4
    assert abs(vals[0] - 0.4636) < 1e-3
