"""Malha não uniforme (progressão geométrica por trecho)."""

import copy

import numpy as np

from mfsim.cases import chapter10_case
from mfsim.geometry import pipe_mesh
from mfsim.simulation import steady_state


def test_graded_mesh_lengths_and_ratio():
    mesh = pipe_mesh([0, 0, 0], [0, 100.0, 100.0], [0, 0, 50.0], [10, 5], grading=[0.2, 4.0])
    h = mesh["h"]
    assert abs(h[:10].sum() - 100.0) < 1e-9 and abs(h[10:].sum() - 50.0) < 1e-9
    assert abs(h[9] / h[0] - 0.2) < 1e-9 and abs(h[14] / h[10] - 4.0) < 1e-9
    assert np.all(np.diff(h[:10]) < 0) and np.all(np.diff(h[10:]) > 0)
    assert abs(mesh["s"][-1] - 150.0) < 1e-9 and np.allclose(mesh["xyz"][-1], [0, 100.0, 50.0])
    uni = pipe_mesh([0, 0, 0], [0, 100.0, 100.0], [0, 0, 50.0], [10, 5])
    assert np.allclose(uni["h"][:10], 10.0) and np.allclose(uni["h"][10:], 10.0)


def test_steady_state_on_graded_mesh_matches_uniform():
    case = chapter10_case("10.1.1")
    c_u = copy.deepcopy(case).set_method("fv_upwind")
    c_u.pipes[0].ic = None
    c_g = copy.deepcopy(c_u)
    c_g.pipes[0].grading = [0.25] * len(c_g.pipes[0].n_elements)
    m_u, X_u, _ = steady_state(c_u, X0="auto", verbose=False)
    m_g, X_g, _ = steady_state(c_g, X0="auto", verbose=False)
    pu, pg = m_u.pipes[0], m_g.pipes[0]
    assert pg.mesh["h"][-1] < 0.5 * pg.mesh["h"][0]
    # pressão e fração de gás nas células: a malha graduada interpola a uniforme
    _, Yu, _, _ = m_u.assemble(0.0, X_u)
    _, Yg, _, _ = m_g.assemble(0.0, X_g)
    xu, xg = pu.scheme.x_cells, pg.scheme.x_cells
    N_u, N_g = pu.scheme.N, pg.scheme.N
    for slot, tol in ((0, 1e-2), (1, 1e-3)):
        ref = np.interp(xg, xu, Yu[0][1 : N_u + 1, slot])
        assert np.max(np.abs(Yg[0][1 : N_g + 1, slot] - ref)) < tol * np.max(np.abs(ref))
