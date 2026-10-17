"""Sensibilidade ao estado inicial propagada no BDF próprio (issue 059, `integration.sensitivity = "state"`): no
problema-modelo e no caso de exemplo, as colunas de S = ∂x(t_f)/∂x(t_0) concordam com diferenças finitas centrais, e a
trajetória é a mesma com e sem sensibilidades."""

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

from mfsim.integrators.implicit import ImplicitBDF
from mfsim.io import case_from_json
from mfsim.simulation import simulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "papers", "01-severe-slugging-two-fluid", "toy"))


def _toy(a=0.95, tf=20.0, rtol=1e-6):
    import model_problem as mp

    p = mp.params()
    x0 = mp.steady(a, p)
    x0[: p["N"]] *= 1.001
    Mv = mp.mass(p)
    n = x0.size

    def run(x_init, sens):
        bdf = ImplicitBDF(
            lambda t, x: mp.rhs(x, a, p),
            S=sp.csc_matrix(np.ones((n, n))),
            rtol=rtol,
            atol=rtol * 1e-2,
            max_step=0.5,
            transform=lambda x: Mv * x,
            inverse=lambda y: y / Mv,
            transform_jac=lambda x: sp.diags(Mv),
            jac_fun=lambda t, x: mp.jac(x, a, p),
        )
        bdf.sensitivity = "state" if sens else None
        T, X, info = bdf.integrate(np.array([0.0, tf]), x_init)
        return X[:, -1], info

    return x0, run


def test_toy_sensitivity_matches_finite_differences():
    x0, run = _toy()
    xf, info = run(x0, True)
    S = info["S"]
    assert S.shape == (x0.size, x0.size)
    xf_plain, _ = run(x0, False)
    assert np.array_equal(xf, xf_plain)  # a mesma trajetória com e sem sensibilidades
    rng = np.random.default_rng(0)
    for j in rng.choice(x0.size, 3, replace=False):
        d = 1e-6 * max(1.0, abs(x0[j]))
        xp, xm = x0.copy(), x0.copy()
        xp[j] += d
        xm[j] -= d
        col_fd = (run(xp, False)[0] - run(xm, False)[0]) / (2 * d)
        assert np.max(np.abs(S[:, j] - col_fd)) < 2e-2 * max(1.0, np.max(np.abs(col_fd)))


def test_hello_world_sensitivity_matches_finite_differences():
    net = case_from_json("cases/examples/hello_world.json")
    net.set_time(tf=20.0, dt_out=20.0)
    res = simulate(net, progress=False, integrator="implicit", backend="jax", sensitivity="state")
    S = res.info["S"]
    X0 = np.asarray(res.X[:, 0], float)
    assert S.shape == (X0.size, X0.size)
    plain = simulate(net, progress=False, integrator="implicit", backend="jax")
    assert np.array_equal(np.asarray(res.X), np.asarray(plain.X))
    slots = res.model.state_slots()
    scale = res.model.state_scales(X0)
    # uma fração de gás e uma vazão de líquido a meio do duto; a coluna da pressão não serve de referência por
    # diferenças finitas: a perturbação excita a acústica e a resposta fica ao nível do ruído do integrador
    for j in (int(np.flatnonzero(slots == 0)[2]), int(np.flatnonzero(slots == 2)[2])):
        d = 1e-4 * max(1.0, abs(X0[j]))
        cols = []
        for sgn in (+1, -1):
            Xp = X0.copy()
            Xp[j] += sgn * d
            r = simulate(net, progress=False, integrator="implicit", backend="jax", X0=Xp)
            cols.append(np.asarray(r.X[:, -1], float))
        col_fd = (cols[0] - cols[1]) / (2 * d)
        assert np.max(np.abs(S[:, j] - col_fd) / scale) < 0.05 * max(1.0, np.max(np.abs(col_fd) / scale))
