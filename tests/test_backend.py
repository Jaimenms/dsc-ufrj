"""Primitivas do backend: numpy e JAX dão os mesmos valores (bit-iguais onde a fórmula é a
mesma) e o JAX tem tangentes finitas onde o numpy puro não teria."""

import numpy as np
import pytest

from mfsim.backend import NUMPY, get_backend


def _both():
    jax = pytest.importorskip("jax")
    return NUMPY, get_backend("jax")


def test_set_add_and_loops():
    for bk in _both():
        xp = bk.xp
        a = xp.zeros(5)
        b = bk.set(a, 2, 3.0)
        c = bk.add(b, xp.array([0, 2]), 1.0)
        assert np.array_equal(bk.to_numpy(c), [1, 0, 4, 0, 0]) and float(bk.to_numpy(a)[2]) == 0.0

        # while_loop com saída por tolerância (Newton de sqrt(2))
        def cond(s):
            return (xp.abs(s[1]) > 1e-12) & (s[2] < 50)

        def body(s):
            x, step, i = s
            step = (x * x - 2.0) / (2.0 * x)
            return x - step, step, i + 1

        x, _, it = bk.while_loop(cond, body, (xp.asarray(1.0), xp.asarray(1.0), 0))
        assert abs(float(x) - np.sqrt(2.0)) < 1e-12
        s = bk.fori_loop(0, 4, lambda i, c: c + i, 0)
        assert int(s) == 6


def test_sigmoid_same_values_and_finite_tangent():
    np_bk, jx = _both()
    import jax

    z = np.array([-800.0, -10.0, -1e-3, 0.0, 1e-3, 10.0, 800.0])
    assert np.array_equal(np_bk.sigmoid(z), np.asarray(jx.sigmoid(jx.asarray(z))))
    t = jax.jvp(jx.sigmoid, (jx.asarray(z),), (jx.xp.ones_like(jx.asarray(z)),))[1]
    assert np.all(np.isfinite(np.asarray(t)))


def test_safe_roots_and_robust_inverse():
    np_bk, jx = _both()
    import jax

    x = np.array([0.0, 1e-300, 4.0])
    assert np.array_equal(np_bk.safe_sqrt(x), np.asarray(jx.safe_sqrt(jx.asarray(x))))
    assert np.array_equal(np_bk.safe_pow(x, 0.4), np.asarray(jx.safe_pow(jx.asarray(x), 0.4)))
    assert np.all(np.isfinite(np.asarray(jax.jvp(jx.safe_sqrt, (jx.asarray(x),), (jx.xp.ones(3),))[1])))
    A = np.array([np.eye(3), [[1, 2, 3], [2, 4, 6], [0, 0, 1]], 2 * np.eye(3)], float)  # a segunda é singular
    inv_np = np_bk.robust_inv(A)
    inv_jx = np.asarray(jx.robust_inv(jx.asarray(A)))
    assert np.allclose(inv_np[0], np.eye(3)) and np.allclose(inv_np[2], 0.5 * np.eye(3))
    assert np.allclose(inv_np, inv_jx, rtol=1e-10, atol=1e-12)


def test_masked_apply_matches_subset_evaluation():
    from mfsim.closures import ClosureInputs, ClosureTerms, bubble

    np_bk, jx = _both()
    n = 6
    rng = np.random.default_rng(0)
    aG = rng.uniform(0.05, 0.4, n)
    A = np.pi * 0.05**2 / 4
    inp = ClosureInputs(
        alphaG=aG,
        alphaL=1 - aG,
        vL=np.full(n, 0.5),
        vLS=0.5 * (1 - aG),
        vG=np.full(n, 0.9),
        vGS=0.9 * aG,
        AreaL=A * (1 - aG),
        AreaG=A * aG,
        D=np.full(n, 0.05),
        eps=np.full(n, 1e-5),
        theta=np.full(n, 0.5),
        rhoL=np.full(n, 900.0),
        rhoG=np.full(n, 6.0),
        muL=np.full(n, 2e-3),
        muG=np.full(n, 1.8e-5),
        CL=np.full(n, 1e-6),
        CG=np.full(n, 1e-5),
        sigma=np.full(n, 0.05),
        beta=np.full(n, 2 * np.pi),
        hD=np.full(n, 1.0),
    )
    mask = np.array([True, False, True, True, False, False])
    out = np_bk.masked_apply(bubble, mask, inp, ClosureTerms)
    full = bubble(inp)
    for nm in out.names():
        assert np.array_equal(getattr(out, nm)[mask], getattr(full, nm)[mask]) and np.all(
            getattr(out, nm)[~mask] == 0.0
        )
