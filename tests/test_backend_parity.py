"""Uma implementação, dois backends: os fechamentos, os equilíbrios, a identificação de
padrão e a ponderação dão com `bk=JAX` os mesmos valores que com numpy (e o caminho JAX
traça/compila sem ramos em valores)."""
import numpy as np
import pytest

from mfsim.backend import NUMPY, get_backend
from mfsim.closures import ClosureInputs, ClosureTerms, RegimeFlags, annular, blend_closures, bubble, equilibrium_fractions, identify_regime, intermittent, stratified
from mfsim.closures.regime import closures_used, fixed_regime
from mfsim.geometry import stratified_level

jax = pytest.importorskip("jax")
JAX = get_backend("jax")


def _inputs(n=48, seed=0, D=0.05):
    rng = np.random.default_rng(seed)
    aG = rng.uniform(0.02, 0.95, n); A = np.pi * D**2 / 4
    vLS = rng.uniform(-0.5, 2.0, n); vGS = rng.uniform(-0.5, 5.0, n)
    th = rng.uniform(-np.pi / 2, np.pi / 2, n)
    P = rng.uniform(2e5, 60e5, n); rhoL = 900 * (P / 1e5) ** 0.01; rhoG = 1.2 * P / 1e5
    hD, beta = stratified_level(aG)
    return ClosureInputs(alphaG=aG, alphaL=1 - aG, vL=vLS / (1 - aG), vLS=vLS, vG=vGS / aG, vGS=vGS, AreaL=A * (1 - aG), AreaG=A * aG,
                         D=np.full(n, D), eps=np.full(n, 1e-5), theta=th, rhoL=rhoL, rhoG=rhoG, muL=np.full(n, 2e-3), muG=np.full(n, 1.8e-5),
                         CL=rhoL / (100 * P), CG=rhoG / (1.05 * P), sigma=np.full(n, 0.05), beta=beta, hD=hD,
                         fi_mult=np.ones(n), fi_vgt=np.full(n, 5.5), bubble_drag=np.zeros(n))


def _to_jax(inp):
    return ClosureInputs(**{k: (JAX.asarray(v) if v is not None else None) for k, v in inp.__dict__.items()})


def _close(a, b, what, rtol=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), f"{what}: finitude difere"
    np.testing.assert_allclose(a[ok], b[ok], rtol=rtol, atol=1e-12 * (1 + np.max(np.abs(b[ok]), initial=0.0)), err_msg=what)


@pytest.mark.parametrize("func", [stratified, annular, bubble, intermittent])
def test_closures_jax_vs_numpy(func):
    inp = _inputs(); inpj = _to_jax(inp)
    S_np = func(inp); S_jx = func(inpj, bk=JAX)
    for nm in S_np.names():
        _close(getattr(S_jx, nm), getattr(S_np, nm), f"{func.__name__}.{nm}")
    # compila (sem ramos em valores)
    S_jit = jax.jit(lambda i: func(i, bk=JAX))(inpj)
    _close(S_jit.GammaL, S_np.GammaL, f"{func.__name__} jit")


def test_stratified_level_jax_vs_numpy():
    aG = np.concatenate([np.array([0.0, 1e-7, 0.5, 1 - 1e-7, 1.0]), np.random.default_rng(3).uniform(1e-3, 0.999, 40)])
    hD, b = stratified_level(aG)
    hDj, bj = stratified_level(JAX.asarray(aG), bk=JAX)
    _close(hDj, hD, "hD", rtol=1e-12); _close(bj, b, "beta", rtol=1e-12)


@pytest.mark.parametrize("smooth,local", [(False, False), (True, False), (True, True)])
def test_regime_identification_jax_vs_numpy(smooth, local):
    inp = _inputs(seed=1); inpj = _to_jax(inp)
    eq = equilibrium_fractions(inp, smooth=smooth); eqj = equilibrium_fractions(inpj, smooth=smooth, bk=JAX)
    for a, b, nm in zip(eqj, eq, ("hD_est", "hD_anu", "aG_est", "aG_anu", "aG_bol")):
        _close(a, b, nm, rtol=1e-12)
    k = 20.0 if smooth else 1e6
    fl, fac = identify_regime(inp, k_switch=k, eq=eq, smooth=smooth, local=local)
    flj, facj = identify_regime(inpj, k_switch=k, eq=eqj, smooth=smooth, local=local, bk=JAX)
    for nm in fl.names():
        _close(getattr(flj, nm), getattr(fl, nm), nm, rtol=1e-10)
    S = blend_closures(inp, fl, threshold=1e-6 if smooth else 0.1)
    Sj = blend_closures(inpj, flj, threshold=1e-6 if smooth else 0.1, bk=JAX)
    for nm in S.names():
        _close(getattr(Sj, nm), getattr(S, nm), f"blend.{nm}")
    # jit e jacfwd finitos
    def f(i):
        e = equilibrium_fractions(i, smooth=smooth, bk=JAX)
        g, _ = identify_regime(i, k_switch=k, eq=e, smooth=smooth, local=local, bk=JAX)
        return blend_closures(i, g, threshold=1e-6 if smooth else 0.1, bk=JAX).GammaL
    out = jax.jit(f)(inpj)
    _close(out, S.GammaL, "jit blend")
    J = jax.jacfwd(lambda a: f(inpj.replace(alphaG=a, alphaL=1 - a)))(inpj.alphaG)
    assert np.all(np.isfinite(np.asarray(J)))


def test_fixed_regime_blend_with_used_set():
    inp = _inputs(seed=2); inpj = _to_jax(inp)
    fl = fixed_regime(4, inp.size); flj = fixed_regime(4, inp.size, bk=JAX)
    used = closures_used(4, 2)
    S = blend_closures(inp, fl, used=used); Sj = blend_closures(inpj, flj, used=used, bk=JAX)
    for nm in S.names():
        _close(getattr(Sj, nm), getattr(S, nm), nm)
    assert np.array_equal(S.GammaL, stratified(inp).GammaL)
