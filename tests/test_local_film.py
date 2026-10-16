"""Ponte do filme do anular avaliada também com a fração de líquido local (`regime.local_film`, issue 046): uma
célula quase cheia de líquido com velocidades superficiais pequenas deixa de ser anular; sem efeito com
`local` desligado; paridade numpy × JAX."""

import numpy as np
import pytest

from mfsim.closures import ClosureInputs, identify_regime
from mfsim.fluid import Fluid
from mfsim.geometry import stratified_level

FLUID = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)


def _inputs(aG, vLS=0.25, vGS=0.06, theta_deg=-3.0, P=7e5, D=0.05):
    aG = np.atleast_1d(np.asarray(aG, float))
    n = aG.size
    rhoL, rhoG, CL, CG = FLUID.props(np.full(n, P))
    A = np.pi * D**2 / 4
    hD, beta = stratified_level(aG)
    one = np.ones(n)
    return ClosureInputs(
        alphaG=aG,
        alphaL=1 - aG,
        vL=vLS / (1 - aG) * one,
        vLS=vLS * one,
        vG=vGS / aG * one,
        vGS=vGS * one,
        AreaL=A * (1 - aG),
        AreaG=A * aG,
        D=D * one,
        eps=1e-5 * one,
        theta=np.deg2rad(theta_deg) * one,
        rhoL=rhoL,
        rhoG=rhoG,
        muL=FLUID.muL * one,
        muG=FLUID.muG * one,
        CL=CL,
        CG=CG,
        sigma=FLUID.sigma * one,
        beta=beta,
        hD=hD,
    )


def test_liquid_filled_cell_is_not_annular():
    inp = _inputs([0.09])
    base, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=True)
    film, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=True, local_film=True)
    assert base.Anu[0] > 0.5  # o defeito: célula com 91 % de líquido como anular
    assert film.Anu[0] < 0.05  # com a fração local: não é anular
    for f in (base, film):
        assert np.allclose(f.Est + f.Anu + f.Bub + f.Int, 1.0)
    # num riser com 95 % de gás a célula continua anular (o filme local é fino)
    riser = _inputs([0.95], theta_deg=90.0)
    f, _ = identify_regime(riser, k_switch=20.0, smooth=True, local=True, local_film=True)
    assert f.Anu[0] > 0.5


def test_without_local_the_flag_does_nothing():
    inp = _inputs(np.linspace(0.05, 0.9, 7))
    a, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=False)
    b, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=False, local_film=True)
    for name in ("Est", "Anu", "Bub", "Int"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_jax_parity():
    pytest.importorskip("jax")
    from mfsim.backend import get_backend

    JAX = get_backend("jax")

    inp = _inputs(np.linspace(0.02, 0.97, 9))
    f_np, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=True, local_film=True)
    inp_j = ClosureInputs(**{k: (JAX.asarray(v) if v is not None else None) for k, v in vars(inp).items()})
    f_jx, _ = identify_regime(inp_j, k_switch=20.0, smooth=True, local=True, local_film=True, bk=JAX)
    for name in ("Est", "Anu", "Bub", "Int"):
        np.testing.assert_allclose(np.asarray(getattr(f_jx, name)), getattr(f_np, name), atol=1e-12)
