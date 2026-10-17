"""Critérios de bolhas com a fração de gás local (`regime.local_bubble`, issue 054): numa célula com 2 % de gás na
curva do riser o padrão deixa de depender da vazão de gás (com max(aG_bol, alphaG) passava de bolhas a
intermitente com vGS crescente, e o arrasto caía ~20 vezes: arrasto efetivo negativo); sem efeito com `local`
desligado; paridade numpy × JAX."""

import numpy as np
import pytest

from mfsim.closures import ClosureInputs, identify_regime
from mfsim.fluid import Fluid
from mfsim.geometry import stratified_level

FLUID = Fluid(rhoL0=850.0, rhoG0=1.2, muL=5e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.03)


def _inputs(aG, vLS=0.58, vGS=0.1, theta_deg=78.5, P=29e5, D=0.1524):
    shape = np.broadcast(np.atleast_1d(aG), np.atleast_1d(vGS)).shape
    aG = np.broadcast_to(np.asarray(aG, float), shape).copy()
    n = aG.size
    vGS = np.broadcast_to(np.asarray(vGS, float), (n,)).copy()
    rhoL, rhoG, CL, CG = FLUID.props(np.full(n, P))
    A = np.pi * D**2 / 4
    hD, beta = stratified_level(aG)
    one = np.ones(n)
    return ClosureInputs(
        alphaG=aG,
        alphaL=1 - aG,
        vL=vLS / (1 - aG) * one,
        vLS=vLS * one,
        vG=vGS / aG,
        vGS=vGS,
        AreaL=A * (1 - aG),
        AreaG=A * aG,
        D=D * one,
        eps=4.5e-5 * one,
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


def test_bubble_pattern_does_not_follow_the_gas_flow():
    vGS = np.linspace(0.1, 0.5, 9)
    inp = _inputs(0.02, vGS=vGS)
    base, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=True, local_film=True)
    loc, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=True, local_film=True, local_bubble=True)
    assert base.Bub[0] > 0.9 and base.Int[-1] > 0.9  # o defeito: 2 % de gás passa a intermitente com a vazão de gás
    assert np.all(loc.Bub > 0.9)  # com a fração local: bolhas em toda a faixa
    for f in (base, loc):
        assert np.allclose(f.Est + f.Anu + f.Bub + f.Int, 1.0)
    # uma célula com 40 % de gás continua intermitente (critério de 0,25 na fração da célula)
    hi, _ = identify_regime(_inputs(0.4, vGS=0.5), k_switch=20.0, smooth=True, local=True, local_bubble=True)
    assert hi.Int[0] > 0.9


def test_without_local_the_flag_does_nothing():
    inp = _inputs(np.linspace(0.02, 0.6, 7), vGS=0.3)
    a, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=False)
    b, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=False, local_bubble=True)
    for name in ("Est", "Anu", "Bub", "Int"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_jax_parity():
    pytest.importorskip("jax")
    from mfsim.backend import get_backend

    JAX = get_backend("jax")

    inp = _inputs(np.linspace(0.02, 0.9, 9), vGS=np.linspace(0.05, 2.0, 9))
    f_np, _ = identify_regime(inp, k_switch=20.0, smooth=True, local=True, local_bubble=True)
    inp_j = ClosureInputs(**{k: (JAX.asarray(v) if v is not None else None) for k, v in vars(inp).items()})
    f_jx, _ = identify_regime(inp_j, k_switch=20.0, smooth=True, local=True, local_bubble=True, bk=JAX)
    for name in ("Est", "Anu", "Bub", "Int"):
        np.testing.assert_allclose(np.asarray(getattr(f_jx, name)), getattr(f_np, name), atol=1e-12)
