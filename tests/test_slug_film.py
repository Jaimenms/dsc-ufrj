"""Opção `closures.intermittent.unit = "film"` (issue 051): unidade de golfada com o filme de equilíbrio. O filme em
queda satisfaz a cinemática da unidade e a lei de Brotz (com derivada exata em JAX), a unidade conserva a massa e
devolve as vazões da célula, degenera no estado da célula nos limites, e o atrito da mistura com a parede é positivo
no estado rico em gás e rápido em que a unidade da tese o dava negativo."""

import numpy as np
import pytest

from mfsim.closures import ClosureInputs, equilibrium_fractions
from mfsim.closures.common import G_CLOSURE
from mfsim.closures.intermittent import BROTZ, falling_film_fraction, film_unit, intermittent
from mfsim.geometry import stratified_level

D = 0.1524


def test_falling_film_root():
    q = np.array([0.2, 0.6, 1.0, 1.5])
    ut = np.array([1.5, 2.0, 2.8, 4.0])
    th = np.full(4, np.pi / 2)
    a = falling_film_fraction(q, ut, np.full(4, D), th)
    uf = -BROTZ * np.sqrt(G_CLOSURE * D * (1 - np.sqrt(a)))
    np.testing.assert_allclose((ut - uf) * (1 - a), q, rtol=1e-9)
    assert np.all((a > 0.5) & (a < 1.0))


def test_falling_film_derivative_jax():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from mfsim.backend import get_backend

    bk = get_backend("jax")

    def f(q):
        return falling_film_fraction(q, jnp.full(3, 2.8), jnp.full(3, D), jnp.full(3, np.pi / 2), bk=bk)

    q0 = jnp.array([0.4, 0.9, 1.3])
    J = np.diag(np.asarray(jax.jacfwd(f)(q0)))
    h = 1e-6
    fd = (np.asarray(f(q0 + h)) - np.asarray(f(q0 - h))) / (2 * h)
    np.testing.assert_allclose(J, fd, rtol=1e-5)


def _unit(aG, theta=0.0, uLS=0.8, uGS=2.0, alphas0=0.25, a_strat=0.6):
    n = np.size(aG)
    aG = np.asarray(aG, float) * np.ones(n)
    ut = 1.2 * (uLS + uGS) + (0.35 * np.sin(theta) + 0.54 * np.cos(theta)) * np.sqrt(G_CLOSURE * D)
    out = film_unit(
        aG,
        np.full(n, alphas0),
        np.full(n, uLS),
        np.full(n, uGS),
        np.full(n, ut),
        np.full(n, theta),
        np.full(n, D),
        np.full(n, a_strat),
    )
    return aG, out


@pytest.mark.parametrize("theta", [0.0, -np.deg2rad(2.0), np.deg2rad(30.0), np.pi / 2])
def test_film_unit_mass_and_fluxes(theta):
    aG, (a_s, a_f, r, uL, ub, uf, uG, acc) = _unit(np.linspace(0.05, 0.95, 19), theta=theta)
    np.testing.assert_allclose(r * a_s + (1 - r) * a_f, aG, atol=1e-12)
    np.testing.assert_allclose(r * (1 - a_s) * uL + (1 - r) * (1 - a_f) * uf, 0.8, rtol=1e-9)
    np.testing.assert_allclose(r * a_s * ub + (1 - r) * a_f * uG, 2.0, rtol=1e-9)
    assert np.all(acc >= 0) and np.all(np.isfinite(acc))


def test_film_unit_limits():
    # célula mais pobre em gás do que o pistão: só pistão, com o estado da célula
    aG, (a_s, a_f, r, uL, ub, uf, uG, acc) = _unit([0.1], alphas0=0.25)
    assert r[0] == 1.0 and a_s[0] == pytest.approx(0.1)
    assert uL[0] == pytest.approx(0.8 / 0.9) and ub[0] == pytest.approx(2.0 / 0.1)
    assert acc[0] == 0.0
    # célula mais rica em gás do que o filme de equilíbrio: só filme, com o estado da célula
    aG, (a_s, a_f, r, uL, ub, uf, uG, acc) = _unit([0.8], a_strat=0.6)
    assert r[0] == 0.0 and a_f[0] == pytest.approx(0.8)
    assert uf[0] == pytest.approx(0.8 / 0.2) and uG[0] == pytest.approx(2.0 / 0.8)
    assert acc[0] == 0.0


def _gas_rich_fast_state(n=1):
    """Célula da linha horizontal do caso offshore a 120 bar com o choke aberto (fluido da Tabela 6): α_G 0,78,
    v_SL 1,9 m/s, v_SG 12 m/s, 15 bar."""
    rhoL, rhoG, P = 646.7, 11.3, 15e5
    A = np.pi * D**2 / 4
    aG, vLS, vGS = 0.78, 1.9, 12.0
    hD, beta = stratified_level(np.full(n, aG))
    one = np.ones(n)
    return ClosureInputs(
        alphaG=aG * one,
        alphaL=(1 - aG) * one,
        vL=vLS / (1 - aG) * one,
        vLS=vLS * one,
        vG=vGS / aG * one,
        vGS=vGS * one,
        AreaL=A * (1 - aG) * one,
        AreaG=A * aG * one,
        D=D * one,
        eps=6e-4 * one,
        theta=0.0 * one,
        rhoL=rhoL * one,
        rhoG=rhoG * one,
        muL=3.14e-4 * one,
        muG=1.21e-5 * one,
        CL=rhoL / (1000 * P) * one,
        CG=rhoG / P * one,
        sigma=0.0163 * one,
        beta=beta,
        hD=hD,
        bubble_drag=one,
        slug_frequency=one,
    )


def test_film_unit_gives_positive_wall_friction_where_thesis_unit_did_not():
    inp = _gas_rich_fast_state()
    eq = equilibrium_fractions(inp, smooth=True)
    t_film = intermittent(inp.replace(slug_film=np.ones(1), aG_film_eq=eq[2]))
    t_cons = intermittent(inp.replace(slug_unit=np.ones(1)))
    assert (t_film.GammaL + t_film.GammaG)[0] > 0.0
    assert (t_cons.GammaL + t_cons.GammaG)[0] < 0.0  # a unidade da tese (mesmo consistente) empurra a mistura


def test_film_unit_jax_matches_numpy():
    pytest.importorskip("jax")
    from mfsim.backend import get_backend

    bk = get_backend("jax")
    th = np.array([0.0, -0.035, 0.6, np.pi / 2])
    aG = np.array([0.3, 0.55, 0.45, 0.6])
    ut = np.array([3.0, 3.2, 3.5, 3.4])
    args = (aG, np.full(4, 0.2), np.full(4, 0.7), np.full(4, 1.8), ut, th, np.full(4, D), np.full(4, 0.62))
    out_n = film_unit(*args)
    out_j = film_unit(*[bk.xp.asarray(a) for a in args], bk=bk)
    for a, b in zip(out_j, out_n):
        np.testing.assert_allclose(np.asarray(a), b, rtol=1e-9, atol=1e-12)
