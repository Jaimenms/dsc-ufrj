"""Opção `closures.intermittent.slug_frequency` (issue 049): a frequência de golfada de Zabaras (2000) confere com a
fórmula publicada (Gregory e Scott com o fator de inclinação), a omissão continua a ser a da tese (eq. 5.83 com o
expoente do código MATLAB) bit a bit, a opção só mexe onde o intermitente pesa, e a chave faz o caminho JSON."""

import numpy as np
import pytest

from mfsim.closures.common import ClosureInputs, G_CLOSURE
from mfsim.closures.intermittent import intermittent, slug_frequency, slug_frequency_zabaras
from mfsim.core.options import ModelSpec


def _gregory_scott(uLS, uGS, D):
    uM = uLS + uGS
    return 0.0226 * (uLS / (G_CLOSURE * D) * (19.75 / uM + uM)) ** 1.2


def test_zabaras_is_gregory_scott_with_inclination_factor():
    uLS, uGS, D = np.array([0.5, 0.2]), np.array([1.0, 3.0]), np.array([0.1, 0.1524])
    gs = _gregory_scott(uLS, uGS, D)
    horiz = slug_frequency_zabaras(uLS, uGS, D, np.zeros(2))
    vert = slug_frequency_zabaras(uLS, uGS, D, np.full(2, np.pi / 2))
    down = slug_frequency_zabaras(uLS, uGS, D, np.full(2, -np.deg2rad(2.0)))
    np.testing.assert_allclose(horiz, 0.836 * gs, rtol=1e-5)  # u_M com 1e-6 m/s
    np.testing.assert_allclose(vert, (0.836 + 2.75) * gs, rtol=1e-5)
    np.testing.assert_allclose(down, horiz, rtol=0, atol=0)  # descendente: fator do horizontal


def test_thesis_formula_uses_matlab_exponent():
    """n = N_f/(260 + 0,85 N_f) (frequenciaSlugs.m), não o 0,85 + 260/N_f do texto da tese."""
    uLS, uGS = np.array([0.3]), np.array([1.2])
    rhoL, rhoG, muL, D, sigma, eps = 700.0, 20.0, 1.2e-3, 0.1524, 0.02, 6e-4
    w = slug_frequency(uLS, uGS, rhoL, rhoG, muL, D, sigma, eps)
    from mfsim.friction import fanning

    fL, _ = fanning(uLS, rhoL, muL, D, eps)
    Eo = np.sqrt(G_CLOSURE * D**2 * (rhoL - rhoG) / sigma)
    Nf = D**1.5 * np.sqrt(rhoL * (rhoL - rhoG) * G_CLOSURE) / muL
    base = 0.0323 / 2 * uLS / uGS * np.sqrt(rhoL / (rhoL - rhoG)) * fL**-0.5 / Eo**0.2
    np.testing.assert_allclose(w, uGS / D * base ** (Nf / (260 + 0.85 * Nf)), rtol=1e-12)


def _inp(theta, flag):
    n = theta.size
    aG = np.full(n, 0.4)
    A = np.pi * 0.1524**2 / 4
    vL, vG = np.full(n, 0.5), np.full(n, 2.0)
    return ClosureInputs(
        alphaG=aG,
        alphaL=1 - aG,
        vL=vL,
        vLS=vL * (1 - aG),
        vG=vG,
        vGS=vG * aG,
        AreaL=A * (1 - aG),
        AreaG=A * aG,
        D=np.full(n, 0.1524),
        eps=np.full(n, 6e-4),
        theta=theta,
        rhoL=np.full(n, 700.0),
        rhoG=np.full(n, 20.0),
        muL=np.full(n, 1.2e-3),
        muG=np.full(n, 1.2e-5),
        CL=np.full(n, 700.0 / (1000 * 20e5)),
        CG=np.full(n, 20.0 / 20e5),
        sigma=np.full(n, 0.02),
        beta=np.full(n, np.pi),
        hD=np.full(n, 0.5),
        bubble_drag=np.ones(n),
        slug_frequency=flag,
    )


def test_option_changes_the_slug_unit_closure():
    th = np.array([0.0, np.pi / 2])
    t_thesis = intermittent(_inp(th, None))
    t_zab = intermittent(_inp(th, np.ones(2)))
    assert np.all(np.isfinite(t_zab.GammaL)) and np.all(np.isfinite(t_zab.GammaG))
    assert not np.allclose(t_thesis.GammaG, t_zab.GammaG)


def test_option_json_roundtrip_and_default():
    assert ModelSpec().slug_frequency == "thesis"
    assert "intermittent" not in ModelSpec().to_json()["closures"]  # a omissão não aparece no JSON
    m = ModelSpec.from_json({"closures": {"intermittent": {"slug_frequency": "zabaras"}}})
    assert m.slug_frequency == "zabaras"
    assert ModelSpec.from_json(m.to_json()) == m
    _, _, extra = m.to_legacy()
    assert extra["slug_frequency"] == "zabaras"
    assert ModelSpec.from_legacy(*m.to_legacy()) == m


def test_zabaras_jax_matches_numpy():
    pytest.importorskip("jax")
    from mfsim.backend import get_backend

    bk = get_backend("jax")
    uLS, uGS, D, th = np.array([0.0, 0.3]), np.array([0.0, 1.5]), np.full(2, 0.1524), np.array([0.0, np.pi / 2])
    a = np.asarray(slug_frequency_zabaras(bk.xp.asarray(uLS), bk.xp.asarray(uGS), bk.xp.asarray(D), th, bk=bk))
    np.testing.assert_allclose(a, slug_frequency_zabaras(uLS, uGS, D, th), rtol=1e-12)
    assert a[0] == 0.0  # duto parado
