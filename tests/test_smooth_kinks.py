"""Opção `closures.kinks = "smooth"` (issue 052): as versões C1 das quinas dos fechamentos (diâmetro de bolha,
escorregamento de Stuhmiller, Andritsos & Hanratty, Kowalski, módulos das velocidades nos equilíbrios) coincidem com
as da tese longe das quinas, têm derivada contínua através delas, a omissão (`sharp`) não muda nada e a chave faz o
caminho JSON."""

import numpy as np
import pytest

from mfsim.closures import ClosureInputs
from mfsim.closures.annular import annular
from mfsim.closures.bubble import bubble
from mfsim.closures.common import smooth_abs, smooth_max, smooth_min
from mfsim.closures.stratified import interfacial_factor_stratified, stratified
from mfsim.core.options import ModelSpec
from mfsim.geometry import stratified_level

D = 0.1524


def _inp(aG, vLS, vGS, theta=0.0, smooth=None):
    n = aG.size
    A = np.pi * D**2 / 4
    rhoL, rhoG, P = 646.7, 15.0, 20e5
    hD, beta = stratified_level(aG)
    one = np.ones(n)
    return ClosureInputs(
        alphaG=aG, alphaL=1 - aG, vL=vLS / (1 - aG), vLS=vLS, vG=vGS / aG, vGS=vGS, AreaL=A * (1 - aG), AreaG=A * aG,
        D=D * one, eps=6e-4 * one, theta=theta * one, rhoL=rhoL * one, rhoG=rhoG * one, muL=3.14e-4 * one,
        muG=1.21e-5 * one, CL=rhoL / (1000 * P) * one, CG=rhoG / P * one, sigma=0.0163 * one, beta=beta, hD=hD,
        bubble_drag=one, smooth_kinks=smooth,
    )  # fmt: skip


def _no_kink(x, y, tol):
    """A derivada numérica não salta mais do que `tol` (relativo) entre pontos vizinhos."""
    d = np.diff(y) / np.diff(x)
    jumps = np.abs(np.diff(d)) / (np.abs(d[:-1]) + np.abs(d[1:]) + 1e-12)
    return float(jumps.max()) < tol


def test_smooth_helpers_agree_away_from_the_kink():
    a, b = np.array([0.0, 1.0, 5.0]), np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(smooth_max(a, b, 0.01), np.maximum(a, b), atol=0.01)
    np.testing.assert_allclose(smooth_min(a, b, 0.01), np.minimum(a, b), atol=0.01)
    np.testing.assert_allclose(smooth_abs(np.array([-3.0, 2.0]), 0.05), [3.0, 2.0], rtol=5e-4)


def test_andritsos_hanratty_is_smooth_at_v_gt():
    v = np.linspace(4.5, 6.5, 801)
    hD = np.full(v.size, 0.4)
    fG = np.full(v.size, 0.005)
    sharp = interfacial_factor_stratified(fG, v, hD)
    smooth = interfacial_factor_stratified(fG, v, hD, smooth=np.ones(v.size))
    assert not _no_kink(v, sharp, 0.5)
    assert _no_kink(v, smooth, 0.05)
    far = (v < 5.0) | (v > 6.0)
    np.testing.assert_allclose(smooth[far], sharp[far], rtol=0.02)


def test_bubble_closure_smooth_in_slip():
    n = 801
    aG = np.full(n, 0.3)
    vLS = np.full(n, 0.5)
    vGS = np.linspace(0.3, 2.0, n)  # o escorregamento cruza os 3 m/s do corte de Stuhmiller
    t_sharp = bubble(_inp(aG, vLS, vGS))
    t_smooth = bubble(_inp(aG, vLS, vGS, smooth=np.ones(n)))
    assert not _no_kink(vGS, t_sharp.dPL, 0.5)
    assert _no_kink(vGS, t_smooth.dPL, 0.05)


def test_annular_and_stratified_smooth_variants_finite():
    n = 201
    aG = np.linspace(0.3, 0.95, n)
    for f in (annular, stratified):
        t = f(_inp(aG, np.full(n, 0.4), np.full(n, 6.0), smooth=np.ones(n)))
        assert np.all(np.isfinite(t.GammaL)) and np.all(np.isfinite(t.GammaG))


def test_sharp_default_is_unchanged_and_json_roundtrip():
    n = 11
    aG = np.linspace(0.2, 0.8, n)
    a = bubble(_inp(aG, np.full(n, 0.5), np.full(n, 1.5)))
    b = bubble(_inp(aG, np.full(n, 0.5), np.full(n, 1.5), smooth=None))
    np.testing.assert_array_equal(a.GammaL, b.GammaL)
    assert ModelSpec().kinks == "sharp" and "kinks" not in ModelSpec().to_json()["closures"]
    m = ModelSpec.from_json({"closures": {"kinks": "smooth"}})
    assert m.kinks == "smooth" and ModelSpec.from_json(m.to_json()) == m
    assert ModelSpec.from_legacy(*m.to_legacy()) == m


def test_smooth_variants_jax_match_numpy():
    pytest.importorskip("jax")
    from mfsim.backend import get_backend

    bk = get_backend("jax")
    n = 7
    inp = _inp(np.linspace(0.2, 0.9, n), np.full(n, 0.4), np.linspace(1.0, 8.0, n), smooth=np.ones(n))
    inp_j = inp.replace(**{f: bk.xp.asarray(getattr(inp, f)) for f in ("alphaG", "alphaL", "vL", "vLS", "vG", "vGS")})
    for f in (bubble, stratified, annular):
        a, b = f(inp), f(inp_j, bk=bk)
        np.testing.assert_allclose(np.asarray(b.GammaL), a.GammaL, rtol=1e-10, atol=1e-9)
