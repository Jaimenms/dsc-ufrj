"""Opção `closures.intermittent.unit = "consistent"` (issue 050): a unidade de golfada conserva a massa
(alphaG lu = alphas ls + alphaf lf) para qualquer fração de gás da célula, reduz-se ao estado da célula nos limites
rico em líquido e rico em gás, as velocidades locais reproduzem as vazões da célula, e a da tese fica igual."""

import numpy as np
import pytest

from mfsim.closures.intermittent import ALPHA_FILM_MAX, slug_unit
from mfsim.core.options import ModelSpec

AG = np.array([0.001, 0.01, 0.05, 0.2, 0.45, 0.6, 0.8, 0.94, 0.97, 0.999])


def _unit(consistent, aG=AG, alphas=0.03, ls=17.4, lu=12.0):
    n = aG.size
    return slug_unit(aG, np.full(n, alphas), np.full(n, ls), np.full(n, lu), consistent=consistent)


def test_consistent_unit_conserves_mass():
    for lu in (5.0, 12.0, 30.0, 300.0):
        a_s, ls, lf, lu_, af = _unit(True, lu=lu)
        np.testing.assert_allclose(a_s * ls + af * lf, AG * lu_, rtol=1e-9, atol=1e-9)
        assert np.all(a_s <= AG + 1e-15) and np.all(ls >= 0) and np.all(lf > 0)
        assert np.all(af[AG <= ALPHA_FILM_MAX] <= ALPHA_FILM_MAX + 1e-12)
        np.testing.assert_allclose(ls + lf, lu_, rtol=1e-12)


def test_thesis_unit_breaks_mass_when_cell_is_gas_rich():
    a_s, ls, lf, lu, af = _unit(False, lu=12.0)  # lu < ls: unidade só pistão, filme cortado
    err = np.abs(a_s * ls + af * lf - AG * lu)
    m = AG > 0.2
    assert np.all(err[m] > 0.1 * AG[m] * lu[m])  # o corte do filme perde mais de 10 % do gás da célula


def test_limits_reduce_to_the_cell_state():
    a_s, ls, lf, lu, af = _unit(True)
    rich_liquid = AG < 0.03
    np.testing.assert_allclose(a_s[rich_liquid], AG[rich_liquid])
    np.testing.assert_allclose(af[rich_liquid], AG[rich_liquid])
    rich_gas = AG > ALPHA_FILM_MAX
    np.testing.assert_allclose(ls[rich_gas], 0.0)
    np.testing.assert_allclose(af[rich_gas], AG[rich_gas])


def test_local_velocities_reproduce_cell_fluxes():
    """Com a unidade consistente, as velocidades locais (relações da tese) devolvem as vazões da célula."""
    aG = np.array([0.1, 0.3, 0.5, 0.8])
    uLS, uGS, ut = 0.6, 1.4, 2.5
    a_s, ls, lf, lu, af = slug_unit(aG, np.full(4, 0.05), np.full(4, 17.4), np.full(4, 20.0), consistent=True)
    Rs, Rf = 1 - a_s, 1 - af
    uL = (uLS + ut * (aG - a_s)) / Rs
    ub = (uGS - ut * (aG - a_s)) / a_s
    uf = ut - (ut - uL) * Rs / Rf
    uG = ut - (ut - ub) * a_s / af
    np.testing.assert_allclose((ls * Rs * uL + lf * Rf * uf) / lu, uLS, rtol=1e-10)
    np.testing.assert_allclose((ls * a_s * ub + lf * af * uG) / lu, uGS, rtol=1e-10)


def test_option_json_roundtrip_and_default():
    assert ModelSpec().slug_unit == "thesis"
    m = ModelSpec.from_json({"closures": {"intermittent": {"unit": "consistent", "slug_frequency": "zabaras"}}})
    assert (m.slug_unit, m.slug_frequency) == ("consistent", "zabaras")
    assert ModelSpec.from_json(m.to_json()) == m
    assert ModelSpec.from_legacy(*m.to_legacy()) == m
    assert "intermittent" not in ModelSpec().to_json()["closures"]


def test_consistent_unit_jax_matches_numpy():
    pytest.importorskip("jax")
    from mfsim.backend import get_backend

    bk = get_backend("jax")
    out_j = slug_unit(
        bk.xp.asarray(AG), bk.xp.full(AG.size, 0.03), bk.xp.full(AG.size, 17.4), bk.xp.full(AG.size, 12.0), True, bk=bk
    )
    out_n = _unit(True)
    for a, b in zip(out_j, out_n):
        np.testing.assert_allclose(np.asarray(a), b, rtol=1e-12, atol=1e-15)
