"""Opção bubble_drag="fisico" do fechamento de bolhas: na velocidade terminal de Mendelson o arrasto iguala o empuxo
(escorregamento de equilíbrio ~0.25 m/s para ar-água), o fechamento da tese dá escorregamento muito maior nas condições
de laboratório, e o caminho JAX reproduz o numpy com a opção ligada."""

import numpy as np
import pytest

from mfsim.closures.bubble import bubble
from mfsim.closures.common import ClosureInputs, G_CLOSURE


def _inp(aG, slip, D=0.0254, rhoL=998.0, rhoG=1.2, sigma=0.072, phys=1.0):
    n = 1
    aG = np.full(n, aG); aL = 1 - aG; A = np.pi * D**2 / 4
    vL = np.full(n, 0.2); vG = vL + slip
    P = 1.013e5
    return ClosureInputs(alphaG=aG, alphaL=aL, vL=vL, vLS=vL * aL, vG=vG, vGS=vG * aG, AreaL=A * aL, AreaG=A * aG,
                         D=np.full(n, D), eps=np.full(n, 1e-5), theta=np.full(n, np.pi / 2), rhoL=np.full(n, rhoL), rhoG=np.full(n, rhoG),
                         muL=np.full(n, 1e-3), muG=np.full(n, 1.8e-5), CL=np.full(n, rhoL / (1000 * P)), CG=np.full(n, rhoG / P),
                         sigma=np.full(n, sigma), beta=np.full(n, 2 * np.pi), hD=np.full(n, 1.0), bubble_drag=np.full(n, phys))


def _equilibrium_slip(phys):
    """escorregamento em que o arrasto interfacial iguala o empuxo das bolhas (bisseção)"""
    def resid(s):
        inp = _inp(0.03, s, phys=phys); t = bubble(inp)
        buoy = inp.alphaG[0] * (inp.rhoL[0] - inp.rhoG[0]) * G_CLOSURE * (inp.AreaL[0] + inp.AreaG[0])
        return buoy - t.GammaG[0]          # GammaG > 0 é o arrasto que trava o gás; positivo = o empuxo vence
    lo, hi = 1e-3, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def test_physical_drag_terminal_velocity_is_mendelson():
    inp = _inp(0.03, 0.3, phys=1.0)
    db = np.sqrt(16 * inp.sigma[0] / G_CLOSURE / (inp.rhoL[0] - inp.rhoG[0]))
    vb = np.sqrt(2.14 * inp.sigma[0] / (inp.rhoL[0] * db) + 0.505 * G_CLOSURE * db)
    assert 0.2 < vb < 0.35
    assert abs(_equilibrium_slip(1.0) - vb) < 2e-2 * vb      # arrasto(vb) = empuxo por construção (db = min(db, dbmax))


def test_thesis_drag_gives_large_slip_at_atmospheric_conditions():
    s_tese, s_fis = _equilibrium_slip(0.0), _equilibrium_slip(1.0)
    assert s_tese > 3.0 and s_fis < 0.35                       # ~4 m/s contra ~0.26 m/s


def test_bubble_drag_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from mfsim.jax_physics import bubble as bubble_j
    for phys in (0.0, 1.0):
        inp = _inp(0.1, 0.5, phys=phys)
        t_np = bubble(inp)
        d = {k: jnp.asarray(getattr(inp, k)) for k in ("alphaG", "alphaL", "vL", "vLS", "vG", "vGS", "AreaL", "AreaG", "D", "eps", "theta", "rhoL", "rhoG", "muL", "muG", "CL", "CG", "sigma", "beta", "hD", "bubble_drag")}
        t_j = bubble_j(d)
        assert np.allclose(np.asarray(t_j["GammaG"]), t_np.GammaG, rtol=1e-10, atol=1e-12)
        assert np.allclose(np.asarray(t_j["GammaL"]), t_np.GammaL, rtol=1e-10, atol=1e-12)
