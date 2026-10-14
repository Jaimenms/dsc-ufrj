"""Identificação automática de padrão (regime 6) do modelo a dois fluidos em JAX:
frações de equilíbrio (Taitel/Pots), indicadores com chaveamento sigmoidal e os
quatro fechamentos ponderados reproduzem o numpy; a Jacobiana automática é finita
nos estados do severe slug (vazões de gás nulas, frações nos limites)."""

import copy

import numpy as np
import pytest

from mfsim.cases import load_chapter10_case
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.closures import ClosureInputs, equilibrium_fractions, identify_regime, intermittent, annular
from mfsim.fluid import Fluid
from mfsim.simulation import build_model

jax = pytest.importorskip("jax")
from mfsim.backend import get_backend  # noqa: E402

JAX = get_backend("jax")


def _severe_model(regime=6, **extra):
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=1000.0, gammaG=1.05, P0=1e5, sigma=0.05)
    D = 0.05
    A = np.pi * D**2 / 4
    qL, qG = 0.3 * 900 * A, 0.1 * fluid.rhoG(5e5) * A
    case = severe_slug_case(
        method="fv_upwind",
        tf=100,
        dt_out=2.0,
        qL=qL,
        qG=qG,
        D=D,
        P_out_bar=4.0,
        L_h1=20.0,
        L_down=300.0,
        slope_down_deg=-3.0,
        H_riser=20.0,
        L_h2=5.0,
        n_h1=2,
        n_down=8,
        n_riser=4,
        n_h2=2,
        regime=regime,
        single_pipe=True,
        fluid=fluid,
        roughness=1e-5,
        start="blocked",
        elbow_radius=5.0,
        n_elbow=3,
        extra={"conservative": True, "art_visc": 0.05, **extra},
    )
    return build_model(case)


def _random_inputs(rng, n):
    """Entradas de fechamento aleatórias (inclinações de -5 a 90 graus, vazões de gás incluindo zero)."""
    aG = rng.uniform(0.02, 0.98, n)
    theta = np.deg2rad(rng.choice([-3.0, 0.0, 30.0, 90.0], n))
    rhoL, rhoG = np.full(n, 900.0), rng.uniform(1.0, 8.0, n)
    vLS = rng.uniform(-0.1, 1.0, n)
    vGS = rng.uniform(-0.05, 2.0, n)
    vGS[: n // 4] = 0.0
    D = np.full(n, 0.05)
    from mfsim.geometry import stratified_level

    hD, beta = stratified_level(aG)
    return ClosureInputs(
        alphaG=aG,
        alphaL=1 - aG,
        vL=vLS / (1 - aG),
        vLS=vLS,
        vG=vGS / aG,
        vGS=vGS,
        AreaL=np.pi * D**2 / 4 * (1 - aG),
        AreaG=np.pi * D**2 / 4 * aG,
        D=D,
        eps=np.full(n, 1e-5),
        theta=theta,
        rhoL=rhoL,
        rhoG=rhoG,
        muL=np.full(n, 2e-3),
        muG=np.full(n, 1.8e-5),
        CL=rhoL / (5e5 * 1000.0),
        CG=rhoG / (5e5 * 1.05),
        sigma=np.full(n, 0.05),
        beta=beta,
        hD=hD,
    )


def _as_jax(inp):
    return ClosureInputs(**{k: (JAX.asarray(v) if v is not None else None) for k, v in inp.__dict__.items()})


def test_equilibrium_and_flags_match_numpy():
    rng = np.random.default_rng(0)
    inp = _random_inputs(rng, 64)
    inp_j = _as_jax(inp)
    eq_np = equilibrium_fractions(inp)
    eq_j = equilibrium_fractions(inp_j, bk=JAX)
    for a, b in zip(eq_np, eq_j):
        assert np.allclose(a, np.asarray(b), rtol=1e-12, atol=1e-14)
    fl_np, _ = identify_regime(inp, k_switch=1e6, eq=eq_np)
    fl_j, _ = identify_regime(inp_j, k_switch=1e6, eq=eq_j, bk=JAX)
    for nm in ("Est", "EsS", "EsW", "Anu", "Bub", "Int"):
        assert np.allclose(getattr(fl_np, nm), np.asarray(getattr(fl_j, nm)), rtol=1e-10, atol=1e-12), nm
    # chaveamento suave (k_switch baixo): também reproduz
    fl_np, _ = identify_regime(inp, k_switch=50.0, eq=eq_np)
    fl_j, _ = identify_regime(inp_j, k_switch=50.0, eq=eq_j, bk=JAX)
    for nm in ("Est", "Anu", "Bub", "Int"):
        assert np.allclose(getattr(fl_np, nm), np.asarray(getattr(fl_j, nm)), rtol=1e-10, atol=1e-12), nm


@pytest.mark.parametrize("func_np, name", [(intermittent, "intermittent"), (annular, "annular")])
def test_closures_match_numpy(func_np, name):
    """Fechamentos que os casos do capítulo 10 com padrão fixo não exercitam em todas as
    inclinações (intermitente, anular) reproduzem o numpy em entradas aleatórias."""
    rng = np.random.default_rng(1)
    inp = _random_inputs(rng, 48)
    inp = inp.replace(
        vLS=np.abs(inp.vLS) + 0.01,
        vGS=np.abs(inp.vGS) + 0.01,
        vL=(np.abs(inp.vLS) + 0.01) / inp.alphaL,
        vG=(np.abs(inp.vGS) + 0.01) / inp.alphaG,
    )
    S_np = func_np(inp)
    S_j = func_np(_as_jax(inp), bk=JAX)
    for nm in ("GammaL", "GammaG", "dPL", "dPG", "ddPLdalphaG", "ddPGdalphaG", "ddPLdP", "ddPGdP"):
        a, b = getattr(S_np, nm), np.asarray(getattr(S_j, nm))
        ok = np.isfinite(a)
        assert np.allclose(a[ok], b[ok], rtol=1e-9, atol=1e-12 * (1 + np.max(np.abs(a[ok])))), nm


def test_regime_auto_rhs_matches_numpy():
    from mfsim.compiled import build_jax_rhs

    model = _severe_model()
    rhs_j, jac_j = build_jax_rhs(model)
    X0 = model.initial_state()
    rng = np.random.default_rng(0)
    pb = model.pipes[0]
    slots = pb.free % 4
    for k in range(6):
        x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))
        x[slots == 0] = np.clip(x[slots == 0] + 0.4 * rng.random((slots == 0).sum()), 0.0, 1.0)
        x[slots == 3] = x[slots == 3] + 0.002 * rng.standard_normal((slots == 3).sum())
        f_np = model.rhs(0.0, x)
        f_jx = rhs_j(0.0, x)
        assert np.all(np.isfinite(f_np))
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
        assert np.all(np.isfinite(jac_j(0.0, x)))
    # estado inicial do bloqueio: vazões de gás exatamente nulas no riser e frações em 0
    assert np.max(np.abs(model.rhs(0.0, X0) - rhs_j(0.0, X0))) < 1e-9
    assert np.all(np.isfinite(jac_j(0.0, X0)))


def test_regime_auto_nodal_schemes():
    """Regime 6 nos esquemas nodais da tese (10.1.1 com identificação automática)."""
    from mfsim.compiled import build_jax_rhs

    case, raw = load_chapter10_case("10.1.1")
    c = copy.deepcopy(case)
    c.set_options(regime=6)
    model = build_model(c)
    rhs_j, jac_j = build_jax_rhs(model)
    X0 = model.initial_state()
    f_np = model.rhs(1.0, X0)
    assert np.all(np.isfinite(f_np))
    assert np.max(np.abs(f_np - rhs_j(1.0, X0))) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
    assert np.all(np.isfinite(jac_j(1.0, X0)))


def test_sigmoid_tangent_finite_at_clip_boundary():
    """Com k = 1e6 o argumento da sigmoide satura (|k (x1 - x2)| > 700) a 7e-4 do limiar;
    a tangente tem de ser finita também aí (a forma 1/(1+exp(clip)) dá inf/inf)."""
    from mfsim.closures.regime import sigmoid

    for dx in (-7.0e-4, -6.9e-4, 6.9e-4, 7.0e-4, 1e-2):
        v, dv = jax.jvp(lambda x: sigmoid(1e6, x, 0.0, bk=JAX), (dx,), (1e7,))
        assert np.isfinite(float(v)) and np.isfinite(float(dv))
    # valores iguais aos do numpy longe do underflow
    x = np.linspace(-1e-3, 1e-3, 11)
    assert np.allclose(
        np.asarray(sigmoid(1e6, JAX.asarray(x), 0.0, bk=JAX)), sigmoid(1e6, x, 0.0), atol=1e-300, rtol=1e-12
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"low_void_regime": 2},
        {"ipe_smooth": True, "k_switch": 20.0, "low_void_regime": 2},
        {"ipe_smooth": True, "ipe_local": True, "k_switch": 20.0, "low_void_regime": 2},
    ],
)
def test_regime_auto_options_match_numpy(extra):
    """Opções do modo automático (mistura de fração baixa, chaveamento relativo): JAX = numpy
    e Jacobiana finita; com as opções os pesos somam 1 em todos os nós."""
    from mfsim.compiled import build_jax_rhs

    model = _severe_model(**extra)
    rhs_j, jac_j = build_jax_rhs(model)
    X0 = model.initial_state()
    rng = np.random.default_rng(5)
    pb = model.pipes[0]
    slots = pb.free % 4
    for k in range(4):
        x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))
        x[slots == 0] = np.clip(x[slots == 0] + 0.4 * rng.random((slots == 0).sum()), 0.0, 1.0)
        f_np = model.rhs(0.0, x)
        f_jx = rhs_j(0.0, x)
        assert np.all(np.isfinite(f_np))
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))
        assert np.all(np.isfinite(jac_j(0.0, x)))
    # partição da unidade dos pesos (com e sem indicadores fracionários)
    E = np.array([[tb(0.0) for tb in row] for row in model.tables])
    Y = pb.assemble(X0[pb.offset : pb.offset + pb.n_states], E[pb.i_from], E[pb.i_to])
    inp, _, _ = pb.model.local_state(Y, pb.scheme.theta_nodes)
    S, flags, eq = pb.model.closures(inp)
    assert np.allclose(flags.Est + flags.Anu + flags.Bub + flags.Int, 1.0, atol=1e-12)
