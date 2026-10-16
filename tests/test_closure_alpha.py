"""Fração de gás dos fechamentos nas faces (`closure_alpha`, issue 044): com "donor" a fração das faces é a da
célula doadora (a mesma do transporte e da gravidade), o lado direito em JAX reproduz o numpy, o padrão "mean"
não muda nada e o estacionário discreto de um riser em bolhas deixa de ter o xadrez par-ímpar."""

import numpy as np
import pytest

from generators.severe_slug import severe_slug_case
from mfsim.core.slots import SLOT_ALPHA
from mfsim.fluid import Fluid
from mfsim.simulation import build_model, state_bounds
from mfsim.steady import newton_steady

FLUID = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)


def _case(closure_alpha=None, regime=4, extra=None):
    D = 0.05
    A = np.pi * D**2 / 4
    ex = {"regime_vertical": 2, "conservative": True, "vertical_blend": 20.0, "art_visc": 0.05}
    ex.update(extra or {})
    if closure_alpha is not None:
        ex["closure_alpha"] = closure_alpha
    return severe_slug_case(
        method="fv_upwind",
        tf=100,
        dt_out=2.0,
        qL=0.3 * 900 * A,
        qG=0.1 * FLUID.rhoG(5e5) * A,
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
        fluid=FLUID,
        roughness=1e-5,
        start="blocked",
        elbow_radius=5.0,
        n_elbow=3,
        extra=ex,
    )


def test_default_is_mean_and_unchanged():
    m0 = build_model(_case())
    m1 = build_model(_case("mean"))
    assert m0.pipes[0].scheme.closure_alpha == "mean"
    X = m0.initial_state()
    assert np.array_equal(m0.rhs(0.0, X), m1.rhs(0.0, X))


def test_donor_faces_take_the_upstream_cell():
    model = build_model(_case("donor"))
    sch = model.pipes[0].scheme
    N = sch.N
    Y = model.assemble(0.0, model.initial_state())[1][0].copy()
    Y[:, 2], Y[:, 3] = 0.5, 0.01  # co-corrente para jusante, vazões bem acima das larguras de chaveamento
    Y[1 : N + 1, SLOT_ALPHA] = np.linspace(0.1, 0.8, N)
    Yc, _, Yf = sch._cell_face_states(Y)
    np.testing.assert_allclose(Yf[1:N, SLOT_ALPHA], Yc[1:N, SLOT_ALPHA], atol=2e-3)
    Y[:, 2], Y[:, 3] = -0.5, -0.01  # refluxo: a doadora é a célula a jusante
    Yc, _, Yf = sch._cell_face_states(Y)
    np.testing.assert_allclose(Yf[1:N, SLOT_ALPHA], Yc[2 : N + 1, SLOT_ALPHA], atol=2e-3)


@pytest.mark.parametrize("regime", [4, 6])
def test_donor_jax_matches_numpy(regime):
    pytest.importorskip("jax")
    from mfsim.compiled import build_jax_rhs

    extra = {"low_void_regime": 2, "virtual_mass": 0.5, "relax_target": "noslip"}
    if regime == 6:
        extra.update(ipe_smooth=True, ipe_local=True, k_switch=20.0)
    model = build_model(_case("donor", regime=regime, extra=extra))
    rhs_j, _ = build_jax_rhs(model)
    rng = np.random.default_rng(1)
    X0 = model.initial_state()
    slots = model.pipes[0].free % 4
    for _ in range(3):
        x = X0 * (1 + 0.05 * rng.standard_normal(X0.size))
        x[slots == 0] = np.clip(x[slots == 0] + 0.3 * rng.random((slots == 0).sum()), 0.0, 1.0)
        f_np, f_jx = model.rhs(0.0, x), rhs_j(0.0, x)
        assert np.max(np.abs(f_np - f_jx)) < 1e-9 * max(1.0, np.max(np.abs(f_np)))


def _riser_profile(closure_alpha, n=10):
    """Fração de gás nas células do estacionário discreto de um riser vertical de 20 m em bolhas (vazões
    impostas na base, pressão no topo), por Newton a partir de um estado uniforme."""
    from mfsim.core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN
    from mfsim.network.graph import BoundaryVertex, Network, Pipe, specs

    D = 0.05
    A = np.pi * D**2 / 4
    qL, qG = 0.3 * 900 * A, 0.1 * FLUID.rhoG(5e5) * A
    extra = {"conservative": True, "relax_target": "noslip", "art_visc": 0.05, "virtual_mass": 0.5}
    extra.update(bubble_drag="fisico", closure_alpha=closure_alpha, switch_frac=0.5)
    pipe = Pipe.legacy(
        "1",
        "1",
        "2",
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 20.0],
        [n],
        D=D,
        roughness=1e-5,
        model=1,
        regime=2,
        method="fv_upwind",
        bc_in=(BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET),
        bc_out=(BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT),
        extra=extra,
    )
    vs = [BoundaryVertex("1", spec=specs(qL=qL, qG=qG)), BoundaryVertex("2", spec=specs(PT=0.4))]
    net = Network.build("riser", FLUID, vs, [pipe], t0=0.0, tf=10.0, dt_out=1.0, PR=1e6, integrator="implicit")
    model = build_model(net)
    pb = model.pipes[0]
    slots, nodes = pb.free % 4, pb.free // 4
    z = pb.scheme.x_nodes[nodes]
    x = np.zeros(pb.n_states)
    x[slots == 0] = 0.15
    x[slots == 1] = (4e5 + 0.85 * 900 * 9.81 * (20.0 - z[slots == 1])) / 1e6
    x[slots == 2], x[slots == 3] = qL, qG
    X, info = newton_steady(model, x, tol=1e-9, verbose=False, bounds=state_bounds(model), max_iter=80)
    assert info["converged"], info
    return model.assemble(0.0, X)[1][0][1 : pb.scheme.N + 1, SLOT_ALPHA]


def test_donor_removes_the_riser_checkerboard():
    """Com a média, os incrementos de fração entre células alternam (modo par-ímpar que cresce para cima);
    com a doadora o perfil é monótono e liso."""
    am, ad = _riser_profile("mean"), _riser_profile("donor")
    assert np.all(np.diff(am) > 0) and np.all(np.diff(ad) > 0)  # mais gás para cima nos dois (expansão)
    odd_even = lambda a: float(np.mean(np.abs(np.diff(a, 2))))  # noqa: E731
    assert odd_even(ad) < 0.5 * odd_even(am)
