"""Espetro da linearização com a matriz de massa (issue 048): no modo conservativo o lado direito devolve dT/dt
(massas por volume), e o espetro de M·δẋ = J·δx coincide com o do mesmo esquema na forma primitiva, que tem o
mesmo estacionário e a mesma dinâmica semi-discreta; os autovalores de J sozinha não."""

import numpy as np

from mfsim.analysis import dense_jacobian, linear_spectrum, mass_matrix
from mfsim.core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN
from mfsim.fluid import Fluid
from mfsim.network.graph import BoundaryVertex, Network, Pipe, specs
from mfsim.simulation import build_model, state_bounds
from mfsim.steady import newton_steady

FLUID = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)


def _riser(conservative, n=10):
    """Estacionário de um riser vertical de 20 m em bolhas (vazões impostas na base, pressão no topo)."""
    D = 0.05
    A = np.pi * D**2 / 4
    qL, qG = 0.3 * 900 * A, 0.1 * FLUID.rhoG(5e5) * A
    extra = {"conservative": conservative, "relax_target": "noslip" if conservative else "raw"}
    extra.update(art_visc=0.05, virtual_mass=0.5, bubble_drag="fisico", closure_alpha="donor", switch_frac=0.5)
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
    X, info = newton_steady(model, x, tol=1e-11, verbose=False, bounds=state_bounds(model), max_iter=80)
    assert info["converged"], info
    return model, X


def test_conservative_spectrum_matches_the_primitive_one():
    mc, Xc = _riser(True)
    mp, Xp = _riser(False)
    assert mc.conservative and not mp.conservative
    np.testing.assert_allclose(Xc, Xp, atol=1e-12)  # o mesmo estacionário
    wc, wp = linear_spectrum(mc, Xc), linear_spectrum(mp, Xp)
    a, b = np.sort(np.abs(wp)), np.sort(np.abs(wc))
    assert np.max(np.abs(a - b) / a) < 1e-6
    np.testing.assert_allclose(wc[:5], wp[:5], rtol=1e-6)  # os modos mais lentos, um a um
    # J sozinha: as linhas das massas vêm escaladas por ρL e ρG e o espetro é outro
    wj = np.linalg.eigvals(dense_jacobian(mc, Xc))
    assert np.max(np.abs(np.sort(np.abs(wj)) - a) / a) > 1.0
    assert np.array_equal(mass_matrix(mp, Xp), np.eye(Xp.size))
