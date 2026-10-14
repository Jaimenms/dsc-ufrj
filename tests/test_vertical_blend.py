"""Transição suave (por ângulo) entre o padrão da linha e o padrão vertical."""

import numpy as np

from generators.severe_slug import severe_slug_case
from mfsim.fluid import Fluid
from mfsim.geometry import pipe_mesh
from mfsim.models.base import PipeParams
from mfsim.models.twofluid import TwoFluidModel


def _model(blend):
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    params = PipeParams(
        D=0.05, eps=1e-5, PR=1e6, regime=4, regime_vertical=2, vertical_angle=45.0, vertical_blend=blend
    )
    return TwoFluidModel(fluid, params)


def _coeffs(model, theta_deg):
    y = np.array([[0.5, 0.6, 0.5, 0.001]])
    A, B = model.quasi_linear(y, np.deg2rad(np.array([theta_deg])))
    return np.concatenate([A.ravel(), B.ravel()])


def test_hard_switch_jumps_and_blend_is_continuous():
    hard, soft = _model(0.0), _model(15.0)
    # chave dura: salto em 45°
    jump = np.abs(_coeffs(hard, 45.5) - _coeffs(hard, 44.5))
    assert jump.max() > 1e-3 * np.abs(_coeffs(hard, 44.5)).max()
    # mistura: pequena variação em 1° e pesos 0/1 fora da faixa
    diff = np.abs(_coeffs(soft, 45.5) - _coeffs(soft, 44.5))
    assert diff.max() < 0.15 * jump.max()
    assert np.allclose(_coeffs(soft, 20.0), _coeffs(hard, 20.0))
    assert np.allclose(_coeffs(soft, 80.0), _coeffs(hard, 80.0))
    w = soft.vertical_weight(np.deg2rad([0.0, 30.0, 45.0, 60.0, 90.0]))
    assert np.allclose(w, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_elbow_arc_geometry():
    case = severe_slug_case(
        method="fv_upwind",
        tf=10,
        qL=0.5,
        qG=0.001,
        D=0.05,
        P_out_bar=4.0,
        L_h1=20.0,
        L_down=300.0,
        slope_down_deg=-3.0,
        H_riser=20.0,
        L_h2=5.0,
        n_h1=2,
        n_down=15,
        n_riser=5,
        n_h2=2,
        regime=4,
        single_pipe=True,
        start="blocked",
        elbow_radius=5.0,
        n_elbow=4,
        extra={"regime_vertical": 2},
    )
    pipe = case.pipes[0]
    mesh = pipe_mesh(pipe.x, pipe.y, pipe.z, pipe.n_elements)
    th = np.rad2deg(mesh["theta"])
    # ângulos do arco crescem de -3° a 90° de forma monótona e o riser continua vertical
    arc = th[2 + 15 : 2 + 15 + 4]
    assert np.all(np.diff(arc) > 0) and arc[0] > -3.0 and arc[-1] < 90.0
    assert np.allclose(th[2 + 15 + 4 : 2 + 15 + 4 + 5], 90.0)
    # topo do riser na mesma cota e saída na mesma posição horizontal do caso sem arco
    z = mesh["xyz"][:, 2]
    y = mesh["xyz"][:, 1]
    assert abs(z[-1] - (300.0 * np.sin(np.deg2rad(-3.0)) + 20.0)) < 1e-9
    assert abs(y[-1] - (20.0 + 300.0 * np.cos(np.deg2rad(-3.0)) + 5.0)) < 1e-9
    # CI bloqueada: bolsão antes do arco, líquido no arco e no riser
    ic = pipe.ic
    s = mesh["s"]
    assert np.all(ic[s < case.meta["s_elbow0"] - 1e-9, 0] > 0.3)
    assert np.all(ic[(s > case.meta["s_elbow0"] + 1e-9) & (s < mesh["L"] - 5.0 - 1e-9), 0] == 0.0)
