"""Estimativa de erro de Milne na ordem 2 (issue 058, `mfsim.integrators.milne`): constante 2/11 no passo constante,
preditor quadrático exato para polinómios de grau 2, e o caso de exemplo integrado até ao mesmo estado que o
`ImplicitBDF` com menos (ou iguais) tentativas."""

import numpy as np

from mfsim.integrators.implicit import milne_order2
from mfsim.integrators.milne import integrate_bdf_milne
from mfsim.io import case_from_json
from mfsim.simulation import build_model, simulate


def test_milne_constant_is_2_over_11_at_constant_step():
    t_hist = [0.0, 1.0, 2.0]
    vals = tuple(np.array([v]) for v in (0.0, 1.0, 2.0))
    _, C = milne_order2(3.0, t_hist, vals)
    assert abs(C - 2.0 / 11.0) < 1e-12


def test_quadratic_predictor_is_exact_for_quadratics():
    t_hist = [0.3, 1.1, 1.6]  # passos desiguais
    poly = lambda t: 2.0 - 0.7 * t + 0.4 * t**2  # noqa: E731
    vals = tuple(np.array([poly(t)]) for t in t_hist)
    pred, C = milne_order2(2.5, t_hist, vals)
    assert abs(pred[0] - poly(2.5)) < 1e-12
    assert 0.0 < C < 1.0


def test_hello_world_reaches_the_same_state_with_fewer_attempts():
    net = case_from_json("cases/examples/hello_world.json")
    net.set_time(tf=60.0, dt_out=10.0)
    res = simulate(net, progress=False, integrator="implicit", backend="numpy")  # referência: estimativa da tese
    res_m = simulate(net, progress=False, integrator="implicit", backend="numpy", error_estimate="milne")
    model = build_model(net)
    X0 = np.asarray(res.X[:, 0], float)
    _, X, info = integrate_bdf_milne(model, X0, np.asarray(res.T, float), model.integration, progress=False)
    assert info["success"] and res.info["success"] and res_m.info["success"]
    scale = model.state_scales(X0)
    assert np.max(np.abs(X[:, -1] - np.asarray(res.X[:, -1])) / scale) < 3e-2  # dentro das tolerâncias (rtol 3e-3)
    assert np.allclose(np.asarray(res_m.X), X)  # o atalho e a opção JSON são o mesmo caminho
    assert info["n_steps"] + info["n_rejected"] <= res.info["n_steps"] + res.info["n_rejected"]
