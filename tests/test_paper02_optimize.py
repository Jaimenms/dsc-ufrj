"""Otimização operacional do artigo 02 (`tools/paper02_optimize.py`): conversão de óleo, potência de compressão por
estágios, enumeração com restrições sobre um substituto sintético; nada simula nem treina."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper02_optimize", ROOT / "tools" / "paper02_optimize.py")
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


def _linear_model():
    """Substituto linear nas entradas padronizadas: q_L sobe com a e P_gas, q_gl e os máximos sobem com P_gas."""
    sg = opt.sg
    W = np.zeros((2, len(sg.NAMES)))
    b = np.zeros(len(sg.NAMES))
    W[0, sg.IDX["qL_top"]], W[1, sg.IDX["qL_top"]] = 0.5, 1.0
    W[1, sg.IDX["q_inj"]] = 1.0
    W[0, sg.IDX["qL_max"]], W[1, sg.IDX["qL_max"]] = 10.0, 5.0
    W[1, sg.IDX["q_inj_max"]] = 1.2
    y_mean = np.zeros(len(sg.NAMES))
    for k, v in (
        ("qL_top", 4.0),
        ("qG_top", 1.0),
        ("q_inj", 0.6),
        ("qL_max", 15.0),
        ("q_inj_max", 0.7),
        ("P_bh", 80.0),
        ("P_inj", 80.0),
        ("P_an", 100.0),
    ):
        y_mean[sg.IDX[k]] = v
    return dict(
        params=[(W, b)],
        x_mean=np.array([0.55, 90.0]),
        x_std=np.array([0.45, 70.0]),
        y_mean=y_mean,
        y_std=np.ones(len(sg.NAMES)),
        span_log=False,
    )


def test_oil_and_compression():
    assert abs(opt.oil_rate_bbl_d(3.94) - 3.94 * 86400 / opt.KG_LIQ_PER_SM3_OIL * opt.BBL_PER_M3) < 1e-9
    p1 = opt.compression_power_kw(1.0, 20.0, 5.0)
    p2 = opt.compression_power_kw(1.0, 160.0, 5.0)
    assert 0.0 < p1 < p2 < 2000.0  # ~1 kg/s de 5 a 160 bar: centenas de kW, em 3 estágios
    assert opt.compression_power_kw(0.0, 160.0, 5.0) == 0.0 and opt.compression_power_kw(1.0, 4.0, 5.0) == 0.0
    two = opt.compression_power_kw(np.array([1.0, 2.0]), np.array([100.0, 100.0]), 5.0)
    assert np.allclose(two[1], 2 * two[0])


def test_optimise_prices_and_constraints():
    model = _linear_model()
    cheap = opt.optimise(model, n_grid=41, oil_price=70.0, energy_price=0.01, p_suction=5.0)
    assert cheap["a"] > 0.99 and cheap["P_gas_bar"] > 159.0 and cheap["constraints"]["active"] == []
    dear = opt.optimise(model, n_grid=41, oil_price=70.0, energy_price=50.0, p_suction=5.0)
    assert dear["P_gas_bar"] < cheap["P_gas_bar"] and dear["fob"] <= cheap["revenue"]
    # restrições nos máximos previstos: a capacidade do compressor trava P_gas, a do separador trava a abertura
    gl = opt.optimise(model, n_grid=41, oil_price=70.0, energy_price=0.01, p_suction=5.0, max_gl=1.0)
    assert "max_gl" in gl["constraints"]["active"] and gl["outputs"]["q_inj_max"] <= 1.0 + 1e-6
    assert gl["P_gas_bar"] < 159.0
    oil = opt.optimise(model, n_grid=41, oil_price=70.0, energy_price=0.01, p_suction=5.0, max_oil=18.0)
    assert "max_oil" in oil["constraints"]["active"] and oil["outputs"]["qL_max"] <= 18.0 + 1e-6
    assert oil["constraints"]["n_feasible"] < oil["constraints"]["n_grid"]
