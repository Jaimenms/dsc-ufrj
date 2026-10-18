"""Paragem adaptativa (issue 041): a assinatura de janela distingue permanente, ciclo repetido e
deriva; a integração em janelas pára quando o sinal assenta, respeita t_min e tf, e mantém o
acumulador dos fluxos de contorno contínuo entre janelas."""

import numpy as np

from mfsim.simulation import settled, simulate, window_signature


def _w(y, T=None):
    T = np.linspace(0.0, 1000.0, y.size) if T is None else T
    return window_signature(T, y)


def test_signature_and_criterion():
    T = np.linspace(0.0, 1000.0, 1001)
    flat = _w(4.0 + 1e-4 * np.sin(T / 10), T)
    assert settled([flat], 0.05, 0.1) == "steady"
    s1 = _w(5.0 + np.sin(2 * np.pi * T / 200.0), T)
    s2 = _w(5.0 + 1.02 * np.sin(2 * np.pi * (T + 30) / 205.0), T)
    assert s1["crossings"] >= 9 and abs(s1["period"] - 200.0) < 25.0
    assert settled([s1], 0.05, 0.1) is None  # uma janela só não chega
    assert settled([s1, s2], 0.05, 0.1) == "cycle"
    ramp1, ramp2 = _w(4.0 + T / 1000.0, T), _w(5.0 + T / 1000.0, T)
    assert ramp1["crossings"] <= 1 and settled([ramp1, ramp2], 0.05, 0.1) is None
    grow = _w(5.0 + 2.0 * np.sin(2 * np.pi * T / 200.0), T)
    assert settled([s1, grow], 0.05, 0.1) is None  # amplitude ainda a mudar


def _thesis_case(**settle):
    from thesis.mf2 import chapter10_case

    net = chapter10_case("10.1.1")
    net.set_time(tf=net.time.t0 + 200.0, dt_out=1.0, settle=tuple(sorted(settle.items())))
    return net


def test_stops_when_steady_after_t_min():
    net = _thesis_case(window=20.0, t_min=40.0, amp_tol=1e-3)
    res = simulate(net, progress=False)
    st = res.info["settle"]
    assert st["reason"] == "steady" and abs(res.T[-1] - (net.time.t0 + 40.0)) < 1e-9
    assert np.all(np.diff(res.T) > 0) and len(st["windows"]) == 2


def test_runs_to_tf_when_never_settled():
    net = _thesis_case(window=50.0, amp_tol=1e-12, rel_tol=1e-12)
    res = simulate(net, progress=False)
    assert res.info["settle"]["reason"] == "tf" and abs(res.T[-1] - (net.time.t0 + 200.0)) < 1e-9
    plain = _thesis_case()
    plain.set_time(settle=())
    ref = simulate(plain, progress=False)
    assert np.array_equal(ref.T, res.T)
    assert np.allclose(ref.X[:, -1], res.X[:, -1], rtol=1e-3, atol=1e-6)


def test_accumulator_is_continuous_between_windows():
    import json
    from pathlib import Path

    from mfsim.io import case_from_json

    d = json.load(open(Path(__file__).resolve().parents[1] / "cases" / "severe_slug" / "reference.json"))
    d["time"].update(tf=60.0, dt_out=2.0)
    plain = simulate(case_from_json(d), progress=False)
    d["time"]["settle"] = {"window": 20.0, "amp_tol": 1e-9, "rel_tol": 1e-9}
    chunked = simulate(case_from_json(d), progress=False)
    a, b = np.asarray(plain.info["accum"]), np.asarray(chunked.info["accum"])
    assert a.shape == b.shape
    scale = np.abs(a).max(axis=1) + 1e-12
    assert np.all(np.abs(a[:, -1] - b[:, -1]) / scale < 0.05)


def test_envelope_match_and_capture():
    """061: `match = "envelope"` aceita ciclos cujo período entre cruzamentos varia; a captura continua N ciclos
    (ou uma janela fixa num estacionário) e regista a janela em info.settle.capture; vários sinais têm de assentar."""
    T = np.linspace(0.0, 1000.0, 1001)
    y1 = 5.0 + np.sin(2 * np.pi * T / 200.0) + 0.3 * np.sin(2 * np.pi * T / 95.0)
    y2 = 5.0 + np.sin(2 * np.pi * (T + 40) / 200.0) + 0.3 * np.sin(2 * np.pi * (T + 40) / 95.0)
    w1, w2 = _w(y1, T), _w(y2, T)
    assert settled([w1, w2], 0.05, 0.1, match="envelope") == "cycle"
    assert "y_min" in w1 and "y_max" in w1
    net = _thesis_case(
        window=20.0, t_min=40.0, amp_tol=1e-3, capture_window=30.0, signals=[{"var": "qL", "amp_tol": 1e-3}]
    )
    res = simulate(net, progress=False)
    st = res.info["settle"]
    assert st["reason"] == "steady" and abs(res.T[-1] - (net.time.t0 + 70.0)) < 1e-9
    assert abs(st["capture"][0] - (net.time.t0 + 40.0)) < 1e-9 and abs(st["capture"][1] - res.T[-1]) < 1e-9
    assert len(st["windows_all"]) == 2 and len(st["signals"]) == 2
    assert net.time.to_json()["settle"]["signals"] == [{"var": "qL", "amp_tol": 1e-3}]
