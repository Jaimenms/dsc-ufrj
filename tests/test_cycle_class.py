"""Classe do ciclo (issue 043): permanente, deriva sem ciclo, ciclo de pressão com saída suave e
golfada severa."""

import numpy as np

from mfsim.analysis import cycle_class

T = np.linspace(0.0, 4000.0, 2001)


def test_classes():
    q = np.full(T.size, 0.5)
    assert cycle_class(T, 6.0 + 0.01 * np.sin(T / 30), q)["cls"] == "stable"
    assert cycle_class(T, 32.0 + 0.1 * np.sin(T / 30), q)["cls"] == "stable"  # 0,2 bar em 32 bar: < 1 %
    drift = cycle_class(T, 20.0 + T / 4000.0, q)
    assert drift["cls"] == "stable" and np.isnan(drift["period"])  # deriva sem ciclo conta como estável
    P = 6.0 + 0.9 * np.sin(2 * np.pi * T / 800.0)
    cyc = cycle_class(T, P, 0.5 + 0.1 * np.sin(2 * np.pi * T / 800.0))
    assert cyc["cls"] == "cycling" and abs(cyc["period"] - 800.0) < 5.0 and abs(cyc["P_max"] - 6.9) < 0.01
    spikes = 0.3 + 5.0 * (np.sin(2 * np.pi * T / 800.0) > 0.98)
    slug = cycle_class(T, P, spikes)
    assert slug["cls"] == "slugging" and slug["surge"] > 2.0
