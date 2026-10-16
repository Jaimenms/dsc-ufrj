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


def test_smooth_ignores_short_oscillations_of_the_riser_column():
    # ciclo de 400 s com oscilações de 8 s (coluna do riser contra o gás do bolsão) que cruzam a média
    T1 = np.arange(0.0, 4000.0, 1.0)
    P = 6.0 + 0.25 * np.sin(2 * np.pi * T1 / 400.0)
    for tk in np.arange(30.0, 4000.0, 400.0):
        on = T1 >= tk
        P[on] += 0.4 * np.sin(2 * np.pi * (T1[on] - tk) / 8.0) * np.exp(-(T1[on] - tk) / 15.0)
    q = np.full(T1.size, 0.5)
    raw, smooth = cycle_class(T1, P, q), cycle_class(T1, P, q, smooth=20.0)
    assert raw["period"] < 300.0  # cruzamentos a mais encurtam o período
    assert smooth["cls"] == "cycling" and abs(smooth["period"] - 400.0) < 5.0
    assert smooth["P_max"] == raw["P_max"]  # extremos do registo, não da média móvel


def test_jansen_criterion_for_the_reference_geometry():
    from mfsim.analysis import jansen_choke_dp_crit, jansen_choke_opening

    dp5, dp10 = jansen_choke_dp_crit(), jansen_choke_dp_crit(Phi=0.87)
    assert 0.70 < dp5 < 0.74 and 0.59 < dp10 < 0.63
    for a, ap in ((0.6, 0.75), (0.85, 0.5)):  # pouco sensível às frações de gás
        assert 0.5 < jansen_choke_dp_crit(alpha=a, alpha_p=ap) < 0.8
    ab = np.array([0.1, 0.2, 0.4])
    q = np.array([0.5, 0.5, 0.5])
    a_c = jansen_choke_opening(ab, q, 0.67, 15.0)
    dp = 0.5**2 / 900.0 / (0.0075 * 15.0 * a_c) ** 2
    assert abs(dp - 0.67) < 1e-9
    assert jansen_choke_opening(ab, q, 100.0, 15.0) is None


def test_cycle_map_draws_one_patch_per_cell():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mfsim.plotting import cycle_map

    rows = [
        dict(ab=a, P_res=P, cls=c, P_mean=6.0, P_max=7.0, period=p, q_mean=q, success=True)
        for (a, P, c, p, q) in [
            (0.1, 10, "stable", np.nan, 0.2),
            (0.3, 10, "slugging", 900.0, 0.3),
            (0.1, 20, "stable", np.nan, 0.4),
            (0.3, 20, "cycling", 60.0, 0.5),
        ]
    ]
    fig, ax = plt.subplots()
    cycle_map(ax, rows)
    assert len(ax.patches) == 4 + 2  # 4 células + 2 contornos da melhor estável por linha
    assert sorted(t.get_text() for t in ax.texts) == ["6,0", "6,0", "7,0\n15 min", "7,0\n60 s"]
    plt.close(fig)
