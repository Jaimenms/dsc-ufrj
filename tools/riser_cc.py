"""Diagnóstico do filme no riser em trajetórias gravadas (partida bloqueada, duto único):
fração do tempo em contracorrente (filme descendo, gás subindo) e com riser quase cheio de gás,
e velocidade média do filme nessas células. uso: uv run python tools/riser_cc.py nome1 nome2 ...  (nomes em notebooks/data/ ou runs/)
"""

import numpy as np, json, sys, warnings

warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
from mfsim.runs import load_run

fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D = 0.05
A = np.pi * D**2 / 4
P_out = 4.0
qL = 0.3 * fluid.rhoL0 * A
qG = 0.1 * fluid.rhoG((P_out + 1) * 1e5) * A
for name in sys.argv[1:]:
    run = load_run(name)
    model = run.model
    res = run.res
    p = res.pipe(0)
    sch = model.pipes[0].scheme
    th = np.rad2deg(sch.theta_nodes)
    riser = np.flatnonzero(th > 80)
    T = run.T
    aG = p["out"]["alphaG"][riser]
    vL = p["out"]["velocidadeL"][riser]
    vG = p["out"]["velocidadeG"][riser]
    gasfull = aG > 0.9
    cc = gasfull & (vL < -0.02) & (vG > 0.02)
    up = gasfull & (vL > 0.5)
    short = name.replace("severe_slug_blocked_conservative_arc_", "")
    print(
        f"{short:28s}: riser com >90% gás {gasfull.mean():5.1%} do tempo | nessas células: filme descendo {cc.sum()/max(1, gasfull.sum()):5.1%}, líquido subindo > 0.5 m/s {up.sum()/max(1, gasfull.sum()):5.1%} | vL médio {vL[gasfull].mean() if gasfull.any() else np.nan:+.2f} m/s, vG médio {vG[gasfull].mean() if gasfull.any() else np.nan:+.2f} m/s"
    )
