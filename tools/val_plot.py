"""Figura rápida de uma corrida de validação: P na entrada (kPa), vazão de líquido na saída, fração de gás na base e no topo do riser.
uso: uv run python tools/val_plot.py tag [t0 t1]   (figura em runs/val_<tag>.png)"""

import sys, json, numpy as np, warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_summary import load_opts, rebuild, REF
from mfsim.simulation import build_model
from mfsim.results import Results

tag = sys.argv[1]
run, opts = load_opts(tag)
T = run.T
case, fluid, A, qL, qG = rebuild(opts, T, T[-1])
model = build_model(case)
p = Results(model, T, run.X).pipe(0)
m = case.meta
x = p["x"]
ib = int(np.argmin(np.abs(x - m["s_elbow0"])))
it_ = int(np.argmin(np.abs(x - (m["s_elbow0"] + m["H_riser"]))))
t0, t1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (T[0], T[-1])
sel = (T >= t0) & (T <= t1)
fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
ax[0].plot(T[sel], p["out"]["P_bar"][1][sel] * 100)
ax[0].set_ylabel("P entrada (kPa)")
key = next((k for k in REF if k in tag), None)
if key:
    ax[0].axhline(REF[key]["P_min"], ls=":", c="k")
    ax[0].axhline(REF[key]["P_max"], ls=":", c="k")
    ax[0].set_title(REF[key]["fonte"] + f" | exp. período {REF[key]['periodo']} s", fontsize=9)
ax[1].plot(T[sel], p["out"]["qL"][-1][sel])
ax[1].axhline(qL, ls=":", c="k")
ax[1].set_ylabel("qL saída (kg/s)")
ax[2].plot(T[sel], p["out"]["alphaG"][ib + 1][sel], label="base do riser")
ax[2].plot(T[sel], p["out"]["alphaG"][it_ - 1][sel], label="topo do riser")
ax[2].plot(T[sel], p["out"]["alphaG"][ib - 2][sel], label="linha, 2 células antes do joelho")
ax[2].legend(fontsize=8)
ax[2].set_ylabel("αG")
vG_ = p["out"]["qG"][ib + 1] / (
    fluid.rhoG(p["out"]["P_bar"][ib + 1] * 1e5) * np.maximum(p["out"]["alphaG"][ib + 1], 1e-3) * A
)
vL_ = p["out"]["qL"][ib + 1] / (fluid.rhoL0 * np.maximum(1 - p["out"]["alphaG"][ib + 1], 1e-3) * A)
ax[3].plot(T[sel], vG_[sel], label="vG base riser")
ax[3].plot(T[sel], vL_[sel], label="vL base riser")
ax[3].set_ylabel("m/s")
ax[3].legend(fontsize=8)
ax[3].set_xlabel("t (s)")
fig.tight_layout()
os.makedirs("runs", exist_ok=True)
fig.savefig(f"runs/val_{tag}.png", dpi=90)
print("saved", f"runs/val_{tag}.png", "keys:", list(p["out"].keys())[:12])
