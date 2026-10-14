import numpy as np, json, warnings, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
"""Análise rápida de uma corrida gravada (picos, período, erro de massa de líquido). uso: uv run python tools/analyze_run.py <tag|nome>"""
from mfsim.runs import load_run
tag = sys.argv[1]
run = load_run(tag); T, X = run.T, run.X; opts = run.meta; case = run.case; fluid = case.fluid
A = np.pi * case.pipes[0].D**2 / 4; qL = case.meta["qL"]
model = run.model; res = run.res; p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qLo = p["out"]["qL"]
h = model.pipes[0].mesh["h"]; mass = np.sum((1-aG[1:-1])*fluid.rhoL(P[1:-1]*1e5)*A*h[:,None], axis=0)
net = np.concatenate([[0.0], np.cumsum(0.5*((qLo[0,1:]-qLo[-1,1:])+(qLo[0,:-1]-qLo[-1,:-1]))*np.diff(T))]); err = mass-mass[0]-net
peaks = [i for i in range(1, T.size-1) if qLo[-1,i] > 2.0 and qLo[-1,i] >= qLo[-1,i-1] and qLo[-1,i] > qLo[-1,i+1]]
tp = T[peaks]
print(f"{tag} ({'fv_upwind'}, malha {opts.get('n_down')}/{opts.get('n_riser')}): picos em {np.round(tp).astype(int).tolist()}")
if len(tp) > 1: print(f"   primeiro slug em t={tp[0]:.0f} s, período médio {np.mean(np.diff(tp)):.0f} s, pico máx {qLo[-1].max():.2f} kg/s")
print(f"   erro de massa: {err[-1]:.2f} kg em {qL*T[-1]:.0f} kg alimentados ({100*err[-1]/(qL*T[-1]):.2f} %); erro em t=500,1000,1500,2000,3000: {[round(float(err[np.argmin(np.abs(T-t))]),2) for t in (500,1000,1500,2000,3000) if t <= T[-1]]}")
