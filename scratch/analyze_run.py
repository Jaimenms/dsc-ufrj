import numpy as np, json, warnings, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
tag = sys.argv[1]
d = np.load(f"scratch/severe_D_{tag}.npz"); T, X = d["T"], d["X"]; opts = json.loads(str(d["opts"]))
fluid = Fluid(rhoL0=opts.get("rhoL", 900.0), rhoG0=1.2, muL=opts.get("muL", 2e-3), muG=1.8e-5, gammaL=opts.get("gammaL", 1000.0), gammaG=1.05, P0=1e5, sigma=0.05)
D = opts.get("D", 0.05); A = np.pi*D**2/4; P_out = opts.get("P_out", 4.0); P_est = (P_out+1.0)*1e5
qL = opts.get("vSL", 0.3)*fluid.rhoL0*A; qG = opts.get("vSG", 0.1)*fluid.rhoG(P_est)*A
extra = {k: v for k, v in {"regime_vertical": opts.get("rv", 2), "q_switch_L": opts.get("fsw", 0.02)*qL, "q_switch_G": opts.get("fsw", 0.02)*qG, "switch_frac": opts.get("switch_frac"),
                           "relax_time": opts.get("relax"), "limit_alpha": opts.get("limit"), "limit_band": opts.get("band")}.items() if v is not None}
case = severe_slug_case(method=opts.get("method", "fv_upwind"), tf=T[-1], dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=opts.get("L_h1", 20.0),
                        L_down=opts.get("L_down", 300.0), slope_down_deg=opts.get("slope", -3.0), H_riser=opts.get("H", 20.0), L_h2=5.0,
                        n_h1=opts.get("n_h1", 2), n_down=opts.get("n_down", 30), n_riser=opts.get("n_riser", 10), n_h2=opts.get("n_h2", 2), regime=4, single_pipe=True,
                        fluid=fluid, roughness=1e-5, extra=extra)
model = build_model(case); res = Results(model, T, X); p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qLo = p["out"]["qL"]
h = model.pipes[0].mesh["h"]; mass = np.sum((1-aG[1:-1])*fluid.rhoL(P[1:-1]*1e5)*A*h[:,None], axis=0)
net = np.concatenate([[0.0], np.cumsum(0.5*((qLo[0,1:]-qLo[-1,1:])+(qLo[0,:-1]-qLo[-1,:-1]))*np.diff(T))]); err = mass-mass[0]-net
peaks = [i for i in range(1, T.size-1) if qLo[-1,i] > 2.0 and qLo[-1,i] >= qLo[-1,i-1] and qLo[-1,i] > qLo[-1,i+1]]
tp = T[peaks]
print(f"{tag} ({opts.get('method','fv_upwind')}, malha {opts.get('n_down',30)}/{opts.get('n_riser',10)}): picos em {np.round(tp).astype(int).tolist()}")
if len(tp) > 1: print(f"   primeiro slug em t={tp[0]:.0f} s, período médio {np.mean(np.diff(tp)):.0f} s, pico máx {qLo[-1].max():.2f} kg/s")
print(f"   erro de massa: {err[-1]:.2f} kg em {qL*T[-1]:.0f} kg alimentados ({100*err[-1]/(qL*T[-1]):.2f} %); erro em t=500,1000,1500,2000,3000: {[round(float(err[np.argmin(np.abs(T-t))]),2) for t in (500,1000,1500,2000,3000) if t <= T[-1]]}")
