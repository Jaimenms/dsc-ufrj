"""Diagnóstico da malha fina graduada (K7): o que oscila em 60-80 s?"""
import numpy as np, warnings, time, json, sys, os
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
nd, nr, gd, gr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); tend = float(sys.argv[5]); tag = sys.argv[6]
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; P_out=4.0; qL=0.3*900*A; qG=0.1*fluid.rhoG((P_out+1)*1e5)*A
case = severe_slug_case(method="fv_upwind", tf=tend, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=nd, n_riser=nr, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked",
                        elbow_radius=5.0, n_elbow=4, grading_down=gd, grading_riser=gr,
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "conservative": True, "vertical_blend": 20.0,
                               **(json.loads(sys.argv[7]) if len(sys.argv) > 7 else {})})
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=3e-3, atol=3e-5, bounds=(lo, hi), min_step=1e-9, verbose=False, max_order=int(os.environ.get("ORDER", 2)),
                  transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
bdf.slow_exit = False   # configuração do severe slug (sem recálculo da Jacobiana dentro do Newton)
import os
WIN = float(os.environ.get("DIAG_WIN", 20.0)); DT = float(os.environ.get("DIAG_DT", 0.05))
t_eval = np.concatenate([np.arange(0, tend - WIN, 2.0), np.arange(tend - WIN, tend + 1e-9, DT)])
t0 = time.time()
T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3)
print("steps", info["n_steps"], "rej", info["n_rejected"], f"{time.time()-t0:.0f}s", flush=True)
np.savez(f"scratch/diag_{tag}.npz", T=T, X=X)
pb = model.pipes[0]; sel = T >= tend - WIN
Xw = X[:, sel]; dX = np.diff(Xw, axis=1)
osc = np.sum(np.sign(dX[:, 1:]) * np.sign(dX[:, :-1]) < 0, axis=1)       # trocas de sinal da derivada
amp = Xw.max(1) - Xw.min(1)
scale = 3e-5 + 3e-3*np.abs(Xw).mean(1)
order = np.argsort(-(osc * amp / scale))[:12]
x_cells = pb.scheme.x_cells; s0 = case.meta["s_elbow0"]
for i in order:
    node, slot = pb.free[i] // 4, pb.free[i] % 4
    pos = (x_cells[node-1] - s0) if 1 <= node <= pb.scheme.N else float("nan")
    print(f"node{node:3d} slot{slot} (s-s_joelho={pos:7.1f} m): trocas de sinal {osc[i]:4d}, amplitude {amp[i]:.4g} ({amp[i]/scale[i]:.0f} tol), média {Xw[i].mean():.4g}")
