import numpy as np, time, warnings, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, superficial_velocities, boe_criterion
from mfsim.simulation import build_model, state_bounds, simulate, steady_state
from mfsim.initial_guess import initial_guess
regime = int(sys.argv[1]) if len(sys.argv) > 1 else 6
case = severe_slug_case(regime=regime, tf=4000.0, dt_out=5.0, qL=1.0, qG=0.05)
vSL, vSG = superficial_velocities(case)
print("vSL, vSG:", vSL, vSG)
model = build_model(case)
X0 = initial_guess(model)
f0 = model.rhs(0.0, X0); print("n_states", model.n_states, "|rhs(X0)|", np.linalg.norm(f0))
t0 = time.time()
res = simulate(case, X0=X0, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, verbose=True)
print("cpu", time.time()-t0, res.info)
np.savez(f"scratch/severe_regime{regime}.npz", T=res.T, X=res.X)
p = res.pipe(0)
aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qL = p["out"]["qL"]
x = p["x"]
ib = np.argmin(np.abs(x - (case.meta["L_h1"] + case.meta["L_down"])))  # riser base
print("riser base node", ib, "x", x[ib])
for it in range(0, res.T.size, max(1, res.T.size//20)):
    print(f"t={res.T[it]:6.0f} P_base={P[ib,it]:7.2f} bar  P_in={P[1,it]:7.2f}  aG_base={aG[ib,it]:.3f} aG_top={aG[-2,it]:.3f} qL_out={qL[-1,it]:6.2f}")
