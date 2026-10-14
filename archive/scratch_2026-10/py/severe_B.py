import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, superficial_velocities, boe_criterion
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.fluid import Fluid
rv = int(sys.argv[1]); tf = float(sys.argv[2]); tag = sys.argv[3]
fluid = Fluid(rhoL0=850.0, rhoG0=18.0, muL=5e-3, muG=1.5e-5, gammaL=1000.0, gammaG=1.1, P0=20e5, sigma=0.03)
case = severe_slug_case(method="fv_upwind", tf=tf, qL=2.0, qG=0.007, P_out_bar=5.0, L_h1=300.0, L_down=2000.0, slope_down_deg=-3.0,
                        H_riser=50.0, L_h2=100.0, n_h1=3, n_down=40, n_riser=10, n_h2=2, regime=4, single_pipe=True,
                        fluid=fluid, extra={"regime_vertical": rv})
vSL, vSG = superficial_velocities(case); m = case.meta
P_base = m["P_out_bar"]*1e5 + 850*9.81*m["H_riser"]
print("vSL, vSG:", vSL, vSG, "Boe:", boe_criterion(vSL, vSG, P_base, 850.0, 0.5, m["L_h1"] + m["L_down"]), flush=True)
model = build_model(case); X0 = initial_guess(model)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-10, verbose=True)
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf+1, 5.0), X0, first_step=1e-3); print("OK", info, time.time()-t0, flush=True)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0); sys.exit(1)
np.savez(f"scratch/severe_B_{tag}.npz", T=T, X=X)
res = Results(model, T, X); p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qL = p["out"]["qL"]; qG = p["out"]["qG"]
ib = np.argmin(np.abs(x - (m["L_h1"]+m["L_down"]))); it_top = ib + 10
for it in range(0, T.size, max(1, T.size//50)):
    print(f"t={T[it]:6.0f} P_in={P[1,it]:6.2f} P_base={P[ib,it]:6.2f} aG_down_end={aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[ib+5,it]:.3f}/{aG[it_top,it]:.3f} qL_base={qL[ib,it]:6.3f} qL_out={qL[-1,it]:6.3f} qG_out={qG[-1,it]:.4f}", flush=True)
