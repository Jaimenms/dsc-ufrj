"""Caso 'de laboratório': ar-água, linha descendente curta, riser 15 m, separador 1.5 bar."""
import numpy as np, warnings, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, superficial_velocities, boe_criterion
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.fluid import Fluid
opts = json.loads(sys.argv[1]); tag = sys.argv[2]
tf = opts.get("tf", 3000.0)
fluid = Fluid(rhoL0=1000.0, rhoG0=1.2, muL=1e-3, muG=1.8e-5, gammaL=1000.0, gammaG=opts.get("gammaG", 1.05), P0=1e5, sigma=0.07)
D = opts.get("D", 0.05); A = np.pi*D**2/4
P_est = opts.get("P_est", 2.5e5)
qL = opts.get("vSL", 0.3)*1000*A
qG = opts.get("vSG", 0.05)*fluid.rhoG(P_est)*A
case = severe_slug_case(method="fv_upwind", tf=tf, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=opts.get("P_out", 1.5), L_h1=opts.get("L_h1", 20.0),
                        L_down=opts.get("L_down", 300.0), slope_down_deg=opts.get("slope", -2.0), H_riser=opts.get("H", 15.0), L_h2=5.0,
                        n_h1=2, n_down=opts.get("n_down", 30), n_riser=opts.get("n_riser", 10), n_h2=2, regime=4, single_pipe=True,
                        fluid=fluid, roughness=1e-5, start=opts.get("start", "guess"), t_ramp=opts.get("t_ramp", 100.0),
                        alpha_line0=opts.get("aline0", 0.5), alpha_riser0=opts.get("ariser0", 0.01),
                        extra={"regime_vertical": opts.get("rv", 2), "relax_time": opts.get("relax", 1.0), "alpha_band": opts.get("band", 0.02),
                               "alpha_min": opts.get("amin", 1e-3), "alpha_max": 1-opts.get("amin", 1e-3)})
vSL, vSG = superficial_velocities(case); m = case.meta
P_base = m["P_out_bar"]*1e5 + 1000*9.81*m["H_riser"]
print("qL, qG:", qL, qG, "vSL, vSG(out):", vSL, vSG, "Boe:", boe_criterion(vSL, vSG, P_base, 1000.0, 0.5, m["L_h1"] + m["L_down"]), flush=True)
model = build_model(case); X0 = model.initial_state() if opts.get("start") == "shut_in" else initial_guess(model)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=opts.get("rtol", 1e-3), atol=opts.get("rtol", 1e-3)*1e-2, bounds=state_bounds(model),
                  min_step=1e-9, verbose=True, max_order=opts.get("order", 2), max_newton=opts.get("maxnewton", 8), newton_tol=opts.get("ntol", 1e-3),
                  jac_reuse=opts.get("jacreuse", 20), fixed_step=opts.get("dt", None))
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf+1, 2.0), X0, first_step=1e-3); print("OK", info, time.time()-t0, flush=True)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0, flush=True); sys.exit(1)
np.savez(f"scratch/severe_C_{tag}.npz", T=T, X=X, opts=json.dumps(opts))
res = Results(model, T, X); p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qLo = p["out"]["qL"]; qGo = p["out"]["qG"]
ib = int(np.argmin(np.abs(x - (m["L_h1"]+m["L_down"]))))
print(f"qL_out: min {qLo[-1].min():.3f} max {qLo[-1].max():.3f} std {qLo[-1][T>tf/2].std():.3f} | P_in min {P[1].min():.3f} max {P[1].max():.3f}")
for it in range(0, T.size, max(1, T.size//60)):
    print(f"t={T[it]:6.0f} P_in={P[1,it]:6.3f} P_base={P[ib,it]:6.3f} aG_line={aG[5,it]:.3f}/{aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[ib+5,it]:.3f}/{aG[ib+10,it]:.3f} qL_out={qLo[-1,it]:6.3f} qG_out={qGo[-1,it]:.5f}", flush=True)
