import numpy as np, warnings, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.fluid import Fluid
opts = json.loads(sys.argv[1]); tag = sys.argv[2]; tend = float(sys.argv[3])
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
extra = {"regime_vertical": opts.get("rv", 2), "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG}
extra.update({k: v for k, v in opts.items() if k in ("switch_frac", "relax_time", "limit_band", "alpha_band")})
case = severe_slug_case(method="fv_upwind", tf=tend, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=30, n_riser=10, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, extra=extra)
model = build_model(case); lo, hi = state_bounds(model); pb = model.pipes[0]; h = pb.mesh["h"]
d = np.load("scratch/diag2600_state.npz"); X0 = d["X"][:, -1]; t0s = float(d["T"][-1])
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=opts.get("rtol", 1e-3), atol=opts.get("rtol", 1e-3)*1e-2, bounds=(lo,hi), min_step=1e-10, max_order=opts.get("order", 2), verbose=True, max_newton=opts.get("maxnewton", 12))
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(t0s, tend+1, 2.0), X0, first_step=1e-3); print("OK", info, time.time()-t0, flush=True)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0, flush=True); sys.exit(1)
np.savez(f"scratch/cont2600_{tag}.npz", T=T, X=X)
res = Results(model, T, X); p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qLo = p["out"]["qL"]; qGo = p["out"]["qG"]
rhoL = fluid.rhoL(P[1:-1]*1e5); mass = np.sum((1 - aG[1:-1]) * rhoL * A * h[:, None], axis=0)
net = np.concatenate([[0.0], np.cumsum(0.5*((qLo[0,1:]-qLo[-1,1:]) + (qLo[0,:-1]-qLo[-1,:-1])) * np.diff(T))])
print(f"massa líquida: inicial {mass[0]:.1f} final {mass[-1]:.1f}; integral {net[-1]:.1f}; erro {mass[-1]-mass[0]-net[-1]:.1f} kg", flush=True)
ib = int(np.argmin(np.abs(x - 320.0)))
print(f"qL_out: min {qLo[-1].min():.3f} max {qLo[-1].max():.3f} std {qLo[-1].std():.3f}", flush=True)
for it in range(0, T.size, max(1, T.size//50)):
    print(f"t={T[it]:6.0f} P_in={P[1,it]:6.3f} P_base={P[ib,it]:6.3f} aG_line={aG[5,it]:.3f}/{aG[15,it]:.3f}/{aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[ib+5,it]:.3f}/{aG[ib+10,it]:.3f} qL_out={qLo[-1,it]:6.3f} qG_out={qGo[-1,it]:.5f}", flush=True)
