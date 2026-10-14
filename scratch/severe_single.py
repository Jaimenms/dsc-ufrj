import numpy as np, warnings, time, sys, collections
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
rv = int(sys.argv[1]); tf = float(sys.argv[2])
case = severe_slug_case(method="fv_upwind", tf=tf, qL=1.0, qG=0.05, n_h1=5, n_down=20, n_riser=10, n_h2=3, regime=4, single_pipe=True, extra={"regime_vertical": rv})
model = build_model(case); X0 = initial_guess(model)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-10, verbose=True)
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf+1, 5.0), X0, first_step=1e-3); print("OK", info, time.time()-t0, flush=True)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0); sys.exit(1)
np.savez(f"scratch/severe_single_rv{rv}_{int(tf)}.npz", T=T, X=X)
res = Results(model, T, X); p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qL = p["out"]["qL"]; qG = p["out"]["qG"]
ib = np.argmin(np.abs(x - 1500)); it_top = ib + 10
for it in range(0, T.size, max(1, T.size//40)):
    print(f"t={T[it]:6.0f} P_in={P[1,it]:6.2f} P_base={P[ib,it]:6.2f} aG_down_end={aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[ib+5,it]:.3f}/{aG[it_top,it]:.3f} qL_base={qL[ib,it]:6.3f} qL_out={qL[-1,it]:6.3f} qG_out={qG[-1,it]:.4f}", flush=True)
