import numpy as np, warnings, time, traceback, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
regimes = [int(c) for c in sys.argv[1]]
tf = float(sys.argv[2]) if len(sys.argv) > 2 else 4000.0
tag = sys.argv[1]
case = severe_slug_case(method="fv_upwind", tf=tf, qL=1.0, qG=0.05, n_h1=5, n_down=20, n_riser=10, n_h2=3,
                        regime=regimes, single_pipe=False, extra={"k_switch": 50.0})
model = build_model(case)
X0 = initial_guess(model)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), verbose=False, min_step=1e-10)
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf + 1, 5.0), X0, first_step=1e-3)
    print("OK", info, time.time() - t0)
except Exception as e:
    traceback.print_exc(limit=2); print("steps", bdf.n_steps, "rej", bdf.n_rejected)
    sys.exit(1)
np.savez(f"scratch/severe3_{tag}.npz", T=T, X=X)
res = Results(model, T, X)
p2 = res.pipe(1); p3 = res.pipe(2); p4 = res.pipe(3)
for it in range(0, T.size, max(1, T.size // 40)):
    print(f"t={T[it]:6.0f} P_in={res.pipe(0)['out']['P_bar'][1,it]:6.2f} P_base={p3['out']['P_bar'][1,it]:6.2f} aG_down_end={p2['out']['alphaG'][-2,it]:.3f} aG_riser={p3['out']['alphaG'][1,it]:.3f}/{p3['out']['alphaG'][5,it]:.3f}/{p3['out']['alphaG'][-2,it]:.3f} qL_out={p4['out']['qL'][-1,it]:6.3f} qG_out={p4['out']['qG'][-1,it]:.4f}")
