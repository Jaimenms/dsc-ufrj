import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
regimes = [int(c) for c in sys.argv[1]]; tf = float(sys.argv[2])
case = severe_slug_case(method="fv_upwind", tf=tf, qL=1.0, qG=0.05, n_h1=5, n_down=20, n_riser=10, n_h2=3, regime=regimes, single_pipe=False, extra={"k_switch": 50.0})
model = build_model(case); X0 = initial_guess(model)
class Dbg(ImplicitBDF):
    def integrate(self, *a, **k):
        self._t_last_print = time.time()
        return super().integrate(*a, **k)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-10, verbose=True)
t0 = time.time()
T, X, info = bdf.integrate(np.arange(0, tf + 1, 5.0), X0, first_step=1e-3)
print(info, time.time()-t0, flush=True)
res = Results(model, T, X); p3 = res.pipe(2); p4 = res.pipe(3); p2 = res.pipe(1)
for it in range(0, T.size, max(1, T.size//30)):
    print(f"t={T[it]:5.0f} P_in={res.pipe(0)['out']['P_bar'][1,it]:6.2f} P_base={p3['out']['P_bar'][1,it]:6.2f} aG_down_end={p2['out']['alphaG'][-2,it]:.3f} aG_riser={p3['out']['alphaG'][1,it]:.3f}/{p3['out']['alphaG'][5,it]:.3f}/{p3['out']['alphaG'][-2,it]:.3f} qL_out={p4['out']['qL'][-1,it]:6.3f} qG_out={p4['out']['qG'][-1,it]:.4f}", flush=True)
np.savez(f"scratch/severe_diag_{sys.argv[1]}_{int(tf)}.npz", T=T, X=X)
