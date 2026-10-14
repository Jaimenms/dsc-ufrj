import numpy as np, warnings, collections, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
tend = float(sys.argv[1]); qsw = float(sys.argv[2]); order = int(sys.argv[3])
fluid = Fluid(rhoL0=1000.0, rhoG0=1.2, muL=1e-3, muG=1.8e-5, gammaL=1000.0, gammaG=1.05, P0=1e5, sigma=0.07)
D=0.05; A=np.pi*D**2/4; qL=0.3*1000*A; qG=0.05*fluid.rhoG(2.5e5)*A
case = severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=1.5, L_h1=20.0, L_down=300.0, slope_down_deg=-2.0, H_riser=15.0, L_h2=5.0,
                        n_h1=2, n_down=30, n_riser=10, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, extra={"regime_vertical": 2, "q_switch": qsw}, start="shut_in")
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model)
stats = collections.Counter()
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats[("ok" if ok else self._fail_reason, nit)] += 1
        return x, ok, nit
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=(lo,hi), min_step=1e-10, max_order=order, verbose=True)
t0 = time.time()
try:
    T, X, info = bdf.integrate([0, tend], X0, first_step=1e-3); print("OK", info, flush=True)
except Exception as e:
    print("FAIL", e, flush=True)
print("time", time.time()-t0, "steps", bdf.n_steps, "rej", bdf.n_rejected, "jac", bdf.n_jac, "rhs", bdf.n_rhs, flush=True)
print("newton outcomes:", sorted(stats.items(), key=lambda kv: -kv[1])[:14], flush=True)
