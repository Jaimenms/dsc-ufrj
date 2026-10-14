import numpy as np, warnings, collections, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=1000.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
case = severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=30, n_riser=10, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, extra={"regime_vertical": 2, "q_switch": 0.01, "relax_time": 0.2})
model = build_model(case); X0 = initial_guess(model); lo, hi = state_bounds(model); pb = model.pipes[0]
def where(i): return f"node{pb.free[i]//4} slot{pb.free[i]%4}"
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=(lo,hi), min_step=1e-10, max_order=2)
T, X, info = bdf.integrate([0, 0.2], X0, first_step=1e-3); x = X[:, -1]; print("to 0.2:", info["n_steps"], flush=True)
stats = collections.Counter(); dom = collections.Counter(); errlog = []
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        self.ncall = getattr(self, "ncall", 0) + 1
        if self.ncall > 600: raise RuntimeError("budget")
        xx, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats[("ok" if ok else self._fail_reason, nit, "clip" if self._clipped else "", "bdf1" if gcoef == 1.0 else "bdf2")] += 1
        scale = self.atol + self.rtol*np.abs(pred); d = np.abs(xx - pred)/scale
        if not ok: dom[where(int(np.argmax(d)))] += 1
        self._lp = (t, h, pred, xx, ok); return xx, ok, nit
bdf2 = Dbg(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=(lo,hi), min_step=1e-10, max_order=2, verbose=False)
t0 = time.time()
# run with a cap on steps: use a small tend and stop by time budget
import threading
try:
    T2, X2, info2 = bdf2.integrate([0.2, 0.3], x, first_step=1e-4); print("to 0.3:", info2, flush=True)
except Exception as e:
    print("FAIL", e)
print("time", time.time()-t0, "steps", bdf2.n_steps, "rej", bdf2.n_rejected, flush=True)
print("outcomes:", sorted(stats.items(), key=lambda kv: -kv[1])[:8], flush=True)
print("dominant components in failures:", dom.most_common(6), flush=True)
t, h, pred, xx, ok = bdf2._lp
E, Ys, _, _ = model.assemble(t, xx); Y = Ys[0]
np.set_printoptions(linewidth=230, precision=4, suppress=True)
print("t", t, "h", h); print("aG:", Y[:,0]); print("P:", Y[:,1]*10); print("qL:", Y[:,2]); print("qG:", Y[:,3])
# error estimate contributors of the last accepted-like step
scale = 1e-5 + 1e-3*np.abs(pred); d = np.abs(xx-pred)/scale
for i in np.argsort(-d)[:6]: print(f"  {where(i)}: pred={pred[i]:.6g} sol={xx[i]:.6g} |d|/scale={d[i]:.3g}")
