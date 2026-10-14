import numpy as np, warnings, time, sys, collections
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
tf = float(sys.argv[1])
fluid = Fluid(rhoL0=850.0, rhoG0=18.0, muL=5e-3, muG=1.5e-5, gammaL=1000.0, gammaG=1.1, P0=20e5, sigma=0.03)
case = severe_slug_case(method="fv_upwind", tf=tf, qL=2.0, qG=0.007, P_out_bar=5.0, L_h1=300.0, L_down=2000.0, slope_down_deg=-3.0,
                        H_riser=50.0, L_h2=100.0, n_h1=3, n_down=40, n_riser=10, n_h2=2, regime=4, single_pipe=True,
                        fluid=fluid, extra={"regime_vertical": 2})
model = build_model(case); X0 = initial_guess(model)
pb = model.pipes[0]
def where(i):
    j = i; return f"node{pb.free[j]//4} slot{pb.free[j]%4}"
stats = collections.Counter()
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats["ok" if ok else f"fail nit={nit}"] += 1
        self._last = (t, h, pred, x, ok); return x, ok, nit
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-6)
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf+1, 5.0), X0, first_step=1e-3); print("OK", info, time.time()-t0, flush=True)
    np.savez("scratch/severe_B_diag.npz", T=T, X=X)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0, flush=True)
print("newton stats", stats, flush=True)
t, h, pred, x, ok = bdf._last
print(f"last newton: t={t:.4f} h={h:.2e} ok={ok}")
scale = bdf.atol + bdf.rtol*np.abs(x); contrib = np.abs(x - pred)/scale
for i in np.argsort(-contrib)[:8]: print(f"  {where(i)}: x={x[i]:.6g} pred={pred[i]:.6g} |x-pred|/scale={contrib[i]:.3g}")
f = model.rhs(t, x); print("rhs finite:", np.all(np.isfinite(f)))
for i in np.argsort(-np.abs(np.nan_to_num(f)))[:8]: print(f"  largest rhs: {where(i)} f={f[i]:.3g} x={x[i]:.5g}")
E, Ys, inf, _ = model.assemble(t, x); Y = Ys[0]
np.set_printoptions(linewidth=200, precision=3, suppress=True)
print("aG:", Y[:,0]); print("P bar:", Y[:,1]*10); print("qL:", Y[:,2]); print("qG:", Y[:,3])
