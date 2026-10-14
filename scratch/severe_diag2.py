import numpy as np, warnings, time, collections, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
regimes = [int(c) for c in sys.argv[1]]; tf = float(sys.argv[2])
case = severe_slug_case(method="fv_upwind", tf=tf, qL=1.0, qG=0.05, n_h1=5, n_down=20, n_riser=10, n_h2=3, regime=regimes, single_pipe=False)
model = build_model(case); X0 = initial_guess(model)
def where(i):
    for pb in model.pipes:
        if pb.offset <= i < pb.offset + pb.n_states:
            j = i - pb.offset; return f"pipe{pb.pipe.index} node{pb.free[j]//4} slot{pb.free[j]%4}"
    return f"vertexA{i-model.n_pipe_states}"
stats = collections.Counter()
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats["ok" if ok else f"fail nit={nit}"] += 1
        self._last = (t, h, pred, x, ok); return x, ok, nit
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-10)
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf+1, 1.0), X0, first_step=1e-3); print(info, time.time()-t0)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0)
print("newton stats", stats)
t, h, pred, x, ok = bdf._last
print(f"last newton: t={t:.4f} h={h:.2e} ok={ok}")
scale = bdf.atol + bdf.rtol*np.abs(x); contrib = np.abs(x - pred)/scale
for i in np.argsort(-contrib)[:8]: print(f"  {where(i)}: x={x[i]:.6g} pred={pred[i]:.6g} |x-pred|/scale={contrib[i]:.3g}")
f = model.rhs(t, x); print("rhs finite:", np.all(np.isfinite(f)))
for i in np.argsort(-np.abs(np.nan_to_num(f)))[:8]: print(f"  largest rhs: {where(i)} f={f[i]:.3g} x={x[i]:.5g}")
E, Ys, inf, _ = model.assemble(t, x)
for k, Y in enumerate(Ys): print(f"pipe{k+1} aG:", np.round(Y[:,0],3)); print(f"       qL:", np.round(Y[:,2],3)); print(f"       qG:", np.round(Y[:,3],4))
print("vertexA", x[model.n_pipe_states:])
