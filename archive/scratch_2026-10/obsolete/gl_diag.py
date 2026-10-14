import numpy as np, warnings, time, collections
warnings.filterwarnings("ignore")
from mfsim.io_legacy import load_case, legacy_states
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
case, raw = load_case("CASO10-3-2b.mf2")
Tl, Xl = legacy_states(raw)
model = build_model(case)
i0 = int(np.argmin(np.abs(Tl - 1520))); X0 = Xl[:, i0]
def where(i):
    for pb in model.pipes:
        if pb.offset <= i < pb.offset + pb.n_states:
            j = i - pb.offset; return f"pipe{pb.pipe.index}({pb.model.name}) node{pb.free[j]//4} slot{pb.free[j]%4}"
    return f"vertexA{i-model.n_pipe_states}"
stats = collections.Counter(); topc = collections.Counter(); reasons = collections.Counter()
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats["newton_fail" if not ok else "newton_ok"] += 1
        if not ok: reasons[f"nit={nit}"] += 1
        self._last_pred = pred; self._last_x = x
        return x, ok, nit
    def integrate(self, *a, **k):
        return super().integrate(*a, **k)
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=1e-4, atol=1e-6, bounds=state_bounds(model), verbose=False, min_step=1e-9)
# monkeypatch error estimate bookkeeping: wrap rhs to count; record err contributions via a hook on accepted steps
orig_integrate = bdf.integrate
t0 = time.time()
T, X, info = bdf.integrate(np.arange(1520, 1541, 1.0), X0, breakpoints=model.spec_breakpoints())
print(info, time.time()-t0)
print("newton stats", stats, reasons)
# error-estimate contributors: recompute from last accepted step
pred, x = bdf._last_pred, bdf._last_x
scale = bdf.atol + bdf.rtol*np.abs(x)
contrib = np.abs(x - pred)/scale
for i in np.argsort(-contrib)[:8]:
    print(f"  {where(i)}: x={x[i]:.6g} pred={pred[i]:.6g} |x-pred|/scale={contrib[i]:.3g}")
# rhs magnitude at x
f = model.rhs(T[-1], X[:, -1])
for i in np.argsort(-np.abs(f))[:5]: print(f"  largest rhs: {where(i)} f={f[i]:.3g} x={X[i,-1]:.5g}")
E, Ys, inf, _ = model.assemble(T[-1], X[:, -1]); print("check valve", inf["check_valves"]); print("vertexA", X[model.n_pipe_states:, -1])
