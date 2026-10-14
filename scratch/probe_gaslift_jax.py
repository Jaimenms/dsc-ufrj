import numpy as np, warnings, time, sys, collections, json
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.jaxrhs import get_jax
mode = sys.argv[1]; tend = float(sys.argv[2])
case, raw = load_chapter10_case("10.3.2b"); tf = 16000.0; case.settings.tf = tf
for v in case.valves:
    if v.vertex == 6: v.opening = (np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, 0.12, 0.12]))
model = build_model(case); X0 = model.initial_state()
def name(i):
    for pb in model.pipes:
        if pb.offset <= i < pb.offset + pb.n_states:
            j = i - pb.offset; return f"duto{pb.pipe.index} nó{pb.free[j]//4} {'aG PT qL qG'.split()[pb.free[j]%4]}"
    return f"vértice A estado {i - model.n_pipe_states}"
stats = collections.Counter(); dom = collections.Counter(); byt = collections.Counter()
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats[("ok" if ok else self._fail_reason)] += 1
        if not ok and self._last_dx is not None:
            scale = self.atol + self.rtol*np.abs(pred); i = int(np.argmax(np.abs(self._last_dx)/scale)); dom[name(i)] += 1; byt[int(t//100)*100] += 1
        return x, ok, nit
fun, jac = (model.rhs, None) if mode == "numpy" else get_jax(model)
bdf = Dbg(fun, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-9, max_order=2, jac_fun=jac)
import os
bdf.slow_exit = bool(int(os.environ.get('SLOW_EXIT', '0')))
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tend + 1, 5.0), X0, breakpoints=model.spec_breakpoints()); print(f"{mode}: OK {info['n_steps']} passos, {info['n_rejected']} rej, {time.time()-t0:.1f} s")
except Exception as e:
    print(f"{mode}: FALHA {e} após {bdf.n_steps} passos, {time.time()-t0:.1f} s")
print("  newton:", dict(stats)); print("  dominantes nas falhas:", dom.most_common(6)); print("  falhas por janela de 100 s:", sorted(byt.items())[-8:])
