"""Configuração rápida do integrador (atol nas vazões, Newton 20) no gas lift intermitente (JAX)."""
import numpy as np, warnings, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.jaxrhs import get_jax
opts = json.loads(sys.argv[1]); tf = 16000.0
case, raw = load_chapter10_case("10.3.2b"); case.settings.tf = tf
for v in case.valves:
    if v.vertex == 6: v.opening = (np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, 0.12, 0.12]))
model = build_model(case); X0 = model.initial_state(); rhs, jac = get_jax(model)
rtol = opts.get("rtol", 1e-3); atol = rtol * 1e-2 * np.ones(X0.size)
for pb in model.pipes:
    sl = pb.free % 4; seg = slice(pb.offset, pb.offset + pb.n_states)
    a = atol[seg]
    if opts.get("atol_q"): a[(sl == 2) | (sl == 3)] = opts["atol_q"]
    if opts.get("atol_P"): a[sl == 1] = opts["atol_P"]
    atol[seg] = a
if opts.get("tol_range"): atol = opts["tol_range"] * model.state_scales(X0)
bdf = ImplicitBDF(rhs, S=model.jac_sparsity(), rtol=rtol, atol=atol, bounds=state_bounds(model), min_step=1e-9, max_order=2, jac_fun=jac, max_newton=opts.get("max_newton", 12))
t0 = time.time(); T, X, info = bdf.integrate(np.arange(0, tf + 1, 5.0), X0, breakpoints=model.spec_breakpoints()); cpu = time.time() - t0
res = Results(model, T, X, info); q = res.elements["valves"][4]["qL"]; sel = T > 2000
pk = [i for i in range(1, T.size - 1) if q[i] > 15 and q[i] >= q[i-1] and q[i] > q[i+1]]
print(f"{json.dumps(opts):60s}: {info['n_steps']:5d} passos, {info['n_rejected']:4d} rej, {cpu:5.1f} s | picos > 15 kg/s em {np.round(T[pk]).astype(int).tolist()} | média {q[sel].mean():.2f} mín {q[sel].min():.2f} máx {q[sel].max():.2f} kg/s")
