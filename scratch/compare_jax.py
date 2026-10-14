"""Compara numpy e JAX num caso dos notebooks: estado estacionário (opcional) e simulação."""
import numpy as np, warnings, time, sys, json, copy
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import simulate, steady_state, build_model
cid = sys.argv[1]; opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
rtol, atol = opts.get("rtol", 1e-4), opts.get("atol", 1e-6)
out = {"case": cid}
def load():
    case, raw = load_chapter10_case(cid)
    if opts.get("method"): case.settings.method = opts["method"]
    if opts.get("tf"): case.settings.tf = opts["tf"]
    if opts.get("choke") is not None:
        tf = case.settings.tf
        for v in case.valves:
            if v.vertex == 6: v.opening = (np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, opts["choke"], opts["choke"]]))
    return case
res = {}
case0 = load()
if opts.get("steady"):
    # estado estacionário nas duas versões (tempos), mas a simulação parte do MESMO
    # estado (o do numpy): o estacionário discreto pode não ser único
    for mode in ("numpy", "jax"):
        c = copy.deepcopy(case0)
        if mode == "jax":
            from mfsim.jaxrhs import get_jax
            get_jax(build_model(c))      # compilação fora da cronometragem
        t0 = time.time()
        model, X, info = steady_state(c, X0=opts.get("X0"), verbose=False, jax=(mode == "jax"))
        out[f"steady_{mode}"] = dict(cpu=time.time() - t0, iters=info.get("iterations"), residual=float(info.get("residual", np.nan)))
        if mode == "numpy":
            case0 = c
for mode in ("numpy", "jax"):
    case = copy.deepcopy(case0)
    if mode == "jax":
        from mfsim.jaxrhs import get_jax
        m = build_model(case); t0 = time.time(); rhs_j, jac_j = get_jax(m); x0 = m.initial_state(); rhs_j(0.0, x0); jac_j(0.0, x0)
        out["jax_compile"] = time.time() - t0
    t0 = time.time()
    r = simulate(case, integrator="implicit", rtol=rtol, atol=atol, max_step=np.inf, progress=False, jax=(mode == "jax"))
    out[f"sim_{mode}"] = dict(cpu=time.time() - t0, steps=r.info["n_steps"], rej=r.info["n_rejected"], jac=r.info["n_jac"])
    res[mode] = r
a, b = res["numpy"], res["jax"]
n = min(a.T.size, b.T.size)
scale = np.max(np.abs(a.X), axis=1) + 1e-12
diff = np.abs(a.X[:, :n] - b.X[:, :n]) / scale[:, None]
out["max_rel_diff"] = float(diff.max()); out["final_rel_diff"] = float(diff[:, -1].max())
print(json.dumps(out))
