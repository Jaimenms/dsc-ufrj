import numpy as np, warnings, time, sys, copy
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import simulate
cid = sys.argv[1]; rtol = float(sys.argv[2]); tf = float(sys.argv[3]) if len(sys.argv) > 3 else None
res = {}
for mode in ("fd", "exact"):
    case, raw = load_chapter10_case(cid)
    if tf:
        case.settings.tf = tf
        for v in case.valves:
            if v.vertex == 6: v.opening = (np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, 0.12, 0.12]))
    for pipe in case.pipes: pipe.extra["bb_derivatives"] = mode
    t0 = time.time(); r = simulate(case, integrator="implicit", rtol=rtol, atol=rtol * 1e-2, max_step=np.inf, progress=False, jax=True); cpu = time.time() - t0
    res[mode] = (r, cpu)
a, b = res["fd"][0], res["exact"][0]; n = min(a.X.shape[1], b.X.shape[1]); sc = np.max(np.abs(a.X), axis=1) + 1e-12
print(f"{cid} (rtol {rtol}): fd {res['fd'][1]:.1f} s / {a.info['n_steps']} passos | exact {res['exact'][1]:.1f} s / {b.info['n_steps']} passos | diferença máx rel dos estados {np.max(np.abs(a.X[:, :n] - b.X[:, :n]) / sc[:, None]):.1e}, final {np.max(np.abs(a.X[:, n-1] - b.X[:, n-1]) / sc):.1e}")
