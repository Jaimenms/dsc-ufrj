"""Saída densa do estado de um caso de laboratório (opções JSON como severe_D.py) entre t0 e t0+dur: variação máxima
por nó e variável entre saídas consecutivas e estado final; tolera falha (usa o que foi integrado).
uso: uv run python -u scratch/diag_lab.py '{json}' t0 dur dt"""
import numpy as np, warnings, time, sys, json
warnings.filterwarnings("ignore"); sys.path.insert(0, "scratch")
from val_summary import rebuild
from mfsim.simulation import build_model, state_bounds
from mfsim.jaxrhs import build_jax_rhs
from mfsim.integrators.implicit import ImplicitBDF
opts = json.loads(sys.argv[1]); t0, dur, dt = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
case, fluid, A, qL, qG = rebuild(opts, None, t0 + dur); model = build_model(case); X0 = model.initial_state(); rhs_j, jac_j = build_jax_rhs(model)
lo, hi = state_bounds(model); atol = opts.get("tol_range", 0.01)*model.state_scales(X0)
kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
bdf = ImplicitBDF(rhs_j, S=model.jac_sparsity(), rtol=opts.get("rtol", 3e-3), atol=atol, bounds=(lo, hi), jac_fun=jac_j, min_step=1e-9, verbose=False, max_order=2, max_newton=opts.get("max_newton", 20), newton_tol=1e-3, **kw_t)
bdf.slow_exit = False
t_eval = np.concatenate([np.arange(0, t0, min(0.5, t0/4)), np.arange(t0, t0+dur+0.5*dt, dt)])
tw = time.time()
try:
    T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3); print("integração:", {k: info[k] for k in ("n_steps", "n_rejected")}, f"{time.time()-tw:.0f}s", flush=True)
except Exception as e:
    print("FAIL", e, bdf.n_steps, flush=True); T, X = np.asarray(bdf.T_out), np.asarray(bdf.X_out)
X = np.asarray(X); X = X.T if X.shape[0] != T.size else X
pb = model.pipes[0]; sch = pb.scheme
sel = T >= t0 - 1e-9
Ys = []
for it in np.flatnonzero(sel):
    E = np.array([[tb(T[it]) for tb in row] for row in model.tables])
    Ys.append(pb.assemble(X[it, pb.offset:pb.offset+pb.n_states], E[model.vpos[pb.pipe.v_in]], E[model.vpos[pb.pipe.v_out]]))
Ys = np.array(Ys); Td = T[sel]
np.set_printoptions(precision=3, linewidth=250, suppress=True)
print("t:", Td[0], "->", Td[-1], "saídas:", Td.size, "| theta:", np.rad2deg(sch.theta_nodes).round(0))
dY = np.abs(np.diff(Ys, axis=0)).max(axis=0)
for s, nm in ((0, "alphaG"), (1, "PT"), (2, "qL"), (3, "qG")):
    print(f"max |delta {nm}| entre saídas por nó:", dY[:, s])
print("alphaG no fim:", Ys[-1, :, 0]); print("PT no fim:", Ys[-1, :, 1]); print("qL no fim:", Ys[-1, :, 2]); print("qG no fim:", Ys[-1, :, 3])
act = np.argsort(dY[:, 2] + dY[:, 3] + dY[:, 0])[-4:]
for i in act:
    k = max(1, Td.size//12)
    print(f"nó {i}: qL {Ys[::k, i, 2].round(4)}  qG {Ys[::k, i, 3].round(6)}  aG {Ys[::k, i, 0].round(4)}  PT {Ys[::k, i, 1].round(5)}")
