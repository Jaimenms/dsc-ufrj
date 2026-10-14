"""Saída densa do estado do severe slug (opções JSON como severe_D.py) entre t0 e t0+dur:
por nó e variável, a maior variação entre saídas consecutivas e os nós com fração extrema;
imprime também as células onde a relaxação da fase ausente está ativa.
uso: uv run python -u tools/diag_state.py '{json}' t0 dur dt"""

import numpy as np, warnings, time, sys, json

warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.fluid import Fluid
from mfsim.compiled import build_jax_rhs
from mfsim.integrators.implicit import ImplicitBDF

opts = json.loads(sys.argv[1])
t0, dur, dt = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
fluid = Fluid(
    rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=opts.get("gammaL", 100.0), gammaG=1.05, P0=1e5, sigma=0.05
)
D = 0.05
A = np.pi * D**2 / 4
P_out = 4.0
qL = 0.3 * fluid.rhoL0 * A
qG = 0.1 * fluid.rhoG((P_out + 1) * 1e5) * A
extra = {
    k: v
    for k, v in {
        "regime_vertical": opts.get("rv", 2),
        "switch_frac": opts.get("switch_frac", 0.5),
        "conservative": 1,
        "art_visc": opts.get("art_visc", 0.05),
        "vertical_blend": opts.get("vblend", 20.0),
        "relax_target": opts.get("relax_target"),
        "relax_time": opts.get("relax"),
        "bb_derivatives": opts.get("bb_derivatives"),
        "drift_relax": opts.get("drift_relax"),
    }.items()
    if v is not None
}
P_out = opts.get("P_out", 4.0)
qL = opts.get("vSL", 0.3) * fluid.rhoL0 * A
qG = opts.get("vSG", 0.1) * fluid.rhoG((P_out + 1) * 1e5) * A
case = severe_slug_case(
    method="fv_upwind",
    tf=t0 + dur,
    dt_out=2.0,
    qL=qL,
    qG=qG,
    D=D,
    P_out_bar=P_out,
    L_h1=20.0,
    L_down=opts.get("L_down", 300.0),
    slope_down_deg=-3.0,
    H_riser=20.0,
    L_h2=5.0,
    n_h1=2,
    n_down=opts.get("n_down", 15),
    n_riser=opts.get("n_riser", 5),
    n_h2=2,
    regime=opts.get("regime", 4),
    model=opts.get("model", 1),
    single_pipe=True,
    fluid=fluid,
    roughness=1e-5,
    start="blocked",
    elbow_radius=5.0,
    n_elbow=4,
    extra=extra,
)
case.set_options(q_switch_L=0.02 * qL, q_switch_G=0.02 * qG)
model = build_model(case)
X0 = model.initial_state()
rhs_j, jac_j = build_jax_rhs(model)
lo, hi = state_bounds(model)
atol = 0.01 * model.state_scales(X0)
kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
bdf = ImplicitBDF(
    rhs_j,
    S=model.jac_sparsity(),
    rtol=3e-3,
    atol=atol,
    bounds=(lo, hi),
    jac_fun=jac_j,
    min_step=1e-9,
    verbose=False,
    max_order=2,
    max_newton=20,
    newton_tol=1e-3,
    **kw_t,
)
bdf.slow_exit = False
t_eval = np.concatenate([np.arange(0, t0, min(0.5, t0 / 4)), np.arange(t0, t0 + dur + 0.5 * dt, dt)])
tw = time.time()
try:
    T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3)
    print("integração:", {k: info[k] for k in ("n_steps", "n_rejected")}, f"{time.time()-tw:.0f}s", flush=True)
except Exception as e:
    print("FAIL", e, bdf.n_steps, flush=True)
    X = np.array(bdf._out).T
    T = np.asarray(bdf._t_eval)[: X.shape[1]]
X = np.asarray(X)
X = X.T if X.shape[0] != T.size else X
pb = model.pipes[0]
sch = pb.scheme
N = sch.N
sel = T >= t0 - 1e-9
Ys = []
for it in np.flatnonzero(sel):
    E = np.array([[tb(T[it]) for tb in row] for row in model.tables])
    Ys.append(pb.assemble(X[it, pb.offset : pb.offset + pb.n_states], E[pb.i_from], E[pb.i_to]))
Ys = np.array(Ys)
Td = T[sel]
np.set_printoptions(precision=3, linewidth=230, suppress=True)
print("t:", Td[0], "->", Td[-1], "saídas:", Td.size, "| theta:", np.rad2deg(sch.theta_nodes).round(0))
dY = np.abs(np.diff(Ys, axis=0)).max(axis=0)
for s, nm in ((0, "alphaG"), (1, "PT"), (2, "qL"), (3, "qG")):
    print(f"max |delta {nm}| entre saídas por nó:", dY[:, s])
print("alphaG no fim:", Ys[-1, :, 0])
print("qL no fim:", Ys[-1, :, 2])
print("qG no fim:", Ys[-1, :, 3])
band = sch.alpha_band
print(
    "células com alphaG > 1 - alpha_band (líquido virtual relaxado):",
    np.flatnonzero(Ys[-1, :, 0] > 1 - band),
    "| < alpha_band (gás virtual):",
    np.flatnonzero(Ys[-1, :, 0] < band),
)
# série temporal nos 3 nós mais ativos em qL/qG
act = np.argsort(dY[:, 2] + dY[:, 3])[-3:]
for i in act:
    print(
        f"nó {i}: qL {Ys[::max(1, Td.size//12), i, 2].round(4)}  qG {Ys[::max(1, Td.size//12), i, 3].round(5)}  aG {Ys[::max(1, Td.size//12), i, 0].round(4)}"
    )
