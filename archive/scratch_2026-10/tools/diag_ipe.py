"""Diagnóstico do IPE em dinâmica: integra o severe slug (opções JSON como severe_D.py) até t0,
depois com saída densa (dt) até t0+dur; imprime por nó o número de mudanças grandes dos pesos,
mudanças de hD_est e o passo médio, e a série temporal no nó mais ativo.
uso: uv run python -u tools/diag_ipe.py '{json}' t0 dur dt"""

import numpy as np, warnings, time, sys, json

warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.fluid import Fluid
from mfsim.compiled import build_jax_rhs
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.closures import identify_regime, equilibrium_fractions

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
        "regime_vertical": opts.get("rv", 0 if opts.get("regime", 4) == 6 else 2),
        "k_switch": opts.get("k_switch"),
        "ipe_smooth": opts.get("ipe_smooth"),
        "ipe_local": opts.get("ipe_local"),
        "q_switch_L": 0.02 * qL,
        "q_switch_G": 0.02 * qG,
        "switch_frac": opts.get("switch_frac", 0.5),
        "conservative": 1,
        "art_visc": opts.get("art_visc", 0.05),
        "vertical_blend": opts.get("vblend"),
        "low_void_regime": opts.get("low_void"),
        "interfacial_pressure": opts.get("delta"),
    }.items()
    if v is not None
}
case = severe_slug_case(
    method="fv_upwind",
    tf=t0 + dur,
    dt_out=2.0,
    qL=qL,
    qG=qG,
    D=D,
    P_out_bar=P_out,
    L_h1=20.0,
    L_down=300.0,
    slope_down_deg=-3.0,
    H_riser=20.0,
    L_h2=5.0,
    n_h1=2,
    n_down=opts.get("n_down", 15),
    n_riser=opts.get("n_riser", 5),
    n_h2=2,
    regime=opts.get("regime", 4),
    single_pipe=True,
    fluid=fluid,
    roughness=1e-5,
    start="blocked",
    elbow_radius=opts.get("elbow_R", 5.0),
    n_elbow=opts.get("n_elbow", 4),
    extra=extra,
)
model = build_model(case)
X0 = model.initial_state()
rhs_j, jac_j = build_jax_rhs(model)
lo, hi = state_bounds(model)
atol = opts.get("tol_range", 0.01) * model.state_scales(X0)
kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
bdf = ImplicitBDF(
    rhs_j,
    S=model.jac_sparsity(),
    rtol=opts.get("rtol", 3e-3),
    atol=atol,
    bounds=(lo, hi),
    jac_fun=jac_j,
    min_step=1e-9,
    verbose=False,
    max_order=2,
    max_newton=opts.get("max_newton", 20),
    newton_tol=1e-3,
    **kw_t,
)
bdf.slow_exit = False
t_eval = np.concatenate([np.arange(0, t0, 5.0), np.arange(t0, t0 + dur + 0.5 * dt, dt)])
tw = time.time()
T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3)
X = np.asarray(X)
X = X.T if X.shape[0] != T.size else X
print("integração:", info, f"{time.time()-tw:.0f}s", flush=True)
pb = model.pipes[0]
sch = pb.scheme
mod = pb.model
p = mod.params
sel = T >= t0 - 1e-9
rows = []
for it in np.flatnonzero(sel):
    E = np.array([[tb(T[it]) for tb in row] for row in model.tables])
    Y = pb.assemble(X[it, pb.offset : pb.offset + pb.n_states], E[pb.i_from], E[pb.i_to])
    inp, P, Area = mod.local_state(Y, sch.theta_nodes)
    eq = equilibrium_fractions(inp, smooth=p.ipe_smooth)
    fl, fac = identify_regime(inp, k_switch=p.k_switch, eq=eq, smooth=p.ipe_smooth, local=p.ipe_local)
    rows.append(
        dict(
            t=T[it],
            aG=Y[:, 0].copy(),
            vLS=inp.vLS.copy(),
            vGS=inp.vGS.copy(),
            hD=eq[0].copy(),
            aG_bol=eq[4].copy(),
            FGs=fac["FGcosteta"].copy(),
            FGc=fac["FGast_c"].copy(),
            FG=fac["FG"].copy(),
            FG_c=fac["FG_c"].copy(),
            W=np.stack([fl.Est, fl.Anu, fl.Bub, fl.Int]),
            qL=Y[:, 2].copy(),
            qG=Y[:, 3].copy(),
            P=Y[:, 1].copy(),
        )
    )
W = np.array([r["W"] for r in rows])
hD = np.array([r["hD"] for r in rows])
ab = np.array([r["aG_bol"] for r in rows])
np.set_printoptions(precision=2, linewidth=230)
th = np.rad2deg(sch.theta_nodes)
print("theta:", th.round(0))
print("|dW|>0.2 por nó:", np.sum(np.abs(np.diff(W, axis=0)) > 0.2, axis=(0, 1)))
print("soma |dW| por nó:", np.sum(np.abs(np.diff(W, axis=0)), axis=(0, 1)).round(1))
print("mudanças hD_est>1e-3 por nó:", np.sum(np.abs(np.diff(hD, axis=0)) > 1e-3, axis=0))
print("pesos médios Est/Anu/Bub/Int:")
print(W.mean(axis=0).round(2))
print("aG médio:", np.array([r["aG"] for r in rows]).mean(axis=0).round(3))
act = np.argmax(np.sum(np.abs(np.diff(W, axis=0)), axis=(0, 1)))
print(f"nó mais ativo: {act} (theta {th[act]:.0f})")
print("   t      aG    vLS    vGS   hD_est aG_bol  FG*    FG*c    FG    FG_c   Est  Anu  Bub  Int    qL     qG")
for r in rows[:: max(1, len(rows) // 60)]:
    i = act
    print(
        f"{r['t']:8.3f} {r['aG'][i]:.3f} {r['vLS'][i]:6.3f} {r['vGS'][i]:6.3f} {r['hD'][i]:.4f} {r['aG_bol'][i]:.3f} {r['FGs'][i]:.4f} {r['FGc'][i]:.4f} {r['FG'][i]:.3f} {r['FG_c'][i]:.3f} "
        f"{r['W'][0][i]:.2f} {r['W'][1][i]:.2f} {r['W'][2][i]:.2f} {r['W'][3][i]:.2f} {r['qL'][i]:6.3f} {r['qG'][i]:.4f}"
    )
