import numpy as np, warnings, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, superficial_velocities, boe_criterion
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.fluid import Fluid
opts = json.loads(sys.argv[1]); tag = sys.argv[2]
tf = opts.get("tf", 4000.0)
fluid = Fluid(rhoL0=opts.get("rhoL", 900.0), rhoG0=opts.get("rhoG0", 1.2), muL=opts.get("muL", 2e-3), muG=opts.get("muG", 1.8e-5), gammaL=opts.get("gammaL", 1000.0), gammaG=opts.get("gammaG", 1.05), P0=opts.get("P0", 1e5), sigma=opts.get("sigma", 0.05))
D = opts.get("D", 0.05); A = np.pi*D**2/4
P_out = opts.get("P_out", 4.0)
P_est = opts.get("P_gas_ref", P_out + 1.0)*1e5     # pressão em que vSG é dada (bar); laboratório: condições padrão (1.013)
qL = opts.get("vSL", 0.3)*fluid.rhoL0*A
qG = opts.get("vSG", 0.1)*fluid.rhoG(P_est)*A
start = opts.get("start", "guess")
dt_out = opts.get("dt_out", 2.0)
case = severe_slug_case(method=opts.get("method", "fv_upwind"), tf=tf, dt_out=dt_out, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=opts.get("L_h1", 20.0),
                        L_down=opts.get("L_down", 300.0), slope_down_deg=opts.get("slope", -3.0), H_riser=opts.get("H", 20.0), L_h2=opts.get("L_h2", 5.0),
                        n_h1=opts.get("n_h1", 2), n_down=opts.get("n_down", 30), n_riser=opts.get("n_riser", 10), n_h2=opts.get("n_h2", 2), regime=opts.get("regime", 4), single_pipe=not opts.get("network", 0),
                        fluid=fluid, roughness=opts.get("roughness", 1e-5), start=start, t_ramp=opts.get("t_ramp", 100.0),
                        pocket_alpha=opts.get("pocket_alpha"), front_liquid_length=opts.get("front", 0.0), dP_margin_bar=opts.get("dP_margin", 0.3),
                        elbow_radius=opts.get("elbow_R", 0.0), n_elbow=opts.get("n_elbow", 0),
                        grading_down=opts.get("g_down", 1.0), grading_riser=opts.get("g_riser", 1.0), model=opts.get("model", 1), upstream_volume=opts.get("V_up", 0.0),
                        extra={k: v for k, v in {"regime_vertical": opts.get("rv", 0 if opts.get("regime", 4) == 6 else 2), "k_switch": opts.get("k_switch"), "ipe_smooth": opts.get("ipe_smooth"), "ipe_local": opts.get("ipe_local"), "alpha_face": opts.get("alpha_face"), "fv_order": opts.get("fv_order"), "relax_target": opts.get("relax_target"), "drift_relax": opts.get("drift_relax"), "bb_derivatives": opts.get("bb_derivatives"), "ic_holdup": opts.get("ic_holdup"), "fi_mult": opts.get("fi_mult"), "fi_vgt": opts.get("fi_vgt"), "virtual_mass": opts.get("virtual_mass"), "vm_lambda": opts.get("vm_lambda"), "bubble_drag": opts.get("bubble_drag"), "q_switch_L": opts.get("fsw", 0.02)*qL, "q_switch_G": opts.get("fsw_G", opts.get("fsw", 0.02))*qG, "switch_frac": opts.get("switch_frac"),
                                                 "relax_time": opts.get("relax"), "limit_alpha": opts.get("limit"), "conservative": opts.get("conservative"),
                                                 "limit_band": opts.get("band"), "flux_band": opts.get("flux_band"), "flux_eps": opts.get("flux_eps"),
                                                 "flux_limit": opts.get("flux_limit"), "vertical_blend": opts.get("vblend"), "art_visc": opts.get("art_visc"),
                                                 "low_void_regime": opts.get("low_void"), "high_void_regime": opts.get("high_void"), "high_void_alpha": opts.get("high_void_alpha"), "interfacial_pressure": opts.get("delta"),
                                                 "vertex_volume": opts.get("vol")}.items() if v is not None})
vSL, vSG = superficial_velocities(case); m = case.meta
P_base = m["P_out_bar"]*1e5 + fluid.rhoL0*9.81*m["H_riser"]
print("qL, qG:", qL, qG, "vSL, vSG(out):", vSL, vSG, "Boe:", boe_criterion(vSL, vSG, P_base, fluid.rhoL0, 0.5, m["L_h1"] + m["L_down"]), flush=True)
model = build_model(case); X0 = model.initial_state() if start in ("shut_in", "blocked") else initial_guess(model)
print("meta:", case.meta, flush=True)
lo, hi = state_bounds(model)
if opts.get("tightbounds", 0):
    pb = model.pipes[0]; slots = pb.free % 4; lo[slots == 0] = 1e-3; hi[slots == 0] = 1 - 1e-3
kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if model.conservative else {}
atol = opts.get("rtol", 1e-3) * 1e-2 * np.ones(X0.size)
_slots = model.pipes[0].free % 4
if opts.get("atol_P"): atol[_slots == 1] = opts["atol_P"]          # tolerância absoluta em PT (P/PR) nas células
if opts.get("atol_q"): atol[(_slots == 2) | (_slots == 3)] = opts["atol_q"]   # idem nas vazões (kg/s)
if opts.get("tol_range"): atol = opts["tol_range"] * model.state_scales(X0)    # tolerância absoluta como fração da faixa
fun_, jac_ = model.rhs, None
if opts.get("jax"):
    from mfsim.jaxrhs import build_jax_rhs
    fun_, jac_ = build_jax_rhs(model)
bdf = ImplicitBDF(fun_, S=model.jac_sparsity(), rtol=opts.get("rtol", 1e-3), atol=atol, bounds=(lo, hi), jac_fun=jac_,
                  min_step=1e-9, verbose=True, max_order=opts.get("order", 2), max_newton=opts.get("max_newton", 12),
                  newton_tol=opts.get("newton_tol", 1e-3), **kw_t)
print("conservative:", model.conservative, flush=True)
bdf.accum = model.boundary_net_fluxes      # integral BDF-consistente dos fluxos de contorno (verificação exata do balanço)
bdf.slow_exit = bool(opts.get("slow", 0))
if opts.get("predict_space"): bdf.predict_space = opts["predict_space"]
if opts.get("jt_refresh"): bdf.jt_refresh = opts["jt_refresh"]
t0 = time.time()
try:
    T, X, info = bdf.integrate(np.arange(0, tf+dt_out/2, dt_out), X0, first_step=1e-3, breakpoints=model.spec_breakpoints()); print("OK", info, time.time()-t0, flush=True)
except Exception as e:
    print("FAIL", e, "steps", bdf.n_steps, "rej", bdf.n_rejected, time.time()-t0, flush=True)
    # guarda o que foi integrado (análise do ponto de falha e dos ciclos já obtidos)
    try:
        Xp = np.array(bdf._out).T; Tp = np.asarray(bdf._t_eval)[:Xp.shape[1]]
        np.savez(f"scratch/severe_D_{tag}_partial.npz", T=Tp, X=Xp, opts=json.dumps(opts)); print("parcial gravado:", Tp.size, "saídas até t =", Tp[-1], flush=True)
    except Exception as e2:
        print("sem parcial:", e2, flush=True)
    sys.exit(1)
np.savez(f"scratch/severe_D_{tag}.npz", T=T, X=X, opts=json.dumps(opts), acc=info["accum"])
if opts.get("network"):
    print("rede: pós-processamento no notebook/compare_network.py", flush=True); sys.exit(0)
res = Results(model, T, X); p = res.pipe(0); x = p["x"]; aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qLo = p["out"]["qL"]; qGo = p["out"]["qG"]
ib = int(np.argmin(np.abs(x - (m["L_h1"]+m["L_down"]))))
sel = T > tf/2
print(f"qL_out: min {qLo[-1][sel].min():.3f} max {qLo[-1][sel].max():.3f} std {qLo[-1][sel].std():.3f} | P_in min {P[1][sel].min():.3f} max {P[1][sel].max():.3f}", flush=True)
# conservação de massa de líquido: estoque vs integral (entrada - saída)
h_el = model.pipes[0].mesh["h"]; rhoL = fluid.rhoL(P[1:-1]*1e5)
mass = np.sum((1 - aG[1:-1]) * rhoL * A * h_el[:, None], axis=0)
if model.conservative:
    # fluxos de massa efetivos (vazões x limitador da célula doadora) nas faces de contorno
    sch = model.pipes[0].scheme
    FL = np.array([sch.mass_fluxes_nodal(p["Y"][it])[0] for it in range(T.size)])
    qin, qout = FL[:, 0], FL[:, -1]
else:
    qin, qout = qLo[0], qLo[-1]
net = np.concatenate([[0.0], np.cumsum(0.5*((qin[1:]-qout[1:]) + (qin[:-1]-qout[:-1])) * np.diff(T))])
print(f"massa líquida: inicial {mass[0]:.1f} kg, final {mass[-1]:.1f} kg; integral (entrada-saída) {net[-1]:.1f} kg; erro {mass[-1]-mass[0]-net[-1]:.1f} kg", flush=True)
print(f"balanço BDF-consistente: líquido erro {mass[-1]-mass[0]-info['accum'][0,-1]:.4f} kg; gás: estoque inicial {np.sum(aG[1:-1,0]*fluid.rhoG(P[1:-1,0]*1e5)*A*h_el):.4f} kg, erro {np.sum(aG[1:-1,-1]*fluid.rhoG(P[1:-1,-1]*1e5)*A*h_el)-np.sum(aG[1:-1,0]*fluid.rhoG(P[1:-1,0]*1e5)*A*h_el)-info['accum'][1,-1]:.5f} kg", flush=True)
for it in range(0, T.size, max(1, T.size//60)):
    print(f"t={T[it]:6.0f} P_in={P[1,it]:6.3f} P_base={P[ib,it]:6.3f} aG_line={aG[5,it]:.3f}/{aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[min(ib+5,aG.shape[0]-1),it]:.3f}/{aG[min(ib+10,aG.shape[0]-1),it]:.3f} qL_out={qLo[-1,it]:6.3f} qG_out={qGo[-1,it]:.5f}", flush=True)
