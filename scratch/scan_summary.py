"""Resumo da varredura da fronteira de intermitência severa: por trajetória gravada, vazões,
pressão, critério de Bøe, padrão da linha pela identificação da tese (estado da linha logo a
jusante da entrada), amplitude da pressão de entrada (t > 300 s), blowdowns e intervalo.
uso: uv run python scratch/scan_summary.py nome1 nome2 ..."""
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, boe_criterion
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
from mfsim.closures import identify_regime, equilibrium_fractions
sys.path.insert(0, "scratch")
from cycle_stats import blowdowns
fluid0 = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D = 0.05; A = np.pi*D**2/4
print(f"{'trajetória':34s} {'vSL':>5s} {'vSG':>5s} {'P_out':>5s} {'L':>5s} {'V_up':>5s} {'Bøe mín':>8s} {'linha (IPE)':>12s} {'passos':>7s} {'ΔP_in':>6s} {'blowdowns':>24s} {'interv.':>8s} {'pico':>6s}")
for name in sys.argv[1:]:
    d = np.load(f"notebooks/data/{name}.npz", allow_pickle=True); mb = json.loads(str(d["meta"]))
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=mb.get("gammaL", 100.0), gammaG=mb.get("gammaG", 1.05), P0=1e5, sigma=0.05)
    P_out = mb.get("P_out", 4.0); vSL = mb.get("vSL", 0.3); vSG_feed = mb.get("vSG", 0.1); L = mb.get("L_down", 300.0); H = mb.get("H_riser", 20.0); th = np.deg2rad(mb.get("slope", -3.0))
    qL = vSL*fluid.rhoL0*A; qG = vSG_feed*fluid.rhoG((P_out+1)*1e5)*A
    reg = mb.get("regime", 4)
    case = severe_slug_case(method="fv_upwind", tf=d["T"][-1], dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=mb.get("L_h1", 20.0), L_down=L, slope_down_deg=mb.get("slope", -3.0),
                            H_riser=H, L_h2=5.0, n_h1=mb.get("n_h1", 2), n_down=mb["n_down"], n_riser=mb["n_riser"], n_h2=2, regime=reg, single_pipe=True, fluid=fluid, roughness=1e-5,
                            start="blocked", elbow_radius=mb["elbow_radius"], n_elbow=mb["n_elbow"], upstream_volume=mb.get("upstream_volume", 0.0),
                            extra={"regime_vertical": mb.get("regime_vertical", 2), "relax_target": mb.get("relax_target", "raw"), "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG,
                                   "switch_frac": mb["switch_frac"], "conservative": mb["conservative"], "vertical_blend": mb.get("vertical_blend", 0.0), "art_visc": mb.get("art_visc", 0.0),
                                   "low_void_regime": mb.get("low_void_regime", 0), "high_void_regime": mb.get("high_void_regime", 0), "virtual_mass": mb.get("virtual_mass", 0.0),
                                   "k_switch": mb.get("k_switch", 1e6), "ipe_smooth": mb.get("ipe_smooth", False), "ipe_local": mb.get("ipe_local", False)})
    model = build_model(case); res = Results(model, d["T"], d["X"]); p = res.pipe(0); T = d["T"]
    P1 = p["out"]["P_bar"][1]; qout = p["out"]["qL"][-1]; sel = T > 300
    amp = P1[sel].max() - P1[sel].min()
    tb = blowdowns(T[sel], P1[sel]) if amp > 0.3 else np.array([]); per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    # Bøe com a pressão média na base do riser e vSG à pressão do separador
    P_base = P_out*1e5 + fluid.rhoL0*9.81*H; vSG_out = qG/(fluid.rhoG(P_out*1e5)*A)
    vmin, severe = boe_criterion(vSL, vSG_out, P_base, fluid.rhoL0, 0.5, mb.get("L_h1", 20.0) + L)   # L de gás = linha de entrada + descendente
    # padrão da linha: IPE da tese no estado médio da célula 5 (descendente) com a fração e vazões médias
    pb = model.pipes[0]; sch = pb.scheme; mod = pb.model
    it = np.flatnonzero(sel)[::10]
    Ymean = np.mean([pb.assemble(d["X"][i, pb.offset:pb.offset+pb.n_states] if d["X"].shape[0] == T.size else d["X"][pb.offset:pb.offset+pb.n_states, i],
                                 np.array([[tb_(T[i]) for tb_ in row] for row in model.tables])[model.vpos[pb.pipe.v_in]],
                                 np.array([[tb_(T[i]) for tb_ in row] for row in model.tables])[model.vpos[pb.pipe.v_out]]) for i in it], axis=0)
    inp, _, _ = mod.local_state(Ymean, sch.theta_nodes)
    fl, _ = identify_regime(inp, k_switch=1e6, eq=equilibrium_fractions(inp))
    th_ = np.rad2deg(sch.theta_nodes); k = int(np.flatnonzero(np.abs(th_ - mb.get("slope", -3.0)) < 0.5)[len(np.flatnonzero(np.abs(th_ - mb.get("slope", -3.0)) < 0.5)) // 2])
    names_ = {1: "anular", 2: "bolhas", 3: "intermit.", 4: "estrat.", 5: "estrat.ond."}
    line = names_.get(int(round(float(fl.reg2[k]))), f"{float(fl.reg2[k]):.1f}")
    short = name.replace("severe_slug_blocked_conservative_arc_", "")
    print(f"{short:34s} {vSL:5.2f} {vSG_out:5.2f} {P_out:5.1f} {L:5.0f} {mb.get('upstream_volume', 0.0):5.1f} {vmin:8.2f} {line:>12s} {mb['n_steps']:7d} {amp:6.2f} {str(np.round(tb).astype(int).tolist()):>24s} {per:7.0f}s {qout[sel].max():6.2f}")
