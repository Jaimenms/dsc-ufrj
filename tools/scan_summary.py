"""Resumo da varredura da fronteira de intermitência severa: por trajetória gravada, vazões,
pressão, critério de Bøe, padrão da linha pela identificação da tese (estado da linha logo a
jusante da entrada), amplitude da pressão de entrada (t > 300 s), blowdowns e intervalo.
uso: uv run python tools/scan_summary.py nome1 nome2 ...  (nomes em notebooks/data/ ou runs/)"""
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, boe_criterion
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
from mfsim.closures import identify_regime, equilibrium_fractions
from mfsim.analysis import blowdowns
from mfsim.runs import load_run
fluid0 = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D = 0.05; A = np.pi*D**2/4
print(f"{'trajetória':34s} {'vSL':>5s} {'vSG':>5s} {'P_out':>5s} {'L':>5s} {'V_up':>5s} {'Bøe mín':>8s} {'linha (IPE)':>12s} {'passos':>7s} {'ΔP_in':>6s} {'blowdowns':>24s} {'interv.':>8s} {'pico':>6s}")
for name in sys.argv[1:]:
    run = load_run(name); mb = run.meta; fluid = run.case.fluid
    P_out = mb.get("P_out", 4.0); vSL = mb.get("vSL", 0.3); vSG_feed = mb.get("vSG", 0.1); L = mb.get("L_down", 300.0); H = mb.get("H_riser", 20.0); th = np.deg2rad(mb.get("slope", -3.0))
    qL = vSL*fluid.rhoL0*A; qG = vSG_feed*fluid.rhoG((P_out+1)*1e5)*A
    reg = mb.get("regime", 4)
    case = run.case; model = run.model; res = run.res; p = res.pipe(0); T = run.T; X_run = run.X
    P1 = p["out"]["P_bar"][1]; qout = p["out"]["qL"][-1]; sel = T > 300
    amp = P1[sel].max() - P1[sel].min()
    tb = blowdowns(T[sel], P1[sel]) if amp > 0.3 else np.array([]); per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    # Bøe com a pressão média na base do riser e vSG à pressão do separador
    P_base = P_out*1e5 + fluid.rhoL0*9.81*H; vSG_out = qG/(fluid.rhoG(P_out*1e5)*A)
    vmin, severe = boe_criterion(vSL, vSG_out, P_base, fluid.rhoL0, 0.5, mb.get("L_h1", 20.0) + L)   # L de gás = linha de entrada + descendente
    # padrão da linha: IPE da tese no estado médio da célula 5 (descendente) com a fração e vazões médias
    pb = model.pipes[0]; sch = pb.scheme; mod = pb.model
    it = np.flatnonzero(sel)[::10]
    Ymean = np.mean([pb.assemble(X_run[pb.offset:pb.offset+pb.n_states, i],
                                 np.array([[tb_(T[i]) for tb_ in row] for row in model.tables])[pb.i_from],
                                 np.array([[tb_(T[i]) for tb_ in row] for row in model.tables])[pb.i_to]) for i in it], axis=0)
    inp, _, _ = mod.local_state(Ymean, sch.theta_nodes)
    fl, _ = identify_regime(inp, k_switch=1e6, eq=equilibrium_fractions(inp))
    th_ = np.rad2deg(sch.theta_nodes); k = int(np.flatnonzero(np.abs(th_ - mb.get("slope", -3.0)) < 0.5)[len(np.flatnonzero(np.abs(th_ - mb.get("slope", -3.0)) < 0.5)) // 2])
    names_ = {1: "anular", 2: "bolhas", 3: "intermit.", 4: "estrat.", 5: "estrat.ond."}
    line = names_.get(int(round(float(fl.reg2[k]))), f"{float(fl.reg2[k]):.1f}")
    short = name.replace("severe_slug_blocked_conservative_arc_", "")
    print(f"{short:34s} {vSL:5.2f} {vSG_out:5.2f} {P_out:5.1f} {L:5.0f} {mb.get('upstream_volume', 0.0):5.1f} {vmin:8.2f} {line:>12s} {mb['n_steps']:7d} {amp:6.2f} {str(np.round(tb).astype(int).tolist()):>24s} {per:7.0f}s {qout[sel].max():6.2f}")
