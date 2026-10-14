"""Estatísticas de ciclo de trajetórias gravadas do severe slug (partida bloqueada, duto único).
uso: uv run python scratch/cycle_stats.py nome1 nome2 ...   (nomes em notebooks/data/, sem .npz)"""
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid

def blowdowns(T_, P_in, min_gap=200.0):
    lvl = P_in.min() + 0.3 * (P_in.max() - P_in.min()); low = P_in < lvl; out = []; i = 0
    while i < T_.size:
        if low[i]:
            j = i
            while j < T_.size and low[j]: j += 1
            k = i + int(np.argmin(P_in[i:j]))
            if not out or T_[k] - out[-1] > min_gap: out.append(T_[k])
            i = j
        else: i += 1
    return np.array(out)

fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D = 0.05; A = np.pi*D**2/4; P_out = 4.0; qL = 0.3*fluid.rhoL0*A; qG = 0.1*fluid.rhoG((P_out+1)*1e5)*A
print(f"{'trajetória':48s} {'passos':>7s} {'CPU':>6s} {'blowdowns':>24s} {'interv.':>7s} {'pico':>6s} {'P_in':>11s} {'prod.':>6s} {'frente':>6s}")
for name in sys.argv[1:]:
    d = np.load(f"notebooks/data/{name}.npz", allow_pickle=True); mb = json.loads(str(d["meta"]))
    reg = mb.get("regime", 4)
    P_out_b = mb.get("P_out", 4.0); qL = mb.get("vSL", 0.3)*fluid.rhoL0*A; qG = mb.get("vSG", 0.1)*fluid.rhoG((P_out_b+1)*1e5)*A
    case = severe_slug_case(method="fv_upwind", tf=d["T"][-1], dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out_b, L_h1=mb.get("L_h1", 20.0), L_down=mb.get("L_down", 300.0),
                            slope_down_deg=mb.get("slope", -3.0), H_riser=mb.get("H_riser", 20.0), L_h2=5.0, n_h1=mb.get("n_h1", 2), n_down=mb["n_down"], n_riser=mb["n_riser"], n_h2=2,
                            regime=reg, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked", elbow_radius=mb["elbow_radius"], n_elbow=mb["n_elbow"],
                            grading_down=mb.get("grading_down", 1.0), grading_riser=mb.get("grading_riser", 1.0), upstream_volume=mb.get("upstream_volume", 0.0),
                            extra={"regime_vertical": mb.get("regime_vertical", 0 if reg == 6 else 2), "k_switch": mb.get("k_switch", 1e6), "ipe_smooth": mb.get("ipe_smooth", False),
                                   "ipe_local": mb.get("ipe_local", False), "alpha_face": mb.get("alpha_face", "upwind"), "relax_target": mb.get("relax_target", "raw"),
                                   "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": mb["switch_frac"], "conservative": mb["conservative"],
                                   "vertical_blend": mb.get("vertical_blend", 0.0), "art_visc": mb.get("art_visc", 0.0), "low_void_regime": mb.get("low_void_regime", 0)})
    model = build_model(case); r = Results(model, d["T"], d["X"]); p = r.pipe(0)
    T = d["T"]; P1 = p["out"]["P_bar"][1]; q = p["out"]["qL"][-1]; aG = p["out"]["alphaG"]; x = p["x"]
    tb = blowdowns(T, P1); per = np.mean(np.diff(tb)) if tb.size > 1 else np.nan; sel = T > 100
    ib = int(np.argmin(np.abs(x - case.meta["s_elbow0"]))); widths = []
    for it in range(T.size):
        seg = aG[ib:, it]; xs = x[ib:]
        if seg.max() > 0.5 and seg.min() < 0.05:
            j1 = np.argmax(seg > 0.1); j9 = np.argmax(seg > 0.9) if (seg > 0.9).any() else len(seg) - 1
            if j9 > j1: widths.append(xs[j9] - xs[j1])
    short = name.replace("severe_slug_blocked_", "")
    print(f"{short:48s} {mb['n_steps']:7d} {mb['cpu']:5.0f}s {str(tb.round(0).astype(int).tolist()):>24s} {per:6.0f}s {q[sel].max():5.2f} {P1[sel].min():5.2f}-{P1[sel].max():4.2f} {q[sel].mean():6.3f} {np.median(widths) if widths else np.nan:5.1f}m")
