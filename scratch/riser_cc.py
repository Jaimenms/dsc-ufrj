"""Diagnóstico do filme no riser em trajetórias gravadas (partida bloqueada, duto único):
fração do tempo em contracorrente (filme descendo, gás subindo) e com riser quase cheio de gás,
e velocidade média do filme nessas células. uso: uv run python scratch/riser_cc.py nome1 nome2 ..."""
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D = 0.05; A = np.pi*D**2/4; P_out = 4.0; qL = 0.3*fluid.rhoL0*A; qG = 0.1*fluid.rhoG((P_out+1)*1e5)*A
for name in sys.argv[1:]:
    d = np.load(f"notebooks/data/{name}.npz", allow_pickle=True); mb = json.loads(str(d["meta"])); reg = mb.get("regime", 4)
    P_out_b = mb.get("P_out", 4.0); qL = mb.get("vSL", 0.3)*fluid.rhoL0*A; qG = mb.get("vSG", 0.1)*fluid.rhoG((P_out_b+1)*1e5)*A
    case = severe_slug_case(method="fv_upwind", tf=d["T"][-1], dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out_b, L_h1=mb.get("L_h1", 20.0), L_down=mb.get("L_down", 300.0),
                            slope_down_deg=mb.get("slope", -3.0), H_riser=mb.get("H_riser", 20.0), L_h2=5.0, n_h1=mb.get("n_h1", 2), n_down=mb["n_down"], n_riser=mb["n_riser"], n_h2=2, regime=reg,
                            single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked", elbow_radius=mb["elbow_radius"], n_elbow=mb["n_elbow"],
                            grading_down=mb.get("grading_down", 1.0), grading_riser=mb.get("grading_riser", 1.0), upstream_volume=mb.get("upstream_volume", 0.0),
                            extra={"regime_vertical": mb.get("regime_vertical", 0 if reg == 6 else 2), "k_switch": mb.get("k_switch", 1e6), "ipe_smooth": mb.get("ipe_smooth", False), "ipe_local": mb.get("ipe_local", False),
                                   "relax_target": mb.get("relax_target", "raw"), "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": mb["switch_frac"], "conservative": mb["conservative"],
                                   "vertical_blend": mb.get("vertical_blend", 0.0), "art_visc": mb.get("art_visc", 0.0), "low_void_regime": mb.get("low_void_regime", 0),
                                   "high_void_regime": mb.get("high_void_regime", 0), "high_void_alpha": mb.get("high_void_alpha", 0.9), "virtual_mass": mb.get("virtual_mass", 0.0),
                                   "interfacial_pressure": mb.get("interfacial_pressure", 0.0)})
    model = build_model(case); res = Results(model, d["T"], d["X"]); p = res.pipe(0); sch = model.pipes[0].scheme
    th = np.rad2deg(sch.theta_nodes); riser = np.flatnonzero(th > 80); T = d["T"]
    aG = p["out"]["alphaG"][riser]; vL = p["out"]["velocidadeL"][riser]; vG = p["out"]["velocidadeG"][riser]
    gasfull = aG > 0.9
    cc = gasfull & (vL < -0.02) & (vG > 0.02)
    up = gasfull & (vL > 0.5)
    short = name.replace("severe_slug_blocked_conservative_arc_", "")
    print(f"{short:28s}: riser com >90% gás {gasfull.mean():5.1%} do tempo | nessas células: filme descendo {cc.sum()/max(1, gasfull.sum()):5.1%}, líquido subindo > 0.5 m/s {up.sum()/max(1, gasfull.sum()):5.1%} | vL médio {vL[gasfull].mean() if gasfull.any() else np.nan:+.2f} m/s, vG médio {vG[gasfull].mean() if gasfull.any() else np.nan:+.2f} m/s")
