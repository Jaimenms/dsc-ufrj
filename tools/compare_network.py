"""Compara a rede de 4 dutos (vértices dinâmicos) com o duto único curvado no severe slug."""
import numpy as np, warnings, json, sys, os
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.analysis import mass_balance
from mfsim.fluid import Fluid
from mfsim.runs import load_run
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
def mk(single, vol=0.05):
    return severe_slug_case(method="fv_upwind", tf=3000, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                            n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=single, fluid=fluid, roughness=1e-5, start="blocked", elbow_radius=5.0, n_elbow=4,
                            extra={"regime_vertical": 2, "conservative": True, "vertical_blend": 20.0, "art_visc": 0.05, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "vertex_volume": vol})
def blowdowns(T, P, min_gap=60.0):
    lvl = P.min() + 0.3 * (P.max() - P.min()); low = P < lvl; out = []; i = 0
    while i < T.size:
        if low[i]:
            j = i
            while j < T.size and low[j]: j += 1
            k = i + int(np.argmin(P[i:j]))
            if not out or T[k] - out[-1] > min_gap: out.append(T[k])
            i = j
        else: i += 1
    return np.array(out)
rows = []
for tag, single, vol in [(sys.argv[1], True, 0.05)] + [(t, False, v) for t, v in zip(sys.argv[2::2], map(float, sys.argv[3::2]))]:
    run_ = load_run(tag); T, X = run_.T, run_.X; opts = run_.meta
    model = build_model(mk(single, vol)); res = Results(model, T, X)
    p_in = res.pipe(0)["out"]["P_bar"][1]; p_last = res.pipe(len(model.pipes) - 1)
    q_out = p_last["out"]["qL"][-1]
    # riser: duto 2 (0-based) na rede, ou células do riser no duto único
    if single:
        p0 = res.pipe(0); x = p0["x"]; s0 = model.case.meta["s_elbow0"]; ib = int(np.argmin(np.abs(x - s0))); aG_base = p0["out"]["alphaG"][ib + 5]
    else:
        aG_base = res.pipe(2)["out"]["alphaG"][1]
    # massa total de líquido: dutos + vértices dinâmicos
    mass = np.zeros(T.size); acc = np.zeros(T.size)
    for k, pb in enumerate(model.pipes):
        m, n, e = mass_balance(res, k, "L"); mass += m
    for vi in model.vA:
        iH, iP = model.vertex_A_index(model.vertices[vi].index); v = model.vertices[vi]
        rhoL = fluid.rhoL(np.maximum(X[iP] * model.PR, 1e3)); mass += v.volume * (1 - X[iH]) * rhoL
    accum = run_.acc
    npipes = len(model.pipes)
    if accum is not None and accum.shape[0] >= 2 * npipes + 2 * len(model.vA):
        # entrada - saída do sistema = soma dos acumuladores dos dutos + dos vértices (BDF-consistente)
        netL = accum[0:2 * npipes:2].sum(axis=0) + accum[2 * npipes::2].sum(axis=0)
    else:
        fin = res.pipe(0)["out"]["qL"][0]; fout = q_out
        netL = np.concatenate([[0.0], np.cumsum(0.5 * ((fin[1:] - fout[1:]) + (fin[:-1] - fout[:-1])) * np.diff(T))])
    tb = blowdowns(T, p_in); per = np.mean(np.diff(tb)) if tb.size > 1 else np.nan
    info = dict(tag=tag, rede=not single, vol=vol if not single else None, passos=opts.get("n_steps"), blowdowns=int(tb.size), periodo=float(per), pico=float(q_out[T > 100].max()),
                P_min=float(p_in[T > 100].min()), P_max=float(p_in[T > 100].max()), erro_massa=float(mass[-1] - mass[0] - netL[-1]))
    rows.append(info); print(json.dumps(info))
    os.makedirs("runs", exist_ok=True); np.savez(f"runs/cmpnet_{tag}.npz", T=T, p_in=p_in, q_out=q_out, aG_base=aG_base, mass=mass - mass[0])
