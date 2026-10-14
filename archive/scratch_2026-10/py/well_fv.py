"""Poço 10.3.1 (B&B): transiente de fechamento/abertura do choke com o esquema nodal da tese
(fd_central) e com o volume finito escalonado de deriva (fv_upwind conservativo); grava
scratch/well_fv_<tag>.npz com T, qL_out, qG_out, P_bottom, P_top, estoque de massa e balanço."""
import numpy as np, warnings, copy, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import build_model, simulate
from mfsim.analysis import mass_balance
tag = sys.argv[1]; opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
case, raw = load_chapter10_case(opts.get("case", "10.3.1"))
if opts.get("fv"):
    case.settings.method = "fv_upwind"
    for pipe in case.pipes:
        if pipe.model == 2:
            pipe.extra.update({"conservative": True, "q_switch": opts.get("q_switch", 0.05), "drift_relax": opts.get("drift_relax", 0.05), "art_visc": opts.get("art_visc", 0.0)})
t0 = time.time()
if opts.get("fv") or opts.get("steady_start"):
    # parte do estacionário do próprio esquema (a CI nodal interpolada não está em equilíbrio no VF)
    from mfsim.simulation import steady_state
    steady_state(case, verbose=False, update_case=True)
    print(f"[{tag}] estacionário em {time.time()-t0:.1f}s", flush=True)
res = simulate(case, progress=False, jax=bool(opts.get("jax", 0)), rtol=opts.get("rtol"), tol_range=opts.get("tol_range"),
               integrator=opts.get("integrator"))
wall = time.time() - t0
p = res.pipe(0); T = res.T
out = dict(T=T, qL_out=p["out"]["qL"][-1], qG_out=p["out"]["qG"][-1], qL_in=p["out"]["qL"][0], P_bottom=p["out"]["P_bar"][1 if opts.get("fv") else 0], P_top=p["out"]["P_bar"][-2 if opts.get("fv") else -1], x=p["x"], aG_end=p["out"]["alphaG"][:, -1], P_end=p["out"]["P_bar"][:, -1])
info = res.info
print(f"[{tag}] passos {info.get('n_steps')} rej {info.get('n_rejected')} wall {wall:.1f}s | qL_out min/max {out['qL_out'].min():.2f}/{out['qL_out'].max():.2f} | P_bottom {out['P_bottom'].min():.1f}-{out['P_bottom'].max():.1f} bar | P_top {out['P_top'].min():.1f}-{out['P_top'].max():.1f}", flush=True)
if opts.get("fv"):
    for ph in ("L", "G"):
        mass, _, _ = mass_balance(res, 0, ph)
        print(f"   balanço {ph}: estoque {mass[0]:.2f} -> {mass[-1]:.2f} kg", flush=True)
np.savez(f"scratch/well_fv_{tag}.npz", **out, info=json.dumps({k: v for k, v in info.items() if k != "accum"}))
