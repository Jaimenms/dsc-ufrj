"""Gas lift com intermitência mais forte: choke de gás mais fechada, simulação longa."""
import numpy as np, warnings, time, sys, copy
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
ab = float(sys.argv[1]); tf = float(sys.argv[2]); tag = sys.argv[3]
rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3; order = int(sys.argv[5]) if len(sys.argv) > 5 else 2
case, raw = load_chapter10_case("10.3.2b")
case.settings.tf = tf
for v in case.valves:
    if v.vertex == 6:
        v.opening = (np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, ab, ab]))
model = build_model(case); X0 = model.initial_state()
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=rtol, atol=rtol*1e-2, bounds=state_bounds(model), verbose=True, min_step=1e-9, max_order=order)
t0 = time.time()
T, X, info = bdf.integrate(np.arange(0, tf+1, 5.0), X0, breakpoints=model.spec_breakpoints())
print("OK", info, time.time()-t0, flush=True)
np.savez(f"scratch/gl_strong_{tag}.npz", T=T, X=X)
res = Results(model, T, X)
v = res.elements["valves"][4]; cv = res.elements["check_valves"][8]
qL = v["qL"]; qG = v["qG"]
sel = T > 1000
print(f"produção de líquido (choke): média {qL[sel].mean():.3f} min {qL[sel].min():.3f} max {qL[sel].max():.3f} std {qL[sel].std():.3f}")
print(f"gás injetado (válvula de retenção): média {cv['qG'][sel].mean():.4f} min {cv['qG'][sel].min():.4f} max {cv['qG'][sel].max():.4f}")
for it in range(0, T.size, max(1, T.size//50)):
    print(f"t={T[it]:6.0f} qL_choke={qL[it]:7.3f} qG_choke={qG[it]:.4f} ZP={v['ZP'][it]:7.2f} qG_inj={cv['qG'][it]:.4f} P_vert8={cv['ZP'][it]:7.2f}", flush=True)
