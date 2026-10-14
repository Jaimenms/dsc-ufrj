import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from mfsim.io_legacy import load_case, legacy_states
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
rtol = float(sys.argv[1]); atol = rtol*1e-2
case, raw = load_case("CASO10-3-2b.mf2")
Tl, Xl = legacy_states(raw)
model = build_model(case)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=rtol, atol=atol, bounds=state_bounds(model), verbose=True, min_step=1e-9)
t0 = time.time()
T, X, info = bdf.integrate(Tl, Xl[:,0], breakpoints=model.spec_breakpoints())
print("OK", info, time.time()-t0, flush=True)
np.savez(f"scratch/X_10_3_2b_rtol{rtol:g}.npz", T=T, X=X)
for k, pb in enumerate(model.pipes):
    slots = pb.free % 4; seg = slice(pb.offset, pb.offset+pb.n_states)
    print(f" pipe{k+1}:", " ".join(f"{nm}:{np.max(np.abs(X[seg][slots==s]-Xl[seg][slots==s])):.2e}/{np.ptp(Xl[seg][slots==s]):.2e}" for s,nm in [(1,'PT'),(2,'qL'),(3,'qG')] if (slots==s).any()), flush=True)
