import numpy as np, time, sys
from mfsim.io_legacy import load_case, legacy_states
from mfsim.simulation import build_model
from mfsim.integrators import integrate_scipy, integrate_implicit
from mfsim.simulation import state_bounds

case, raw = load_case("CASO10-1-1.mf2")
model = build_model(case)
Tl, Xl = legacy_states(raw)
X0 = Xl[:, 0]
pb = model.pipes[0]; slots = pb.free % 4
def report(X, label):
    for s, name in enumerate(["alphaG", "PT", "qL", "qG"]):
        m = slots == s
        print(f"  {label} {name}: max|diff|={np.max(np.abs(X[m]-Xl[m])):.3e}  legacy range [{Xl[m].min():.4g},{Xl[m].max():.4g}]")
which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("scipy", "both"):
    t0 = time.time()
    T, X, info = integrate_scipy(model, X0, Tl, rtol=1e-4, atol=1e-6, max_step=np.inf)
    print("scipy n_steps(t_eval)", info, time.time()-t0)
    report(X, "scipy")
    np.save("scratch/X_10_1_1_scipy.npy", X)
if which in ("implicit", "both"):
    t0 = time.time()
    T, X, info = integrate_implicit(model, X0, Tl, rtol=1e-4, atol=1e-6, max_step=np.inf, bounds=state_bounds(model), verbose=True)
    print(info, time.time()-t0)
    report(X, "implicit")
    np.save("scratch/X_10_1_1_implicit.npy", X)
