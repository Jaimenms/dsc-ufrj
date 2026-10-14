import numpy as np, time
from scipy.integrate import solve_ivp
from mfsim.io_legacy import load_case, legacy_states
from mfsim.simulation import build_model
case, raw = load_case("CASO10-1-1.mf2")
model = build_model(case)
Tl, Xl = legacy_states(raw)
X0 = Xl[:, 0]
S = model.jac_sparsity()
n=[0]
def f(t,x):
    n[0]+=1; return model.rhs(t,x)
t0=time.time()
sol = solve_ivp(f, (0, 300), X0, method="BDF", rtol=1e-4, atol=1e-6, jac_sparsity=S, t_eval=Tl[Tl<=300])
print("BDF 0-300s: steps", sol.t.size, "nfev", sol.nfev, "njev", sol.njev, "nlu", sol.nlu, "rhs calls", n[0], "time", time.time()-t0)
# step history
import scipy
sol2 = solve_ivp(f, (0, 300), X0, method="BDF", rtol=1e-4, atol=1e-6, jac_sparsity=S, dense_output=False)
print("internal steps:", sol2.t.size, "min dt", np.min(np.diff(sol2.t)), "median dt", np.median(np.diff(sol2.t)))
