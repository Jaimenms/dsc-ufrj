import numpy as np, time
from mfsim.io_legacy import load_case, legacy_states
from mfsim.simulation import build_model, simulate

case, raw = load_case("CASO10-1-1.mf2")
print("settings:", case.settings)
print("pipe bc:", case.pipes[0].bc_in, case.pipes[0].bc_out, "n_el", case.pipes[0].n_elements, "ic", None if case.pipes[0].ic is None else case.pipes[0].ic.shape)
model = build_model(case)
X0 = model.initial_state()
Tl, Xl = legacy_states(raw)
print("n_states", model.n_states, "legacy", Xl.shape)
print("max |X0 - Xlegacy(0)|:", np.max(np.abs(X0 - Xl[:, 0])))
t0 = time.time(); f0 = model.rhs(0.0, X0); print("rhs time", time.time()-t0)
print("|rhs(X0)| =", np.linalg.norm(f0), "max", np.max(np.abs(f0)))
# compare rhs at a later legacy state
for it in [15, 20, 50]:
    f = model.rhs(Tl[it], Xl[:, it])
    print(f"t={Tl[it]}: |rhs| = {np.linalg.norm(f):.3e}")
# sparsity check vs numerical jacobian
S = model.jac_sparsity()
n = model.n_states
J = np.empty((n, n))
for j in range(n):
    h = 1e-6*abs(X0[j]) + 1e-8
    xp = X0.copy(); xp[j] += h
    J[:, j] = (model.rhs(0.0, xp) - f0)/h
nz = np.abs(J) > 1e-12*np.max(np.abs(J))
print("sparsity ok:", not np.any(nz & ~S), " nnz num", nz.sum(), " nnz struct", S.sum())
res = simulate(case, rtol=1e-4, atol=1e-6, max_step=1.0)
X = res.X
err = np.max(np.abs(X - Xl), axis=1)
print("max abs diff per state group: alpha", err[0::4][:3], "...")
# per variable type
pb = model.pipes[0]
slots = pb.free % 4
for s, name in enumerate(["alphaG", "PT", "qL", "qG"]):
    m = slots == s
    print(name, "max diff", np.max(np.abs(X[m] - Xl[m])), "legacy range", Xl[m].min(), Xl[m].max())
np.save("scratch/X_10_1_1_galerkin.npy", X)
