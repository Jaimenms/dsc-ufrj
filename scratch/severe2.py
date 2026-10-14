import numpy as np, time, warnings, sys, traceback
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, superficial_velocities
from mfsim.simulation import build_model, state_bounds, simulate, steady_state
from mfsim.initial_guess import initial_guess
from mfsim.integrators.jacobian import colored_jacobian, color_groups
regimes = [4, 4, 3, 3]
for method in ["fv_upwind", "fd_central"]:
    print("=========", method)
    case = severe_slug_case(regime=regimes, single_pipe=False, method=method, tf=3000.0, dt_out=5.0, qL=1.0, qG=0.05,
                            n_h1=5, n_down=20, n_riser=10, n_h2=3, extra={"k_switch": 50.0})
    print("vSL, vSG:", superficial_velocities(case))
    try:
        t0 = time.time()
        model, X, info = steady_state(case, X0="auto", verbose=False)
        print("SS:", info["converged"], f"{info['residual']:.2e}", info["iterations"], f"{time.time()-t0:.1f}s")
        E, Ys, _, _ = model.assemble(0.0, X)
        for k, Y in enumerate(Ys):
            print(f"  pipe{k+1}: aG in/out {Y[1,0]:.3f}/{Y[-2,0]:.3f}  P in/out {Y[1,1]*10:.2f}/{Y[-2,1]*10:.2f} bar  qL in {Y[0,2]:.3f} qG in {Y[0,3]:.4f}")
        # eigenvalues
        f0 = model.rhs(0.0, X); n = model.n_states
        J = np.empty((n, n))
        for j in range(n):
            h = 1e-7*abs(X[j]) + 1e-9; xp = X.copy(); xp[j] += h; J[:, j] = (model.rhs(0.0, xp) - f0)/h
        ev = np.linalg.eigvals(J)
        pos = ev[ev.real > 1e-9]
        print("  n eig with Re>0:", pos.size, " max Re:", ev.real.max(), " unstable:", pos[np.argsort(-pos.real)][:5])
        t0 = time.time()
        res = simulate(case, X0=X*(1+1e-3*np.random.default_rng(0).standard_normal(n)), integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=True)
        print("  dyn cpu", time.time()-t0)
        np.savez(f"scratch/severe2_{method}.npz", T=res.T, X=res.X)
        p = res.pipe(2); P = p["out"]["P_bar"]; aG = p["out"]["alphaG"]
        q = res.pipe(3)["out"]["qL"]
        for it in range(0, res.T.size, max(1, res.T.size//15)):
            print(f"   t={res.T[it]:6.0f} P_riser_base={P[1,it]:7.3f} bar aG_base={aG[1,it]:.3f} aG_top={aG[-2,it]:.3f} qL_out={q[-1,it]:6.3f}")
    except Exception:
        traceback.print_exc()
