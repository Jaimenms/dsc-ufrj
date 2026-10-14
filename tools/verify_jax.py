"""Verifica rhs JAX vs numpy e a Jacobiana em todos os casos dos notebooks."""
import numpy as np, warnings, time, sys, copy
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model
from mfsim.compiled import build_jax_rhs
from mfsim.fluid import Fluid
from mfsim.integrators.jacobian import colored_jacobian, color_groups

def severe(cons):
    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
    return severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                            n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked", elbow_radius=5.0, n_elbow=4,
                            extra={"regime_vertical": 2, "conservative": cons, "vertical_blend": 20.0, "art_visc": 0.05, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5})

cases = []
for cid in ("10.1.1", "10.1.2", "10.1.3", "10.1.4", "10.2.1A", "10.2.1B", "10.2.2", "10.3.1", "10.3.2a", "10.3.2b", "10.3.3"):
    case, raw = load_chapter10_case(cid); cases.append((cid, case))
c = copy.deepcopy(cases[0][1]); c.settings.method = "fd_central"; cases.append(("10.1.1 fd_central", c))
c = copy.deepcopy(cases[0][1]); c.settings.method = "petrov_galerkin"; cases.append(("10.1.1 petrov_galerkin", c))
c = copy.deepcopy(cases[0][1]); c.settings.method = "fd_upwind"; cases.append(("10.1.1 fd_upwind", c))
c = copy.deepcopy(cases[0][1]); c.settings.method = "fv_upwind"; c.pipes[0].ic = None; cases.append(("10.1.1 fv_upwind primitivo", c))
cases.append(("severe slug conservativo", severe(True))); cases.append(("severe slug primitivo", severe(False)))
sel = sys.argv[1:] 
rng = np.random.default_rng(1)
for name, case in cases:
    if sel and not any(s in name for s in sel): continue
    try:
        model = build_model(case)
        if case.pipes[0].ic is None and "fv_upwind" in name:
            from mfsim.initial_guess import initial_guess
            X0 = initial_guess(model)
        else:
            X0 = model.initial_state()
        t0 = time.time(); rhs_j, jac_j = build_jax_rhs(model); t_build = time.time() - t0
        tt = case.settings.t0 + 1.0
        worst = 0.0; scale = 0.0
        states = [X0]
        for k in range(3):
            x = X0 * (1 + 0.02 * rng.standard_normal(X0.size)); states.append(x)
        for x in states:
            f_np = model.rhs(tt, x); f_jx = rhs_j(tt, x)
            worst = max(worst, np.max(np.abs(f_np - f_jx))); scale = max(scale, np.max(np.abs(f_np)))
        J = jac_j(tt, X0); finite = bool(np.all(np.isfinite(J)))
        t0 = time.time(); [rhs_j(tt, X0) for _ in range(50)]; t_r = (time.time() - t0) / 50
        t0 = time.time(); [model.rhs(tt, X0) for _ in range(20)]; t_n = (time.time() - t0) / 20
        t0 = time.time(); [jac_j(tt, X0) for _ in range(10)]; t_j = (time.time() - t0) / 10
        S = model.jac_sparsity(); g = color_groups(S)
        print(f"{name:28s} n={X0.size:4d} | max|Δf| {worst:9.2e} (|f|max {scale:9.2e}, rel {worst/max(scale,1e-300):8.1e}) | J finita {finite} | rhs {1e3*t_n:6.2f} -> {1e3*t_r:5.2f} ms | jac {1e3*t_n*len(g):6.1f} -> {1e3*t_j:5.2f} ms | build {t_build:4.1f} s", flush=True)
    except Exception as e:
        import traceback
        print(f"{name:28s} ERRO: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if sel: traceback.print_exc()
