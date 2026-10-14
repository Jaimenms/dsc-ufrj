import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=1000.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
tf = float(sys.argv[1]); tend = float(sys.argv[2]); use_bp = int(sys.argv[3])
case = severe_slug_case(method="fv_upwind", tf=tf, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=30, n_riser=10, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, extra={"regime_vertical": 2, "q_switch": 0.01, "relax_time": 0.2})
model = build_model(case); X0 = initial_guess(model)
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-10, max_order=2, verbose=True)
t0 = time.time()
kw = dict(breakpoints=model.spec_breakpoints()) if use_bp else {}
print("breakpoints:", model.spec_breakpoints(), flush=True)
T, X, info = bdf.integrate([0, tend], X0, first_step=1e-3, **kw); print("DONE", info, f"{time.time()-t0:.1f}s", flush=True)
