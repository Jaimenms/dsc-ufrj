"""Inversão do sentido do escoamento num duto entre dois vértices de pressão (esquema
escalonado conservativo): a fração de gás que entra pela saída invertida é a do separador."""
import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import Case, Vertex, Pipe, SimulatorSettings, BC_NEUMANN, BC_ABSENT, BC_DIRICHLET
from mfsim.fluid import Fluid
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.jaxrhs import build_jax_rhs
from mfsim.results import Results
from mfsim.analysis import mass_balance
alpha_out_bc = sys.argv[1] if len(sys.argv) > 1 else "dirichlet"
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D = 0.05; A = np.pi * D**2 / 4; PR = 1e6; N = 20; L = 100.0; tf = 200.0
settings = SimulatorSettings(t0=0.0, tf=tf, dt_out=1.0, PR=PR, method="fv_upwind", max_step=np.inf, atol=1e-6, rtol=1e-4, quadrature_order=2, integrator="implicit")
# pressão de entrada: 5.3 bar até 60 s, cai a 4.7 bar em 100 s (inversão), permanece
P_in = (np.array([0.0, 60.0, 100.0, tf]), np.array([5.3e5, 5.3e5, 4.7e5, 4.7e5]) / PR)
v1 = Vertex(index=1, kind="B", spec={"PT": P_in, "alphaG": 0.5})
v2 = Vertex(index=2, kind="B", spec={"PT": 5.0e5 / PR, "alphaG": 0.0})      # separador: devolve líquido
bc_out_alpha = BC_DIRICHLET if alpha_out_bc == "dirichlet" else BC_ABSENT
pipe = Pipe(index=1, v_in=1, v_out=2, x=[0, 0], y=[0, L], z=[0, 0], n_elements=[N], D=D, roughness=1e-5, model=1, regime=4,
            bc_in=(BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT), bc_out=(bc_out_alpha, BC_DIRICHLET, BC_ABSENT, BC_ABSENT),
            extra={"conservative": True, "art_visc": 0.05, "q_switch_L": 0.01, "q_switch_G": 1e-4})
s_nodes = np.linspace(0, L, N + 1)
pipe.ic = np.stack([np.full(N + 1, 0.5), np.linspace(5.3e5, 5.0e5, N + 1) / PR, np.zeros(N + 1), np.zeros(N + 1)], axis=1)
case = Case(name="inversao", fluid=fluid, settings=settings, vertices=[v1, v2], pipes=[pipe])
case.settings.extra["state_ref"] = dict(PT=0.6, qL=2.0, qG=0.01)
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model)
rhs_j, jac_j = build_jax_rhs(model)
bdf = ImplicitBDF(rhs_j, S=model.jac_sparsity(), rtol=3e-3, atol=0.01 * model.state_scales(X0), bounds=(lo, hi), min_step=1e-9, jac_fun=jac_j, max_newton=20,
                  transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian)
bdf.accum = model.boundary_net_fluxes
t0 = time.time(); T, X, info = bdf.integrate(np.arange(0, tf + 1e-9, 1.0), X0, breakpoints=model.spec_breakpoints()); cpu = time.time() - t0
res = Results(model, T, X); p = res.pipe(0); qL = p["out"]["qL"]; aG = p["out"]["alphaG"]
mL, nL, eL = mass_balance(res, 0, "L"); mG, nG, eG = mass_balance(res, 0, "G")
print(f"saída alphaG {alpha_out_bc:9s}: {info['n_steps']} passos, {cpu:.1f} s | erro de massa L {mL[-1]-mL[0]-info['accum'][0,-1]:+.2e} kg, G {mG[-1]-mG[0]-info['accum'][1,-1]:+.2e} kg")
for t in (50, 80, 100, 130, 200):
    i = int(np.argmin(np.abs(T - t)))
    print(f"   t={t:3d} s: qL entrada {qL[0,i]:+.3f} saída {qL[-1,i]:+.3f} kg/s | alphaG células 1, N/2, N: {aG[1,i]:.3f} {aG[N//2,i]:.3f} {aG[N,i]:.3f} | fantasma de saída {aG[N+1,i]:.3f}")
