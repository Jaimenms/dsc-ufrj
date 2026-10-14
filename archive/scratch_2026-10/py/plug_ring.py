"""Excitação do modo acústico do tampão: três colunas de 20 m (D = 0.05 m, 20 células de 1 m)
entre duas pressões fixas, partindo do repouso com um degrau de pressão entre as extremidades.
  B  : líquido monofásico (modelo 3);
  A' : modelo a dois fluidos com a coluna cheia de líquido (alphaG = 0 em todas as células);
  A  : modelo a dois fluidos com bolsões de gás nas extremidades (tampão entre dois bolsões).
Saída densa (1 ms) da vazão de líquido no meio da coluna: frequência dominante e taxa de
decaimento do envelope. uso: uv run python -u scratch/plug_ring.py '{json de opções}' tag"""
import numpy as np, warnings, sys, json, time
warnings.filterwarnings("ignore")
from mfsim.fluid import Fluid
from mfsim.network.data import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN, Case, Pipe, SimulatorSettings, Vertex
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.jaxrhs import build_jax_rhs
opts = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}; tag = sys.argv[2] if len(sys.argv) > 2 else "x"
gammaL = opts.get("gammaL", 100.0)
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=gammaL, gammaG=1.05, P0=1e5, sigma=0.05)
PR = 1e6; D = 0.05; A = np.pi*D**2/4; L = opts.get("L", 20.0); N = opts.get("N", 20); h = L/N
P0 = 5e5; dP = opts.get("dP", 0.2e5)           # degrau inicial de pressão entre as extremidades (Pa)
tf = opts.get("tf", 3.0); dt = opts.get("dt", 1e-3)
which = opts.get("case", "B")
x_nodes = np.linspace(0, L, N+1)
settings = SimulatorSettings(t0=0.0, tf=tf, dt_out=dt, PR=PR, method="fv_upwind", max_step=np.inf, atol=1e-6, rtol=opts.get("rtol", 3e-3),
                             quadrature_order=2, integrator="implicit")
excite = opts.get("excite", "flow")     # "flow": vazão senoidal (modo c/2L, extremidades a pressão fixa); "step": degrau de pressão
closed = bool(opts.get("closed", 0))    # extremidades fechadas (vazões nulas) em vez de pressão fixa
q0 = opts.get("q0", 0.2)                # amplitude da vazão inicial (kg/s)
ic = np.zeros((N+1, 4))
if excite == "step":
    ic[:, 1] = (P0 + dP*(x_nodes < L/2))/PR if closed else (P0 + dP*(1 - x_nodes/L))/PR
else:
    ic[:, 1] = P0/PR; ic[:, 2] = q0*np.sin(np.pi*x_nodes/L)
if closed:
    vertices = [Vertex(index=1, kind="B", spec={"qL": 0.0, "qG": 0.0}), Vertex(index=2, kind="B", spec={"qL": 0.0, "qG": 0.0})]
else:
    vertices = [Vertex(index=1, kind="B", spec={"PT": (P0 + (dP if excite == "step" else 0.0))/PR}), Vertex(index=2, kind="B", spec={"PT": P0/PR})]
if which == "B":
    model_id = 3; ic[:, 0] = 0.0
    if closed:
        bc_in = (BC_ABSENT, BC_ABSENT, BC_DIRICHLET, BC_ABSENT); bc_out = (BC_ABSENT, BC_ABSENT, BC_DIRICHLET, BC_ABSENT)
    else:
        bc_in = (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT); bc_out = (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
    extra = {"q_switch": 1e-3}
else:
    model_id = 1
    if which == "Ap":
        ic[:, 0] = 0.0
    else:   # bolsões de gás de Lp m nas extremidades com alphaG = aG_pocket (tampão entre dois bolsões)
        Lp = opts.get("Lp", 4.0); ic[:, 0] = np.where((x_nodes < Lp) | (x_nodes > L - Lp), opts.get("aG_pocket", 0.7), 0.0)
        if excite != "step": ic[:, 2] = q0*np.sin(np.pi*np.clip((x_nodes - Lp)/(L - 2*Lp), 0, 1))
    if closed:
        bc_in = (BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET); bc_out = (BC_ABSENT, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET)
    else:
        bc_in = (BC_NEUMANN, BC_DIRICHLET, BC_ABSENT, BC_ABSENT); bc_out = (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
    extra = {k: v for k, v in {"regime_vertical": 0, "q_switch_L": 1e-3, "q_switch_G": 1e-5, "switch_frac": 0.5, "conservative": opts.get("conservative", 1),
                               "relax_time": opts.get("relax"), "flux_limit": opts.get("flux_limit"), "low_void_regime": opts.get("low_void"),
                               "interfacial_pressure": opts.get("delta"), "art_visc": opts.get("art_visc"), "alpha_min": opts.get("alpha_min"),
                               "alpha_band": opts.get("alpha_band"), "limit_alpha": opts.get("limit_alpha"), "alpha_face": opts.get("alpha_face"),
                               "flux_band": opts.get("flux_band"), "relax_target": opts.get("relax_target")}.items() if v is not None}
pipe = Pipe(index=1, v_in=1, v_out=2, x=[0.0, 0.0], y=[0.0, L], z=[0.0, 0.0], n_elements=[N], D=D, roughness=1e-5, model=model_id,
            regime=opts.get("regime", 4), bc_in=bc_in, bc_out=bc_out, ic=ic, extra=extra)
case = Case(name="plug", fluid=fluid, settings=settings, vertices=vertices, pipes=[pipe])
model = build_model(case); X0 = model.initial_state()
rhs_j, jac_j = build_jax_rhs(model)
lo, hi = state_bounds(model)
atol = 1e-3*np.ones(X0.size); slots = model.pipes[0].free % 4
atol[slots == 1] = 1e-3; atol[(slots == 2) | (slots == 3)] = 1e-3
kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if model.conservative else {}
bdf = ImplicitBDF(rhs_j, S=model.jac_sparsity(), rtol=opts.get("rtol", 3e-3), atol=atol, bounds=(lo, hi), jac_fun=jac_j, min_step=1e-9,
                  verbose=False, max_order=opts.get("order", 2), max_newton=20, newton_tol=1e-3, **kw_t)
bdf.slow_exit = False
t0 = time.time(); T, X, info = bdf.integrate(np.arange(0, tf + 0.5*dt, dt), X0, first_step=1e-4); wall = time.time() - t0
X = np.asarray(X); X = X.T if X.shape[0] != T.size else X
pb = model.pipes[0]; sch = pb.scheme
slot_q = 2
Y = np.array([pb.assemble(X[i, pb.offset:pb.offset+pb.n_states], np.array([[tb(T[i]) for tb in row] for row in model.tables])[model.vpos[pb.pipe.v_in]],
                          np.array([[tb(T[i]) for tb in row] for row in model.tables])[model.vpos[pb.pipe.v_out]]) for i in range(T.size)])
q_mid = Y[:, N//2, slot_q]; P_mid = Y[:, N//2 + 1, 1]*PR
c = np.sqrt(gammaL*P0/fluid.rhoL0)
# frequência dominante e decaimento do envelope (segunda metade: regime de oscilação livre)
sel = T > opts.get("t_skip", 0.3)
q = q_mid[sel] - np.mean(q_mid[sel]); tt = T[sel]
freqs = np.fft.rfftfreq(q.size, dt); spec = np.abs(np.fft.rfft(q*np.hanning(q.size)))
f_dom = freqs[np.argmax(spec[1:]) + 1]
# envelope: máximos locais de |q|
from scipy.signal import find_peaks
pk, _ = find_peaks(np.abs(q), distance=max(1, int(0.5/f_dom/dt)))
env_t, env = tt[pk], np.abs(q[pk])
rate = np.nan
if env.size > 4 and np.all(env > 0):
    rate = -np.polyfit(env_t, np.log(env), 1)[0]      # 1/s (positivo = decai)
print(f"[{tag}] caso {which} {excite}{' fechado' if closed else ''} N={N} gammaL={gammaL}: passos {info['n_steps']} rej {info['n_rejected']} wall {wall:.1f}s | c={c:.0f} m/s, c/2L={c/(2*L):.2f} Hz | "
      f"f_dom={f_dom:.2f} Hz, decaimento {rate:+.2f} 1/s (envelope {env[:1].round(4)} -> {env[-1:].round(4)} kg/s), |q| máx {np.abs(q_mid).max():.3f}, q final {q_mid[-1]:.4f}", flush=True)
np.savez(f"scratch/plug_ring_{tag}.npz", T=T, q_mid=q_mid, P_mid=P_mid, Y=Y, opts=json.dumps(opts), info=json.dumps({k: v for k, v in info.items() if k != "accum"}))
