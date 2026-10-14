"""Reinicia a trajetória M1 num instante com tampão de líquido (toque acústico) e integra
30 s com variantes de configuração: conta passos e mede a amplitude da oscilação."""
import numpy as np, warnings, time, json, sys, os
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
t0 = float(sys.argv[1]); dur = float(sys.argv[2]); tag = sys.argv[3]; over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
use_jax = bool(over.pop("jax", 0)); tol_range = over.pop("tol_range", None); state_ref = over.pop("state_ref", None); fl_over = over.pop("fluid", {}); bdf_over = over.pop("bdf", {}); rtolP = over.pop("rtol_P", None); patch = over.pop("patch", {}); atolP = over.pop("atol_P", None); atolq = over.pop("atol_q", None)
fluid = Fluid(**{**dict(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05), **fl_over})
D=0.05; A=np.pi*D**2/4; P_out=4.0; qL=0.3*900*A; qG=0.1*fluid.rhoG((P_out+1)*1e5)*A
if patch:
    import mfsim.closures.bubble as bub
    for k, v in patch.items(): setattr(bub, k, v)
case = severe_slug_case(method="fv_upwind", tf=t0 + dur, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked", elbow_radius=5.0, n_elbow=4,
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "conservative": True, "vertical_blend": 20.0, "art_visc": 0.01, **over})
if state_ref: case.settings.extra["state_ref"] = state_ref
model = build_model(case); lo, hi = state_bounds(model)
d = np.load("notebooks/data/severe_slug_blocked_conservative_arc_av.npz"); T, X = d["T"], d["X"]
X0 = X[:, int(np.argmin(np.abs(T - t0)))].copy()
rtol = 3e-3 * np.ones(X0.size); atol = 3e-5 * np.ones(X0.size); slots = model.pipes[0].free % 4
if rtolP: rtol[slots == 1] = rtolP
if atolP: atol[slots == 1] = atolP
if atolq: atol[(slots == 2) | (slots == 3)] = atolq
if tol_range: atol = tol_range * model.state_scales(X0)
fun_, jac_ = model.rhs, None
if use_jax:
    from mfsim.jaxrhs import build_jax_rhs
    fun_, jac_ = build_jax_rhs(model)
bdf = ImplicitBDF(fun_, S=model.jac_sparsity(), rtol=rtol, atol=atol, bounds=(lo, hi), min_step=1e-9, verbose=False, jac_fun=jac_,
                  transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian, **bdf_over)
bdf.slow_exit = False   # configuração do severe slug (sem recálculo da Jacobiana dentro do Newton)
t_eval = np.arange(t0, t0 + dur + 1e-9, 0.02)
tc = time.time(); Tr, Xr, info = bdf.integrate(t_eval, X0, first_step=1e-3); cpu = time.time() - tc
pb = model.pipes[0]; sch = pb.scheme
E, Ys, _, _ = model.assemble(Tr[0], Xr[:, 0])
# amplitude da oscilação rápida de qL (desvio da média móvel de 1 s) nas faces do riser e P na entrada
Yall = np.array([model.assemble(Tr[i], Xr[:, i])[1][0] for i in range(0, Tr.size, 1)])
qLf = Yall[:, :, 2]; k = 50
mov = np.array([np.convolve(qLf[:, j], np.ones(k)/k, mode="same") for j in range(qLf.shape[1])]).T
amp = np.sqrt(np.mean((qLf - mov)**2, axis=0))
Pin = Yall[:, 1, 1] * model.PR / 1e5
Pref = None
try:
    from mfsim.results import Results
    m_ref = build_model(severe_slug_case(method="fv_upwind", tf=T[-1], dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05),
                        roughness=1e-5, start="blocked", elbow_radius=5.0, n_elbow=4, extra={"regime_vertical": 2, "conservative": True, "vertical_blend": 20.0, "art_visc": 0.01}))
    j = int(np.argmin(np.abs(T - (t0 + dur)))); Pref = m_ref.assemble(T[j], X[:, j])[1][0][1, 1] * m_ref.PR / 1e5
except Exception as e:
    pass
print(f"{tag:16s}: {info['n_steps']:6d} passos, {info['n_rejected']:5d} rej, {info['n_jac']:5d} jac, {cpu:5.0f} s | osc. rápida qL máx {amp.max():.3f} kg/s | P_in {Pin[0]:.3f} -> {Pin[-1]:.3f} bar (M1: {Pref if Pref is None else round(float(Pref), 3)}) | qL_out médio {Yall[:, -1, 2].mean():.3f}", flush=True)
np.savez(f"scratch/restart_{tag}.npz", T=Tr, X=Xr)
