"""Reinício em t0 com estatísticas do Newton/controlador (fase cara)."""
import numpy as np, warnings, time, json, sys, collections
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
t0 = float(sys.argv[1]); dur = float(sys.argv[2]); over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
bdf_over = over.pop("bdf", {}); attrs = over.pop("attrs", {}); use_jax = bool(over.pop("jax", 0))
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; P_out=4.0; qL=0.3*900*A; qG=0.1*fluid.rhoG((P_out+1)*1e5)*A
case = severe_slug_case(method="fv_upwind", tf=t0 + dur, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked", elbow_radius=5.0, n_elbow=4,
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "conservative": True, "vertical_blend": 20.0, "art_visc": 0.01, **over})
model = build_model(case); lo, hi = state_bounds(model); pb = model.pipes[0]; sch = pb.scheme; s0 = case.meta["s_elbow0"]
d = np.load("notebooks/data/severe_slug_blocked_conservative_arc_av.npz"); T, X = d["T"], d["X"]
X0 = X[:, int(np.argmin(np.abs(T - t0)))].copy()
def name(i):
    node, slot = pb.free[i] // 4, pb.free[i] % 4
    pos = (sch.x_cells[node-1] - s0) if 1 <= node <= sch.N else float("nan")
    return f"{'aG P qL qG'.split()[slot]}@{pos:+.1f}m"
stats = collections.Counter(); dom_fail = collections.Counter(); dom_err = collections.Counter(); errslot = np.zeros(4); nerr = 0; nits = []
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        v = self._last_err_vec
        global nerr, errslot
        if v is not None:
            if np.sqrt(np.mean(v**2)) > 1:
                i = int(np.argmax(np.abs(v))); dom_err[name(i)] += 1
                slots = pb.free % 4
                for s_ in range(4): errslot[s_] += np.mean(v[slots == s_]**2)
                nerr += 1
            self._last_err_vec = None
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats[("ok" if ok else self._fail_reason)] += 1; nits.append(nit)
        if not ok and self._last_dx is not None:
            scale = self.atol + self.rtol*np.abs(pred); i = int(np.argmax(np.abs(self._last_dx)/scale)); dom_fail[name(i)] += 1
        return x, ok, nit
fun_, jac_ = model.rhs, None
if use_jax:
    from mfsim.jaxrhs import build_jax_rhs
    fun_, jac_ = build_jax_rhs(model)
bdf = Dbg(fun_, S=model.jac_sparsity(), rtol=3e-3, atol=3e-5, bounds=(lo, hi), min_step=1e-9, verbose=False, jac_fun=jac_,
          transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian, **bdf_over)
bdf.slow_exit = False   # configuração do severe slug (sem recálculo da Jacobiana dentro do Newton)
for k_, v_ in attrs.items(): setattr(bdf, k_, v_)
tc = time.time(); Tr, Xr, info = bdf.integrate([t0, t0 + dur], X0, first_step=1e-3); cpu = time.time() - tc
print(f"t0={t0} dur={dur} over={over} {bdf_over}: {info['n_steps']} passos, {info['n_rejected']} rej, {info['n_jac']} jac, {cpu:.0f} s")
print("  newton:", dict(stats), "| iterações médias:", round(float(np.mean(nits)), 2))
print("  rejeições por erro:", nerr, "| erro por slot (RMS^2 médio) [aG, P, qL, qG]:", np.round(errslot/max(nerr,1), 2))
print("  componentes dominantes do erro:", dom_err.most_common(6))
print("  componentes dominantes nas falhas do Newton:", dom_fail.most_common(6))
