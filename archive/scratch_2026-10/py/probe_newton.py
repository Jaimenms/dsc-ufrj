"""Traça a convergência do Newton (modo conservativo vs primitivo) até t=tend."""
import numpy as np, warnings, collections, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
cons = bool(int(sys.argv[1])); tend = float(sys.argv[2]); extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
case = severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked",
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "conservative": cons, **extra})
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model); pb = model.pipes[0]
kw = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if cons else {}
pat = collections.Counter(); domdx = collections.Counter(); hs = []
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        if not ok and self._fail_reason == "maxit":
            e = np.array(self._err_hist)
            tail = e[-4:]
            ratios = tail[1:] / tail[:-1]
            kind = "oscila" if np.any(ratios > 1.0) else ("lento" if ratios.min() > 0.5 else "converge")
            pat[(kind, f"{np.log10(e[-1]):.0f}")] += 1
            scale = self.atol + self.rtol*np.abs(pred); i = int(np.argmax(np.abs(self._last_dx)/scale))
            domdx[f"node{pb.free[i]//4} slot{pb.free[i]%4}"] += 1
            if len(hs) < 6: hs.append((round(t, 4), h, np.round(e, 4).tolist()))
        return x, ok, nit
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=3e-3, atol=3e-5, bounds=(lo,hi), min_step=1e-10, **kw)
class Stop(Exception): pass
orig_f = bdf._f
def f_lim(t, x):
    if t > tend: raise Stop()
    return orig_f(t, x)
bdf._f = f_lim
try:
    bdf.integrate(np.arange(0, 3001, 2.0), X0, first_step=1e-3)
except Stop: pass
print(f"cons={cons} extra={extra}: {bdf.n_steps} passos, {bdf.n_rejected} rej")
print("  padrões maxit (tipo, log10 err final):", sorted(pat.items(), key=lambda kv: -kv[1])[:8])
print("  componente dominante do último dx:", domdx.most_common(6))
for t, h, e in hs: print(f"  t={t} h={h:.2e} errs={e}")
