"""Procura descontinuidades de f ao longo da direção de Newton num passo que estagna."""
import numpy as np, warnings, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
cons = bool(int(sys.argv[1])); tend = float(sys.argv[2]); nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 3
case = severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked",
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "conservative": cons})
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model); pb = model.pipes[0]
kw = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if cons else {}
def name(i): return f"node{pb.free[i]//4} slot{pb.free[i]%4}"
found = [0]
class Dbg(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        if not ok and self._fail_reason == "maxit" and found[0] < nmax:
            e = np.array(self._err_hist); tail = e[-4:]
            if np.any(tail[1:] / tail[:-1] > 1.0) and e[-1] > 1e-2:
                found[0] += 1
                dx = self._last_dx; gamma = gcoef * h
                ss = np.linspace(-1.0, 1.0, 81)
                F = np.array([self._f(t, self._project(x - s * dx)) for s in ss])
                R = np.array([self._T(self._project(x - s * dx)) - c_hist - gamma * self._f(t, self._project(x - s * dx)) for s in ss])
                scale = self.atol + self.rtol * np.abs(pred)
                print(f"\n== maxit oscilante em t={t:.5f} h={h:.2e} errs(últimos)={np.round(tail,3).tolist()}")
                # saltos: maior diferença entre amostras consecutivas vs variação total
                dF = np.abs(np.diff(F, axis=0)); tot = F.max(0) - F.min(0) + 1e-300
                jump = dF.max(0) / tot
                big = np.argsort(-jump)[:8]
                for j in big:
                    if jump[j] > 0.3 and tot[j] > 1e-8 * (np.abs(F[:, j]).max() + 1e-300):
                        k = int(np.argmax(dF[:, j]))
                        print(f"  f[{name(j)}]: salto {jump[j]:.2f} da variação total em s={ss[k]:.3f}..{ss[k+1]:.3f}; f={F[k, j]:.4g} -> {F[k+1, j]:.4g}; |f|max {np.abs(F[:, j]).max():.3g}")
                # componentes dominantes do dx e seus valores em x
                i = np.argsort(-np.abs(dx) / scale)[:5]
                print("  dx dominantes:", [(name(j), f"x={x[j]:.5g}", f"dx={dx[j]:.3g}") for j in i])
                # resíduo escalado total ao longo de s
                rn = np.sqrt(np.mean((R / scale) ** 2, axis=1))
                print("  |R| escalado em s=-1,-0.5,0,0.25,0.5,0.75,1:", [f"{rn[np.argmin(np.abs(ss - v))]:.3g}" for v in (-1, -0.5, 0, 0.25, 0.5, 0.75, 1)])
        return x, ok, nit
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=3e-3, atol=3e-5, bounds=(lo,hi), min_step=1e-10, **kw)
class Stop(Exception): pass
orig_f = bdf._f
def f_lim(t, x):
    if t > tend or found[0] >= nmax: raise Stop()
    return orig_f(t, x)
bdf._f = f_lim
try:
    bdf.integrate(np.arange(0, 3001, 2.0), X0, first_step=1e-3)
except Stop: pass
