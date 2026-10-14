import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=1000.0, rhoG0=1.2, muL=1e-3, muG=1.8e-5, gammaL=1000.0, gammaG=1.05, P0=1e5, sigma=0.07)
D=0.05; A=np.pi*D**2/4; qL=0.3*1000*A; qG=0.05*fluid.rhoG(2.5e5)*A
case = severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=1.5, L_h1=20.0, L_down=300.0, slope_down_deg=-2.0, H_riser=15.0, L_h2=5.0,
                        n_h1=2, n_down=30, n_riser=10, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, extra={"regime_vertical": 2}, start="shut_in")
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model)
pb = model.pipes[0]
def where(i): return f"node{pb.free[i]//4} slot{pb.free[i]%4}"
class Probe(ImplicitBDF):
    def _newton(self, t, h, pred, c_hist, gcoef):
        x = pred.copy(); gamma = gcoef*h
        if self._LU is None or self._LU_h != gamma: self._factor(gamma)
        scale = self.atol + self.rtol*np.abs(pred); errs = []; xs = [x.copy()]
        for it in range(self.max_newton):
            f = self._f(t, x)
            if not np.all(np.isfinite(f)):
                bad = np.flatnonzero(~np.isfinite(f)); print("NONFINITE f at", [where(i) for i in bad][:8], "t=", t, flush=True)
                E, Ys, _, _ = model.assemble(t, x); Y = Ys[0]; nb = sorted(set(pb.free[i]//4 for i in bad))[:3]
                for nd in nb: print("   Y[node]=", Y[max(nd-1,0):nd+2].round(5).tolist())
                return x, False, it+1
            res = x - c_hist - gamma*f; dx = self._LU.solve(res)
            err = np.sqrt(np.mean((dx/scale)**2)); errs.append(err)
            x = self._project(x - dx); xs.append(x.copy())
            if err < self.newton_tol: return x, True, it+1
        # failed
        if t > 0.3:
            print(f"\nFALHA (jac_age={self._jac_age}): t={t:.4f} h={h:.3e} errs={np.array(errs).round(5)}", flush=True)
            k = int(np.argmax(np.abs(dx/scale))); print("  componente dominante:", where(k), "x=", x[k], "dx/scale=", dx[k]/scale[k])
            # scan f along the last update
            a, b = xs[-2], xs[-1]
            fs = np.array([self._f(t, a + s*(b-a)) for s in np.linspace(0, 1, 41)])
            jumps = np.max(np.abs(np.diff(fs, axis=0)), axis=0); typ = np.max(np.abs(fs), axis=0) + 1e-12
            r = jumps/typ
            for j in np.argsort(-r)[:6]: print(f"  salto f em {where(j)}: rel {r[j]:.3g}, |f| típico {typ[j]:.3g}, x={b[j]:.5g}")
            # where are the clipped states?
            at = np.flatnonzero((b <= lo + 1e-12) | (b >= hi - 1e-12)); print("  estados no limite:", [where(i) for i in at][:12])
            E, Ys, _, _ = model.assemble(t, b); Y = Ys[0]
            np.set_printoptions(linewidth=220, precision=4, suppress=True)
            print("  aG:", Y[:,0]); print("  qL:", Y[:,2]); print("  qG:", Y[:,3])
            np.savez("scratch/newton_fail_state.npz", x=xs[0], pred=pred, c_hist=c_hist, t=t, h=h, gcoef=gcoef)
            sys.exit(0)
        return x, False, self.max_newton
bdf = Probe(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=(lo,hi), min_step=1e-10, max_order=1)
T, X, info = bdf.integrate([0, 100.0], X0, first_step=1e-3); print("no failure up to 100 s", info)
