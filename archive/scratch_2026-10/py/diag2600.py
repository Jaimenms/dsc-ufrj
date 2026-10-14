import numpy as np, warnings, collections, time, sys, json
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.initial_guess import initial_guess
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
case = severe_slug_case(method="fv_upwind", tf=3000, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=30, n_riser=10, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5,
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG})
model = build_model(case); X0 = initial_guess(model); lo, hi = state_bounds(model); pb = model.pipes[0]; h = pb.mesh["h"]
def where(i): return f"node{pb.free[i]//4} slot{pb.free[i]%4}"
bdf = ImplicitBDF(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=(lo,hi), min_step=1e-10, max_order=2)
t0 = time.time()
T, X, info = bdf.integrate(np.arange(0, 2551, 10.0), X0, first_step=1e-3); print("to 2550:", info, time.time()-t0, flush=True)
np.savez("scratch/diag2600_state.npz", T=T, X=X)
res = Results(model, T, X); p = res.pipe(0); aG = p["out"]["alphaG"]; P = p["out"]["P_bar"]; qLo = p["out"]["qL"]; qGo = p["out"]["qG"]; x = p["x"]
rhoL = fluid.rhoL(P[1:-1]*1e5); mass = np.sum((1 - aG[1:-1]) * rhoL * A * h[:, None], axis=0)
net = np.concatenate([[0.0], np.cumsum(0.5*((qLo[0,1:]-qLo[-1,1:]) + (qLo[0,:-1]-qLo[-1,:-1])) * np.diff(T))])
print(f"massa líquida: inicial {mass[0]:.1f} final {mass[-1]:.1f}; integral {net[-1]:.1f}; erro {mass[-1]-mass[0]-net[-1]:.1f} kg", flush=True)
ib = int(np.argmin(np.abs(x - 320.0)))
for it in range(0, T.size, 25):
    print(f"t={T[it]:6.0f} P_in={P[1,it]:6.3f} P_base={P[ib,it]:6.3f} aG_line={aG[5,it]:.3f}/{aG[15,it]:.3f}/{aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[ib+5,it]:.3f}/{aG[ib+10,it]:.3f} qL_out={qLo[-1,it]:6.3f} qG_out={qGo[-1,it]:.5f}", flush=True)
np.set_printoptions(linewidth=230, precision=3, suppress=True)
print("aG(2550):", aG[:, -1]); print("qL(2550):", qLo[:, -1]); print("qG(2550):", qGo[:, -1]); print("P(2550):", P[:, -1])
# now instrumented continuation
stats = collections.Counter(); dom = collections.Counter()
class Dbg(ImplicitBDF):
    def _newton(self, t, h_, pred, c_hist, gcoef):
        self.ncall = getattr(self, "ncall", 0) + 1
        if self.ncall > 2500: raise RuntimeError("budget")
        xx, ok, nit = super()._newton(t, h_, pred, c_hist, gcoef)
        stats[("ok" if ok else self._fail_reason, nit, "clip" if self._clipped else "")] += 1
        scale = self.atol + self.rtol*np.abs(pred); d = np.abs(xx - pred)/scale
        if not ok: dom[where(int(np.argmax(d)))] += 1
        self._lp = (t, h_, xx); return xx, ok, nit
bdf2 = Dbg(model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=(lo,hi), min_step=1e-10, max_order=2, verbose=True)
try:
    T2, X2, info2 = bdf2.integrate(np.arange(2550, 2801, 10.0), X[:, -1], first_step=1e-3); print("to 2800:", info2, flush=True)
except Exception as e:
    print("stopped:", e, "at t=", bdf2._lp[0], "steps", bdf2.n_steps, flush=True)
print("outcomes:", sorted(stats.items(), key=lambda kv: -kv[1])[:8], flush=True)
print("dominant:", dom.most_common(6), flush=True)
t, h_, xx = bdf2._lp; E, Ys, _, _ = model.assemble(t, xx); Y = Ys[0]
print("t", t, "h", h_); print("aG:", Y[:,0]); print("qL:", Y[:,2]); print("qG:", Y[:,3]); print("P:", Y[:,1]*10)
