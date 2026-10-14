import numpy as np, warnings, collections, time, sys
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case
from mfsim.simulation import build_model, state_bounds
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.fluid import Fluid
fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
D=0.05; A=np.pi*D**2/4; qL=0.3*900*A; qG=0.1*fluid.rhoG(5e5)*A
cons = bool(int(sys.argv[1])); tend = float(sys.argv[2]); long_horizon = bool(int(sys.argv[3]))
case = severe_slug_case(method="fv_upwind", tf=100, dt_out=2.0, qL=qL, qG=qG, D=D, P_out_bar=4.0, L_h1=20.0, L_down=300.0, slope_down_deg=-3.0, H_riser=20.0, L_h2=5.0,
                        n_h1=2, n_down=15, n_riser=5, n_h2=2, regime=4, single_pipe=True, fluid=fluid, roughness=1e-5, start="blocked",
                        elbow_radius=float(__import__("os").environ.get("ELBOW_R", 0)), n_elbow=int(__import__("os").environ.get("N_ELBOW", 0)),
                        extra={"regime_vertical": 2, "q_switch_L": 0.02*qL, "q_switch_G": 0.02*qG, "switch_frac": 0.5, "conservative": cons, **(__import__("json").loads(sys.argv[7]) if len(sys.argv) > 7 else {})})
model = build_model(case); X0 = model.initial_state(); lo, hi = state_bounds(model)
kw = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if cons else {}
import collections
stats = collections.Counter(); dom = collections.Counter(); byt = collections.Counter()
nclip = collections.Counter()
errslot = {"acc": np.zeros(4), "rej": np.zeros(4)}; nerr = {"acc": 0, "rej": 0}
class Dbg(ImplicitBDF):
    def _project(self, x):
        xp = super()._project(x)
        for i in np.nonzero(xp != x)[0]:
            pb = model.pipes[0]; nclip[f"node{pb.free[i]//4} slot{pb.free[i]%4} {'lo' if x[i] < lo[i] else 'hi'}"] += 1
        return xp
    def _newton(self, t, h, pred, c_hist, gcoef):
        v = self._last_err_vec
        if v is not None:
            key = "rej" if np.sqrt(np.mean(v**2)) > 1 else "acc"
            slots = model.pipes[0].free % 4
            for s_ in range(4): errslot[key][s_] += np.mean(v[slots == s_]**2)
            nerr[key] += 1; self._last_err_vec = None
        x, ok, nit = super()._newton(t, h, pred, c_hist, gcoef)
        stats[("ok" if ok else self._fail_reason, "clip" if self._clipped else "")] += 1
        if not ok:
            scale = self.atol + self.rtol*np.abs(pred); i = int(np.argmax(np.abs(x-pred)/scale)); pb = model.pipes[0]
            dom[f"node{pb.free[i]//4} slot{pb.free[i]%4}"] += 1; byt[int(t)] += 1
        return x, ok, nit
import os
rtol = 3e-3*np.ones(X0.size)
if os.environ.get("RTOL_P"):
    rtol[model.pipes[0].free % 4 == 1] = float(os.environ["RTOL_P"])
bdf = Dbg(model.rhs, S=model.jac_sparsity(), rtol=rtol, atol=3e-5, bounds=(lo,hi), min_step=1e-10, **kw)
bdf.slow_exit = False   # configuração do severe slug (sem recálculo da Jacobiana dentro do Newton)
bdf.cooldown_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
if len(sys.argv) > 5: bdf.max_order = int(sys.argv[5])
if len(sys.argv) > 6: bdf.jac_reuse = int(sys.argv[6])
if len(sys.argv) > 8: bdf.predict_space = sys.argv[8]
if len(sys.argv) > 9: bdf.jt_refresh = sys.argv[9]
if len(sys.argv) > 10: bdf.error_space = sys.argv[10]
t_eval = np.arange(0, 3001, 2.0) if long_horizon else [0, tend]
class Stop(Exception): pass
orig_f = bdf._f
def f_lim(t, x):
    if t > tend: raise Stop()
    return orig_f(t, x)
bdf._f = f_lim
t0 = time.time()
try:
    T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3)
except Stop:
    pass
print(f"cons={cons} long_horizon={long_horizon}: até t={tend}: {bdf.n_steps} passos, {bdf.n_rejected} rejeitados, jac {bdf.n_jac}, {time.time()-t0:.1f}s")
print("  erro local por slot (RMS^2 médio) aceitos:", np.round(errslot["acc"]/max(nerr["acc"],1), 3), "rejeitados:", np.round(errslot["rej"]/max(nerr["rej"],1), 3)); print("  clips:", nclip.most_common(4)); print("  newton:", dict(stats)); print("  dominantes:", dom.most_common(5)); print("  falhas por segundo simulado:", sorted(byt.items()))
