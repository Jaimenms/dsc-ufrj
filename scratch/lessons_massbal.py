"""Balanço de massa (acumulador BDF-consistente) nos casos da tese com os esquemas legados e com fv_upwind."""
import numpy as np, warnings, copy, sys, time
warnings.filterwarnings("ignore")
from mfsim.cases import load_chapter10_case
from mfsim.simulation import build_model, state_bounds, steady_state
from mfsim.integrators.implicit import ImplicitBDF
from mfsim.results import Results
from mfsim.analysis import mass_balance
from mfsim.jaxrhs import get_jax
cid, method = sys.argv[1], sys.argv[2]; extra = dict(e.split("=") for e in sys.argv[3:])
case, raw = load_chapter10_case(cid); case.settings.method = method
if method == "fv_upwind":
    for p in case.pipes:
        p.ic = None
        if extra.get("conservative"): p.extra["conservative"] = True
        if extra.get("art_visc"): p.extra["art_visc"] = float(extra["art_visc"])
    model, X0, _ = steady_state(case, X0="auto", verbose=False, jax=True)
else:
    model = build_model(case); X0 = model.initial_state()
rtol = float(extra.get("rtol", 1e-4))
rhs, jac = get_jax(model)
kw = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if model.conservative else {}
bdf = ImplicitBDF(rhs, S=model.jac_sparsity(), rtol=rtol, atol=rtol * 1e-2, bounds=state_bounds(model), min_step=1e-9, jac_fun=jac, **kw)
bdf.accum = model.boundary_net_fluxes
t_eval = np.arange(case.settings.t0, case.settings.tf + 1e-9, case.settings.dt_out)
t0 = time.time(); T, X, info = bdf.integrate(t_eval, X0, breakpoints=model.spec_breakpoints()); cpu = time.time() - t0
res = Results(model, T, X, info)
line = f"{cid} {method:10s}{' cons' if model.conservative else ''}: {info['n_steps']:5d} passos, {cpu:5.1f} s |"
for k, pb in enumerate(model.pipes):
    if pb.model.name in ("liquid", "gas"):
        continue
    mL, nL, eL = mass_balance(res, k, "L"); mG, nG, eG = mass_balance(res, k, "G")
    accL, accG = info["accum"][2 * k], info["accum"][2 * k + 1]
    fedL = np.trapezoid(np.abs(res.pipe(k)["out"]["qL"][0]), T) + 1e-300; fedG = np.trapezoid(np.abs(res.pipe(k)["out"]["qG"][0]), T) + 1e-300
    line += f" duto {pb.pipe.index}: erro L {mL[-1]-mL[0]-accL[-1]:+8.3f} kg ({100*(mL[-1]-mL[0]-accL[-1])/fedL:+6.2f} % do alimentado), G {mG[-1]-mG[0]-accG[-1]:+8.4f} kg ({100*(mG[-1]-mG[0]-accG[-1])/fedG:+6.2f} %) |"
print(line)
