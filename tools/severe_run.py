"""Corrida do severe slug com as opções do antigo driver `scratch/severe_D.py` (dialeto
curto: vSL, vSG, H, rv, fsw, delta, vol, ...; ver `mfsim.runs.case_from_severe_opts`).
Invólucro fino: o caso vem de `case_from_severe_opts`, a integração de `integrate_run`
(trajetória parcial gravada em caso de falha) e a gravação de `save_run` (npz v2 com o
caso completo em `case_json`). Para casos novos prefira `mfsim run caso.json`.

uso: uv run python tools/severe_run.py '{"vSL":0.3,"vSG":0.1,...}' tag [--out runs/tag.npz]
"""

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import boe_criterion, superficial_velocities  # noqa: E402
from mfsim.results import Results  # noqa: E402
from mfsim.runs import case_from_severe_opts, integrate_run, meta_from_severe_opts, save_run  # noqa: E402

opts = json.loads(sys.argv[1])
tag = sys.argv[2]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join("runs", f"{tag}.npz")
case = case_from_severe_opts(opts)
fluid = case.fluid
m = case.meta
vSL, vSG = superficial_velocities(case)
P_base = m["P_out_bar"] * 1e5 + fluid.rhoL0 * 9.81 * m["H_riser"]
print(
    "qL, qG:",
    m["qL"],
    m["qG"],
    "vSL, vSG(out):",
    vSL,
    vSG,
    "Boe:",
    boe_criterion(vSL, vSG, P_base, fluid.rhoL0, 0.5, m["L_h1"] + m["L_down"]),
    flush=True,
)
print("meta:", m, flush=True)
t0 = time.time()
model, T, X, info = integrate_run(case, opts, verbose=True, start=opts.get("start", "guess"), partial=True)
print("conservative:", model.conservative, flush=True)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
save_run(
    out,
    T,
    X,
    meta_from_severe_opts(opts, info),
    acc=info.get("accum"),
    info=info,
    command=" ".join(sys.argv),
    case=case,
)
if not info.get("success", True):
    print(
        "FAIL",
        info.get("error"),
        "steps",
        info.get("n_steps"),
        "rej",
        info.get("n_rejected"),
        time.time() - t0,
        flush=True,
    )
    print("parcial gravado:", T.size, "saídas até t =", T[-1] if T.size else None, "->", out, flush=True)
    sys.exit(1)
print("OK", info, time.time() - t0, "->", out, flush=True)
if opts.get("network"):
    print("rede: pós-processamento no notebook/tools/compare_network.py", flush=True)
    sys.exit(0)
res = Results(model, T, X)
p = res.pipe(0)
x = p["x"]
aG = p["out"]["alphaG"]
P = p["out"]["P_bar"]
qLo = p["out"]["qL"]
qGo = p["out"]["qG"]
ib = int(np.argmin(np.abs(x - (m["L_h1"] + m["L_down"]))))
sel = T > T[-1] / 2
print(
    f"qL_out: min {qLo[-1][sel].min():.3f} max {qLo[-1][sel].max():.3f} std {qLo[-1][sel].std():.3f} | P_in min {P[1][sel].min():.3f} max {P[1][sel].max():.3f}",
    flush=True,
)
A = np.pi * m["D"] ** 2 / 4 if "D" in m else np.pi * case.pipes[0].D ** 2 / 4
h_el = model.pipes[0].mesh["h"]
rhoL = fluid.rhoL(P[1:-1] * 1e5)
mass = np.sum((1 - aG[1:-1]) * rhoL * A * h_el[:, None], axis=0)
if model.conservative:
    sch = model.pipes[0].scheme
    FL = np.array([sch.mass_fluxes_nodal(p["Y"][it])[0] for it in range(T.size)])
    qin, qout = FL[:, 0], FL[:, -1]
else:
    qin, qout = qLo[0], qLo[-1]
net = np.concatenate([[0.0], np.cumsum(0.5 * ((qin[1:] - qout[1:]) + (qin[:-1] - qout[:-1])) * np.diff(T))])
print(
    f"massa líquida: inicial {mass[0]:.1f} kg, final {mass[-1]:.1f} kg; integral (entrada-saída) {net[-1]:.1f} kg; erro {mass[-1]-mass[0]-net[-1]:.1f} kg",
    flush=True,
)
if info.get("accum") is not None:
    acc = info["accum"]
    print(
        f"balanço BDF-consistente: líquido erro {mass[-1]-mass[0]-acc[0,-1]:.4f} kg; gás: estoque inicial {np.sum(aG[1:-1,0]*fluid.rhoG(P[1:-1,0]*1e5)*A*h_el):.4f} kg, erro {np.sum(aG[1:-1,-1]*fluid.rhoG(P[1:-1,-1]*1e5)*A*h_el)-np.sum(aG[1:-1,0]*fluid.rhoG(P[1:-1,0]*1e5)*A*h_el)-acc[1,-1]:.5f} kg",
        flush=True,
    )
for it in range(0, T.size, max(1, T.size // 60)):
    print(
        f"t={T[it]:6.0f} P_in={P[1,it]:6.3f} P_base={P[ib,it]:6.3f} aG_line={aG[5,it]:.3f}/{aG[ib-1,it]:.3f} aG_riser={aG[ib+1,it]:.3f}/{aG[min(ib+5,aG.shape[0]-1),it]:.3f}/{aG[min(ib+10,aG.shape[0]-1),it]:.3f} qL_out={qLo[-1,it]:6.3f} qG_out={qGo[-1,it]:.5f}",
        flush=True,
    )
