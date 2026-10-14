"""Resumo das corridas de validação com dados de laboratório (item 8): período, pressões e picos
contra os valores publicados. uso: uv run python tools/val_summary.py [tags...]  (tags em runs/<tag>.npz ou archive/scratch_2026-10/npz/severe_D_<tag>.npz,
ou nomes em notebooks/data/)"""
import numpy as np, json, sys, re, warnings
warnings.filterwarnings("ignore")
from mfsim.cases.severe_slug import severe_slug_case, boe_criterion
from mfsim.simulation import build_model
from mfsim.results import Results
from mfsim.fluid import Fluid
from mfsim.runs import case_from_severe_opts, flows_from_severe_opts, load_run

# referências experimentais: período (s), pressão na linha mín-máx (kPa abs), fonte
REF = {
    "fabre2": dict(periodo=137.0, P_min=126.0, P_max=235.0, fonte="Fabre et al. 1990 caso 2 (Tengesdal 2002, fig. 3-5): vSL 0.13, vSG0 0.20"),
    "fabre1": dict(periodo=70.0, P_min=128.0, P_max=210.0, fonte="Fabre et al. 1990 caso 1 (Tengesdal 2002, fig. 3-3): vSL 0.13, vSG0 0.45"),
    "jansen": dict(periodo=46.6, P_min=111.0, P_max=133.0, fonte="Jansen et al. 1996 tab. 1 (choke C = 120 kPa s2/m2): ULS 0.0959, UGSO 0.0753"),
    "teng1": dict(periodo=79.0, P_min=121.0, P_max=222.0, fonte="Tengesdal 2002 teste 1 (fig. 4-12/4-13, -1 grau): vSL 0.32, vSG 0.35"),
}

def rebuild(opts, T, tf):
    """Caso do dialeto do driver (mfsim.runs.case_from_severe_opts) e (fluido, A, qL, qG)."""
    case = case_from_severe_opts(opts, tf=tf)
    qL, qG, A, _ = flows_from_severe_opts(opts, case.fluid)
    return case, case.fluid, A, qL, qG


def load_opts(tag):
    """Corrida (`load_run`) e opções no dialeto do driver: as próprias nos ficheiros do driver,
    ou o `meta` das corridas gravadas traduzido (H, low_void, ...)."""
    run = load_run(tag)
    if run.opts is not None:
        return run, dict(run.opts)
    opts = dict(run.meta); opts.setdefault("H", opts.get("H_riser", 20.0)); opts.setdefault("low_void", opts.get("low_void_regime")); opts.setdefault("high_void", opts.get("high_void_regime")); opts.setdefault("vblend", opts.get("vertical_blend")); opts.setdefault("relax", opts.get("relax_time"))
    return run, opts


def blowdowns(T_, P_in, min_gap):
    lvl = P_in.min() + 0.3*(P_in.max() - P_in.min()); low = P_in < lvl; out = []; i = 0
    while i < T_.size:
        if low[i]:
            j = i
            while j < T_.size and low[j]: j += 1
            k = i + int(np.argmin(P_in[i:j]))
            if not out or T_[k] - out[-1] > min_gap: out.append(T_[k])
            i = j
        else: i += 1
    return np.array(out)

def summarize(tag):
    run, opts = load_opts(tag)
    T = run.T; case, fluid, A, qL, qG = rebuild(opts, T, T[-1]); model = build_model(case); r = Results(model, T, run.X); p = r.pipe(0)
    P1 = p["out"]["P_bar"][1]*100.0; q = p["out"]["qL"][-1]; x = p["x"]; aG = p["out"]["alphaG"]
    key = next((k for k in REF if k in tag), None); ref = REF.get(key, {})
    per_ref = ref.get("periodo", 100.0)
    sel = T > 0.2*T[-1]
    tb = blowdowns(T[sel], P1[sel], 0.3*per_ref); per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    m = case.meta; P_base = m["P_out_bar"]*1e5 + fluid.rhoL0*9.81*m["H_riser"]
    vSL = qL/fluid.rhoL0/A; vSG_out = qG/fluid.rhoG(m["P_out_bar"]*1e5)/A
    vmin, sev = boe_criterion(vSL, vSG_out, P_base, fluid.rhoL0, 1 - float(np.mean(1 - aG[1:m.get("n_h1", 2) + opts["n_down"], sel])), m["L_h1"] + m["L_down"])
    ib = int(np.argmin(np.abs(x - m["s_elbow0"])))
    print(f"{tag:18s} passos {opts.get('_n_steps', '?'):>6} | blowdowns {len(tb)} | período {per:6.1f} s (exp {ref.get('periodo', np.nan):6.1f}) | P_in {P1[sel].min():6.1f}-{P1[sel].max():6.1f} kPa (exp {ref.get('P_min', np.nan):.0f}-{ref.get('P_max', np.nan):.0f}) | pico líquido {q[sel].max():.3f} kg/s ({q[sel].max()/qL:.1f}x) | αG base riser {aG[ib+1, sel].min():.2f}-{aG[ib+1, sel].max():.2f} | Bøe vSL_min {vmin:.3f} ({'sim' if sev else 'não'})")
    return dict(tag=tag, tb=tb, per=per, P_min=P1[sel].min(), P_max=P1[sel].max(), ref=ref)

if __name__ == "__main__":
    for tag in sys.argv[1:]:
        try:
            summarize(tag)
        except Exception as e:
            print(tag, "erro:", e)
