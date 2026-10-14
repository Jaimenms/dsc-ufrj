"""Converte scratch/severe_D_<tag>.npz (+ log) em notebooks/data/<nome>.npz com meta.
uso: uv run python scratch/save_run_data.py <tag> <nome>"""
import numpy as np, json, re, sys, ast
tag, name = sys.argv[1], sys.argv[2]
d = np.load(f"scratch/severe_D_{tag}.npz"); opts = json.loads(str(d["opts"]))
log = open(f"scratch/severe_D_{tag}.log").read()
info = {}
for key, pat in (("n_steps", r"'n_steps': (\d+)"), ("n_rejected", r"'n_rejected': (\d+)"), ("cpu", r"'cpu': ([0-9.]+)")):
    mm = re.search(pat, log)
    if mm:
        info[key] = float(mm.group(1)) if key == "cpu" else int(mm.group(1))
meta = dict(parcial=bool(opts.get("_parcial", False)), gammaG=opts.get("gammaG", 1.05), gammaL=opts.get("gammaL", 1000.0), L_h1=opts.get("L_h1", 20.0), n_h1=opts.get("n_h1", 2), vSL=opts.get("vSL", 0.3), vSG=opts.get("vSG", 0.1), L_down=opts.get("L_down", 300.0), P_out=opts.get("P_out", 4.0), H_riser=opts.get("H", 20.0), slope=opts.get("slope", -3.0), tf=opts.get("tf", 4000.0),
            n_down=opts.get("n_down", 30), n_riser=opts.get("n_riser", 10), relax_time=opts.get("relax", 0.05), switch_frac=opts.get("switch_frac", 0.5), rtol=opts.get("rtol", 1e-3),
            start=opts.get("start", "guess"), conservative=bool(opts.get("conservative", 0)),
            elbow_radius=opts.get("elbow_R", 0.0), n_elbow=opts.get("n_elbow", 0), vertical_blend=opts.get("vblend", 0.0),
            grading_down=opts.get("g_down", 1.0), grading_riser=opts.get("g_riser", 1.0),
            art_visc=opts.get("art_visc", 0.0), low_void_regime=opts.get("low_void", 0), high_void_regime=opts.get("high_void", 0), high_void_alpha=opts.get("high_void_alpha", 0.9), interfacial_pressure=opts.get("delta", 0.0),
            max_newton=opts.get("max_newton", 12), atol_P=opts.get("atol_P"), atol_q=opts.get("atol_q"),
            network=bool(opts.get("network", 0)), vertex_volume=opts.get("vol", 0.05), tol_range=opts.get("tol_range"), jax=bool(opts.get("jax", 0)),
            regime=opts.get("regime", 4), regime_vertical=opts.get("rv", 0 if opts.get("regime", 4) == 6 else 2), k_switch=opts.get("k_switch", 1e6), ipe_smooth=bool(opts.get("ipe_smooth", 0)), ipe_local=bool(opts.get("ipe_local", 0)), fv_order=opts.get("fv_order", 1), alpha_face=opts.get("alpha_face", "upwind"), relax_target=opts.get("relax_target", "raw"), model=opts.get("model", 1), drift_relax=opts.get("drift_relax", 0.05), upstream_volume=opts.get("V_up", 0.0), fi_mult=opts.get("fi_mult", 1.0), fi_vgt=opts.get("fi_vgt", 5.5), virtual_mass=opts.get("virtual_mass", 0.0), vm_lambda=opts.get("vm_lambda", 1.0), bubble_drag=opts.get("bubble_drag", "tese"), t_ramp=opts.get("t_ramp", 100.0), fsw=opts.get("fsw", 0.02), fsw_G=opts.get("fsw_G", opts.get("fsw", 0.02)), slow=bool(opts.get("slow", 0)), dP_margin=opts.get("dP_margin", 0.3),
            D=opts.get("D", 0.05), rhoL=opts.get("rhoL", 900.0), muL=opts.get("muL", 2e-3), rhoG0=opts.get("rhoG0", 1.2), muG=opts.get("muG", 1.8e-5), P0=opts.get("P0", 1e5), sigma=opts.get("sigma", 0.05), P_gas_ref=opts.get("P_gas_ref"), L_h2=opts.get("L_h2", 5.0), dt_out=opts.get("dt_out", 2.0), roughness=opts.get("roughness", 1e-5),
            n_steps=info.get("n_steps"), n_rejected=info.get("n_rejected"), cpu=info.get("cpu"))
extra = {"acc": d["acc"]} if "acc" in d.files else {}
np.savez(f"notebooks/data/{name}.npz", T=d["T"], X=d["X"], meta=json.dumps(meta), **extra)
print("saved", name, meta)
