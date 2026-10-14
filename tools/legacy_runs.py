"""Corridas gravadas no formato antigo (v1): o `meta` com as opções da corrida no dialeto do
antigo driver `scratch/severe_D.py` e os defaults do notebook `nb_severe_slug`.

Só para conversão e para o lançador no dialeto antigo (`tools/severe_run.py`) e `tools/convert_runs.py`: o simulador
e as análises consomem o formato v2 (`case_json`, lido por `mfsim.runs.load_run`).
`tools/convert_runs.py` chama `convert_run`, que reconstrói o caso do `meta` (`case_from_meta`,
cópia verificada do carregador do notebook: γL = 100 por omissão) e completa a secção
`integration` com o que o driver fazia (primeiro passo 1e-3, ordem 2, passo mínimo 1e-9,
acumulador dos fluxos de contorno, tolerância por faixa ou `atol = rtol/100`).
"""

from __future__ import annotations

import json
import os

import numpy as np

from generators.severe_slug import severe_slug_case
from mfsim.fluid import Fluid

# fluido do caso base do notebook (usado quando o meta não grava o campo)
NOTEBOOK_FLUID = dict(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)



def case_from_meta(meta: dict, tf: float, dt_out=None):
    """Grafo (`severe_slug_case`) reconstruído do `meta` de uma corrida gravada, com os
    defaults do notebook. `tf` é o instante final da trajetória (`T[-1]`)."""
    mb = meta
    fluid = Fluid(
        rhoL0=mb.get("rhoL", NOTEBOOK_FLUID["rhoL0"]),
        rhoG0=mb.get("rhoG0", NOTEBOOK_FLUID["rhoG0"]),
        muL=mb.get("muL", NOTEBOOK_FLUID["muL"]),
        muG=mb.get("muG", NOTEBOOK_FLUID["muG"]),
        gammaL=mb.get("gammaL", NOTEBOOK_FLUID["gammaL"]),
        gammaG=mb.get("gammaG", NOTEBOOK_FLUID["gammaG"]),
        P0=mb.get("P0", NOTEBOOK_FLUID["P0"]),
        sigma=mb.get("sigma", NOTEBOOK_FLUID["sigma"]),
    )
    D = mb.get("D", 0.05)
    A = np.pi * D**2 / 4
    P_out = mb.get("P_out", 4.0)
    P_gas = mb.get("P_gas_ref") or (P_out + 1.0)  # pressão (bar) em que vSG é dada
    qL = mb.get("vSL", 0.3) * fluid.rhoL0 * A
    qG = mb.get("vSG", 0.1) * fluid.rhoG(P_gas * 1e5) * A
    regime = mb.get("regime", 4)
    extra = {
        "regime_vertical": mb.get("regime_vertical", 0 if regime == 6 else 2),
        "k_switch": mb.get("k_switch", 1e6),
        "ipe_smooth": mb.get("ipe_smooth", False),
        "ipe_local": mb.get("ipe_local", False),
        "alpha_face": mb.get("alpha_face", "upwind"),
        "relax_target": mb.get("relax_target", "raw"),
        "fi_mult": mb.get("fi_mult", 1.0),
        "fi_vgt": mb.get("fi_vgt", 5.5),
        "virtual_mass": mb.get("virtual_mass", 0.0),
        "vm_lambda": mb.get("vm_lambda", 1.0),
        "bubble_drag": mb.get("bubble_drag", "tese"),
        "q_switch_L": 0.02 * qL,
        "q_switch_G": 0.02 * qG,
        "switch_frac": mb["switch_frac"],
        "relax_time": mb["relax_time"],
        "conservative": mb["conservative"],
        "vertical_blend": mb.get("vertical_blend", 0.0),
        "art_visc": mb.get("art_visc", 0.0),
        "low_void_regime": mb.get("low_void_regime", 0),
        "interfacial_pressure": mb.get("interfacial_pressure", 0.0),
        "high_void_regime": mb.get("high_void_regime", 0),
        "high_void_alpha": mb.get("high_void_alpha", 0.9),
        "vertex_volume": mb.get("vertex_volume", 0.05),
    }
    return severe_slug_case(
        method="fv_upwind",
        tf=tf,
        dt_out=mb.get("dt_out", 2.0) if dt_out is None else dt_out,
        qL=qL,
        qG=qG,
        D=D,
        P_out_bar=P_out,
        L_h1=mb.get("L_h1", 20.0),
        L_down=mb.get("L_down", 300.0),
        slope_down_deg=mb.get("slope", -3.0),
        H_riser=mb.get("H_riser", 20.0),
        L_h2=mb.get("L_h2", 5.0),
        n_h1=mb.get("n_h1", 2),
        n_down=mb["n_down"],
        n_riser=mb["n_riser"],
        n_h2=2,
        regime=regime,
        single_pipe=not mb.get("network", False),
        fluid=fluid,
        roughness=mb.get("roughness", 1e-5),
        start=mb.get("start", "blocked"),
        t_ramp=mb.get("t_ramp", 100.0),
        dP_margin_bar=mb.get("dP_margin", 0.3),
        elbow_radius=mb.get("elbow_radius", 0.0),
        n_elbow=mb.get("n_elbow", 0),
        grading_down=mb.get("grading_down", 1.0),
        grading_riser=mb.get("grading_riser", 1.0),
        upstream_volume=mb.get("upstream_volume", 0.0),
        extra=extra,
    )



def integrator_opts_from_meta(meta: dict) -> dict:
    """Opções do integrador gravadas no meta (nomes de `scratch/severe_D.py`)."""
    return dict(
        rtol=meta.get("rtol", 1e-3),
        tol_range=meta.get("tol_range"),
        atol_P=meta.get("atol_P"),
        atol_q=meta.get("atol_q"),
        order=2,
        max_newton=meta.get("max_newton", 12),
        newton_tol=1e-3,
        slow=bool(meta.get("slow", False)),
        jax=bool(meta.get("jax", False)),
    )



def fluid_from_severe_opts(opts) -> Fluid:
    return Fluid(
        rhoL0=opts.get("rhoL", 900.0),
        rhoG0=opts.get("rhoG0", 1.2),
        muL=opts.get("muL", 2e-3),
        muG=opts.get("muG", 1.8e-5),
        gammaL=opts.get("gammaL", 1000.0),
        gammaG=opts.get("gammaG", 1.05),
        P0=opts.get("P0", 1e5),
        sigma=opts.get("sigma", 0.05),
    )



def flows_from_severe_opts(opts, fluid=None):
    """(qL, qG, A, P_out) do dialeto do driver: vSL/vSG com vSG dada à pressão P_gas_ref (bar;
    P_out + 1 por omissão; laboratório: condições padrão)."""
    fluid = fluid or fluid_from_severe_opts(opts)
    D = opts.get("D", 0.05)
    A = np.pi * D**2 / 4
    P_out = opts.get("P_out", 4.0)
    P_est = opts.get("P_gas_ref", P_out + 1.0) * 1e5
    return opts.get("vSL", 0.3) * fluid.rhoL0 * A, opts.get("vSG", 0.1) * fluid.rhoG(P_est) * A, A, P_out



def case_from_severe_opts(opts, tf=None, dt_out=None):
    """Grafo `severe_slug_case` construído como o driver `scratch/severe_D.py` (verbatim)."""
    fluid = fluid_from_severe_opts(opts)
    qL, qG, A, P_out = flows_from_severe_opts(opts, fluid)
    D = opts.get("D", 0.05)
    tf = opts.get("tf", 4000.0) if tf is None else tf
    dt_out = opts.get("dt_out", 2.0) if dt_out is None else dt_out
    start = opts.get("start", "guess")
    extra = {
        k: v
        for k, v in {
            "regime_vertical": opts.get("rv", 0 if opts.get("regime", 4) == 6 else 2),
            "k_switch": opts.get("k_switch"),
            "ipe_smooth": opts.get("ipe_smooth"),
            "ipe_local": opts.get("ipe_local"),
            "alpha_face": opts.get("alpha_face"),
            "fv_order": opts.get("fv_order"),
            "relax_target": opts.get("relax_target"),
            "drift_relax": opts.get("drift_relax"),
            "bb_derivatives": opts.get("bb_derivatives"),
            "ic_holdup": opts.get("ic_holdup"),
            "fi_mult": opts.get("fi_mult"),
            "fi_vgt": opts.get("fi_vgt"),
            "virtual_mass": opts.get("virtual_mass"),
            "vm_lambda": opts.get("vm_lambda"),
            "bubble_drag": opts.get("bubble_drag"),
            "q_switch_L": opts.get("fsw", 0.02) * qL,
            "q_switch_G": opts.get("fsw_G", opts.get("fsw", 0.02)) * qG,
            "switch_frac": opts.get("switch_frac"),
            "relax_time": opts.get("relax"),
            "limit_alpha": opts.get("limit"),
            "conservative": opts.get("conservative"),
            "limit_band": opts.get("band"),
            "flux_band": opts.get("flux_band"),
            "flux_eps": opts.get("flux_eps"),
            "flux_limit": opts.get("flux_limit"),
            "vertical_blend": opts.get("vblend"),
            "art_visc": opts.get("art_visc"),
            "low_void_regime": opts.get("low_void"),
            "high_void_regime": opts.get("high_void"),
            "high_void_alpha": opts.get("high_void_alpha"),
            "interfacial_pressure": opts.get("delta"),
            "vertex_volume": opts.get("vol"),
        }.items()
        if v is not None
    }
    return severe_slug_case(
        method=opts.get("method", "fv_upwind"),
        tf=tf,
        dt_out=dt_out,
        qL=qL,
        qG=qG,
        D=D,
        P_out_bar=P_out,
        L_h1=opts.get("L_h1", 20.0),
        L_down=opts.get("L_down", 300.0),
        slope_down_deg=opts.get("slope", -3.0),
        H_riser=opts.get("H", 20.0),
        L_h2=opts.get("L_h2", 5.0),
        n_h1=opts.get("n_h1", 2),
        n_down=opts.get("n_down", 30),
        n_riser=opts.get("n_riser", 10),
        n_h2=opts.get("n_h2", 2),
        regime=opts.get("regime", 4),
        single_pipe=not opts.get("network", 0),
        fluid=fluid,
        roughness=opts.get("roughness", 1e-5),
        start=start,
        t_ramp=opts.get("t_ramp", 100.0),
        pocket_alpha=opts.get("pocket_alpha"),
        front_liquid_length=opts.get("front", 0.0),
        dP_margin_bar=opts.get("dP_margin", 0.3),
        elbow_radius=opts.get("elbow_R", 0.0),
        n_elbow=opts.get("n_elbow", 0),
        grading_down=opts.get("g_down", 1.0),
        grading_riser=opts.get("g_riser", 1.0),
        model=opts.get("model", 1),
        upstream_volume=opts.get("V_up", 0.0),
        extra=extra,
    )



def meta_from_severe_opts(opts, info=None) -> dict:
    """`meta` v1 de uma corrida a partir das opções do driver (era scratch/save_run_data.py)."""
    info = info or {}
    return dict(
        parcial=bool(opts.get("_parcial", False)) or (info.get("success") is False),
        gammaG=opts.get("gammaG", 1.05),
        gammaL=opts.get("gammaL", 1000.0),
        L_h1=opts.get("L_h1", 20.0),
        n_h1=opts.get("n_h1", 2),
        vSL=opts.get("vSL", 0.3),
        vSG=opts.get("vSG", 0.1),
        L_down=opts.get("L_down", 300.0),
        P_out=opts.get("P_out", 4.0),
        H_riser=opts.get("H", 20.0),
        slope=opts.get("slope", -3.0),
        tf=opts.get("tf", 4000.0),
        n_down=opts.get("n_down", 30),
        n_riser=opts.get("n_riser", 10),
        relax_time=opts.get("relax", 0.05),
        switch_frac=opts.get("switch_frac", 0.5),
        rtol=opts.get("rtol", 1e-3),
        start=opts.get("start", "guess"),
        conservative=bool(opts.get("conservative", 0)),
        elbow_radius=opts.get("elbow_R", 0.0),
        n_elbow=opts.get("n_elbow", 0),
        vertical_blend=opts.get("vblend", 0.0),
        grading_down=opts.get("g_down", 1.0),
        grading_riser=opts.get("g_riser", 1.0),
        art_visc=opts.get("art_visc", 0.0),
        low_void_regime=opts.get("low_void", 0),
        high_void_regime=opts.get("high_void", 0),
        high_void_alpha=opts.get("high_void_alpha", 0.9),
        interfacial_pressure=opts.get("delta", 0.0),
        max_newton=opts.get("max_newton", 12),
        atol_P=opts.get("atol_P"),
        atol_q=opts.get("atol_q"),
        network=bool(opts.get("network", 0)),
        vertex_volume=opts.get("vol", 0.05),
        tol_range=opts.get("tol_range"),
        jax=bool(opts.get("jax", 0)),
        regime=opts.get("regime", 4),
        regime_vertical=opts.get("rv", 0 if opts.get("regime", 4) == 6 else 2),
        k_switch=opts.get("k_switch", 1e6),
        ipe_smooth=bool(opts.get("ipe_smooth", 0)),
        ipe_local=bool(opts.get("ipe_local", 0)),
        fv_order=opts.get("fv_order", 1),
        alpha_face=opts.get("alpha_face", "upwind"),
        relax_target=opts.get("relax_target", "raw"),
        model=opts.get("model", 1),
        drift_relax=opts.get("drift_relax", 0.05),
        upstream_volume=opts.get("V_up", 0.0),
        fi_mult=opts.get("fi_mult", 1.0),
        fi_vgt=opts.get("fi_vgt", 5.5),
        virtual_mass=opts.get("virtual_mass", 0.0),
        vm_lambda=opts.get("vm_lambda", 1.0),
        bubble_drag=opts.get("bubble_drag", "tese"),
        t_ramp=opts.get("t_ramp", 100.0),
        fsw=opts.get("fsw", 0.02),
        fsw_G=opts.get("fsw_G", opts.get("fsw", 0.02)),
        slow=bool(opts.get("slow", 0)),
        dP_margin=opts.get("dP_margin", 0.3),
        D=opts.get("D", 0.05),
        rhoL=opts.get("rhoL", 900.0),
        muL=opts.get("muL", 2e-3),
        rhoG0=opts.get("rhoG0", 1.2),
        muG=opts.get("muG", 1.8e-5),
        P0=opts.get("P0", 1e5),
        sigma=opts.get("sigma", 0.05),
        P_gas_ref=opts.get("P_gas_ref"),
        L_h2=opts.get("L_h2", 5.0),
        dt_out=opts.get("dt_out", 2.0),
        roughness=opts.get("roughness", 1e-5),
        n_steps=info.get("n_steps"),
        n_rejected=info.get("n_rejected"),
        cpu=info.get("cpu"),
    )



META_RUN_KEYS = ("n_steps", "n_rejected", "cpu", "parcial", "note")


def driver_opts_from_meta(meta) -> dict:
    """`meta` v1 traduzido para o dialeto do driver (H, low_void, high_void, vblend, relax)."""
    opts = dict(meta)
    opts.setdefault("H", opts.get("H_riser", 20.0))
    opts.setdefault("low_void", opts.get("low_void_regime"))
    opts.setdefault("high_void", opts.get("high_void_regime"))
    opts.setdefault("vblend", opts.get("vertical_blend"))
    opts.setdefault("relax", opts.get("relax_time"))
    return opts


def apply_driver_integration(net, meta):
    """Secção `integration` e política de estado inicial de um grafo a partir do `meta`/`opts`
    do driver: o que `integrate_run`/`severe_D.py` impunham (verbatim)."""
    o = integrator_opts_from_meta(meta)
    kw = dict(integrator="implicit", rtol=o["rtol"], tol_range=o["tol_range"], atol_P=o["atol_P"], atol_q=o["atol_q"],
              max_newton=int(o["max_newton"]), newton_tol=1e-3, max_order=2, first_step=1e-3, min_step=1e-9,
              slow_exit=bool(o["slow"]), backend="jax" if o["jax"] else "numpy", accumulate=True,
              tight_bounds=bool(meta.get("tightbounds", False)))
    if o["tol_range"] is None:
        kw["atol"] = o["rtol"] * 1e-2  # base da tolerância absoluta do driver (vetor uniforme)
    if meta.get("predict_space"):
        kw["predict_space"] = meta["predict_space"]
    if meta.get("jt_refresh"):
        kw["jt_refresh"] = meta["jt_refresh"]
    net.set_integration(**kw)
    net.initial = "edges" if meta.get("start", "blocked") in ("blocked", "shut_in") else "guess"
    return net


def network_from_meta(meta, T, name):
    """Grafo completo (caso + integração + meta com os parâmetros do gerador) de uma corrida v1."""
    net = case_from_meta(meta, tf=float(T[-1]))
    apply_driver_integration(net, meta)
    net.name = name
    net.meta = dict(net.meta, run=name, generator="severe_slug_case",
                    generator_params={k: v for k, v in meta.items() if k not in META_RUN_KEYS})
    return net


def convert_run(src, dst=None):
    """Converte um ficheiro de corrida v1 (`meta`, ou `opts` do driver) para o formato v2
    (`case_json`, `info_json`, `provenance_json`); devolve o caminho escrito, ou None se o
    ficheiro já está em v2. Escrita atómica (ficheiro temporário + substituição)."""
    from mfsim.runs import save_run

    dst = dst or src
    with np.load(src, allow_pickle=False) as d:
        if "case_json" in d.files:
            return None
        if "meta" in d.files:
            meta = json.loads(str(d["meta"]))
        elif "opts" in d.files:
            meta = meta_from_severe_opts(json.loads(str(d["opts"])))
        else:
            raise ValueError(f"{src}: sem meta nem opts")
        missing = [k for k in ("n_down", "n_riser", "switch_frac", "relax_time", "conservative") if k not in meta]
        if missing:
            raise ValueError(f"{src}: meta incompleto (faltam {missing}); o caso não é reconstruível")
        T, X = np.asarray(d["T"], float), np.asarray(d["X"], float)
        acc = np.asarray(d["acc"]) if "acc" in d.files else None
    name = os.path.splitext(os.path.basename(src))[0]
    net = network_from_meta(meta, T, name)
    info = {k: meta[k] for k in ("n_steps", "n_rejected", "cpu") if meta.get(k) is not None}
    info["success"] = not bool(meta.get("parcial", False))
    if meta.get("note"):
        info["note"] = meta["note"]
    tmp = dst + ".tmp.npz"
    save_run(tmp, T, X, net, info=info, acc=acc, command=f"tools/convert_runs.py {src}")
    os.replace(tmp, dst)
    return dst
