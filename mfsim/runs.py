"""Corridas gravadas: um só carregador para os ficheiros `notebooks/data/*.npz`.

Substitui as seis cópias da lógica "reconstruir o caso a partir do `meta`" (notebook
`load_blocked_run`, cycle_stats, scan_summary, riser_cc, val_summary.rebuild,
analyze_run). Os defaults são os do notebook `nb_severe_slug` — em particular
`gammaL = 100`, o valor com que as corridas de referência foram feitas (67 dos 80
ficheiros não gravam `gammaL`; `scratch/severe_D.py` usava 1000 por omissão, mas os
resultados guardados são bit-idênticos aos obtidos com 100).

Formato dos ficheiros (v1): arrays `T` (nt,), `X` (n_states, nt), opcional `acc`
(integral BDF-consistente dos fluxos de contorno) e `meta` (JSON com as opções da
corrida nos nomes de `scratch/save_run_data.py`). `save_run` grava ainda `info_json` e
`provenance_json`; na fase 1 da reorganização o `meta` passa a derivar de `case_json`.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

from .cases.severe_slug import severe_slug_case
from .fluid import Fluid
from .simulation import build_model, state_bounds

# fluido do caso base do notebook (usado quando o meta não grava o campo)
NOTEBOOK_FLUID = dict(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)


def case_from_meta(meta: dict, tf: float, dt_out=None):
    """Caso `severe_slug_case` reconstruído do `meta` de uma corrida gravada, com os
    defaults do notebook. `tf` é o instante final da trajetória (`T[-1]`)."""
    mb = meta
    fluid = Fluid(rhoL0=mb.get("rhoL", NOTEBOOK_FLUID["rhoL0"]), rhoG0=mb.get("rhoG0", NOTEBOOK_FLUID["rhoG0"]),
                  muL=mb.get("muL", NOTEBOOK_FLUID["muL"]), muG=mb.get("muG", NOTEBOOK_FLUID["muG"]),
                  gammaL=mb.get("gammaL", NOTEBOOK_FLUID["gammaL"]), gammaG=mb.get("gammaG", NOTEBOOK_FLUID["gammaG"]),
                  P0=mb.get("P0", NOTEBOOK_FLUID["P0"]), sigma=mb.get("sigma", NOTEBOOK_FLUID["sigma"]))
    D = mb.get("D", 0.05)
    A = np.pi * D**2 / 4
    P_out = mb.get("P_out", 4.0)
    P_gas = mb.get("P_gas_ref") or (P_out + 1.0)          # pressão (bar) em que vSG é dada
    qL = mb.get("vSL", 0.3) * fluid.rhoL0 * A
    qG = mb.get("vSG", 0.1) * fluid.rhoG(P_gas * 1e5) * A
    regime = mb.get("regime", 4)
    extra = {"regime_vertical": mb.get("regime_vertical", 0 if regime == 6 else 2), "k_switch": mb.get("k_switch", 1e6),
             "ipe_smooth": mb.get("ipe_smooth", False), "ipe_local": mb.get("ipe_local", False),
             "alpha_face": mb.get("alpha_face", "upwind"), "relax_target": mb.get("relax_target", "raw"),
             "fi_mult": mb.get("fi_mult", 1.0), "fi_vgt": mb.get("fi_vgt", 5.5), "virtual_mass": mb.get("virtual_mass", 0.0),
             "vm_lambda": mb.get("vm_lambda", 1.0), "bubble_drag": mb.get("bubble_drag", "tese"),
             "q_switch_L": 0.02 * qL, "q_switch_G": 0.02 * qG, "switch_frac": mb["switch_frac"],
             "relax_time": mb["relax_time"], "conservative": mb["conservative"],
             "vertical_blend": mb.get("vertical_blend", 0.0), "art_visc": mb.get("art_visc", 0.0),
             "low_void_regime": mb.get("low_void_regime", 0), "interfacial_pressure": mb.get("interfacial_pressure", 0.0),
             "high_void_regime": mb.get("high_void_regime", 0), "high_void_alpha": mb.get("high_void_alpha", 0.9),
             "vertex_volume": mb.get("vertex_volume", 0.05)}
    return severe_slug_case(method="fv_upwind", tf=tf, dt_out=mb.get("dt_out", 2.0) if dt_out is None else dt_out,
                            qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=mb.get("L_h1", 20.0), L_down=mb.get("L_down", 300.0),
                            slope_down_deg=mb.get("slope", -3.0), H_riser=mb.get("H_riser", 20.0), L_h2=mb.get("L_h2", 5.0),
                            n_h1=mb.get("n_h1", 2), n_down=mb["n_down"], n_riser=mb["n_riser"], n_h2=2,
                            regime=regime, single_pipe=not mb.get("network", False), fluid=fluid,
                            roughness=mb.get("roughness", 1e-5), start=mb.get("start", "blocked"),
                            t_ramp=mb.get("t_ramp", 100.0), dP_margin_bar=mb.get("dP_margin", 0.3),
                            elbow_radius=mb.get("elbow_radius", 0.0), n_elbow=mb.get("n_elbow", 0),
                            grading_down=mb.get("grading_down", 1.0), grading_riser=mb.get("grading_riser", 1.0),
                            upstream_volume=mb.get("upstream_volume", 0.0), extra=extra)


def integrator_opts_from_meta(meta: dict) -> dict:
    """Opções do integrador gravadas no meta (nomes de `scratch/severe_D.py`)."""
    return dict(rtol=meta.get("rtol", 1e-3), tol_range=meta.get("tol_range"), atol_P=meta.get("atol_P"),
                atol_q=meta.get("atol_q"), order=2, max_newton=meta.get("max_newton", 12), newton_tol=1e-3,
                slow=bool(meta.get("slow", False)), jax=bool(meta.get("jax", False)))


def integrator_opts_from_case(case) -> dict:
    """Opções do integrador de um caso JSON (secção `integration`, que `Network.to_legacy_case`
    escreve em `settings.extra`), nos nomes de `integrate_run`."""
    ex = dict(getattr(case.settings, "extra", {}) or {})
    return dict(rtol=float(case.settings.rtol), tol_range=ex.get("tol_range"), atol_P=ex.get("atol_P"), atol_q=ex.get("atol_q"),
                order=int(ex.get("max_order", 2)), max_newton=int(ex.get("max_newton", 12)), newton_tol=float(ex.get("newton_tol", 1e-3)),
                slow=bool(ex.get("slow_exit", False)), jax=str(ex.get("backend", "numpy")) == "jax",
                first_step=float(ex.get("first_step", 1e-3)))


def integrate_run(case, opts=None, verbose=False, t_eval=None, start=None, partial=False):
    """Integra um caso como o driver `scratch/severe_D.py`: estado inicial da partida
    (bloqueada/linha fechada) ou estimativa, tolerância absoluta por faixa
    (`tol_range`), Jacobiana exata em JAX (`jax`), acumulador dos fluxos de contorno.

    opts: rtol (1e-3), tol_range, atol_P, atol_q, order (2), max_newton (12),
    newton_tol (1e-3), slow (False), predict_space, jt_refresh, jax (False),
    tightbounds (False), first_step (1e-3). Devolve (model, T, X, info); com
    `partial=True` uma falha do integrador devolve a trajetória integrada até aí
    (info["success"] = False, info["error"]) em vez de levantar a exceção."""
    import time as _time
    from .initial_guess import initial_guess
    from .integrators.implicit import ImplicitBDF
    opts = dict(opts or {})
    model = build_model(case)
    start = start or getattr(case, "meta", {}).get("start")
    X0 = model.initial_state() if start in ("shut_in", "blocked") or case.pipes[0].ic is not None else initial_guess(model)
    lo, hi = state_bounds(model)
    if opts.get("tightbounds"):
        pb = model.pipes[0]; slots = pb.free % 4
        lo[slots == 0] = 1e-3; hi[slots == 0] = 1 - 1e-3
    kw_t = dict(transform=model.transform, inverse=model.inverse_transform, transform_jac=model.transform_jacobian) if model.conservative else {}
    rtol = opts.get("rtol", 1e-3)
    atol = rtol * 1e-2 * np.ones(X0.size)
    slots = model.pipes[0].free % 4
    if opts.get("atol_P"):
        atol[slots == 1] = opts["atol_P"]
    if opts.get("atol_q"):
        atol[(slots == 2) | (slots == 3)] = opts["atol_q"]
    if opts.get("tol_range"):
        atol = opts["tol_range"] * model.state_scales(X0)
    fun, jac = model.rhs, None
    if opts.get("jax"):
        from .compiled import build_jax_rhs
        fun, jac = build_jax_rhs(model)
    bdf = ImplicitBDF(fun, S=model.jac_sparsity(), rtol=rtol, atol=atol, bounds=(lo, hi), jac_fun=jac, min_step=1e-9,
                      verbose=verbose, max_order=opts.get("order", 2), max_newton=opts.get("max_newton", 12),
                      newton_tol=opts.get("newton_tol", 1e-3), **kw_t)
    bdf.accum = model.boundary_net_fluxes
    bdf.slow_exit = bool(opts.get("slow", False))
    if opts.get("predict_space"):
        bdf.predict_space = opts["predict_space"]
    if opts.get("jt_refresh"):
        bdf.jt_refresh = opts["jt_refresh"]
    if t_eval is None:
        dt_out = case.settings.dt_out
        t_eval = np.arange(0.0, case.settings.tf + dt_out / 2, dt_out)
    t_start = _time.time()
    try:
        T, X, info = bdf.integrate(t_eval, X0, first_step=float(opts.get("first_step", 1e-3)), breakpoints=model.spec_breakpoints())
    except Exception as e:
        if not partial:
            raise
        out = getattr(bdf, "_out", None) or []
        X = np.array(out).T if out else np.zeros((X0.size, 0))
        T = np.asarray(getattr(bdf, "_t_eval", t_eval), float)[:X.shape[1]]
        info = dict(success=False, error=f"{type(e).__name__}: {e}", n_steps=bdf.n_steps, n_rejected=bdf.n_rejected,
                    n_rhs=bdf.n_rhs, n_jac=bdf.n_jac, cpu=_time.time() - t_start)
    return model, T, X, info


@dataclass
class Run:
    """Trajetória gravada + caso reconstruído (modelo e resultados preguiçosos)."""

    path: str
    T: np.ndarray
    X: np.ndarray
    meta: dict
    acc: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    _case: object = None
    _model: object = None
    _res: object = None

    case_json: dict | None = None
    ic_arrays: dict = field(default_factory=dict)
    opts: dict | None = None          # opções do driver (ficheiros antigos de scratch/severe_D)

    @property
    def case(self):
        if self._case is None:
            if self.case_json:
                from .io.spec import case_from_json
                self._case = case_from_json(self.case_json, ic_arrays=self.ic_arrays)
                self._case.settings.tf = float(self.T[-1]) if self.T.size else self._case.settings.tf
            else:
                self._case = case_from_meta(self.meta, tf=float(self.T[-1]))
        return self._case

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.case)
        return self._model

    @property
    def res(self):
        if self._res is None:
            from .results import Results
            self._res = Results(self.model, self.T, self.X)
        return self._res

    def pipe(self, k=0):
        return self.res.pipe(k)

    @property
    def p_last(self):
        """Saídas do duto que termina no separador (o último, em rede)."""
        return self.res.pipe(len(self.model.pipes) - 1)

    @property
    def ib(self):
        """Célula no início do joelho (base do riser) do primeiro duto."""
        p = self.pipe(0)
        return int(np.argmin(np.abs(p["x"] - self.case.meta["s_elbow0"])))

    def aG_riser_base(self):
        if self.meta.get("network", False):
            return self.res.pipe(2)["out"]["alphaG"][1]
        return self.pipe(0)["out"]["alphaG"][self.ib + self.meta.get("n_elbow", 0) + 1]

    def balance(self, phase="L", k=0):
        from .analysis import mass_balance
        return mass_balance(self.res, k, phase)

    def signature(self, min_gap=200.0, t_skip=0.0):
        from .analysis import cycle_signature
        P_in = self.pipe(0)["out"]["P_bar"][1]
        q_out = self.p_last["out"]["qL"][-1]
        return cycle_signature(self.T, P_in, q_out, min_gap=min_gap, t_skip=t_skip)

    def integrator_opts(self):
        return integrator_opts_from_meta(self.meta)


RUN_DIRS = ("runs", os.path.join("notebooks", "data"), os.path.join("archive", "scratch_2026-10", "npz"))


def find_run(name) -> str:
    """Caminho de uma corrida: o próprio caminho, ou `<nome>.npz` / `severe_D_<nome>.npz` em
    `runs/`, `notebooks/data/` e `archive/scratch_2026-10/npz/`."""
    name = str(name)
    if os.path.exists(name):
        return name
    base = name[:-4] if name.endswith(".npz") else name
    for d in RUN_DIRS:
        for fn in (f"{base}.npz", f"severe_D_{base}.npz"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"corrida não encontrada: {name} (procurado em {RUN_DIRS})")


def load_run(path) -> Run:
    """Lê um npz gravado (caminho, ou nome procurado por `find_run`). Os ficheiros antigos do
    driver (`opts` em vez de `meta`) recebem o `meta` derivado por `meta_from_severe_opts`."""
    path = find_run(path)
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        opts = json.loads(str(d["opts"])) if "opts" in d.files else None
        if not meta and opts is not None:
            meta = meta_from_severe_opts(opts)
        info = json.loads(str(d["info_json"])) if "info_json" in d.files else {}
        prov = json.loads(str(d["provenance_json"])) if "provenance_json" in d.files else {}
        case_json = json.loads(str(d["case_json"])) if "case_json" in d.files else None
        ic_arrays = {k[3:]: np.asarray(d[k], float) for k in d.files if k.startswith("ic/")}
        return Run(path=str(path), T=np.asarray(d["T"], float), X=np.asarray(d["X"], float), meta=meta,
                   acc=np.asarray(d["acc"]) if "acc" in d.files else None, info=info, provenance=prov,
                   case_json=case_json, ic_arrays=ic_arrays, opts=opts)


def provenance(command=None) -> dict:
    """Proveniência de uma corrida: commit git, versões e plataforma."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        sha = ""
    import numpy, scipy
    try:
        import jax
        jax_v = jax.__version__
    except ImportError:
        jax_v = None
    return dict(git=sha, python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__, jax=jax_v,
                platform=platform.platform(), command=command or " ".join(sys.argv))


def save_run(path, T, X, meta, acc=None, info=None, command=None, case=None):
    """Grava uma corrida: `T`, `X`, `acc` (opcional), `meta` (JSON v1, compatibilidade),
    `info_json` (n_steps, n_rejected, cpu, n_rhs, n_jac, success, error) e
    `provenance_json`; com `case` (Case legado ou Network) grava ainda `case_json`
    (o caso completo em `mfsim-case/1`) e as condições iniciais explícitas em `ic/<duto>`
    (formato v2: `load_run` reconstrói o caso do JSON sem passar pelo `meta`)."""
    extra = {}
    if acc is not None:
        extra["acc"] = np.asarray(acc)
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in (info or {}).items() if k in ("n_steps", "n_rejected", "cpu", "n_rhs", "n_jac", "success", "error")}
    if case is not None:
        from .io.spec import network_to_json
        from .network.graph import Network
        net = case if isinstance(case, Network) else Network.from_legacy_case(case)
        d, ic_store = network_to_json(net)
        extra["case_json"] = json.dumps(d)
        for k, v in ic_store.items():
            extra[f"ic/{k}"] = np.asarray(v, float)
    np.savez(path, T=np.asarray(T, float), X=np.asarray(X, float), meta=json.dumps(meta),
             info_json=json.dumps(info), provenance_json=json.dumps(provenance(command)), **extra)
    return path


# ----------------------------------------------------------------------------- dialeto do driver
# As opções do antigo `scratch/severe_D.py` (agora `tools/severe_run.py`): nomes curtos
# ("H", "rv", "fsw", "delta", "vol", "elbow_R", "g_down", "low_void", ...) com os defaults do
# driver (gammaL 1000). `meta_from_severe_opts` é o mapa (antes em scratch/save_run_data.py)
# para o `meta` v1 das corridas gravadas; `case_from_severe_opts` constrói o caso exatamente
# como o driver (inclui opções que o `meta` não guarda: pocket_alpha, front, bb_derivatives,
# ic_holdup, model, fsw).

def fluid_from_severe_opts(opts) -> Fluid:
    return Fluid(rhoL0=opts.get("rhoL", 900.0), rhoG0=opts.get("rhoG0", 1.2), muL=opts.get("muL", 2e-3), muG=opts.get("muG", 1.8e-5),
                 gammaL=opts.get("gammaL", 1000.0), gammaG=opts.get("gammaG", 1.05), P0=opts.get("P0", 1e5), sigma=opts.get("sigma", 0.05))


def flows_from_severe_opts(opts, fluid=None):
    """(qL, qG, A, P_out) do dialeto do driver: vSL/vSG com vSG dada à pressão P_gas_ref (bar;
    P_out + 1 por omissão; laboratório: condições padrão)."""
    fluid = fluid or fluid_from_severe_opts(opts)
    D = opts.get("D", 0.05); A = np.pi * D**2 / 4
    P_out = opts.get("P_out", 4.0)
    P_est = opts.get("P_gas_ref", P_out + 1.0) * 1e5
    return opts.get("vSL", 0.3) * fluid.rhoL0 * A, opts.get("vSG", 0.1) * fluid.rhoG(P_est) * A, A, P_out


def case_from_severe_opts(opts, tf=None, dt_out=None):
    """Caso `severe_slug_case` construído como o driver `scratch/severe_D.py` (verbatim)."""
    fluid = fluid_from_severe_opts(opts)
    qL, qG, A, P_out = flows_from_severe_opts(opts, fluid)
    D = opts.get("D", 0.05)
    tf = opts.get("tf", 4000.0) if tf is None else tf
    dt_out = opts.get("dt_out", 2.0) if dt_out is None else dt_out
    start = opts.get("start", "guess")
    extra = {k: v for k, v in {"regime_vertical": opts.get("rv", 0 if opts.get("regime", 4) == 6 else 2), "k_switch": opts.get("k_switch"), "ipe_smooth": opts.get("ipe_smooth"), "ipe_local": opts.get("ipe_local"), "alpha_face": opts.get("alpha_face"), "fv_order": opts.get("fv_order"), "relax_target": opts.get("relax_target"), "drift_relax": opts.get("drift_relax"), "bb_derivatives": opts.get("bb_derivatives"), "ic_holdup": opts.get("ic_holdup"), "fi_mult": opts.get("fi_mult"), "fi_vgt": opts.get("fi_vgt"), "virtual_mass": opts.get("virtual_mass"), "vm_lambda": opts.get("vm_lambda"), "bubble_drag": opts.get("bubble_drag"), "q_switch_L": opts.get("fsw", 0.02) * qL, "q_switch_G": opts.get("fsw_G", opts.get("fsw", 0.02)) * qG, "switch_frac": opts.get("switch_frac"),
                               "relax_time": opts.get("relax"), "limit_alpha": opts.get("limit"), "conservative": opts.get("conservative"),
                               "limit_band": opts.get("band"), "flux_band": opts.get("flux_band"), "flux_eps": opts.get("flux_eps"),
                               "flux_limit": opts.get("flux_limit"), "vertical_blend": opts.get("vblend"), "art_visc": opts.get("art_visc"),
                               "low_void_regime": opts.get("low_void"), "high_void_regime": opts.get("high_void"), "high_void_alpha": opts.get("high_void_alpha"), "interfacial_pressure": opts.get("delta"),
                               "vertex_volume": opts.get("vol")}.items() if v is not None}
    return severe_slug_case(method=opts.get("method", "fv_upwind"), tf=tf, dt_out=dt_out, qL=qL, qG=qG, D=D, P_out_bar=P_out, L_h1=opts.get("L_h1", 20.0),
                            L_down=opts.get("L_down", 300.0), slope_down_deg=opts.get("slope", -3.0), H_riser=opts.get("H", 20.0), L_h2=opts.get("L_h2", 5.0),
                            n_h1=opts.get("n_h1", 2), n_down=opts.get("n_down", 30), n_riser=opts.get("n_riser", 10), n_h2=opts.get("n_h2", 2), regime=opts.get("regime", 4), single_pipe=not opts.get("network", 0),
                            fluid=fluid, roughness=opts.get("roughness", 1e-5), start=start, t_ramp=opts.get("t_ramp", 100.0),
                            pocket_alpha=opts.get("pocket_alpha"), front_liquid_length=opts.get("front", 0.0), dP_margin_bar=opts.get("dP_margin", 0.3),
                            elbow_radius=opts.get("elbow_R", 0.0), n_elbow=opts.get("n_elbow", 0),
                            grading_down=opts.get("g_down", 1.0), grading_riser=opts.get("g_riser", 1.0), model=opts.get("model", 1), upstream_volume=opts.get("V_up", 0.0),
                            extra=extra)


def meta_from_severe_opts(opts, info=None) -> dict:
    """`meta` v1 de uma corrida a partir das opções do driver (era scratch/save_run_data.py)."""
    info = info or {}
    return dict(parcial=bool(opts.get("_parcial", False)) or (info.get("success") is False), gammaG=opts.get("gammaG", 1.05), gammaL=opts.get("gammaL", 1000.0), L_h1=opts.get("L_h1", 20.0), n_h1=opts.get("n_h1", 2), vSL=opts.get("vSL", 0.3), vSG=opts.get("vSG", 0.1), L_down=opts.get("L_down", 300.0), P_out=opts.get("P_out", 4.0), H_riser=opts.get("H", 20.0), slope=opts.get("slope", -3.0), tf=opts.get("tf", 4000.0),
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
