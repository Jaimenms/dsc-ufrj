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


def integrate_run(case, opts=None, verbose=False, t_eval=None, start=None):
    """Integra um caso como o driver `scratch/severe_D.py`: estado inicial da partida
    (bloqueada/linha fechada) ou estimativa, tolerância absoluta por faixa
    (`tol_range`), Jacobiana exata em JAX (`jax`), acumulador dos fluxos de contorno.

    opts: rtol (1e-3), tol_range, atol_P, atol_q, order (2), max_newton (12),
    newton_tol (1e-3), slow (False), predict_space, jt_refresh, jax (False),
    tightbounds (False). Devolve (model, T, X, info)."""
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
        from .jaxrhs import build_jax_rhs
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
    T, X, info = bdf.integrate(t_eval, X0, first_step=1e-3, breakpoints=model.spec_breakpoints())
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

    @property
    def case(self):
        if self._case is None:
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


def load_run(path) -> Run:
    """Lê um npz gravado (`notebooks/data/<nome>.npz`, ou o nome sem extensão)."""
    if not os.path.exists(path) and not str(path).endswith(".npz"):
        path = os.path.join("notebooks", "data", f"{path}.npz")
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        info = json.loads(str(d["info_json"])) if "info_json" in d.files else {}
        prov = json.loads(str(d["provenance_json"])) if "provenance_json" in d.files else {}
        return Run(path=str(path), T=np.asarray(d["T"], float), X=np.asarray(d["X"], float), meta=meta,
                   acc=np.asarray(d["acc"]) if "acc" in d.files else None, info=info, provenance=prov)


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


def save_run(path, T, X, meta, acc=None, info=None, command=None):
    """Grava uma corrida no formato v1 (T, X, acc, meta JSON) com `info_json` e
    `provenance_json`. `info` guarda n_steps, n_rejected, cpu, n_rhs, n_jac."""
    extra = {}
    if acc is not None:
        extra["acc"] = np.asarray(acc)
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in (info or {}).items() if k in ("n_steps", "n_rejected", "cpu", "n_rhs", "n_jac", "success")}
    np.savez(path, T=np.asarray(T, float), X=np.asarray(X, float), meta=json.dumps(meta),
             info_json=json.dumps(info), provenance_json=json.dumps(provenance(command)), **extra)
    return path
