"""Opções dos submodelos como dataclasses congeladas (hashable: estáticas num jit) com
conversão de e para a forma legada (`Pipe.model`, `Pipe.regime`, `Pipe.extra`,
`SimulatorSettings`) e de e para o dicionário JSON (`mfsim-case/1`).

A conversão legado -> spec -> legado é a identidade nos campos usados, o que garante que o
`NetworkModel` atual constrói exatamente os mesmos objetos (teste de round-trip).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace, asdict
from typing import Optional

REGIME_NAMES = {
    1: "annular",
    2: "bubble",
    3: "intermittent",
    4: "stratified_smooth",
    5: "stratified_wavy",
    6: "automatic",
}
REGIME_IDS = {v: k for k, v in REGIME_NAMES.items()}
MODEL_NAMES = {1: "two_fluid", 2: "drift_flux_bb", 3: "single_liquid", 4: "single_gas"}
MODEL_IDS = {v: k for k, v in MODEL_NAMES.items()}
BUBBLE_DRAG_NAMES = {"tese": "thesis", "fisico": "physical"}
BUBBLE_DRAG_IDS = {v: k for k, v in BUBBLE_DRAG_NAMES.items()}

# chaves de Pipe.extra que o NetworkModel passa a PipeParams (lista branca de network/model.py)
PIPE_PARAM_KEYS = (
    "k_switch",
    "ipe_smooth",
    "ipe_local",
    "alpha_min",
    "alpha_max",
    "legacy_convective",
    "regime_vertical",
    "vertical_angle",
    "vertical_blend",
    "interfacial_pressure",
    "ip_slip_max",
    "low_void_regime",
    "low_void_alpha",
    "high_void_regime",
    "high_void_alpha",
    "bb_derivatives",
    "fi_mult",
    "fi_vgt",
    "virtual_mass",
    "vm_lambda",
    "bubble_drag",
)
# chaves de Pipe.extra que vão para as definições do esquema (idem)
SCHEME_KEYS = (
    "alpha_face",
    "pg_alpha",
    "q_switch",
    "q_switch_L",
    "q_switch_G",
    "switch_frac",
    "alpha_band",
    "relax_time",
    "limit_alpha",
    "limit_band",
    "conservative",
    "flux_band",
    "flux_eps",
    "flux_limit",
    "art_visc",
    "relax_target",
    "drift_relax",
    "drift_den_min",
    "ic_holdup",
)


def _regime_name(code):
    if code is None:
        return None
    code = int(code)
    return REGIME_NAMES[code] if code else None


def _regime_id(name):
    return 0 if name is None else REGIME_IDS[name]


@dataclass(frozen=True)
class RegimeSpec:
    """Política de padrão de escoamento: fixo (com misturas por ângulo e por fração) ou
    identificação automática (regime 6 da tese)."""

    type: str = "fixed"  # "fixed" | "automatic"
    regime: Optional[str] = "stratified_smooth"  # padrão fixo
    vertical_regime: Optional[str] = None  # padrão acima de vertical_angle (None = o mesmo)
    vertical_angle: float = 45.0  # graus
    vertical_blend: float = 0.0  # graus de transição suave (0 = chave dura)
    low_void_regime: Optional[str] = None
    low_void_alpha: float = 0.01
    high_void_regime: Optional[str] = None
    high_void_alpha: float = 0.9
    k_switch: float = 1e6  # automático: suavidade do chaveamento
    smooth: bool = False  # automático: ipe_smooth
    local: bool = False  # automático: ipe_local

    def to_json(self) -> dict:
        d = {"type": self.type}
        if self.type == "fixed":
            d["regime"] = self.regime
        else:
            d.update(k_switch=self.k_switch, smooth=self.smooth, local=self.local)
        if self.vertical_regime:
            d["vertical"] = {
                "regime": self.vertical_regime,
                "angle_deg": self.vertical_angle,
                "blend_deg": self.vertical_blend,
            }
        if self.low_void_regime:
            d["low_void"] = {"regime": self.low_void_regime, "alpha": self.low_void_alpha}
        if self.high_void_regime:
            d["high_void"] = {"regime": self.high_void_regime, "alpha": self.high_void_alpha}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "RegimeSpec":
        v = d.get("vertical") or {}
        lv = d.get("low_void") or {}
        hv = d.get("high_void") or {}
        return cls(
            type=d.get("type", "fixed"),
            regime=d.get("regime", "stratified_smooth") if d.get("type", "fixed") == "fixed" else None,
            vertical_regime=v.get("regime"),
            vertical_angle=float(v.get("angle_deg", 45.0)),
            vertical_blend=float(v.get("blend_deg", 0.0)),
            low_void_regime=lv.get("regime"),
            low_void_alpha=float(lv.get("alpha", 0.01)),
            high_void_regime=hv.get("regime"),
            high_void_alpha=float(hv.get("alpha", 0.9)),
            k_switch=float(d.get("k_switch", 1e6)),
            smooth=bool(d.get("smooth", False)),
            local=bool(d.get("local", False)),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Modelo de escoamento de um duto e os seus submodelos (fechamentos, política de
    padrão, opções). `extra` guarda chaves legadas não reconhecidas (passagem direta)."""

    type: str = "two_fluid"
    regime: RegimeSpec = field(default_factory=RegimeSpec)
    # fechamentos
    fi_mult: float = 1.0  # estratificado: multiplicador do atrito interfacial
    fi_vgt: float = 5.5  # estratificado: v_GT de Andritsos & Hanratty (inf = Taitel & Dukler)
    bubble_drag: str = "thesis"  # bolhas: "thesis" | "physical"
    # opções do modelo
    alpha_min: float = 1e-3
    alpha_max: float = 1.0 - 1e-3
    legacy_convective: bool = True
    interfacial_pressure: float = 0.0
    ip_slip_max: float = 10.0
    virtual_mass: float = 0.0
    vm_lambda: float = 1.0
    derivatives: str = "exact"  # deriva (Beggs & Brill): "exact" (JAX) | "fd"
    extra: tuple = ()  # pares (chave, valor) legados sem equivalente

    # --- legado ------------------------------------------------------------
    @classmethod
    def from_legacy(cls, model_id: int, regime_id: int, extra: dict) -> "ModelSpec":
        e = dict(extra or {})
        reg = RegimeSpec(
            type="automatic" if int(regime_id) == 6 else "fixed",
            regime=None if int(regime_id) == 6 else REGIME_NAMES[int(regime_id)],
            vertical_regime=_regime_name(e.get("regime_vertical", 0)),
            vertical_angle=float(e.get("vertical_angle", 45.0)),
            vertical_blend=float(e.get("vertical_blend", 0.0)),
            low_void_regime=_regime_name(e.get("low_void_regime", 0)),
            low_void_alpha=float(e.get("low_void_alpha", 0.01)),
            high_void_regime=_regime_name(e.get("high_void_regime", 0)),
            high_void_alpha=float(e.get("high_void_alpha", 0.9)),
            k_switch=float(e.get("k_switch", 1e6)),
            smooth=bool(e.get("ipe_smooth", False)),
            local=bool(e.get("ipe_local", False)),
        )
        known = set(PIPE_PARAM_KEYS) | set(SCHEME_KEYS) | {k for k in e if k.startswith("fv_")}
        rest = tuple(sorted((k, v) for k, v in e.items() if k not in known))
        return cls(
            type=MODEL_NAMES[int(model_id)],
            regime=reg,
            fi_mult=float(e.get("fi_mult", 1.0)),
            fi_vgt=float(e.get("fi_vgt", 5.5)),
            bubble_drag=BUBBLE_DRAG_NAMES[e.get("bubble_drag", "tese")],
            alpha_min=float(e.get("alpha_min", 1e-3)),
            alpha_max=float(e.get("alpha_max", 1.0 - 1e-3)),
            legacy_convective=bool(e.get("legacy_convective", True)),
            interfacial_pressure=float(e.get("interfacial_pressure", 0.0)),
            ip_slip_max=float(e.get("ip_slip_max", 10.0)),
            virtual_mass=float(e.get("virtual_mass", 0.0)),
            vm_lambda=float(e.get("vm_lambda", 1.0)),
            derivatives=e.get("bb_derivatives", "exact"),
            extra=rest,
        )

    def to_legacy(self):
        """(model_id, regime_id, extra) com as mesmas chaves/valores que o caso legado tinha."""
        r = self.regime
        e = {
            "k_switch": r.k_switch,
            "ipe_smooth": r.smooth,
            "ipe_local": r.local,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "legacy_convective": self.legacy_convective,
            "regime_vertical": _regime_id(r.vertical_regime),
            "vertical_angle": r.vertical_angle,
            "vertical_blend": r.vertical_blend,
            "interfacial_pressure": self.interfacial_pressure,
            "ip_slip_max": self.ip_slip_max,
            "low_void_regime": _regime_id(r.low_void_regime),
            "low_void_alpha": r.low_void_alpha,
            "high_void_regime": _regime_id(r.high_void_regime),
            "high_void_alpha": r.high_void_alpha,
            "bb_derivatives": self.derivatives,
            "fi_mult": self.fi_mult,
            "fi_vgt": self.fi_vgt,
            "virtual_mass": self.virtual_mass,
            "vm_lambda": self.vm_lambda,
            "bubble_drag": BUBBLE_DRAG_IDS[self.bubble_drag],
        }
        e.update(dict(self.extra))
        regime_id = 6 if r.type == "automatic" else REGIME_IDS[r.regime]
        return MODEL_IDS[self.type], regime_id, e

    # --- JSON --------------------------------------------------------------
    def to_json(self) -> dict:
        d = {
            "type": self.type,
            "regime": self.regime.to_json(),
            "closures": {
                "stratified": {"fi_mult": self.fi_mult, "v_gt": self.fi_vgt},
                "bubble": {"drag": self.bubble_drag},
            },
            "options": {
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "legacy_convective": self.legacy_convective,
                "interfacial_pressure": self.interfacial_pressure,
                "ip_slip_max": self.ip_slip_max,
                "virtual_mass": {"C_vm": self.virtual_mass, "lambda": self.vm_lambda},
                "derivatives": self.derivatives,
            },
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ModelSpec":
        c = d.get("closures") or {}
        st = c.get("stratified") or {}
        bu = c.get("bubble") or {}
        o = d.get("options") or {}
        vm = o.get("virtual_mass") or {}
        if isinstance(vm, (int, float)):
            vm = {"C_vm": vm}
        return cls(
            type=d.get("type", "two_fluid"),
            regime=RegimeSpec.from_json(d.get("regime") or {}),
            fi_mult=float(st.get("fi_mult", 1.0)),
            fi_vgt=float(st.get("v_gt", 5.5)),
            bubble_drag=bu.get("drag", "thesis"),
            alpha_min=float(o.get("alpha_min", 1e-3)),
            alpha_max=float(o.get("alpha_max", 1.0 - 1e-3)),
            legacy_convective=bool(o.get("legacy_convective", True)),
            interfacial_pressure=float(o.get("interfacial_pressure", 0.0)),
            ip_slip_max=float(o.get("ip_slip_max", 10.0)),
            virtual_mass=float(vm.get("C_vm", 0.0)),
            vm_lambda=float(vm.get("lambda", 1.0)),
            derivatives=o.get("derivatives", "exact"),
            extra=tuple(sorted((d.get("extra") or {}).items())),
        )


@dataclass(frozen=True)
class SchemeSpec:
    """Discretização espacial de um duto: tipo (galerkin, petrov_galerkin, fd_central,
    fd_upwind, fv_upwind, fv_upwind_cons) e opções (as chaves do esquema em Pipe.extra)."""

    type: str = "galerkin"
    options: tuple = ()  # pares (chave, valor) ordenados

    @classmethod
    def from_legacy(cls, method: str, extra: dict) -> "SchemeSpec":
        e = extra or {}
        opts = {k: v for k, v in e.items() if k in SCHEME_KEYS or k.startswith("fv_")}
        return cls(type=method, options=tuple(sorted(opts.items())))

    def to_legacy_extra(self) -> dict:
        return dict(self.options)

    def to_json(self) -> dict:
        return {"type": self.type, "options": dict(self.options)}

    @classmethod
    def from_json(cls, d: dict) -> "SchemeSpec":
        return cls(type=d.get("type", "galerkin"), options=tuple(sorted((d.get("options") or {}).items())))


@dataclass(frozen=True)
class TimeSpec:
    t0: float = 0.0
    tf: float = 200.0
    dt_out: float = 2.0

    def to_json(self):
        return {"t0": self.t0, "tf": self.tf, "dt_out": self.dt_out}

    @classmethod
    def from_json(cls, d):
        return cls(float(d.get("t0", 0.0)), float(d.get("tf", 200.0)), float(d.get("dt_out", 2.0)))


@dataclass(frozen=True)
class ReferenceSpec:
    """Pressão de referência PR (PT = P/PR) e faixas de referência dos estados (state_ref)."""

    PR: float = 100e5
    state_ref: tuple = ()  # pares (PT|qL|qG, valor)

    def to_json(self):
        d = {"PR": self.PR}
        if self.state_ref:
            d["state_ref"] = dict(self.state_ref)
        return d

    @classmethod
    def from_json(cls, d):
        return cls(float(d.get("PR", 100e5)), tuple(sorted((d.get("state_ref") or {}).items())))


@dataclass(frozen=True)
class IntegrationSpec:
    """Integrador e tolerâncias (SimulatorSettings + opções do ImplicitBDF usadas pelos drivers)."""

    integrator: str = "bdf"  # bdf (scipy) | implicit | radau | LSODA
    method_quadrature: int = 1  # ordem de quadratura (Galerkin): 1->3, 2->5, 3->8, 4->12 pontos
    max_step: float = 1.0
    atol: float = 1e-3
    rtol: float = 1e-4
    backend: str = "numpy"  # numpy | jax
    tol_range: Optional[float] = None
    first_step: float = 1e-3
    max_order: int = 2
    max_newton: int = 12
    newton_tol: float = 1e-3
    slow_exit: bool = False
    galerkin_band: Optional[int] = None
    extra: tuple = ()

    def to_json(self):
        d = {
            "integrator": self.integrator,
            "quadrature_order": self.method_quadrature,
            "max_step": None if self.max_step == float("inf") else self.max_step,
            "atol": self.atol,
            "rtol": self.rtol,
            "backend": self.backend,
            "tol_range": self.tol_range,
            "first_step": self.first_step,
            "max_order": self.max_order,
            "max_newton": self.max_newton,
            "newton_tol": self.newton_tol,
            "slow_exit": self.slow_exit,
        }
        if self.galerkin_band is not None:
            d["galerkin_band"] = self.galerkin_band
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_json(cls, d):
        ms = d.get("max_step", 1.0)
        return cls(
            integrator=d.get("integrator", "bdf"),
            method_quadrature=int(d.get("quadrature_order", 1)),
            max_step=float("inf") if ms is None else float(ms),
            atol=float(d.get("atol", 1e-3)),
            rtol=float(d.get("rtol", 1e-4)),
            backend=d.get("backend", "numpy"),
            tol_range=d.get("tol_range"),
            first_step=float(d.get("first_step", 1e-3)),
            max_order=int(d.get("max_order", 2)),
            max_newton=int(d.get("max_newton", 12)),
            newton_tol=float(d.get("newton_tol", 1e-3)),
            slow_exit=bool(d.get("slow_exit", False)),
            galerkin_band=d.get("galerkin_band"),
            extra=tuple(sorted((d.get("extra") or {}).items())),
        )
