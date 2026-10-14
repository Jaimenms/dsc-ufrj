"""Lado direito da rede compilado (JAX) com Jacobiana por diferenciação automática.

`build_jax_rhs(model)` devolve `(rhs, jac)`:

    rhs(t, x) -> dx/dt   (numpy, igual a NetworkModel.rhs a precisão de máquina)
    jac(t, x) -> d(rhs)/dx  (numpy denso, exato)

Cobre os modelos de duto (dois fluidos com padrão fixo e misturas, deriva de
Beggs & Brill, monofásicos), os esquemas nodais (diferenças centrais e
regressivas, Galerkin, Petrov-Galerkin) e o escalonado de dois fluidos, vértices
de contorno e dinâmicos, acoplamentos, chokes, reservatórios (IPR), válvulas de
retenção, filtro anti-refluxo e tabelas no tempo (avaliadas em numpy a cada
instante e passadas como argumentos). A identificação automática de padrão do
modelo a dois fluidos (regime 6), o esquema escalonado de ordem 2 (MUSCL) e o
escalonado monofásico são cobertos; o que não é (volumes finitos para o modelo
de deriva) levanta NotImplementedError na construção.
"""

from __future__ import annotations

import numpy as np

try:
    import os
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    # cache persistente de compilação: a compilação (1-5 s por modelo) é feita uma vez
    # por máquina e reaproveitada entre processos/notebooks
    _cache_dir = os.environ.get("MFSIM_JAX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "mfsim_jax"))
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:                   # pragma: no cover
        pass
    HAVE_JAX = True
except ImportError:                     # pragma: no cover
    HAVE_JAX = False

from .jax_physics import check_valve_flows, fluid_props, reservoir_flows, valve_upstream_pressure
from .jax_schemes import make_scheme_j
from .models.base import SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
from .network.data import BC_DIRICHLET, BC_NEUMANN


def _make_assemble(pb):
    """PipeBlock.assemble + scheme.fill_derived em JAX (estrutura estática)."""
    sch = pb.scheme
    n_nodes = pb.n_nodes
    free = jnp.asarray(pb.free)
    base_Y = jnp.asarray(pb.base_Y())
    bc_in, bc_out = tuple(int(b) for b in pb.bc_in), tuple(int(b) for b in pb.bc_out)
    staggered = sch.name == "fv_upwind"
    N = sch.N
    phases = [s for s, _ in sch.phases] if staggered else []
    cell_slots = list(getattr(sch, "cell_slots", (SLOT_ALPHA, SLOT_P))) if staggered else []

    def assemble(xseg, E_in, E_out):
        Y = base_Y.ravel().at[free].set(xseg).reshape(n_nodes, 4)
        for s in range(4):
            if bc_in[s] == BC_DIRICHLET:
                Y = Y.at[0, s].set(E_in[s])
            if bc_out[s] == BC_DIRICHLET:
                Y = Y.at[-1, s].set(E_out[s])
        for s in range(4):
            if bc_in[s] == BC_NEUMANN:
                Y = Y.at[0, s].set(Y[1, s])
            if bc_out[s] == BC_NEUMANN:
                Y = Y.at[-1, s].set(Y[-2, s])
        if staggered:
            for s in cell_slots:
                if bc_in[s] != BC_DIRICHLET:
                    Y = Y.at[0, s].set(Y[1, s])
                if bc_out[s] != BC_DIRICHLET:
                    Y = Y.at[N + 1, s].set(Y[N, s])
            for s in phases:
                if bc_out[s] == BC_DIRICHLET:
                    Y = Y.at[N, s].set(Y[N + 1, s])
                elif bc_out[s] == BC_NEUMANN:
                    Y = Y.at[N, s].set(Y[N - 1, s])
                    Y = Y.at[N + 1, s].set(Y[N, s])
                else:
                    Y = Y.at[N + 1, s].set(Y[N, s])
                if bc_in[s] == BC_NEUMANN:
                    Y = Y.at[0, s].set(Y[1, s])
        return Y

    return assemble


def build_jax_rhs(model):
    """Constrói (rhs, jac) compilados para um NetworkModel."""
    if not HAVE_JAX:
        raise ImportError("jax não está instalado")
    fl = model.fluid
    PR = float(model.PR)
    n_states = model.n_states
    pipes = []
    for pb in model.pipes:
        pipes.append(dict(pb=pb, assemble=_make_assemble(pb), rhs=make_scheme_j(pb),
                          free=jnp.asarray(pb.free), sl=slice(pb.offset, pb.offset + pb.n_states),
                          vin=model.vpos[pb.pipe.v_in], vout=model.vpos[pb.pipe.v_out]))
    specified = np.array([[tb.is_specified for tb in row] for row in model.tables], dtype=bool)
    vA = list(model.vA)
    vA_index = {vi: model.vertex_A_index(model.vertices[vi].index) for vi in vA}
    couplings = [(vu, vd, [model.pipes[k].state_index(model.pipes[k].inlet_P_node, SLOT_P) for k in model.pipes_out[vd]])
                 for vu, vd in model.couplings]
    valves = []
    for j, (vi, v, k, op, mo, sp) in enumerate(model.valves):
        pb = model.pipes[k]
        valves.append(dict(vi=vi, CV=float(v.CV), iL=pb.state_index(pb.outlet_q_node, SLOT_QL),
                           iG=pb.state_index(pb.outlet_q_node, SLOT_QG), j=j))
    reservoirs = []
    for vi, r, k in model.reservoirs:
        pb = model.pipes[k]
        reservoirs.append(dict(vi=vi, iP=pb.state_index(pb.inlet_P_node, SLOT_P), IPL=float(r.IPL), IPG=float(r.IPG), P=float(r.P)))
    checks = [dict(vi=vi, iH=vA_index[vi][0], iP=vA_index[vi][1], CV=float(c.CV), dP_min=float(c.dP_min))
              for vi, c in model.check_valves]
    coupling_flows = {}
    for vu, vd in model.couplings:
        lst = coupling_flows.setdefault(vd, [])
        for k in model.pipes_in[vu]:
            pb = model.pipes[k]
            lst.append((pb.state_index(pb.outlet_q_node, SLOT_QL), pb.state_index(pb.outlet_q_node, SLOT_QG)))
    a_min, a_max, band = model.alpha_limits
    vert_in = {vi: [(k, model.pipes[k].outlet_q_node) for k in model.pipes_in[vi]] for vi in vA}
    vert_out = {vi: [(k, model.pipes[k].inlet_q_node) for k in model.pipes_out[vi]] for vi in vA}
    # fluxos de contorno de cada duto (limitados nos dutos fv_upwind conservativos)
    from .jax_schemes import make_boundary_fluxes_j
    bflux = [make_boundary_fluxes_j(pb) for pb in model.pipes]
    volumes = {vi: float(model.vertices[vi].volume) for vi in vA}
    vert_conservative = bool(vA) and bool(getattr(model, "conservative", False))

    def rhs_core(x, Et, opt):
        E = Et
        for vi in vA:
            for s in (SLOT_QL, SLOT_QG):
                if not specified[vi, s]:
                    E = E.at[vi, s].set(0.0)
        for vu, vd, idxs in couplings:
            for i in idxs:
                if i is not None:
                    E = E.at[vu, SLOT_P].set(x[i])
        for vv in valves:
            qL = x[vv["iL"]] if vv["iL"] is not None else 0.0
            qG = x[vv["iG"]] if vv["iG"] is not None else 0.0
            ZP = valve_upstream_pressure(E[vv["vi"], SLOT_P], qL, qG, opt[vv["j"]], vv["CV"], fl, PR)
            E = E.at[vv["vi"], SLOT_P].set(ZP)
        for rr in reservoirs:
            P = x[rr["iP"]] if rr["iP"] is not None else E[rr["vi"], SLOT_P]
            ZL, ZG = reservoir_flows(P, rr["IPL"], rr["IPG"], rr["P"], PR)
            E = E.at[rr["vi"], SLOT_QL].set(ZL).at[rr["vi"], SLOT_QG].set(ZG)
        for cc in checks:
            qL, qG = check_valve_flows(x[cc["iP"]], E[cc["vi"], SLOT_P], x[cc["iH"]], cc["CV"], cc["dP_min"], fl, PR)
            E = E.at[cc["vi"], SLOT_QL].set(-qL).at[cc["vi"], SLOT_QG].set(-qG)
        for vd, lst in coupling_flows.items():
            qL = sum((x[i] for i, _ in lst if i is not None), 0.0)
            qG = sum((x[i] for _, i in lst if i is not None), 0.0)
            E = E.at[vd, SLOT_QL].set(qL).at[vd, SLOT_QG].set(qG)
        for vi in vA:
            iH, iP = vA_index[vi]
            E = E.at[vi, SLOT_ALPHA].set(x[iH]).at[vi, SLOT_P].set(x[iP])
        Ys = []
        parts = []
        for pp in pipes:
            Y = pp["assemble"](x[pp["sl"]], E[pp["vin"]], E[pp["vout"]])
            Ys.append(Y)
            dY = pp["rhs"](Y)
            parts.append(dY.ravel()[pp["free"]])
        for vi in vA:
            iH, iP = vA_index[vi]
            QL = E[vi, SLOT_QL]
            QG = E[vi, SLOT_QG]
            for k, node in vert_in[vi]:
                (_, _), (fL, fG) = bflux[k](Ys[k])
                QL = QL + fL
                QG = QG + fG
            for k, node in vert_out[vi]:
                (fL, fG), (_, _) = bflux[k](Ys[k])
                QL = QL - fL
                QG = QG - fG
            HG = x[iH]
            PT = x[iP]
            if vert_conservative:
                vol = volumes[vi]
                parts.append(jnp.stack([QG / vol, QL / vol]))
                continue
            rhoL, rhoG, CL, CG = fluid_props(fl, jnp.maximum(PT * PR, 1e3))
            HL = 1.0 - HG
            CVL = HL / rhoL * CL
            CVG = HG / rhoG * CG
            CVT = CVL + CVG
            vol = volumes[vi]
            BVL = QL / vol / rhoL
            BVG = QG / vol / rhoG
            BVT = BVL + BVG
            dPT = BVT / CVT / PR
            dHG = BVG - CVG * BVT / CVT
            r_lo = jnp.clip((HG - a_min) / band, 0.0, 1.0)
            r_hi = jnp.clip((a_max - HG) / band, 0.0, 1.0)
            dHG = jnp.where(dHG < 0, dHG * r_lo, dHG * r_hi)
            parts.append(jnp.stack([dHG, dPT]))
        dX = jnp.concatenate(parts) if parts else jnp.zeros(n_states)
        return dX

    rhs_jit = jax.jit(rhs_core)
    jac_jit = jax.jit(jax.jacfwd(rhs_core, argnums=0))
    tables = model.tables
    valve_tables = [op for (_, _, _, op, _, _) in model.valves]
    const = all(tb.breakpoints.size == 0 for row in tables for tb in row) and all(op.breakpoints.size == 0 for op in valve_tables)
    cache = {}

    def _inputs(t):
        if const and "E" in cache:
            return cache["E"], cache["opt"]
        key = float(t)
        if cache.get("t") == key:
            return cache["E"], cache["opt"]
        E = np.array([[tb(t) for tb in row] for row in tables], dtype=float)
        opt = np.array([op(t) for op in valve_tables], dtype=float) if valve_tables else np.zeros(0)
        cache.update(t=key, E=E, opt=opt)
        return E, opt

    def rhs(t, x):
        E, opt = _inputs(t)
        return np.asarray(rhs_jit(x, E, opt))

    def jac(t, x):
        E, opt = _inputs(t)
        return np.asarray(jac_jit(x, E, opt))

    rhs.jit = rhs_jit
    rhs.inputs = _inputs
    return rhs, jac


def get_jax(model):
    """(rhs, jac) JAX do modelo, construídos uma vez e guardados no próprio modelo."""
    cached = getattr(model, "_jax_functions", None)
    if cached is None:
        cached = build_jax_rhs(model)
        model._jax_functions = cached
    return cached
