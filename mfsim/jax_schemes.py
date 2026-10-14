"""Discretizações em JAX: esquemas nodais (diferenças centrais e regressivas,
Galerkin, Petrov-Galerkin) e volumes finitos escalonados (dois fluidos, modos
primitivo e conservativo, ordens 1 e 2; monofásicos). `make_scheme_j(pb)` devolve `rhs_pipe(Y) -> dY`
(n_nodes x 4) equivalente a `pb.scheme.rhs(Y)`.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from .jax_physics import make_model_j, quasi_linear
from .models.base import G, SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


# ----------------------------------------------------------------------------
# esquemas nodais
# ----------------------------------------------------------------------------
def _make_expand(pb, mj):
    slots = list(mj.slots)
    n_nodes = pb.n_nodes
    limit = 0 in slots
    p = pb.model.params if hasattr(pb.model, "params") else None
    band = float(pb.scheme.settings.get("alpha_band", 0.02))

    def expand(dys, Y):
        dY = jnp.zeros((n_nodes, 4)).at[:, slots].set(dys)
        if limit:
            aG = Y[:, 0]
            r_lo = jnp.clip((aG - p.alpha_min) / band, 0.0, 1.0)
            r_hi = jnp.clip((p.alpha_max - aG) / band, 0.0, 1.0)
            d = dY[:, 0]
            dY = dY.at[:, 0].set(jnp.where(d < 0, d * r_lo, d * r_hi))
        return dY

    return expand


def make_fd_central(pb):
    mj = make_model_j(pb.model)
    slots = list(mj.slots)
    theta = jnp.asarray(pb.scheme.theta_nodes)
    h = jnp.asarray(pb.scheme.h)
    expand = _make_expand(pb, mj)

    def rhs(Y):
        A, B = quasi_linear(mj, Y, theta)
        ys = Y[:, slots]
        first = (ys[1] - ys[0]) / h[0]
        last = (ys[-1] - ys[-2]) / h[-1]
        mid = (ys[2:] - ys[:-2]) / (h[:-1] + h[1:])[:, None]
        dydx = jnp.concatenate([first[None, :], mid, last[None, :]], axis=0)
        dys = -(jnp.einsum("nij,nj->ni", A, dydx) + B)
        return expand(dys, Y)

    return rhs


def make_fd_upwind(pb):
    mj = make_model_j(pb.model)
    slots = list(mj.slots)
    theta = jnp.asarray(pb.scheme.theta_nodes)
    h = jnp.asarray(pb.scheme.h)
    expand = _make_expand(pb, mj)

    def rhs(Y):
        A, B = quasi_linear(mj, Y, theta)
        ys = Y[:, slots]
        first = (ys[1] - ys[0]) / h[0]
        rest = (ys[1:] - ys[:-1]) / h[:, None]
        dydx = jnp.concatenate([first[None, :], rest], axis=0)
        dys = -(jnp.einsum("nij,nj->ni", A, dydx) + B)
        return expand(dys, Y)

    return rhs


def make_galerkin(pb, petrov=False):
    sch = pb.scheme
    mj = make_model_j(pb.model)
    slots = list(mj.slots)
    nv = mj.nv
    N = sch.N
    xi = jnp.asarray(sch.xi)
    wq = jnp.asarray(sch.wq)
    nq = sch.xi.size
    phi_l = jnp.asarray(sch._phi_left)
    phi_r = jnp.asarray(sch._phi_right)
    Minv = jnp.asarray(sch.Minv)
    h = jnp.asarray(sch.h)
    thq = jnp.asarray(np.repeat(sch.theta_elem, nq))
    expand = _make_expand(pb, mj)
    a2 = (sch.alpha / 2.0) if petrov else 0.0

    def rhs(Y):
        yl = Y[:-1][:, None, :]
        yr = Y[1:][:, None, :]
        Yq = (yl + (yr - yl) * xi[None, :, None]).reshape(-1, 4)
        A, B = quasi_linear(mj, Yq, thq)
        A = A.reshape(N, nq, nv, nv)
        B = B.reshape(N, nq, nv)
        ys = Y[:, slots]
        dy = ys[1:] - ys[:-1]
        if not petrov:
            Ady = jnp.einsum("eqij,ej->eqi", A, dy)
            integrand = Ady + h[:, None, None] * B
            R_left = jnp.einsum("q,eqi->ei", wq * phi_l, integrand)
            R_right = jnp.einsum("q,eqi->ei", wq * phi_r, integrand)
        else:
            sA = jnp.sign(A)
            wl = wq[None, :, None, None] * (phi_l[None, :, None, None] - a2 * sA)
            wr = wq[None, :, None, None] * (phi_r[None, :, None, None] + a2 * sA)
            Al = jnp.einsum("eqij,eqij->eij", wl, A)
            Ar = jnp.einsum("eqij,eqij->eij", wr, A)
            sB = jnp.sign(B)
            wlB = wq[None, :, None] * (phi_l[None, :, None] - a2 * sB)
            wrB = wq[None, :, None] * (phi_r[None, :, None] + a2 * sB)
            Bl = jnp.einsum("eqi,eqi->ei", wlB, B) * h[:, None]
            Br = jnp.einsum("eqi,eqi->ei", wrB, B) * h[:, None]
            R_left = jnp.einsum("eij,ej->ei", Al, dy) + Bl
            R_right = jnp.einsum("eij,ej->ei", Ar, dy) + Br
        R = jnp.zeros((N + 1, nv)).at[:-1].add(R_left).at[1:].add(R_right)
        dys = -(Minv @ R)
        return expand(dys, Y)

    return rhs


# ----------------------------------------------------------------------------
# volumes finitos escalonados (dois fluidos)
# ----------------------------------------------------------------------------
def make_staggered_single(pb):
    """Volumes finitos escalonados para os modelos monofásicos (StaggeredFVScheme._rhs_single):
    pressão nas células, vazão nas faces, momento com velocidade doadora (ou MUSCL)."""
    sch = pb.scheme
    mj = make_model_j(pb.model)
    N = sch.N
    h = jnp.asarray(sch.h)
    dxf = jnp.asarray(sch.dx_faces)
    th_nodes = jnp.asarray(sch.theta_nodes)
    th_faces = jnp.asarray(sch.theta_faces)
    sin_f = jnp.sin(th_faces)
    Area = float(np.pi * pb.model.params.D**2 / 4.0)
    PR = pb.model.params.PR
    slot_q = sch.phases[0][0]
    q_floor, switch_frac, order = float(sch.q_switch), float(sch.switch_frac), int(sch.order)

    def switch(q):
        aq = jnp.abs(q)
        local = aq
        local = local.at[1:].set(jnp.maximum(local[1:], aq[:-1]))
        local = local.at[:-1].set(jnp.maximum(local[:-1], aq[1:]))
        return 0.5 * (1.0 + jnp.tanh(q / (switch_frac * local + q_floor)))

    def rhs(Y):
        q = Y[0:N + 1, slot_q]
        Yc = Y[0:N + 2].at[0, slot_q].set(q[0]).at[N + 1, slot_q].set(q[N])
        Ycc = Yc.at[1:N + 1, slot_q].set(0.5 * (q[:-1] + q[1:]))
        tc = mj.conservative_terms(Ycc, th_nodes)
        Yf = jnp.stack([0.5 * (Yc[0:N + 1, 0] + Yc[1:N + 2, 0]), 0.5 * (Yc[0:N + 1, 1] + Yc[1:N + 2, 1]),
                        Y[0:N + 1, 2], Y[0:N + 1, 3]], axis=1)
        tf = mj.conservative_terms(Yf, th_faces)
        dPT = -(1.0 / Area / tc["C"][1:N + 1] / PR) * (q[1:] - q[:-1]) / h
        vf = q / (tf["rho"] * Area)
        qc = jnp.concatenate([q[:1], 0.5 * (q[:-1] + q[1:]), q[-1:]])
        w = switch(qc[1:N + 1])
        if order == 2:
            dv = vf[1:] - vf[:-1]
            slope = jnp.concatenate([jnp.zeros(1), _minmod(dv[:-1], dv[1:]), jnp.zeros(1)])
            vmid = w * (vf[:-1] + 0.5 * slope[:-1]) + (1.0 - w) * (vf[1:] - 0.5 * slope[1:])
        else:
            vmid = w * vf[:-1] + (1.0 - w) * vf[1:]
        vdon = jnp.concatenate([vf[:1], vmid, vf[-1:]])
        mom_c = qc * vdon
        Pc = tc["P"]
        dq = -(mom_c[1:] - mom_c[:-1]) / dxf - Area * (Pc[1:] - Pc[:-1]) / dxf - tf["Gamma"] - G * tf["rho"] * Area * sin_f
        dY = jnp.zeros((N + 2, 4))
        return dY.at[1:N + 1, SLOT_P].set(dPT).at[0:N + 1, slot_q].set(dq)

    return rhs


def _staggered_common(pb):
    """Funções e constantes partilhadas pelos esquemas escalonados bifásicos (dois fluidos
    e deriva): chaveamento doador, limitador de fluxo da célula doadora e momento por fase."""
    sch = pb.scheme
    N = sch.N
    dxf = jnp.asarray(sch.dx_faces)
    sin_f = jnp.sin(jnp.asarray(sch.theta_faces))
    Area = float(np.pi * pb.model.params.D**2 / 4.0)
    q_floor = {"L": float(sch.q_switch_L), "G": float(sch.q_switch_G), None: float(sch.q_switch)}
    switch_frac = float(sch.switch_frac)
    flux_band, flux_eps, flux_limit = float(sch.flux_band), float(sch.flux_eps), bool(sch.flux_limit)
    art_visc, order = float(sch.art_visc), int(sch.order)
    from .network.data import BC_DIRICHLET as _BCD
    ghost_in_free = sch.bc_in[SLOT_ALPHA] != _BCD
    ghost_out_free = sch.bc_out[SLOT_ALPHA] != _BCD

    def switch(q, phase, frac=None, floor_scale=1.0):
        aq = jnp.abs(q)
        local = aq
        local = local.at[1:].set(jnp.maximum(local[1:], aq[:-1]))
        local = local.at[:-1].set(jnp.maximum(local[:-1], aq[1:]))
        e = (switch_frac if frac is None else frac) * local + q_floor[phase] * floor_scale
        return 0.5 * (1.0 + jnp.tanh(q / e))

    def flux_limiter(aG_c, q, phase):
        if not flux_limit:
            return jnp.ones(N + 1)
        a = aG_c if phase == "G" else 1.0 - aG_c
        u = jnp.clip(a / flux_band, 0.0, 1.0)
        r = u * u * (3.0 - 2.0 * u)
        if ghost_in_free:
            r = r.at[0].set(1.0)
        if ghost_out_free:
            r = r.at[N + 1].set(1.0)
        s_ = 0.5 * (1.0 + jnp.tanh(q / (flux_eps * q_floor[phase])))
        return s_ * r[:-1] + (1.0 - s_) * r[1:]

    def mass_fluxes(Y):
        qL = Y[0:N + 1, SLOT_QL]
        qG = Y[0:N + 1, SLOT_QG]
        aG_raw_c = Y[0:N + 2, SLOT_ALPHA]
        return qL * flux_limiter(aG_raw_c, qL, "L"), qG * flux_limiter(aG_raw_c, qG, "G")

    def momentum(qf, rho_f, alpha_f, Pc, dPc, Gamma_f, phase, comp_c):
        vf = qf / (rho_f * alpha_f * Area)
        qc = jnp.concatenate([qf[:1], 0.5 * (qf[:-1] + qf[1:]), qf[-1:]])
        w = switch(qc[1:N + 1], phase)
        if order == 2:
            dv = vf[1:] - vf[:-1]
            sl = _minmod(dv[:-1], dv[1:])
            slope = jnp.concatenate([jnp.zeros(1), sl, jnp.zeros(1)])
            vmid = w * (vf[:-1] + 0.5 * slope[:-1]) + (1.0 - w) * (vf[1:] - 0.5 * slope[1:])
        else:
            vmid = w * vf[:-1] + (1.0 - w) * vf[1:]
        vdon = jnp.concatenate([vf[:1], vmid, vf[-1:]])
        mom_c = qc * vdon
        dq = (-(mom_c[1:] - mom_c[:-1]) / dxf - Area * alpha_f * (Pc[1:] - Pc[:-1]) / dxf
              - Area * (dPc[1:] - dPc[:-1]) / dxf - Gamma_f - G * rho_f * alpha_f * Area * sin_f)
        if art_visc > 0 and comp_c is not None:
            c_c = 1.0 / jnp.sqrt(jnp.maximum(comp_c[1:N + 1], 1e-300))
            F = art_visc * c_c * (qf[1:] - qf[:-1])
            dq = dq.at[1:N].add((F[1:] - F[:-1]) / dxf[1:N])
        return dq, (mom_c[1:] - mom_c[:-1]) / dxf

    return dict(switch=switch, flux_limiter=flux_limiter, mass_fluxes=mass_fluxes, momentum=momentum, Area=Area, N=N)


def make_staggered_drift(pb):
    """Modelo de deriva (Beggs & Brill) no escalonado: StaggeredFVScheme._rhs_drift em JAX."""
    sch = pb.scheme
    mj = make_model_j(pb.model)
    fl = pb.model.fluid
    N = sch.N
    h = jnp.asarray(sch.h)
    th_nodes = jnp.asarray(sch.theta_nodes)
    th_faces = jnp.asarray(sch.theta_faces)
    PR = pb.model.params.PR
    cm = _staggered_common(pb)
    switch, mass_fluxes, momentum, Area = cm["switch"], cm["mass_fluxes"], cm["momentum"], cm["Area"]
    conservative, limit_alpha, limit_band = bool(sch.conservative), bool(sch.limit_alpha), float(sch.limit_band)
    tau_c, den_min = float(sch.drift_relax), float(sch.drift_den_min)
    from .network.data import BC_DIRICHLET as _BCD
    ghost_in_free = sch.bc_in[SLOT_ALPHA] != _BCD
    ghost_out_free = sch.bc_out[SLOT_ALPHA] != _BCD

    def mass_rates_primitive(Y, qL, qG, ic):
        dqL = (qL[1:] - qL[:-1]) / h
        dqG = (qG[1:] - qG[:-1]) / h
        rhoL, rhoG, CL, CG = ic["rhoL"][1:N + 1], ic["rhoG"][1:N + 1], ic["CL"][1:N + 1], ic["CG"][1:N + 1]
        aGr = jnp.clip(Y[1:N + 1, SLOT_ALPHA], 0.0, 1.0)
        aLr = 1.0 - aGr
        auxr = aLr / fl.gammaL + aGr / fl.gammaG
        rL = (aLr / fl.gammaL) / auxr
        rG = (aGr / fl.gammaG) / auxr
        rL2 = (1.0 / fl.gammaL) / auxr
        rG2 = (1.0 / fl.gammaG) / auxr
        F1 = -rG / (Area * rhoL)
        F2 = rL / (Area * rhoG)
        F3 = rL2 / (Area * PR * CL)
        F4 = rG2 / (Area * PR * CG)
        return -(F1 * dqL + F2 * dqG), -(F3 * dqL + F4 * dqG)

    def rhs(Y):
        Yc = Y[0:N + 2]
        Yc = Yc.at[0, SLOT_QL].set(Y[0, SLOT_QL]).at[0, SLOT_QG].set(Y[0, SLOT_QG])
        Yc = Yc.at[N + 1, SLOT_QL].set(Y[N, SLOT_QL]).at[N + 1, SLOT_QG].set(Y[N, SLOT_QG])
        Ycc = Yc.at[1:N + 1, SLOT_QL].set(0.5 * (Y[0:N, SLOT_QL] + Y[1:N + 1, SLOT_QL]))
        Ycc = Ycc.at[1:N + 1, SLOT_QG].set(0.5 * (Y[0:N, SLOT_QG] + Y[1:N + 1, SLOT_QG]))
        tc = mj.conservative_terms(Ycc, th_nodes)
        aG_f0 = 0.5 * (Yc[0:N + 1, SLOT_ALPHA] + Yc[1:N + 2, SLOT_ALPHA])
        qm = Y[:, SLOT_QL] + Y[:, SLOT_QG]
        if not ghost_in_free:
            w0 = switch(qm[0:1], "L", frac=0.02, floor_scale=0.05)[0]
            aG_f0 = aG_f0.at[0].set(w0 * Yc[0, SLOT_ALPHA] + (1.0 - w0) * Yc[1, SLOT_ALPHA])
        if not ghost_out_free:
            wN = switch(qm[N:N + 1], "L", frac=0.02, floor_scale=0.05)[0]
            aG_f0 = aG_f0.at[N].set(wN * Yc[N, SLOT_ALPHA] + (1.0 - wN) * Yc[N + 1, SLOT_ALPHA])
        Yf = jnp.stack([aG_f0, 0.5 * (Yc[0:N + 1, SLOT_P] + Yc[1:N + 2, SLOT_P]), Y[0:N + 1, SLOT_QL], Y[0:N + 1, SLOT_QG]], axis=1)
        tf = mj.conservative_terms(Yf, th_faces)
        qL = Y[0:N + 1, SLOT_QL]
        qG = Y[0:N + 1, SLOT_QG]
        qM = qL + qG
        daG_c, dPT_c = mass_rates_primitive(Y, qL, qG, tc["inp"])
        if conservative:
            FL, FG = mass_fluxes(Y)
            dalpha = -(FG[1:] - FG[:-1]) / (Area * h)
            dPT = -(FL[1:] - FL[:-1]) / (Area * h)
        else:
            dalpha, dPT = daG_c, dPT_c
            if limit_alpha:
                aG = Y[1:N + 1, SLOT_ALPHA]
                r_lo = jnp.clip(aG / limit_band, 0.0, 1.0)
                r_hi = jnp.clip((1.0 - aG) / limit_band, 0.0, 1.0)
                dalpha = jnp.where(dalpha < 0, dalpha * r_lo, dalpha * r_hi)
        ones_f = jnp.ones(N + 1)
        Gamma_f = jnp.pi * pb.model.params.D * tf["tension"]
        dqM, _ = momentum(qM, tf["rhoS"], ones_f, tc["P"], jnp.zeros(N + 2), Gamma_f, None, tc["C_mix"])
        HG_c = jnp.clip(Yc[:, SLOT_ALPHA], 0.0, 1.0)
        w = switch(qM, None)
        HG_f = w * HG_c[:-1] + (1.0 - w) * HG_c[1:]
        dHG_c = jnp.concatenate([jnp.zeros(1), daG_c, jnp.zeros(1)])
        dPT_cf = jnp.concatenate([jnp.zeros(1), dPT_c, jnp.zeros(1)])
        dHG_f = w * dHG_c[:-1] + (1.0 - w) * dHG_c[1:]
        dP_f = 0.5 * (dPT_cf[:-1] + dPT_cf[1:]) * PR
        den = jnp.maximum(tf["dHGdqG"] - tf["dHGdqL"], den_min)
        g = tf["HG"] - HG_f
        dqG = (dHG_f - tf["dHGdP"] * dP_f - tf["dHGdqL"] * dqM - g / tau_c) / den
        dY = jnp.zeros((N + 2, 4))
        dY = dY.at[1:N + 1, SLOT_ALPHA].set(dalpha).at[1:N + 1, SLOT_P].set(dPT)
        dY = dY.at[0:N + 1, SLOT_QL].set(dqM - dqG).at[0:N + 1, SLOT_QG].set(dqG)
        return dY

    return rhs


def make_staggered_twofluid(pb, only_fluxes=False):
    sch = pb.scheme
    if pb.model.name != "twofluid":
        raise NotImplementedError("rhs JAX: fv_upwind (dois fluidos) só com o modelo a dois fluidos")
    if sch.order not in (1, 2) or sch.alpha_face not in ("upwind", "central", "muscl"):
        raise NotImplementedError("rhs JAX: fv_upwind com fv_order em (1, 2) e alpha_face em ('upwind', 'central', 'muscl')")
    order, alpha_central, alpha_muscl = int(sch.order), sch.alpha_face == "central", sch.alpha_face == "muscl"
    mj = make_model_j(pb.model)
    fl = pb.model.fluid
    N = sch.N
    h = jnp.asarray(sch.h)
    dxf = jnp.asarray(sch.dx_faces)
    th_nodes = jnp.asarray(sch.theta_nodes)
    th_faces = jnp.asarray(sch.theta_faces)
    sin_f = jnp.sin(th_faces)
    Area = mj.Area
    PR = pb.model.params.PR
    q_floor = {"L": float(sch.q_switch_L), "G": float(sch.q_switch_G)}
    switch_frac, alpha_band, relax_time = float(sch.switch_frac), float(sch.alpha_band), float(sch.relax_time)
    flux_band, flux_eps, flux_limit = float(sch.flux_band), float(sch.flux_eps), bool(sch.flux_limit)
    art_visc, conservative, limit_alpha, limit_band = float(sch.art_visc), bool(sch.conservative), bool(sch.limit_alpha), float(sch.limit_band)
    relax_noslip = getattr(sch, "relax_target", "raw") == "noslip"
    virtual_mass = float(getattr(pb.model.params, "virtual_mass", 0.0))
    vm_lambda = float(getattr(pb.model.params, "vm_lambda", 1.0))
    x_faces = jnp.asarray(sch.x_faces)
    a_min = float(pb.model.params.alpha_min)
    from .network.data import BC_DIRICHLET as _BCD
    ghost_in_free = sch.bc_in[SLOT_ALPHA] != _BCD      # fantasma de entrada: fluxo definido pelo contorno (r = 1)
    ghost_out_free = sch.bc_out[SLOT_ALPHA] != _BCD

    def switch(q, phase, frac=None, floor_scale=1.0):
        aq = jnp.abs(q)
        local = aq
        local = local.at[1:].set(jnp.maximum(local[1:], aq[:-1]))
        local = local.at[:-1].set(jnp.maximum(local[:-1], aq[1:]))
        e = (switch_frac if frac is None else frac) * local + q_floor[phase] * floor_scale
        return 0.5 * (1.0 + jnp.tanh(q / e))

    def flux_limiter(aG_c, q, phase):
        if not flux_limit:
            return jnp.ones(N + 1)
        a = aG_c if phase == "G" else 1.0 - aG_c
        u = jnp.clip(a / flux_band, 0.0, 1.0)
        r = u * u * (3.0 - 2.0 * u)
        if ghost_in_free:
            r = r.at[0].set(1.0)
        if ghost_out_free:
            r = r.at[N + 1].set(1.0)
        s = 0.5 * (1.0 + jnp.tanh(q / (flux_eps * q_floor[phase])))
        return s * r[:-1] + (1.0 - s) * r[1:]

    def momentum(qf, rho_f, alpha_f, Pc, dPc, Gamma_f, phase, comp_c):
        vf = qf / (rho_f * alpha_f * Area)
        qc = jnp.concatenate([qf[:1], 0.5 * (qf[:-1] + qf[1:]), qf[-1:]])
        w = switch(qc[1:N + 1], phase)
        if order == 2:
            # correção MUSCL (minmod) sobre a velocidade doadora (StaggeredFVScheme._momentum)
            dv = vf[1:] - vf[:-1]
            sl = _minmod(dv[:-1], dv[1:])
            slope = jnp.concatenate([jnp.zeros(1), sl, jnp.zeros(1)])
            vmid = w * (vf[:-1] + 0.5 * slope[:-1]) + (1.0 - w) * (vf[1:] - 0.5 * slope[1:])
        else:
            vmid = w * vf[:-1] + (1.0 - w) * vf[1:]
        vdon = jnp.concatenate([vf[:1], vmid, vf[-1:]])
        mom_c = qc * vdon
        dq = (-(mom_c[1:] - mom_c[:-1]) / dxf - Area * alpha_f * (Pc[1:] - Pc[:-1]) / dxf
              - Area * (dPc[1:] - dPc[:-1]) / dxf - Gamma_f - G * rho_f * alpha_f * Area * sin_f)
        if art_visc > 0:
            c_c = 1.0 / jnp.sqrt(jnp.maximum(comp_c[1:N + 1], 1e-300))
            F = art_visc * c_c * (qf[1:] - qf[:-1])
            dq = dq.at[1:N].add((F[1:] - F[:-1]) / dxf[1:N])
        return dq, (mom_c[1:] - mom_c[:-1]) / dxf

    if only_fluxes:
        def mass_fluxes(Y):
            qL = Y[0:N + 1, SLOT_QL]
            qG = Y[0:N + 1, SLOT_QG]
            aG_raw_c = Y[0:N + 2, SLOT_ALPHA]
            return qL * flux_limiter(aG_raw_c, qL, "L"), qG * flux_limiter(aG_raw_c, qG, "G")
        return mass_fluxes

    def rhs(Y):
        Yc = Y[0:N + 2]
        Yc = Yc.at[0, SLOT_QL].set(Y[0, SLOT_QL]).at[0, SLOT_QG].set(Y[0, SLOT_QG])
        Yc = Yc.at[N + 1, SLOT_QL].set(Y[N, SLOT_QL]).at[N + 1, SLOT_QG].set(Y[N, SLOT_QG])
        Ycc = Yc.at[1:N + 1, SLOT_QL].set(0.5 * (Y[0:N, SLOT_QL] + Y[1:N + 1, SLOT_QL]))
        Ycc = Ycc.at[1:N + 1, SLOT_QG].set(0.5 * (Y[0:N, SLOT_QG] + Y[1:N + 1, SLOT_QG]))
        tc = mj.conservative_terms(Ycc, th_nodes)
        aG_f0 = 0.5 * (Yc[0:N + 1, SLOT_ALPHA] + Yc[1:N + 2, SLOT_ALPHA])
        qm = Y[:, SLOT_QL] + Y[:, SLOT_QG]
        if not ghost_in_free:
            w0 = switch(qm[0:1], "L", frac=0.02, floor_scale=0.05)[0]
            aG_f0 = aG_f0.at[0].set(w0 * Yc[0, SLOT_ALPHA] + (1.0 - w0) * Yc[1, SLOT_ALPHA])
        if not ghost_out_free:
            wN = switch(qm[N:N + 1], "L", frac=0.02, floor_scale=0.05)[0]
            aG_f0 = aG_f0.at[N].set(wN * Yc[N, SLOT_ALPHA] + (1.0 - wN) * Yc[N + 1, SLOT_ALPHA])
        Yf = jnp.stack([aG_f0, 0.5 * (Yc[0:N + 1, SLOT_P] + Yc[1:N + 2, SLOT_P]),
                        Y[0:N + 1, SLOT_QL], Y[0:N + 1, SLOT_QG]], axis=1)
        tf = mj.conservative_terms(Yf, th_faces)
        ic, iff, Sc, Sf = tc["inp"], tf["inp"], tc["S"], tf["S"]
        qL = Y[0:N + 1, SLOT_QL]
        qG = Y[0:N + 1, SLOT_QG]
        if conservative:
            aG_raw_c = Yc[:, SLOT_ALPHA]
            FL = qL * flux_limiter(aG_raw_c, qL, "L")
            FG = qG * flux_limiter(aG_raw_c, qG, "G")
            dalpha = -(FG[1:] - FG[:-1]) / (Area * h)
            dPT = -(FL[1:] - FL[:-1]) / (Area * h)
        else:
            dqL = (qL[1:] - qL[:-1]) / h
            dqG = (qG[1:] - qG[:-1]) / h
            rhoL, rhoG, CL, CG = ic["rhoL"][1:N + 1], ic["rhoG"][1:N + 1], ic["CL"][1:N + 1], ic["CG"][1:N + 1]
            aGr = jnp.clip(Y[1:N + 1, SLOT_ALPHA], 0.0, 1.0)
            aLr = 1.0 - aGr
            auxr = aLr / fl.gammaL + aGr / fl.gammaG
            rL = (aLr / fl.gammaL) / auxr
            rG = (aGr / fl.gammaG) / auxr
            rL2 = (1.0 / fl.gammaL) / auxr
            rG2 = (1.0 / fl.gammaG) / auxr
            F1 = -rG / (Area * rhoL)
            F2 = rL / (Area * rhoG)
            F3 = rL2 / (Area * PR * CL)
            F4 = rG2 / (Area * PR * CG)
            dalpha = -(F1 * dqL + F2 * dqG)
            dPT = -(F3 * dqL + F4 * dqG)
            if limit_alpha:
                aG = Y[1:N + 1, SLOT_ALPHA]
                r_lo = jnp.clip(aG / limit_band, 0.0, 1.0)
                r_hi = jnp.clip((1.0 - aG) / limit_band, 0.0, 1.0)
                dalpha = jnp.where(dalpha < 0, dalpha * r_lo, dalpha * r_hi)
        aG_c = ic["alphaG"]
        if alpha_central:
            aG_fL = aG_fG = iff["alphaG"]
        else:
            wL = switch(qL, "L")
            wG = switch(qG, "G")
            if alpha_muscl:
                da = aG_c[1:] - aG_c[:-1]
                slope = jnp.concatenate([jnp.zeros(1), _minmod(da[:-1], da[1:]), jnp.zeros(1)])
                aG_left, aG_right = aG_c[:-1] + 0.5 * slope[:-1], aG_c[1:] - 0.5 * slope[1:]
            else:
                aG_left, aG_right = aG_c[:-1], aG_c[1:]
            aG_fL = wL * aG_left + (1.0 - wL) * aG_right
            aG_fG = wG * aG_left + (1.0 - wG) * aG_right
        corrL = ic["alphaL"] * Sc["dPL"]
        corrG = ic["alphaG"] * Sc["dPG"]
        corrL = corrL.at[0].set(corrL[1]).at[-1].set(corrL[-2])
        corrG = corrG.at[0].set(corrG[1]).at[-1].set(corrG[-2])
        dqLf, dmomL = momentum(qL, iff["rhoL"], 1.0 - aG_fL, tc["P"], corrL, Sf["GammaL"], "L", ic["CL"])
        dqGf, dmomG = momentum(qG, iff["rhoG"], aG_fG, tc["P"], corrG, Sf["GammaG"], "G", ic["CG"])
        if virtual_mass > 0:
            IL = iff["rhoL"] * (1.0 - aG_fL) * Area
            IG = iff["rhoG"] * aG_fG * Area
            wv = tc["vm_weight"]
            K = virtual_mass * 0.5 * (wv[:-1] + wv[1:]) * aG_fG * iff["rhoL"] * Area
            FL = dqLf + dmomL
            FG = dqGf + dmomG
            if vm_lambda != 2.0:
                vG_f = qG / (iff["rhoG"] * aG_fG * Area)
                vL_f = qL / (iff["rhoL"] * (1.0 - aG_fL) * Area)
                dvG = jnp.gradient(vG_f, x_faces)
                dvL = jnp.gradient(vL_f, x_faces)
                extra = -K * (vG_f - vL_f) * ((vm_lambda - 2.0) * dvG + (1.0 - vm_lambda) * dvL)
                FG = FG + extra
                FL = FL - extra
            det = IG * IL + K * (IG + IL)
            aG_acc = (FG * (IL + K) + K * FL) / det
            aL_acc = (FL * (IG + K) + K * FG) / det
            dqGf = IG * aG_acc - dmomG
            dqLf = IL * aL_acc - dmomL
        if relax_time > 0:
            aG_f = iff["alphaG"]
            aG_c_raw = jnp.clip(Yc[:, SLOT_ALPHA], 0.0, 1.0)
            wGs = switch(qG, "G", frac=0.02, floor_scale=0.05)
            wLs = switch(qL, "L", frac=0.02, floor_scale=0.05)
            aG_raw = wGs * aG_c_raw[:-1] + (1.0 - wGs) * aG_c_raw[1:]
            aL_raw = 1.0 - (wLs * aG_c_raw[:-1] + (1.0 - wLs) * aG_c_raw[1:])
            wGr = jnp.clip(aG_raw / alpha_band, 0.0, 1.0)
            wLr = jnp.clip(aL_raw / alpha_band, 0.0, 1.0)
            vL_f = qL / (iff["rhoL"] * (1.0 - aG_f) * Area)
            vG_f = qG / (iff["rhoG"] * aG_f * Area)
            if relax_noslip:
                qG_ns = iff["rhoG"] * jnp.maximum(aG_raw, a_min) * Area * vL_f
                qL_ns = iff["rhoL"] * jnp.maximum(aL_raw, a_min) * Area * vG_f
            else:
                qG_ns = iff["rhoG"] * aG_raw * Area * vL_f
                qL_ns = iff["rhoL"] * aL_raw * Area * vG_f
            dqGf = wGr * dqGf + (1.0 - wGr) * (qG_ns - qG) / relax_time
            dqLf = wLr * dqLf + (1.0 - wLr) * (qL_ns - qL) / relax_time
        dY = jnp.zeros((N + 2, 4))
        dY = dY.at[1:N + 1, SLOT_ALPHA].set(dalpha).at[1:N + 1, SLOT_P].set(dPT)
        dY = dY.at[0:N + 1, SLOT_QL].set(dqLf).at[0:N + 1, SLOT_QG].set(dqGf)
        return dY

    return rhs


def make_boundary_fluxes_j(pb):
    """Fluxos de massa (L, G) nas faces de entrada e saída do duto a partir de Y
    (limitados nos dutos fv_upwind conservativos; vazões das faces nos demais)."""
    sch = pb.scheme
    if sch.name == "fv_upwind" and sch.conservative and pb.model.name in ("twofluid", "beggs_brill"):
        lim = _staggered_common(pb)["mass_fluxes"]

        def fluxes(Y):
            FL, FG = lim(Y)
            return (FL[0], FG[0]), (FL[-1], FG[-1])
        return fluxes
    i0, i1 = pb.inlet_q_node, pb.outlet_q_node

    def fluxes(Y):
        return (Y[i0, SLOT_QL], Y[i0, SLOT_QG]), (Y[i1, SLOT_QL], Y[i1, SLOT_QG])
    return fluxes


def make_scheme_j(pb):
    name = pb.scheme.name
    if name == "fd_central":
        return make_fd_central(pb)
    if name == "fd_upwind":
        return make_fd_upwind(pb)
    if name == "galerkin":
        return make_galerkin(pb, petrov=False)
    if name == "petrov_galerkin":
        return make_galerkin(pb, petrov=True)
    if name == "fv_upwind":
        if pb.model.name == "twofluid":
            return make_staggered_twofluid(pb)
        if pb.model.name in ("liquid", "gas"):
            return make_staggered_single(pb)
        if pb.model.name == "beggs_brill":
            return make_staggered_drift(pb)
        raise NotImplementedError(f"rhs JAX: fv_upwind com o modelo {pb.model.name}")
    raise NotImplementedError(f"rhs JAX: esquema {name}")
