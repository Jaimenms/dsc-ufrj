"""Modelo da rede: montagem do vetor de estados e do sistema de EDOs
(equivalente a calculos_iniciais.m + modelo_rede.m + modelo_vertice.m).
"""

from __future__ import annotations

import numpy as np

from ..discretization import SCHEMES
from ..geometry import pipe_mesh
from ..models import MODEL_BY_LEGACY_ID
from ..models.base import PipeParams, SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
from .data import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN, Case
from .elements import check_valve_flows, reservoir_flows, valve_upstream_pressure
from .specs import UNSPECIFIED, TimeTable

TOL_Q = 1e-6


class PipeBlock:
    """Um duto discretizado dentro da rede."""

    def __init__(self, pipe, fluid, settings, method, scheme_settings):
        self.pipe = pipe
        self.mesh = pipe_mesh(pipe.x, pipe.y, pipe.z, pipe.n_elements, grading=getattr(pipe, "grading", None))
        params = PipeParams(D=pipe.D, eps=pipe.roughness, PR=settings.PR, regime=pipe.regime,
                            **{k: v for k, v in pipe.extra.items()
                               if k in ("k_switch", "ipe_smooth", "ipe_local", "alpha_min", "alpha_max", "legacy_convective",
                                        "regime_vertical", "vertical_angle", "vertical_blend", "interfacial_pressure", "ip_slip_max", "low_void_regime", "low_void_alpha", "high_void_regime", "high_void_alpha", "bb_derivatives", "fi_mult", "fi_vgt", "virtual_mass", "vm_lambda", "bubble_drag")})
        self.model = MODEL_BY_LEGACY_ID[pipe.model](fluid, params)
        ss = dict(scheme_settings)
        ss.update(dict(n_quad=settings.n_quad, bc_in=pipe.bc_in, bc_out=pipe.bc_out))
        ss.update({k: v for k, v in pipe.extra.items()
                   if k.startswith("fv_") or k in ("alpha_face", "pg_alpha", "q_switch", "q_switch_L", "q_switch_G", "switch_frac", "alpha_band", "relax_time", "limit_alpha", "limit_band", "conservative", "flux_band", "flux_eps", "flux_limit", "art_visc", "relax_target", "drift_relax", "drift_den_min", "ic_holdup")})
        self.scheme = SCHEMES[method](self.model, self.mesh, ss)
        self.n_nodes = self.scheme.n_nodes
        self.slots = list(self.model.slots)
        self.bc_in = tuple(pipe.bc_in)
        self.bc_out = tuple(pipe.bc_out)
        self.inlet_P_node = getattr(self.scheme, "inlet_P_node", 0)
        self.outlet_P_node = getattr(self.scheme, "outlet_P_node", self.n_nodes - 1)
        self.inlet_q_node = getattr(self.scheme, "inlet_q_node", 0)
        self.outlet_q_node = getattr(self.scheme, "outlet_q_node", self.n_nodes - 1)

        mask = self.scheme.state_mask()
        # remove condições de contorno Dirichlet/Neumann das extremidades
        for s in range(4):
            if self.bc_in[s] in (BC_DIRICHLET, BC_NEUMANN):
                mask[0, s] = False
            if self.bc_out[s] in (BC_DIRICHLET, BC_NEUMANN):
                mask[-1, s] = False
        self.mask = mask
        self.free = np.flatnonzero(mask.ravel())
        self.n_states = self.free.size
        self.offset = 0  # definido pela rede

    def state_index(self, node, slot):
        """Índice global do estado (node, slot) ou None se não for estado."""
        if node < 0:
            node += self.n_nodes
        if not self.mask[node, slot]:
            return None
        flat = node * 4 + slot
        pos = np.searchsorted(self.free, flat)
        return self.offset + int(pos)

    def base_Y(self):
        """Arranjo nodal com valores padrão para slots não modelados."""
        Y = np.zeros((self.n_nodes, 4))
        if self.model.name == "gas":
            Y[:, SLOT_ALPHA] = 1.0
        elif self.model.name == "liquid":
            Y[:, SLOT_ALPHA] = 0.0
        return Y

    def assemble(self, Xseg, E_in, E_out):
        """Monta Y (n_nodes x 4) a partir do segmento de estados e das
        especificações dos vértices de entrada/saída (vetores de 4)."""
        Y = self.base_Y()
        Yr = Y.ravel()
        Yr[self.free] = Xseg
        Y = Yr.reshape(self.n_nodes, 4)
        for s in range(4):
            if self.bc_in[s] == BC_DIRICHLET:
                Y[0, s] = E_in[s]
            if self.bc_out[s] == BC_DIRICHLET:
                Y[-1, s] = E_out[s]
        for s in range(4):
            if self.bc_in[s] == BC_NEUMANN:
                Y[0, s] = Y[1, s]
            if self.bc_out[s] == BC_NEUMANN:
                Y[-1, s] = Y[-2, s]
        Y = self.scheme.fill_derived(Y)
        return Y

    def ic_nodal(self, ic):
        """Interpola uma condição inicial nodal (nós da malha) para os nós do esquema."""
        s_mesh = self.mesh["s"]
        x = self.scheme.x_nodes
        out = np.empty((self.n_nodes, 4))
        for s in range(4):
            out[:, s] = np.interp(x, s_mesh, ic[:, s])
        return out


class NetworkModel:
    """Sistema de EDOs da rede: dX/dt = rhs(t, X)."""

    def __init__(self, case: Case, method=None, scheme_settings=None):
        self.case = case
        self.fluid = case.fluid
        self.settings = case.settings
        self.method = method or case.settings.method
        self.PR = case.settings.PR
        scheme_settings = scheme_settings or {}

        # vértices
        self.vertices = list(case.vertices)
        self.vpos = {v.index: i for i, v in enumerate(self.vertices)}
        self.NV = len(self.vertices)
        self.vA = [i for i, v in enumerate(self.vertices) if v.kind == "A"]
        self.vA_pos = {vi: k for k, vi in enumerate(self.vA)}

        # tabelas de especificação
        t0, tf = case.settings.t0, case.settings.tf
        self.tables = []
        for v in self.vertices:
            row = []
            for key in ("alphaG", "PT", "qL", "qG"):
                row.append(TimeTable(v.spec.get(key), t0, tf))
            self.tables.append(row)

        # dutos
        self.pipes = [PipeBlock(p, self.fluid, case.settings, self.method, scheme_settings)
                      for p in case.pipes]
        off = 0
        for pb in self.pipes:
            pb.offset = off
            off += pb.n_states
        self.n_pipe_states = off
        self.n_states = off + 2 * len(self.vA)
        self.pipes_in = {i: [] for i in range(self.NV)}    # dutos que chegam ao vértice
        self.pipes_out = {i: [] for i in range(self.NV)}   # dutos que saem do vértice
        for k, pb in enumerate(self.pipes):
            self.pipes_out[self.vpos[pb.pipe.v_in]].append(k)
            self.pipes_in[self.vpos[pb.pipe.v_out]].append(k)

        # elementos especiais
        self.valves = [(self.vpos[v.vertex], v, self.pipes_in[self.vpos[v.vertex]][0],
                        TimeTable(v.opening, t0, tf), TimeTable(v.mode, t0, tf),
                        TimeTable(v.setpoint, t0, tf)) for v in case.valves]
        self.reservoirs = [(self.vpos[r.vertex], r, self.pipes_out[self.vpos[r.vertex]][0])
                           for r in case.reservoirs]
        self.check_valves = [(self.vpos[c.vertex], c) for c in case.check_valves]
        for vi, c in self.check_valves:
            if vi not in self.vA_pos:
                raise ValueError(f"válvula de retenção exige vértice dinâmico (vértice {c.vertex})")
        self.couplings = [(self.vpos[c.v_up], self.vpos[c.v_down]) for c in case.couplings]

        # o filtro anti-refluxo do código original (Pipe.check) foi removido: o refluxo é
        # resolvido pelas equações e pelo tratamento das extremidades (fantasma doador)
        self.check_idx = np.zeros(0, dtype=int)

        self._last_E = None
        # limites de fração de gás (dos parâmetros dos dutos bifásicos) e faixa do limitador
        a_min, a_max = 1e-3, 1 - 1e-3
        for pb in self.pipes:
            pa = getattr(pb.model, "params", None)
            if pa is not None and pb.model.name == "twofluid":
                a_min, a_max = pa.alpha_min, pa.alpha_max
        self.alpha_limits = (a_min, a_max, float(scheme_settings.get("alpha_band", 0.02)))

    # ------------------------------------------------------------------
    def vertex_A_index(self, vertex_index):
        """Índices globais (HG, PT) de um vértice dinâmico."""
        k = self.vA_pos[self.vpos[vertex_index]]
        base = self.n_pipe_states + 2 * k
        return base, base + 1

    def spec_breakpoints(self):
        """Instantes em que as especificações mudam de inclinação."""
        bp = set()
        for row in self.tables:
            for tb in row:
                bp.update(tb.breakpoints.tolist())
        for _, _, _, op, mo, sp in self.valves:
            for tb in (op, mo, sp):
                bp.update(tb.breakpoints.tolist())
        return np.array(sorted(bp))

    # ------------------------------------------------------------------
    def specifications(self, t, X):
        """Vetor de especificações E (NV x 4) no instante t para o estado X."""
        E = np.array([[tb(t) for tb in row] for row in self.tables], dtype=float)
        PR = self.PR
        info = {"valves": [], "reservoirs": [], "check_valves": []}

        # vértices dinâmicos: vazões externas não especificadas = 0
        for vi in self.vA:
            for s in (SLOT_QL, SLOT_QG):
                if E[vi, s] == UNSPECIFIED:
                    E[vi, s] = 0.0

        # conexões: pressão do vértice a montante = pressão no início da aresta a jusante
        for vu, vd in self.couplings:
            for k in self.pipes_out[vd]:
                pb = self.pipes[k]
                i = pb.state_index(pb.inlet_P_node, SLOT_P)
                E[vu, SLOT_P] = X[i] if i is not None else E[vu, SLOT_P]

        # válvulas
        for vi, v, k, op, mo, sp in self.valves:
            pb = self.pipes[k]
            qL = qG = 0.0
            i = pb.state_index(pb.outlet_q_node, SLOT_QL)
            if i is not None:
                qL = X[i]
            i = pb.state_index(pb.outlet_q_node, SLOT_QG)
            if i is not None:
                qG = X[i]
            opening = op(t)
            ZP, dP = valve_upstream_pressure(E[vi, SLOT_P], qL, qG, opening, v.CV, self.fluid, PR)
            info["valves"].append(dict(vertex=v.vertex, Ab=opening, ZP=ZP * PR / 1e5, dP=dP,
                                       qL=qL, qG=qG, P_down=E[vi, SLOT_P] * PR / 1e5))
            E[vi, SLOT_P] = ZP

        # reservatórios
        for vi, r, k in self.reservoirs:
            pb = self.pipes[k]
            i = pb.state_index(pb.inlet_P_node, SLOT_P)
            P = X[i] if i is not None else E[vi, SLOT_P]
            ZL, ZG, dP = reservoir_flows(P, r.IPL, r.IPG, r.P, PR)
            info["reservoirs"].append(dict(vertex=r.vertex, ZL=ZL, ZG=ZG, dP=dP, P=P * PR / 1e5))
            E[vi, SLOT_QL] = ZL
            E[vi, SLOT_QG] = ZG

        # válvulas de retenção (vértice dinâmico -> pressão externa do vértice)
        for vi, c in self.check_valves:
            iH, iP = self.vertex_A_index(c.vertex)
            qL, qG, dP = check_valve_flows(X[iP], E[vi, SLOT_P], X[iH], c.CV, c.dP_min, self.fluid, PR)
            info["check_valves"].append(dict(vertex=c.vertex, ZP=X[iP] * PR / 1e5, dP=dP, qL=qL, qG=qG))
            E[vi, SLOT_QL] = -qL
            E[vi, SLOT_QG] = -qG

        # conexões: vazões do vértice a jusante = soma das vazões no fim das arestas a montante
        acc = {}
        for vu, vd in self.couplings:
            acc.setdefault(vd, [0.0, 0.0])
            for k in self.pipes_in[vu]:
                pb = self.pipes[k]
                for j, s in enumerate((SLOT_QL, SLOT_QG)):
                    i = pb.state_index(pb.outlet_q_node, s)
                    if i is not None:
                        acc[vd][j] += X[i]
        for vd, (qL, qG) in acc.items():
            E[vd, SLOT_QL] = qL
            E[vd, SLOT_QG] = qG

        # vértices dinâmicos: hold-up e pressão do vértice
        for vi in self.vA:
            iH, iP = self.vertex_A_index(self.vertices[vi].index)
            E[vi, SLOT_ALPHA] = X[iH]
            E[vi, SLOT_P] = X[iP]
        return E, info

    def assemble(self, t, X):
        X = np.asarray(X, float)
        E, info = self.specifications(t, X)
        Ys = []
        for pb in self.pipes:
            seg = X[pb.offset:pb.offset + pb.n_states]
            E_in = E[self.vpos[pb.pipe.v_in]]
            E_out = E[self.vpos[pb.pipe.v_out]]
            Ys.append(pb.assemble(seg, E_in, E_out))
        return E, Ys, info, X

    def rhs(self, t, X):
        E, Ys, info, X = self.assemble(t, X)
        dX = np.zeros(self.n_states)
        for pb, Y in zip(self.pipes, Ys):
            dY = pb.scheme.rhs(Y)
            dX[pb.offset:pb.offset + pb.n_states] = dY.ravel()[pb.free]
        if self.vA:
            dX[self.n_pipe_states:] = self._vertex_rhs(X, E, Ys)
        return dX

    def _pipe_boundary_fluxes(self, kp, Y):
        """Fluxos de massa (L, G) nas faces de entrada e saída do duto kp: os fluxos
        limitados das equações de massa nos dutos fv_upwind conservativos (para que o
        balanço dos vértices seja exatamente consistente com o dos dutos), as vazões
        das faces nos demais."""
        pb = self.pipes[kp]
        sch = pb.scheme
        if hasattr(sch, "mass_fluxes_nodal") and getattr(sch, "conservative", False):
            FL, FG = sch.mass_fluxes_nodal(Y)
            return (FL[0], FG[0]), (FL[-1], FG[-1])
        i0, i1 = pb.inlet_q_node, pb.outlet_q_node
        return (Y[i0, SLOT_QL], Y[i0, SLOT_QG]), (Y[i1, SLOT_QL], Y[i1, SLOT_QG])

    def vertex_net_inflows(self, X, E, Ys):
        """Vazões líquidas (L, G) que entram em cada vértice dinâmico (externas + dutos)."""
        out = []
        for vi in self.vA:
            QL = E[vi, SLOT_QL]
            QG = E[vi, SLOT_QG]
            for kp in self.pipes_in[vi]:
                _, (fL, fG) = self._pipe_boundary_fluxes(kp, Ys[kp])
                QL += fL
                QG += fG
            for kp in self.pipes_out[vi]:
                (fL, fG), _ = self._pipe_boundary_fluxes(kp, Ys[kp])
                QL -= fL
                QG -= fG
            out.append((QL, QG))
        return out

    def _vertex_rhs(self, X, E, Ys):
        """Dinâmica dos vértices tipo A (balanços de massa, modelo_vertice.m)."""
        PR = self.PR
        out = np.zeros(2 * len(self.vA))
        inflows = self.vertex_net_inflows(X, E, Ys)
        for k, vi in enumerate(self.vA):
            v = self.vertices[vi]
            QL, QG = inflows[k]
            iH, iP = self.vertex_A_index(v.index)
            HG = X[iH]
            PT = X[iP]
            rhoL, rhoG, CL, CG = self.fluid.props(max(PT * PR, 1e3))
            if self.conservative:
                # modo conservativo: d(mG)/dt = QG/V, d(mL)/dt = QL/V (massas por volume; balanço exato)
                out[2 * k] = QG / v.volume
                out[2 * k + 1] = QL / v.volume
                continue
            HL = 1.0 - HG
            CVL = HL / rhoL * CL
            CVG = HG / rhoG * CG
            CVT = CVL + CVG
            BVL = QL / v.volume / rhoL
            BVG = QG / v.volume / rhoG
            BVT = BVL + BVG
            out[2 * k + 1] = BVT / CVT / PR
            dHG = BVG - CVG * BVT / CVT
            # limitador suave: derivada para fora dos limites físicos anulada numa
            # faixa junto a [alpha_min, alpha_max] (evita estados presos no limite)
            a_min, a_max, band = self.alpha_limits
            r_lo = min(max((HG - a_min) / band, 0.0), 1.0)
            r_hi = min(max((a_max - HG) / band, 0.0), 1.0)
            out[2 * k] = dHG * (r_lo if dHG < 0 else r_hi)
        return out

    # ------------------------------------------------------------------
    def initial_state(self):
        """Vetor de estados inicial a partir das condições iniciais do caso."""
        X = np.zeros(self.n_states)
        for pb in self.pipes:
            ic = pb.pipe.ic
            if ic is None:
                raise ValueError(f"duto {pb.pipe.index} sem condição inicial")
            Yic = pb.ic_nodal(np.asarray(ic, float))
            if hasattr(pb.scheme, "prepare_ic"):
                Yic = pb.scheme.prepare_ic(Yic)
            if hasattr(pb.scheme, "from_primitive"):
                Yic = pb.scheme.from_primitive(Yic)
            X[pb.offset:pb.offset + pb.n_states] = Yic.ravel()[pb.free]
        for k, vi in enumerate(self.vA):
            v = self.vertices[vi]
            iH, iP = self.vertex_A_index(v.index)
            HG0, PT0 = v.HG0, v.PT0
            if HG0 is None or PT0 is None:
                # estimativa: média das extremidades dos dutos conectados
                aG, PT = [], []
                for kp in self.pipes_in[vi]:
                    Y = self.pipes[kp].ic_nodal(np.asarray(self.pipes[kp].pipe.ic, float))
                    aG.append(Y[-1, SLOT_ALPHA]); PT.append(Y[-1, SLOT_P])
                for kp in self.pipes_out[vi]:
                    Y = self.pipes[kp].ic_nodal(np.asarray(self.pipes[kp].pipe.ic, float))
                    aG.append(Y[0, SLOT_ALPHA]); PT.append(Y[0, SLOT_P])
                HG0 = HG0 if HG0 is not None else float(np.mean(aG))
                PT0 = PT0 if PT0 is not None else float(np.mean(PT))
            X[iH] = HG0
            X[iP] = PT0
        return X

    def state_to_case_ic(self, X, t=0.0):
        """Escreve o estado X nas condições iniciais do caso (dutos e vértices A)."""
        E, Ys, info, X = self.assemble(t, X)
        for pb, Y in zip(self.pipes, Ys):
            if hasattr(pb.scheme, "to_primitive"):
                Y = pb.scheme.to_primitive(Y)
            x = pb.scheme.x_nodes
            s_mesh = pb.mesh["s"]
            ic = np.empty((s_mesh.size, 4))
            for s in range(4):
                ic[:, s] = np.interp(s_mesh, x, Y[:, s])
            pb.pipe.ic = ic
        for vi in self.vA:
            v = self.vertices[vi]
            iH, iP = self.vertex_A_index(v.index)
            v.HG0, v.PT0 = float(X[iH]), float(X[iP])

    # ------------------------------------------------------------------
    @property
    def conservative(self):
        return any(getattr(pb.scheme, "conservative", False) for pb in self.pipes)

    def _apply_mass_map(self, X, direction):
        from ..discretization.staggered import _mass_map, StaggeredFVConsScheme
        X = np.asarray(X, float).copy()
        for pb in self.pipes:
            if not getattr(pb.scheme, "conservative", False):
                continue
            seg = X[pb.offset:pb.offset + pb.n_states]
            Y = pb.base_Y().ravel()
            Y[pb.free] = seg
            Y = _mass_map(pb.scheme, Y.reshape(pb.n_nodes, 4), direction)
            X[pb.offset:pb.offset + pb.n_states] = Y.ravel()[pb.free]
        if self.vA and self.conservative:
            # vértices dinâmicos: (HG, PT) <-> massas por volume (mG, mL) = (rhoG HG, rhoL (1 - HG))
            # (o volume V é constante e fica fora do mapa; a EDO do vértice devolve QG/V, QL/V)
            fl = self.fluid
            for vi in self.vA:
                iH, iP = self.vertex_A_index(self.vertices[vi].index)
                if direction == "to_mass":
                    P = max(X[iP] * self.PR, 1e3)
                    rhoL, rhoG, _, _ = fl.props(P)
                    HG = X[iH]
                    X[iH], X[iP] = rhoG * HG, rhoL * (1.0 - HG)
                else:
                    mG, mL = X[iH], X[iP]
                    pb0 = self.pipes[0]
                    P = StaggeredFVConsScheme._solve_P(pb0.scheme, np.array([mL]), np.array([mG]), P0=None, clip=False)[0]
                    rhoL, rhoG, _, _ = fl.props(P)
                    X[iH], X[iP] = mG / rhoG, P / self.PR
        return X

    def transform(self, X):
        """Mapa conservativo T(X): massas por volume nas células dos dutos conservativos."""
        return self._apply_mass_map(X, "to_mass")

    def inverse_transform(self, Y):
        return self._apply_mass_map(Y, "to_primitive")

    def state_scales(self, X0=None):
        """Ordem de grandeza (faixa) de cada estado, para tolerâncias absolutas uniformes.

        alphaG/HG: 1; PT: maior pressão adimensional das especificações/estado inicial
        (>= 1); qL, qG: maior vazão especificada (vértices, IPR dos reservatórios em
        pressão nula, estado inicial) por fase (>= 1e-3 kg/s). Pode ser sobrescrita
        pelo caso em settings.extra["state_ref"] = {"PT": ..., "qL": ..., "qG": ...}.
        """
        ref = dict(self.settings.extra.get("state_ref", {})) if hasattr(self.settings, "extra") else {}
        E = np.array([[tb(self.settings.t0) for tb in row] for row in self.tables], dtype=float)
        spec = {}
        for s_, key in ((SLOT_P, "PT"), (SLOT_QL, "qL"), (SLOT_QG, "qG")):
            vals = [abs(v) for v in E[:, s_] if v != UNSPECIFIED]
            for row in self.tables:
                tb = row[s_]
                if tb.is_specified:
                    vals.append(float(np.max(np.abs(tb.v))))
            spec[key] = max(vals) if vals else 0.0
        for vi, r, k in self.reservoirs:
            spec["qL"] = max(spec["qL"], abs(r.IPL * r.P))
            spec["qG"] = max(spec["qG"], abs(r.IPG * r.P))
        if X0 is None:
            try:
                X0 = self.initial_state()
            except ValueError:
                X0 = None
        if X0 is not None:
            for pb in self.pipes:
                seg = X0[pb.offset:pb.offset + pb.n_states]
                sl = pb.free % 4
                for s_, key in ((SLOT_P, "PT"), (SLOT_QL, "qL"), (SLOT_QG, "qG")):
                    if np.any(sl == s_):
                        spec[key] = max(spec[key], float(np.max(np.abs(seg[sl == s_]))))
        PT_ref = float(ref.get("PT", max(spec["PT"], 1.0)))
        qL_ref = float(ref.get("qL", max(spec["qL"], 1e-3)))
        qG_ref = float(ref.get("qG", max(spec["qG"], 1e-3)))
        scale = np.ones(self.n_states)
        for pb in self.pipes:
            sl = pb.free % 4
            seg = slice(pb.offset, pb.offset + pb.n_states)
            sc = scale[seg]
            sc[sl == SLOT_P] = PT_ref
            sc[sl == SLOT_QL] = qL_ref
            sc[sl == SLOT_QG] = qG_ref
            scale[seg] = sc
        for vi in self.vA:
            iH, iP = self.vertex_A_index(self.vertices[vi].index)
            scale[iP] = PT_ref
        return scale

    def boundary_net_fluxes(self, t, X):
        """Vazões líquidas de entrada (entrada - saída, kg/s) de líquido e gás de cada
        duto, concatenadas [L_0, G_0, L_1, G_1, ...]. Nos dutos fv_upwind conservativos
        são os fluxos limitados (os mesmos das equações de massa). Usado como
        acumulador do integrador para verificar a conservação de massa."""
        E, Ys, _, _ = self.assemble(t, X)
        out = []
        for kp, (pb, Y) in enumerate(zip(self.pipes, Ys)):
            (fLi, fGi), (fLo, fGo) = self._pipe_boundary_fluxes(kp, Y)
            out += [fLi - fLo, fGi - fGo]
        # vértices dinâmicos: vazões líquidas de entrada (L, G), após os dutos
        for QL, QG in self.vertex_net_inflows(X, E, Ys):
            out += [QL, QG]
        return np.array(out)

    def transform_jacobian(self, X):
        """Jacobiana analítica (esparsa, bloco-diagonal) de T(X).

        Por célula, na base (alphaG, PT): d(mG)/d(alphaG) = rhoG, d(mG)/d(PT) = PR CG alphaG,
        d(mL)/d(alphaG) = -rhoL, d(mL)/d(PT) = PR CL (1 - alphaG). Identidade nos demais estados.
        (Analítica para não depender de diferenças finitas junto aos limites 0 e 1 de alphaG.)"""
        import scipy.sparse as sp
        X = np.asarray(X, float)
        n = X.size
        rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.ones(n)]
        for pb in self.pipes:
            if not getattr(pb.scheme, "conservative", False):
                continue
            ia, ip = self._mass_state_indices(pb)
            if ia.size == 0:
                continue
            aG = np.clip(X[ia], -0.01, 1.01)            # undershoot/overshoot admitidos pelos limites
            P = np.maximum(X[ip] * self.PR, 1e3)
            rhoL, rhoG, CL, CG = self.fluid.props(P)
            rows.append(np.concatenate([ia, ia, ip, ip]))
            cols.append(np.concatenate([ia, ip, ia, ip]))
            # linhas: mG (slot 0) e mL (slot 1); colunas: alphaG, PT (vetorizado sobre as células)
            vals.append(np.concatenate([rhoG - 1.0, self.PR * CG * aG, -rhoL, self.PR * CL * (1.0 - aG) - 1.0]))
        if self.vA and self.conservative:
            for vi in self.vA:
                iH, iP = self.vertex_A_index(self.vertices[vi].index)
                HG = min(max(X[iH], -0.01), 1.01)
                P = max(X[iP] * self.PR, 1e3)
                rhoL, rhoG, CL, CG = self.fluid.props(P)
                rows.append(np.array([iH, iH, iP, iP]))
                cols.append(np.array([iH, iP, iH, iP]))
                vals.append(np.array([rhoG - 1.0, self.PR * CG * HG, -rhoL, self.PR * CL * (1.0 - HG) - 1.0]))
        return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    def _mass_state_indices(self, pb):
        """Índices globais (alphaG, PT) das células do duto com ambos os estados livres (cache)."""
        cache = getattr(self, "_mass_idx_cache", None)
        if cache is None:
            cache = self._mass_idx_cache = {}
        key = id(pb)
        if key not in cache:
            ia, ip = [], []
            for c in range(1, pb.scheme.N + 1):
                a, b = pb.state_index(c, 0), pb.state_index(c, 1)
                if a is not None and b is not None:
                    ia.append(a); ip.append(b)
            cache[key] = (np.array(ia, dtype=int), np.array(ip, dtype=int))
        return cache[key]

    def jac_sparsity(self, exact_galerkin=False):
        """Padrão de esparsidade (estrutural, generoso) da Jacobiana.

        Para Galerkin a inversa da matriz de massa é densa dentro do duto; por
        padrão usa-se uma banda (suficiente para o Newton modificado do
        integrador); com exact_galerkin=True usa-se o bloco denso (Newton do
        estado estacionário).
        """
        n = self.n_states
        S = np.zeros((n, n), dtype=bool)
        groups = {}  # vértice -> conjunto de estados "de contorno" conectados
        for pb in self.pipes:
            a, b = pb.offset, pb.offset + pb.n_states
            nodes = pb.free // 4
            if self.method in ("galerkin", "petrov_galerkin"):
                # M^-1 é densa, mas decai exponencialmente (razão ~0.27 por nó):
                # banda de +-band nós dá uma Jacobiana aproximada suficiente
                # para o Newton modificado
                band = 10**6 if exact_galerkin else int(self.settings.extra.get("galerkin_band", 8))
            else:
                band = 2
            d = np.abs(nodes[:, None] - nodes[None, :]) <= band
            S[a:b, a:b] = d
            nn = pb.n_nodes
            first = pb.offset + np.flatnonzero(pb.free // 4 <= 2)
            last = pb.offset + np.flatnonzero(pb.free // 4 >= nn - 3)
            vin = self.vpos[pb.pipe.v_in]
            vout = self.vpos[pb.pipe.v_out]
            groups.setdefault(vin, set()).update(first.tolist())
            groups.setdefault(vout, set()).update(last.tolist())
        for vi in self.vA:
            iH, iP = self.vertex_A_index(self.vertices[vi].index)
            groups.setdefault(vi, set()).update([iH, iP])
        # une grupos ligados por acoplamentos
        parent = {v: v for v in groups}

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        for vu, vd in self.couplings:
            if vu in parent and vd in parent:
                parent[find(vu)] = find(vd)
        merged = {}
        for v, g in groups.items():
            merged.setdefault(find(v), set()).update(g)
        for g in merged.values():
            idx = np.array(sorted(g))
            S[np.ix_(idx, idx)] = True
        return S
