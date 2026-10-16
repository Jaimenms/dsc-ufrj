"""Sistema de EDOs da rede montado sobre o grafo (`Network`): layout do vetor de estados,
especificações no tempo, plano de avaliação das arestas de comprimento zero
(`network.ports`), lado direito puro no backend injetado e rotinas de apoio ao integrador
(estado inicial, mapa conservativo, escalas, esparsidade, fluxos de contorno).

Equivalente a calculos_iniciais.m + modelo_rede.m + modelo_vertice.m. O layout do vetor
de estados é o legado (dutos na ordem do grafo, nós livres por nó e slot; depois (HG, PT)
por tanque), logo as comparações com `estados.X` do MATLAB mantêm-se. `rhs_core(x, E_t,
opt_t)` é função pura das entradas no instante (`time_inputs(t)`, em numpy) e corre igual
em numpy e em JAX (`mfsim.compiled`).
"""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY
from ..core.options import PIPE_PARAM_KEYS, SCHEME_KEYS
from ..core.slots import BC_DIRICHLET, BC_NEUMANN, SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
from ..discretization import SCHEMES
from ..geometry import pipe_mesh
from ..models import MODEL_BY_LEGACY_ID
from ..models.base import PipeParams
from .graph import CheckValve, Choke, Connector, JunctionVertex, Network, Reservoir, TankVertex, ZeroLengthEdge
from .ports import build_plan
from .schedules import UNSPECIFIED, Unspecified

SLOT_KEYS = ("alphaG", "PT", "qL", "qG")
N_QUAD = (3, 5, 8, 12)


class PipeBlock:
    """Um duto do grafo discretizado dentro da rede."""

    def __init__(self, index, pipe, fluid, PR, n_quad, scheme_settings, bk=None):
        self.index = index
        self.pipe = pipe
        self.bk = bk or NUMPY
        self.fluid, self.PR = fluid, PR
        pts = np.asarray(pipe.points, float)
        self.mesh = pipe_mesh(
            pts[:, 0].copy(),
            pts[:, 1].copy(),
            pts[:, 2].copy(),
            np.asarray(pipe.n_elements, int),
            grading=None if pipe.grading is None else list(pipe.grading),
        )
        model_id, regime_id, extra = pipe.model.to_legacy()
        extra.update(pipe.scheme.to_legacy_extra())
        self.extra = extra
        params = PipeParams(
            D=float(pipe.D),
            eps=float(pipe.roughness),
            PR=PR,
            regime=regime_id,
            **{k: v for k, v in extra.items() if k in PIPE_PARAM_KEYS},
        )
        self.model = MODEL_BY_LEGACY_ID[model_id](fluid, params, bk=self.bk)
        self.bc_in = tuple(k.legacy for k in pipe.ports_in)
        self.bc_out = tuple(k.legacy for k in pipe.ports_out)
        ss = dict(scheme_settings)
        ss.update(dict(n_quad=n_quad, bc_in=self.bc_in, bc_out=self.bc_out))
        ss.update({k: v for k, v in extra.items() if k.startswith("fv_") or k in SCHEME_KEYS})
        self.scheme = SCHEMES[pipe.scheme.type](self.model, self.mesh, ss)
        self.n_nodes = self.scheme.n_nodes
        self.slots = list(self.model.slots)
        self.inlet_P_node = self.scheme.inlet_P_node
        self.outlet_P_node = self.scheme.outlet_P_node
        self.inlet_q_node = self.scheme.inlet_q_node
        self.outlet_q_node = self.scheme.outlet_q_node
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
        self.offset = 0  # definidos pela rede
        self.i_from = self.i_to = None
        self._base_Y_flat = self.bk.xp.asarray(self.base_Y().ravel())
        self._ic = None

    # --- identidade -------------------------------------------------------
    @property
    def v_from(self):
        return self.pipe.v_from

    @property
    def v_to(self):
        return self.pipe.v_to

    @property
    def ic(self):
        """Condição inicial nodal (nós da malha x 4): os arrays do duto ou a receita materializada."""
        if self._ic is None:
            if self.pipe.ic is not None:
                self._ic = np.asarray(self.pipe.ic, float)
            elif self.pipe.ic_recipe is not None:
                from ..initial.recipes import materialize

                self._ic = np.asarray(
                    materialize(
                        self.pipe.ic_recipe,
                        self.pipe.points,
                        self.pipe.n_elements,
                        self.pipe.grading,
                        self.fluid,
                        self.PR,
                    ),
                    float,
                )
        return self._ic

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
        especificações dos vértices de entrada/saída (vetores de 4); função pura."""
        bk = self.bk
        Y = bk.set(self._base_Y_flat, self.free, Xseg).reshape(self.n_nodes, 4)
        for s in range(4):
            if self.bc_in[s] == BC_DIRICHLET:
                Y = bk.set(Y, (0, s), E_in[s])
            if self.bc_out[s] == BC_DIRICHLET:
                Y = bk.set(Y, (-1, s), E_out[s])
        for s in range(4):
            if self.bc_in[s] == BC_NEUMANN:
                Y = bk.set(Y, (0, s), Y[1, s])
            if self.bc_out[s] == BC_NEUMANN:
                Y = bk.set(Y, (-1, s), Y[-2, s])
        return self.scheme.fill_derived(Y)

    def ic_nodal(self, ic):
        """Condição inicial nos nós do esquema: interpolada dos nós da malha, ou tal e qual se o
        duto a guarda nos nós do esquema (`ic_at = "scheme"`)."""
        if getattr(self.pipe, "ic_at", "mesh") == "scheme":
            ic = np.asarray(ic, float)
            if ic.shape != (self.n_nodes, 4):
                raise ValueError(
                    f"duto {self.pipe.id}: condição inicial nos nós do esquema com forma {ic.shape}, esperado {(self.n_nodes, 4)}"
                )
            return ic.copy()
        s_mesh = self.mesh["s"]
        x = self.scheme.x_nodes
        out = np.empty((self.n_nodes, 4))
        for s in range(4):
            out[:, s] = np.interp(x, s_mesh, ic[:, s])
        return out


class NetworkSystem:
    """Sistema de EDOs da rede: dX/dt = rhs(t, X), montado sobre um `Network`."""

    def __init__(self, network: Network, scheme_settings=None, bk=None):
        self.network = network.validate()
        self.bk = bk or NUMPY
        self.xp = self.bk.xp
        self.fluid = network.fluid
        self.PR = float(network.reference.PR)
        self.state_ref = dict(network.reference.state_ref)
        self.t0, self.tf, self.dt_out = float(network.time.t0), float(network.time.tf), float(network.time.dt_out)
        self.meta = dict(network.meta or {})
        self.integration = network.integration
        self.scheme_settings = dict(scheme_settings or {})

        # vértices (ids; os ids numéricos também indexam `vpos` como inteiros, compatibilidade)
        self.vertices = list(network.vertices)
        self.vpos = {}
        for i, v in enumerate(self.vertices):
            self.vpos[v.id] = i
            if v.id.lstrip("-").isdigit():
                self.vpos.setdefault(int(v.id), i)
        self.NV = len(self.vertices)
        self.vA = [i for i, v in enumerate(self.vertices) if isinstance(v, TankVertex)]
        self.vA_pos = {vi: k for k, vi in enumerate(self.vA)}

        # tabelas de especificação (Schedule por porta; ausente = Unspecified)
        self.tables = [
            [
                v.spec.get(key) if hasattr(v, "spec") and v.spec.get(key) is not None else Unspecified()
                for key in SLOT_KEYS
            ]
            for v in self.vertices
        ]
        self._specified = np.array([[tb.is_specified for tb in row] for row in self.tables], dtype=bool)

        # dutos
        n_quad = N_QUAD[max(0, min(3, int(network.integration.method_quadrature) - 1))]
        self.pipes = [
            PipeBlock(k, p, self.fluid, self.PR, n_quad, self.scheme_settings, bk=self.bk)
            for k, p in enumerate(network.pipes)
        ]
        off = 0
        for pb in self.pipes:
            pb.offset = off
            off += pb.n_states
            pb.i_from, pb.i_to = self.vpos[pb.pipe.v_from], self.vpos[pb.pipe.v_to]
        self.n_pipe_states = off
        self.n_states = off + 2 * len(self.vA)
        self.pipes_in = {i: [] for i in range(self.NV)}  # dutos que chegam ao vértice
        self.pipes_out = {i: [] for i in range(self.NV)}  # dutos que saem do vértice
        for k, pb in enumerate(self.pipes):
            self.pipes_out[pb.i_from].append(k)
            self.pipes_in[pb.i_to].append(k)
        self.method = self.pipes[0].scheme.name if self.pipes else None
        # dutos que chegam a / partem de cada vértice também através das junções dos chokes
        # (Jc -> v) e dos reservatórios (R -> Jr): é o que o legado via como "dutos do vértice"
        self.arriving = {i: self._through(i, self.pipes_in, Choke, "v_to", "v_from") for i in range(self.NV)}
        self.leaving = {i: self._through(i, self.pipes_out, Reservoir, "v_from", "v_to") for i in range(self.NV)}
        # linha de E que representa cada vértice nas saídas (`Results`): a da junção que o
        # desdobra (Jc<v>: pressão a montante do choke; Jr<v>: vazões do reservatório), como
        # o vértice legado reportava; os outros vértices representam-se a si próprios
        self.report_row = {i: i for i in range(self.NV)}
        for e in network.edges:
            if isinstance(e, Choke) and isinstance(network.vertex(e.v_from), JunctionVertex):
                self.report_row[self.vpos[e.v_to]] = self.vpos[e.v_from]
            elif isinstance(e, Reservoir) and isinstance(network.vertex(e.v_to), JunctionVertex):
                self.report_row[self.vpos[e.v_from]] = self.vpos[e.v_to]

        # arestas de comprimento zero: plano estático de avaliação
        self.plan, self.chokes = build_plan(self)
        self.connectors = network.edges_of(Connector)
        self.reservoirs = network.edges_of(Reservoir)
        self.check_valves = network.edges_of(CheckValve)
        self.couplings = [(self.vpos[c.v_from], self.vpos[c.v_to]) for c in self.connectors]

        # limites de fração de gás (dos parâmetros dos dutos bifásicos) e faixa do limitador
        a_min, a_max = 1e-3, 1 - 1e-3
        for pb in self.pipes:
            pa = getattr(pb.model, "params", None)
            if pa is not None and pb.model.name == "twofluid":
                a_min, a_max = pa.alpha_min, pa.alpha_max
        self.alpha_limits = (a_min, a_max, float(self.scheme_settings.get("alpha_band", 0.02)))

    def _through(self, vi, direct, cls, near, far, _seen=None):
        seen = set() if _seen is None else _seen
        out = list(direct[vi])
        for e in self.network.edges:
            if isinstance(e, cls) and self.vpos[getattr(e, near)] == vi and e.id not in seen:
                seen.add(e.id)
                out += self._through(self.vpos[getattr(e, far)], direct, cls, near, far, seen)
        return out

    # ------------------------------------------------------------------
    def vertex_pos(self, v):
        """Posição de um vértice dado o id (ou índice legado inteiro)."""
        return self.vpos[v]

    def vertex_A_index(self, vertex):
        """Índices globais (HG, PT) de um vértice dinâmico (id ou índice legado)."""
        k = self.vA_pos[self.vpos[vertex]]
        base = self.n_pipe_states + 2 * k
        return base, base + 1

    def state_slots(self):
        """Slot (0 alphaG, 1 PT, 2 qL, 3 qG) de cada estado do vetor X: dutos (nós livres)
        e, no fim, (HG, PT) de cada vértice dinâmico (0, 1)."""
        parts = [pb.free % 4 for pb in self.pipes] + [np.array([0, 1], dtype=int) for _ in self.vA]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

    def spec_breakpoints(self):
        """Instantes em que as especificações (vértices e aberturas dos chokes) mudam de inclinação."""
        bp = {self.t0}
        for row in self.tables:
            for tb in row:
                bp.update(np.asarray(tb.breakpoints, float).tolist())
        for c in self.chokes:
            for sch in (c.opening, c.mode, c.setpoint):
                if sch is not None:
                    bp.update(np.asarray(sch.breakpoints, float).tolist())
        return np.array(sorted(bp))

    # ------------------------------------------------------------------
    def time_inputs(self, t):
        """Entradas no instante t (numpy): tabelas dos vértices E_t (NV x 4) e aberturas
        dos chokes opt_t (uma por choke, na ordem do grafo)."""
        E = np.array([[tb(t) for tb in row] for row in self.tables], dtype=float)
        opt = np.array([c.opening(t) for c in self.chokes], dtype=float) if self.chokes else np.zeros(0)
        return E, opt

    def specs_core(self, x, E_t, opt_t, info=None):
        """Vetor de especificações E (NV x 4) para o estado x a partir das entradas no
        instante; função pura no backend. `info` (só no lado numpy) recebe os diagnósticos
        dos chokes, reservatórios e válvulas de retenção."""
        bk = self.bk
        E = bk.asarray(E_t)
        # vértices dinâmicos: vazões externas não especificadas = 0
        for vi in self.vA:
            for s in (SLOT_QL, SLOT_QG):
                if not self._specified[vi, s]:
                    E = bk.set(E, (vi, s), 0.0)
        # arestas de comprimento zero (plano estático)
        for step in self.plan:
            E = step(x, E, opt_t, info, bk)
        # vértices dinâmicos: hold-up e pressão do vértice
        for vi in self.vA:
            iH, iP = self.vertex_A_index(self.vertices[vi].id)
            E = bk.set(E, (vi, SLOT_ALPHA), x[iH])
            E = bk.set(E, (vi, SLOT_P), x[iP])
        return E

    def specifications(self, t, X):
        """Vetor de especificações E (NV x 4) no instante t para o estado X e os
        diagnósticos dos elementos (lado numpy)."""
        info = {"valves": [], "reservoirs": [], "check_valves": []}
        E_t, opt_t = self.time_inputs(t)
        E = self.specs_core(X, E_t, opt_t, info)
        return E, info

    def assemble(self, t, X):
        X = np.asarray(X, float)
        E, info = self.specifications(t, X)
        Ys = [pb.assemble(X[pb.offset : pb.offset + pb.n_states], E[pb.i_from], E[pb.i_to]) for pb in self.pipes]
        return E, Ys, info, X

    def rhs_core(self, x, E_t, opt_t):
        """dX/dt a partir do estado x e das entradas no instante (função pura no backend)."""
        xp = self.xp
        E = self.specs_core(x, E_t, opt_t)
        Ys = []
        parts = []
        for pb in self.pipes:
            Y = pb.assemble(x[pb.offset : pb.offset + pb.n_states], E[pb.i_from], E[pb.i_to])
            Ys.append(Y)
            dY = pb.scheme.rhs(Y)
            parts.append(dY.ravel()[pb.free])
        if self.vA:
            parts.append(self._vertex_rhs(x, E, Ys))
        return xp.concatenate(parts) if parts else xp.zeros(self.n_states)

    def rhs(self, t, X):
        X = np.asarray(X, float)
        E_t, opt_t = self.time_inputs(t)
        return self.bk.to_numpy(self.rhs_core(X, E_t, opt_t))

    def _pipe_boundary_fluxes(self, kp, Y):
        """Fluxos de massa (L, G) nas faces de entrada e saída do duto kp (os fluxos
        limitados das equações de massa nos dutos fv_upwind conservativos)."""
        return self.pipes[kp].scheme.boundary_fluxes(Y)

    def _vertex_inflow(self, vi, E, Ys):
        """Vazões líquidas (L, G) que entram no vértice vi (externas + dutos)."""
        QL = E[vi, SLOT_QL]
        QG = E[vi, SLOT_QG]
        for kp in self.pipes_in[vi]:
            _, (fL, fG) = self._pipe_boundary_fluxes(kp, Ys[kp])
            QL = QL + fL
            QG = QG + fG
        for kp in self.pipes_out[vi]:
            (fL, fG), _ = self._pipe_boundary_fluxes(kp, Ys[kp])
            QL = QL - fL
            QG = QG - fG
        return QL, QG

    def vertex_net_inflows(self, X, E, Ys):
        """Vazões líquidas (L, G) que entram em cada vértice dinâmico (externas + dutos)."""
        return [self._vertex_inflow(vi, E, Ys) for vi in self.vA]

    def _vertex_rhs(self, X, E, Ys):
        """Dinâmica dos vértices dinâmicos (balanços de massa, modelo_vertice.m)."""
        xp = self.xp
        PR = self.PR
        out = []
        a_min, a_max, band = self.alpha_limits
        for vi in self.vA:
            v = self.vertices[vi]
            QL, QG = self._vertex_inflow(vi, E, Ys)
            iH, iP = self.vertex_A_index(v.id)
            HG = X[iH]
            PT = X[iP]
            if self.conservative:
                # modo conservativo: d(mG)/dt = QG/V, d(mL)/dt = QL/V (massas por volume; balanço exato)
                out += [QG / v.volume, QL / v.volume]
                continue
            rhoL, rhoG, CL, CG = self.fluid.props(xp.maximum(PT * PR, 1e3))
            HL = 1.0 - HG
            CVL = HL / rhoL * CL
            CVG = HG / rhoG * CG
            CVT = CVL + CVG
            BVL = QL / v.volume / rhoL
            BVG = QG / v.volume / rhoG
            BVT = BVL + BVG
            dPT = BVT / CVT / PR
            dHG = BVG - CVG * BVT / CVT
            # limitador suave: derivada para fora dos limites físicos anulada numa
            # faixa junto a [alpha_min, alpha_max] (evita estados presos no limite)
            r_lo = xp.clip((HG - a_min) / band, 0.0, 1.0)
            r_hi = xp.clip((a_max - HG) / band, 0.0, 1.0)
            dHG = xp.where(dHG < 0, dHG * r_lo, dHG * r_hi)
            out += [dHG, dPT]
        return xp.stack(out)

    # ------------------------------------------------------------------
    def initial_state(self):
        """Vetor de estados inicial a partir das condições iniciais do grafo."""
        X = np.zeros(self.n_states)
        for pb in self.pipes:
            ic = pb.ic
            if ic is None:
                raise ValueError(f"duto {pb.pipe.id} sem condição inicial")
            Yic = pb.ic_nodal(np.asarray(ic, float))
            Yic = pb.scheme.prepare_ic(Yic)
            if hasattr(pb.scheme, "from_primitive"):
                Yic = pb.scheme.from_primitive(Yic)
            X[pb.offset : pb.offset + pb.n_states] = Yic.ravel()[pb.free]
        for vi in self.vA:
            v = self.vertices[vi]
            iH, iP = self.vertex_A_index(v.id)
            HG0, PT0 = v.HG0, v.PT0
            if HG0 is None or PT0 is None:
                # estimativa: média das extremidades dos dutos conectados
                aG, PT = [], []
                for kp in self.pipes_in[vi]:
                    Y = self.pipes[kp].ic_nodal(np.asarray(self.pipes[kp].ic, float))
                    aG.append(Y[-1, SLOT_ALPHA])
                    PT.append(Y[-1, SLOT_P])
                for kp in self.pipes_out[vi]:
                    Y = self.pipes[kp].ic_nodal(np.asarray(self.pipes[kp].ic, float))
                    aG.append(Y[0, SLOT_ALPHA])
                    PT.append(Y[0, SLOT_P])
                HG0 = HG0 if HG0 is not None else float(np.mean(aG))
                PT0 = PT0 if PT0 is not None else float(np.mean(PT))
            X[iH] = HG0
            X[iP] = PT0
        return X

    def state_to_network_ic(self, X, t=0.0, exact=False):
        """Escreve o estado X nas condições iniciais do grafo (dutos e tanques). Por omissão nos
        nós da malha (interpolado, como a tese guarda); com `exact` nos nós do esquema
        (`ic_at = "scheme"`), o que torna a ida e volta estado -> caso -> estado a identidade
        também nos esquemas escalonados (issue 040)."""
        E, Ys, info, X = self.assemble(t, X)
        for pb, Y in zip(self.pipes, Ys):
            if hasattr(pb.scheme, "to_primitive"):
                Y = pb.scheme.to_primitive(Y)
            if exact:
                pb.pipe.ic = np.asarray(Y, float).copy()
                pb.pipe.ic_at = "scheme"
            else:
                x = pb.scheme.x_nodes
                s_mesh = pb.mesh["s"]
                ic = np.empty((s_mesh.size, 4))
                for s in range(4):
                    ic[:, s] = np.interp(s_mesh, x, Y[:, s])
                pb.pipe.ic = ic
                pb.pipe.ic_at = "mesh"
            pb.pipe.ic_recipe = None
            pb._ic = None
        for vi in self.vA:
            v = self.vertices[vi]
            iH, iP = self.vertex_A_index(v.id)
            v.HG0, v.PT0 = float(X[iH]), float(X[iP])

    state_to_case_ic = state_to_network_ic

    # ------------------------------------------------------------------
    @property
    def conservative(self):
        return any(getattr(pb.scheme, "conservative", False) for pb in self.pipes)

    def _apply_mass_map(self, X, direction):
        from ..discretization.staggered import _mass_map, solve_pressure

        X = np.asarray(X, float).copy()
        for pb in self.pipes:
            if not getattr(pb.scheme, "conservative", False):
                continue
            seg = X[pb.offset : pb.offset + pb.n_states]
            Y = pb.base_Y().ravel()
            Y[pb.free] = seg
            Y = _mass_map(pb.scheme, Y.reshape(pb.n_nodes, 4), direction)
            X[pb.offset : pb.offset + pb.n_states] = Y.ravel()[pb.free]
        if self.vA and self.conservative:
            # vértices dinâmicos: (HG, PT) <-> massas por volume (mG, mL) = (rhoG HG, rhoL (1 - HG))
            # (o volume V é constante e fica fora do mapa; a EDO do vértice devolve QG/V, QL/V)
            fl = self.fluid
            for vi in self.vA:
                iH, iP = self.vertex_A_index(self.vertices[vi].id)
                if direction == "to_mass":
                    P = max(X[iP] * self.PR, 1e3)
                    rhoL, rhoG, _, _ = fl.props(P)
                    HG = X[iH]
                    X[iH], X[iP] = rhoG * HG, rhoL * (1.0 - HG)
                else:
                    mG, mL = X[iH], X[iP]
                    P = solve_pressure(fl, np.array([mL]), np.array([mG]), P0=None, clip=False)[0]
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
        pelo caso em `reference.state_ref` = {"PT": ..., "qL": ..., "qG": ...}.
        """
        ref = dict(self.state_ref)
        E = np.array([[tb(self.t0) for tb in row] for row in self.tables], dtype=float)
        spec = {}
        for s_, key in ((SLOT_P, "PT"), (SLOT_QL, "qL"), (SLOT_QG, "qG")):
            vals = [abs(v) for v in E[:, s_] if v != UNSPECIFIED]
            for row in self.tables:
                tb = row[s_]
                if tb.is_specified:
                    vals.append(float(np.max(np.abs(tb.values))))
            spec[key] = max(vals) if vals else 0.0
        for r in self.reservoirs:
            spec["qL"] = max(spec["qL"], abs(r.IPL * r.P))
            spec["qG"] = max(spec["qG"], abs(r.IPG * r.P))
        if X0 is None:
            try:
                X0 = self.initial_state()
            except ValueError:
                X0 = None
        if X0 is not None:
            for pb in self.pipes:
                seg = X0[pb.offset : pb.offset + pb.n_states]
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
            iH, iP = self.vertex_A_index(self.vertices[vi].id)
            scale[iP] = PT_ref
        return scale

    def boundary_net_fluxes(self, t, X):
        """Vazões líquidas de entrada (entrada - saída, kg/s) de líquido e gás de cada
        duto, concatenadas [L_0, G_0, L_1, G_1, ...], seguidas das dos vértices dinâmicos.
        Nos dutos fv_upwind conservativos são os fluxos limitados (os mesmos das equações
        de massa). Usado como acumulador do integrador para verificar a conservação."""
        E, Ys, _, _ = self.assemble(t, X)
        out = []
        for kp, (pb, Y) in enumerate(zip(self.pipes, Ys)):
            (fLi, fGi), (fLo, fGo) = self._pipe_boundary_fluxes(kp, Y)
            out += [fLi - fLo, fGi - fGo]
        for QL, QG in self.vertex_net_inflows(X, E, Ys):
            out += [QL, QG]
        return np.array(out)

    def transform_jacobian(self, X):
        """Jacobiana analítica (esparsa, bloco-diagonal) de T(X).

        Por célula, na base (alphaG, PT): d(mG)/d(alphaG) = rhoG, d(mG)/d(PT) = PR CG alphaG,
        d(mL)/d(alphaG) = -rhoL, d(mL)/d(PT) = PR CL (1 - alphaG). Identidade nos demais estados."""
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
            aG = np.clip(X[ia], -0.01, 1.01)  # undershoot/overshoot admitidos pelos limites
            P = np.maximum(X[ip] * self.PR, 1e3)
            rhoL, rhoG, CL, CG = self.fluid.props(P)
            rows.append(np.concatenate([ia, ia, ip, ip]))
            cols.append(np.concatenate([ia, ip, ia, ip]))
            vals.append(np.concatenate([rhoG - 1.0, self.PR * CG * aG, -rhoL, self.PR * CL * (1.0 - aG) - 1.0]))
        if self.vA and self.conservative:
            for vi in self.vA:
                iH, iP = self.vertex_A_index(self.vertices[vi].id)
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
                    ia.append(a)
                    ip.append(b)
            cache[key] = (np.array(ia, dtype=int), np.array(ip, dtype=int))
        return cache[key]

    def jac_sparsity(self, exact_galerkin=False):
        """Padrão de esparsidade (estrutural, generoso) da Jacobiana.

        Dentro de cada duto, uma banda de nós (Galerkin: a inversa da matriz de massa
        decai exponencialmente, banda `integration.galerkin_band`; `exact_galerkin=True`
        usa o bloco denso, para o Newton do estado estacionário). As extremidades dos
        dutos e os estados dos tanques ligados por um vértice, e os vértices ligados por
        arestas de comprimento zero, formam blocos densos."""
        n = self.n_states
        S = np.zeros((n, n), dtype=bool)
        groups = {i: set() for i in range(self.NV)}  # vértice -> conjunto de estados "de contorno" conectados
        gb = self.integration.galerkin_band
        for pb in self.pipes:
            a, b = pb.offset, pb.offset + pb.n_states
            nodes = pb.free // 4
            if pb.scheme.name in ("galerkin", "petrov_galerkin"):
                band = 10**6 if exact_galerkin else int(8 if gb is None else gb)
            else:
                band = 2
            d = np.abs(nodes[:, None] - nodes[None, :]) <= band
            S[a:b, a:b] = d
            nn = pb.n_nodes
            first = pb.offset + np.flatnonzero(pb.free // 4 <= 2)
            last = pb.offset + np.flatnonzero(pb.free // 4 >= nn - 3)
            groups.setdefault(pb.i_from, set()).update(first.tolist())
            groups.setdefault(pb.i_to, set()).update(last.tolist())
        for vi in self.vA:
            iH, iP = self.vertex_A_index(self.vertices[vi].id)
            groups.setdefault(vi, set()).update([iH, iP])
        # une grupos ligados por arestas de comprimento zero
        parent = {v: v for v in groups}

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        for e in self.network.edges:
            if isinstance(e, ZeroLengthEdge):
                parent[find(self.vpos[e.v_from])] = find(self.vpos[e.v_to])
        merged = {}
        for v, g in groups.items():
            merged.setdefault(find(v), set()).update(g)
        for g in merged.values():
            if g:
                idx = np.array(sorted(g))
                S[np.ix_(idx, idx)] = True
        return S
