"""Volumes finitos em malha escalonada com upwind pela velocidade de fase.

Proposta de esquema robusto para o modelo a dois fluidos (e monofásicos):

  * alphaG e PT nos centros das N células; qL e qG nas N+1 faces;
  * balanços de massa exatos por célula (as vazões nas faces são estados,
    não é preciso interpolar fluxo de massa);
  * convecção de momento por "donor cell": o fluxo de momento no centro da
    célula usa a velocidade da face a montante segundo o sentido local do
    escoamento de cada fase;
  * gradiente de pressão e correções de pressão centrados entre células
    vizinhas (acoplamento pressão-vazão naturalmente estável na malha
    escalonada, sem modos par-ímpar);
  * termos fonte (atrito, gravidade) avaliados nas faces com os
    fechamentos por padrão de escoamento.

Ao contrário do Galerkin/diferenças centrais, o esquema introduz
dissipação numérica apenas nos termos convectivos (ondas de fração de
área), não depende da hiperbolicidade do sistema e é conservativo.

Layout nodal (n_nodes = N+2):
  nó 0      : contorno de entrada (alphaG, PT: valores de contorno; qL, qG: face 0)
  nós 1..N  : células (alphaG, PT) e faces 1..N (qL, qG)
  nó N+1    : contorno de saída (alphaG, PT: valores de contorno; qL, qG: cópia da face N)

Classes: `StaggeredFVBase` (malha, chaveamento doador, momento por fase, limitador de
fluxo, contornos) e as concretas por modelo — `StaggeredTwoFluidFV` (dois fluidos),
`StaggeredDriftFluxFV` (deriva de Beggs & Brill) e `StaggeredSinglePhaseFV`
(monofásicos); `staggered_scheme(model, mesh, settings)` escolhe a classe pelo modelo.
`StaggeredTwoFluidConsFV` (fv_upwind_cons) tem as massas por volume como estados.
"""

from __future__ import annotations

import numpy as np

from ..backend import NUMPY
from ..core.slots import G, SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
from ..core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN
from .base import PipeScheme


def _minmod(a, b, xp=np):
    return xp.where(a * b > 0, xp.sign(a) * xp.minimum(xp.abs(a), xp.abs(b)), 0.0)


def solve_pressure(fluid, mL, mG, P0=None, clip=True, bk=NUMPY, max_iter=30):
    """Pressão (Pa) tal que mL/rhoL(P) + mG/rhoG(P) = 1 (Newton escalar por célula, com
    passo amortecido; o laço pára quando todas as células convergem a 1e-9 relativo ou
    após `max_iter` iterações).

    clip=False admite massas ligeiramente negativas (undershoot de alphaG no modo
    conservativo do esquema primitivo); a equação continua bem posta enquanto
    alphaG > -gammaG/gammaL."""
    xp = bk.xp
    mL = bk.asarray(mL)
    mG = bk.asarray(mG)
    if clip:
        mL = xp.maximum(mL, 0.0)
        mG = xp.maximum(mG, 0.0)
    if P0 is None:
        # estimativa inicial: pressão que satisfaz a fase líquida sozinha (boa quando
        # mG é pequeno) limitada a um intervalo razoável
        P0 = fluid.P0 * xp.clip(xp.maximum(mL, 1e-3) / fluid.rhoL0, 0.5, 2.0) ** fluid.gammaL
    P = xp.maximum(bk.asarray(P0) * xp.ones(mL.shape), 1e3)

    def body(c):
        P, i, _ = c
        rhoL, rhoG, CL, CG = fluid.props(P)
        f = mL / rhoL + mG / rhoG - 1.0
        df = -(mL / rhoL**2 * CL + mG / rhoG**2 * CG)
        df = xp.where(xp.abs(df) < 1e-300, -1e-300, df)
        step = f / df
        # amortece passos grandes (log-escala) e mantém P > 0
        step = xp.clip(step, -0.5 * P, 0.5 * P)
        P = P - step
        return P, i + 1, xp.all(xp.abs(step) <= 1e-9 * P)

    def cond(c):
        return (c[1] < max_iter) & ~c[2]

    P, _, _ = bk.while_loop(cond, body, (P, xp.asarray(0), xp.asarray(False)))
    return P


class StaggeredFVBase(PipeScheme):
    """Malha escalonada, chaveamento doador, momento por fase e contornos (comum aos
    modelos). As subclasses definem `two_phase`, `cell_slots` e `_rhs_phases`."""

    name = "fv_upwind"
    two_phase = True
    # estados de célula: [alphaG, PT] nos modelos bifásicos (no modelo de deriva o
    # hold-up HG é estado de célula, embora não seja estado do modelo nodal), só PT nos
    # monofásicos
    cell_slots = (SLOT_ALPHA, SLOT_P)

    def __init__(self, model, mesh, settings=None):
        super().__init__(model, mesh, settings)
        self.bc_in = tuple(self.settings.get("bc_in", (1, 1, 2, 2)))
        self.bc_out = tuple(self.settings.get("bc_out", (1, 2, 1, 1)))
        self.order = int(self.settings.get("fv_order", 1))  # 1 = upwind, 2 = MUSCL/minmod
        self.alpha_face = self.settings.get(
            "alpha_face", "upwind"
        )  # upwind | central | muscl (reconstrução TVD/minmod)
        self.q_switch = float(self.settings.get("q_switch", 1e-3))  # kg/s, largura do chaveamento upwind
        self.q_switch_L = float(self.settings.get("q_switch_L", self.q_switch))
        self.q_switch_G = float(self.settings.get("q_switch_G", 1e-2 * self.q_switch))
        self.switch_frac = float(self.settings.get("switch_frac", 0.2))
        self.alpha_band = float(self.settings.get("alpha_band", 0.02))  # faixa de transição junto aos limites de alpha
        self.relax_time = float(self.settings.get("relax_time", 0.05))  # s, relaxação da fase que desaparece
        # resíduo de massa conservativo (massas por volume): só nos modelos bifásicos; nos
        # monofásicos a equação da pressão já é o balanço de massa da única fase
        self.conservative = bool(self.settings.get("conservative", False)) and self.two_phase
        self.limit_alpha = bool(self.settings.get("limit_alpha", not self.conservative))
        self.limit_band = float(self.settings.get("limit_band", 1e-4))
        # modo conservativo: banda (em alpha) do limitador de fluxo da fase que some
        # na célula doadora (positividade exata das massas sem perder conservação)
        self.flux_band = float(self.settings.get("flux_band", 1e-3))
        self.flux_eps = float(self.settings.get("flux_eps", 0.05))  # largura do chaveamento doador (x q_switch)
        self.flux_limit = bool(self.settings.get("flux_limit", True))  # liga o limitador de fluxo (modo conservativo)
        # viscosidade artificial linear nas equações de momento, nu = art_visc * c * h
        # (c = velocidade do som da fase, h = tamanho da célula): amortece os modos
        # acústicos na escala da malha (tampões de líquido entre bolsões de gás
        # "tocam" a ~c/2L e prendem o passo de tempo); desaparece com h -> 0 e não
        # afeta ondas longas (taxa ~ art_visc c h k^2)
        self.art_visc = float(self.settings.get("art_visc", 0.0))
        # alvo da relaxação da fase que desaparece: "raw" (vazão sem escorregamento com a
        # fração REAL da célula doadora: com fração nula o alvo é vazão nula, isto é, a
        # fase ausente fica PARADA) ou "noslip" (idem com a fração doadora com piso em
        # alpha_min: a fase virtual, que os fechamentos veem com fração alpha_min,
        # acompanha a outra fase sem escorregamento). Com a fase virtual parada o
        # estratificado e o anular exercem sobre o líquido uma força interfacial espúria
        # de 40 a 60 vezes o atrito de parede em toda célula cheia de líquido; "noslip" a
        # elimina. Só no modo conservativo (o limitador de fluxo impede que a vazão
        # virtual drene massa de uma célula sem a fase); no primitivo mantém "raw".
        self.relax_target = str(self.settings.get("relax_target", "raw"))
        self.relax_ramp = str(self.settings.get("relax_ramp", "linear"))  # "linear" | "smooth" (C1; issue 052)
        if self.relax_target not in ("raw", "noslip"):
            raise ValueError(f"relax_target: {self.relax_target}")
        if self.relax_target != "raw" and not self.conservative:
            raise ValueError("relax_target='noslip' exige o modo conservativo (limitador de fluxo)")
        # fração de gás com que se avaliam os fechamentos nas faces (atrito de parede e interfacial,
        # velocidades da relaxação): "mean" (média das células vizinhas, como até aqui) ou "donor"
        # (célula doadora pela vazão volumétrica da mistura, a mesma fração dos termos de transporte e de
        # gravidade). Com "mean" cada equação de momento de face mistura as duas frações e o estacionário
        # discreto ganha um modo par-ímpar: xadrez de alphaG no riser e um modo local instável na curva
        # (issue 044)
        self.closure_alpha = str(self.settings.get("closure_alpha", "mean"))
        if self.closure_alpha not in ("mean", "donor"):
            raise ValueError(f"closure_alpha: {self.closure_alpha}")
        N = self.N
        h = self.h
        # centros das células e faces
        s = np.asarray(mesh["s"], float)
        self.x_faces = s  # N+1 faces
        self.x_cells = 0.5 * (s[:-1] + s[1:])  # N células
        self.dx_faces = np.empty(N + 1)  # distância entre centros vizinhos à face
        self.dx_faces[0] = 0.5 * h[0]
        self.dx_faces[-1] = 0.5 * h[-1]
        self.dx_faces[1:-1] = 0.5 * (h[:-1] + h[1:])
        th = self.theta_elem
        self.theta_faces = np.empty(N + 1)
        self.theta_faces[0] = th[0]
        self.theta_faces[-1] = th[-1]
        self.theta_faces[1:-1] = 0.5 * (th[:-1] + th[1:])
        self.phases = self._phases()
        # modelo de deriva: constante de tempo da estabilização (Baumgarte) da restrição
        # HG_BB(P, qL, qG) = HG das células nas faces, e piso do coeficiente dHG/dqG - dHG/dqL
        self.drift_relax = float(self.settings.get("drift_relax", 0.05))
        self.drift_den_min = float(self.settings.get("drift_den_min", 1e-3))
        # hold-up inicial das células no modelo de deriva: "correlation" (das vazões e pressão
        # da CI nodal, que na tese não carrega hold-up) ou "given" (coluna alphaG da CI)
        self.ic_holdup = str(self.settings.get("ic_holdup", "correlation"))
        # constantes no backend (lado direito)
        xp = self.xp
        self._h_x = xp.asarray(h)
        self._dxf_x = xp.asarray(self.dx_faces)
        self._xf_x = xp.asarray(self.x_faces)
        self._thn_x = xp.asarray(self.theta_nodes)
        self._thf_x = xp.asarray(self.theta_faces)
        self._sin_f_x = xp.asarray(np.sin(self.theta_faces))

    def _phases(self):
        """Lista de fases modeladas: (slot_q, 'L'/'G')."""
        slots = self.model.slots
        out = []
        if SLOT_QL in slots:
            out.append((SLOT_QL, "L"))
        if SLOT_QG in slots:
            out.append((SLOT_QG, "G"))
        return out

    # --- layout ---------------------------------------------------------------
    @property
    def n_nodes(self):
        return self.N + 2

    @property
    def x_nodes(self):
        return np.concatenate([[self.x_faces[0]], self.x_cells, [self.x_faces[-1]]])

    @property
    def theta_nodes(self):
        return np.concatenate([[self.theta_elem[0]], self.theta_elem, [self.theta_elem[-1]]])

    @property
    def inlet_P_node(self):
        return 1

    @property
    def outlet_P_node(self):
        return self.N

    @property
    def inlet_q_node(self):
        return 0

    @property
    def outlet_q_node(self):
        return self.N

    def state_mask(self):
        mask = np.zeros((self.n_nodes, 4), dtype=bool)
        for s in self.cell_slots:
            mask[1 : self.N + 1, s] = True
        for s, _ in self.phases:
            mask[0 : self.N + 1, s] = True
            if self.bc_in[s] != BC_ABSENT:
                mask[0, s] = False
            if self.bc_out[s] != BC_ABSENT:
                mask[self.N, s] = False
        return mask

    def fill_derived(self, Y):
        bk = self.bk
        N = self.N
        for s in self.cell_slots:
            if self.bc_in[s] != BC_DIRICHLET:
                Y = bk.set(Y, (0, s), Y[1, s])
            if self.bc_out[s] != BC_DIRICHLET:
                Y = bk.set(Y, (N + 1, s), Y[N, s])
        return self._fill_flows(Y)

    def _fill_flows(self, Y):
        bk = self.bk
        N = self.N
        for s, _ in self.phases:
            if self.bc_out[s] == BC_DIRICHLET:
                Y = bk.set(Y, (N, s), Y[N + 1, s])  # valor imposto no nó de contorno -> face N
            elif self.bc_out[s] == BC_NEUMANN:
                Y = bk.set(Y, (N, s), Y[N - 1, s])
                Y = bk.set(Y, (N + 1, s), Y[N, s])
            else:
                Y = bk.set(Y, (N + 1, s), Y[N, s])
            if self.bc_in[s] == BC_NEUMANN:
                Y = bk.set(Y, (0, s), Y[1, s])
        return Y

    # --- chaveamento doador e reconstruções --------------------------------------
    def _switch(self, q, phase=None, frac=None, floor_scale=1.0):
        """Peso suave do lado a montante: ~1 se q >> 0, ~0 se q << 0.

        A largura do chaveamento é adaptativa: uma fração (switch_frac) do maior
        |q| entre as faces vizinhas, com um piso por fase (q_switch_L / q_switch_G)
        -- assim o chaveamento é suave em relação às vazões locais tanto no
        escoamento lento quanto nos transientes rápidos (blowdown)."""
        xp, bk = self.xp, self.bk
        q = bk.asarray(q)
        floor = {"L": self.q_switch_L, "G": self.q_switch_G}.get(phase, self.q_switch)
        aq = xp.abs(q)
        if aq.shape[0] > 2:
            local = bk.set(aq, slice(1, None), xp.maximum(aq[1:], aq[:-1]))  # vizinho à esquerda
            local = bk.set(local, slice(None, -1), xp.maximum(local[:-1], aq[1:]))  # vizinho à direita
        else:
            local = aq
        eps = (self.switch_frac if frac is None else frac) * local + floor * floor_scale
        return 0.5 * (1.0 + xp.tanh(q / eps))

    def _face_states(self, a_c):
        """Valores de uma grandeza de célula (N+2, com fantasmas) extrapolados para as
        N+1 faces pelo lado esquerdo e pelo lado direito: constantes por célula
        (upwind, 1ª ordem) ou com reconstrução linear limitada por minmod (muscl, 2ª
        ordem TVD: o valor na face fica entre os das células vizinhas, sem extremos
        novos; nas células fantasmas o declive é nulo)."""
        xp = self.xp
        a_c = self.bk.asarray(a_c)
        if self.alpha_face != "muscl":
            return a_c[:-1], a_c[1:]
        da = a_c[1:] - a_c[:-1]
        slope = xp.concatenate([xp.zeros(1), _minmod(da[:-1], da[1:], xp), xp.zeros(1)])
        return a_c[:-1] + 0.5 * slope[:-1], a_c[1:] - 0.5 * slope[1:]

    def _flux_limiter(self, aG_c, q, phase):
        """Fator em [0, 1] do fluxo de massa de uma fase nas faces (modo conservativo).

        Vale ~1 quando a célula doadora (a montante pelo sinal de q, chaveamento
        bem estreito) tem a fase e cai a 0 quando não tem (fração < flux_band);
        nas células fantasmas o fator é 1 (o contorno define o fluxo)."""
        xp, bk = self.xp, self.bk
        N = self.N
        if not self.flux_limit:
            return xp.ones(N + 1)
        aG_c = bk.asarray(aG_c)
        a = aG_c if phase == "G" else 1.0 - aG_c
        u = xp.clip(a / self.flux_band, 0.0, 1.0)
        # rampa C1 (smoothstep): sem quina na borda da banda, onde as células
        # quase vazias tendem a se estabilizar (o Newton estagnaria numa quina)
        r = u * u * (3.0 - 2.0 * u)
        # células fantasmas: o contorno define o fluxo (r = 1), exceto quando a fração
        # de gás é imposta pelo vértice dinâmico (Dirichlet): aí o vértice é a célula
        # doadora e não pode ceder uma fase que não tem
        if self.bc_in[SLOT_ALPHA] != BC_DIRICHLET:
            r = bk.set(r, 0, 1.0)
        if self.bc_out[SLOT_ALPHA] != BC_DIRICHLET:
            r = bk.set(r, N + 1, 1.0)
        eps = self.flux_eps * (self.q_switch_G if phase == "G" else self.q_switch_L)
        s = 0.5 * (1.0 + xp.tanh(bk.asarray(q) / eps))
        return s * r[:-1] + (1.0 - s) * r[1:]

    def mass_fluxes(self, aG_c, qL, qG):
        """Fluxos de massa (kg/s) de líquido e gás nas N+1 faces usados no balanço
        conservativo (vazões das faces vezes o limitador da célula doadora).
        aG_c: fração de gás nas N+2 células (com fantasmas)."""
        return qL * self._flux_limiter(aG_c, qL, "L"), qG * self._flux_limiter(aG_c, qG, "G")

    def mass_fluxes_nodal(self, Y):
        """Idem a partir do arranjo nodal completo Y (n_nodes x 4) do duto."""
        Y = self.bk.asarray(Y)
        N = self.N
        return self.mass_fluxes(Y[0 : N + 2, SLOT_ALPHA], Y[0 : N + 1, SLOT_QL], Y[0 : N + 1, SLOT_QG])

    def boundary_fluxes(self, Y):
        """Nos dutos conservativos, os fluxos limitados das equações de massa (para que o
        balanço dos vértices seja exatamente consistente com o dos dutos)."""
        if self.conservative:
            FL, FG = self.mass_fluxes_nodal(Y)
            return (FL[0], FG[0]), (FL[-1], FG[-1])
        return super().boundary_fluxes(Y)

    # --- blocos das equações ---------------------------------------------------
    def _cell_face_states(self, Y):
        """Estados nas células com fantasmas (Yc), nas células com vazões médias das
        faces (Ycc, para os fechamentos) e nas faces (Yf)."""
        xp, bk = self.xp, self.bk
        N = self.N
        model = self.model
        # vazões nas células fantasmas: cópia das faces de contorno (só para fechamentos)
        Yc = Y
        for s, _ in self.phases:
            Yc = bk.set(Yc, (N + 1, s), Y[N, s])
        # para avaliar propriedades nas células, as vazões "de célula" são médias das faces
        Ycc = Yc
        for s, _ in self.phases:
            Ycc = bk.set(Ycc, (slice(1, N + 1), s), 0.5 * (Y[0:N, s] + Y[1 : N + 1, s]))
        # faces: média das células vizinhas para alphaG, PT; vazões das faces
        aG_f = 0.5 * (Yc[0 : N + 1, SLOT_ALPHA] + Yc[1 : N + 2, SLOT_ALPHA])
        P_f = 0.5 * (Yc[0 : N + 1, SLOT_P] + Yc[1 : N + 2, SLOT_P])
        if self.closure_alpha == "donor" and SLOT_ALPHA in model.slots:
            # célula doadora pela vazão volumétrica da mistura, escrita como vazão mássica equivalente de
            # líquido (qL + qG rhoL/rhoG) para usar as larguras do chaveamento do líquido
            rhoL_f, rhoG_f, _, _ = model.fluid.props(model.params.PR * P_f)
            w = self._switch(Y[0 : N + 1, SLOT_QL] + Y[0 : N + 1, SLOT_QG] * (rhoL_f / rhoG_f), "L")
            aG_f = w * Yc[0 : N + 1, SLOT_ALPHA] + (1.0 - w) * Yc[1 : N + 2, SLOT_ALPHA]
        # faces de contorno com alphaG imposto pelo vértice (Dirichlet): a fração da
        # face é a do lado doador (célula em saída, vértice em entrada), não a média —
        # o conteúdo do vértice só entra no duto quando o escoamento entra
        if SLOT_ALPHA in self.cell_slots:
            qm = Y[:, SLOT_QL] + Y[:, SLOT_QG]
            if self.bc_in[SLOT_ALPHA] == BC_DIRICHLET:
                w = self._switch(qm[0:1], "L", frac=0.02, floor_scale=0.05)[0]  # ~1 se entra (q > 0)
                aG_f = bk.set(aG_f, 0, w * Yc[0, SLOT_ALPHA] + (1.0 - w) * Yc[1, SLOT_ALPHA])
            if self.bc_out[SLOT_ALPHA] == BC_DIRICHLET:
                w = self._switch(qm[N : N + 1], "L", frac=0.02, floor_scale=0.05)[0]  # ~1 se sai (q > 0)
                aG_f = bk.set(aG_f, N, w * Yc[N, SLOT_ALPHA] + (1.0 - w) * Yc[N + 1, SLOT_ALPHA])
        if SLOT_ALPHA not in model.slots:
            # modelos sem a fração de área como estado: coluna constante do fantasma de entrada
            aG_f = Yc[0, SLOT_ALPHA] * xp.ones(N + 1)
        Yf = xp.stack([aG_f, P_f, Y[0 : N + 1, SLOT_QL], Y[0 : N + 1, SLOT_QG]], axis=1)
        return Yc, Ycc, Yf

    def _momentum(self, qf, rho_f, alpha_f, Pc, dPc, Gamma_f, Area, phase=None, comp_c=None):
        """Equação de momento de uma fase nas faces: devolve (dq/dt, d(fluxo de momento)/dx).

        qf      : vazões nas faces (N+1)
        rho_f, alpha_f : densidade e fração de área nas faces (N+1)
        Pc      : pressão (Pa) nas células (N+2)
        dPc     : alpha*deltaP nas células (N+2)  (correção de pressão)
        Gamma_f : força de atrito por metro nas faces (N+1)
        """
        xp, bk = self.xp, self.bk
        N = self.N
        dxf, sin_t = self._dxf_x, self._sin_f_x
        # velocidades nas faces
        vf = qf / (rho_f * alpha_f * Area)
        # fluxo de massa nos centros das células 1..N (interpolação) e nos fantasmas
        qc = xp.concatenate([qf[:1], 0.5 * (qf[:-1] + qf[1:]), qf[-1:]])
        # velocidade donor no centro i: face i-1 se qc >= 0, senão face i
        # (chaveamento suavizado para manter o lado direito contínuo)
        w = self._switch(qc[1 : N + 1], phase)
        if self.order == 2:
            # correção MUSCL (minmod) sobre a velocidade donor
            dv = vf[1:] - vf[:-1]
            slope = xp.concatenate([xp.zeros(1), _minmod(dv[:-1], dv[1:], xp), xp.zeros(1)])
            vmid = w * (vf[:-1] + 0.5 * slope[:-1]) + (1.0 - w) * (vf[1:] - 0.5 * slope[1:])
        else:
            vmid = w * vf[:-1] + (1.0 - w) * vf[1:]
        vdon = xp.concatenate([vf[:1], vmid, vf[-1:]])
        mom_c = qc * vdon
        # faces 0..N: derivada do fluxo de momento entre centros vizinhos
        dmom = (mom_c[1:] - mom_c[:-1]) / dxf
        dP = (Pc[1:] - Pc[:-1]) / dxf
        dcorr = (dPc[1:] - dPc[:-1]) / dxf
        dq = -dmom - Area * alpha_f * dP - Area * dcorr - Gamma_f - G * rho_f * alpha_f * Area * sin_t
        if self.art_visc > 0 and comp_c is not None:
            # difusão de q entre faces vizinhas através da célula k (0-based, entre as
            # faces k e k+1): F_k = nu_k (q_{k+1} - q_k)/h_k, nu_k = art_visc c_k h_k
            c_c = 1.0 / xp.sqrt(xp.maximum(comp_c[1 : N + 1], 1e-300))  # velocidade do som nas células 1..N
            F = self.art_visc * c_c * (qf[1:] - qf[:-1])
            dq = bk.add(dq, slice(1, N), (F[1:] - F[:-1]) / dxf[1:N])
        return dq, dmom

    def _mass_primitive(self, Y, qL, qG, tc, limit):
        """Taxas (dalphaG/dt, dPT/dt) das células 1..N na forma primitiva, com o
        limitador suave de fração de área nos extremos se `limit`."""
        xp = self.xp
        N = self.N
        model = self.model
        p = model.params
        Area = tc["Area"]
        ic = tc["inp"]
        h = self._h_x
        dqL = (qL[1:] - qL[:-1]) / h
        dqG = (qG[1:] - qG[:-1]) / h
        rhoL, rhoG, CL, CG = ic.rhoL[1 : N + 1], ic.rhoG[1 : N + 1], ic.CL[1 : N + 1], ic.CG[1 : N + 1]
        # coeficientes de massa com a fração de gás REAL (não cortada em alpha_min):
        # quando alphaG -> 0 a compressibilidade do gás desaparece (rG -> 0) e a
        # acumulação de líquido vai para a pressão (compressibilidade do líquido);
        # assim a fração de gás não ultrapassa os limites e a massa é conservada
        fl = model.fluid
        aGr = xp.clip(Y[1 : N + 1, SLOT_ALPHA], 0.0, 1.0)
        aLr = 1.0 - aGr
        auxr = aLr / fl.gammaL + aGr / fl.gammaG
        rL = (aLr / fl.gammaL) / auxr
        rG = (aGr / fl.gammaG) / auxr
        rL2 = (1.0 / fl.gammaL) / auxr
        rG2 = (1.0 / fl.gammaG) / auxr
        F1 = -rG / (Area * rhoL)
        F2 = rL / (Area * rhoG)
        F3 = rL2 / (Area * p.PR * CL)
        F4 = rG2 / (Area * p.PR * CG)
        daG = -(F1 * dqL + F2 * dqG)
        dPT = -(F3 * dqL + F4 * dqG)
        # limitador suave de fração de área nos extremos (ligado por padrão: sem ele
        # o Newton do integrador estagna com estados presos no limite físico; a
        # relaxação da vazão da fase que desaparece complementa)
        if limit:
            # guarda de consistência muito fina junto a 0 e 1 (banda limit_band):
            # anula a derivada para fora do intervalo físico; a perda de massa é
            # desprezível porque a relaxação já anula o fluxo da fase ausente
            aG = Y[1 : N + 1, SLOT_ALPHA]
            d = self.limit_band
            r_lo = xp.clip(aG / d, 0.0, 1.0)
            r_hi = xp.clip((1.0 - aG) / d, 0.0, 1.0)
            daG = xp.where(daG < 0, daG * r_lo, daG * r_hi)
        return daG, dPT

    def _mass_rates(self, Y, Yc, qL, qG, tc):
        """(dalphaG/dt, dPT/dt) das células: fluxos limitados (modo conservativo) ou forma
        primitiva com limitador."""
        N = self.N
        if self.conservative:
            # taxas das massas por volume (mG, mL) nas células: o integrador usa o
            # mapa T(x) = (rhoG alphaG, rhoL alphaL) no resíduo (balanço exato).
            # Fluxos limitados pela fase disponível na célula doadora: o mesmo fluxo
            # entra no balanço das duas células vizinhas (conservação exata) e se
            # anula quando a doadora não tem a fase (positividade sem limitador de alpha)
            Area, h = tc["Area"], self._h_x
            FL, FG = self.mass_fluxes(Yc[:, SLOT_ALPHA], qL, qG)
            return -(FG[1:] - FG[:-1]) / (Area * h), -(FL[1:] - FL[:-1]) / (Area * h)
        return self._mass_primitive(Y, qL, qG, tc, self.limit_alpha)

    # --- esquema --------------------------------------------------------------
    def rhs(self, Y):
        Y = self.bk.asarray(Y)
        Yc, Ycc, Yf = self._cell_face_states(Y)
        terms_c = self.model.conservative_terms(Ycc, self._thn_x)
        terms_f = self.model.conservative_terms(Yf, self._thf_x)
        return self._rhs_phases(Y, Yc, Yf, terms_c, terms_f)

    def _rhs_phases(self, Y, Yc, Yf, tc, tf):
        raise NotImplementedError

    def _pack(self, daG, dPT, dq):
        """dY (n_nodes x 4) a partir das taxas das células e das faces ({slot: dq})."""
        xp, bk = self.xp, self.bk
        N = self.N
        dY = xp.zeros((N + 2, 4))
        if daG is not None:
            dY = bk.set(dY, (slice(1, N + 1), SLOT_ALPHA), daG)
        dY = bk.set(dY, (slice(1, N + 1), SLOT_P), dPT)
        for s, v in dq.items():
            dY = bk.set(dY, (slice(0, N + 1), s), v)
        return dY


class StaggeredTwoFluidFV(StaggeredFVBase):
    """Modelo a dois fluidos: massas (primitivas ou conservativas), momento por fase
    com upwind doador, massa virtual, relaxação da fase que desaparece."""

    def _rhs_phases(self, Y, Yc, Yf, tc, tf):
        xp, bk = self.xp, self.bk
        N = self.N
        p = self.model.params
        Area = tc["Area"]
        ic, iff = tc["inp"], tf["inp"]
        Sc, Sf = tc["S"], tf["S"]

        qL = Y[0 : N + 1, SLOT_QL]
        qG = Y[0 : N + 1, SLOT_QG]
        daG, dPT = self._mass_rates(Y, Yc, qL, qG, tc)

        # --- momento
        Pc = tc["P"]
        # fração de área nas faces: upwind pelo sentido do escoamento de cada fase, média,
        # ou reconstrução MUSCL (minmod) de 2ª ordem a partir da célula doadora
        if self.alpha_face in ("upwind", "muscl"):
            aG_c = ic.alphaG
            wL = self._switch(qL, "L")
            wG = self._switch(qG, "G")
            aG_left, aG_right = self._face_states(aG_c)
            aG_fL = wL * aG_left + (1.0 - wL) * aG_right
            aG_fG = wG * aG_left + (1.0 - wG) * aG_right
        else:
            aG_fL = aG_fG = iff.alphaG
        aL_fL = 1.0 - aG_fL
        # correções de pressão: gradiente nulo nas células fantasmas (os valores
        # de contorno de alphaG/P não definem um estado interno consistente)
        corrL = ic.alphaL * Sc.dPL
        corrG = ic.alphaG * Sc.dPG
        corrL = bk.set(bk.set(corrL, 0, corrL[1]), -1, corrL[-2])
        corrG = bk.set(bk.set(corrG, 0, corrG[1]), -1, corrG[-2])
        dqL, dmomL = self._momentum(qL, iff.rhoL, aL_fL, Pc, corrL, Sf.GammaL, Area, "L", comp_c=ic.CL)
        dqG, dmomG = self._momentum(qG, iff.rhoG, aG_fG, Pc, corrG, Sf.GammaG, Area, "G", comp_c=ic.CG)
        if p.virtual_mass > 0:
            # massa virtual (Drew & Lahey): IG aG = FG - K (aG - aL), IL aL = FL + K (aG - aL), com as
            # acelerações materiais a = (dq/dt + d(q v)/dx)/I, I = rho alpha A e K = C_vm w alphaG rhoL A
            # (w = indicador de bolhas); resolvido face a face (2x2) e devolvido na forma dq/dt
            IL = iff.rhoL * aL_fL * Area
            IG = iff.rhoG * aG_fG * Area
            K = p.virtual_mass * 0.5 * (tc["vm_weight"][:-1] + tc["vm_weight"][1:]) * aG_fG * iff.rhoL * Area
            FL = dqL + dmomL
            FG = dqG + dmomG
            if p.vm_lambda != 2.0:
                # forma objetiva de Drew & Lahey: força sobre o gás
                # -K (vG - vL) [(lambda - 2) dvG/dx + (1 - lambda) dvL/dx] (reação sobre o líquido)
                vG_f = qG / (iff.rhoG * aG_fG * Area)
                vL_f = qL / (iff.rhoL * aL_fL * Area)
                xf = self._xf_x
                dvG = xp.gradient(vG_f, xf)
                dvL = xp.gradient(vL_f, xf)
                extra = -K * (vG_f - vL_f) * ((p.vm_lambda - 2.0) * dvG + (1.0 - p.vm_lambda) * dvL)
                FG = FG + extra
                FL = FL - extra
            det = IG * IL + K * (IG + IL)
            aG_acc = (FG * (IL + K) + K * FL) / det
            aL_acc = (FL * (IG + K) + K * FG) / det
            dqG = IG * aG_acc - dmomG
            dqL = IL * aL_acc - dmomL
        # fase que desaparece: relaxa a sua vazão para a condição sem escorregamento
        # (v_fase = v_outra fase), evitando velocidades enormes quando alpha -> 0
        tau = self.relax_time
        if tau > 0:
            dband = self.alpha_band
            aG_f = iff.alphaG  # valor cortado (fechamentos)
            # fração REAL da célula doadora (upwind pelo sentido da vazão de cada fase):
            # uma célula sem gás não pode ser drenada de gás
            # (chaveamento bem mais abrupto aqui: a mistura suave "vazaria" parte da
            # fração da célula vizinha e sustentaria fluxo de uma fase ausente)
            aG_c_raw = xp.clip(Yc[:, SLOT_ALPHA], 0.0, 1.0)
            wGs = self._switch(qG, "G", frac=0.02, floor_scale=0.05)
            wLs = self._switch(qL, "L", frac=0.02, floor_scale=0.05)
            aG_raw = wGs * aG_c_raw[:-1] + (1.0 - wGs) * aG_c_raw[1:]
            aL_raw = 1.0 - (wLs * aG_c_raw[:-1] + (1.0 - wLs) * aG_c_raw[1:])
            # pesos pela fração REAL da célula doadora: o fluxo da fase ausente na
            # célula a montante é relaxado para o valor sem escorregamento (-> 0)
            wG = xp.clip(aG_raw / dband, 0.0, 1.0)  # 1 longe do limite
            wL = xp.clip(aL_raw / dband, 0.0, 1.0)
            if self.relax_ramp == "smooth":
                # rampa C1 (smoothstep; issue 052): a rampa linear tem quinas em 0 e em dband
                wG = wG * wG * (3.0 - 2.0 * wG)
                wL = wL * wL * (3.0 - 2.0 * wL)
            vL_f = qL / (iff.rhoL * (1.0 - aG_f) * Area)
            vG_f = qG / (iff.rhoG * aG_f * Area)
            if self.relax_target == "noslip":
                # fase virtual sem escorregamento: fração doadora com piso alpha_min (NÃO a
                # média da face, que numa interface vale dezenas de vezes a fração doadora)
                a_min = p.alpha_min
                qG_ns = iff.rhoG * xp.maximum(aG_raw, a_min) * Area * vL_f
                qL_ns = iff.rhoL * xp.maximum(aL_raw, a_min) * Area * vG_f
            else:
                qG_ns = iff.rhoG * aG_raw * Area * vL_f
                qL_ns = iff.rhoL * aL_raw * Area * vG_f
            dqG = wG * dqG + (1.0 - wG) * (qG_ns - qG) / tau
            dqL = wL * dqL + (1.0 - wL) * (qL_ns - qL) / tau
        return self._pack(daG, dPT, {SLOT_QL: dqL, SLOT_QG: dqG})


class StaggeredDriftFluxFV(StaggeredFVBase):
    """Modelo de deriva (Beggs & Brill) no esquema escalonado.

    Células: hold-up HG e pressão (equações de massa das duas fases, iguais às do
    modelo a dois fluidos, conservativas ou primitivas). Faces: as vazões qL e qG
    são estados; a soma qM = qL + qG obedece à equação de momento da mistura da
    tese (fluxo rhoS vM^2 com velocidade doadora, gradiente de pressão, atrito
    bifásico e gravidade) e a divisão entre as fases obedece à relação de
    escorregamento HG_BB(P, qL, qG) = HG da face (hold-up doador das células),
    imposta na forma diferenciada no tempo com estabilização de Baumgarte
    (constante drift_relax): d/dt[HG_BB - HG_f] = -(HG_BB - HG_f)/tau. É a mesma
    ideia do modelo nodal da tese (hold-up algébrico diferenciado no tempo pela
    matriz MA), sem matriz de massa nem inversão da correlação."""

    def prepare_ic(self, Y):
        """Condição inicial nodal -> estado do esquema: o hold-up das células vem da
        correlação (as CI nodais da tese não o carregam)."""
        Y = np.asarray(Y, float).copy()
        if self.ic_holdup == "correlation":
            N = self.N
            Yc = Y[0 : N + 2].copy()
            for s_, _ in self.phases:
                Yc[1 : N + 1, s_] = 0.5 * (Y[0:N, s_] + Y[1 : N + 1, s_])
            tc = self.model.conservative_terms(Yc, self.theta_nodes)
            Y[0 : N + 2, SLOT_ALPHA] = self.bk.to_numpy(tc["HG"])
        return Y

    def _rhs_phases(self, Y, Yc, Yf, tc, tf):
        xp = self.xp
        N = self.N
        p = self.model.params
        Area = tc["Area"]
        qL = Y[0 : N + 1, SLOT_QL]
        qG = Y[0 : N + 1, SLOT_QG]
        qM = qL + qG
        # --- massas (iguais ao modelo a dois fluidos)
        daG, dPT = self._mass_rates(Y, Yc, qL, qG, tc)
        daG_c, dPT_c = self._mass_primitive(Y, qL, qG, tc, limit=False)  # taxas primitivas (restrição)
        # --- momento da mistura nas faces
        ones_f = xp.ones(N + 1)
        Gamma_f = xp.pi * p.D * tf["tension"]
        dqM, _ = self._momentum(
            qM, tf["rhoS"], ones_f, tc["P"], xp.zeros(N + 2), Gamma_f, Area, None, comp_c=tc["C_mix"]
        )
        # --- divisão entre as fases: restrição de escorregamento diferenciada
        HG_c = xp.clip(Yc[:, SLOT_ALPHA], 0.0, 1.0)
        w = self._switch(qM, None)  # peso doador pela vazão da mistura
        HG_f = w * HG_c[:-1] + (1.0 - w) * HG_c[1:]
        dHG_c = xp.concatenate([xp.zeros(1), daG_c, xp.zeros(1)])  # fantasmas: taxa nula
        dPT_cf = xp.concatenate([xp.zeros(1), dPT_c, xp.zeros(1)])
        dHG_f = w * dHG_c[:-1] + (1.0 - w) * dHG_c[1:]
        dP_f = 0.5 * (dPT_cf[:-1] + dPT_cf[1:]) * p.PR
        den = tf["dHGdqG"] - tf["dHGdqL"]  # > 0 (mais gás -> mais HG; mais líquido -> menos HG)
        den = xp.maximum(den, self.drift_den_min)
        g = tf["HG"] - HG_f
        dqG = (dHG_f - tf["dHGdP"] * dP_f - tf["dHGdqL"] * dqM - g / self.drift_relax) / den
        return self._pack(daG, dPT, {SLOT_QG: dqG, SLOT_QL: dqM - dqG})


class StaggeredSinglePhaseFV(StaggeredFVBase):
    """Modelos monofásicos: pressão nas células, vazão nas faces, momento com velocidade
    doadora (ou MUSCL)."""

    two_phase = False
    cell_slots = (SLOT_P,)

    def _rhs_phases(self, Y, Yc, Yf, tc, tf):
        xp = self.xp
        N = self.N
        p = self.model.params
        slot_q = self.phases[0][0]
        Area = tc["Area"]
        h = self._h_x
        q = Y[0 : N + 1, slot_q]
        dq_dx = (q[1:] - q[:-1]) / h
        C = tc["C"][1 : N + 1]
        dPT = -(1.0 / Area / C / p.PR) * dq_dx
        ones_f = xp.ones(N + 1)
        dq, _ = self._momentum(q, tf["rho"], ones_f, tc["P"], xp.zeros(N + 2), tf["Gamma"], Area)
        return self._pack(None, dPT, {slot_q: dq})


_STAGGERED_BY_MODEL = {
    "twofluid": StaggeredTwoFluidFV,
    "beggs_brill": StaggeredDriftFluxFV,
    "liquid": StaggeredSinglePhaseFV,
    "gas": StaggeredSinglePhaseFV,
}


def staggered_scheme(model, mesh, settings=None):
    """Esquema escalonado (fv_upwind) concreto para o modelo do duto."""
    try:
        cls = _STAGGERED_BY_MODEL[model.name]
    except KeyError:
        raise NotImplementedError(f"fv_upwind com o modelo {model.name}") from None
    return cls(model, mesh, settings)


def _mass_map(scheme, Y, direction):
    """T (direction='to_mass') ou T^-1 ('to_primitive') nas células de Y (n_nodes x 4);
    numpy (lado do integrador)."""
    Y = np.asarray(Y, float).copy()
    N = scheme.N
    fl = scheme.model.fluid
    PR = scheme.model.params.PR
    if direction == "to_mass":
        P = np.maximum(Y[1 : N + 1, 1] * PR, 1e3)
        # sem corte em [0, 1] (cópia: a coluna é sobrescrita logo abaixo): T é suave e
        # invertível também num pequeno undershoot/overshoot de alphaG, desde que
        # alphaG > -gammaG/gammaL (ali o determinante de dT/dx se anula)
        aG = Y[1 : N + 1, 0].copy()
        rhoL, rhoG, _, _ = fl.props(P)
        Y[1 : N + 1, 0] = rhoG * aG
        Y[1 : N + 1, 1] = rhoL * (1.0 - aG)
    else:
        mG = Y[1 : N + 1, 0]
        mL = Y[1 : N + 1, 1]
        P = solve_pressure(fl, mL, mG, P0=None, clip=False, bk=NUMPY)
        rhoL, rhoG, _, _ = fl.props(P)
        Y[1 : N + 1, 0] = mG / rhoG
        Y[1 : N + 1, 1] = P / PR
    return Y


class StaggeredTwoFluidConsFV(StaggeredTwoFluidFV):
    """Variante conservativa: os estados das células são as massas por volume
    mG = rhoG alphaG e mL = rhoL alphaL (slots 0 e 1, respectivamente), de modo
    que o balanço de massa discreto por célula é exato em cada passo de tempo:

        d mG/dt = -(qG_{i} - qG_{i-1}) / (A h_i),   d mL/dt = -(qL_{i} - qL_{i-1}) / (A h_i)

    alphaG e P são recuperados algebricamente de (mL, mG) resolvendo
    mL/rhoL(P) + mG/rhoG(P) = 1 (Newton escalar por célula). Os nós de contorno
    (0 e N+1) continuam com a semântica [alphaG, PT] para a rede. A equação de
    momento, os fechamentos e o tratamento da fase que desaparece são os da
    classe base; o limitador de fração de gás não é necessário (massas >= 0).
    """

    name = "fv_upwind_cons"

    @property
    def inlet_P_node(self):
        return 0

    @property
    def outlet_P_node(self):
        return self.N + 1

    # --- conversões ---------------------------------------------------------
    def _solve_P(self, mL, mG, P0=None, clip=True):
        """Pressão (Pa) tal que mL/rhoL(P) + mG/rhoG(P) = 1 (ver `solve_pressure`)."""
        return solve_pressure(self.model.fluid, mL, mG, P0=P0, clip=clip, bk=self.bk)

    def to_primitive(self, Y):
        """Arranjo nodal com [alphaG, PT, qL, qG] (células convertidas de massas)."""
        xp, bk = self.xp, self.bk
        Y = bk.asarray(Y)
        N = self.N
        PR = self.model.params.PR
        mG = Y[1 : N + 1, 0]
        mL = Y[1 : N + 1, 1]
        P = self._solve_P(mL, mG)
        rhoL, rhoG, _, _ = self.model.fluid.props(P)
        aG = xp.clip(xp.maximum(mG, 0.0) / rhoG, 0.0, 1.0)
        Yp = bk.set(Y, (slice(1, N + 1), 0), aG)
        return bk.set(Yp, (slice(1, N + 1), 1), P / PR)

    def from_primitive(self, Yp):
        """Inverso de to_primitive: células [alphaG, PT] -> [mG, mL]."""
        xp, bk = self.xp, self.bk
        Yp = bk.asarray(Yp)
        N = self.N
        PR = self.model.params.PR
        P = xp.maximum(Yp[1 : N + 1, 1] * PR, 1e3)
        aG = xp.clip(Yp[1 : N + 1, 0], 0.0, 1.0)
        rhoL, rhoG, _, _ = self.model.fluid.props(P)
        Y = bk.set(Yp, (slice(1, N + 1), 0), rhoG * aG)
        return bk.set(Y, (slice(1, N + 1), 1), rhoL * (1.0 - aG))

    def fill_derived(self, Y):
        bk = self.bk
        N = self.N
        # nós de contorno recebem alphaG/PT convertidos das células vizinhas
        # (a menos que Dirichlet); vazões como na classe base
        Yp_cells = self.to_primitive(Y)
        for s_ in (SLOT_ALPHA, SLOT_P):
            if self.bc_in[s_] != BC_DIRICHLET:
                Y = bk.set(Y, (0, s_), Yp_cells[1, s_])
            if self.bc_out[s_] != BC_DIRICHLET:
                Y = bk.set(Y, (N + 1, s_), Yp_cells[N, s_])
        return self._fill_flows(Y)

    def outputs(self, Y):
        return super().outputs(self.to_primitive(Y))

    # --- esquema --------------------------------------------------------------
    def rhs(self, Y):
        if self.model.name != "twofluid":
            raise NotImplementedError("fv_upwind_cons só está implementado para o modelo a dois fluidos")
        bk = self.bk
        Y = bk.asarray(Y)
        Yp = self.to_primitive(Y)  # células em variáveis primitivas (contornos já preenchidos)
        dY = super().rhs(Yp)  # momento (+ relaxação) com as variáveis primitivas
        # substitui as equações de massa pela forma conservativa
        N = self.N
        Area = self.xp.pi * self.model.params.D**2 / 4.0
        qL = Y[0 : N + 1, SLOT_QL]
        qG = Y[0 : N + 1, SLOT_QG]
        dY = bk.set(dY, (slice(1, N + 1), 0), -(qG[1:] - qG[:-1]) / (Area * self._h_x))
        return bk.set(dY, (slice(1, N + 1), 1), -(qL[1:] - qL[:-1]) / (Area * self._h_x))


# nomes anteriores
StaggeredFVScheme = StaggeredFVBase
StaggeredFVConsScheme = StaggeredTwoFluidConsFV
