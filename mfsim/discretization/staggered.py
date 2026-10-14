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
"""

from __future__ import annotations

import numpy as np

from ..models.base import G, SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
from ..network.data import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN
from .base import PipeScheme


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


class StaggeredFVScheme(PipeScheme):
    name = "fv_upwind"

    def __init__(self, model, mesh, settings=None):
        super().__init__(model, mesh, settings)
        self.bc_in = tuple(self.settings.get("bc_in", (1, 1, 2, 2)))
        self.bc_out = tuple(self.settings.get("bc_out", (1, 2, 1, 1)))
        self.order = int(self.settings.get("fv_order", 1))     # 1 = upwind, 2 = MUSCL/minmod
        self.alpha_face = self.settings.get("alpha_face", "upwind")  # upwind | central | muscl (reconstrução TVD/minmod)
        self.q_switch = float(self.settings.get("q_switch", 1e-3))   # kg/s, largura do chaveamento upwind
        self.q_switch_L = float(self.settings.get("q_switch_L", self.q_switch))
        self.q_switch_G = float(self.settings.get("q_switch_G", 1e-2 * self.q_switch))
        self.switch_frac = float(self.settings.get("switch_frac", 0.2))
        self.alpha_band = float(self.settings.get("alpha_band", 0.02))  # faixa de transição junto aos limites de alpha
        self.relax_time = float(self.settings.get("relax_time", 0.05))   # s, relaxação da fase que desaparece
        # resíduo de massa conservativo (massas por volume): só no modelo a dois fluidos; nos
        # monofásicos a equação da pressão já é o balanço de massa da única fase
        self.conservative = bool(self.settings.get("conservative", False)) and model.name in ("twofluid", "beggs_brill")
        self.limit_alpha = bool(self.settings.get("limit_alpha", not self.conservative))
        self.limit_band = float(self.settings.get("limit_band", 1e-4))
        # modo conservativo: banda (em alpha) do limitador de fluxo da fase que some
        # na célula doadora (positividade exata das massas sem perder conservação)
        self.flux_band = float(self.settings.get("flux_band", 1e-3))
        self.flux_eps = float(self.settings.get("flux_eps", 0.05))      # largura do chaveamento doador (x q_switch)
        self.flux_limit = bool(self.settings.get("flux_limit", True))   # liga o limitador de fluxo (modo conservativo)
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
        if self.relax_target not in ("raw", "noslip"):
            raise ValueError(f"relax_target: {self.relax_target}")
        if self.relax_target != "raw" and not self.conservative:
            raise ValueError("relax_target='noslip' exige o modo conservativo (limitador de fluxo)")
        N = self.N
        h = self.h
        # centros das células e faces
        s = np.asarray(mesh["s"], float)
        self.x_faces = s                                  # N+1 faces
        self.x_cells = 0.5 * (s[:-1] + s[1:])             # N células
        self.dx_faces = np.empty(N + 1)                   # distância entre centros vizinhos à face
        self.dx_faces[0] = 0.5 * h[0]
        self.dx_faces[-1] = 0.5 * h[-1]
        self.dx_faces[1:-1] = 0.5 * (h[:-1] + h[1:])
        th = self.theta_elem
        self.theta_faces = np.empty(N + 1)
        self.theta_faces[0] = th[0]
        self.theta_faces[-1] = th[-1]
        self.theta_faces[1:-1] = 0.5 * (th[:-1] + th[1:])
        self.phases = self._phases()
        # estados de célula: [alphaG, PT] nos modelos bifásicos (no modelo de deriva o
        # hold-up HG é estado de célula, embora não seja estado do modelo nodal), só PT nos
        # monofásicos
        self.cell_slots = (SLOT_ALPHA, SLOT_P) if model.name in ("twofluid", "beggs_brill") else (SLOT_P,)
        # modelo de deriva: constante de tempo da estabilização (Baumgarte) da restrição
        # HG_BB(P, qL, qG) = HG das células nas faces, e piso do coeficiente dHG/dqG - dHG/dqL
        self.drift_relax = float(self.settings.get("drift_relax", 0.05))
        self.drift_den_min = float(self.settings.get("drift_den_min", 1e-3))
        # hold-up inicial das células no modelo de deriva: "correlation" (das vazões e pressão
        # da CI nodal, que na tese não carrega hold-up) ou "given" (coluna alphaG da CI)
        self.ic_holdup = str(self.settings.get("ic_holdup", "correlation"))

    def _switch(self, q, phase=None, frac=None, floor_scale=1.0):
        """Peso suave do lado a montante: ~1 se q >> 0, ~0 se q << 0.

        A largura do chaveamento é adaptativa: uma fração (switch_frac) do maior
        |q| entre as faces vizinhas, com um piso por fase (q_switch_L / q_switch_G)
        -- assim o chaveamento é suave em relação às vazões locais tanto no
        escoamento lento quanto nos transientes rápidos (blowdown)."""
        q = np.asarray(q, float)
        floor = {"L": self.q_switch_L, "G": self.q_switch_G}.get(phase, self.q_switch)
        aq = np.abs(q)
        if aq.size > 2:
            local = aq.copy()
            np.maximum(local[1:], aq[:-1], out=local[1:])       # vizinho à esquerda
            np.maximum(local[:-1], aq[1:], out=local[:-1])      # vizinho à direita
        else:
            local = aq
        eps = (self.switch_frac if frac is None else frac) * local + floor * floor_scale
        return 0.5 * (1.0 + np.tanh(q / eps))

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
            mask[1:self.N + 1, s] = True
        for s, _ in self.phases:
            mask[0:self.N + 1, s] = True
            if self.bc_in[s] != BC_ABSENT:
                mask[0, s] = False
            if self.bc_out[s] != BC_ABSENT:
                mask[self.N, s] = False
        return mask

    def prepare_ic(self, Y):
        """Condição inicial nodal -> estado do esquema: no modelo de deriva o hold-up das
        células vem da correlação (as CI nodais da tese não o carregam)."""
        Y = np.asarray(Y, float).copy()
        if self.model.name == "beggs_brill" and self.ic_holdup == "correlation":
            N = self.N
            Yc = Y[0:N + 2].copy()
            for s_, _ in self.phases:
                Yc[1:N + 1, s_] = 0.5 * (Y[0:N, s_] + Y[1:N + 1, s_])
            tc = self.model.conservative_terms(Yc, self.theta_nodes)
            Y[0:N + 2, SLOT_ALPHA] = tc["HG"]
        return Y

    def fill_derived(self, Y):
        N = self.N
        for s in self.cell_slots:
            if self.bc_in[s] != BC_DIRICHLET:
                Y[0, s] = Y[1, s]
            if self.bc_out[s] != BC_DIRICHLET:
                Y[N + 1, s] = Y[N, s]
        for s, _ in self.phases:
            if self.bc_out[s] == BC_DIRICHLET:
                Y[N, s] = Y[N + 1, s]        # valor imposto no nó de contorno -> face N
            elif self.bc_out[s] == BC_NEUMANN:
                Y[N, s] = Y[N - 1, s]
                Y[N + 1, s] = Y[N, s]
            else:
                Y[N + 1, s] = Y[N, s]
            if self.bc_in[s] == BC_NEUMANN:
                Y[0, s] = Y[1, s]
        return Y

    # --- esquema --------------------------------------------------------------
    def rhs(self, Y):
        Y = np.asarray(Y, float)
        N = self.N
        model = self.model
        p = model.params
        Area = np.pi * p.D**2 / 4.0
        h = self.h
        cells = slice(1, N + 1)

        # --- estados nas células (incluindo nós de contorno como "células fantasmas")
        Yc = Y[0:N + 2].copy()
        # vazões nas células fantasmas: cópia das faces de contorno (só para fechamentos)
        for s, _ in self.phases:
            Yc[0, s] = Y[0, s]
            Yc[N + 1, s] = Y[N, s]
        # para avaliar propriedades nas células, as vazões "de célula" são médias das faces
        Ycc = Yc.copy()
        for s, _ in self.phases:
            Ycc[1:N + 1, s] = 0.5 * (Y[0:N, s] + Y[1:N + 1, s])
        terms_c = model.conservative_terms(Ycc, self.theta_nodes)

        # --- estados nas faces (média das células vizinhas para alphaG, PT)
        Yf = np.empty((N + 1, 4))
        for s in (SLOT_ALPHA, SLOT_P):
            Yf[:, s] = 0.5 * (Yc[0:N + 1, s] + Yc[1:N + 2, s])
        for s, _ in self.phases:
            Yf[:, s] = Y[0:N + 1, s]
        # faces de contorno com alphaG imposto pelo vértice (Dirichlet): a fração da
        # face é a do lado doador (célula em saída, vértice em entrada), não a média —
        # o conteúdo do vértice só entra no duto quando o escoamento entra
        if SLOT_ALPHA in self.cell_slots:
            qm = Y[:, SLOT_QL] + Y[:, SLOT_QG]
            if self.bc_in[SLOT_ALPHA] == BC_DIRICHLET:
                w = self._switch(qm[0:1], "L", frac=0.02, floor_scale=0.05)[0]   # ~1 se entra (q > 0)
                Yf[0, SLOT_ALPHA] = w * Yc[0, SLOT_ALPHA] + (1.0 - w) * Yc[1, SLOT_ALPHA]
            if self.bc_out[SLOT_ALPHA] == BC_DIRICHLET:
                w = self._switch(qm[N:N + 1], "L", frac=0.02, floor_scale=0.05)[0]  # ~1 se sai (q > 0)
                Yf[N, SLOT_ALPHA] = w * Yc[N, SLOT_ALPHA] + (1.0 - w) * Yc[N + 1, SLOT_ALPHA]
        for s in (SLOT_ALPHA, SLOT_P, SLOT_QL, SLOT_QG):
            if s not in model.slots and s not in (SLOT_QL, SLOT_QG):
                Yf[:, s] = Yc[0, s]
        terms_f = model.conservative_terms(Yf, self.theta_faces)

        dY = np.zeros_like(Y)
        if model.name == "twofluid":
            self._rhs_twofluid(Y, Yc, Yf, terms_c, terms_f, dY)
        elif model.name == "beggs_brill":
            self._rhs_drift(Y, Yc, Yf, terms_c, terms_f, dY)
        else:
            self._rhs_single(Y, Yc, Yf, terms_c, terms_f, dY)
        return dY

    # -------------------------------------------------------------------------
    def _momentum(self, qf, rho_c, alpha_c, rho_f, alpha_f, Pc, dPc, Gamma_f, Area, theta_f, sin_t, dY_slot, phase=None, comp_c=None):
        """Equação de momento de uma fase nas faces.

        qf      : vazões nas faces (N+1)
        rho_c, alpha_c : densidade e fração de área nas células (N+2, com fantasmas)
        rho_f, alpha_f : idem nas faces (N+1)
        Pc      : pressão (Pa) nas células (N+2)
        dPc     : alpha*deltaP nas células (N+2)  (correção de pressão)
        Gamma_f : força de atrito por metro nas faces (N+1)
        """
        N = self.N
        h = self.h
        # velocidades nas faces
        vf = qf / (rho_f * alpha_f * Area)
        # fluxo de massa nos centros das células 1..N (interpolação) e nos fantasmas
        qc = np.empty(N + 2)
        qc[1:N + 1] = 0.5 * (qf[:-1] + qf[1:])
        qc[0] = qf[0]
        qc[N + 1] = qf[N]
        # velocidade donor no centro i: face i-1 se qc >= 0, senão face i
        # (chaveamento suavizado para manter o lado direito contínuo)
        vdon = np.empty(N + 2)
        w = self._switch(qc[1:N + 1], phase)
        vdon[1:N + 1] = w * vf[:-1] + (1.0 - w) * vf[1:]
        if self.order == 2:
            # correção MUSCL (minmod) sobre a velocidade donor
            dv = np.diff(vf)
            slope = np.zeros(N + 1)
            slope[1:-1] = _minmod(dv[:-1], dv[1:])
            vdon[1:N + 1] = w * (vf[:-1] + 0.5 * slope[:-1]) + (1.0 - w) * (vf[1:] - 0.5 * slope[1:])
        vdon[0] = vf[0]
        vdon[N + 1] = vf[N]
        mom_c = qc * vdon
        # faces 0..N: derivada do fluxo de momento entre centros vizinhos
        dmom = (mom_c[1:] - mom_c[:-1]) / self.dx_faces
        dP = (Pc[1:] - Pc[:-1]) / self.dx_faces
        dcorr = (dPc[1:] - dPc[:-1]) / self.dx_faces
        dq = -dmom - Area * alpha_f * dP - Area * dcorr - Gamma_f - G * rho_f * alpha_f * Area * sin_t
        if self.art_visc > 0 and comp_c is not None:
            # difusão de q entre faces vizinhas através da célula k (0-based, entre as
            # faces k e k+1): F_k = nu_k (q_{k+1} - q_k)/h_k, nu_k = art_visc c_k h_k
            c_c = 1.0 / np.sqrt(np.maximum(comp_c[1:N + 1], 1e-300))      # velocidade do som nas células 1..N
            F = self.art_visc * c_c * (qf[1:] - qf[:-1])
            dq[1:N] += (F[1:] - F[:-1]) / self.dx_faces[1:N]
        dY_slot[0:N + 1] = dq
        return dmom

    def _rhs_twofluid(self, Y, Yc, Yf, tc, tf, dY):
        N = self.N
        model = self.model
        p = model.params
        Area = tc["Area"]
        ic, iff = tc["inp"], tf["inp"]
        Sc, Sf = tc["S"], tf["S"]
        h = self.h
        sin_t = np.sin(self.theta_faces)

        qL = Y[0:N + 1, SLOT_QL]
        qG = Y[0:N + 1, SLOT_QG]

        if self.conservative:
            # taxas das massas por volume (mG, mL) nas células: o integrador usa o
            # mapa T(x) = (rhoG alphaG, rhoL alphaL) no resíduo (balanço exato).
            # Fluxos limitados pela fase disponível na célula doadora: o mesmo fluxo
            # entra no balanço das duas células vizinhas (conservação exata) e se
            # anula quando a doadora não tem a fase (positividade sem limitador de alpha)
            Area_ = tc["Area"]
            FL, FG = self.mass_fluxes(Yc[:, SLOT_ALPHA], qL, qG)
            dY[1:N + 1, SLOT_ALPHA] = -(FG[1:] - FG[:-1]) / (Area_ * h)
            dY[1:N + 1, SLOT_P] = -(FL[1:] - FL[:-1]) / (Area_ * h)
        else:
            self._mass_twofluid(Y, qL, qG, tc, dY)

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
        corrL[0], corrL[-1] = corrL[1], corrL[-2]
        corrG[0], corrG[-1] = corrG[1], corrG[-2]
        dmomL = self._momentum(qL, ic.rhoL, ic.alphaL, iff.rhoL, aL_fL, Pc, corrL,
                               Sf.GammaL, Area, self.theta_faces, sin_t, dY[:, SLOT_QL], "L", comp_c=ic.CL)
        dmomG = self._momentum(qG, ic.rhoG, ic.alphaG, iff.rhoG, aG_fG, Pc, corrG,
                               Sf.GammaG, Area, self.theta_faces, sin_t, dY[:, SLOT_QG], "G", comp_c=ic.CG)
        if p.virtual_mass > 0:
            # massa virtual (Drew & Lahey): IG aG = FG - K (aG - aL), IL aL = FL + K (aG - aL), com as
            # acelerações materiais a = (dq/dt + d(q v)/dx)/I, I = rho alpha A e K = C_vm w alphaG rhoL A
            # (w = indicador de bolhas); resolvido face a face (2x2) e devolvido na forma dq/dt
            IL = iff.rhoL * aL_fL * Area
            IG = iff.rhoG * aG_fG * Area
            K = p.virtual_mass * 0.5 * (tc["vm_weight"][:-1] + tc["vm_weight"][1:]) * aG_fG * iff.rhoL * Area
            FL = dY[0:N + 1, SLOT_QL] + dmomL
            FG = dY[0:N + 1, SLOT_QG] + dmomG
            if p.vm_lambda != 2.0:
                # forma objetiva de Drew & Lahey: força sobre o gás
                # -K (vG - vL) [(lambda - 2) dvG/dx + (1 - lambda) dvL/dx] (reação sobre o líquido)
                vG_f = qG / (iff.rhoG * aG_fG * Area)
                vL_f = qL / (iff.rhoL * aL_fL * Area)
                xf = self.x_faces
                dvG = np.gradient(vG_f, xf)
                dvL = np.gradient(vL_f, xf)
                extra = -K * (vG_f - vL_f) * ((p.vm_lambda - 2.0) * dvG + (1.0 - p.vm_lambda) * dvL)
                FG = FG + extra
                FL = FL - extra
            det = IG * IL + K * (IG + IL)
            aG_acc = (FG * (IL + K) + K * FL) / det
            aL_acc = (FL * (IG + K) + K * FG) / det
            dY[0:N + 1, SLOT_QG] = IG * aG_acc - dmomG
            dY[0:N + 1, SLOT_QL] = IL * aL_acc - dmomL
        # fase que desaparece: relaxa a sua vazão para a condição sem escorregamento
        # (v_fase = v_outra fase), evitando velocidades enormes quando alpha -> 0
        tau = self.relax_time
        if tau > 0:
            dband = self.alpha_band
            aG_f = iff.alphaG                                   # valor cortado (fechamentos)
            # fração REAL da célula doadora (upwind pelo sentido da vazão de cada fase):
            # uma célula sem gás não pode ser drenada de gás
            # (chaveamento bem mais abrupto aqui: a mistura suave "vazaria" parte da
            # fração da célula vizinha e sustentaria fluxo de uma fase ausente)
            aG_c_raw = np.clip(Yc[:, SLOT_ALPHA], 0.0, 1.0)
            wGs = self._switch(qG, "G", frac=0.02, floor_scale=0.05)
            wLs = self._switch(qL, "L", frac=0.02, floor_scale=0.05)
            aG_raw = wGs * aG_c_raw[:-1] + (1.0 - wGs) * aG_c_raw[1:]
            aL_raw = 1.0 - (wLs * aG_c_raw[:-1] + (1.0 - wLs) * aG_c_raw[1:])
            # pesos pela fração REAL da célula doadora: o fluxo da fase ausente na
            # célula a montante é relaxado para o valor sem escorregamento (-> 0)
            wG = np.clip(aG_raw / dband, 0.0, 1.0)          # 1 longe do limite
            wL = np.clip(aL_raw / dband, 0.0, 1.0)
            vL_f = qL / (iff.rhoL * (1.0 - aG_f) * Area)
            vG_f = qG / (iff.rhoG * aG_f * Area)
            if self.relax_target == "noslip":
                # fase virtual sem escorregamento: fração doadora com piso alpha_min (NÃO a
                # média da face, que numa interface vale dezenas de vezes a fração doadora)
                a_min = p.alpha_min
                qG_ns = iff.rhoG * np.maximum(aG_raw, a_min) * Area * vL_f
                qL_ns = iff.rhoL * np.maximum(aL_raw, a_min) * Area * vG_f
            else:
                qG_ns = iff.rhoG * aG_raw * Area * vL_f
                qL_ns = iff.rhoL * aL_raw * Area * vG_f
            dY[0:N + 1, SLOT_QG] = wG * dY[0:N + 1, SLOT_QG] + (1.0 - wG) * (qG_ns - qG) / tau
            dY[0:N + 1, SLOT_QL] = wL * dY[0:N + 1, SLOT_QL] + (1.0 - wL) * (qL_ns - qL) / tau

    def _face_states(self, a_c):
        """Valores de uma grandeza de célula (N+2, com fantasmas) extrapolados para as
        N+1 faces pelo lado esquerdo e pelo lado direito: constantes por célula
        (upwind, 1ª ordem) ou com reconstrução linear limitada por minmod (muscl, 2ª
        ordem TVD: o valor na face fica entre os das células vizinhas, sem extremos
        novos; nas células fantasmas o declive é nulo)."""
        a_c = np.asarray(a_c, float)
        if self.alpha_face != "muscl":
            return a_c[:-1], a_c[1:]
        da = np.diff(a_c)
        slope = np.zeros_like(a_c)
        slope[1:-1] = _minmod(da[:-1], da[1:])
        return a_c[:-1] + 0.5 * slope[:-1], a_c[1:] - 0.5 * slope[1:]

    def _flux_limiter(self, aG_c, q, phase):
        """Fator em [0, 1] do fluxo de massa de uma fase nas faces (modo conservativo).

        Vale ~1 quando a célula doadora (a montante pelo sinal de q, chaveamento
        bem estreito) tem a fase e cai a 0 quando não tem (fração < flux_band);
        nas células fantasmas o fator é 1 (o contorno define o fluxo)."""
        N = self.N
        if not self.flux_limit:
            return np.ones(N + 1)
        a = np.asarray(aG_c, float) if phase == "G" else 1.0 - np.asarray(aG_c, float)
        u = np.clip(a / self.flux_band, 0.0, 1.0)
        # rampa C1 (smoothstep): sem quina na borda da banda, onde as células
        # quase vazias tendem a se estabilizar (o Newton estagnaria numa quina)
        r = u * u * (3.0 - 2.0 * u)
        # células fantasmas: o contorno define o fluxo (r = 1), exceto quando a fração
        # de gás é imposta pelo vértice dinâmico (Dirichlet): aí o vértice é a célula
        # doadora e não pode ceder uma fase que não tem
        if self.bc_in[SLOT_ALPHA] != BC_DIRICHLET:
            r[0] = 1.0
        if self.bc_out[SLOT_ALPHA] != BC_DIRICHLET:
            r[N + 1] = 1.0
        eps = self.flux_eps * (self.q_switch_G if phase == "G" else self.q_switch_L)
        s = 0.5 * (1.0 + np.tanh(np.asarray(q, float) / eps))
        return s * r[:-1] + (1.0 - s) * r[1:]

    def mass_fluxes(self, aG_c, qL, qG):
        """Fluxos de massa (kg/s) de líquido e gás nas N+1 faces usados no balanço
        conservativo (vazões das faces vezes o limitador da célula doadora).
        aG_c: fração de gás nas N+2 células (com fantasmas)."""
        return qL * self._flux_limiter(aG_c, qL, "L"), qG * self._flux_limiter(aG_c, qG, "G")

    def mass_fluxes_nodal(self, Y):
        """Idem a partir do arranjo nodal completo Y (n_nodes x 4) do duto."""
        Y = np.asarray(Y, float)
        N = self.N
        return self.mass_fluxes(Y[0:N + 2, SLOT_ALPHA], Y[0:N + 1, SLOT_QL], Y[0:N + 1, SLOT_QG])

    def _mass_rates_primitive(self, Y, qL, qG, tc):
        """Taxas (dalphaG/dt, dPT/dt) das células 1..N na forma primitiva, sem limitador."""
        N = self.N
        model = self.model
        p = model.params
        Area = tc["Area"]
        ic = tc["inp"]
        h = self.h
        dqL = (qL[1:] - qL[:-1]) / h
        dqG = (qG[1:] - qG[:-1]) / h
        rhoL, rhoG, CL, CG = ic.rhoL[1:N + 1], ic.rhoG[1:N + 1], ic.CL[1:N + 1], ic.CG[1:N + 1]
        fl = model.fluid
        aGr = np.clip(Y[1:N + 1, SLOT_ALPHA], 0.0, 1.0)
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
        return -(F1 * dqL + F2 * dqG), -(F3 * dqL + F4 * dqG)

    def _mass_twofluid(self, Y, qL, qG, tc, dY):
        """Equações de massa nas células, forma primitiva (dalphaG/dt, dPT/dt)."""
        N = self.N
        model = self.model
        p = model.params
        Area = tc["Area"]
        ic = tc["inp"]
        h = self.h
        dqL = (qL[1:] - qL[:-1]) / h
        dqG = (qG[1:] - qG[:-1]) / h
        rhoL, rhoG, CL, CG = ic.rhoL[1:N + 1], ic.rhoG[1:N + 1], ic.CL[1:N + 1], ic.CG[1:N + 1]
        # coeficientes de massa com a fração de gás REAL (não cortada em alpha_min):
        # quando alphaG -> 0 a compressibilidade do gás desaparece (rG -> 0) e a
        # acumulação de líquido vai para a pressão (compressibilidade do líquido);
        # assim a fração de gás não ultrapassa os limites e a massa é conservada
        fl = model.fluid
        aGr = np.clip(Y[1:N + 1, SLOT_ALPHA], 0.0, 1.0)
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
        # relaxação da vazão da fase que desaparece, abaixo, complementa)
        if self.limit_alpha:
            # guarda de consistência muito fina junto a 0 e 1 (banda limit_band):
            # anula a derivada para fora do intervalo físico; a perda de massa é
            # desprezível porque a relaxação acima já anula o fluxo da fase ausente
            aG = Y[1:N + 1, SLOT_ALPHA]
            d = self.limit_band
            r_lo = np.clip(aG / d, 0.0, 1.0)
            r_hi = np.clip((1.0 - aG) / d, 0.0, 1.0)
            daG = np.where(daG < 0, daG * r_lo, daG * r_hi)
        dY[1:N + 1, SLOT_ALPHA] = daG
        dY[1:N + 1, SLOT_P] = dPT


    def _rhs_drift(self, Y, Yc, Yf, tc, tf, dY):
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
        N = self.N
        model = self.model
        p = model.params
        Area = tc["Area"]
        h = self.h
        sin_t = np.sin(self.theta_faces)
        qL = Y[0:N + 1, SLOT_QL]
        qG = Y[0:N + 1, SLOT_QG]
        qM = qL + qG
        # --- massas (iguais ao modelo a dois fluidos)
        if self.conservative:
            FL, FG = self.mass_fluxes(Yc[:, SLOT_ALPHA], qL, qG)
            dY[1:N + 1, SLOT_ALPHA] = -(FG[1:] - FG[:-1]) / (Area * h)
            dY[1:N + 1, SLOT_P] = -(FL[1:] - FL[:-1]) / (Area * h)
        else:
            self._mass_twofluid(Y, qL, qG, tc, dY)
        daG_c, dPT_c = self._mass_rates_primitive(Y, qL, qG, tc)     # taxas primitivas (restrição)
        # --- momento da mistura nas faces
        ones_c = np.ones(N + 2)
        ones_f = np.ones(N + 1)
        Gamma_f = np.pi * p.D * tf["tension"]
        dqM_full = np.zeros(N + 2)
        self._momentum(qM, tc["rhoS"], ones_c, tf["rhoS"], ones_f, tc["P"], np.zeros(N + 2),
                       Gamma_f, Area, self.theta_faces, sin_t, dqM_full, None, comp_c=tc["C_mix"])
        dqM = dqM_full[0:N + 1]
        # --- divisão entre as fases: restrição de escorregamento diferenciada
        HG_c = np.clip(Yc[:, SLOT_ALPHA], 0.0, 1.0)
        w = self._switch(qM, None)                                   # peso doador pela vazão da mistura
        HG_f = w * HG_c[:-1] + (1.0 - w) * HG_c[1:]
        dHG_c = np.zeros(N + 2)
        dHG_c[1:N + 1] = daG_c                                       # fantasmas: taxa nula
        dPT_cf = np.zeros(N + 2)
        dPT_cf[1:N + 1] = dPT_c
        dHG_f = w * dHG_c[:-1] + (1.0 - w) * dHG_c[1:]
        dP_f = 0.5 * (dPT_cf[:-1] + dPT_cf[1:]) * p.PR
        den = tf["dHGdqG"] - tf["dHGdqL"]                            # > 0 (mais gás -> mais HG; mais líquido -> menos HG)
        den = np.maximum(den, self.drift_den_min)
        g = tf["HG"] - HG_f
        dqG = (dHG_f - tf["dHGdP"] * dP_f - tf["dHGdqL"] * dqM - g / self.drift_relax) / den
        dY[0:N + 1, SLOT_QG] = dqG
        dY[0:N + 1, SLOT_QL] = dqM - dqG

    def _rhs_single(self, Y, Yc, Yf, tc, tf, dY):
        N = self.N
        model = self.model
        p = model.params
        slot_q = self.phases[0][0]
        Area = tc["Area"]
        h = self.h
        q = Y[0:N + 1, slot_q]
        dq = (q[1:] - q[:-1]) / h
        C = tc["C"][1:N + 1]
        dY[1:N + 1, SLOT_P] = -(1.0 / Area / C / p.PR) * dq
        ones_c = np.ones(N + 2)
        ones_f = np.ones(N + 1)
        self._momentum(q, tc["rho"], ones_c, tf["rho"], ones_f, tc["P"], np.zeros(N + 2),
                       tf["Gamma"], Area, self.theta_faces, np.sin(self.theta_faces), dY[:, slot_q])


def _mass_map(scheme, Y, direction):
    """T (direction='to_mass') ou T^-1 ('to_primitive') nas células de Y (n_nodes x 4)."""
    Y = np.asarray(Y, float).copy()
    N = scheme.N
    fl = scheme.model.fluid
    PR = scheme.model.params.PR
    if direction == "to_mass":
        P = np.maximum(Y[1:N + 1, 1] * PR, 1e3)
        # sem corte em [0, 1] (cópia: a coluna é sobrescrita logo abaixo): T é suave e
        # invertível também num pequeno undershoot/overshoot de alphaG, desde que
        # alphaG > -gammaG/gammaL (ali o determinante de dT/dx se anula)
        aG = Y[1:N + 1, 0].copy()
        rhoL, rhoG, _, _ = fl.props(P)
        Y[1:N + 1, 0] = rhoG * aG
        Y[1:N + 1, 1] = rhoL * (1.0 - aG)
    else:
        mG = Y[1:N + 1, 0]
        mL = Y[1:N + 1, 1]
        P = StaggeredFVConsScheme._solve_P(scheme, mL, mG, P0=None, clip=False)
        rhoL, rhoG, _, _ = fl.props(P)
        Y[1:N + 1, 0] = mG / rhoG
        Y[1:N + 1, 1] = P / PR
    return Y


class StaggeredFVConsScheme(StaggeredFVScheme):
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
        """Pressão (Pa) tal que mL/rhoL(P) + mG/rhoG(P) = 1.

        clip=False admite massas ligeiramente negativas (undershoot de alphaG no
        modo conservativo do esquema primitivo); a equação continua bem posta
        enquanto alphaG > -gammaG/gammaL."""
        fl = self.model.fluid
        mL = np.asarray(mL, float)
        mG = np.asarray(mG, float)
        if clip:
            mL = np.maximum(mL, 0.0)
            mG = np.maximum(mG, 0.0)
        if P0 is None:
            # estimativa inicial: pressão que satisfaz a fase líquida sozinha (boa quando
            # mG é pequeno) limitada a um intervalo razoável
            P0 = fl.P0 * np.clip(np.maximum(mL, 1e-3) / fl.rhoL0, 0.5, 2.0) ** fl.gammaL
        P = np.maximum(np.asarray(P0, float) * np.ones(mL.shape), 1e3)
        for _ in range(30):
            rhoL, rhoG, CL, CG = fl.props(P)
            f = mL / rhoL + mG / rhoG - 1.0
            df = -(mL / rhoL**2 * CL + mG / rhoG**2 * CG)
            df = np.where(np.abs(df) < 1e-300, -1e-300, df)
            step = f / df
            # amortece passos grandes (log-escala) e mantém P > 0
            step = np.clip(step, -0.5 * P, 0.5 * P)
            P = P - step
            if np.all(np.abs(step) <= 1e-9 * P):
                break
        return P

    def to_primitive(self, Y):
        """Arranjo nodal com [alphaG, PT, qL, qG] (células convertidas de massas)."""
        Y = np.asarray(Y, float)
        N = self.N
        fl = self.model.fluid
        PR = self.model.params.PR
        Yp = Y.copy()
        mG = Y[1:N + 1, 0]
        mL = Y[1:N + 1, 1]
        P = self._solve_P(mL, mG)
        rhoL, rhoG, _, _ = fl.props(P)
        aG = np.clip(np.maximum(mG, 0.0) / rhoG, 0.0, 1.0)
        Yp[1:N + 1, 0] = aG
        Yp[1:N + 1, 1] = P / PR
        return Yp

    def from_primitive(self, Yp):
        """Inverso de to_primitive: células [alphaG, PT] -> [mG, mL]."""
        Yp = np.asarray(Yp, float)
        N = self.N
        fl = self.model.fluid
        PR = self.model.params.PR
        Y = Yp.copy()
        P = np.maximum(Yp[1:N + 1, 1] * PR, 1e3)
        aG = np.clip(Yp[1:N + 1, 0], 0.0, 1.0)
        rhoL, rhoG, _, _ = fl.props(P)
        Y[1:N + 1, 0] = rhoG * aG
        Y[1:N + 1, 1] = rhoL * (1.0 - aG)
        return Y

    def fill_derived(self, Y):
        N = self.N
        # nós de contorno recebem alphaG/PT convertidos das células vizinhas
        # (a menos que Dirichlet); vazões como na classe base
        Yp_cells = self.to_primitive(Y)
        for s_ in (SLOT_ALPHA, SLOT_P):
            if self.bc_in[s_] != BC_DIRICHLET:
                Y[0, s_] = Yp_cells[1, s_]
            if self.bc_out[s_] != BC_DIRICHLET:
                Y[N + 1, s_] = Yp_cells[N, s_]
        for s_, _ in self.phases:
            if self.bc_out[s_] == BC_DIRICHLET:
                Y[N, s_] = Y[N + 1, s_]
            elif self.bc_out[s_] == BC_NEUMANN:
                Y[N, s_] = Y[N - 1, s_]
                Y[N + 1, s_] = Y[N, s_]
            else:
                Y[N + 1, s_] = Y[N, s_]
            if self.bc_in[s_] == BC_NEUMANN:
                Y[0, s_] = Y[1, s_]
        return Y

    def outputs(self, Y):
        return super().outputs(self.to_primitive(Y))

    # --- esquema --------------------------------------------------------------
    def rhs(self, Y):
        if self.model.name != "twofluid":
            raise NotImplementedError("fv_upwind_cons só está implementado para o modelo a dois fluidos")
        Y = np.asarray(Y, float)
        Yp = self.to_primitive(Y)          # células em variáveis primitivas (contornos já preenchidos)
        dY = super().rhs(Yp)               # momento (+ relaxação) com as variáveis primitivas
        # substitui as equações de massa pela forma conservativa
        N = self.N
        Area = np.pi * self.model.params.D**2 / 4.0
        qL = Y[0:N + 1, SLOT_QL]
        qG = Y[0:N + 1, SLOT_QG]
        dY[1:N + 1, 0] = -(qG[1:] - qG[:-1]) / (Area * self.h)
        dY[1:N + 1, 1] = -(qL[1:] - qL[:-1]) / (Area * self.h)
        return dY
