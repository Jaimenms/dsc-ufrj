"""Modelo bifásico a dois fluidos com 4 equações (modelo_duto.m).

Equações (forma quase-linear, x = coordenada axial):

  d(alphaG)/dt = -(F1 dqL/dx + F2 dqG/dx)
  d(PT)/dt     = -(F3 dqL/dx + F4 dqG/dx)
  d(qL)/dt     = -(F5 dalphaG/dx + F6 dPT/dx + F7 dqL/dx) - F11
  d(qG)/dt     = -(F8 dalphaG/dx + F9 dPT/dx + F10 dqG/dx) - F12

obtidas dos balanços de massa e de momento por fase com equação de estado
politrópica e termos de fechamento (Gamma_N, dP_N) dependentes do padrão
de escoamento.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from ..closures import (
    REGIME_AUTO,
    ClosureInputs,
    blend_closures,
    equilibrium_fractions,
    fixed_regime,
    identify_regime,
)
from ..closures.regime import RegimeFlags, closures_used
from ..geometry import stratified_level
from .base import G, PipeModel, PipeParams


class TwoFluidModel(PipeModel):
    name = "twofluid"
    slots = (0, 1, 2, 3)
    A_pattern = np.array([[0, 0, 1, 2], [0, 0, 3, 4], [5, 6, 7, 0], [8, 9, 0, 10]])
    B_pattern = np.array([0, 0, 11, 12])
    nF = 12

    def local_state(self, y, theta):
        """Variáveis locais e entradas de fechamento a partir do estado nodal."""
        xp, bk = self.xp, self.bk
        p = self.params
        aG, PT, qL, qG = self.unpack(y)
        aG = xp.clip(aG, p.alpha_min, p.alpha_max)
        aL = 1.0 - aG
        n = aG.shape[0]
        P = p.PR * PT
        rhoL, rhoG, CL, CG = self.fluid.props(P)
        hD, beta = stratified_level(aG, bk=bk)
        ones = xp.ones(n)
        D = p.D * ones
        Area = xp.pi * p.D**2 / 4.0
        AreaG = Area * aG
        AreaL = Area * aL
        vLS = qL / (rhoL * Area)
        vGS = qG / (rhoG * Area)
        vL = vLS / aL
        vG = vGS / aG
        inp = ClosureInputs(
            alphaG=aG,
            alphaL=aL,
            vL=vL,
            vLS=vLS,
            vG=vG,
            vGS=vGS,
            AreaL=AreaL,
            AreaG=AreaG,
            D=D,
            eps=p.eps * ones,
            theta=xp.asarray(theta, dtype=float) * ones,
            rhoL=rhoL,
            rhoG=rhoG,
            muL=self.fluid.muL * ones,
            muG=self.fluid.muG * ones,
            CL=CL,
            CG=CG,
            sigma=self.fluid.sigma * ones,
            beta=beta,
            hD=hD,
            fi_mult=p.fi_mult * ones,
            fi_vgt=p.fi_vgt * ones,
            bubble_drag=(1.0 if p.bubble_drag == "fisico" else 0.0) * ones,
            slug_frequency=None if p.slug_frequency == "thesis" else ones,
            slug_unit=None if p.slug_unit in ("thesis", "film") else ones,
            slug_film=ones if p.slug_unit == "film" else None,
            smooth_kinks=ones if p.kinks == "smooth" else None,
        )
        return inp, P, Area

    @property
    def used_closures(self):
        """Conjunto estático de fechamentos que podem ter peso não nulo (poda no jit)."""
        p = self.params
        return closures_used(p.regime, p.regime_vertical, p.low_void_regime, p.high_void_regime)

    @property
    def blend_threshold(self):
        # com mistura por ângulo/fração os pesos pequenos também contam (sem o corte
        # de 0.1, que criaria um salto na borda da faixa de transição)
        p = self.params
        blended = (
            p.regime != REGIME_AUTO
            and ((p.regime_vertical and p.vertical_blend > 0) or p.low_void_regime or p.high_void_regime)
        ) or (p.regime == REGIME_AUTO and (p.low_void_regime or p.ipe_smooth or p.ipe_local))
        return 1e-6 if blended else 0.1

    @staticmethod
    def _mix_flags(flags, other, w):
        return RegimeFlags(**{nm: (1.0 - w) * getattr(flags, nm) + w * getattr(other, nm) for nm in flags.names()})

    def closures(self, inp, want_eq=False):
        xp, bk = self.xp, self.bk
        p = self.params
        n = inp.size
        eq = None
        if p.regime == REGIME_AUTO:
            eq = equilibrium_fractions(inp, smooth=p.ipe_smooth, bk=bk)
            flags, _ = identify_regime(
                inp,
                k_switch=p.k_switch,
                eq=eq,
                smooth=p.ipe_smooth,
                local=p.ipe_local,
                local_film=p.ipe_local_film,
                local_bubble=p.ipe_local_bubble,
                bk=bk,
            )
            if p.low_void_regime:
                # sem gás não há padrão a identificar: os critérios degeneram (FG* e
                # FG*c nulos, empates permanentes); abaixo de low_void_alpha vale o
                # padrão disperso, com a mesma rampa C1 do modo de padrão fixo
                u = xp.clip((2.0 * p.low_void_alpha - inp.alphaG) / p.low_void_alpha, 0.0, 1.0)
                wl = u * u * (3.0 - 2.0 * u)
                flags = self._mix_flags(flags, fixed_regime(p.low_void_regime, n, bk=bk), wl)
        else:
            flags = fixed_regime(p.regime, n, bk=bk)
            if p.regime_vertical:
                # padrão distinto nos trechos quase verticais (ex.: riser): peso w do
                # padrão vertical em função do ângulo -- chave dura em vertical_angle
                # (vertical_blend = 0) ou rampa C1 (smoothstep) entre
                # vertical_angle -/+ vertical_blend; os fechamentos são somados com
                # pesos (1 - w) e w, o que remove a descontinuidade espacial no joelho
                w = self.vertical_weight(inp.theta)
                flags = self._mix_flags(flags, fixed_regime(p.regime_vertical, n, bk=bk), w)
            if p.low_void_regime:
                # fração de gás muito baixa (camada de gás de ~1 % da área): o
                # estratificado é mal posto nesse limite (velocidades características
                # complexas mesmo sem escorregamento); mistura C1 para o padrão
                # disperso entre 2*low_void_alpha e low_void_alpha
                u = xp.clip((2.0 * p.low_void_alpha - inp.alphaG) / p.low_void_alpha, 0.0, 1.0)
                wl = u * u * (3.0 - 2.0 * u)
                flags = self._mix_flags(flags, fixed_regime(p.low_void_regime, n, bk=bk), wl)
            if p.high_void_regime:
                # fração de gás alta (duto quase cheio de gás): o líquido é um filme de parede
                # (anular), seja qual for o padrão prescrito; rampa C1 de +-0.05 em torno de
                # high_void_alpha (simétrica à regra de fração baixa)
                u = xp.clip((inp.alphaG - (p.high_void_alpha - 0.05)) / 0.1, 0.0, 1.0)
                wh = u * u * (3.0 - 2.0 * u)
                flags = self._mix_flags(flags, fixed_regime(p.high_void_regime, n, bk=bk), wh)
            if want_eq:
                eq = equilibrium_fractions(inp, bk=bk)
        if p.slug_unit == "film":
            # filme da unidade de golfada: estratificado de equilíbrio às vazões da célula (issue 051)
            eq_f = eq if eq is not None else equilibrium_fractions(inp, smooth=p.ipe_smooth, bk=bk)
            inp = inp.replace(aG_film_eq=eq_f[2])
        S = blend_closures(inp, flags, threshold=self.blend_threshold, used=self.used_closures, bk=bk)
        if p.interfacial_pressure > 0:
            S = self._add_interfacial_pressure(inp, S)
        return S, flags, eq

    def _add_interfacial_pressure(self, inp, S):
        """Termo de pressão interfacial regularizante (forma de Bestion/CATHARE):

            dP_i = delta * aG aL rhoG rhoL / (aG rhoL + aL rhoG) * (vG - vL)^2,

        somado às correções de pressão das duas fases (P_k - P_i). Com delta >= 1 o
        modelo incompressível a dois fluidos tem velocidades características reais
        para qualquer escorregamento (regularização dos fechamentos nas células em
        que a hidrostática do estratificado ou o termo 0.25 rhoL slip^2 das bolhas
        não bastam, p.ex. fase que desaparece com escorregamento residual).
        Derivadas em alphaG e P incluídas (as em velocidade são desprezadas, como
        nas demais correções de pressão)."""
        xp = self.xp
        p = self.params
        aG, aL = inp.alphaG, inp.alphaL
        rhoL, rhoG, CL, CG = inp.rhoL, inp.rhoG, inp.CL, inp.CG
        slip = xp.clip(inp.vG - inp.vL, -p.ip_slip_max, p.ip_slip_max)
        den = aG * rhoL + aL * rhoG
        g = aG * aL / den
        dP = p.interfacial_pressure * rhoG * rhoL * g * slip**2
        # d/daG [aG aL / den] = ((aL - aG) den - aG aL (rhoL - rhoG)) / den^2
        dg_daG = ((aL - aG) * den - aG * aL * (rhoL - rhoG)) / den**2
        ddP_daG = p.interfacial_pressure * rhoG * rhoL * dg_daG * slip**2
        # d/dP: rhoG, rhoL e den dependem de P (CL = drhoL/dP, CG = drhoG/dP)
        dden_dP = aG * CL + aL * CG
        ddP_dP = (
            p.interfacial_pressure
            * slip**2
            * aG
            * aL
            * ((CG * rhoL + rhoG * CL) * den - rhoG * rhoL * dden_dP)
            / den**2
        )
        return replace(
            S,
            dPL=S.dPL + dP,
            dPG=S.dPG + dP,
            ddPLdalphaG=S.ddPLdalphaG + ddP_daG,
            ddPGdalphaG=S.ddPGdalphaG + ddP_daG,
            ddPLdP=S.ddPLdP + ddP_dP,
            ddPGdP=S.ddPGdP + ddP_dP,
        )

    def vertical_weight(self, theta):
        """Peso (0..1) do padrão vertical em função do ângulo de inclinação (rad)."""
        xp = self.xp
        p = self.params
        ath = xp.rad2deg(xp.abs(xp.asarray(theta, dtype=float)))
        if p.vertical_blend > 0:
            u = xp.clip((ath - (p.vertical_angle - p.vertical_blend)) / (2.0 * p.vertical_blend), 0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)
        return xp.where(ath > p.vertical_angle, 1.0, 0.0)

    def coefficients(self, y, theta, outputs=False):
        xp = self.xp
        p = self.params
        fl = self.fluid
        inp, P, Area = self.local_state(y, theta)
        S, flags, eq = self.closures(inp, want_eq=outputs)
        aG, aL = inp.alphaG, inp.alphaL
        rhoL, rhoG, CL, CG = inp.rhoL, inp.rhoG, inp.CL, inp.CG
        vL, vG = inp.vL, inp.vG
        AreaL, AreaG = inp.AreaL, inp.AreaG
        sin_t = xp.sin(inp.theta)

        auxr = aL / fl.gammaL + aG / fl.gammaG
        rL = (aL / fl.gammaL) / auxr
        rG = (aG / fl.gammaG) / auxr
        rL2 = (1.0 / fl.gammaL) / auxr
        rG2 = (1.0 / fl.gammaG) / auxr

        if p.legacy_convective:
            vL2, vG2 = vL * xp.abs(vL), vG * xp.abs(vG)
            cL, cG = 2.0 * xp.abs(vL), 2.0 * xp.abs(vG)
        else:
            vL2, vG2 = vL * vL, vG * vG
            cL, cG = 2.0 * vL, 2.0 * vG

        F = xp.stack(
            [
                -rG / (Area * rhoL),
                rL / (Area * rhoG),
                rL2 / (Area * p.PR * CL),
                rG2 / (Area * p.PR * CG),
                -Area * (S.dPL - aL * S.ddPLdalphaG - vL2 * rhoL),
                p.PR * AreaL * (1.0 - vL2 * CL + S.ddPLdP),
                cL,
                Area * (S.dPG + aG * S.ddPGdalphaG - vG2 * rhoG),
                p.PR * AreaG * (1.0 - vG2 * CG + S.ddPGdP),
                cG,
                S.GammaL + G * AreaL * rhoL * sin_t,
                S.GammaG + G * AreaG * rhoG * sin_t,
            ]
        )
        if not outputs:
            return F
        out = self._outputs(y, inp, S, flags, eq, F, Area)
        return F, out

    def _outputs(self, y, inp, S, flags, eq, F, Area):
        aG0, PT, qL, qG = self.unpack(y)
        p = self.params
        rhoL, rhoG = inp.rhoL, inp.rhoG
        out = dict(
            inclinacao=inp.theta,
            diametro=inp.D,
            hD=inp.hD,
            alfaL=1 - aG0,
            alfaG=aG0,
            pressao=PT * p.PR / 1e5,
            vazaoMassaL=qL,
            vazaoMassaG=qG,
            vazaoVolL=qL / rhoL,
            vazaoVolG=qG / rhoG,
            vazaoMassa=qL + qG,
            vazaoVol=qL / rhoL + qG / rhoG,
            velocidadeL=inp.vL,
            velocidadeG=inp.vG,
            velocidadeSL=inp.vLS,
            velocidadeSG=inp.vGS,
            velocidadeI=inp.vG - inp.vL,
            velSomL=1 / np.sqrt(inp.CL),
            velSomG=1 / np.sqrt(inp.CG),
            velSomM=np.sqrt((inp.alphaL / rhoL + aG0 / rhoG) / (inp.alphaL * inp.CL / rhoL + aG0 * inp.CG / rhoG)),
            dpresL=S.dPL,
            dpresG=S.dPG,
            cargaL=F[10] / inp.AreaL * 100 / 1e5,
            cargaG=F[11] / inp.AreaG * 100 / 1e5,
            densidadeL=rhoL,
            densidadeG=rhoG,
            viscosidadeL=inp.muL,
            viscosidadeG=inp.muG,
            reg1=flags.reg1,
            reg2=flags.reg2,
            GammaL=S.GammaL,
            GammaG=S.GammaG,
            tauL=S.tauL,
            tauG=S.tauG,
        )
        for i in range(12):
            out[f"F{i+1}"] = F[i]
        if eq is not None:
            hD_est, hD_anu, aG_est, aG_anu, aG_bol = eq
            out.update(alfaG_est=aG_est, alfaG_anu=aG_anu, alfaG_bol=aG_bol, hD_est=hD_est)
            fl_c, _ = identify_regime(
                inp,
                k_switch=p.k_switch,
                eq=eq,
                smooth=p.ipe_smooth,
                local=p.ipe_local,
                local_film=p.ipe_local_film,
                local_bubble=p.ipe_local_bubble,
            )
            out["reg1C"] = fl_c.reg1
        return out

    def eigenvalues(self, y, theta, virtual_mass=None, vm_lambda=None):
        """Autovalores da matriz A (velocidades características) em cada ponto.

        virtual_mass (C_vm, pesado pelo indicador de bolhas; None = params.virtual_mass):
        o termo de massa virtual de Drew & Lahey acopla as acelerações materiais das
        fases, IG aG = forças_G - K (aG - aL), IL aL = forças_L + K (aG - aL), com
        K = C_vm w alphaG rhoL A; as linhas de momento da matriz quase-linear passam a
        ser combinações das linhas de forças das duas fases mais os fluxos convectivos."""
        A, _ = self.quasi_linear(y, theta)
        A = np.nan_to_num(A)
        C_vm = self.params.virtual_mass if virtual_mass is None else virtual_mass
        if C_vm > 0:
            A = self._quasi_linear_virtual_mass(
                y, theta, A, C_vm, self.params.vm_lambda if vm_lambda is None else vm_lambda
            )
        ev = np.linalg.eigvals(A)
        order = np.argsort(ev.real, axis=1)
        return np.take_along_axis(ev, order, axis=1)

    def _quasi_linear_virtual_mass(self, y, theta, A, C_vm, lam=1.0):
        """Matriz quase-linear com o termo de massa virtual (ver eigenvalues). lam é o
        parâmetro da forma objetiva de Drew & Lahey: além da diferença de acelerações
        materiais, a força sobre o gás tem (lam-2)(vG-vL) dvG/dx + (1-lam)(vG-vL) dvL/dx
        (vezes C_vm w alphaG rhoL); lam = 2 reduz à forma simples."""
        p = self.params
        inp, P, Area = self.local_state(y, theta)
        S, flags, eq = self.closures(inp)
        aG, aL = inp.alphaG, inp.alphaL
        rhoL, rhoG, CL, CG = inp.rhoL, inp.rhoG, inp.CL, inp.CG
        vL, vG = inp.vL, inp.vG
        n = aG.size
        # linhas convectivas d(q v)/dx em (alphaG, PT, qL, qG)
        convL = np.stack(
            [-Area * vL**2 * rhoL * (-1.0), -(vL**2) * CL * p.PR * Area * aL, 2.0 * vL, np.zeros(n)], axis=1
        )
        convG = np.stack([-Area * vG**2 * rhoG, -(vG**2) * CG * p.PR * Area * aG, np.zeros(n), 2.0 * vG], axis=1)
        FL_row = A[:, 2, :] - convL  # forças = -(FL_row . dy/dx) - B
        FG_row = A[:, 3, :] - convG
        IL = rhoL * aL * Area
        IG = rhoG * aG * Area
        K = C_vm * np.asarray(flags.Bub, float) * aG * rhoL * Area
        if lam != 2.0:
            # termos objetivos: dvG/dx e dvL/dx em (alphaG, PT, qL, qG)
            dvG = (
                np.stack([-vG * Area * rhoG, -vG * Area * aG * CG * p.PR, np.zeros(n), np.ones(n)], axis=1)
                / IG[:, None]
            )
            dvL = (
                np.stack([vL * Area * rhoL, -vL * Area * aL * CL * p.PR, np.ones(n), np.zeros(n)], axis=1) / IL[:, None]
            )
            extra = (K * (vG - vL))[:, None] * (
                (lam - 2.0) * dvG + (1.0 - lam) * dvL
            )  # força sobre o gás = -extra . dy/dx
            FG_row = FG_row + extra
            FL_row = FL_row - extra
        det = IG * IL + K * (IG + IL)
        A2 = A.copy()
        A2[:, 3, :] = (IG * (IL + K) / det)[:, None] * FG_row + (IG * K / det)[:, None] * FL_row + convG
        A2[:, 2, :] = (IL * (IG + K) / det)[:, None] * FL_row + (IL * K / det)[:, None] * FG_row + convL
        return np.nan_to_num(A2)

    # ------------------------------------------------------------------
    # Forma conservativa (para volumes finitos escalonados)
    # ------------------------------------------------------------------
    def conservative_terms(self, y, theta):
        """Termos para o esquema de volumes finitos:

        retorna dict com rhoL, rhoG, CL, CG, alphaL, alphaG, vL, vG,
        dPL, dPG (correções de pressão), GammaL, GammaG (atrito por m) e
        os coeficientes de massa (rL, rG, rL2, rG2).
        """
        inp, P, Area = self.local_state(y, theta)
        S, flags, _ = self.closures(inp)
        fl = self.fluid
        aG, aL = inp.alphaG, inp.alphaL
        auxr = aL / fl.gammaL + aG / fl.gammaG
        return dict(
            inp=inp,
            S=S,
            flags=flags,
            Area=Area,
            P=P,
            vm_weight=flags.Bub,
            rL=(aL / fl.gammaL) / auxr,
            rG=(aG / fl.gammaG) / auxr,
            rL2=(1.0 / fl.gammaL) / auxr,
            rG2=(1.0 / fl.gammaG) / auxr,
        )
