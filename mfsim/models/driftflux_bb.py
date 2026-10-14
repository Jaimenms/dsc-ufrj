"""Modelo de mistura a 3 equações com correlação de Beggs & Brill (modelo_duto_BB.m).

Variáveis de estado: PT, qL, qG (slots 1, 2, 3). A fração de área de gás é
obtida da correlação de hold-up de Beggs & Brill (1973) e o fator de atrito
bifásico também. O sistema quase-linear é

    MA dy/dt + MB dy/dx + MC = 0  ->  dy/dt = -(MA^-1 MB) dy/dx - MA^-1 MC

com derivadas das relações de fechamento por diferenças finitas de 1e-3 relativo
(como no código original; backend numpy) ou jvp exatas (backend JAX com
`bb_derivatives="exact"`). Uma implementação para os dois backends (`bk`).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..backend import NUMPY
from .base import G, PipeModel, SLOT_P, SLOT_QG, SLOT_QL


def sigmoid(k, x1, x2, bk=NUMPY):
    return bk.sigmoid(k * (x1 - x2))


def beggs_brill(P, qL, qG, D, eps, theta, fluid, regime=6, bk=NUMPY):
    """Relações de fechamento de Beggs & Brill.

    Retorna dict com tension (tensão cisalhante), rhoS (densidade da mistura),
    vM (velocidade da mistura ponderada por massa), HL, HG, rhoL, rhoG e flags.
    """
    xp = bk.xp
    g = 9.80665
    P = xp.maximum(1e5, P)
    rhoL = fluid.rhoL0 * xp.ones_like(P)
    rhoG = fluid.rhoG0 * (P / fluid.P0) ** (1.0 / fluid.gammaG)
    Area = xp.pi * D**2 / 4.0
    vSL = qL / rhoL / Area
    vSG = qG / rhoG / Area
    direction = xp.where(vSL > 0, 1.0, -1.0)
    vSL = xp.abs(vSL)
    vSG = xp.abs(vSG)
    vm = vSL + vSG
    lambdaL = vSL / (vm + 1e-9)
    lambdaG = vSG / (vm + 1e-9)
    ix = lambdaL > 1
    lambdaL = xp.where(ix, 1.0, lambdaL)
    lambdaG = xp.where(ix, 0.0, lambdaG)
    ix = lambdaG > 1
    lambdaG = xp.where(ix, 1.0, lambdaG)
    lambdaL = xp.where(ix, 0.0, lambdaL)
    rhoN = rhoL * lambdaL + rhoG * lambdaG
    muN = fluid.muL * lambdaL + fluid.muG * lambdaG
    NFR = vm**2 / (g * D)
    NLV = vSL * (rhoL / fluid.sigma / g) ** 0.25
    NRE = rhoN * vm * D / muN

    # identificação de padrão
    L1 = 316.0 * lambdaL**0.3020
    L2 = 0.0009252 * lambdaL ** (-2.4684)
    L3 = 0.10 * lambdaL ** (-1.4516)
    L4 = 0.5 * lambdaL ** (-6.7380)
    k = 1e1
    with bk.errstate(divide="ignore", invalid="ignore"):
        tA = sigmoid(k, lambdaL / 0.01, 1, bk)
        tB = sigmoid(k, lambdaL / 0.4, 1, bk)
        tC = sigmoid(k, L1 / NFR, 1, bk)
        tD = sigmoid(k, L2 / NFR, 1, bk)
        tE = sigmoid(k, L3 / NFR, 1, bk)
        tF = sigmoid(k, L4 / NFR, 1, bk)
    fEst = xp.clip((1 - tA) * tC + tA * tD, 0, 1)
    fTra = xp.clip(tA * tE * (1 - tD), 0, 1)
    fInt = xp.clip((1 - tB) * tA * tC * (1 - tE) + tB * tF * (1 - tE), 0, 1)
    fDis = xp.clip((1 - tB) * (1 - tC) + tB * (1 - tF), 0, 1)
    if regime != 6:
        z = xp.zeros_like(lambdaL)
        o = xp.ones_like(lambdaL)
        fEst, fTra, fInt, fDis = z, z, z, z
        if regime == 1:
            fTra = o
        elif regime == 2:
            fDis = o
        elif regime == 3:
            fInt = o
        elif regime in (4, 5):
            fEst = o

    # hold-up horizontal
    with bk.errstate(divide="ignore", invalid="ignore"):
        HL0_Est = xp.clip(xp.maximum(0.98 * lambdaL**0.4846 / NFR**0.0868, lambdaL), lambdaL, 1)
        HL0_Int = xp.clip(xp.maximum(0.845 * lambdaL**0.5351 / NFR**0.0173, lambdaL), lambdaL, 1)
        HL0_Dis = xp.clip(xp.maximum(1.065 * lambdaL**0.5824 / NFR**0.0609, lambdaL), lambdaL, 1)

        # correção de inclinação
        flagAsc = (1 + xp.sign(direction) * xp.sign(theta)) / 2.0
        flagDesc = 1 - flagAsc
        s18 = xp.sin(1.8 * theta) - 0.333 * xp.sin(1.8 * theta) ** 3

        def corr(Cd, Ce, Cf, Cg):
            C = (1 - lambdaL) * xp.log(Cd * lambdaL**Ce * NLV**Cf * NFR**Cg)
            C = xp.maximum(xp.nan_to_num(C, nan=0.0, neginf=0.0), 0.0)
            return 1 + C * s18

        qsiDesc = corr(4.70, -0.3692, 0.1244, -0.5056)
        qsi_Est = corr(0.011, -3.768, 3.539, -1.614) * flagAsc + qsiDesc * flagDesc
        qsi_Int = corr(2.96, 0.305, -0.4473, 0.0978) * flagAsc + qsiDesc * flagDesc
        qsi_Dis = 1.0 * flagAsc + qsiDesc * flagDesc

        HL_Est = HL0_Est * qsi_Est
        HL_Int = HL0_Int * qsi_Int
        HL_Dis = HL0_Dis * qsi_Dis
        fi = (L3 - NFR) / (L3 - L2)
        HL_Tra = fi * HL_Est + (1 - fi) * HL_Int
    HL = HL_Est * fEst + HL_Int * fInt + HL_Dis * fDis + HL_Tra * fTra
    HL = xp.clip(xp.nan_to_num(xp.where(xp.isnan(HL), lambdaL, HL)), 1e-9, 1 - 1e-9)
    HG = 1 - HL
    ix1 = HG < 1e-6
    HG = xp.where(ix1, 1e-6, HG)
    HL = xp.where(ix1, 1 - 1e-6, HL)
    ix2 = HL < 1e-6
    HL = xp.where(ix2, 1e-6, HL)
    HG = xp.where(ix2, 1 - 1e-6, HG)

    vL = direction * vSL / HL
    vG = direction * vSG / HG
    rhoS = HL * rhoL + HG * rhoG

    # fator de atrito (Churchill, Darcy = 4 Fanning)
    NRE_s = xp.where(NRE == 0, 1e-30, NRE)
    A1 = (2.457 * xp.log(1.0 / (0.27 * eps / D + (7.0 / NRE_s) ** 0.9))) ** 16
    B1 = (37530.0 / NRE_s) ** 16
    f = 4.0 * (2.0 * ((8.0 / NRE_s) ** 12 + 1.0 / (A1 + B1) ** 1.5) ** (1.0 / 12.0))
    with bk.errstate(divide="ignore", invalid="ignore"):
        kk = xp.log(lambdaL / HL**2)
        S = xp.log(2.2 * xp.exp(kk) - 1.2)
        ix = (kk <= 0) | (kk >= 0.1823)
        S_alt = (-0.0523 / kk + 3.1820 - 0.8725 * kk + 0.01853 * kk**3) ** (-1)
        S = xp.where(ix, S_alt, S)
    S = xp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = xp.where(ix1, 0.0, S)
    S = xp.where(ix2, 0.0, S)
    ftp = f * xp.exp(S)
    tension = direction * (ftp * rhoS * vm**2) / 8.0
    vM = (HL * rhoL * vL + HG * rhoG * vG) / (HL * rhoL + HG * rhoG)
    return dict(
        tension=tension,
        rhoS=rhoS,
        vM=vM,
        HL=HL,
        HG=HG,
        rhoL=rhoL,
        rhoG=rhoG,
        vL=vL,
        vG=vG,
        vm=vm,
        flTra=fTra,
        flEst=fEst,
        flInt=fInt,
        flDis=fDis,
        NFR=NFR,
        NLV=NLV,
        NRE=NRE,
        lambdaL=lambdaL,
    )


class BeggsBrillModel(PipeModel):
    name = "beggs_brill"
    slots = (SLOT_P, SLOT_QL, SLOT_QG)
    A_pattern = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B_pattern = np.array([10, 11, 12])
    nF = 12

    @property
    def exact_derivatives(self):
        """Derivadas internas por jvp (só com diferenciação automática); senão diferenças finitas."""
        return self.params.bb_derivatives == "exact" and self.bk.has_ad

    def coefficients(self, y, theta, outputs=False):
        xp, bk = self.xp, self.bk
        p = self.params
        fl = self.fluid
        y = xp.asarray(y, dtype=float).reshape(-1, 4)
        PT = y[:, SLOT_P]
        qL = xp.maximum(0.001, y[:, SLOT_QL])
        qG = xp.maximum(0.001, y[:, SLOT_QG])
        n = PT.shape[0]
        th = xp.asarray(theta, dtype=float) * xp.ones(n)
        P = p.PR * PT

        bb = beggs_brill(P, qL, qG, p.D, p.eps, th, fl, p.regime, bk=bk)
        rhoL, rhoG = bb["rhoL"], bb["rhoG"]
        cL = rhoL / (P * fl.gammaL)
        cG = rhoG / (P * fl.gammaG)
        alphaL, alphaG = bb["HL"], bb["HG"]
        Mom = bb["rhoS"] * bb["vM"]
        Vel = bb["rhoS"] * bb["vM"] ** 2
        if self.exact_derivatives:
            # derivadas internas exatas: jvp do fechamento (nó a nó, pois cada nó só
            # depende dos seus próprios P, qL, qG) nas três direções
            def gfun(P_, qL_, qG_):
                b = beggs_brill(P_, qL_, qG_, p.D, p.eps, th, fl, p.regime, bk=bk)
                return b["rhoS"] * b["vM"], b["rhoS"] * b["vM"] ** 2, b["HG"]

            o, z = xp.ones_like(P), xp.zeros_like(P)
            _, (dMomdP, dVeldP, daGdP) = bk.jvp(gfun, (P, qL, qG), (o, z, z))
            _, (dMomdqL, dVeldqL, daGdqL) = bk.jvp(gfun, (P, qL, qG), (z, o, z))
            _, (dMomdqG, dVeldqG, daGdqG) = bk.jvp(gfun, (P, qL, qG), (z, z, o))
        else:
            P2 = P * (1 + 1e-3)
            qL2 = qL * (1 + 1e-3) + 1e-6
            qG2 = qG * (1 + 1e-3) + 1e-6
            bP = beggs_brill(P2, qL, qG, p.D, p.eps, th, fl, p.regime, bk=bk)
            bL = beggs_brill(P, qL2, qG, p.D, p.eps, th, fl, p.regime, bk=bk)
            bG = beggs_brill(P, qL, qG2, p.D, p.eps, th, fl, p.regime, bk=bk)

            def d(a, b, h):
                return (a - b) / h

            dMomdP = d(bP["rhoS"] * bP["vM"], Mom, P2 - P)
            dMomdqL = d(bL["rhoS"] * bL["vM"], Mom, qL2 - qL)
            dMomdqG = d(bG["rhoS"] * bG["vM"], Mom, qG2 - qG)
            dVeldP = d(bP["rhoS"] * bP["vM"] ** 2, Vel, P2 - P)
            dVeldqL = d(bL["rhoS"] * bL["vM"] ** 2, Vel, qL2 - qL)
            dVeldqG = d(bG["rhoS"] * bG["vM"] ** 2, Vel, qG2 - qG)
            daGdP = d(bP["HG"], alphaG, P2 - P)
            daGdqL = d(bL["HG"], alphaG, qL2 - qL)
            daGdqG = d(bG["HG"], alphaG, qG2 - qG)

        Area = xp.pi * p.D**2 / 4.0
        sgn = xp.sign(bb["vM"])
        z = xp.zeros(n)
        MA = xp.stack(
            [
                xp.stack([p.PR * (alphaL * cL - rhoL * daGdP), -rhoL * daGdqL, -rhoL * daGdqG], axis=1),
                xp.stack([p.PR * (alphaG * cG + rhoG * daGdP), rhoG * daGdqL, rhoG * daGdqG], axis=1),
                xp.stack([p.PR * dMomdP, dMomdqL, dMomdqG], axis=1),
            ],
            axis=1,
        )
        MB = xp.stack(
            [
                xp.stack([z, xp.full(n, 1.0 / Area), z], axis=1),
                xp.stack([z, z, xp.full(n, 1.0 / Area)], axis=1),
                xp.stack([p.PR * (1.0 + sgn * dVeldP), sgn * dVeldqL, sgn * dVeldqG], axis=1),
            ],
            axis=1,
        )
        MC = xp.stack([z, z, xp.pi * p.D * bb["tension"] / Area + bb["rhoS"] * G * xp.sin(th)], axis=1)
        # inversa robusta: pontos com matriz singular (fase gás extinta, derivadas
        # do hold-up nulas) recebem regularização pela pseudo-inversa
        invA = bk.robust_inv(MA)
        Amat = invA @ MB
        Bvec = xp.einsum("nij,nj->ni", invA, MC)
        F = xp.concatenate([Amat.reshape(n, 9).T, Bvec.T], axis=0)
        if not outputs:
            return F
        out = dict(
            inclinacao=th,
            diametro=np.full(n, p.D),
            hD=np.zeros(n),
            alfaL=alphaL,
            alfaG=alphaG,
            pressao=P / 1e5,
            vazaoMassaL=qL,
            vazaoMassaG=qG,
            vazaoVolL=qL / rhoL,
            vazaoVolG=qG / rhoG,
            vazaoMassa=qL + qG,
            vazaoVol=qL / rhoL + qG / rhoG,
            velocidadeL=bb["vL"],
            velocidadeG=bb["vG"],
            velocidadeSL=qL / rhoL / Area,
            velocidadeSG=qG / rhoG / Area,
            velocidadeI=bb["vL"] - bb["vG"],
            densidadeL=rhoL,
            densidadeG=rhoG,
            flTra=bb["flTra"],
            flEst=bb["flEst"],
            flInt=bb["flInt"],
            flDis=bb["flDis"],
            tensao=bb["tension"],
        )
        return F, out

    # ------------------------------------------------------------------
    # Termos para o esquema de volumes finitos escalonados (StaggeredFVScheme._rhs_drift)
    # ------------------------------------------------------------------
    def conservative_terms(self, y, theta):
        """Por nó: hold-up de Beggs & Brill (HG, HL) e suas derivadas em P, qL e qG (diferenças
        finitas de 1e-3 relativo, como em coefficients, ou jvp exatas avaliadas a 1e-6 kg/s do
        lado do escoamento), densidade da mistura rhoS, velocidade da mistura vM (ponderada
        por massa: rhoS vM = (qL + qG)/A), tensão de parede (com sinal), e as propriedades do
        fluido para as equações de massa (rhoL, rhoG, CL, CG de Fluid.props)."""
        xp, bk = self.xp, self.bk
        p = self.params
        fl = self.fluid
        y = xp.asarray(y, dtype=float).reshape(-1, 4)
        n = y.shape[0]
        th = xp.asarray(theta, dtype=float) * xp.ones(n)
        P = xp.maximum(p.PR * y[:, SLOT_P], 1e3)
        qL = y[:, SLOT_QL]
        qG = y[:, SLOT_QG]
        Area = xp.pi * p.D**2 / 4.0

        def hg(P_, qL_, qG_):
            return beggs_brill(P_, qL_, qG_, p.D, p.eps, th, fl, p.regime, bk=bk)["HG"]

        bb = beggs_brill(P, qL, qG, p.D, p.eps, th, fl, p.regime, bk=bk)
        HG = bb["HG"]
        if self.exact_derivatives:
            # derivadas exatas avaliadas a 1e-6 kg/s do lado do escoamento: em vazão de gás
            # exatamente nula a correlação está na região cortada (HG = 1e-6, derivada zero)
            # e o coeficiente da restrição de escorregamento cairia no piso; a derivada
            # lateral (como a diferença finita avançada da tese) é a sensibilidade real
            o, z = xp.ones_like(P), xp.zeros_like(P)
            qLs = qL + 1e-6 * xp.sign(qL + 1e-300)
            qGs = qG + 1e-6 * xp.sign(qG + 1e-300)
            _, dHGdP = bk.jvp(hg, (P, qLs, qGs), (o, z, z))
            _, dHGdqL = bk.jvp(hg, (P, qLs, qGs), (z, o, z))
            _, dHGdqG = bk.jvp(hg, (P, qLs, qGs), (z, z, o))
        else:
            # derivadas internas do hold-up (perturbação relativa 1e-3 com piso absoluto)
            dP_ = 1e-3 * P
            dqL = 1e-3 * xp.abs(qL) + 1e-6
            dqG = 1e-3 * xp.abs(qG) + 1e-6
            dHGdP = (hg(P + dP_, qL, qG) - HG) / dP_
            dHGdqL = (hg(P, qL + dqL, qG) - HG) / dqL
            dHGdqG = (hg(P, qL, qG + dqG) - HG) / dqG
        rhoL, rhoG, CL, CG = fl.props(P)
        HL = bb["HL"]
        inp = SimpleNamespace(rhoL=rhoL, rhoG=rhoG, CL=CL, CG=CG, alphaG=HG, alphaL=HL)
        return dict(
            Area=Area,
            P=P,
            inp=inp,
            HG=HG,
            HL=HL,
            rhoS=bb["rhoS"],
            vM=bb["vM"],
            tension=bb["tension"],
            dHGdP=dHGdP,
            dHGdqL=dHGdqL,
            dHGdqG=dHGdqG,
            C_mix=HL * CL + HG * CG,
        )
