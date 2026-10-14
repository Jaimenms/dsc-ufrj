"""Modelo de mistura a 3 equações com correlação de Beggs & Brill (modelo_duto_BB.m).

Variáveis de estado: PT, qL, qG (slots 1, 2, 3). A fração de área de gás é
obtida da correlação de hold-up de Beggs & Brill (1973) e o fator de atrito
bifásico também. O sistema quase-linear é

    MA dy/dt + MB dy/dx + MC = 0  ->  dy/dt = -(MA^-1 MB) dy/dx - MA^-1 MC

com derivadas das relações de fechamento obtidas por diferenças finitas,
como no código original.
"""

from __future__ import annotations

import numpy as np

from .base import G, PipeModel, SLOT_P, SLOT_QG, SLOT_QL


def sigmoid(k, x1, x2):
    return 1.0 / (1.0 + np.exp(np.clip(-k * (x1 - x2), -700, 700)))


def beggs_brill(P, qL, qG, D, eps, theta, fluid, regime=6):
    """Relações de fechamento de Beggs & Brill.

    Retorna dict com tension (tensão cisalhante), rhoS (densidade da mistura),
    vM (velocidade da mistura ponderada por massa), HL, HG, rhoL, rhoG e flags.
    """
    g = 9.80665
    P = np.maximum(1e5, P)
    rhoL = fluid.rhoL0 * np.ones_like(P)
    rhoG = fluid.rhoG0 * (P / fluid.P0) ** (1.0 / fluid.gammaG)
    Area = np.pi * D**2 / 4.0
    vSL = qL / rhoL / Area
    vSG = qG / rhoG / Area
    direction = np.where(vSL > 0, 1.0, -1.0)
    vSL = np.abs(vSL)
    vSG = np.abs(vSG)
    vm = vSL + vSG
    lambdaL = vSL / (vm + 1e-9)
    lambdaG = vSG / (vm + 1e-9)
    ix = lambdaL > 1
    lambdaL[ix], lambdaG[ix] = 1.0, 0.0
    ix = lambdaG > 1
    lambdaG[ix], lambdaL[ix] = 1.0, 0.0
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
    with np.errstate(divide="ignore", invalid="ignore"):
        tA = sigmoid(k, lambdaL / 0.01, 1)
        tB = sigmoid(k, lambdaL / 0.4, 1)
        tC = sigmoid(k, L1 / NFR, 1)
        tD = sigmoid(k, L2 / NFR, 1)
        tE = sigmoid(k, L3 / NFR, 1)
        tF = sigmoid(k, L4 / NFR, 1)
    fEst = np.clip((1 - tA) * tC + tA * tD, 0, 1)
    fTra = np.clip(tA * tE * (1 - tD), 0, 1)
    fInt = np.clip((1 - tB) * tA * tC * (1 - tE) + tB * tF * (1 - tE), 0, 1)
    fDis = np.clip((1 - tB) * (1 - tC) + tB * (1 - tF), 0, 1)
    if regime != 6:
        z = np.zeros_like(lambdaL)
        fEst, fTra, fInt, fDis = z.copy(), z.copy(), z.copy(), z.copy()
        if regime == 1:
            fTra[:] = 1
        elif regime == 2:
            fDis[:] = 1
        elif regime == 3:
            fInt[:] = 1
        elif regime in (4, 5):
            fEst[:] = 1

    # hold-up horizontal
    with np.errstate(divide="ignore", invalid="ignore"):
        HL0_Est = np.clip(np.maximum(0.98 * lambdaL**0.4846 / NFR**0.0868, lambdaL), lambdaL, 1)
        HL0_Int = np.clip(np.maximum(0.845 * lambdaL**0.5351 / NFR**0.0173, lambdaL), lambdaL, 1)
        HL0_Dis = np.clip(np.maximum(1.065 * lambdaL**0.5824 / NFR**0.0609, lambdaL), lambdaL, 1)

        # correção de inclinação
        flagAsc = (1 + np.sign(direction) * np.sign(theta)) / 2.0
        flagDesc = 1 - flagAsc
        s18 = np.sin(1.8 * theta) - 0.333 * np.sin(1.8 * theta) ** 3

        def corr(Cd, Ce, Cf, Cg):
            C = (1 - lambdaL) * np.log(Cd * lambdaL**Ce * NLV**Cf * NFR**Cg)
            C = np.maximum(np.nan_to_num(C, nan=0.0, neginf=0.0), 0.0)
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
    HL = np.clip(np.nan_to_num(HL, nan=lambdaL), 1e-9, 1 - 1e-9)
    HG = 1 - HL
    ix1 = HG < 1e-6
    HG[ix1], HL[ix1] = 1e-6, 1 - 1e-6
    ix2 = HL < 1e-6
    HL[ix2], HG[ix2] = 1e-6, 1 - 1e-6

    vL = direction * vSL / HL
    vG = direction * vSG / HG
    rhoS = HL * rhoL + HG * rhoG

    # fator de atrito (Churchill, Darcy = 4 Fanning)
    NRE_s = np.where(NRE == 0, 1e-30, NRE)
    A1 = (2.457 * np.log(1.0 / (0.27 * eps / D + (7.0 / NRE_s) ** 0.9))) ** 16
    B1 = (37530.0 / NRE_s) ** 16
    f = 4.0 * (2.0 * ((8.0 / NRE_s) ** 12 + 1.0 / (A1 + B1) ** 1.5) ** (1.0 / 12.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        kk = np.log(lambdaL / HL**2)
        S = np.log(2.2 * np.exp(kk) - 1.2)
        ix = (kk <= 0) | (kk >= 0.1823)
        S_alt = (-0.0523 / kk + 3.1820 - 0.8725 * kk + 0.01853 * kk**3) ** (-1)
        S = np.where(ix, S_alt, S)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S[ix1] = 0
    S[ix2] = 0
    ftp = f * np.exp(S)
    tension = direction * (ftp * rhoS * vm**2) / 8.0
    vM = (HL * rhoL * vL + HG * rhoG * vG) / (HL * rhoL + HG * rhoG)
    return dict(tension=tension, rhoS=rhoS, vM=vM, HL=HL, HG=HG, rhoL=rhoL, rhoG=rhoG,
                vL=vL, vG=vG, vm=vm, flTra=fTra, flEst=fEst, flInt=fInt, flDis=fDis,
                NFR=NFR, NLV=NLV, NRE=NRE, lambdaL=lambdaL)


class BeggsBrillModel(PipeModel):
    name = "beggs_brill"
    slots = (SLOT_P, SLOT_QL, SLOT_QG)
    A_pattern = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
    B_pattern = np.array([10, 11, 12])
    nF = 12

    def coefficients(self, y, theta, outputs=False):
        p = self.params
        fl = self.fluid
        y = np.asarray(y, float).reshape(-1, 4)
        PT = y[:, SLOT_P]
        qL = np.maximum(0.001, y[:, SLOT_QL])
        qG = np.maximum(0.001, y[:, SLOT_QG])
        n = PT.size
        th = np.asarray(theta, float) * np.ones(n)
        P = p.PR * PT

        bb = beggs_brill(P, qL, qG, p.D, p.eps, th, fl, p.regime)
        P2 = P * (1 + 1e-3)
        qL2 = qL * (1 + 1e-3) + 1e-6
        qG2 = qG * (1 + 1e-3) + 1e-6
        bP = beggs_brill(P2, qL, qG, p.D, p.eps, th, fl, p.regime)
        bL = beggs_brill(P, qL2, qG, p.D, p.eps, th, fl, p.regime)
        bG = beggs_brill(P, qL, qG2, p.D, p.eps, th, fl, p.regime)

        rhoL, rhoG = bb["rhoL"], bb["rhoG"]
        cL = rhoL / (P * fl.gammaL)
        cG = rhoG / (P * fl.gammaG)
        alphaL, alphaG = bb["HL"], bb["HG"]

        def d(a, b, h):
            return (a - b) / h

        Mom = bb["rhoS"] * bb["vM"]
        Vel = bb["rhoS"] * bb["vM"] ** 2
        dMomdP = d(bP["rhoS"] * bP["vM"], Mom, P2 - P)
        dMomdqL = d(bL["rhoS"] * bL["vM"], Mom, qL2 - qL)
        dMomdqG = d(bG["rhoS"] * bG["vM"], Mom, qG2 - qG)
        dVeldP = d(bP["rhoS"] * bP["vM"] ** 2, Vel, P2 - P)
        dVeldqL = d(bL["rhoS"] * bL["vM"] ** 2, Vel, qL2 - qL)
        dVeldqG = d(bG["rhoS"] * bG["vM"] ** 2, Vel, qG2 - qG)
        daGdP = d(bP["HG"], alphaG, P2 - P)
        daGdqL = d(bL["HG"], alphaG, qL2 - qL)
        daGdqG = d(bG["HG"], alphaG, qG2 - qG)

        Area = np.pi * p.D**2 / 4.0
        sgn = np.sign(bb["vM"])
        MA = np.empty((n, 3, 3))
        MA[:, 0, 0] = p.PR * (alphaL * cL - rhoL * daGdP)
        MA[:, 0, 1] = -rhoL * daGdqL
        MA[:, 0, 2] = -rhoL * daGdqG
        MA[:, 1, 0] = p.PR * (alphaG * cG + rhoG * daGdP)
        MA[:, 1, 1] = rhoG * daGdqL
        MA[:, 1, 2] = rhoG * daGdqG
        MA[:, 2, 0] = p.PR * dMomdP
        MA[:, 2, 1] = dMomdqL
        MA[:, 2, 2] = dMomdqG
        MB = np.zeros((n, 3, 3))
        MB[:, 0, 1] = 1.0 / Area
        MB[:, 1, 2] = 1.0 / Area
        MB[:, 2, 0] = p.PR * (1.0 + sgn * dVeldP)
        MB[:, 2, 1] = sgn * dVeldqL
        MB[:, 2, 2] = sgn * dVeldqG
        MC = np.zeros((n, 3))
        MC[:, 2] = np.pi * p.D * bb["tension"] / Area + bb["rhoS"] * G * np.sin(th)

        # inversa robusta: pontos com matriz singular (fase gás extinta, derivadas
        # do hold-up nulas) recebem regularização pela pseudo-inversa
        try:
            invA = np.linalg.inv(MA)
            if not np.all(np.isfinite(invA)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            invA = np.empty_like(MA)
            for i in range(n):
                try:
                    invA[i] = np.linalg.inv(MA[i])
                    if not np.all(np.isfinite(invA[i])):
                        raise np.linalg.LinAlgError
                except np.linalg.LinAlgError:
                    try:
                        invA[i] = np.linalg.pinv(MA[i], rcond=1e-12)
                    except np.linalg.LinAlgError:
                        invA[i] = np.nan
        Amat = invA @ MB
        Bvec = np.einsum("nij,nj->ni", invA, MC)
        F = np.empty((12, n))
        F[0:9] = Amat.reshape(n, 9).T
        F[9:12] = Bvec.T
        if not outputs:
            return F
        out = dict(
            inclinacao=th, diametro=np.full(n, p.D), hD=np.zeros(n), alfaL=alphaL, alfaG=alphaG,
            pressao=P / 1e5, vazaoMassaL=qL, vazaoMassaG=qG, vazaoVolL=qL / rhoL, vazaoVolG=qG / rhoG,
            vazaoMassa=qL + qG, vazaoVol=qL / rhoL + qG / rhoG,
            velocidadeL=bb["vL"], velocidadeG=bb["vG"],
            velocidadeSL=qL / rhoL / Area, velocidadeSG=qG / rhoG / Area,
            velocidadeI=bb["vL"] - bb["vG"], densidadeL=rhoL, densidadeG=rhoG,
            flTra=bb["flTra"], flEst=bb["flEst"], flInt=bb["flInt"], flDis=bb["flDis"],
            tensao=bb["tension"],
        )
        return F, out

    # ------------------------------------------------------------------
    # Termos para o esquema de volumes finitos escalonados (StaggeredFVScheme._rhs_drift)
    # ------------------------------------------------------------------
    def conservative_terms(self, y, theta):
        """Por nó: hold-up de Beggs & Brill (HG, HL) e suas derivadas em P, qL e qG (diferenças
        finitas de 1e-3 relativo, como em coefficients), densidade da mistura rhoS, velocidade
        da mistura vM (ponderada por massa: rhoS vM = (qL + qG)/A), tensão de parede (com
        sinal), e as propriedades do fluido para as equações de massa (rhoL, rhoG, CL, CG de
        Fluid.props, as mesmas do mapa conservativo)."""
        from types import SimpleNamespace
        p = self.params
        fl = self.fluid
        y = np.asarray(y, float).reshape(-1, 4)
        n = y.shape[0]
        th = np.asarray(theta, float) * np.ones(n)
        P = np.maximum(p.PR * y[:, SLOT_P], 1e3)
        qL = y[:, SLOT_QL]
        qG = y[:, SLOT_QG]
        Area = np.pi * p.D**2 / 4.0
        bb = beggs_brill(P, qL, qG, p.D, p.eps, th, fl, p.regime)
        # derivadas internas do hold-up (perturbação relativa 1e-3 com piso absoluto)
        dP_ = 1e-3 * P
        dqL = 1e-3 * np.abs(qL) + 1e-6
        dqG = 1e-3 * np.abs(qG) + 1e-6
        bP = beggs_brill(P + dP_, qL, qG, p.D, p.eps, th, fl, p.regime)
        bL = beggs_brill(P, qL + dqL, qG, p.D, p.eps, th, fl, p.regime)
        bG = beggs_brill(P, qL, qG + dqG, p.D, p.eps, th, fl, p.regime)
        rhoL, rhoG, CL, CG = fl.props(P)
        HL, HG = bb["HL"], bb["HG"]
        inp = SimpleNamespace(rhoL=rhoL, rhoG=rhoG, CL=CL, CG=CG, alphaG=HG, alphaL=HL)
        return dict(Area=Area, P=P, inp=inp, HG=HG, HL=HL, rhoS=bb["rhoS"], vM=bb["vM"], tension=bb["tension"],
                    dHGdP=(bP["HG"] - HG) / dP_, dHGdqL=(bL["HG"] - HG) / dqL, dHGdqG=(bG["HG"] - HG) / dqG,
                    C_mix=HL * CL + HG * CG)
