"""Sistema de produção offshore do artigo 01: poço vertical, linha no fundo do mar com um trecho final descendente até
à base do riser, curva, riser vertical até à superfície e choke no topo, a descarregar num separador.

Geometria do caso offshore da dissertação de mestrado de Gustavo Alves de Carvalho (UERJ, 2022, §5.5 e §5.7): poço
vertical de 1500 m (D 0,127 m, rugosidade 0,2 mm), linha de 6000 m e riser de 300 m (D 0,1524 m, rugosidade 0,6 mm).
Fluido da Tabela 6 (§5.7, o do sistema do modelo substituto: 165 bar e 13,5 kg/s no fundo do poço). Diferenças:

- a linha termina num trecho a −2° (`L_down`, medido ao longo do duto, dentro dos 6000 m) até à curva da base do riser
  (arco de raio 5 m), e o riser sobe até ao nível do mar (300 m + o desnível desse trecho), com um troço horizontal
  curto no topo até ao choke;
- o poço é descrito com as variáveis de mercado: IPR linear com índice de produtividade `PI` em Sm³/d/bar de óleo de
  tanque, pressão do reservatório `P_res_bar` e razão gás–óleo `GOR` em Sm³/Sm³ (condições padrão da ANP, 1 atm e
  20 °C). O PI de omissão faz a IPR passar pelo ponto nominal do Gustavo com P_res = 200 bar;
- o mfsim não tem transferência de massa nem equação de energia: o fluido é um par politrópico ajustado ao flash de
  Peng–Robinson (cmfsim) a 40 °C — densidades e viscosidades a 20 bar, gás ideal entre 10 e 60 bar — e a divisão de
  massa entre as fases fica congelada na das condições da linha (`P_REF`, `T_REF`): óleo saturado com o gás dissolvido
  R_s(P_REF) e o resto do gás livre. A fração mássica de gás livre é
  w_G = ρ_gf,std (RGO − R_s)/[ρ_o,std + ρ_gd,std R_s + ρ_gf,std (RGO − R_s)], e o índice de produtividade mássico
  PI [ρ_o,std + ρ_gd,std R_s + ρ_gf,std (RGO − R_s)]/86400 kg/s/bar divide-se nas duas fases com w_G. Com a RGO do
  próprio fluido (122 Sm³/Sm³) w_G = 0,099, a do flash. Mais ou menos gás vindo do poço é mais ou menos RGO com o
  mesmo óleo e o mesmo gás livre.

O poço e a linha são dois dutos (diâmetros diferentes) ligados por um conector na cabeça do poço; a linha com o riser é
o primeiro duto da rede (`model.pipes[0]`), o poço o segundo.

    uv run python -m generators.offshore --out cases/paper01/offshore.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

G = 9.80665
PR = 1e6

# PVT do fluido da Tabela 6 (flash de Peng–Robinson do cmfsim; condições padrão 1,01325 bar e 20 °C)
P_REF, T_REF = 20e5, 313.15  # condições da linha em que a divisão entre as fases fica congelada
PVT = dict(
    GOR=122.13,  # Sm³/Sm³ (separação em dois estágios: P_REF, T_REF e condições padrão)
    Rs=23.24,  # Sm³/Sm³ dissolvidos no óleo a P_REF, T_REF
    rho_o_std=673.25,  # kg/m³, óleo de tanque
    rho_gd_std=1.1547,  # kg/m³, gás dissolvido
    rho_gf_std=0.7763,  # kg/m³, gás livre a P_REF
)
FLUID = dict(
    type="polytropic",
    rhoL0=646.7,
    rhoG0=15.0,
    muL=3.14e-4,
    muG=1.21e-5,
    gammaL=1000.0,
    gammaG=1.0,
    P0=P_REF,
    sigma=0.0163,
)
# o fluido do §5.5 (Tabela 19, congelado da mesma maneira; fração de gás 0,29 a 20 bar) não mostrou golfada severa
FLUID_55 = dict(
    type="polytropic",
    rhoL0=696.0,
    rhoG0=17.08,
    muL=1.25e-3,
    muG=1.18e-5,
    gammaL=1000.0,
    gammaG=0.95,
    P0=P_REF,
    sigma=0.0218,
)


def market_to_mass(PI, GOR, pvt=PVT):
    """(PI mássico total em kg/s/bar, fração mássica de gás livre a P_REF, kg de fluido por Sm³ de óleo) a partir do
    índice de produtividade em Sm³/d/bar de óleo de tanque e da RGO em Sm³/Sm³."""
    free = max(GOR - pvt["Rs"], 0.0)
    m_oil = pvt["rho_o_std"] + pvt["rho_gd_std"] * min(GOR, pvt["Rs"]) + pvt["rho_gf_std"] * free
    return PI * m_oil / 86400.0, pvt["rho_gf_std"] * free / m_oil, m_oil


DEFAULTS = dict(
    # poço (dissertação: 1500 m vertical, D 0,127 m, rugosidade 0,2 mm)
    L_well=1500.0,
    D_well=0.127,
    eps_well=2e-4,
    n_well=15,
    # linha (6000 m, D 0,1524 m, rugosidade 0,6 mm) com o trecho final descendente
    L_line=6000.0,
    L_down=1000.0,
    slope_down_deg=-2.0,
    D=0.1524,
    eps=6e-4,
    n_h1=25,
    n_down=10,
    grading_down=1.0,
    # curva da base do riser e riser até ao nível do mar
    water_depth=300.0,
    elbow_radius=5.0,
    n_elbow=4,
    n_riser=16,
    grading_riser=1.0,
    L_h2=10.0,
    n_h2=2,
    # fluido e poço (variáveis de mercado; PI de omissão: ponto nominal do Gustavo, 13,5 kg/s a 165 bar, com
    # P_res = 200 bar)
    fluid=None,
    GOR=PVT["GOR"],
    PI=round(13.5 / market_to_mass(1.0, PVT["GOR"])[0] / (200.0 - 165.0), 2),
    P_res_bar=200.0,
    P_sep_bar=10.0,
    choke_cv=150.0,
    choke_opening=0.2,
    # modelo e integração
    slug_frequency="zabaras",
    slug_unit="film",
    k_switch=20.0,
    tf=20000.0,
    dt_out=5.0,
    rtol=3e-3,
    tol_range=0.01,
    max_step=None,
    q_ref_factor=10.0,
    newton_damping="legacy",  # "armijo" (issue 053) nos casos novos; "legacy" enquanto os lotes de 2026-10-17 correm
)


def _model(p):
    return {
        "type": "two_fluid",
        "regime": {
            "type": "automatic",
            "k_switch": float(p["k_switch"]),
            "smooth": True,
            "local": True,
            "local_film": True,
            "low_void": {"regime": "bubble", "alpha": 0.01},
        },
        "closures": {
            "stratified": {"fi_mult": 1.0, "v_gt": 5.5},
            "bubble": {"drag": "physical"},
            "intermittent": {"slug_frequency": p["slug_frequency"], "unit": p["slug_unit"]},
        },
        "options": {
            "alpha_min": 0.001,
            "alpha_max": 0.999,
            "legacy_convective": True,
            "interfacial_pressure": 0.0,
            "ip_slip_max": 10.0,
            "virtual_mass": {"C_vm": 0.5, "lambda": 1.0},
            "derivatives": "exact",
        },
    }


def _scheme(qL, qG):
    return {
        "type": "fv_upwind",
        "options": {
            "alpha_face": "upwind",
            "art_visc": 0.05,
            "conservative": True,
            "q_switch_G": 0.02 * qG,
            "q_switch_L": 0.02 * qL,
            "relax_target": "noslip",
            "relax_time": 0.05,
            "switch_frac": 0.5,
            "closure_alpha": "donor",
        },
    }


def line_geometry(p):
    """Pontos (y horizontal, z vertical; z = 0 ao nível do mar) e segmentos da linha com o riser, a partir da cabeça
    do poço no fundo do mar. Devolve (points, segments, info)."""
    th = np.deg2rad(p["slope_down_deg"])
    L_h1 = p["L_line"] - p["L_down"]
    z0 = -p["water_depth"]
    yC = L_h1 + p["L_down"] * np.cos(th)
    zC = z0 + p["L_down"] * np.sin(th)  # joelho (base do riser)
    R = p["elbow_radius"]
    n_el = p["n_elbow"]
    t_tan = R * (1.0 - np.sin(th)) / np.cos(th)  # distância do joelho aos pontos de tangência do arco
    y1, z1 = yC - t_tan * np.cos(th), zC - t_tan * np.sin(th)
    ang = th + (np.pi / 2 - th) * np.arange(1, n_el + 1) / n_el
    y_arc = y1 + R * (np.sin(ang) - np.sin(th))
    z_arc = z1 - R * (np.cos(ang) - np.cos(th))
    # o arco acaba na vertical do joelho (y = yC), t_tan acima dele; o riser sobe daí até z = 0
    y = [0.0, L_h1, y1] + y_arc.tolist() + [yC, yC + p["L_h2"]]
    z = [z0, z0, z1] + z_arc.tolist() + [0.0, 0.0]
    n = [p["n_h1"], p["n_down"]] + [1] * n_el + [p["n_riser"], p["n_h2"]]
    grading = [1.0, p["grading_down"]] + [1.0] * n_el + [p["grading_riser"], 1.0]
    segments = [{"n": int(k)} if g == 1.0 else {"n": int(k), "grading": float(g)} for k, g in zip(n, grading)]
    points = [{"xyz": [0.0, float(a), float(b)]} for a, b in zip(y, z)]
    info = dict(
        L_h1=L_h1,
        z_wellhead=z0,
        z_riser_base=float(zC),
        drop_down=float(-p["L_down"] * np.sin(th)),
        H_riser=float(-zC),
        s_elbow0=float(L_h1 + p["L_down"] - t_tan),
        t_tan=float(t_tan),
    )
    return points, segments, info


def offshore_case(**kw):
    """Caso JSON (`mfsim-case/1`) do sistema offshore; `kw` substitui os `DEFAULTS`."""
    p = dict(DEFAULTS, **kw)
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"parâmetros desconhecidos: {sorted(unknown)}")
    PI_mass, w_gas, m_oil = market_to_mass(p["PI"], p["GOR"])
    IPL, IPG = PI_mass * (1.0 - w_gas), PI_mass * w_gas
    # estimativa inicial das vazões: um quarto da vazão aberta (só a estimativa; o caso fixa a IPR)
    qL0, qG0 = 0.25 * IPL * p["P_res_bar"], max(0.25 * IPG * p["P_res_bar"], 1e-6)
    points, segments, info = line_geometry(p)
    z0 = info["z_wellhead"]
    well_points = [{"xyz": [0.0, 0.0, z0 - p["L_well"]]}, {"xyz": [0.0, 0.0, z0]}]
    rhoL0 = (p["fluid"] or FLUID)["rhoL0"]
    ports_in = {"alphaG": "neumann", "PT": "free", "qL": "dirichlet", "qG": "dirichlet"}
    ports_out = {"alphaG": "free", "PT": "dirichlet", "qL": "free", "qG": "free"}
    PT_sep = p["P_sep_bar"] * 1e5 / PR
    # pressão na cabeça do poço só para a estimativa inicial (o conector impõe a da entrada da linha)
    PT_head = (p["P_sep_bar"] * 1e5 + 0.5 * rhoL0 * G * info["H_riser"] + 5e5) / PR
    state_ref = {
        "PT": (p["P_sep_bar"] * 1e5 + 1.1 * rhoL0 * G * info["H_riser"] + 2e5) / PR,
        "qL": p["q_ref_factor"] * qL0,
        "qG": p["q_ref_factor"] * qG0,
    }
    case = {
        "schema": "mfsim-case/1",
        "name": "offshore",
        "description": (
            "Sistema offshore do artigo 01 (geometria do caso offshore da dissertação de Gustavo Alves de Carvalho, "
            "UERJ 2022, com o fluido da Tabela 6): poço vertical de 1500 m com IPR linear no fundo "
            f"(PI {p['PI']:g} Sm³/d/bar, P_res {p['P_res_bar']:g} bar, RGO {p['GOR']:g} Sm³/Sm³), linha de 6000 m com "
            f"trecho final a {p['slope_down_deg']:g}° ({p['L_down']:g} m), curva de raio {p['elbow_radius']:g} m, "
            f"riser até à superfície, choke CV {p['choke_cv']:g} no topo e separador a {p['P_sep_bar']:g} bar; fluido "
            "congelado num par politrópico, padrão de escoamento dinâmico, unidade de golfada com o filme de "
            "equilíbrio (051) e frequência de Zabaras (049). Gerado por generators/offshore.py."
        ),
        "meta": {
            "generator": "generators.offshore",
            "generator_params": {k: v for k, v in p.items()},
            "geometry": info,
            "market": {"PI_Sm3_d_bar": p["PI"], "GOR_Sm3_Sm3": p["GOR"], "P_res_bar": p["P_res_bar"]},
            "mass": {"PI_kg_s_bar": PI_mass, "w_gas": w_gas, "kg_per_Sm3_oil": m_oil},
            "pvt": dict(PVT, P_ref_bar=P_REF / 1e5, T_ref_C=T_REF - 273.15, std="1,01325 bar e 20 °C"),
            "paper": "01-severe-slugging-two-fluid",
            "map_base": True,
            "run": None,
        },
        "fluid": dict(p["fluid"] or FLUID),
        "reference": {"PR": PR, "state_ref": state_ref},
        "time": {"t0": 0.0, "tf": float(p["tf"]), "dt_out": float(p["dt_out"])},
        "network": {
            "vertices": [
                {"id": "1", "kind": "boundary", "spec": {"qL": qL0, "qG": qG0}},
                {"id": "Jr1", "kind": "junction", "spec": {"qL": qL0, "qG": qG0}},
                {"id": "3", "kind": "boundary", "spec": {"PT": PT_head}},
                {"id": "4", "kind": "boundary", "spec": {"qL": qL0, "qG": qG0}},
                {"id": "Jc2", "kind": "junction", "spec": {"PT": PT_sep}},
                {"id": "2", "kind": "boundary", "spec": {"PT": PT_sep}},
            ],
            "edges": [
                {
                    "id": "1",
                    "type": "pipe",
                    "from": "4",
                    "to": "Jc2",
                    "D": p["D"],
                    "roughness": p["eps"],
                    "geometry": {"points": points, "segments": segments},
                    "model": _model(p),
                    "scheme": _scheme(qL0, qG0),
                    "ports": {"in": dict(ports_in), "out": dict(ports_out)},
                },
                {
                    "id": "well",
                    "type": "pipe",
                    "from": "Jr1",
                    "to": "3",
                    "D": p["D_well"],
                    "roughness": p["eps_well"],
                    "geometry": {"points": well_points, "segments": [{"n": int(p["n_well"])}]},
                    "model": _model(p),
                    "scheme": _scheme(qL0, qG0),
                    "ports": {"in": dict(ports_in), "out": dict(ports_out)},
                },
                {"id": "connector_3_4", "type": "connector", "from": "3", "to": "4"},
                {
                    "id": "reservoir_1",
                    "type": "reservoir",
                    "from": "1",
                    "to": "Jr1",
                    "IPL": IPL,
                    "IPG": IPG,
                    "P_bar": float(p["P_res_bar"]),
                },
                {
                    "id": "choke_2",
                    "type": "choke",
                    "from": "Jc2",
                    "to": "2",
                    "CV": float(p["choke_cv"]),
                    "law": "homogeneous",
                    "opening": float(p["choke_opening"]),
                },
            ],
        },
        "integration": {
            "integrator": "implicit",
            "quadrature_order": 2,
            "max_step": p["max_step"],
            "atol": 1e-06,
            "rtol": float(p["rtol"]),
            "backend": "jax",
            "tol_range": float(p["tol_range"]),
            "first_step": 0.001,
            "max_order": 2,
            "max_newton": 20,
            "newton_tol": 0.001,
            "min_step": 1e-09,
            "slow_exit": False,
            "accumulate": True,
            **({"newton_damping": p["newton_damping"]} if p["newton_damping"] != "legacy" else {}),
        },
        "initial_state": {"source": "guess"},
    }
    return case


# ------------------------------------------------------------------------------------------ estado inicial
def _equilibrium_alpha(pb, fluid, P, qL, qG, theta):
    """Fração de gás de equilíbrio do padrão provável num nó: estratificado (Taitel e Dukler) até 45°, deriva
    (C0 1,25 e bolha de Harmathy) acima; a sem escorregamento quando o equilíbrio não existe."""
    from mfsim.closures import ClosureInputs, equilibrium_fractions
    from mfsim.geometry import stratified_level

    pp = pb.model.params
    rhoL, rhoG, CL, CG = fluid.props(P)
    A = np.pi * pp.D**2 / 4
    vLS, vGS = qL / rhoL / A, qG / rhoG / A
    lam = vGS / (vLS + vGS)
    hD, beta = stratified_level(np.array([0.5]))
    one = np.ones(1)
    inp = ClosureInputs(
        alphaG=0.5 * one, alphaL=0.5 * one, vL=2 * vLS * one, vLS=vLS * one, vG=2 * vGS * one, vGS=vGS * one,
        AreaL=A / 2 * one, AreaG=A / 2 * one, D=pp.D * one, eps=pp.eps * one, theta=theta * one, rhoL=rhoL * one,
        rhoG=rhoG * one, muL=fluid.muL * one, muG=fluid.muG * one, CL=CL * one, CG=CG * one, sigma=fluid.sigma * one,
        beta=beta, hD=hD,
    )  # fmt: skip
    _, _, aG_est, _, aG_bol = equilibrium_fractions(inp)
    a = float(aG_est[0]) if abs(theta) < np.pi / 4 else float(aG_bol[0])
    return float(np.clip(a if np.isfinite(a) else lam, 0.02, 0.98)), float(lam)


def _node_alpha(mode, pb, fluid, P, qL, qG, theta, is_line, is_top):
    """Fração de gás de um nó segundo o modo da estimativa: "noslip" (sem escorregamento em todo o lado) ou
    "blocked" (partida bloqueada: linha estratificada no equilíbrio nos troços horizontais e descendentes, curva e
    riser cheios de líquido, topo e poço sem escorregamento)."""
    rhoL, rhoG, _, _ = fluid.props(P)
    lam = float(np.clip((qG / rhoG) / (qL / rhoL + qG / rhoG), 0.02, 0.98))
    if mode == "noslip" or not is_line or is_top:
        return lam
    deg = np.rad2deg(theta)
    if deg > 1.0:
        return 0.02
    a, _ = _equilibrium_alpha(pb, fluid, P, qL, qG, theta)
    return max(a, lam)


def _march(pb, fluid, P_out, qL, qG, alpha_of, n_sub=4):
    """Pressão (Pa) e fração de gás nos nós do esquema de um duto, marchando da saída para a entrada com
    dP/ds = rho g sin(theta) + 2 f rho_ns v_m^2/D (densidade local com a fração do nó no termo hidrostático, sem
    escorregamento no atrito)."""
    from mfsim.friction import fanning

    sch = pb.scheme
    x = np.asarray(sch.x_nodes, float)
    th_n = np.asarray(sch.theta_nodes, float)
    zc = np.concatenate([[0.0], np.cumsum(np.sin(pb.mesh["theta"]) * pb.mesh["h"])])
    z = np.interp(x, pb.mesh["s"], zc)
    D, eps = float(pb.model.params.D), float(pb.model.params.eps)
    A = np.pi * D**2 / 4
    P, a = np.zeros_like(x), np.zeros_like(x)
    P[-1] = P_out
    a[-1] = alpha_of(P_out, th_n[-1])

    def grad(Pi, dz_ds, th):
        rhoL, rhoG, _, _ = fluid.props(Pi)
        aG = alpha_of(Pi, th)
        lam = (qG / rhoG) / (qL / rhoL + qG / rhoG)
        rho, rho_ns = (1 - aG) * rhoL + aG * rhoG, (1 - lam) * rhoL + lam * rhoG
        v_m = (qL / rhoL + qG / rhoG) / A
        f, _ = fanning(np.array([v_m]), rho_ns, fluid.muL, D, eps)
        return rho * G * dz_ds + 2.0 * float(f[0]) * rho_ns * v_m**2 / D

    for i in range(x.size - 1, 0, -1):
        ds = x[i] - x[i - 1]
        dz_ds = (z[i] - z[i - 1]) / ds if ds > 0 else 0.0
        th = float(th_n[i - 1])
        Pi, h = P[i], ds / n_sub
        for _ in range(n_sub):
            Pi = Pi + h * grad(Pi, dz_ds, th)
        P[i - 1] = Pi
        a[i - 1] = alpha_of(Pi, th)
    return P, a


def initial_state_guess(model, net, w_gas, mode="noslip"):
    """Estado inicial do sistema offshore (vetor de estados do modelo): vazão total da IPR por bissecção, pressão
    marchada do separador (choke, linha com riser, poço) e fração de gás por nó segundo `mode`."""
    from mfsim.core.slots import SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
    from mfsim.network.elements import valve_upstream_pressure
    from mfsim.network.graph import Choke, Reservoir

    PR, fluid = model.PR, model.fluid
    ch = [e for e in net.edges if isinstance(e, Choke)][0]
    rs = [e for e in net.edges if isinstance(e, Reservoir)][0]
    P_sep = [v for v in net.vertices if v.id == ch.v_to][0].spec["PT"](0.0) * PR
    ab = float(ch.opening(0.0))
    line, well = model.pipes[0], model.pipes[1]
    s_top = float(np.asarray(line.scheme.x_nodes)[-1]) - float(net.meta["generator_params"]["L_h2"]) - 1.0

    def profile(m):
        qL, qG = (1 - w_gas) * m, w_gas * m
        ZP, _ = valve_upstream_pressure(P_sep / PR, qL, qG, ab, ch.CV, fluid, PR, law=ch.law)
        xs = np.asarray(line.scheme.x_nodes, float)
        th_n = np.asarray(line.scheme.theta_nodes, float)
        # o topo horizontal da linha identifica-se pela posição (a marcha dá o ângulo, não a posição)
        pos = {float(t): x for x, t in zip(xs, th_n)}
        P_line, a_line = _march(
            line,
            fluid,
            float(ZP) * PR,
            qL,
            qG,
            lambda P, th: _node_alpha(mode, line, fluid, P, qL, qG, th, True, False),
        )
        if mode == "blocked":
            a_line = np.array(
                [
                    _node_alpha(mode, line, fluid, P_line[i], qL, qG, th_n[i], True, xs[i] > s_top)
                    for i in range(xs.size)
                ]
            )
        P_well, a_well = _march(
            well, fluid, P_line[0], qL, qG, lambda P, th: _node_alpha(mode, well, fluid, P, qL, qG, th, False, False)
        )
        return qL, qG, (P_line, a_line), (P_well, a_well)

    def resid(m):
        _, _, _, (P_well, _) = profile(m)
        return m - (rs.IPL + rs.IPG) * max(0.0, rs.P - P_well[0] / 1e5)

    lo, hi = 1e-3, (rs.IPL + rs.IPG) * rs.P
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            hi = mid
        else:
            lo = mid
    m = max(0.5 * (lo + hi), 1e-3)
    qL, qG, (P_line, a_line), (P_well, a_well) = profile(m)
    X = np.zeros(model.n_states)
    for pb, P, a in ((line, P_line, a_line), (well, P_well, a_well)):
        Y = pb.base_Y()
        Y[:, SLOT_ALPHA] = np.clip(a, 0.02, 0.98)
        Y[:, SLOT_P] = P / PR
        Y[:, SLOT_QL] = qL
        Y[:, SLOT_QG] = qG
        if hasattr(pb.scheme, "from_primitive"):
            Y = pb.scheme.from_primitive(Y)
        X[pb.offset : pb.offset + pb.n_states] = Y.ravel()[pb.free]
    return X, m


def write_case(case, out, ic=None):
    """Grava o caso; com `ic` ("noslip" | "blocked") calcula a estimativa inicial e grava-a nos nós do esquema
    (`<nome>.ic.npz` ao lado, `initial_state.source = "edges"`), para que o caso seja autocontido."""
    from mfsim.io import case_from_json, case_to_json
    from mfsim.simulation import build_model

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ic is None:
        out.write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return out
    net = case_from_json(case)
    model = build_model(net)
    X, m = initial_state_guess(model, net, case["meta"]["mass"]["w_gas"], mode=ic)
    model.state_to_case_ic(X, 0.0, exact=True)
    net.initial = "edges"
    net.meta = dict(net.meta or {}, initial_guess=dict(mode=ic, m_total=float(m)))
    case_to_json(net, str(out), ic_file=f"{out.stem}.ic.npz")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", action="append", default=[], help="nome=valor (JSON) para substituir um parâmetro")
    ap.add_argument("--ic", choices=["noslip", "blocked"], default=None, help="estimativa inicial gravada no caso")
    a = ap.parse_args(argv)
    kw = {}
    for item in a.param:
        k, v = item.split("=", 1)
        kw[k] = json.loads(v)
    case = offshore_case(**kw)
    print(write_case(case, a.out, ic=a.ic))


if __name__ == "__main__":
    main()
