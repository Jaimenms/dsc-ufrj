"""Sistema de produção offshore do artigo 01: poço vertical, linha no fundo do mar com um trecho final descendente até
à base do riser, curva, riser vertical até à superfície e choke no topo, a descarregar num separador.

Geometria do caso offshore da dissertação de mestrado de Gustavo Alves de Carvalho (UERJ, 2022, §5.5 e §5.7): poço
vertical de 1500 m (D 0,127 m, rugosidade 0,2 mm), linha de 6000 m — aqui encurtada para 2000 m, para ciclos 3× mais
curtos (decisão do dono, 2026-10-17) — e riser de 300 m (D 0,1524 m, rugosidade 0,6 mm).
Fluido da Tabela 6 (§5.7, o do sistema do modelo substituto: 165 bar e 13,5 kg/s no fundo do poço). Diferenças:

- a linha termina num trecho a −2° (`L_down`, medido ao longo do duto, dentro de `L_line`) até à curva da base do riser
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
    # linha (dissertação: 6000 m; aqui 2000 m, decisão do dono em 2026-10-17: o período da golfada severa é
    # proporcional ao volume de gás da linha e os ciclos ficam 3× mais curtos sem sair da região de Bøe) com o
    # trecho final descendente; D 0,1524 m, rugosidade 0,6 mm; células de ~190 m e 100 m
    L_line=2000.0,
    L_down=500.0,
    slope_down_deg=-2.0,
    D=0.1524,
    eps=6e-4,
    n_h1=8,
    n_down=5,
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
    choke_opening=1.0,  # ponto base = choke aberto (a omissão 0,2 enganou a sonda e as malhas de 2026-10-17)
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
    newton_damping="armijo_tol",  # busca em linha no Newton só antes da convergência (053 + 055); "legacy" = tese
    error_estimate="milne",  # estimativa de erro de ordem consistente na ordem 2 (058); "thesis" = a da tese
    low_void_alpha=0.025,  # limiar da regra de fração baixa (zera o estratificado mal-posto; decisão do dono, 17/10)
    local_bubble=True,  # critérios de bolhas com a fração de gás da célula (054)
    # gas lift (pedido do dono, 2026-10-18): linha de gás da plataforma à cabeça do poço, anular até à válvula de
    # retenção junto ao packer, injeção no tubing; tudo com o modelo a dois fluidos. Dimensões da prática offshore
    # brasileira: linha de gas lift de 4" ao lado da produção de 6"; anular 9 5/8" × 5 1/2" (área equivalente);
    # válvula de orifício dimensionada para ~1 kg/s a 60 bar de queda; manifold de gas lift a 150 bar.
    # Desligar = pressão baixa na plataforma (20 bar): a retenção fica fechada e o ramo inerte.
    gas_lift=False,
    P_gas_bar=150.0,  # pressão do gás na plataforma (especificação de pressão)
    D_gas_line=0.1016,  # 4" (linha flexível de gas lift); segue o traçado da linha de produção (com o joelho)
    gas_mesh_factor=0.5,  # células da linha de gás = fração das da linha de produção, por segmento (mín. 1; pedido do dono)
    D_annulus=0.171,  # diâmetro de área equivalente do anular 9 5/8" (ID 0,2205 m) × 5 1/2" (OD 0,1397 m)
    n_annulus=10,
    L_inj=30.0,  # m: válvula operadora acima do fundo do poço (junto ao packer)
    n_well_lower=1,
    CV_check=1.5,  # q = 0,0075 CV sqrt(rho dP[bar]) [kg/s]: ~1 kg/s a 60 bar
    dP_min_check_bar=0.0,
    V_tank=1.0,  # m³: volume do anular sob a válvula (tanque que alimenta a retenção)
    gas_alpha=0.998,  # fração de gás inicial da linha de gás, do anular e do tanque: quase seca (o modelo admite até 1 − alpha_min); com 0,98 o líquido escorre durante horas e custa 5× os passos
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
            "local_bubble": bool(p["local_bubble"]),
            "low_void": {"regime": "bubble", "alpha": float(p["low_void_alpha"])},
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


def gas_lift_geometry(p, info):
    """Pontos da linha de gás (plataforma → base do riser → cabeça do poço) e cotas do anular e da injeção."""
    th = np.deg2rad(p["slope_down_deg"])
    L_h1 = p["L_line"] - p["L_down"]
    yC = L_h1 + p["L_down"] * np.cos(th)
    z0 = info["z_wellhead"]
    z_bot = z0 - p["L_well"]
    return dict(
        y_top=yC + p["L_h2"],
        yC=yC,
        zC=info["z_riser_base"],
        z0=z0,
        z_bot=z_bot,
        z_inj=z_bot + p["L_inj"],
    )


def _reversed_path(points, segments, factor=1.0):
    """O mesmo traçado percorrido ao contrário (a linha de gás desce da plataforma pelo caminho da linha de
    produção): pontos invertidos e, em cada segmento, a razão de graduação invertida e o número de células
    multiplicado por `factor` (arredondado, mínimo 1: a malha da linha de gás pode ser mais aberta)."""
    pts = list(reversed(points))
    segs = []
    for sg in reversed(segments):
        g = sg.get("grading", 1.0)
        n = max(1, int(round(sg["n"] * factor)))
        segs.append({"n": n} if g == 1.0 else {"n": n, "grading": 1.0 / g})
    return pts, segs


def _gas_lift_block(p, info, PR, qL0, qG0, ports_in, ports_out, rhoL0, P_sep_bar, line_points, line_segments):
    """Vértices e arestas do ramo de gas lift; a injeção é o vértice `inj`, onde o tubing inferior e o anular (pela
    retenção) alimentam o tubing superior, como no caso 10.3.2 da tese. A linha de gás segue o traçado da linha
    de produção ao contrário (plataforma → riser → joelho → descida → horizontal → cabeça do poço), com uma malha
    mais aberta (`gas_mesh_factor`)."""
    g = gas_lift_geometry(p, info)
    gas_points, gas_segments = _reversed_path(line_points, line_segments, float(p["gas_mesh_factor"]))
    fl = p["fluid"] or FLUID
    P_gas = p["P_gas_bar"] * 1e5
    rhoG = fl["rhoG0"] * (P_gas / fl["P0"]) ** (1.0 / fl["gammaG"])
    qG_gl0 = 0.5  # kg/s: só a estimativa inicial das vazões do ramo (o estado inicial recalcula)
    P_wh_gas = P_gas + rhoG * G * (0.0 - g["z0"])
    P_inj_guess = P_sep_bar * 1e5 + 0.5 * rhoL0 * G * (info["H_riser"] + p["L_well"] - p["L_inj"])
    vertices = [
        {"id": "gl_src", "kind": "boundary", "spec": {"PT": P_gas / PR, "qL": 0.0}},
        {"id": "gl_wh", "kind": "boundary", "spec": {"PT": P_wh_gas / PR}},
        {"id": "an_top", "kind": "boundary", "spec": {"qL": 0.0, "qG": qG_gl0}},
        {"id": "an_bot", "kind": "tank", "volume": float(p["V_tank"])},
        {"id": "inj", "kind": "boundary", "spec": {"qL": qL0, "qG": qG0 + qG_gl0}},
        {"id": "wl_out", "kind": "boundary", "spec": {"PT": P_inj_guess / PR}},
    ]
    ports_gas_in = {"alphaG": "neumann", "PT": "dirichlet", "qL": "dirichlet", "qG": "free"}
    edges = [
        {
            "id": "well_lower",
            "type": "pipe",
            "from": "Jr1",
            "to": "wl_out",
            "D": p["D_well"],
            "roughness": p["eps_well"],
            "geometry": {
                "points": [{"xyz": [0.0, 0.0, g["z_bot"]]}, {"xyz": [0.0, 0.0, g["z_inj"]]}],
                "segments": [{"n": int(p["n_well_lower"])}],
            },
            "model": _model(p),
            "scheme": _scheme(qL0, qG0),
            "ports": {"in": dict(ports_in), "out": dict(ports_out)},
        },
        {"id": "connector_wl_inj", "type": "connector", "from": "wl_out", "to": "inj"},
        {
            "id": "gas_line",
            "type": "pipe",
            "from": "gl_src",
            "to": "gl_wh",
            "D": p["D_gas_line"],
            "roughness": p["eps"],
            "geometry": {"points": gas_points, "segments": gas_segments},
            "model": _model(p),
            "scheme": _scheme(1e-3, qG_gl0),
            "ports": {"in": dict(ports_gas_in), "out": dict(ports_out)},
        },
        {"id": "connector_gl_an", "type": "connector", "from": "gl_wh", "to": "an_top"},
        {
            "id": "annulus",
            "type": "pipe",
            "from": "an_top",
            "to": "an_bot",
            "D": p["D_annulus"],
            "roughness": p["eps_well"],
            "geometry": {
                "points": [{"xyz": [0.0, 0.0, g["z0"]]}, {"xyz": [0.0, 0.0, g["z_inj"]]}],
                "segments": [{"n": int(p["n_annulus"])}],
            },
            "model": _model(p),
            "scheme": _scheme(1e-3, qG_gl0),
            "ports": {"in": dict(ports_in), "out": dict(ports_out)},
        },
        {
            "id": "check_an",
            "type": "check_valve",
            "from": "an_bot",
            "to": "inj",
            "CV": float(p["CV_check"]),
            "dP_min_bar": float(p["dP_min_check_bar"]),
        },
        {"id": "connector_an_inj", "type": "connector", "from": "an_bot", "to": "inj"},
    ]
    return vertices, edges, g


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
            f"(PI {p['PI']:g} Sm³/d/bar, P_res {p['P_res_bar']:g} bar, RGO {p['GOR']:g} Sm³/Sm³), linha de {p['L_line']:g} m com "
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
            **({"error_estimate": p["error_estimate"]} if p["error_estimate"] != "thesis" else {}),
        },
        "initial_state": {"source": "guess"},
    }
    if p["gas_lift"]:
        vertices, edges, g = _gas_lift_block(
            p, info, PR, qL0, qG0, ports_in, ports_out, rhoL0, p["P_sep_bar"], points, segments
        )
        well = next(e for e in case["network"]["edges"] if e["id"] == "well")
        well["from"] = "inj"
        well["geometry"]["points"] = [{"xyz": [0.0, 0.0, g["z_inj"]]}, {"xyz": [0.0, 0.0, g["z0"]]}]
        case["network"]["vertices"] += vertices
        case["network"]["edges"] += edges
        case["name"] = "offshore_gaslift"
        case["description"] += (
            f" Gas lift: linha de gás de {p['D_gas_line']:g} m da plataforma a {p['P_gas_bar']:g} bar até à cabeça "
            f"do poço, anular (D equivalente {p['D_annulus']:g} m) até {p['L_inj']:g} m do fundo, retenção CV "
            f"{p['CV_check']:g} para o tubing; todos os dutos com o modelo a dois fluidos."
        )
        case["meta"]["geometry"] = dict(info, **{f"gl_{k}": v for k, v in g.items()})
        case["integration"]["slow_exit"] = True  # eventos da retenção (issue 033)
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

    from mfsim.network.elements import check_valve_flows
    from mfsim.network.graph import CheckValve, TankVertex

    PR, fluid = model.PR, model.fluid
    ch = [e for e in net.edges if isinstance(e, Choke)][0]
    rs = [e for e in net.edges if isinstance(e, Reservoir)][0]
    P_sep = [v for v in net.vertices if v.id == ch.v_to][0].spec["PT"](0.0) * PR
    ab = float(ch.opening(0.0))
    pipes = {pb.pipe.id: pb for pb in model.pipes}
    line, well = pipes["1"], pipes["well"]
    gas_lift = "gas_line" in pipes
    s_top = float(np.asarray(line.scheme.x_nodes)[-1]) - float(net.meta["generator_params"]["L_h2"]) - 1.0
    q_inj = 0.0
    if gas_lift:
        # coluna de gás estática da plataforma à válvula (linha de gás e anular): pressão na base do anular
        cv = [e for e in net.edges if isinstance(e, CheckValve)][0]
        P_gas = [v for v in net.vertices if v.id == "gl_src"][0].spec["PT"](0.0) * PR
        P_gl, _ = _gas_column(pipes["gas_line"], fluid, P_gas)
        P_an, _ = _gas_column(pipes["annulus"], fluid, P_gl[-1])
        P_an_bot = P_an[-1]
        a_gas = float(net.meta["generator_params"].get("gas_alpha", 0.98))

    def profile(m, q_inj):
        qL, qG = (1 - w_gas) * m, w_gas * m
        ZP, _ = valve_upstream_pressure(P_sep / PR, qL, qG + q_inj, ab, ch.CV, fluid, PR, law=ch.law)
        xs = np.asarray(line.scheme.x_nodes, float)
        th_n = np.asarray(line.scheme.theta_nodes, float)
        qG_up = qG + q_inj  # acima da injeção a linha e o tubing superior levam também o gás do gas lift
        P_line, a_line = _march(
            line,
            fluid,
            float(ZP) * PR,
            qL,
            qG_up,
            lambda P, th: _node_alpha(mode, line, fluid, P, qL, qG_up, th, True, False),
        )
        if mode == "blocked":
            a_line = np.array(
                [
                    _node_alpha(mode, line, fluid, P_line[i], qL, qG_up, th_n[i], True, xs[i] > s_top)
                    for i in range(xs.size)
                ]
            )
        P_well, a_well = _march(
            well,
            fluid,
            P_line[0],
            qL,
            qG_up,
            lambda P, th: _node_alpha(mode, well, fluid, P, qL, qG_up, th, False, False),
        )
        out = dict(qL=qL, qG=qG, qG_up=qG_up, line=(P_line, a_line), well=(P_well, a_well), P_bottom=P_well[0])
        if gas_lift:
            wl = pipes["well_lower"]
            P_wl, a_wl = _march(
                wl, fluid, P_well[0], qL, qG, lambda P, th: _node_alpha(mode, wl, fluid, P, qL, qG, th, False, False)
            )
            out.update(well_lower=(P_wl, a_wl), P_bottom=P_wl[0], P_inj=P_well[0])
        return out

    def resid(m):
        return m - (rs.IPL + rs.IPG) * max(0.0, rs.P - profile(m, q_inj)["P_bottom"] / 1e5)

    def solve_m():
        lo, hi = 1e-3, (rs.IPL + rs.IPG) * rs.P
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0:
                hi = mid
            else:
                lo = mid
        return max(0.5 * (lo + hi), 1e-3)

    m = solve_m()
    if gas_lift:
        # injeção da lei da retenção com a pressão do tubing na injeção; uma repetição chega para a estimativa
        for _ in range(2):
            P_inj = profile(m, q_inj)["P_inj"]
            _, qG_v, _ = check_valve_flows(P_an_bot / PR, P_inj / PR, a_gas, cv.CV, cv.dP_min, fluid, PR)
            q_inj = max(float(qG_v), 0.0)
            m = solve_m()
    pr = profile(m, q_inj)
    X = np.zeros(model.n_states)
    blocks = [(line, pr["line"], pr["qL"], pr["qG_up"]), (well, pr["well"], pr["qL"], pr["qG_up"])]
    if gas_lift:
        blocks.append((pipes["well_lower"], pr["well_lower"], pr["qL"], pr["qG"]))
        blocks.append((pipes["gas_line"], (P_gl, np.full(P_gl.size, a_gas)), 0.0, q_inj))
        blocks.append((pipes["annulus"], (P_an, np.full(P_an.size, a_gas)), 0.0, q_inj))
    for pb, (P, a), qL_b, qG_b in blocks:
        Y = pb.base_Y()
        Y[:, SLOT_ALPHA] = np.clip(a, 0.02, 0.98 if pb.pipe.id not in ("gas_line", "annulus") else a_gas)
        Y[:, SLOT_P] = P / PR
        Y[:, SLOT_QL] = qL_b
        Y[:, SLOT_QG] = qG_b
        if hasattr(pb.scheme, "from_primitive"):
            Y = pb.scheme.from_primitive(Y)
        X[pb.offset : pb.offset + pb.n_states] = Y.ravel()[pb.free]
    if gas_lift:
        for v in net.vertices:
            if isinstance(v, TankVertex):
                iH, iP = model.vertex_A_index(v.id)
                X[iH], X[iP] = a_gas, P_an_bot / PR
    return X, m


def _gas_column(pb, fluid, P_in):
    """Pressão (Pa) nos nós do esquema de um duto só com gás, marchada da entrada com a coluna estática
    (dP = rho_G g dz, sem atrito); devolve (P, z)."""
    sch = pb.scheme
    x = np.asarray(sch.x_nodes, float)
    zc = np.concatenate([[0.0], np.cumsum(np.sin(pb.mesh["theta"]) * pb.mesh["h"])])
    z = np.interp(x, pb.mesh["s"], zc)
    P = np.zeros_like(x)
    P[0] = P_in
    for i in range(1, x.size):
        _, rhoG, _, _ = fluid.props(P[i - 1])
        P[i] = P[i - 1] - rhoG * G * (z[i] - z[i - 1])
    return P, z


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
