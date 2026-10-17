"""Sistema de produção offshore do artigo 01: poço vertical, linha no fundo do mar com um trecho final descendente até
à base do riser, curva, riser vertical até à superfície e choke no topo, a descarregar num separador.

A geometria e o fluido são os do caso offshore da dissertação de mestrado de Gustavo Alves de Carvalho (UERJ, 2022,
§5.5): poço vertical de 1500 m (D 0,127 m, rugosidade 0,2 mm), linha de 6000 m e riser de 300 m (D 0,1524 m,
rugosidade 0,6 mm), fluido de 9 componentes com C7+ (Tabela 19), 170 bar e 10 kg/s no fundo do poço. Diferenças:

- a linha termina num trecho a −2° (`L_down`, medido ao longo do duto, dentro dos 6000 m) até à curva da base do riser
  (arco de raio 5 m), e o riser sobe até ao nível do mar (300 m + o desnível desse trecho), com um troço horizontal
  curto no topo até ao choke;
- o mfsim não tem transferência de massa nem equação de energia: o fluido é um par politrópico ajustado ao flash de
  Peng–Robinson do fluido do Gustavo (cmfsim) a 40 °C — densidades e viscosidades a 20 bar, expoente do gás ajustado
  entre 10 e 100 bar — e a fração mássica de gás fica congelada na das condições da linha e do riser (0,29 a 20 bar;
  0,24 a 100 bar, 0,25 no fundo do poço);
- o fundo do poço recebe uma IPR linear que passa pelo ponto nominal (170 bar, 10 kg/s) com a pressão do reservatório
  `P_res_nominal_bar`; a pressão do reservatório do caso (`P_res_bar`) é a variável do mapa (depleção) e o índice de
  produtividade fica fixo.

O poço e a linha são dois dutos (diâmetros diferentes) ligados por um conector na cabeça do poço; a linha com o riser é
o primeiro duto da rede (`model.pipes[0]`, como no exemplo anterior), o poço o segundo.

    uv run python -m generators.offshore --out cases/paper01/offshore.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

G = 9.80665
PR = 1e6

# fluido do §5.5 congelado (flash de Peng–Robinson do cmfsim a 40 °C; ver docstring)
FLUID = dict(
    type="polytropic",
    rhoL0=696.0,
    rhoG0=17.08,
    muL=1.25e-3,
    muG=1.18e-5,
    gammaL=1000.0,
    gammaG=0.95,
    P0=20e5,
    sigma=0.0218,
)
W_GAS = 0.29  # fração mássica de gás congelada (condições da linha, 20 bar e 40 °C)
# fluido da Tabela 6 (§5.7, o do modelo substituto, no mesmo sistema; 165 bar e 13,5 kg/s no fundo do poço), congelado
# da mesma maneira: flash a 40 °C, densidades e viscosidades a 20 bar, gás ideal entre 10 e 60 bar
FLUID_T6 = dict(
    type="polytropic",
    rhoL0=646.7,
    rhoG0=15.0,
    muL=3.14e-4,
    muG=1.21e-5,
    gammaL=1000.0,
    gammaG=1.0,
    P0=20e5,
    sigma=0.0163,
)
W_GAS_T6 = 0.099

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
    # fluido e condições de contorno
    fluid=None,
    w_gas=W_GAS,
    m_nominal=10.0,
    P_wf_nominal_bar=170.0,
    P_res_nominal_bar=200.0,
    P_res_bar=200.0,
    P_sep_bar=10.0,
    choke_cv=150.0,
    choke_opening=0.2,
    # modelo e integração
    slug_frequency="zabaras",
    slug_unit="consistent",
    k_switch=20.0,
    tf=20000.0,
    dt_out=5.0,
    rtol=3e-3,
    tol_range=0.01,
    max_step=None,
    q_ref_factor=10.0,
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
    PI = p["m_nominal"] / (p["P_res_nominal_bar"] - p["P_wf_nominal_bar"])  # kg/s/bar (total)
    IPL, IPG = PI * (1.0 - p["w_gas"]), PI * p["w_gas"]
    qL0 = IPL * (p["P_res_bar"] - p["P_wf_nominal_bar"])  # estimativa inicial das vazões
    qG0 = IPG * (p["P_res_bar"] - p["P_wf_nominal_bar"])
    qL0, qG0 = max(qL0, 0.1 * IPL * p["P_res_bar"]), max(qG0, 0.1 * IPG * p["P_res_bar"])
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
            "Sistema offshore do artigo 01 (geometria e fluido do caso §5.5 da dissertação de Gustavo Alves de "
            "Carvalho, UERJ 2022): poço vertical de 1500 m com IPR linear no fundo, linha de 6000 m com trecho final a "
            f"{p['slope_down_deg']:g}° ({p['L_down']:g} m), curva de raio {p['elbow_radius']:g} m, riser até à "
            f"superfície, choke CV {p['choke_cv']:g} no topo e separador a {p['P_sep_bar']:g} bar; fluido do Gustavo "
            "congelado num par politrópico, padrão de escoamento dinâmico, frequência de golfada de Zabaras (049). "
            "Gerado por generators/offshore.py."
        ),
        "meta": {
            "generator": "generators.offshore",
            "generator_params": {k: v for k, v in p.items()},
            "geometry": info,
            "PI_total": PI,
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
        },
        "initial_state": {"source": "guess"},
    }
    return case


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", action="append", default=[], help="nome=valor (JSON) para substituir um parâmetro")
    a = ap.parse_args(argv)
    kw = {}
    for item in a.param:
        k, v = item.split("=", 1)
        kw[k] = json.loads(v)
    case = offshore_case(**kw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
