"""Tabelas de propriedades para o OLGA a partir de um caso do mfsim (pedido do dono, 2026-11-10; ver
docs/olga/README.md).

O OLGA lê as propriedades de um ficheiro `.tab` com uma malha bidimensional em pressão e temperatura e, em cada nó,
as colunas PT, TM, ROG, ROHL, DROGDP, DROHLDP, DROGDT, DROHLDT, RS, VISG, VISHL, CPG, CPHL, HG, HHL, TCG, TCHL,
SIGGHL, SEG, SEHL. O fluido do mfsim é um par politrópico sem equação de energia nem transferência de massa, pelo
que as densidades e as suas derivadas em pressão vêm da lei ρ = ρ0 (P/P0)^(1/γ), as derivadas em temperatura são
zero, as viscosidades e a tensão interfacial são constantes, a fração mássica de gás (RS) é a fração congelada w_G
da RGO do caso (`meta.mass.w_gas`), e as colunas térmicas (capacidades, entalpias, condutividades, entropias) são
preenchidas com valores plausíveis e coerentes entre si, que não contam quando o OLGA corre com o cálculo de
temperatura desligado. A malha de pressão inclui a pressão padrão (1,01325 bar) e vai até `--P-max`; a temperatura é
repetida em dois valores (`--T`), que têm de enquadrar a temperatura padrão.

    uv run python tools/olga_tables.py cases/paper01/offshore.json --out docs/olga/offshore_gor50.csv
    uv run python tools/olga_tables.py cases/paper01/offshore_gor30.json --gor 30 --out docs/olga/offshore_gor30.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = (
    ("PT", "Pa"),
    ("TM", "C"),
    ("ROG", "kg/m3"),
    ("ROHL", "kg/m3"),
    ("DROGDP", "s2/m2"),
    ("DROHLDP", "s2/m2"),
    ("DROGDT", "kg/m3/C"),
    ("DROHLDT", "kg/m3/C"),
    ("RS", "-"),
    ("VISG", "Ns/m2"),
    ("VISHL", "Ns/m2"),
    ("CPG", "J/kg/C"),
    ("CPHL", "J/kg/C"),
    ("HG", "J/kg"),
    ("HHL", "J/kg"),
    ("TCG", "W/m/C"),
    ("TCHL", "W/m/C"),
    ("SIGGHL", "N/m"),
    ("SEG", "J/kg/C"),
    ("SEHL", "J/kg/C"),
)
P_STD, T_STD = 1.01325e5, 20.0  # condições padrão da ANP (as do caso: `meta.pvt.std`)
# colunas térmicas que o mfsim não tem: valores típicos de um gás natural leve e de um óleo leve, constantes
THERMAL = dict(CPG=2200.0, CPHL=2000.0, TCG=0.03, TCHL=0.13)
R_GAS = 8.314  # J/mol/K

# malha de pressão de omissão (bar): densa abaixo de 30 bar, onde a densidade do gás muda mais em termos relativos
P_GRID_BAR = [P_STD / 1e5, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] + [
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
]


def fluid_mass_split(case, gor=None):
    """Fração mássica de gás livre congelada do caso (`meta.mass.w_gas`) ou recalculada para outra RGO com o PVT do
    caso (mesma regra do gerador do offshore: óleo saturado com R_s e o resto do gás livre)."""
    meta = case.get("meta", {})
    pvt = meta.get("pvt")
    if gor is None:
        return float(meta["mass"]["w_gas"]), meta["market"]["GOR_Sm3_Sm3"], pvt
    free = max(gor - pvt["Rs"], 0.0)
    m_oil = pvt["rho_o_std"] + pvt["rho_gd_std"] * min(gor, pvt["Rs"]) + pvt["rho_gf_std"] * free
    return pvt["rho_gf_std"] * free / m_oil, gor, pvt


def property_table(case, P_bar=P_GRID_BAR, T_C=(20.0, 60.0), gor=None):
    """Linhas da tabela (uma por nó, pressão exterior, temperatura interior) com as colunas do OLGA; devolve também
    os metadados do cabeçalho (RGO, densidades padrão, fração de gás)."""
    f = case["fluid"]
    if f.get("type", "polytropic") != "polytropic":
        raise SystemExit(f"fluido {f.get('type')}: só o par politrópico é suportado")
    w_gas, gor_used, pvt = fluid_mass_split(case, gor)
    M_gas = f["rhoG0"] * R_GAS * 313.15 / f["P0"]  # kg/mol do gás ideal ajustado a P0 e 40 °C
    rows = []
    for Pb in P_bar:
        P = Pb * 1e5
        rhoG = f["rhoG0"] * (P / f["P0"]) ** (1.0 / f["gammaG"])
        rhoL = f["rhoL0"] * (P / f["P0"]) ** (1.0 / f["gammaL"])
        for T in T_C:
            TK = T + 273.15
            rows.append(
                dict(
                    PT=P,
                    TM=T,
                    ROG=rhoG,
                    ROHL=rhoL,
                    DROGDP=rhoG / (f["gammaG"] * P),
                    DROHLDP=rhoL / (f["gammaL"] * P),
                    DROGDT=0.0,
                    DROHLDT=0.0,
                    RS=w_gas,
                    VISG=f["muG"],
                    VISHL=f["muL"],
                    CPG=THERMAL["CPG"],
                    CPHL=THERMAL["CPHL"],
                    HG=THERMAL["CPG"] * T,  # referência: 0 °C
                    HHL=THERMAL["CPHL"] * T,
                    TCG=THERMAL["TCG"],
                    TCHL=THERMAL["TCHL"],
                    SIGGHL=f["sigma"],
                    SEG=THERMAL["CPG"] * np.log(TK / 273.15) - R_GAS / M_gas * np.log(P / P_STD),
                    SEHL=THERMAL["CPHL"] * np.log(TK / 273.15),
                )
            )
    header = dict(
        GOR_Sm3_Sm3=gor_used,
        w_gas=w_gas,
        rho_oil_std=pvt["rho_o_std"] if pvt else None,
        rho_gas_std=f["rhoG0"] * (P_STD / f["P0"]) ** (1.0 / f["gammaG"]),
        P_std_Pa=P_STD,
        T_std_C=T_STD,
        M_gas_kg_mol=M_gas,
    )
    return rows, header


def write_csv(rows, header, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# " + json.dumps(header) + "\n")
        fh.write("# " + ",".join(f"{c} [{u}]" for c, u in COLUMNS) + "\n")
        w = csv.DictWriter(fh, fieldnames=[c for c, _ in COLUMNS])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--out", required=True, help="CSV de saída")
    ap.add_argument("--gor", type=float, default=None, help="RGO (Sm³/Sm³) em vez da do caso: muda só a fração de gás")
    ap.add_argument("--P-max", type=float, default=300.0, help="pressão máxima da malha [bar]")
    ap.add_argument("--T", type=float, nargs=2, default=(20.0, 60.0), help="as duas temperaturas da malha [°C]")
    a = ap.parse_args(argv)
    case = json.load(open(a.case, encoding="utf-8"))
    P = [p for p in P_GRID_BAR if p <= a.P_max + 1e-9]
    if not (min(a.T) <= T_STD <= max(a.T)):
        raise SystemExit(f"as temperaturas {a.T} têm de enquadrar a padrão ({T_STD} °C)")
    rows, header = property_table(case, P_bar=P, T_C=a.T, gor=a.gor)
    out = write_csv(rows, header, a.out)
    print(
        f"{out}: {len(rows)} nós ({len(P)} pressões × {len(a.T)} temperaturas); RGO {header['GOR_Sm3_Sm3']:g}, w_gas {header['w_gas']:.4f}"
    )


if __name__ == "__main__":
    main()
