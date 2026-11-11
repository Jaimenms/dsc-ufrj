"""Tabelas de propriedades para o OLGA (`tools/olga_tables.py`): a tabela reproduz a lei politrópica do caso, a fração
de gás congelada e as regras do OLGA (malha bidimensional com a pressão padrão, duas fases em todos os nós)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("olga_tables", ROOT / "tools" / "olga_tables.py")
ot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ot)


def test_table_matches_the_polytropic_fluid_and_olga_rules(tmp_path):
    case = json.load(open(ROOT / "cases" / "paper01" / "offshore.json", encoding="utf-8"))
    rows, header = ot.property_table(case)
    f = case["fluid"]
    assert len(rows) == len(ot.P_GRID_BAR) * 2 and abs(rows[0]["PT"] - ot.P_STD) < 1e-6
    names = [c for c, _ in ot.COLUMNS]
    assert all(list(r.keys()) == names for r in rows)
    for r in rows:
        P = r["PT"]
        assert np.isclose(r["ROG"], f["rhoG0"] * (P / f["P0"]) ** (1 / f["gammaG"]))
        assert np.isclose(r["DROGDP"], r["ROG"] / (f["gammaG"] * P)) and np.isclose(
            r["DROHLDP"], r["ROHL"] / (f["gammaL"] * P)
        )
        assert r["DROGDT"] == 0.0 and r["RS"] == case["meta"]["mass"]["w_gas"] and r["ROHL"] > 0 and r["ROG"] > 0
        assert all(np.isfinite(v) for v in r.values())
    assert header["GOR_Sm3_Sm3"] == case["meta"]["market"]["GOR_Sm3_Sm3"]
    # outra RGO só muda a fração de gás (mesmo óleo e gás)
    rows2, header2 = ot.property_table(case, gor=122.13)
    assert abs(header2["w_gas"] - 0.0988) < 1e-3 and rows2[0]["ROG"] == rows[0]["ROG"]
    out = ot.write_csv(rows, header, tmp_path / "t.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# {") and lines[2] == ",".join(names) and len(lines) == 3 + len(rows)
