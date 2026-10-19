"""Ferramenta da massa de dados do artigo 02 (`tools/paper02_dataset.py`): desenho de Sobol em cadeias de vizinhos,
marcadores de um sinal e escrita da tabela; nada simula."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper02_dataset", ROOT / "tools" / "paper02_dataset.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def test_design_covers_ranges_and_chains():
    ranges = ds.parse_ranges(["P_gas_bar=60:150"])
    assert ranges["P_gas_bar"] == (60.0, 150.0) and ranges["PI"] == ds.DEFAULT_RANGES["PI"]
    X = ds.sobol_design(40, ranges, seed=3)
    assert X.shape == (40, 5)
    for j, k in enumerate(ds.INPUTS):
        assert ranges[k][0] <= X[:, j].min() and X[:, j].max() <= ranges[k][1]
    assert np.array_equal(X, ds.sobol_design(40, ranges, seed=3))  # determinista com a semente
    chain, order = ds.make_chains(X, ranges, 4, seed=3)
    assert set(chain.tolist()) == set(range(4)) and chain.min() >= 0
    for c in range(4):
        idx = np.flatnonzero(chain == c)
        assert sorted(order[idx].tolist()) == list(range(idx.size))  # a ordem é uma permutação dentro da cadeia
    chain1, order1 = ds.make_chains(X, ranges, 1, seed=3, start="centre")
    assert chain1.max() == 0 and sorted(order1.tolist()) == list(range(40))
    U = ds.scaled(X, ranges)
    assert U.min() >= -1.0 and U.max() <= 1.0 and U.shape == X.shape
    start = int(np.flatnonzero(order1 == 0)[0])
    assert start == int(np.argmin(np.linalg.norm(U - U.mean(axis=0), axis=1)))  # parte do ponto mais perto do centro
    _, order_r = ds.make_chains(X, ranges, 1, seed=3, start="random")  # sorteado, determinista com a semente
    _, order_r2 = ds.make_chains(X, ranges, 1, seed=3, start="random")
    assert np.array_equal(order_r, order_r2) and sorted(order_r.tolist()) == list(range(40))


def test_signal_markers():
    T = np.arange(0.0, 4000.0, 5.0)
    const = ds.signal_markers(T, 20.0 + 0.01 * np.sin(T / 50.0), amp_tol=0.1)
    assert const["fmin"] == 0.0 and const["fmax"] == 0.0 and abs(const["mean"] - 20.0) < 0.01
    per = ds.signal_markers(T, 20.0 + 3.0 * np.sin(2 * np.pi * T / 500.0), amp_tol=0.1)
    assert abs(per["fmin"] - 1 / 500.0) < 2e-5 and abs(per["fmax"] - 1 / 500.0) < 2e-5
    assert abs(per["max"] - 23.0) < 1e-6 and abs(per["min"] - 17.0) < 1e-6
    # sopros alternados: máximos a intervalos de 400 e 800 s → f_min = 1/800, f_max = 1/400
    y = np.zeros_like(T)
    for t in (300.0, 700.0, 1500.0, 1900.0, 2700.0, 3100.0, 3900.0):
        y += 5.0 * np.exp(-(((T - t) / 40.0) ** 2))
    alt = ds.signal_markers(T, 20.0 + y, amp_tol=0.1)
    assert abs(alt["fmin"] - 1 / 800.0) < 2e-5 and abs(alt["fmax"] - 1 / 400.0) < 2e-5


def test_write_dataset(tmp_path):
    design = dict(ranges={k: list(v) for k, v in ds.DEFAULT_RANGES.items()}, settle={})
    (tmp_path / "design.json").write_text(json.dumps(design))
    row = dict(
        id="p0",
        chain=0,
        order=0,
        start="blocked",
        from_point=None,
        P_res_bar=200.0,
        PI=5.0,
        GOR=50.0,
        P_gas_bar=20.0,
        choke_opening=1.0,
    )
    row.update({"class": "slugging", "period": 1300.0, "wellhead_P_mean": 25.5, "q_inj_mean": 0.0})
    path = ds.write_dataset(tmp_path, [row], design)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["class"] == "slugging" and rows[0]["wellhead_P_mean"] == "25.5"
    cols = ds.columns()
    assert len(rows[0]) == len(cols) and "riser_top_qL_max" in cols and "annulus_base_P_mean" in cols
    meta = json.loads((tmp_path / "dataset.json").read_text())
    assert meta["n_rows"] == 1 and meta["columns"] == cols


def test_shortest_path_from_centre():
    rng = np.random.default_rng(0)
    U = rng.random((8, 2))
    start = int(np.argmin(np.linalg.norm(U - U.mean(axis=0), axis=1)))
    path = ds.shortest_path_from(U, start)
    assert path[0] == start and sorted(path) == list(range(8))
    D = np.linalg.norm(U[:, None] - U[None, :], axis=-1)
    # exato: nenhuma troca 2-opt melhora
    for i in range(1, 7):
        for j in range(i + 1, 8):
            cand = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]
            assert ds.path_length(D, cand) >= ds.path_length(D, path) - 1e-12
    big = ds.shortest_path_from(rng.random((25, 3)), 0)
    assert big[0] == 0 and sorted(big) == list(range(25))


def test_rounds_continue_the_sobol_sequence(tmp_path):
    ranges = ds.DEFAULT_RANGES
    X20 = ds.sobol_design(20, ranges, seed=5)
    assert np.allclose(ds.sobol_design(10, ranges, seed=5, skip=10), X20[10:])  # a 2.ª ronda continua a sequência
    design = dict(
        seed=5,
        n=10,
        round_size=10,
        rounds=1,
        ranges={k: list(v) for k, v in ranges.items()},
        test=False,
        points=[
            dict(id=f"p{i:03d}", x={k: float(X20[i, j]) for j, k in enumerate(ds.INPUTS)}, chain=0, order=i, round=0)
            for i in range(10)
        ],
    )
    new = ds.append_round(design)
    assert len(new) == 10 and design["n"] == 20 and design["rounds"] == 2
    assert all(p["chain"] == 1 and p["round"] == 1 for p in new) and sorted(p["order"] for p in new) == list(range(10))
    assert np.allclose([[p["x"][k] for k in ds.INPUTS] for p in sorted(new, key=lambda p: p["sobol_index"])], X20[10:])


def test_design_never_overwrites_runs(tmp_path):
    import pytest

    ds.main(
        ["design", "--flat", "--start", "random", "--out", str(tmp_path), "--n", "4", "--chains", "1", "--seed", "2"]
    )
    d0 = json.loads((tmp_path / "design.json").read_text())
    first = d0["points"][0]["id"]
    assert len(first) == 6 and first == ds.point_id(d0["points"][0]["x"])
    (tmp_path / first).mkdir()
    (tmp_path / first / "point.json").write_text("{}")  # um ponto feito (só point.json + final_state.npz ficam)
    with pytest.raises(SystemExit):
        ds.main(
            [
                "design",
                "--flat",
                "--start",
                "random",
                "--out",
                str(tmp_path),
                "--n",
                "4",
                "--chains",
                "1",
                "--seed",
                "2",
                "--force",
            ]
        )
    assert json.loads((tmp_path / "design.json").read_text()) == d0
    ds.main(["design", "--flat", "--start", "random", "--out", str(tmp_path), "--n", "4", "--chains", "1", "--append"])
    d1 = json.loads((tmp_path / "design.json").read_text())
    assert d1["n"] == 8 and d1["rounds"] == 2 and d1["points"][:4] == d0["points"]
    assert ds.point_status(tmp_path, first) == "done" and ds.point_status(tmp_path, d1["points"][4]["id"]) == "pending"
    assert len({p["id"] for p in d1["points"]}) == 8


def test_rounds_in_dated_folders(tmp_path):
    import pytest

    root = tmp_path / "rounds"
    ds.main(["design", "--start", "random", "--out", str(root), "--n", "4", "--seed", "9"])
    master = json.loads((root / "rounds.json").read_text())
    assert master["round_size"] == 4 and master["n"] == 4 and len(master["rounds"]) == 1
    d0 = root / master["rounds"][0]["folder"]
    assert d0.name.isdigit() and len(d0.name) == 10 and d0.name.endswith("01") and (d0 / "design.json").exists()
    with pytest.raises(SystemExit):
        ds.main(
            ["design", "--start", "random", "--out", str(root), "--n", "4", "--seed", "9"]
        )  # a raiz nunca se substitui
    ds.main(["design", "--start", "random", "--out", str(root), "--append"])
    master = json.loads((root / "rounds.json").read_text())
    assert len(master["rounds"]) == 2 and master["n"] == 8
    d1 = root / master["rounds"][1]["folder"]
    assert d1 != d0 and d1.name == d0.name[:8] + "02"  # contador por dia, não a hora
    pts1 = json.loads((d1 / "design.json").read_text())["points"]
    X8 = ds.sobol_design(8, ds.DEFAULT_RANGES, seed=9)
    assert [p["sobol_index"] for p in sorted(pts1, key=lambda p: p["sobol_index"])] == [4, 5, 6, 7]
    assert all(p["id"] == ds.point_id(p["x"]) for p in pts1) and len(master["ids"]) == 8 == len(set(master["ids"]))
    assert np.allclose([[p["x"][k] for k in ds.INPUTS] for p in sorted(pts1, key=lambda p: p["sobol_index"])], X8[4:])
    assert sorted(p["order"] for p in pts1) == [0, 1, 2, 3] and all(p["round"] == 1 for p in pts1)


def test_first_point_closest_to_base_and_resequence(tmp_path):
    ranges = ds.DEFAULT_RANGES
    X = ds.sobol_design(12, ranges, seed=4)
    x_base = {"P_res_bar": 200.0, "PI": 5.0, "GOR": 50.0, "P_gas_bar": 20.0, "choke_opening": 1.0}
    chain, order = ds.make_chains(X, ranges, 1, seed=4, start="base", x_base=x_base)
    U = ds.scaled(X, ranges)
    ub = ds.scaled(np.array([[x_base[k] for k in ds.INPUTS]]), ranges)[0]
    assert int(np.flatnonzero(order == 0)[0]) == int(np.argmin(np.linalg.norm(U - ub, axis=1)))
    # reordenar uma ronda sem corridas com a regra do caso base
    root = tmp_path / "rounds"
    ds.main(["design", "--start", "random", "--out", str(root), "--n", "6", "--seed", "4"])
    master = json.loads((root / "rounds.json").read_text())
    d = root / master["rounds"][0]["folder"]
    master.update(start="base", x_base=x_base, base_run=None)
    ds.resequence_round(d, master)
    pts = json.loads((d / "design.json").read_text())["points"]
    first = next(p for p in pts if p["order"] == 0)
    Xr = np.array([[p["x"][k] for k in ds.INPUTS] for p in sorted(pts, key=lambda p: p["sobol_index"])])
    Ur = ds.scaled(Xr, ranges)
    assert (
        sorted(pts, key=lambda p: p["sobol_index"])[int(np.argmin(np.linalg.norm(Ur - ub, axis=1)))]["id"]
        == first["id"]
    )
    (d / first["id"]).mkdir()
    (d / first["id"] / "point.json").write_text("{}")
    import pytest

    with pytest.raises(SystemExit):
        ds.resequence_round(d, master)  # com corridas feitas não se reordena


def test_point_id_is_a_hash_of_the_inputs():
    x = {"P_res_bar": 200.0, "PI": 5.0, "GOR": 50.0, "P_gas_bar": 20.0, "choke_opening": 1.0}
    assert ds.point_id(x) == ds.point_id(dict(x)) and len(ds.point_id(x)) == 6
    assert ds.point_id(dict(x, PI=5.0001)) != ds.point_id(x)
    ids = ds.unique_ids([x, dict(x)])  # o mesmo tuplo duas vezes: o segundo leva mais caracteres
    assert ids[0] != ids[1] and ids[1].startswith(ids[0])


def test_point_record_is_what_the_table_reads(tmp_path):
    rec = dict(
        id="abc123",
        x={"P_res_bar": 200.0, "PI": 5.0, "GOR": 50.0, "P_gas_bar": 20.0, "choke_opening": 1.0},
        chain=0,
        order=0,
        round=0,
        sobol_index=0,
        start="base",
        from_point="base",
        markers={"class": "stable", "wellhead_P_mean": 17.9, "q_inj_mean": 0.0},
    )
    ds.save_point(tmp_path, rec, np.arange(5.0))
    assert ds.point_status(tmp_path, "abc123") == "done"
    assert np.array_equal(ds._final_state(tmp_path, "abc123"), np.arange(5.0))
    row = ds._row_from_record(json.loads(ds.point_path(tmp_path, "abc123").read_text()))
    assert row["wellhead_P_mean"] == 17.9 and row["P_res_bar"] == 200.0 and row["start"] == "base"
    assert sorted(p.name for p in (tmp_path / "abc123").iterdir()) == ["final_state.npz", "point.json"]


def test_grid_decisions_snake_is_the_shortest_warm_start_path(tmp_path):
    ranges = ds.DEFAULT_RANGES
    well = {"P_res_bar": 150.0, "PI": 7.0, "GOR": 80.0}
    x_base = {"P_res_bar": 200.0, "PI": 5.0, "GOR": 50.0, "P_gas_bar": 20.0, "choke_opening": 1.0}
    seq = ds.grid_decisions(well, ranges, 6, 6, x_base)
    assert len(seq) == 36 and len({(x["P_gas_bar"], x["choke_opening"]) for x in seq}) == 36
    assert all(x["P_res_bar"] == 150.0 and x["GOR"] == 80.0 for x in seq)
    # começa no canto mais perto do caso base: gás mínimo (20 bar), choke aberto (1,0)
    assert seq[0]["P_gas_bar"] == 20.0 and seq[0]["choke_opening"] == 1.0
    g = np.linspace(20.0, 160.0, 6)
    o = np.linspace(0.1, 1.0, 6)
    for x0, x1 in zip(seq[:-1], seq[1:]):
        dg = abs(np.searchsorted(g, x1["P_gas_bar"]) - np.searchsorted(g, x0["P_gas_bar"]))
        do = abs(np.searchsorted(o, x1["choke_opening"]) - np.searchsorted(o, x0["choke_opening"]))
        assert dg + do == 1  # cada passo muda uma só decisão de um só nível
        assert x1["P_gas_bar"] >= x0["P_gas_bar"]  # P_gas nunca desce
    # raiz em modo malha: uma ronda por estado do poço, 36 pontos, poço seguinte na ronda seguinte
    root = tmp_path / "grid"
    ds.main(["design", "--start", "random", "--out", str(root), "--grid-decisions", "6x6", "--seed", "3"])
    master = json.loads((root / "rounds.json").read_text())
    assert master["mode"] == "grid" and master["round_size"] == 36 and master["n"] == 1
    d0 = root / master["rounds"][0]["folder"]
    pts = json.loads((d0 / "design.json").read_text())["points"]
    assert len(pts) == 36 and [p["order"] for p in pts] == list(range(36)) and all(p["chain"] == 0 for p in pts)
    ds.main(["design", "--start", "random", "--out", str(root), "--append"])
    master = json.loads((root / "rounds.json").read_text())
    assert master["n"] == 2 and master["rounds"][1]["well"] != master["rounds"][0]["well"]
    W = ds.sobol_wells(2, ranges, seed=3)
    assert np.allclose([master["rounds"][1]["well"][k] for k in ds.WELL_INPUTS], W[1])


def test_fixed_well_sobol_in_two_dimensions():
    ranges = dict(ds.DEFAULT_RANGES, P_res_bar=(200.0, 200.0), PI=(5.0, 5.0), GOR=(50.0, 50.0))
    X = ds.sobol_design(25, ranges, seed=1)
    assert X.shape == (25, 5)
    assert np.all(X[:, 0] == 200.0) and np.all(X[:, 1] == 5.0) and np.all(X[:, 2] == 50.0)
    assert len(set(np.round(X[:, 3], 6))) == 25 and X[:, 3].min() >= 20.0 and X[:, 4].max() <= 1.0
    U = ds.scaled(X, ranges)
    assert np.all(U[:, :3] == 0.0) and U[:, 3:].min() >= -1.0 and U[:, 3:].max() <= 1.0
    x_base = {"P_res_bar": 200.0, "PI": 5.0, "GOR": 50.0, "P_gas_bar": 20.0, "choke_opening": 1.0}
    chain, order = ds.make_chains(X, ranges, 1, seed=1, start="base", x_base=x_base)
    ub = ds.scaled(np.array([[x_base[k] for k in ds.INPUTS]]), ranges)[0]
    assert int(np.flatnonzero(order == 0)[0]) == int(np.argmin(np.linalg.norm(U - ub, axis=1)))
