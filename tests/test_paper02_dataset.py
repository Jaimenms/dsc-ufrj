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
    chain1, order1 = ds.make_chains(X, ranges, 1, seed=3)
    assert chain1.max() == 0 and sorted(order1.tolist()) == list(range(40))


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
