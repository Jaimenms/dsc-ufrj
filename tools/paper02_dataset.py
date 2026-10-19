"""Massa de dados do artigo 02: do hipercubo das entradas aos marcadores para o substituto (plano do artigo 02).

Entradas (5): P_res_bar, PI, GOR, P_gas_bar, choke_opening, sobre a geometria com gas lift do artigo 01
(`generators/offshore.py` com `gas_lift=true`). Desenho de Sobol embaralhado, agrupado em cadeias de vizinhos
(k-means e ordem do vizinho mais próximo); cada cadeia corre em sequência num processo: o primeiro ponto da partida
bloqueada, os seguintes com arranque quente a partir do estado final do anterior. Cada corrida usa a paragem adaptativa
com captura de ciclos (061): pára quando P na cabeça do poço e q_L no topo do riser assentam (estacionário ou ciclo
repetido) e continua N_c ciclos, que são a janela de captura dos marcadores.

Dois modos de desenho. `sobol` (omissão): rondas de 25 pontos de Sobol em 5 dimensões, cada cadeia a começar no ponto
mais perto do caso base do artigo 01 (choke aberto, 200 bar, PI 5, RGO 50, retenção fechada), arrancando a quente do
estado final desse caso (`runs/paper01/gaslift/gl_P20.npz`), e a seguir pelo caminho mais curto (exato até 10 pontos)
com as entradas reescalonadas a [−1, +1]. `grid` (`--grid-decisions 6x6`, pedido do dono): cada ronda é um estado do
poço (Sobol em P_res, PI, RGO) com a malha regular das decisões (P_gas × abertura) varrida em serpentina a partir do
canto mais perto do caso base, com P_gas monótono (a linha de gás nunca se esvazia) e a abertura de ida e volta: cada
passo muda uma só decisão de um só nível, e o varrimento dá o mapa completo das decisões desse poço. Ao acabar uma
ronda, `run --rounds` abre a seguinte (os pontos ou o poço seguintes da mesma sequência de Sobol).

Vários `run` na mesma raiz repartem o trabalho sem duplicar (pedido do dono): cada cadeia tem um trinco com o pid do
processo que a corre (`chain<k>.lock`, tomado se o processo morreu), cada instância escolhe a ronda mais antiga com uma
cadeia livre por fazer e, não havendo nenhuma, abre a ronda seguinte (uma de cada vez, com `rounds.lock`); com
`--chains-per-round k` cada ronda nova tem k cadeias para k instâncias.

Tudo é retomável e nada se perde: o desenho é um ficheiro (`design.json`, só cresce: nunca se substitui quando há
corridas), cada corrida é um `.npz` gravado atomicamente (ou um `.failed.json`), a ferramenta nunca apaga corridas, e
uma cadeia retoma do último ponto concluído; um ficheiro `STOP` na pasta pára as cadeias depois da corrida em curso
(paragem quente) e impede rondas novas; matar os processos perde só a corrida em curso; `markers` só lê. Os marcadores (média, mínimo, máximo, f_min, f_max de
P, alpha_G, q_L e q_G em oito pontos característicos, mais o gás injetado e a classe) vão para `dataset.csv` com o
dicionário das colunas em `dataset.json`.

Organização (pedido do dono): `runs/paper02/cases/rounds.json` é a raiz (semente, faixas, configuração, lista de
rondas); cada ronda vive em `runs/paper02/cases/YYYYMMDDXX/` (data de abertura e contador de dois dígitos por dia)
com o seu `design.json` e a sua `dataset.csv`; cada ponto tem a sua pasta `<id>/` (id = 6 caracteres do SHA-1 do
tuplo das entradas, o mesmo tuplo dá sempre o mesmo id) com `point.json` (entradas, metadados e marcadores: o que o
treino usa) e `final_state.npz` (para o arranque quente do ponto seguinte); a trajetória, o caso e o estado inicial
não se guardam (regeneram-se das entradas; `run --keep-runs` guarda-os para depuração, `compact` remove-os).
A raiz recebe a tabela de todas as rondas. Omissão: 25 pontos por ronda, uma cadeia por ronda. `--flat` mantém o modo
de uma só pasta (os pontos continuam em pastas próprias).

    uv run python tools/paper02_dataset.py design --out runs/paper02/cases --seed 1         # raiz + ronda 0 (25 pontos)
    uv run python tools/paper02_dataset.py run --out runs/paper02/cases --rounds -1         # continuamente, até STOP
    uv run python tools/paper02_dataset.py markers --out runs/paper02/cases                # tabelas por ronda e total
    uv run python tools/paper02_dataset.py status --out runs/paper02/cases
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUTS = ("P_res_bar", "PI", "GOR", "P_gas_bar", "choke_opening")
DEFAULT_RANGES = {
    "P_res_bar": (120.0, 220.0),
    "PI": (2.0, 10.0),
    "GOR": (30.0, 200.0),
    "P_gas_bar": (20.0, 160.0),
    "choke_opening": (0.1, 1.0),
}
VARS = ("P", "alphaG", "qL", "qG")
PERIOD_FAST = (
    300.0  # 062: classe "fast" — ciclo da pressão na cabeça com período < 300 s (os ciclos assentados de golfada
)
# e de pressão têm ≥ 546 s nas três campanhas; a intermitência do topo do riser com gás rápido tem 52–108 s)
MARKERS = ("mean", "min", "max", "fmin", "fmax")
AMP_TOL = {"P": 0.1, "alphaG": 0.01, "qL": 0.05, "qG": 0.01}  # amplitude abaixo da qual um sinal é permanente
SMOOTH = 20.0  # s: média móvel antes de procurar os máximos (como no artigo 01)
# pontos característicos: (rótulo, duto, linha das matrizes de saída para P/alphaG, linha para qL/qG)
# linhas: 0 = fantasma de entrada (vazão que entra), 1..N = células, N+1 ou -1 = fantasma de saída (vazão que sai)
POINTS = (
    ("well_bottom", "well_lower", 1, 0),
    ("injection", "well", 1, 0),
    ("wellhead", "well", -2, -1),
    ("riser_base", "1", "low", "low"),
    ("riser_top", "1", -2, -1),
    ("gas_platform", "gas_line", 1, 0),
    ("gas_wellhead", "gas_line", -2, -1),
    ("annulus_base", "annulus", "tank", -1),
)


# ------------------------------------------------------------------------------------------------ desenho
def parse_ranges(items):
    """`nome=lo:hi` → dict; os nomes têm de ser entradas conhecidas."""
    ranges = dict(DEFAULT_RANGES)
    for it in items or ():
        k, v = it.split("=", 1)
        if k not in INPUTS:
            raise SystemExit(f"entrada desconhecida: {k} (conhecidas: {', '.join(INPUTS)})")
        lo, hi = (float(x) for x in v.split(":"))
        ranges[k] = (lo, hi)
    return ranges


def scaled(X, ranges):
    """Entradas reescalonadas dimensão a dimensão para [−1, +1] (pedido do dono): é neste espaço que se medem as
    distâncias do caminho mais curto, o centro da distribuição e os grupos. Uma entrada fixa (faixa nula) vale 0."""
    lo = np.array([ranges[k][0] for k in INPUTS])
    hi = np.array([ranges[k][1] for k in INPUTS])
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.where(hi > lo, 2.0 * (np.asarray(X, float) - lo) / span - 1.0, 0.0)


def sobol_design(n, ranges, seed, skip=0):
    """`n` pontos de Sobol embaralhado nas faixas (matriz n × 5 na ordem de INPUTS), a partir do ponto `skip` da
    sequência: rondas sucessivas da mesma semente continuam a mesma sequência e, juntas, são um só desenho. As
    entradas de faixa nula ficam fixas e não contam para a dimensão da sequência (pedido do dono: só as decisões
    variam, o poço é o do caso base)."""
    import warnings

    from scipy.stats import qmc

    lo = np.array([ranges[k][0] for k in INPUTS])
    hi = np.array([ranges[k][1] for k in INPUTS])
    free = np.flatnonzero(hi > lo)
    gen = qmc.Sobol(d=int(free.size), scramble=True, seed=seed)
    if skip:
        gen.fast_forward(int(skip))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # n fora das potências de 2 é deliberado (rondas)
        u = gen.random(int(n))
    X = np.tile(lo, (int(n), 1))
    X[:, free] = lo[free] + u * (hi[free] - lo[free])
    return X


def path_length(D, path):
    return float(sum(D[path[k], path[k + 1]] for k in range(len(path) - 1)))


def shortest_path_from(U, start, exact_max=10):
    """Caminho aberto mais curto que começa em `start` e visita todos os pontos (caixeiro viajante sem regresso,
    a sequência de arranque quente proposta pelo dono): exato por enumeração até `exact_max` pontos, vizinho mais
    próximo melhorado por 2-opt acima disso. Devolve a lista de índices."""
    from itertools import permutations

    n = U.shape[0]
    D = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=-1)
    others = [i for i in range(n) if i != start]
    if n <= exact_max:
        best, best_len = None, np.inf
        for perm in permutations(others):
            L = D[start, perm[0]] + sum(D[perm[k], perm[k + 1]] for k in range(len(perm) - 1)) if perm else 0.0
            if L < best_len:
                best, best_len = perm, L
        return [start] + list(best or ())
    cur, left, path = start, set(others), [start]
    while left:
        nxt = min(left, key=lambda j: D[cur, j])
        left.remove(nxt)
        path.append(nxt)
        cur = nxt
    improved = True
    while improved:  # 2-opt num caminho aberto (o início fica fixo)
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cand = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]
                if path_length(D, cand) < path_length(D, path) - 1e-12:
                    path, improved = cand, True
    return path


def make_chains(X, ranges, n_chains, seed, start="base", x_base=None):
    """Agrupa os pontos em `n_chains` cadeias de vizinhos (k-means nas entradas reescalonadas a [−1, 1]) e, em cada
    cadeia, escolhe o primeiro ponto — `"base"` (pedido do dono, 2026-10-18): o mais perto do caso base do artigo
    01, `x_base`, de onde a cadeia arranca a quente; `"random"`: sorteado; `"centre"`: o mais perto do centro do
    grupo — e segue o caminho mais curto por todos os outros (`shortest_path_from`). Devolve (cadeia, ordem) por
    ponto."""
    from scipy.cluster.vq import kmeans2

    U = scaled(X, ranges)
    n_chains = max(1, min(int(n_chains), U.shape[0]))
    rng = np.random.default_rng(seed)
    if n_chains == 1:
        labels = np.zeros(U.shape[0], int)
    else:
        _, labels = kmeans2(U, n_chains, minit="++", seed=rng)
    chain = np.full(U.shape[0], -1, int)
    order = np.full(U.shape[0], -1, int)
    c_id = 0
    for lab in range(n_chains):
        idx = np.flatnonzero(labels == lab)
        if idx.size == 0:
            continue
        if start == "base" and x_base is not None:
            ub = scaled(np.asarray([x_base[k] for k in INPUTS], float)[None, :], ranges)[0]
            first = int(np.argmin(np.linalg.norm(U[idx] - ub, axis=1)))
        elif start == "centre":
            centre = U[idx].mean(axis=0)
            first = int(np.argmin(np.linalg.norm(U[idx] - centre, axis=1)))
        else:  # sorteado antes de se determinar o caminho
            first = int(rng.integers(idx.size))
        tour = [int(idx[i]) for i in shortest_path_from(U[idx], first)]
        for k, i in enumerate(tour):
            chain[i], order[i] = c_id, k
        c_id += 1
    return chain, order


def _settle_spec(a):
    return dict(
        window=float(a.window),
        t_min=float(a.t_min),
        t_min_warm=float(a.t_min_warm),
        amp_tol=AMP_TOL["P"],
        rel_tol=float(a.rel_tol),
        match="envelope",
        capture_cycles=int(a.capture),
        capture_window=float(a.capture_window),
        window_periods=int(a.window_periods),
        window_min=float(a.window_min),
        signal={"edge": "1", "node": 1, "var": "P", "amp_tol": AMP_TOL["P"]},
        signals=[{"edge": "1", "node": -1, "var": "qL", "amp_tol": AMP_TOL["qL"]}],
    )


def _steady_spec(a):
    """Partida estacionária (`--start steady`): estimativa do gerador de onde parte o Newton, perturbação das
    pressões do estacionário e iterações máximas."""
    return dict(guess=str(a.guess), perturb=float(a.perturb), max_iter=int(a.newton_iter))


def point_id(x, n=6):
    """Identificador do ponto (pedido do dono): os primeiros `n` caracteres do SHA-1 do tuplo das entradas, na
    ordem de INPUTS e com a representação completa dos floats — o mesmo tuplo dá sempre o mesmo id."""
    key = json.dumps([float(x[k]) for k in INPUTS], separators=(",", ":"))
    return hashlib.sha1(key.encode()).hexdigest()[:n]


def unique_ids(xs, taken=()):
    """ids de 6 caracteres; em colisão (dentro do conjunto ou com `taken`) o ponto leva mais caracteres."""
    ids, seen = [], set(taken)
    for x in xs:
        n = 6
        pid = point_id(x, n)
        while pid in seen:
            n += 2
            pid = point_id(x, n)
        seen.add(pid)
        ids.append(pid)
    return ids


def design_path(out):
    return Path(out) / "design.json"


def load_design(out):
    return json.load(open(design_path(out), encoding="utf-8"))


def save_design(out, d):
    p = design_path(out)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def append_round(design, n=None, chains=1):
    """Acrescenta uma ronda ao desenho (ideia do dono, 2026-10-18: ao acabar os pontos, sortear outros tantos e
    recomeçar do centro da nova ronda com a partida bloqueada): `n` pontos seguintes da mesma sequência de Sobol,
    em `chains` cadeias novas (ordem do caminho mais curto a partir do centro). Devolve os pontos novos."""
    ranges = {k: tuple(v) for k, v in design["ranges"].items()}
    n = int(n or design.get("round_size") or design["n"])
    X = sobol_design(n, ranges, int(design["seed"]), skip=int(design["n"]))
    if design.get("test"):
        chain, order = np.arange(n), np.zeros(n, int)
    else:
        chain, order = make_chains(
            X,
            ranges,
            chains,
            int(design["seed"]) + int(design["n"]),
            start=design.get("start", "base"),
            x_base=design.get("x_base"),
        )
    c0 = 1 + max((p["chain"] for p in design["points"]), default=-1)
    rnd = 1 + max((p.get("round", 0) for p in design["points"]), default=-1)
    i0 = int(design["n"])
    xs = [{k: float(X[i, j]) for j, k in enumerate(INPUTS)} for i in range(n)]
    ids = unique_ids(xs, taken=[p["id"] for p in design["points"]])
    new = [
        dict(id=ids[i], x=xs[i], chain=int(c0 + chain[i]), order=int(order[i]), round=int(rnd), sobol_index=i0 + i)
        for i in range(n)
    ]
    design["points"].extend(new)
    design["n"] = i0 + n
    design["rounds"] = rnd + 1
    return new


WELL_INPUTS = ("P_res_bar", "PI", "GOR")  # estado do poço (sorteado); as decisões (P_gas, abertura) vão em malha
DECISIONS = ("P_gas_bar", "choke_opening")


def sobol_wells(n, ranges, seed, skip=0):
    """`n` estados do poço (P_res, PI, RGO) de Sobol embaralhado em 3 dimensões, a partir do ponto `skip`."""
    import warnings

    from scipy.stats import qmc

    gen = qmc.Sobol(d=len(WELL_INPUTS), scramble=True, seed=seed)
    if skip:
        gen.fast_forward(int(skip))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        u = gen.random(int(n))
    lo = np.array([ranges[k][0] for k in WELL_INPUTS])
    hi = np.array([ranges[k][1] for k in WELL_INPUTS])
    return lo + u * (hi - lo)


def grid_decisions(well, ranges, n_gas, n_open, x_base, inner="choke_opening"):
    """Malha regular n_gas × n_open nas decisões para um estado do poço, na **melhor sequência de arranque quente**
    (pedido do dono): varrimento em serpentina (boustrophedon) a partir do canto da malha mais perto do caso base,
    com o eixo exterior (por omissão P_gas) monótono — a pressão de gás nunca desce, por isso a linha de gás e o
    anular nunca se esvaziam para a plataforma — e o eixo interior (a abertura) de ida e volta. Cada passo muda uma
    só decisão, de um só nível; o caminho visita cada nó uma vez e é o mais curto possível numa malha. Devolve a
    lista de entradas x na ordem."""
    outer = "P_gas_bar" if inner == "choke_opening" else "choke_opening"
    levels = {
        "P_gas_bar": np.linspace(ranges["P_gas_bar"][0], ranges["P_gas_bar"][1], n_gas),
        "choke_opening": np.linspace(ranges["choke_opening"][0], ranges["choke_opening"][1], n_open),
    }
    # canto mais perto do caso base, eixo a eixo (no espaço reescalonado a [−1, 1] a distância por eixo é a mesma)
    for k in (outer, inner):
        if x_base is not None and abs(levels[k][-1] - x_base[k]) < abs(levels[k][0] - x_base[k]):
            levels[k] = levels[k][::-1]
    seq = []
    for i, v_out in enumerate(levels[outer]):
        inner_levels = levels[inner] if i % 2 == 0 else levels[inner][::-1]
        for v in inner_levels:
            x = dict(well)
            x[outer] = float(v_out)
            x[inner] = float(v)
            seq.append(x)
    return seq


ROUNDS_FILE = "rounds.json"  # raiz das rondas (pedido do dono): uma pasta datada por ronda


def rounds_path(root):
    return Path(root) / ROUNDS_FILE


def is_rounds_root(out):
    return rounds_path(out).exists()


def load_rounds(root):
    return json.load(open(rounds_path(root), encoding="utf-8"))


def save_rounds(root, master):
    p = rounds_path(root)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(master, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def new_round_dir(root):
    """Pasta da ronda (pedido do dono): YYYYMMDDXX, a data de abertura e um contador de dois dígitos por dia
    (01, 02, …), não a hora."""
    day = time.strftime("%Y%m%d")
    used = [int(q.name[8:10]) for q in Path(root).glob(f"{day}[0-9][0-9]") if q.is_dir()]
    return Path(root) / f"{day}{(max(used) + 1 if used else 1):02d}"


def create_round(root, master, n=None, chains=1):
    """Ronda nova numa pasta datada, com um `design.json` próprio; atualiza `rounds.json`. Modo `sobol`: os próximos
    `n` pontos da sequência de Sobol do conjunto, no caminho mais curto a partir do ponto mais perto do caso base.
    Modo `grid` (pedido do dono): o próximo estado do poço da sequência de Sobol em 3 dimensões e a malha
    n_gas × n_open das decisões em serpentina (`grid_decisions`), uma cadeia. Devolve a pasta."""
    ranges = {k: tuple(v) for k, v in master["ranges"].items()}
    rnd = len(master["rounds"])
    if master.get("mode") == "grid":
        g = master["grid"]
        skip = int(master["n"])  # estados do poço já consumidos
        W = sobol_wells(1, ranges, int(master["seed"]), skip=skip)[0]
        well = {k: float(W[j]) for j, k in enumerate(WELL_INPUTS)}
        xs = grid_decisions(
            well, ranges, int(g["n_gas"]), int(g["n_open"]), master.get("x_base"), inner=g.get("inner", "choke_opening")
        )
        n = len(xs)
        ids = unique_ids(xs, taken=master.get("ids", []))
        points = [
            dict(id=ids[i], x=xs[i], chain=0, order=i, round=rnd, well_index=skip, grid_index=i) for i in range(n)
        ]
        master["ids"] = list(master.get("ids", [])) + ids
        design = {k: v for k, v in master.items() if k not in ("rounds", "ids")}
        design.update(n=n, round_size=n, rounds=1, round=rnd, well=well, well_index=skip, points=points)
        d = new_round_dir(root)
        d.mkdir(parents=True, exist_ok=False)
        save_design(d, design)
        master["rounds"].append(
            dict(folder=d.name, n=n, well_index=skip, well=well, created=time.strftime("%Y-%m-%d %H:%M"))
        )
        master["n"] = skip + 1
        save_rounds(root, master)
        return d
    n = int(n or master["round_size"])
    skip = int(master["n"])
    X = sobol_design(n, ranges, int(master["seed"]), skip=skip)
    chain, order = make_chains(
        X, ranges, chains, int(master["seed"]) + skip, start=master.get("start", "base"), x_base=master.get("x_base")
    )
    xs = [{k: float(X[i, j]) for j, k in enumerate(INPUTS)} for i in range(n)]
    ids = unique_ids(xs, taken=master.get("ids", []))
    points = [
        dict(id=ids[i], x=xs[i], chain=int(chain[i]), order=int(order[i]), round=rnd, sobol_index=skip + i)
        for i in range(n)
    ]
    master["ids"] = list(master.get("ids", [])) + ids
    design = {k: v for k, v in master.items() if k not in ("rounds", "ids")}
    design.update(n=n, round_size=n, rounds=1, round=rnd, sobol_skip=skip, points=points)
    d = new_round_dir(root)
    d.mkdir(parents=True, exist_ok=False)
    save_design(d, design)
    master["rounds"].append(dict(folder=d.name, n=n, skip=skip, created=time.strftime("%Y-%m-%d %H:%M")))
    master["n"] = skip + n
    save_rounds(root, master)
    return d


def base_point(base_run):
    """Caso base do artigo 01 de onde as cadeias arrancam: caminho da corrida e as suas cinco entradas (lidas dos
    parâmetros do gerador gravados na corrida). Sem corrida (`""`), None."""
    if not base_run:
        return dict(run=None, x=None)
    from mfsim.runs import load_run

    path = Path(base_run)
    if not path.is_absolute():
        path = ROOT / path
    run = load_run(str(path))
    gp = run.case_json["meta"]["generator_params"]
    return dict(
        run=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), x={k: float(gp[k]) for k in INPUTS}
    )


def resequence_round(d, master, chains=1):
    """Reordena os pontos de uma ronda sem corridas feitas com a regra atual do primeiro ponto (`start`, `x_base`)."""
    design = load_design(d)
    if any(point_status(d, p["id"]) != "pending" for p in design["points"]):
        raise SystemExit(f"{d}: já tem corridas; uma ronda com corridas não se reordena")
    ranges = {k: tuple(v) for k, v in master["ranges"].items()}
    pts = sorted(design["points"], key=lambda p: p.get("sobol_index", 0))
    X = np.array([[p["x"][k] for k in INPUTS] for p in pts], float)
    chain, order = make_chains(
        X,
        ranges,
        chains,
        int(master["seed"]) + int(design.get("sobol_skip", 0)),
        start=master.get("start", "base"),
        x_base=master.get("x_base"),
    )
    for i, p in enumerate(pts):
        p["chain"], p["order"] = int(chain[i]), int(order[i])
    design.update(points=pts, start=master.get("start"), base_run=master.get("base_run"), x_base=master.get("x_base"))
    save_design(d, design)


def cmd_design(a):
    out = Path(a.out)
    base = base_point(a.base_run if (a.start == "base" or a.fix_well) else "")
    if a.start == "base" and base["x"] is None:
        raise SystemExit("--start base precisa de --base-run (corrida do caso base do artigo 01)")
    if a.fix_well:
        a.range = list(a.range) + [f"{k}={base['x'][k]}:{base['x'][k]}" for k in WELL_INPUTS]
    if not a.flat:
        # omissão: raiz de rondas, cada ronda na sua pasta datada
        if is_rounds_root(out):
            master = load_rounds(out)
            if a.resequence:
                # reordenar a última ronda com a regra atual (só sem corridas feitas: nada se perde)
                master.update(start=a.start, base_run=base["run"], x_base=base["x"])
                d = out / master["rounds"][-1]["folder"]
                resequence_round(d, master, chains=a.chains)
                save_rounds(out, master)
                pts = load_design(d)["points"]
                tour = [p["id"] for p in sorted(pts, key=lambda p: p["order"])]
                print(f"ronda {master['rounds'][-1]['folder']} reordenada ({a.start}): {' → '.join(tour)}")
                return
            if not a.append:
                raise SystemExit(f"{rounds_path(out)} já existe (--append para abrir uma ronda nova)")
            d = create_round(out, master, n=a.n, chains=a.chains)
            print(f"ronda {len(master['rounds']) - 1}: {d} ({master['rounds'][-1]['n']} pontos)")
            return
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} existe e não está vazia: um conjunto nunca se substitui")
        ranges = parse_ranges(a.range)
        grid = None
        if a.grid_decisions:
            ng, no = (int(v) for v in a.grid_decisions.lower().split("x"))
            grid = dict(n_gas=ng, n_open=no, inner=a.inner)
        master = dict(
            schema="paper02-rounds/1",
            created=time.strftime("%Y-%m-%d %H:%M"),
            seed=int(a.seed),
            n=0,
            mode="grid" if grid else "sobol",
            grid=grid,
            round_size=int(a.n or (grid["n_gas"] * grid["n_open"] if grid else 25)),
            start=a.start,
            steady=_steady_spec(a),
            base_run=base["run"],
            x_base=base["x"],
            test=bool(a.test),
            ranges={k: list(v) for k, v in ranges.items()},
            inputs=list(INPUTS),
            tf_max=float(a.tf_max),
            dt_out=float(a.dt_out),
            max_cpu=float(a.max_cpu),
            settle=_settle_spec(a),
            rounds=[],
        )
        out.mkdir(parents=True, exist_ok=True)
        save_rounds(out, master)
        d = create_round(out, master, chains=a.chains)
        dsg = load_design(d)
        pts = dsg["points"]
        tour = [p["id"] for p in sorted(pts, key=lambda p: p["order"])]
        print(f"{rounds_path(out)}: ronda 0 em {d} com {len(pts)} pontos (semente {a.seed}, modo {master['mode']})")
        if dsg.get("well"):
            print(f"  estado do poço: {{{', '.join(f'{k}: {v:.3g}' for k, v in dsg['well'].items())}}}")
        print(f"  sequência: {' → '.join(tour)}")
        return
    if a.append:
        design = load_design(out)
        new = append_round(design, n=a.n, chains=a.chains)
        save_design(out, design)
        print(
            f"{design_path(out)}: ronda {design['rounds'] - 1} com {len(new)} pontos ({new[0]['id']}–{new[-1]['id']})"
        )
        return
    if design_path(out).exists():
        n_runs = len(list(out.glob("*/run.npz"))) + len(list(out.glob("*/point.json")))
        if n_runs or not a.force:
            raise SystemExit(
                f"{design_path(out)} já existe com {n_runs} corridas: um desenho nunca se substitui "
                "(use --append para rondas novas, ou outra pasta)"
            )
    ranges = parse_ranges(a.range)
    X = sobol_design(a.n or 25, ranges, a.seed)
    if a.test:
        chain, order = np.arange(X.shape[0]), np.zeros(X.shape[0], int)  # teste: todos da partida bloqueada
    else:
        chain, order = make_chains(X, ranges, a.chains, a.seed, start=a.start, x_base=base["x"])
    xs = [{k: float(X[i, j]) for j, k in enumerate(INPUTS)} for i in range(X.shape[0])]
    ids = unique_ids(xs)
    points = [
        dict(id=ids[i], x=xs[i], chain=int(chain[i]), order=int(order[i]), round=0, sobol_index=i)
        for i in range(X.shape[0])
    ]
    settle = _settle_spec(a)
    d = dict(
        schema="paper02-dataset/1",
        created=time.strftime("%Y-%m-%d %H:%M"),
        seed=int(a.seed),
        n=int(X.shape[0]),
        round_size=int(X.shape[0]),
        rounds=1,
        start=a.start,
        steady=_steady_spec(a),
        base_run=base["run"],
        x_base=base["x"],
        test=bool(a.test),
        ranges={k: list(v) for k, v in ranges.items()},
        inputs=list(INPUTS),
        tf_max=float(a.tf_max),
        dt_out=float(a.dt_out),
        max_cpu=float(a.max_cpu),
        settle=settle,
        points=points,
    )
    out.mkdir(parents=True, exist_ok=True)
    save_design(out, d)
    n_chains = len(set(chain.tolist()))
    print(f"{design_path(out)}: {X.shape[0]} pontos em {n_chains} cadeias (semente {a.seed})")
    if not a.test:
        U = scaled(X, ranges)
        for c in range(n_chains):
            idx = np.flatnonzero(chain == c)
            tour = idx[np.argsort(order[idx])]
            L = sum(float(np.linalg.norm(U[tour[k + 1]] - U[tour[k]])) for k in range(tour.size - 1))
            print(f"  cadeia {c}: {' → '.join(points[i]['id'] for i in tour)} (distância em [−1, 1]: {L:.2f})")


# ------------------------------------------------------------------------------------------------ corridas
def point_dir(out, pid):
    """Pasta do ponto: `<ronda>/<id>/` com point.json (entradas, metadados e marcadores: o que o treino usa),
    final_state.npz (o estado final, só para o arranque quente do ponto seguinte) e, a pedido (`--keep-runs`),
    run.npz com a trajetória; failed.json quando a corrida rebenta. Pedido do dono: não guardar o que o treino
    não usa (nem a trajetória, nem o caso, nem o estado inicial — regeneram-se das entradas)."""
    return Path(out) / pid


def run_path(out, pid):
    return point_dir(out, pid) / "run.npz"


def point_path(out, pid):
    return point_dir(out, pid) / "point.json"


def state_path(out, pid):
    return point_dir(out, pid) / "final_state.npz"


def fail_path(out, pid):
    return point_dir(out, pid) / "failed.json"


def point_status(out, pid):
    if point_path(out, pid).exists() or run_path(out, pid).exists():
        return "done"
    if fail_path(out, pid).exists():
        return "failed"
    return "pending"


def _settle_for(design, start):
    s = dict(design["settle"])
    t_min_warm = s.pop("t_min_warm", None)
    if start in ("warm", "base", "resume") and t_min_warm is not None:
        s["t_min"] = float(t_min_warm)
    return s


def _node_index(case, edge, node):
    """`node` negativo conta do fim (−1 = última célula) no esquema do duto `edge`."""
    if node >= 0:
        return node
    e = next(x for x in case["network"]["edges"] if x["id"] == edge)
    n_cells = sum(int(sg["n"]) for sg in e["geometry"]["segments"])
    return n_cells + 1 + node


def build_network(design, point, start, X_prev=None, prev_id=None, keep_case=None):
    """Caso de um ponto em memória (gerador do offshore com gas lift, paragem adaptativa com captura, orçamento),
    com a partida bloqueada ou o arranque quente a partir de `X_prev`; devolve o `Network` pronto a simular. Com
    `keep_case` (caminho) grava também o JSON e o estado inicial, para depuração."""
    from generators.offshore import initial_state_guess, offshore_case
    from mfsim.io import case_from_json, case_to_json
    from mfsim.simulation import build_model

    x = point["x"]
    case = offshore_case(
        gas_lift=True,
        P_res_bar=x["P_res_bar"],
        PI=x["PI"],
        GOR=x["GOR"],
        P_gas_bar=x["P_gas_bar"],
        choke_opening=x["choke_opening"],
        tf=float(design["tf_max"]),
        dt_out=float(design["dt_out"]),
    )
    settle = _settle_for(design, start)
    for sg in [settle["signal"]] + settle["signals"]:
        sg["node"] = _node_index(case, sg["edge"], int(sg["node"]))
    case["time"]["settle"] = settle
    case["integration"]["max_cpu"] = float(design["max_cpu"])
    case["name"] = f"paper02_{point['id']}"
    net = case_from_json(case)
    model = build_model(net)
    if start in ("blocked", "steady"):
        sd = dict(design.get("steady") or {})
        mode = str(sd.get("guess", "noslip")) if start == "steady" else "blocked"
        X0, m = initial_state_guess(model, net, case["meta"]["mass"]["w_gas"], mode=mode)
        net.meta = dict(net.meta or {}, initial_guess=dict(mode=mode, m_total=float(m)))
    else:
        if np.asarray(X_prev).shape[0] != model.n_states:
            raise ValueError(
                f"estado de arranque com {np.asarray(X_prev).shape[0]} estados; o caso tem {model.n_states}"
            )
        X0 = np.asarray(X_prev, float)
        net.meta = dict(net.meta or {}, initial_guess=dict(mode=start, from_point=prev_id))
    model.state_to_case_ic(X0, 0.0, exact=True)
    net.initial = "edges"
    if start == "steady":
        # partida estacionária (pedido do dono, 2026-10-19): Newton (issue 040) a partir da estimativa do gerador
        # gravada nos nós, sem caso base nem ponto anterior; se não converge, segue da estimativa (partida fria);
        # a perturbação das pressões tira a trajetória de um equilíbrio instável
        sd = dict(design.get("steady") or {})
        net.initial = "steady"
        net.initial_options = {
            "from": "edges",
            "on_failure": "start",
            "perturb": float(sd.get("perturb", 0.01)),
            "max_iter": int(sd.get("max_iter", 100)),
        }
    if keep_case:
        Path(keep_case).parent.mkdir(parents=True, exist_ok=True)
        case_to_json(net, str(keep_case), ic_file="case.ic.npz")
    return net


def _final_state(out, pid):
    """Estado final de um ponto feito: `final_state.npz`, ou a última coluna de `run.npz` (pontos antigos)."""
    if state_path(out, pid).exists():
        return np.load(state_path(out, pid))["X"]
    from mfsim.runs import load_run

    return np.asarray(load_run(str(run_path(out, pid))).X)[:, -1]


def point_record(design, point, start, prev_id, run, tail=4000.0, initial=None):
    """O que fica de um ponto: entradas, cadeia, metadados da corrida, o Newton da partida estacionária (`initial`,
    quando há) e marcadores (`point.json`)."""
    return dict(
        id=point["id"],
        x=point["x"],
        chain=point["chain"],
        order=point["order"],
        round=point.get("round", 0),
        sobol_index=point.get("sobol_index"),
        start=start,
        from_point=prev_id,
        initial=initial,
        markers=run_markers(run, tail=tail),
        written=time.strftime("%Y-%m-%d %H:%M"),
    )


def save_point(out, rec, X_final):
    pdir = point_dir(out, rec["id"])
    pdir.mkdir(parents=True, exist_ok=True)
    tmp = pdir / "final_state.partial.npz"
    np.savez(tmp, X=np.asarray(X_final, float))
    os.replace(tmp, state_path(out, rec["id"]))
    tmp = pdir / "point.partial.json"
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, point_path(out, rec["id"]))


def _newton_note(info):
    """Sufixo do registo: iterações do Newton da partida estacionária e se convergiu."""
    ini = info.get("initial") or {}
    if ini.get("source") != "steady":
        return ""
    tail = "" if ini.get("converged") else " (não convergiu: seguiu da estimativa)"
    return f", Newton {ini.get('iterations')} it{tail}"


def run_point(design, point, out, start, X_prev=None, prev_id=None, progress=True):
    """Corre um ponto e grava `point.json` + `final_state.npz` (e `run.npz` com `keep_runs`), ou `failed.json`;
    devolve o estado."""
    from mfsim.runs import load_run, save_run
    from mfsim.simulation import simulate

    pid = point["id"]
    t0 = time.time()
    keep_runs = bool(design.get("keep_runs"))
    try:
        pdir = point_dir(out, pid)
        pdir.mkdir(parents=True, exist_ok=True)
        net = build_network(
            design, point, start, X_prev=X_prev, prev_id=prev_id, keep_case=(pdir / "case.json") if keep_runs else None
        )
        res = simulate(net, progress=False, partial=True)
        info = dict(res.info)
        info["cpu_total"] = time.time() - t0
        info["note"] = json.dumps(dict(start=start, from_point=prev_id, chain=point["chain"], order=point["order"]))
        tmp = pdir / "run.partial.npz"  # gravação atómica: uma morte a meio nunca deixa um .npz truncado
        save_run(
            str(tmp),
            res.T,
            res.X,
            res.model.network,
            info=info,
            acc=info.get("accum"),
            command=f"tools/paper02_dataset.py run --out {out} (ponto {pid})",
        )
        rec = point_record(
            design,
            point,
            start,
            prev_id,
            load_run(str(tmp)),
            tail=float(design.get("tail", 4000.0)),
            initial=info.get("initial"),
        )
        save_point(out, rec, np.asarray(res.X)[:, -1])
        if keep_runs:
            os.replace(tmp, run_path(out, pid))
        else:
            tmp.unlink()  # a trajetória não se guarda (pedido do dono): fica point.json + final_state.npz
        if fail_path(out, pid).exists():
            fail_path(out, pid).unlink()
        if progress:
            st = info.get("settle") or {}
            print(
                f"{pid} ({start}): OK {st.get('reason', '?')} em t={st.get('t_end', float('nan')):.0f} s, "
                f"{info.get('n_steps')} passos, {info['cpu_total']:.0f} s" + _newton_note(info),
                flush=True,
            )
        return "done"
    except Exception as e:  # noqa: BLE001  (um ponto que rebenta não trava a cadeia)
        point_dir(out, pid).mkdir(parents=True, exist_ok=True)
        fail_path(out, pid).write_text(
            json.dumps(dict(error=f"{type(e).__name__}: {e}", start=start, from_point=prev_id, cpu=time.time() - t0)),
            encoding="utf-8",
        )
        if progress:
            print(f"{pid} ({start}): FALHOU {type(e).__name__}: {str(e)[:120]}", flush=True)
        return "failed"


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def acquire_lock(path):
    """Trinco por ficheiro com o pid do dono (criação atómica): True se ficou nosso; um trinco de um processo morto
    é tomado. É o que permite vários `run` na mesma raiz sem duplicar pontos (pedido do dono)."""
    path = Path(path)
    for _ in range(2):
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            return True
        except FileExistsError:
            try:
                pid = int(path.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid) and pid != os.getpid():
                return False
            try:
                path.unlink()  # dono morto: toma-se o trinco
            except FileNotFoundError:
                pass
    return False


def release_lock(path):
    try:
        if int(Path(path).read_text(encoding="utf-8").strip() or 0) == os.getpid():
            Path(path).unlink()
    except (OSError, ValueError):
        pass


def chain_lock(out, chain_id):
    return Path(out) / f"chain{int(chain_id)}.lock"


def chain_is_free(out, chain_id):
    lk = chain_lock(out, chain_id)
    if not lk.exists():
        return True
    try:
        pid = int(lk.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return True
    return not (pid and _pid_alive(pid))


def run_chain(args):
    """Uma cadeia em sequência: retoma do último ponto concluído; para com o ficheiro STOP. Só corre se conseguir o
    trinco da cadeia (outro processo vivo com a mesma cadeia → devolve sem fazer nada)."""
    out, chain_id, progress, only, resume = (tuple(args) + (None, False))[:5]
    if not acquire_lock(chain_lock(out, chain_id)):
        return chain_id, []
    try:
        return _run_chain_locked(out, chain_id, progress, only=only, resume=resume)
    finally:
        release_lock(chain_lock(out, chain_id))


def _base_state(design, chain_id, progress):
    """Estado final do caso base do artigo 01 (`base_run`), ou None se não existe ou não se lê."""
    if not design.get("base_run") or design.get("test"):
        return None
    try:
        from mfsim.runs import load_run

        return np.asarray(load_run(str(ROOT / design["base_run"])).X)[:, -1]
    except Exception as e:  # noqa: BLE001
        if progress:
            print(f"cadeia {chain_id}: caso base ilegível ({e}); partida bloqueada", flush=True)
        return None


def is_exhausted(out, pid):
    """Ponto feito cuja corrida esgotou o orçamento (marcadores da trajetória parcial, `reason = failure`)."""
    try:
        rec = json.loads(point_path(out, pid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (rec.get("markers") or {}).get("reason") == "failure"


def _run_chain_locked(out, chain_id, progress, only=None, resume=False):
    design = load_design(out)
    pts = sorted((p for p in design["points"] if p["chain"] == chain_id), key=lambda p: p["order"])
    done = []
    prev_done = None
    # `from_base` (pedido do dono, 2026-10-19): todos os pontos arrancam do estado final do caso base, nenhum do
    # anterior — a mesma condição inicial para todos; acima de 0,5 de abertura o arranque quente de um vizinho
    # esgotava o orçamento e o do caso base passava em < 70 s
    X_base_all = _base_state(design, chain_id, progress) if design.get("from_base") else None
    for p in pts:
        st = point_status(out, p["id"])
        if only is not None and p["id"] not in only:
            if st == "done":
                prev_done = p["id"]
            continue  # `run --only`: só os pontos pedidos; os outros pendentes ficam para depois
        if st == "done" and resume and is_exhausted(out, p["id"]):
            # `run --resume-exhausted` (pedido do dono, 2026-10-19): retomar do próprio estado final, com o orçamento
            # inteiro outra vez; os marcadores passam a ser os da continuação
            if (Path(out) / "STOP").exists() or (Path(out).parent / "STOP").exists():
                break
            st = run_point(
                design, p, out, "resume", X_prev=_final_state(out, p["id"]), prev_id=p["id"], progress=progress
            )
            done.append((p["id"], st))
            prev_done = p["id"]
            continue
        if st == "done":
            prev_done = p["id"]
            continue
        if st == "failed" and not design.get("retry_failed", False):
            continue
        if (Path(out) / "STOP").exists() or (Path(out).parent / "STOP").exists():
            if progress:
                print(f"cadeia {chain_id}: STOP encontrado, parada antes de {p['id']}", flush=True)
            break
        if design.get("start") == "steady":
            st = run_point(design, p, out, "steady", progress=progress)
        elif X_base_all is not None:
            st = run_point(design, p, out, "base", X_prev=X_base_all, prev_id="base", progress=progress)
        elif prev_done is None and design.get("start", "base") == "base" and design.get("base_run"):
            X_base = _base_state(design, chain_id, progress)
            if X_base is not None:
                st = run_point(design, p, out, "base", X_prev=X_base, prev_id="base", progress=progress)
            else:
                st = run_point(design, p, out, "blocked", progress=progress)
        elif prev_done is None or design.get("test"):
            st = run_point(design, p, out, "blocked", progress=progress)
        else:
            st = run_point(
                design, p, out, "warm", X_prev=_final_state(out, prev_done), prev_id=prev_done, progress=progress
            )
        if st == "done":
            prev_done = p["id"]
        done.append((p["id"], st))
    return chain_id, done


def _only_set(a):
    """`run --only`: conjunto de ids (separados por vírgulas, ou um ficheiro com um id por linha); None = todos."""
    v = getattr(a, "only", None)
    if not v:
        return None
    if Path(v).exists():
        return {x.strip() for x in Path(v).read_text(encoding="utf-8").split() if x.strip()}
    return {x.strip() for x in str(v).split(",") if x.strip()}


def _run_options_changed(a, design):
    """Opções de `run` que ficam gravadas no desenho (uma vez ligadas, valem para a ronda toda)."""
    return (
        (a.retry_failed and not design.get("retry_failed"))
        or (a.from_base and not design.get("from_base"))
        or bool(a.keep_runs) != bool(design.get("keep_runs"))
    )


def _apply_run_options(a, design):
    design["retry_failed"] = bool(design.get("retry_failed") or a.retry_failed)
    design["from_base"] = bool(design.get("from_base") or a.from_base)
    design["keep_runs"] = bool(a.keep_runs)


def free_pending_chains(out, design, resume=False):
    """Cadeias com pontos por fazer (com `resume`, também os esgotados a retomar) e sem trinco de um processo vivo."""
    chains = sorted(
        {
            p["chain"]
            for p in design["points"]
            if point_status(out, p["id"]) == "pending" or (resume and is_exhausted(out, p["id"]))
        }
    )
    return [c for c in chains if chain_is_free(out, c)]


def _run_pending(out, design, jobs, only=None, resume=False):
    """Corre as cadeias livres com pontos por fazer (com `only`, só os pontos desse conjunto; com `resume`, também os
    esgotados, retomados do próprio estado final); devolve (feitas, falhadas)."""
    chains = free_pending_chains(out, design, resume=resume)
    if only is not None:
        ids = {
            p["id"]
            for p in design["points"]
            if p["id"] in only and (point_status(out, p["id"]) == "pending" or (resume and is_exhausted(out, p["id"])))
        }
        chains = [c for c in chains if any(p["chain"] == c and p["id"] in ids for p in design["points"])]
    if not chains:
        return 0, 0
    work = [(str(out), c, True, only, resume) for c in chains]
    if jobs > 1 and len(chains) > 1:
        from concurrent.futures import ProcessPoolExecutor

        from mfsim.runs import _worker_init

        with ProcessPoolExecutor(max_workers=int(jobs), initializer=_worker_init) as ex:
            results = list(ex.map(run_chain, work))
    else:
        results = [run_chain(w) for w in work]
    n_ok = sum(1 for _, d in results for _, st in d if st == "done")
    n_bad = sum(1 for _, d in results for _, st in d if st == "failed")
    return n_ok, n_bad


def cmd_run_rounds(a):
    """Raiz de rondas: corre a ronda mais recente; ao acabar, abre uma ronda nova (pasta datada) até STOP."""
    root = Path(a.out)
    stop = root / "STOP"
    if stop.exists() and not a.keep_stop:
        stop.unlink()
    tot_ok = tot_bad = 0
    while True:
        # a ronda mais antiga com uma cadeia livre por fazer (vários `run` na mesma raiz repartem as cadeias)
        master = load_rounds(root)
        claimed = False
        for r in master["rounds"]:
            cur = root / r["folder"]
            design = load_design(cur)
            if _run_options_changed(a, design):
                _apply_run_options(a, design)
                save_design(cur, design)
            chains = free_pending_chains(cur, design, resume=bool(a.resume_exhausted))
            if not chains:
                continue
            print(f"ronda {design.get('round', 0)} ({cur.name}): cadeias livres {chains}", flush=True)
            n_ok, n_bad = _run_pending(cur, design, a.jobs, only=_only_set(a), resume=bool(a.resume_exhausted))
            tot_ok, tot_bad = tot_ok + n_ok, tot_bad + n_bad
            claimed = True
            if n_ok == 0 and n_bad > 0:
                # guarda contra a fuga: uma cadeia inteira falhada sem uma corrida boa é um erro de código ou de
                # ambiente, e abrir rondas novas só multiplicaria falhas
                print(
                    f"ronda {cur.name}: {n_bad} falhas e nenhuma corrida feita — a parar (ver failed.json)", flush=True
                )
                stop.touch()
            break
        if stop.exists():
            print("STOP encontrado: sem rondas novas", flush=True)
            break
        if claimed:
            continue
        if a.rounds == 0 or (a.rounds > 0 and len(master["rounds"]) >= a.rounds):
            busy = any(
                not chain_is_free(root / r["folder"], c)
                for r in master["rounds"]
                for c in {p["chain"] for p in load_design(root / r["folder"])["points"]}
            )
            if not busy:
                break
            time.sleep(60.0)  # outras instâncias ainda a correr: esperar sem abrir rondas
            continue
        if acquire_lock(root / "rounds.lock"):
            try:
                master = load_rounds(root)
                if not any(
                    free_pending_chains(root / r["folder"], load_design(root / r["folder"])) for r in master["rounds"]
                ):
                    d = create_round(root, master, chains=a.chains_per_round)
                    print(f"ronda nova em {d}", flush=True)
            finally:
                release_lock(root / "rounds.lock")
        else:
            time.sleep(5.0)
    print(f"feito: {tot_ok} corridas, {tot_bad} falhas; estado em `status`", flush=True)


def cmd_run(a):
    out = Path(a.out)
    if is_rounds_root(out):
        return cmd_run_rounds(a)
    design = load_design(out)
    if _run_options_changed(a, design):
        _apply_run_options(a, design)
        save_design(out, design)
    stop = out / "STOP"
    if stop.exists() and not a.keep_stop:
        stop.unlink()
    tot_ok = tot_bad = 0
    while True:
        design = load_design(out)
        pending = [
            p
            for p in design["points"]
            if point_status(out, p["id"]) == "pending" or (a.resume_exhausted and is_exhausted(out, p["id"]))
        ]
        if pending:
            chains = sorted({p["chain"] for p in pending})
            print(f"{len(pending)} pontos por correr em {len(chains)} cadeias ({a.jobs} processos)", flush=True)
            n_ok, n_bad = _run_pending(out, design, a.jobs, only=_only_set(a), resume=bool(a.resume_exhausted))
            tot_ok, tot_bad = tot_ok + n_ok, tot_bad + n_bad
        if stop.exists():
            print("STOP encontrado: sem rondas novas", flush=True)
            break
        # rondas seguintes (ideia do dono): sortear outros tantos pontos e recomeçar do centro da ronda nova
        rounds_done = int(design.get("rounds", 1))
        if a.rounds == 0 or (a.rounds > 0 and rounds_done >= a.rounds):
            break
        design = load_design(out)
        new = append_round(design, chains=a.chains_per_round)
        save_design(out, design)
        print(f"ronda {design['rounds'] - 1}: {len(new)} pontos novos ({new[0]['id']}–{new[-1]['id']})", flush=True)
    print(f"feito: {tot_ok} corridas, {tot_bad} falhas; estado em `status`", flush=True)


def _round_dirs(root):
    return [Path(root) / r["folder"] for r in load_rounds(root)["rounds"]]


def cmd_status(a):
    out = Path(a.out)
    if is_rounds_root(out):
        master = load_rounds(out)
        print(
            f"{out}: {len(master['rounds'])} ronda(s), {master['n']} pontos sorteados (ronda de {master['round_size']})"
        )
        for d in _round_dirs(out):
            design = load_design(d)
            c = {"done": 0, "failed": 0, "pending": 0}
            for p in design["points"]:
                c[point_status(d, p["id"])] += 1
            busy = sorted(ch for ch in {p["chain"] for p in design["points"]} if not chain_is_free(d, ch))
            print(
                f"  {d.name}: {c['done']} feitos, {c['failed']} falhados, {c['pending']} por fazer"
                + (f"; cadeias em curso {busy}" if busy else "")
            )
        if (out / "STOP").exists():
            print("  STOP presente (as cadeias param depois da corrida em curso)")
        return
    design = load_design(out)
    counts = {"done": 0, "failed": 0, "pending": 0}
    by_chain = {}
    for p in design["points"]:
        st = point_status(out, p["id"])
        counts[st] += 1
        by_chain.setdefault(p["chain"], {"done": 0, "failed": 0, "pending": 0})[st] += 1
    print(
        f"{design['n']} pontos em {design.get('rounds', 1)} ronda(s): {counts['done']} feitos, {counts['failed']} falhados, "
        f"{counts['pending']} por fazer"
    )
    for c in sorted(by_chain):
        b = by_chain[c]
        print(f"  cadeia {c}: {b['done']} feitos, {b['failed']} falhados, {b['pending']} por fazer")
    if (out / "STOP").exists():
        print("  STOP presente (as cadeias param depois da corrida em curso)")
    csv_path = out / "dataset.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print(f"  dataset.csv: {n} linhas")


# ------------------------------------------------------------------------------------------------ marcadores
def signal_markers(T, y, amp_tol, smooth=SMOOTH):
    """Média, mínimo, máximo e frequências mínima e máxima de um sinal numa janela: as frequências vêm dos intervalos
    entre máximos consecutivos do sinal suavizado (proeminência de 5 % da faixa); um sinal de amplitude menor que
    `amp_tol` é permanente e tem frequências nulas."""
    from scipy.signal import find_peaks

    from mfsim.analysis import moving_average

    T, y = np.asarray(T, float), np.asarray(y, float)
    out = dict(mean=float(np.mean(y)), min=float(np.min(y)), max=float(np.max(y)), fmin=0.0, fmax=0.0)
    rng = out["max"] - out["min"]
    if rng < amp_tol or T.size < 4:
        return out
    ys = moving_average(T, y, smooth)
    dt = float(np.median(np.diff(T)))
    pk, _ = find_peaks(ys, prominence=0.05 * rng, distance=max(1, int(round(3 * smooth / dt))))
    if pk.size >= 2:
        iv = np.diff(T[pk])
        out["fmin"], out["fmax"] = float(1.0 / iv.max()), float(1.0 / iv.min())
    return out


def _low_point_row(model):
    """Linha (célula) do ponto mais baixo do duto `1`."""
    pb = next(p for p in model.pipes if p.pipe.id == "1")
    z = np.concatenate([[0.0], np.cumsum(np.sin(pb.mesh["theta"]) * pb.mesh["h"])])
    zc = 0.5 * (z[1:] + z[:-1])
    return int(np.argmin(zc)) + 1


def run_series(run):
    """Séries dos pontos característicos de uma corrida: {ponto: {var: série}} mais `q_inj` (gás pela retenção)."""
    ids = [pb.pipe.id for pb in run.model.pipes]
    T = np.asarray(run.T)
    out = {}
    low = _low_point_row(run.model)
    for label, pid, row_s, row_q in POINTS:
        if pid not in ids:
            continue
        p = run.pipe(ids.index(pid))["out"]
        if row_s == "tank":
            iH, iP = run.model.vertex_A_index("an_bot")
            P = np.asarray(run.X)[iP] * run.model.PR / 1e5
            aG = np.asarray(run.X)[iH]
        else:
            rs = low if row_s == "low" else row_s
            P, aG = np.asarray(p["P_bar"][rs]), np.asarray(p["alfaG"][rs])
        rq = low if row_q == "low" else row_q
        out[label] = dict(P=P, alphaG=aG, qL=np.asarray(p["qL"][rq]), qG=np.asarray(p["qG"][rq]))
    if "injection" in out and "well_bottom" in out:
        wl = run.pipe(ids.index("well_lower"))["out"]
        out["q_inj"] = np.asarray(run.pipe(ids.index("well"))["out"]["qG"][0]) - np.asarray(wl["qG"][-1])
    return T, out


def run_markers(run, tail=4000.0):
    """Linha de marcadores de uma corrida: metadados, classe e os marcadores de cada sinal na janela de captura
    (`info.settle.capture`; sem ela, os últimos `tail` s)."""
    from mfsim.analysis import cycle_class

    T, series = run_series(run)
    info = run.info or {}
    st = info.get("settle") or {}
    cap = st.get("capture")
    if cap:
        sel = (T >= cap[0] - 1e-9) & (T <= cap[1] + 1e-9)
    else:
        sel = T >= T[-1] - tail
    if np.count_nonzero(sel) < 4:
        sel = np.ones(T.size, bool)
    Tc = T[sel]
    row = dict(
        t_end=float(T[-1]),
        reason=st.get("reason", "none"),
        t_cap0=float(Tc[0]),
        t_cap1=float(Tc[-1]),
        n_steps=info.get("n_steps"),
        cpu=info.get("cpu"),
        compile=info.get("compile"),
        success=info.get("success"),
    )
    if "wellhead" in series and "riser_top" in series:
        c = cycle_class(
            Tc, series["wellhead"]["P"][sel], series["riser_top"]["qL"][sel], smooth=SMOOTH, period_fast=PERIOD_FAST
        )
        row.update({"class": c["cls"], "period": c["period"], "surge": c["surge"]})
    for label, sig in series.items():
        if label == "q_inj":
            m = signal_markers(Tc, sig[sel], AMP_TOL["qG"])
            row.update({f"q_inj_{k}": v for k, v in m.items()})
            continue
        for var in VARS:
            m = signal_markers(Tc, sig[var][sel], AMP_TOL[var])
            row.update({f"{label}_{var}_{k}": v for k, v in m.items()})
    return row


def columns():
    cols = ["id", "chain", "order", "start", "from_point"] + list(INPUTS)
    cols += ["t_end", "reason", "t_cap0", "t_cap1", "n_steps", "cpu", "compile", "success", "class", "period", "surge"]
    for label, *_ in POINTS:
        for var in VARS:
            cols += [f"{label}_{var}_{k}" for k in MARKERS]
    cols += [f"q_inj_{k}" for k in MARKERS]
    return cols


def write_dataset(out, rows, design):
    """`dataset.csv` (uma linha por corrida feita) e `dataset.json` (dicionário das colunas e metadados)."""
    cols = columns()
    out = Path(out)
    with open(out / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    desc = {
        "id": "identificador do ponto: 6 caracteres do SHA-1 do tuplo das entradas (pasta do ponto na ronda)",
        "chain": "cadeia (processo) a que o ponto pertence",
        "order": "posição na cadeia (0 = partida bloqueada)",
        "start": "base (estado final do caso base do artigo 01), warm (estado final do ponto anterior), blocked (estado bloqueado do gerador), steady (estacionário de Newton a partir da estimativa do gerador, perturbado) ou resume (continuação do próprio estado final de uma corrida que esgotou o orçamento)",
        "from_point": "ponto cujo estado final arrancou esta corrida",
        "t_end": "tempo simulado alcançado [s]",
        "reason": "paragem: steady, cycle, tf (tempo máximo sem assentar), failure, none",
        "t_cap0/t_cap1": "janela de captura dos marcadores [s]",
        "class": "stable, fast (ciclo rápido do topo do riser, período < 300 s; 062), cycling (ciclo de pressão) ou slugging (golfada severa) na janela de captura",
        "period": "período entre cruzamentos ascendentes da média de P na cabeça [s]",
        "surge": "razão pico/média de q_L no choke",
        "<ponto>_<var>_<marcador>": "marcadores (mean, min, max, fmin, fmax [1/s]) de P [bar], alphaG, qL e qG [kg/s] no ponto",
        "q_inj_*": "gás injetado pela retenção [kg/s]",
    }
    meta = dict(
        schema="paper02-dataset/1",
        design=str(design_path(out)),
        inputs=list(INPUTS),
        ranges=design.get("ranges"),
        settle=design.get("settle"),
        points={label: dict(pipe=pid, row_state=str(rs), row_flow=str(rq)) for label, pid, rs, rq in POINTS},
        markers=list(MARKERS),
        amp_tol=AMP_TOL,
        smooth_s=SMOOTH,
        columns=cols,
        descriptions=desc,
        n_rows=len(rows),
        written=time.strftime("%Y-%m-%d %H:%M"),
    )
    (out / "dataset.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out / "dataset.csv"


def _row_from_record(rec):
    row = dict(
        id=rec["id"], chain=rec["chain"], order=rec["order"], start=rec.get("start"), from_point=rec.get("from_point")
    )
    row.update(rec["x"])
    row.update(rec["markers"])
    return row


def _record_from_run(design, p, out, tail):
    from mfsim.runs import load_run

    run = load_run(str(run_path(out, p["id"])))
    note = {}
    try:
        note = json.loads((run.info or {}).get("note") or "{}")
    except (TypeError, ValueError):
        pass
    return point_record(design, p, note.get("start"), note.get("from_point"), run, tail=float(tail)), run


def _rows_of(out, design, tail):
    """Linhas da tabela: de `point.json` quando existe; senão (pontos antigos) da trajetória `run.npz`."""
    rows = []
    for p in design["points"]:
        if point_status(out, p["id"]) != "done":
            continue
        if point_path(out, p["id"]).exists():
            rows.append(_row_from_record(json.loads(point_path(out, p["id"]).read_text(encoding="utf-8"))))
        else:
            rows.append(_row_from_record(_record_from_run(design, p, out, tail)[0]))
    return rows


def compact_dir(out, design, tail=4000.0, progress=True):
    """Pontos antigos com trajetória: escreve point.json + final_state.npz e remove run.npz, case.json e
    case.ic.npz (pedido do dono: só fica o que o treino usa). Devolve o número de pontos compactados."""
    n = 0
    for p in design["points"]:
        if not run_path(out, p["id"]).exists():
            continue
        rec, run = _record_from_run(design, p, out, tail)
        save_point(out, rec, np.asarray(run.X)[:, -1])
        for name in ("run.npz", "case.json", "case.ic.npz"):
            f = point_dir(out, p["id"]) / name
            if f.exists():
                f.unlink()
        n += 1
        if progress:
            print(f"{p['id']}: compactado", flush=True)
    return n


def cmd_compact(a):
    out = Path(a.out)
    dirs = _round_dirs(out) if is_rounds_root(out) else [out]
    n = sum(compact_dir(d, load_design(d), tail=float(a.tail)) for d in dirs)
    print(f"{n} pontos compactados")


def cmd_markers(a):
    out = Path(a.out)
    if is_rounds_root(out):
        # uma tabela por ronda e a tabela de todas as rondas na raiz
        master = load_rounds(out)
        all_rows = []
        for d in _round_dirs(out):
            design = load_design(d)
            rows = _rows_of(d, design, a.tail)
            write_dataset(d, rows, design)
            all_rows.extend(rows)
        path = write_dataset(out, all_rows, master)
        print(f"{path}: {len(all_rows)} linhas de {len(master['rounds'])} ronda(s), {len(columns())} colunas")
        return
    design = load_design(out)
    rows = _rows_of(out, design, a.tail)
    path = write_dataset(out, rows, design)
    print(f"{path}: {len(rows)} linhas, {len(columns())} colunas")


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design", help="desenho de Sobol em cadeias (design.json)")
    d.add_argument("--out", required=True)
    d.add_argument(
        "--n", type=int, default=None, help="pontos por ronda (omissão: 25; em --append, os da ronda anterior)"
    )
    d.add_argument("--chains", type=int, default=1, help="cadeias por ronda (1 = uma sequência por ronda)")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--range", action="append", default=[], help="nome=lo:hi (omissão: faixas do plano)")
    d.add_argument(
        "--fix-well",
        action="store_true",
        help="o poço é o do caso base (P_res, PI, RGO fixos nos valores da corrida base); só P_gas e a abertura variam (pedido do dono)",
    )
    d.add_argument("--test", action="store_true", help="conjunto de teste: todos os pontos da partida bloqueada")
    d.add_argument("--tf-max", type=float, default=24000.0)
    d.add_argument("--dt-out", type=float, default=5.0)
    d.add_argument("--max-cpu", type=float, default=2400.0, help="orçamento de relógio por corrida [s] (045)")
    d.add_argument("--window", type=float, default=3000.0, help="janela da paragem adaptativa [s]")
    d.add_argument("--t-min", type=float, default=4000.0, help="tempo mínimo antes de parar, partida bloqueada [s]")
    d.add_argument("--t-min-warm", type=float, default=1500.0, help="tempo mínimo, arranque quente [s]")
    d.add_argument("--rel-tol", type=float, default=0.1)
    d.add_argument("--capture", type=int, default=3, help="ciclos capturados depois de assentar (N_c)")
    d.add_argument("--capture-window", type=float, default=600.0, help="janela de captura de um estacionário [s]")
    d.add_argument(
        "--window-periods",
        type=int,
        default=10,
        help="062: ciclos rápidos avaliados em janelas deste número de períodos quando cabem na janela (0 = desligado)",
    )
    d.add_argument("--window-min", type=float, default=600.0, help="062: janela mínima dos ciclos rápidos [s]")
    d.add_argument("--force", action="store_true")
    d.add_argument("--flat", action="store_true", help="conjunto numa só pasta (sem rondas datadas; o modo dos testes)")
    d.add_argument(
        "--start",
        choices=["base", "random", "centre", "steady"],
        default="base",
        help="primeiro ponto de cada cadeia: o mais perto do caso base do artigo 01, arrancando do seu estado final (omissão, pedido do dono); sorteado; o mais perto do centro; ou `steady`: todos os pontos independentes, da partida estacionária (Newton a partir da estimativa do gerador, sem caso base nem ponto anterior; pedido do dono, 2026-10-19)",
    )
    d.add_argument(
        "--guess",
        choices=["noslip", "blocked"],
        default="noslip",
        help="estimativa do gerador de onde parte o Newton da partida estacionária (--start steady)",
    )
    d.add_argument(
        "--perturb",
        type=float,
        default=0.01,
        help="partida estacionária: as pressões do estacionário multiplicam-se por 1 + perturb, para a trajetória sair de um equilíbrio instável (issue 040)",
    )
    d.add_argument("--newton-iter", type=int, default=100, help="iterações máximas do Newton da partida estacionária")
    d.add_argument(
        "--base-run",
        default="runs/paper01/gaslift/gl_P20.npz",
        help="corrida do caso base (estado final = arranque das cadeias; as suas entradas = ponto de referência)",
    )
    d.add_argument(
        "--resequence", action="store_true", help="reordenar a última ronda (sem corridas feitas) com a regra atual"
    )
    d.add_argument(
        "--grid-decisions",
        default=None,
        metavar="NxM",
        help="modo malha: por ronda, um estado do poço (Sobol em P_res, PI, RGO) e a malha N (P_gas) × M (abertura) das decisões em serpentina",
    )
    d.add_argument(
        "--inner",
        choices=["choke_opening", "P_gas_bar"],
        default="choke_opening",
        help="eixo interior da serpentina (o exterior é monótono)",
    )
    d.add_argument(
        "--append",
        action="store_true",
        help="acrescentar uma ronda de --n pontos (--chains cadeias) a um desenho existente",
    )
    d.set_defaults(func=cmd_design)
    r = sub.add_parser("run", help="corre as cadeias (retomável; STOP na pasta para parar a quente)")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--keep-stop", action="store_true", help="não apagar um STOP existente ao arrancar")
    r.add_argument(
        "--only",
        default=None,
        help="só estes pontos (ids separados por vírgulas, ou ficheiro com um id por linha); os outros pendentes ficam para depois; não fica gravado",
    )
    r.add_argument(
        "--resume-exhausted",
        action="store_true",
        help="retomar os pontos que esgotaram o orçamento a partir do próprio estado final (orçamento inteiro outra vez; os marcadores passam a ser os da continuação); não fica gravado",
    )
    r.add_argument(
        "--from-base",
        action="store_true",
        help="todos os pontos arrancam do estado final do caso base (`base_run`), nenhum do ponto anterior; fica gravado na ronda (pedido do dono, 2026-10-19)",
    )
    r.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="rondas no total: ao acabar os pontos, sortear outra ronda e recomeçar do centro (0 = só o que existe; -1 = até STOP)",
    )
    r.add_argument("--chains-per-round", type=int, default=1, help="cadeias de cada ronda nova")
    r.add_argument(
        "--keep-runs", action="store_true", help="guardar também run.npz, case.json e case.ic.npz (depuração)"
    )
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compact", help="pontos antigos com trajetória → point.json + final_state.npz, sem run.npz/case")
    c.add_argument("--out", required=True)
    c.add_argument("--tail", type=float, default=4000.0)
    c.set_defaults(func=cmd_compact)
    m = sub.add_parser("markers", help="marcadores das corridas feitas → dataset.csv")
    m.add_argument("--out", required=True)
    m.add_argument("--tail", type=float, default=4000.0, help="janela final [s] quando não há captura")
    m.set_defaults(func=cmd_markers)
    s = sub.add_parser("status", help="pontos feitos, falhados e por fazer")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_status)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
